from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conditional_encoder.HorizonEncoder_70x70 import HorizonEncoderA
from models.conditional_encoder.RMSVelocityEncoderAligning_70x70 import RMSVelocityEncoderA
from models.conditional_encoder.SeismicImageEncoder_70x70 import SeismicImageEncoderA
from models.conditional_encoder.WellLogEncoder_70x70 import WellLogEncoderA


# ====================== Shared Blocks ======================

def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8):
    """
    Softmax over modality scores while ignoring missing modalities.

    Args:
        scores: Tensor shaped like ``(B, M, 1, 1)``.
        mask: Tensor with the same modality axis where ``0`` marks a missing modality.
    """
    del eps
    scores = scores.masked_fill(mask == 0, float("-inf"))

    # Fall back to uniform weights if every modality is missing.
    all_zero = mask.sum(dim=dim, keepdim=True) == 0
    scores = scores.clone()
    scores[all_zero.expand_as(scores)] = 0.0
    w = torch.softmax(scores, dim=dim)
    if all_zero.any():
        w = torch.where(all_zero.expand_as(w), torch.full_like(w, 1.0 / w.size(dim)), w)
    return w


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, act=True):
        super().__init__()
        p = ((k - 1) // 2) * d if isinstance(k, int) else (((k[0] - 1) // 2) * d, ((k[1] - 1) // 2) * d)
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Standard residual block with optional input projection."""

    def __init__(self, in_ch, out_ch, k=3, d1=1, d2=1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, k=k, d=d1, act=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNAct(out_ch, out_ch, k=k, d=d2, act=False)
        self.act = nn.GELU()
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x):
        y = self.conv2(self.drop(self.conv1(x)))
        return self.act(self.proj(x) + y)


class ModalityScorer(nn.Module):
    """Produce one global confidence score per modality."""

    def __init__(self, ch_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch_in, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x, 1)
        return self.net(g)


# ====================== Fusion Pyramid ======================

class CondFusionPyramid70(nn.Module):
    """
    Fuse four condition branches defined on 70x70 grids and emit multi-scale features.

    Output:
        {
            "s16": (B, C16, 16, 16),
            "s32": (B, C32, 32, 32),
            "s64": (B, C64, 64, 64),
            "s70": (B, C70, 70, 70),
            "weights": (B, M, 1, 1),
        }
    """

    def __init__(self,
                 in_channels: Dict[str, int] = {"rms_vel": 64, "migrated_image": 64, "horizon": 16, "well_log": 16},
                 C_per_scale: Dict[str, int] = {"s16": 64, "s32": 64, "s64": 64, "s70": 32},
                 unify_per_mod: int = 32,
                 score_hidden: int = 32,
                 modality_dropout_p: float = 0.0):
        super().__init__()
        self.names = list(in_channels.keys())
        self.Cs = C_per_scale
        self.unify = nn.ModuleDict({n: nn.Conv2d(cin, unify_per_mod, 1) for n, cin in in_channels.items()})
        self.scorers = nn.ModuleDict({n: ModalityScorer(unify_per_mod, score_hidden) for n in self.names})
        self.modality_dropout_p = modality_dropout_p
        self.cond_encoder = nn.ModuleDict({
            "rms_vel": RMSVelocityEncoderA(C_t=16, C_mid=64, C_out=in_channels["rms_vel"], n_res2d=3, dropout2d=0.0),
            "migrated_image": SeismicImageEncoderA(c1=32, c2=64, c3=96, c4=128,
                                                   C_out=in_channels["migrated_image"], use_se=True, dropout=0.0),
            "horizon": HorizonEncoderA(C_out=in_channels["horizon"], c1=32, c2=48, c3=64, k_gauss=7,
                                       sigma_gauss=1.5, use_se=True, use_spatial_attn=True, dropout=0.0),
            "well_log": WellLogEncoderA(C_t=16, C2d_1=32, C2d_2=48, C_out=16, k_soft=7, sigma_soft=1.5,
                                        dropout=0.0, use_coords=True),
        })

        total_unify = unify_per_mod * len(self.names)
        self.head70 = nn.Sequential(nn.Conv2d(total_unify, self.Cs["s70"], 1, bias=False),
                                    nn.BatchNorm2d(self.Cs["s70"]), nn.GELU(),
                                    ResBlock(self.Cs["s70"], self.Cs["s70"], d1=1, d2=1))
        self.head64 = nn.Sequential(nn.Conv2d(total_unify, self.Cs["s64"], 1, bias=False),
                                    nn.BatchNorm2d(self.Cs["s64"]), nn.GELU(),
                                    ResBlock(self.Cs["s64"], self.Cs["s64"], d1=1, d2=1))
        self.head32 = nn.Sequential(nn.Conv2d(total_unify, self.Cs["s32"], 1, bias=False),
                                    nn.BatchNorm2d(self.Cs["s32"]), nn.GELU(),
                                    ResBlock(self.Cs["s32"], self.Cs["s32"], d1=1, d2=1))
        self.head16 = nn.Sequential(nn.Conv2d(total_unify, self.Cs["s16"], 1, bias=False),
                                    nn.BatchNorm2d(self.Cs["s16"]), nn.GELU(),
                                    ResBlock(self.Cs["s16"], self.Cs["s16"], d1=1, d2=1))

        self.decoder = CondS16ToZDecoder(in_ch=self.Cs["s16"], mid_ch=48, out_ch=16, dropout=0.0, dilations=(1, 2))

    def _maybe_dropout(self, feats: Dict[str, Optional[torch.Tensor]]):
        if not self.training or self.modality_dropout_p <= 0:
            return feats
        out = {}
        for name, f in feats.items():
            if f is None:
                out[name] = None
            else:
                drop = torch.rand((), device=f.device) < self.modality_dropout_p
                out[name] = None if drop else f

        # Ensure at least one modality survives the stochastic dropout.
        if all(v is None for v in out.values()):
            avail = [k for k, v in feats.items() if v is not None]
            if len(avail) > 0:
                pick = avail[torch.randint(len(avail), (1,)).item()]
                out[pick] = feats[pick]
        return out

    def _fuse_at_scale(self, feats70: Dict[str, torch.Tensor], H: int, W: int, head: nn.Module):
        """Project each available modality to the target scale and gate them before fusion."""
        xs, scores, masks = [], [], []
        for name in self.names:
            f70 = feats70.get(name, None)
            if f70 is None:
                ref = next(v for v in feats70.values() if v is not None)
                B = ref.size(0)
                device = ref.device
                scores.append(torch.zeros(B, 1, 1, 1, device=device))
                masks.append(torch.zeros(B, 1, 1, 1, device=device))
                continue
            u = self.unify[name](f70)
            if (u.shape[-2], u.shape[-1]) != (H, W):
                u = F.adaptive_avg_pool2d(u, (H, W))
            xs.append(u)
            s = self.scorers[name](u)
            scores.append(s)
            masks.append(torch.ones_like(s))
        scores = torch.stack(scores, dim=1)
        masks = torch.stack(masks, dim=1)
        weights = masked_softmax(scores, masks, dim=1)

        gated = []
        m_idx = 0
        for name in self.names:
            f70 = feats70.get(name, None)
            if f70 is None:
                m_idx += 1
                continue
            w = weights[:, m_idx, ...]
            u = self.unify[name](f70)
            if (u.shape[-2], u.shape[-1]) != (H, W):
                u = F.adaptive_avg_pool2d(u, (H, W))
            gated.append(u * w)
            m_idx += 1

        if len(gated) == 0:
            raise ValueError("No available modality features to fuse at this scale.")
        x = torch.cat(gated, dim=1)
        y = head(x)
        return y, weights

    def forward(self, cond_dict: Dict[str, Optional[torch.Tensor]]):
        """
        Encode each provided modality and fuse them into a multi-scale pyramid.

        Missing modalities may be omitted or set to ``None``.
        """
        feats70 = {}
        for k in cond_dict.keys():
            feats70.update({k: self.cond_encoder[k](cond_dict[k])})
        feats70 = self._maybe_dropout(feats70)

        s70, w = self._fuse_at_scale(feats70, 70, 70, self.head70)
        s64, _ = self._fuse_at_scale(feats70, 64, 64, self.head64)
        s32, _ = self._fuse_at_scale(feats70, 32, 32, self.head32)
        s16, _ = self._fuse_at_scale(feats70, 16, 16, self.head16)

        return {"s16": s16, "s32": s32, "s64": s64, "s70": s70, "weights": w}


# ====================== Decoder Blocks ======================

class ConvBNGELU2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, bias=False, act=True):
        super().__init__()
        p = ((k - 1) // 2) * d
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, padding=p, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlockSame(nn.Module):
    """Residual block that keeps the same channel count and spatial size."""

    def __init__(self, ch, d1=1, d2=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNGELU2d(ch, ch, k=3, d=d1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNGELU2d(ch, ch, k=3, d=d2, act=False)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv2(self.drop(self.conv1(x)))
        return self.act(x + y)


class CondS16ToZDecoder(nn.Module):
    """
    Lightweight decoder that maps the fused ``s16`` feature map into the VAE latent space.
    """

    def __init__(self, in_ch=64, mid_ch=48, out_ch=16, dropout=0.0, dilations=(1, 2)):
        super().__init__()
        self.stem = ConvBNGELU2d(in_ch, mid_ch, k=1)
        self.block1 = ResBlockSame(mid_ch, d1=dilations[0], d2=1, dropout=dropout)
        self.block2 = ResBlockSame(mid_ch, d1=1, d2=dilations[1], dropout=dropout)
        self.head = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=True)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, s16):
        h = self.stem(s16)
        h = self.block2(self.block1(h))
        z_hat = self.head(h) + self.skip(s16)
        return z_hat


if __name__ == "__main__":
    conds = {
        "rms_vel": torch.randn(2, 1, 1000, 70),
        "migrated_image": torch.randn(2, 1, 1000, 70),
        "horizon": torch.randn(2, 1, 70, 70),
        "well_log": torch.randn(2, 1, 70, 70),
    }
    model = CondFusionPyramid70()
    retur = model(conds)
    print("s16:", retur["s16"].shape)
    latent = model.decoder(retur["s16"])
    print("latent:", latent.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")
