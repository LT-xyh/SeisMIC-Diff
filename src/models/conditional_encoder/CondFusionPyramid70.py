from typing import Dict, Optional

import torch
import torch.nn as nn
from models.conditional_encoder.HorizonEncoder_70x70 import HorizonEncoderA
from models.conditional_encoder.RMSVelocityEncoderAligning_70x70 import RMSVelocityEncoderA
from models.conditional_encoder.SeismicImageEncoder_70x70 import SeismicImageEncoderA
from models.conditional_encoder.WellLogEncoder_70x70 import WellLogEncoderA


# ====================== 通用小组件 ======================

def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8):
    """
    scores, mask: (B, M, 1, 1)  —— M 个模态的全局分数；mask=0 表示该模态缺失
    """
    scores = scores.masked_fill(mask == 0, float("-inf"))
    # 处理全 0：退化为平均
    all_zero = (mask.sum(dim=dim, keepdim=True) == 0)
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
    """
    标准残差块，in/out 通道可不同（用 1x1 跳连对齐）
    """

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
    """
    为每个模态生成一个标量分数（GAP -> 1x1 -> GELU -> 1x1）
    """

    def __init__(self, ch_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch_in, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x, 1)
        return self.net(g)  # (B,1,1,1)


# ====================== 条件融合金字塔（70 → 64/32/16） ======================

class CondFusionPyramid70(nn.Module):
    """
    把四路条件 (均为 70x70) 融合，并输出多尺度条件图：
      out: {
        's16': (B, C16, 16, 16),
        's32': (B, C32, 32, 32),
        's64': (B, C64, 64, 64),
        's70': (B, C70, 70, 70),
        'weights': (B, M, 1, 1)  # 模态门控权重（可解释）
      }
    设计：逐尺度对每个模态做 1x1 通道对齐 + 下采样 → 门控加权 concat → 1x1 压回 C_s → ResBlock 细化
    """

    def __init__(self,
                 in_channels: Dict[str, int] = {'rms_vel': 64, 'migrated_image': 64, 'horizon': 16, 'well_log': 16},
                 C_per_scale: Dict[str, int] = {'s16': 64, 's32': 64, 's64': 64, 's70': 32}, unify_per_mod: int = 32,
                 # 每个模态先对齐到这个中间维度再融合
                 score_hidden: int = 32, modality_dropout_p: float = 0.0):
        super().__init__()
        self.names = list(in_channels.keys())
        self.Cs = C_per_scale
        self.unify = nn.ModuleDict({n: nn.Conv2d(cin, unify_per_mod, 1) for n, cin in in_channels.items()})
        self.scorers = nn.ModuleDict({n: ModalityScorer(unify_per_mod, score_hidden) for n in self.names})
        self.modality_dropout_p = modality_dropout_p
        self.cond_encoder = nn.ModuleDict(
            {'rms_vel': RMSVelocityEncoderA(C_t=16, C_mid=64, C_out=in_channels['rms_vel'], n_res2d=3, dropout2d=0.0),
             'migrated_image': SeismicImageEncoderA(c1=32, c2=64, c3=96, c4=128, C_out=in_channels['migrated_image'],
                                                    use_se=True, dropout=0.0),
             'horizon': HorizonEncoderA(C_out=in_channels['horizon'], c1=32, c2=48, c3=64, k_gauss=7, sigma_gauss=1.5,
                                        use_se=True, use_spatial_attn=True, dropout=0.0),
             'well_log': WellLogEncoderA(C_t=16, C2d_1=32, C2d_2=48, C_out=16, k_soft=7, sigma_soft=1.5, dropout=0.0,
                                         use_coords=True), })

        # 每个尺度一个融合头（concat -> 1x1 -> ResBlock）
        total_unify = unify_per_mod * len(self.names)
        self.head70 = nn.Sequential(nn.Conv2d(total_unify, self.Cs['s70'], 1, bias=False),
                                    nn.BatchNorm2d(self.Cs['s70']), nn.GELU(),
                                    ResBlock(self.Cs['s70'], self.Cs['s70'], d1=1, d2=1))
        self.head64 = nn.Sequential(nn.Conv2d(total_unify, self.Cs['s64'], 1, bias=False),
                                    nn.BatchNorm2d(self.Cs['s64']), nn.GELU(),
                                    ResBlock(self.Cs['s64'], self.Cs['s64'], d1=1, d2=1))
        self.head32 = nn.Sequential(nn.Conv2d(total_unify, self.Cs['s32'], 1, bias=False),
                                    nn.BatchNorm2d(self.Cs['s32']), nn.GELU(),
                                    ResBlock(self.Cs['s32'], self.Cs['s32'], d1=1, d2=1))
        self.head16 = nn.Sequential(nn.Conv2d(total_unify, self.Cs['s16'], 1, bias=False),
                                    nn.BatchNorm2d(self.Cs['s16']), nn.GELU(),
                                    ResBlock(self.Cs['s16'], self.Cs['s16'], d1=1, d2=1))

        self.decoder = CondS16ToZDecoder(in_ch=self.Cs['s16'], mid_ch=48, out_ch=16, dropout=0.0, dilations=(1, 2))

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
        # 防止全丢
        if all(v is None for v in out.values()):
            avail = [k for k, v in feats.items() if v is not None]
            if len(avail) > 0:
                pick = avail[torch.randint(len(avail), (1,)).item()]
                out[pick] = feats[pick]
        return out

    def _fuse_at_scale(self, feats70: Dict[str, torch.Tensor], H: int, W: int, head: nn.Module):
        """
        在目标尺度 (H,W) 融合：下采到该尺度、门控加权、concat、head
        """
        xs, scores, masks = [], [], []
        for name in self.names:
            f70 = feats70.get(name, None)
            if f70 is None:
                # 缺失模态
                B = next(iter(feats70.values())).size(0)
                device = next(iter(feats70.values())).device
                scores.append(torch.zeros(B, 1, 1, 1, device=device))
                masks.append(torch.zeros(B, 1, 1, 1, device=device))
                continue
            u = self.unify[name](f70)  # (B,unify,H70,W70)
            if (u.shape[-2], u.shape[-1]) != (H, W):
                u = F.adaptive_avg_pool2d(u, (H, W))  # 下采到该尺度
            xs.append(u)
            s = self.scorers[name](u)  # (B,1,1,1)
            scores.append(s)
            masks.append(torch.ones_like(s))
        scores = torch.stack(scores, dim=1)  # (B,M,1,1)
        masks = torch.stack(masks, dim=1)  # (B,M,1,1)
        weights = masked_softmax(scores, masks, dim=1)  # (B,M,1,1)

        # 对存在的 xs 乘对应权重并 concat（按 self.names 前后顺序）
        gated = []
        m_idx = 0
        for name in self.names:
            f70 = feats70.get(name, None)
            if f70 is None:
                m_idx += 1
                continue
            w = weights[:, m_idx, ...]  # (B,1,1,1)
            u = self.unify[name](f70)
            if (u.shape[-2], u.shape[-1]) != (H, W):
                u = F.adaptive_avg_pool2d(u, (H, W))
            gated.append(u * w)
            m_idx += 1

        if len(gated) == 0:
            raise ValueError("No available modality features to fuse at this scale.")
        x = torch.cat(gated, dim=1)  # (B, M*unify, H, W)
        y = head(x)  # (B, C_s, H, W)
        return y, weights

    def forward(self, cond_dict: Dict[str, Optional[torch.Tensor]]):
        """
        feats70: dict 包含四路条件：
          {'rms_vel': (B,64,70,70), 'migrated_image': (B,64,70,70), 'horizon': (B,16,70,70), 'well_log': (B,16,70,70)}
          可为 None 表示缺失。
        """
        feats70 = {}
        for k in cond_dict.keys():
            feats70.update({k: self.cond_encoder[k](cond_dict[k])})
        feats70 = self._maybe_dropout(feats70)

        # 70
        s70, w = self._fuse_at_scale(feats70, 70, 70, self.head70)
        # 64/32/16
        s64, _ = self._fuse_at_scale(feats70, 64, 64, self.head64)
        s32, _ = self._fuse_at_scale(feats70, 32, 32, self.head32)
        s16, _ = self._fuse_at_scale(feats70, 16, 16, self.head16)

        return {'s16': s16, 's32': s32, 's64': s64, 's70': s70, 'weights': w}


import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- 小组件 ---------
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
    """同通道残差块（保持 H×W），可选扩张卷积增强感受野"""

    def __init__(self, ch, d1=1, d2=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNGELU2d(ch, ch, k=3, d=d1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNGELU2d(ch, ch, k=3, d=d2, act=False)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv2(self.drop(self.conv1(x)))
        return self.act(x + y)


# --------- Cond s16 -> z 的轻量解码器 ---------
class CondS16ToZDecoder(nn.Module):
    """
    输入:  s16 ∈ [B, 64, 16, 16]  （CondEncoder在16×16尺度的特征）
    输出:  z_hat ∈ [B, 16, 16, 16] （与 AE latent 对齐，用于计算对齐损失）
    结构: 1×1降维 → 2个残差块 → 1×1投影到16通道 + 直接跳连(1×1)（更稳）
    """

    def __init__(self, in_ch=64, mid_ch=48, out_ch=16, dropout=0.0, dilations=(1, 2)):
        super().__init__()
        self.stem = ConvBNGELU2d(in_ch, mid_ch, k=1)  # 64→48
        self.block1 = ResBlockSame(mid_ch, d1=dilations[0], d2=1, dropout=dropout)
        self.block2 = ResBlockSame(mid_ch, d1=1, d2=dilations[1], dropout=dropout)
        self.head = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=True)  # 48→16
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)  # 64→16（残差直连）

    def forward(self, s16):  # (B,64,16,16)
        h = self.stem(s16)
        h = self.block2(self.block1(h))
        z_hat = self.head(h) + self.skip(s16)  # 稳定对齐
        return z_hat  # (B,16,16,16)


# ====================== 自测（维度） ======================
if __name__ == "__main__":
    conds = {"rms_vel": torch.randn(2, 1, 1000, 70), "migrated_image": torch.randn(2, 1, 1000, 70),
        "horizon": torch.randn(2, 1, 70, 70), "well_log": torch.randn(2, 1, 70, 70), }
    model = CondFusionPyramid70()
    retur = model(conds)
    print("s16:", retur['s16'].shape)
    latent = model.decoder(retur['s16'])
    print("latent:", latent.shape)
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Total trainable parameters: {total_params / 1e6:.2f} M")  # print("pred:", retur)  # -> (B,1,70,70)  # for value in retur.values():  #     print(value.shape)
