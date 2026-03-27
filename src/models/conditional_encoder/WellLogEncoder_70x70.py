import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== Building Blocks =====================

class ConvBNAct2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, int):
                p = ((k - 1) // 2) * d
            else:
                p = (((k[0] - 1) // 2) * d, ((k[1] - 1) // 2) * d)
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock2d(nn.Module):
    def __init__(self, ch, d1=1, d2=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNAct2d(ch, ch, k=3, d=d1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNAct2d(ch, ch, k=3, d=d2, act=False)
        self.out = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return self.out(x + y)


class ConvBNAct1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):  # (B, C, T)
        return self.act(self.bn(self.conv(x)))


class ColumnAttn(nn.Module):
    """Column attention that highlights lateral positions containing well traces."""

    def __init__(self, ch, k=7):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size=(1, k), padding=(0, k // 2), groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=True)

    def forward(self, x):
        m = x.mean(dim=2, keepdim=True)
        w = torch.sigmoid(self.pw(self.dw(m)))
        return x * w


def add_coord_channels(x):
    """Append normalized y/x coordinates to a 70x70 feature tensor."""
    B, _, H, W = x.shape
    device = x.device
    yy = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
    xx = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, yy, xx], dim=1)


def gaussian1d_kernel(k=7, sigma=1.5, device="cpu"):
    """Create a width-only Gaussian kernel used to spread sparse well columns."""
    ax = torch.arange(k, device=device) - (k - 1) / 2.0
    ker = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
    ker = ker / ker.sum()
    return ker


class FixedGaussianBlurWidth(nn.Module):
    """Horizontal 1xk Gaussian blur that only spreads signals along the width axis."""

    def __init__(self, k=7, sigma=1.5):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.register_buffer("weight", torch.zeros(1, 1, 1, k), persistent=False)
        self._init = False

    def _maybe_init(self, device):
        if not self._init or self.weight.device != device:
            ker = gaussian1d_kernel(self.k, self.sigma, device=device).view(1, 1, 1, self.k)
            self.weight.data = ker
            self._init = True

    def forward(self, x):
        self._maybe_init(x.device)
        return F.conv2d(x, self.weight, stride=1, padding=(0, self.k // 2))


class WellLogEncoderA(nn.Module):
    """
    Dual-branch encoder for sparse well-log maps.

    Input:
        (B, 1, 70, 70) where active columns contain well-log values and the rest are zero.
    """

    def __init__(self, C_t: int = 16, C2d_1: int = 32, C2d_2: int = 48, C_out: int = 16,
                 k_soft: int = 7, sigma_soft: float = 1.5, dropout: float = 0.0, use_coords: bool = True):
        super().__init__()
        self.soft = FixedGaussianBlurWidth(k=k_soft, sigma=sigma_soft)
        self.use_coords = use_coords

        self.t_conv1 = ConvBNAct1d(1, C_t, k=7, s=1)
        self.t_conv2 = ConvBNAct1d(C_t, C_t, k=5, s=1)

        in2d = 5 if use_coords else 3  # [x, mask, soft] plus optional [y, x].
        self.s2d_stem = ConvBNAct2d(in2d, C2d_1, k=(5, 3))
        self.s2d_res1 = ResBlock2d(C2d_1, d1=1, d2=2, dropout=dropout)
        self.s2d_att1 = ColumnAttn(C2d_1, k=7)

        self.s2d_to2 = ConvBNAct2d(C2d_1, C2d_2, k=3)
        self.s2d_res2 = ResBlock2d(C2d_2, d1=2, d2=1, dropout=dropout)
        self.s2d_att2 = ColumnAttn(C2d_2, k=7)

        self.fuse = ConvBNAct2d(C2d_2 + C_t, C2d_2, k=1)
        self.mix = ResBlock2d(C2d_2, d1=1, d2=1, dropout=dropout)
        self.out = nn.Conv2d(C2d_2, C_out, kernel_size=1, bias=True)

    def _build_aug_input(self, x):
        """Build the augmented input ``[x, mask, soft]`` and optionally append coordinates."""
        mask = (x.abs() > 0).float()
        soft = self.soft(x)
        xin = torch.cat([x, mask, soft], dim=1)
        if self.use_coords:
            xin = add_coord_channels(xin)
        return xin

    def _column_1d(self, x):
        """Encode each well column independently before restoring a 2D layout."""
        B, _, H, W = x.shape
        t = x.permute(0, 3, 1, 2).contiguous().view(B * W, 1, H)
        t = self.t_conv2(self.t_conv1(t))
        t = t.view(B, W, -1, H).permute(0, 2, 3, 1).contiguous()
        return t

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 1 and H == 70 and W == 70, "Expected input shape (B, 1, 70, 70)."

        feat_1d = self._column_1d(x)

        xin = self._build_aug_input(x)
        h = self.s2d_stem(xin)
        h = self.s2d_att1(self.s2d_res1(h))
        h = self.s2d_to2(h)
        h = self.s2d_att2(self.s2d_res2(h))

        h = torch.cat([h, feat_1d], dim=1)
        h = self.fuse(h)
        h = self.mix(h)
        y = self.out(h)
        return y


if __name__ == "__main__":
    # Build a toy sample with wells at columns 10 and 55.
    B = 2
    x = torch.zeros(B, 1, 70, 70)
    x[:, 0, :, 10] = torch.linspace(2.0, 4.0, 70)
    x[:, 0, :, 55] = torch.linspace(3.0, 5.0, 70)

    net = WellLogEncoderA(C_t=16, C2d_1=32, C2d_2=48, C_out=16, k_soft=7, sigma_soft=1.5, dropout=0.0,
                          use_coords=True)
    y = net(x)
    print("Output:", y.shape)  # Expected: (B, 16, 70, 70)
