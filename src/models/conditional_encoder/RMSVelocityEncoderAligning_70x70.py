import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Encoder Blocks ----------
class ConvBNGELU1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):  # (B, C, T)
        return self.act(self.bn(self.conv(x)))


class ConvBNGELU2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):  # (B, C, H, W)
        return self.act(self.bn(self.conv(x)))


class ResBlock2d(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNGELU2d(ch, ch, k=3, s=1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNGELU2d(ch, ch, k=3, s=1)
        self.out = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return self.out(x + y)


class RMSVelocityEncoderA(nn.Module):
    """
    Encode RMS velocity traces from ``(B, 1, 1000, 70)`` into ``(B, C_out, 70, 70)``.

    The model first runs 1D convolutions trace by trace, then reshapes the result into
    a 2D feature map for spatial refinement.
    """

    def __init__(self, C_t: int = 16, C_mid: int = 64, C_out: int = 64, n_res2d: int = 3, dropout2d: float = 0.0):
        super().__init__()
        self.C_t = C_t
        self.C_mid = C_mid
        self.C_out = C_out

        self.temporal1 = ConvBNGELU1d(1, C_t, k=7, s=1)
        self.temporal2 = ConvBNGELU1d(C_t, C_t, k=5, s=1)
        self.pool1d = nn.AdaptiveAvgPool1d(70)  # Compress 1000 time samples down to 70.

        self.stem2d = ConvBNGELU2d(C_t, C_mid, k=3)
        blocks = [ResBlock2d(C_mid, dropout=dropout2d) for _ in range(n_res2d)]
        self.refine2d = nn.Sequential(*blocks)

        self.proj = nn.Conv2d(C_mid, C_out, kernel_size=1, bias=True)
        self.decoder = RMS2DepthDecoder(C_in=self.C_out)

    def forward(self, x):
        B, C, T, W = x.shape
        assert C == 1 and T == 1000 and W == 70, "Expected input shape (B, 1, 1000, 70)."

        # Convert each lateral column into a standalone 1D trace batch.
        t = x.permute(0, 3, 1, 2).contiguous().view(B * W, 1, T)

        h = self.temporal1(t)
        h = self.temporal2(h)
        h = self.pool1d(h)

        # Restore the pooled traces as a 2D feature map aligned to 70x70.
        h = h.view(B, W, self.C_t, 70).permute(0, 2, 3, 1).contiguous()

        h = self.stem2d(h)
        h = self.refine2d(h)

        y = self.proj(h)
        return y


# ---------- Decoder Blocks ----------
class DecoderConvBNGELU2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderResBlock2d(nn.Module):
    def __init__(self, ch, dilation=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return self.act(x + y)


def add_coord_channels(x):
    """Append normalized y/x coordinates to the feature tensor."""
    B, _, H, W = x.shape
    device = x.device
    yy = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
    xx = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, yy, xx], dim=1)


class RMS2DepthDecoder(nn.Module):
    """
    Decode RMS encoder features ``[B, C_in, 70, 70]`` into depth velocity ``[B, 1, 70, 70]``.
    """

    def __init__(self, C_in: int, C_mid: int = 64, n_blocks: int = 4, dilations=(1, 2, 4, 1),
                 use_coords: bool = True, dropout: float = 0.0, out_activation: str = "none"):
        super().__init__()
        self.use_coords = use_coords
        stem_in = C_in + (2 if use_coords else 0)

        self.stem = DecoderConvBNGELU2d(stem_in, C_mid, k=3)

        blocks = []
        for i in range(n_blocks):
            d = dilations[i % len(dilations)]
            blocks.append(DecoderResBlock2d(C_mid, dilation=d, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        # Build a small coarse-context branch and merge it back into the 70x70 stream.
        self.down = nn.Conv2d(C_mid, C_mid, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_bn = nn.BatchNorm2d(C_mid)
        self.ctx = DecoderResBlock2d(C_mid, dilation=2, dropout=dropout)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.mix = DecoderConvBNGELU2d(C_mid, C_mid, k=3)

        self.head = nn.Conv2d(C_mid, 1, kernel_size=1, bias=True)
        self.out_activation = out_activation

    def forward(self, f_rms):
        h = add_coord_channels(f_rms) if self.use_coords else f_rms
        h = self.stem(h)
        h = self.blocks(h)

        c = F.gelu(self.down_bn(self.down(h)))
        c = self.ctx(c)
        c = self.up(c)
        h = self.mix(h + c)

        y = self.head(h)

        if self.out_activation == "relu":
            y = F.relu(y)
        elif self.out_activation == "softplus":
            y = F.softplus(y)
        elif self.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        return y


if __name__ == "__main__":
    B = 2
    x = torch.randn(B, 1, 1000, 70)
    model = RMSVelocityEncoderA(C_t=16, C_mid=64, C_out=64, n_res2d=3, dropout2d=0.0)
    y = model(x)
    decoder = RMS2DepthDecoder(C_in=64)
    z = decoder(y)
    print("Output shape:", y.shape, z.shape)  # Expected: (B, 64, 70, 70) and (B, 1, 70, 70)
