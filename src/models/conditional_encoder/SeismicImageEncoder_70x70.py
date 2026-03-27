import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- Building Blocks -----------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, int):
                p = ((k - 1) // 2) * d
            else:
                p = (((k[0] - 1) // 2) * d, ((k[1] - 1) // 2) * d)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, k=3, s=1, d=dilation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNAct(ch, ch, k=3, s=1, d=1, act=False)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv2(self.drop(self.conv1(x)))
        return self.act(x + y)


class SEBlock(nn.Module):
    """Optional lightweight channel attention."""

    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), 1)
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class SeismicImageEncoderA(nn.Module):
    """
    Encoder for migrated seismic images with strong vertical anisotropy.

    Input:
        (B, 1, 1000, 70)

    Output:
        (B, C_out, 70, 70)
    """

    def __init__(self, c1=32, c2=64, c3=96, c4=128, C_out=64, use_se=True, se_ratio=8, dropout=0.0):
        super().__init__()

        self.stem = ConvBNAct(1, c1, k=3, s=1)

        # Downsample only along the vertical axis to preserve lateral resolution.
        self.aa1 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv1 = ConvBNAct(c1, c1, k=3, s=1)
        self.res1 = ResBlock(c1, dilation=2, dropout=dropout)
        self.to2 = ConvBNAct(c1, c2, k=3, s=1)

        self.aa2 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv2 = ConvBNAct(c2, c2, k=3, s=1)
        self.res2 = ResBlock(c2, dilation=2, dropout=dropout)
        self.to3 = ConvBNAct(c2, c3, k=3, s=1)

        self.aa3 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv3 = ConvBNAct(c3, c3, k=3, s=1)
        self.res3 = ResBlock(c3, dilation=2, dropout=dropout)
        self.to4 = ConvBNAct(c3, c4, k=3, s=1)

        self.se = SEBlock(c4, r=se_ratio) if use_se else nn.Identity()
        self.align = nn.AdaptiveAvgPool2d((70, 70))

        self.mix = nn.Sequential(ConvBNAct(c4, c4, k=3, s=1), ResBlock(c4, dilation=1, dropout=dropout))
        self.out = nn.Conv2d(c4, C_out, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 1 and H == 1000 and W == 70, "Expected input shape (B, 1, 1000, 70)."

        h = self.stem(x)

        h = self.aa1(h)
        h = self.conv1(h)
        h = self.res1(h)
        h = self.to2(h)

        h = self.aa2(h)
        h = self.conv2(h)
        h = self.res2(h)
        h = self.to3(h)

        h = self.aa3(h)
        h = self.conv3(h)
        h = self.res3(h)
        h = self.to4(h)

        h = self.se(h)
        h = self.align(h)

        h = self.mix(h)
        y = self.out(h)
        return y


if __name__ == "__main__":
    x = torch.randn(2, 1, 1000, 70)
    enc = SeismicImageEncoderA(c1=32, c2=64, c3=96, c4=128, C_out=64, use_se=True, dropout=0.0)
    y = enc(x)
    print("Output:", y.shape)  # Expected: (2, 64, 70, 70)
