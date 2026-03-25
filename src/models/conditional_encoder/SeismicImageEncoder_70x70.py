import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- 基础模块 -----------------
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
    """ 轻量通道注意力（可选） """

    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), 1)
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


# ----------------- 方案A：各向异性 CNN 金字塔 -----------------
class SeismicImageEncoderA(nn.Module):
    """
    输入:  (B, 1, 1000, 70)  —— [垂向H=1000, 横向W=70]
    输出:  (B, C_out, 70, 70)

    设计:
      * 竖向优先的各向异性下采样：AvgPool2d((2,1), stride=(2,1)) * 3  -> H: 1000→500→250→125
      * 每级若干 3×3 卷积 + 1 个空洞残差块 (dilation=2) 扩大垂向感受野
      * 最后以 AdaptiveAvgPool2d 对齐到 (70,70)，再 1×1 投影到 C_out
      * 可选 SE 通道注意力
    """

    def __init__(self, c1=32, c2=64, c3=96, c4=128,  # 各级通道
                 C_out=64,  # 输出通道
                 use_se=True, se_ratio=8, dropout=0.0):
        super().__init__()

        # 预平滑/特征提取
        self.stem = ConvBNAct(1, c1, k=3, s=1)

        # Stage 1: 1000 -> 500
        self.aa1 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))  # 抗混叠下采样(竖向)
        self.conv1 = ConvBNAct(c1, c1, k=3, s=1)
        self.res1 = ResBlock(c1, dilation=2, dropout=dropout)
        self.to2 = ConvBNAct(c1, c2, k=3, s=1)

        # Stage 2: 500 -> 250
        self.aa2 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv2 = ConvBNAct(c2, c2, k=3, s=1)
        self.res2 = ResBlock(c2, dilation=2, dropout=dropout)
        self.to3 = ConvBNAct(c2, c3, k=3, s=1)

        # Stage 3: 250 -> 125
        self.aa3 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv3 = ConvBNAct(c3, c3, k=3, s=1)
        self.res3 = ResBlock(c3, dilation=2, dropout=dropout)
        self.to4 = ConvBNAct(c3, c4, k=3, s=1)

        # 可选通道注意力（在 125×70 尺度）
        self.se = SEBlock(c4, r=se_ratio) if use_se else nn.Identity()

        # 对齐到 (70,70)
        self.align = nn.AdaptiveAvgPool2d((70, 70))

        # 细化 + 输出
        self.mix = nn.Sequential(ConvBNAct(c4, c4, k=3, s=1), ResBlock(c4, dilation=1, dropout=dropout))
        self.out = nn.Conv2d(c4, C_out, kernel_size=1, bias=True)

    def forward(self, x):
        """
        x: (B,1,1000,70)
        """
        B, C, H, W = x.shape
        assert C == 1 and H == 1000 and W == 70, "期望输入为 (B,1,1000,70)"

        h = self.stem(x)  # (B,c1,1000,70)

        h = self.aa1(h)  # (B,c1, 500,70)
        h = self.conv1(h)
        h = self.res1(h)
        h = self.to2(h)  # (B,c2, 500,70)

        h = self.aa2(h)  # (B,c2, 250,70)
        h = self.conv2(h)
        h = self.res2(h)
        h = self.to3(h)  # (B,c3, 250,70)

        h = self.aa3(h)  # (B,c3, 125,70)
        h = self.conv3(h)
        h = self.res3(h)
        h = self.to4(h)  # (B,c4, 125,70)

        h = self.se(h)  # 可选SE
        h = self.align(h)  # (B,c4,  70,70)

        h = self.mix(h)  # (B,c4,  70,70)
        y = self.out(h)  # (B,C_out,70,70)
        return y


# ----------------- 简单自测 -----------------
if __name__ == "__main__":
    x = torch.randn(2, 1, 1000, 70)
    enc = SeismicImageEncoderA(c1=32, c2=64, c3=96, c4=128, C_out=64, use_se=True, dropout=0.0)
    y = enc(x)
    print("Output:", y.shape)  # 期望 (2, 64, 70, 70)
