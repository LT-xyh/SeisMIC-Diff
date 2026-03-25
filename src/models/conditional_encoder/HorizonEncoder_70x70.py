import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 基础与增强组件 =====================

class ConvBNGELU2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1, bias=False, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, int):
                p = ((k - 1) // 2) * d
            else:
                p = (((k[0] - 1) // 2) * d, ((k[1] - 1) // 2) * d)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock2d(nn.Module):
    def __init__(self, ch, k=3, d1=1, d2=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNGELU2d(ch, ch, k=k, d=d1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNGELU2d(ch, ch, k=k, d=d2, act=False)
        self.out = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return self.out(x + y)


class SEBlock(nn.Module):
    """ 通道注意力（Squeeze-Excitation） """

    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), 1)
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class SpatialAttn(nn.Module):
    """ CBAM 风格的空间注意力：聚焦细线位置 """

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        m = torch.cat([x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)], dim=1)
        w = torch.sigmoid(self.conv(m))
        return x * w


# ---- 固定高斯模糊：生成 soft-band（增粗稀疏细线） ----
def make_gaussian_kernel(size=7, sigma=1.5, device="cpu"):
    ax = torch.arange(size, device=device) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma))
    ker = ker / ker.sum()
    return ker  # (K,K)


class FixedGaussianBlur2d(nn.Module):
    """ 单通道固定高斯核，作为可微“增粗/平滑” """

    def __init__(self, k=7, sigma=1.5):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.register_buffer("weight", torch.zeros(1, 1, k, k), persistent=False)
        self._inited = False

    def _maybe_init(self, device):
        if (not self._inited) or (self.weight.device != device):
            ker = make_gaussian_kernel(self.k, self.sigma, device=device)
            self.weight.data = ker.view(1, 1, self.k, self.k)
            self._inited = True

    def forward(self, x):
        self._maybe_init(x.device)
        return F.conv2d(x, self.weight, stride=1, padding=self.k // 2)


def add_coord_channels(x):
    """ 为 [B,1,70,70] 或 [B,C,70,70] 添加 y/x 坐标通道（归一化到[-1,1]） """
    B, _, H, W = x.shape
    device = x.device
    yy = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
    xx = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, yy, xx], dim=1)


# ===================== Horizon Encoder（方案A） =====================

class HorizonEncoderA(nn.Module):
    """
    稀疏友好 Horizon 编码器（方案A）
    输入:  (B, 1, 70, 70)  —— 二值层位图：界面=1，其它=0（或接近0）
    输出:  (B, C_out, 70, 70)

    设计要点：
      • 输入增强：拼接 soft-band（高斯模糊增粗）与 y/x 坐标 -> 4 通道
      • 保持原分辨率（不下采样），用扩张卷积扩大感受野，避免细线丢失
      • 轻量注意力：SE + SpatialAttn 聚焦界面区域
      • 1×1 投影到 C_out（默认 16），可直接供融合模块/下游使用
    """

    def __init__(self, C_out: int = 16, c1: int = 32, c2: int = 48, c3: int = 64, k_gauss: int = 7,
                 sigma_gauss: float = 1.5, use_se: bool = True, use_spatial_attn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.blur = FixedGaussianBlur2d(k=k_gauss, sigma=sigma_gauss)

        in_ch = 4  # [mask, soft_band, y, x]
        self.stem = ConvBNGELU2d(in_ch, c1, k=3)

        # 不降采样，堆叠带扩张的残差块，提升感受野同时保留像素级定位
        self.block1 = ResBlock2d(c1, d1=1, d2=1, dropout=dropout)
        self.block2 = ResBlock2d(c1, d1=2, d2=2, dropout=dropout)
        self.to_c2 = ConvBNGELU2d(c1, c2, k=3)

        self.block3 = ResBlock2d(c2, d1=1, d2=2, dropout=dropout)
        self.block4 = ResBlock2d(c2, d1=4, d2=1, dropout=dropout)
        self.to_c3 = ConvBNGELU2d(c2, c3, k=3)

        self.block5 = ResBlock2d(c3, d1=2, d2=4, dropout=dropout)

        self.se = SEBlock(c3, r=8) if use_se else nn.Identity()
        self.spa = SpatialAttn(k=7) if use_spatial_attn else nn.Identity()

        # 轻混合与输出
        self.mix = ConvBNGELU2d(c3, c3, k=3)
        self.head = nn.Conv2d(c3, C_out, kernel_size=1, bias=True)

    def _build_aug_input(self, x):
        """
        x: (B,1,70,70)
        返回: (B,4,70,70) -> [mask, soft_band, y, x]
        """
        soft = self.blur(x)
        xin = add_coord_channels(torch.cat([x, soft], dim=1))  # (B,4,70,70)
        return xin

    def forward(self, x):  # x: (B,1,70,70)
        B, C, H, W = x.shape
        assert C == 1 and H == 70 and W == 70, "期望输入为 (B,1,70,70)"

        xin = self._build_aug_input(x)  # (B,4,70,70)

        h = self.stem(xin)  # (B,c1,70,70)
        h = self.block2(self.block1(h))  # (B,c1,70,70)

        h = self.to_c2(h)  # (B,c2,70,70)
        h = self.block4(self.block3(h))  # (B,c2,70,70)

        h = self.to_c3(h)  # (B,c3,70,70)
        h = self.block5(h)  # (B,c3,70,70)

        h = self.se(h)
        h = self.spa(h)

        h = self.mix(h)  # (B,c3,70,70)
        y = self.head(h)  # (B,C_out,70,70)
        return y


# ===================== 简单自测 =====================
if __name__ == "__main__":
    B = 2
    x = torch.zeros(B, 1, 70, 70)
    # 造几条示例层位线
    x[:, :, 20, :] = 1.0
    x[:, :, 35, 10:60] = 1.0
    x[:, :, 50, 5:40] = 1.0

    net = HorizonEncoderA(C_out=16, c1=32, c2=48, c3=64, k_gauss=7, sigma_gauss=1.5, use_se=True, use_spatial_attn=True,
                          dropout=0.0)
    y = net(x)
    print("Output:", y.shape)  # 期望 (B, 16, 70, 70)
