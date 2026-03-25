import torch.nn as nn


# ---------- 基础模块 ----------
class ConvBNGELU1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):  # (B,C,T)
        return self.act(self.bn(self.conv(x)))


class ConvBNGELU2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):  # (B,C,H,W)
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


# ---------- 方案A：RMS 1D→70 下采样 + 2D 精炼 ----------
class RMSVelocityEncoderA(nn.Module):
    """
    输入:  x ∈ R^{B, 1, 1000, 70}  (time, x)
    输出:  y ∈ R^{B, C_out, 70, 70}

    流程:
      1) 逐列 1D 卷积堆叠 (核 7/5, GELU, BN)，提取纵向趋势
      2) 自适应平均池化到 70 (防混叠重采样) : 1000 -> 70
      3) 重排为 2D 特征 (B, C_t, 70, 70)
      4) 小型 2D CNN (2-3个3x3残差块) 做横向一致性/去噪
      5) 1x1 投影到 C_out (默认 32)
    """

    def __init__(self, C_t: int = 16,  # 1D阶段产生的时序通道数 (建议 8/16)
                 C_mid: int = 64,  # 2D阶段的中间通道
                 C_out: int = 64,  # 最终输出通道
                 n_res2d: int = 3,  # 2D残差块数量 (2或3)
                 dropout2d: float = 0.0  # 2D残差块中的Dropout
                 ):
        super().__init__()
        self.C_t = C_t
        self.C_mid = C_mid
        self.C_out = C_out

        # 1) 逐列 1D 卷积堆叠: 1 -> C_t
        self.temporal1 = ConvBNGELU1d(1, C_t, k=7, s=1)  # (B*W, C_t, T)
        self.temporal2 = ConvBNGELU1d(C_t, C_t, k=5, s=1)  # (B*W, C_t, T)
        self.pool1d = nn.AdaptiveAvgPool1d(70)  # 1000 -> 70 (防混叠)

        # 2D 精炼: 先把 C_t 提升到 C_mid
        self.stem2d = ConvBNGELU2d(C_t, C_mid, k=3)
        blocks = [ResBlock2d(C_mid, dropout=dropout2d) for _ in range(n_res2d)]
        self.refine2d = nn.Sequential(*blocks)

        # 1x1 投影得到目标通道
        self.proj = nn.Conv2d(C_mid, C_out, kernel_size=1, bias=True)

        self.decoder = RMS2DepthDecoder(C_in=self.C_out)

    def forward(self, x):  # x: (B,1,1000,70)
        B, C, T, W = x.shape
        assert C == 1 and T == 1000 and W == 70, "期望输入为 (B,1,1000,70)"

        # --- 逐列处理为 1D 序列 ---
        # (B,1,T,W) -> (B,W,1,T) -> (B*W,1,T)
        t = x.permute(0, 3, 1, 2).contiguous().view(B * W, 1, T)

        h = self.temporal1(t)  # (B*W, C_t, T)
        h = self.temporal2(h)  # (B*W, C_t, T)
        h = self.pool1d(h)  # (B*W, C_t, 70)

        # 还原为 2D 特征图: (B,W,C_t,70) -> (B,C_t,70,W=70)
        h = h.view(B, W, self.C_t, 70).permute(0, 2, 3, 1).contiguous()  # (B,C_t,70,70)

        # 2D 细化
        h = self.stem2d(h)  # (B,C_mid,70,70)
        h = self.refine2d(h)  # (B,C_mid,70,70)

        # 1x1 投影
        y = self.proj(h)  # (B,C_out,70,70)
        return y


import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 小模块 ----------
class ConvBNGELU2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = (k - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):  # (B,C,H,W)
        return self.act(self.bn(self.conv(x)))


class ResBlock2d(nn.Module):
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
    """ 给 [B,C,H,W] 添加 2 个坐标通道 (y,x) ∈ [-1,1]，返回 [B,C+2,H,W] """
    B, _, H, W = x.shape
    device = x.device
    yy = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
    xx = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, yy, xx], dim=1)


# ---------- 解码器主体 ----------
class RMS2DepthDecoder(nn.Module):
    """
    将 RMS encoder 输出的特征 [B, C_in, 70, 70] 映射为深度域速度 [B, 1, 70, 70]。
    结构要点：
      * 坐标增强：默认拼接 (y,x) 坐标，帮助网络学“深度位置语义”
      * 多感受野：堆叠 dilated 残差块 (d=1,2,4)
      * 轻量多尺度上下文：35×35 的粗尺度上下文上采并残差合并
    """

    def __init__(self, C_in: int,  # 来自 encoder 的通道数
                 C_mid: int = 64, n_blocks: int = 4, dilations=(1, 2, 4, 1), use_coords: bool = True,
                 dropout: float = 0.0, out_activation: str = "none"  # "none" | "relu" | "softplus" | "sigmoid"
                 ):
        super().__init__()
        self.use_coords = use_coords
        stem_in = C_in + (2 if use_coords else 0)

        # stem
        self.stem = ConvBNGELU2d(stem_in, C_mid, k=3)

        # 一组 dilated ResBlocks
        blocks = []
        for i in range(n_blocks):
            d = dilations[i % len(dilations)]
            blocks.append(ResBlock2d(C_mid, dilation=d, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        # 粗尺度上下文：下采到 35×35，再上采回 70×70，残差融合
        self.down = nn.Conv2d(C_mid, C_mid, kernel_size=3, stride=2, padding=1, bias=False)  # 70->35
        self.down_bn = nn.BatchNorm2d(C_mid)
        self.ctx = ResBlock2d(C_mid, dilation=2, dropout=dropout)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # 35->70
        self.mix = ConvBNGELU2d(C_mid, C_mid, k=3)

        # 输出
        self.head = nn.Conv2d(C_mid, 1, kernel_size=1, bias=True)
        self.out_activation = out_activation

    def forward(self, f_rms):  # (B,C_in,70,70)
        h = add_coord_channels(f_rms) if self.use_coords else f_rms
        h = self.stem(h)  # (B,C_mid,70,70)
        h = self.blocks(h)  # 多感受野聚合

        # 粗尺度上下文
        c = F.gelu(self.down_bn(self.down(h)))  # (B,C_mid,35,35)
        c = self.ctx(c)
        c = self.up(c)  # (B,C_mid,70,70)
        h = self.mix(h + c)  # 融合

        y = self.head(h)  # (B,1,70,70)

        if self.out_activation == "relu":
            y = F.relu(y)
        elif self.out_activation == "softplus":
            y = F.softplus(y)  # 若速度非负、且希望可导且>0
        elif self.out_activation == "sigmoid":
            y = torch.sigmoid(y)  # 若已做归一化到 [0,1]
        return y


# ------------------ 简单自测 ------------------
if __name__ == "__main__":
    B = 2
    x = torch.randn(B, 1, 1000, 70)
    model = RMSVelocityEncoderA(C_t=16, C_mid=64, C_out=64, n_res2d=3, dropout2d=0.0)
    y = model(x)
    decoder = RMS2DepthDecoder(C_in=64)
    z = decoder(y)
    print("Output shape:", y.shape, z.shape)  # 应为 (B, 32, 70, 70)
