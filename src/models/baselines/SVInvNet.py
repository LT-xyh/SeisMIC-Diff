import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# basic blocks
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
                                 nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), )

    def forward(self, x):
        return self.net(x)


class DenseBlock3(nn.Module):
    """
    Paper: each dense block has 3 layers, each Conv2d+BN+ReLU,
    with c_j = 64, output concatenation => 192 channels. :contentReference[oaicite:4]{index=4}
    """

    def __init__(self, in_ch: int, growth: int = 64):
        super().__init__()
        self.growth = growth
        self.l1 = ConvBNReLU(in_ch, growth, k=3, s=1, p=1)
        self.l2 = ConvBNReLU(growth, growth, k=3, s=1, p=1)
        self.l3 = ConvBNReLU(growth * 2, growth, k=3, s=1, p=1)

    def forward(self, x0):
        x1 = self.l1(x0)  # (B, g, H, W)
        x2 = self.l2(x1)  # (B, g, H, W)
        x12 = torch.cat([x1, x2], dim=1)  # (B, 2g, H, W)
        x3 = self.l3(x12)  # (B, g, H, W)
        y0 = torch.cat([x1, x2, x3], dim=1)  # (B, 3g=192, H, W)
        return y0


class DenseStage(nn.Module):
    """
    Repeat N dense blocks at same spatial size.
    Collect each block's compressed output for later concatenation (as paper says
    outputs of dense blocks with same feature size are concatenated before transition). :contentReference[oaicite:5]{index=5}
    """

    def __init__(self, ch: int, num_blocks: int, growth: int = 64):
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([DenseBlock3(ch, growth=growth) for _ in range(num_blocks)])
        self.compress = nn.ModuleList([ConvBNReLU(3 * growth, ch, k=1, s=1, p=0) for _ in range(num_blocks)])

    def forward(self, x):
        outs = []
        for blk, comp in zip(self.blocks, self.compress):
            y = blk(x)  # (B, 192, H, W)
            y = comp(y)  # (B, ch, H, W)
            x = y  # keep same-size feature flowing
            outs.append(y)
        cat = torch.cat(outs, dim=1)  # (B, num_blocks*ch, H, W)
        return x, cat


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x, size_hw):
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.conv(x)


# -------------------------
# SVInvNet multi-constraint (OpenFWI-like 70x70) version
# -------------------------
class MultiConstraintSVInvNet(nn.Module):
    """
    Input:
      migrated_image: (B,1,1000,70)
      rms_vel:        (B,1,1000,70)
      horizon:        (B,1,70,70)
      well_log:       (B,1,70,70)
      well_mask:      (B,1,70,70)  (recommended)

    Output:
      depth_vel_pred: (B,1,70,70)  in [-1,1] by tanh
    """

    def __init__(self, base_ch=64, cond_ch=16, growth=64, use_tanh=True):
        super().__init__()
        self.use_tanh = use_tanh

        # ---- 1D-CNN stem (time reduction) ----
        # keep the "1D kernels to reduce time axis" idea from the paper :contentReference[oaicite:6]{index=6}
        # use Conv2d with kernel (k,1): acts like 1D along time
        self.stem = nn.Sequential(ConvBNReLU(2, base_ch, k=(7, 1), s=(2, 1), p=(3, 0)),  # 1000 -> 500
                                  ConvBNReLU(base_ch, base_ch, k=(7, 1), s=(2, 1), p=(3, 0)),  # 500 -> 250
                                  ConvBNReLU(base_ch, base_ch, k=(7, 1), s=(2, 1), p=(3, 0)),  # 250 -> 125
                                  ConvBNReLU(base_ch, base_ch, k=(7, 1), s=(2, 1), p=(3, 0)),  # 125 -> 63
                                  ConvBNReLU(base_ch, base_ch, k=(3, 1), s=(1, 1), p=(1, 0)), )
        # ensure exact (70,70) without changing the rest of SVInvNet’s encoder/decoder shapes
        self.stem_pool = nn.AdaptiveAvgPool2d((70, 70))

        # ---- condition embedding on depth grid (70x70) ----
        self.cond_embed = nn.Sequential(ConvBNReLU(2, cond_ch, k=3, s=1, p=1),
                                        ConvBNReLU(cond_ch, cond_ch, k=3, s=1, p=1), )
        self.fuse70 = ConvBNReLU(base_ch + cond_ch, base_ch, k=1, s=1, p=0)

        # ---- Encoder: 70 -> 18 -> 9 -> 6 (min size 6x6 as paper) :contentReference[oaicite:7]{index=7}
        self.enc70 = DenseStage(base_ch, num_blocks=3,
                                growth=growth)  # like "34x34 has 3 dense blocks" :contentReference[oaicite:8]{index=8}
        self.down70_18 = ConvBNReLU(base_ch * 3, base_ch * 2, k=3, s=4, p=1)  # 70 -> 18

        self.enc18 = DenseStage(base_ch * 2, num_blocks=2,
                                growth=growth)  # paper: 18x18 has 2 dense blocks :contentReference[oaicite:9]{index=9}
        self.down18_9 = ConvBNReLU((base_ch * 2) * 2, base_ch * 4, k=3, s=2, p=1)  # 18 -> 9

        self.enc9 = DenseStage(base_ch * 4, num_blocks=1,
                               growth=growth)  # paper: 9x9 has 1 dense block :contentReference[oaicite:10]{index=10}
        self.down9_6 = ConvBNReLU(base_ch * 4, base_ch * 8, k=4, s=1, p=0)  # 9 -> 6

        self.enc6 = DenseStage(base_ch * 8, num_blocks=1,
                               growth=growth)  # paper: 6x6 has 1 dense block :contentReference[oaicite:11]{index=11}

        # ---- Decoder: 6 -> 9 -> 18 -> 35 -> 70 ----
        self.up6_9 = UpConv(base_ch * 8, base_ch * 4)
        self.up9_18 = UpConv(base_ch * 4, base_ch * 2)

        # skip at 18x18 (paper concatenates encoder dense-block outputs to decoder) :contentReference[oaicite:12]{index=12}
        self.fuse18 = ConvBNReLU(base_ch * 2 + (base_ch * 2) * 2, base_ch * 2, k=1, s=1, p=0)

        self.up18_35 = UpConv(base_ch * 2, base_ch)
        self.up35_70 = UpConv(base_ch, base_ch)

        # skip at 70x70 (openfwi-sized variant; align with "34x34 skip" idea) :contentReference[oaicite:13]{index=13}
        self.fuse70_dec = ConvBNReLU(base_ch + base_ch * 3, base_ch, k=1, s=1, p=0)
        self.dec70_dense = DenseStage(base_ch, num_blocks=1, growth=growth)

        self.out = nn.Conv2d(base_ch, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, migrated_image: torch.Tensor, rms_vel: torch.Tensor, horizon: torch.Tensor,
                well_log: torch.Tensor, ):
        # time-domain inputs: (B,2,1000,70)
        x_time = torch.cat([migrated_image, rms_vel], dim=1)

        # stem -> (B,base,70,70)
        x = self.stem(x_time)
        x = self.stem_pool(x)

        # depth conditions -> (B,cond,70,70)

        x_cond_in = torch.cat([horizon, well_log], dim=1)
        x_cond = self.cond_embed(x_cond_in)

        # fuse at 70x70
        x = self.fuse70(torch.cat([x, x_cond], dim=1))

        # ---- Encoder ----
        x70, skip70 = self.enc70(x)  # skip70: (B, 3*base, 70, 70)
        x18_in = self.down70_18(skip70)  # (B, 2*base, 18, 18)

        x18, skip18 = self.enc18(x18_in)  # skip18: (B, 2*(2*base), 18, 18)
        x9_in = self.down18_9(skip18)  # (B, 4*base, 9, 9)

        x9, skip9 = self.enc9(x9_in)  # skip9: (B, 1*(4*base), 9, 9)
        x6_in = self.down9_6(skip9)  # (B, 8*base, 6, 6)

        x6, _ = self.enc6(x6_in)  # (B, 8*base, 6, 6)

        # ---- Decoder ----
        d9 = self.up6_9(x6, (9, 9))  # (B, 4*base, 9, 9)
        d18 = self.up9_18(d9, (18, 18))  # (B, 2*base, 18, 18)

        # skip at 18
        d18 = self.fuse18(torch.cat([d18, skip18], dim=1))

        d35 = self.up18_35(d18, (35, 35))  # (B, base, 35, 35)
        d70 = self.up35_70(d35, (70, 70))  # (B, base, 70, 70)

        # skip at 70
        d70 = self.fuse70_dec(torch.cat([d70, skip70], dim=1))
        d70, _ = self.dec70_dense(d70)

        y = self.out(d70)
        if self.use_tanh:
            y = torch.tanh(y)
        else:
            y = torch.sigmoid(y)  # if you later switch dataset normalization to [0,1]
        return y


def _quick_test():
    B = 2
    migrated = torch.randn(B, 1, 1000, 70)
    rms = torch.randn(B, 1, 1000, 70)
    horizon = torch.randn(B, 1, 70, 70)
    well = torch.randn(B, 1, 70, 70)

    net = MultiConstraintSVInvNet(base_ch=64, cond_ch=16, growth=64, use_tanh=True)
    y = net(migrated, rms, horizon, well)
    print("out:", y.shape)


if __name__ == "__main__":
    _quick_test()
