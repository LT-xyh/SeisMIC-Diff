import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k, s=1, p=0, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(0.2,
                                inplace=True) if act else nn.Identity()  # paper uses alpha=0.2:contentReference[oaicite:8]{index=8}

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeconvBNAct(nn.Module):
    def __init__(self, cin, cout, k, s=2, p=1, out_pad=0, act=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(cin, cout, kernel_size=k, stride=s, padding=p,
                                         output_padding=out_pad, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class MultiConstraintInversionNet(nn.Module):
    """
    Inputs (all in [-1,1]):
      migrated_image: (B,1,1000,70)
      rms_vel       : (B,1,1000,70)
      horizon       : (B,1,70,70)
      well_log      : (B,1,70,70)   (sparse stripes, others -1)

    Output:
      depth_vel_pred: (B,1,70,70) in [-1,1] (if use_tanh=True)
    """

    def __init__(self, base=32, use_tanh: bool = False):
        super().__init__()
        self.use_tanh = use_tanh

        # ---- Time branch: (B,2,1000,70) -> feature on ~70x70
        # Use "time-direction" kernels (k_t,1) to reduce time axis first (paper motivation):contentReference[oaicite:9]{index=9}
        self.time_reduce = nn.Sequential(
            ConvBNAct(2, base, k=(7, 1), s=(2, 1), p=(3, 0)),  # 1000 -> 500
            ConvBNAct(base, base, k=(7, 1), s=(2, 1), p=(3, 0)),  # 500  -> 250
            ConvBNAct(base, base * 2, k=(7, 1), s=(2, 1), p=(3, 0)),  # 250  -> 125
            ConvBNAct(base * 2, base * 2, k=(7, 1), s=(2, 1), p=(3, 0)),  # 125  -> 63
            ConvBNAct(base * 2, base * 2, k=3, s=1, p=1),
        )

        # ---- Depth constraint encoder: (B,2/3,70,70) -> feature on 70x70
        depth_in_ch = 2
        self.depth_enc0 = nn.Sequential(
            ConvBNAct(depth_in_ch, base, k=3, s=1, p=1),
            ConvBNAct(base, base, k=3, s=1, p=1),
        )

        # ---- Fusion + InversionNet-style encoder-decoder
        # Decoder in paper uses deconv blocks to upsample:contentReference[oaicite:10]{index=10}
        fusion_in = base * 2 + base

        # encoder downs: 70 -> 35 -> 18 -> 9 -> 5  (k=3,s=2,p=1 gives ceil-ish)
        self.enc1 = nn.Sequential(ConvBNAct(fusion_in, base * 2, k=3, s=2, p=1),
                                  ConvBNAct(base * 2, base * 2, k=3, s=1, p=1))  # 35
        self.enc2 = nn.Sequential(ConvBNAct(base * 2, base * 4, k=3, s=2, p=1),
                                  ConvBNAct(base * 4, base * 4, k=3, s=1, p=1))  # 18
        self.enc3 = nn.Sequential(ConvBNAct(base * 4, base * 6, k=3, s=2, p=1),
                                  ConvBNAct(base * 6, base * 6, k=3, s=1, p=1))  # 9
        self.enc4 = nn.Sequential(ConvBNAct(base * 6, base * 8, k=3, s=2, p=1),
                                  ConvBNAct(base * 8, base * 8, k=3, s=1, p=1))  # 5

        # decoder ups: 5 -> 9 -> 18 -> 35 -> 70  (use output_padding to hit exact sizes)
        self.dec3 = nn.Sequential(DeconvBNAct(base * 8, base * 6, k=3, s=2, p=1, out_pad=0),
                                  ConvBNAct(base * 6, base * 6, k=3, s=1, p=1))  # 9
        self.dec2 = nn.Sequential(DeconvBNAct(base * 6, base * 4, k=3, s=2, p=1, out_pad=1),
                                  ConvBNAct(base * 4, base * 4, k=3, s=1, p=1))  # 18
        self.dec1 = nn.Sequential(DeconvBNAct(base * 4, base * 2, k=3, s=2, p=1, out_pad=0),
                                  ConvBNAct(base * 2, base * 2, k=3, s=1, p=1))  # 35
        self.dec0 = nn.Sequential(DeconvBNAct(base * 2, base, k=3, s=2, p=1, out_pad=1),
                                  ConvBNAct(base, base, k=3, s=1, p=1))  # 70

        # final regression (paper uses 1x1 to regress per-pixel):contentReference[oaicite:11]{index=11}
        self.head = nn.Conv2d(base, 1, kernel_size=1, stride=1, padding=0)

    def forward(self,
                migrated_image: torch.Tensor,
                rms_vel: torch.Tensor,
                horizon: torch.Tensor,
                well_log: torch.Tensor,
                ) -> torch.Tensor:
        # time inputs
        xt = torch.cat([migrated_image, rms_vel], dim=1)  # (B,2,1000,70)
        ft = self.time_reduce(xt)  # (B,C,~63,70)
        # print(ft.shape)  [3, 64, 63, 70]
        ft = F.interpolate(ft, size=(70, 70), mode="bilinear", align_corners=False)

        # depth constraints
        xd = torch.cat([horizon, well_log], dim=1)
        fd = self.depth_enc0(xd)  # (B,base,70,70)
        # print(fd.shape)  [3, 32, 70, 70]

        x = torch.cat([ft, fd], dim=1)  # (B,base*3,70,70)

        # encoder-decoder
        x = self.enc1(x)  # 35
        # print(f'e1: {x.shape}')
        x = self.enc2(x)  # 18
        # print(f'e2: {x.shape}')
        x = self.enc3(x)  # 9
        # print(f'e3: {x.shape}')
        x = self.enc4(x)  # 5
        # print(f'e4: {x.shape}')

        x = self.dec3(x)  # 9
        # print(f'd3: {x.shape}')
        x = self.dec2(x)  # 18
        # print(f'd2: {x.shape}')
        x = self.dec1(x)  # 35
        # print(f'd1: {x.shape}')
        x = self.dec0(x)  # 70
        # print(f'd0: {x.shape}')

        y = self.head(x)  # (B,1,70,70)
        return torch.tanh(y) if self.use_tanh else y


def test_InversionNet():
    batch_size = 3
    pstm = torch.randn(batch_size, 1, 1000, 70)
    vrms = torch.randn(batch_size, 1, 1000, 70)
    horizons = torch.randn(batch_size, 1, 70, 70)
    well_log = torch.randn(batch_size, 1, 70, 70)
    model = MultiConstraintInversionNet()

    out = model(pstm, vrms, horizons, well_log)
    print(out.shape)


if __name__ == '__main__':
    test_InversionNet()
