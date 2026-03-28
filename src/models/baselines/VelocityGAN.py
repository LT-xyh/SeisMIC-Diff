from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# utils: center crop / pad
# -------------------------
def center_crop(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Crops the center of a tensor to size out_hw (H, W).
    """
    _, _, H, W = x.shape
    oh, ow = out_hw
    if H == oh and W == ow:
        return x
    top = max((H - oh) // 2, 0)
    left = max((W - ow) // 2, 0)
    # Ensure we don't go out of bounds if input is smaller (defensive)
    h_end = min(top + oh, H)
    w_end = min(left + ow, W)
    return x[:, :, top:h_end, left:w_end]


def center_pad(x: torch.Tensor, out_hw: Tuple[int, int], value: float = 0.0) -> torch.Tensor:
    """
    Pads a tensor to size out_hw (H, W) with value.
    """
    _, _, H, W = x.shape
    oh, ow = out_hw
    pad_h = max(oh - H, 0)
    pad_w = max(ow - W, 0)
    if pad_h == 0 and pad_w == 0:
        return x
    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bot), mode="constant", value=value)


def pad_time_to(x: torch.Tensor, target_H: int, value: float = 0.0) -> torch.Tensor:
    """
    Pads only the Height (Time) dimension to target_H.
    """
    _, _, H, _ = x.shape
    if H >= target_H:
        return x
    pad = target_H - H
    pad_top = pad // 2
    pad_bot = pad - pad_top
    return F.pad(x, (0, 0, pad_top, pad_bot), mode="constant", value=value)


# -------------------------
# basic blocks (BN + LeakyReLU)
# -------------------------
class ConvBNLReLU(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True), )

    def forward(self, x): return self.net(x)


class DeconvBNLReLU(nn.Module):
    def __init__(self, cin, cout, k, s, p, out_pad=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=k, stride=s, padding=p, output_padding=out_pad, bias=False),
            nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True), )

    def forward(self, x): return self.net(x)


# -------------------------
# Generator (Physics-Aware Domain Separation)
# -------------------------
class VelocityGAN_Generator_MC(nn.Module):
    """
    Multi-constraint generator with Domain Separation:
    1. Encoder: Processes Time-Domain data (Migrated Image, RMS Vel).
    2. Decoder: Reconstructs Depth-Domain features.
    3. Injection: Depth-Domain constraints (Wells, Horizon) are injected
       at the end of the Decoder to preserve spatial accuracy.

    Inputs:
      - migrated_image: (B,1,1000,70) [Time]
      - rms_vel:        (B,1,1000,70) [Time]
      - horizon:        (B,1,70,70)   [Depth]
      - well_log:       (B,1,70,70)   [Depth]
      - well_mask:      (B,1,70,70)   [Depth] 1.0=valid, 0.0=invalid
    Output:
      - depth_vel_pred: (B,1,70,70) in [-1,1]
    """

    def __init__(self, in_ch: int = 2, cond_ch: int = 2, base: int = 32, target_hw=(70, 70)):
        super().__init__()
        self.target_hw = target_hw

        # ==========================================
        # 1. Time-Domain Encoder (Input: Mig + RMS)
        # ==========================================
        # Make time H from 1024 -> 64 in 4 downsamples.
        # Channels: in_ch (2) -> base
        self.t1 = ConvBNLReLU(in_ch, base, k=(7, 1), s=(2, 1), p=(3, 0))  # 1024->512
        self.t2 = ConvBNLReLU(base, base * 2, k=(3, 1), s=(2, 1), p=(1, 0))  # 512->256
        self.t3 = ConvBNLReLU(base * 2, base * 4, k=(3, 1), s=(2, 1), p=(1, 0))  # 256->128
        self.t4 = ConvBNLReLU(base * 4, base * 8, k=(3, 1), s=(2, 1), p=(1, 0))  # 128->64

        # ==========================================
        # 2. Domain Transition (64x64 -> 1x1)
        # ==========================================
        def enc_block(cin, cout):
            return nn.Sequential(ConvBNLReLU(cin, cout, k=3, s=2, p=1), ConvBNLReLU(cout, cout, k=3, s=1, p=1), )

        self.e1 = enc_block(base * 8, base * 8)  # 64->32
        self.e2 = enc_block(base * 8, base * 8)  # 32->16
        self.e3 = enc_block(base * 8, base * 8)  # 16->8

        # "Global" bottleneck
        self.conv_global = nn.Sequential(nn.Conv2d(base * 8, base * 16, kernel_size=8, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 16), nn.LeakyReLU(0.2, inplace=True), )

        # ==========================================
        # 3. Depth-Domain Decoder (1x1 -> 64x64)
        # ==========================================
        self.d0 = DeconvBNLReLU(base * 16, base * 8, k=8, s=1, p=0)  # 1->8
        self.d1 = DeconvBNLReLU(base * 8, base * 4, k=4, s=2, p=1)  # 8->16
        self.d2 = DeconvBNLReLU(base * 4, base * 2, k=4, s=2, p=1)  # 16->32
        self.d3 = DeconvBNLReLU(base * 2, base, k=4, s=2, p=1)  # 32->64

        # ==========================================
        # 4. Constraint Fusion Block
        # ==========================================
        # Input channels = Decoder features (base) + Conditions (cond_ch: well+mask+horizon)
        self.fusion_conv = ConvBNLReLU(base + cond_ch, base, k=3, s=1, p=1)

        self.out_conv = nn.Conv2d(base, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, migrated_image: torch.Tensor, rms_vel: torch.Tensor, horizon: torch.Tensor,
                well_log: torch.Tensor) -> torch.Tensor:
        # -----------------------------------------------
        # Step 1: Encoder (Time Domain Processing)
        # -----------------------------------------------
        # Only concatenate Time-domain inputs
        x_time = torch.cat([migrated_image, rms_vel], dim=1)  # (B, 2, 1000, 70)

        # Pad time to 1024 for clean downsampling
        x_time = pad_time_to(x_time, 1024, value=0.0)  # (B, 2, 1024, 70)

        # Time-axis reduction (Squareification)
        x = self.t1(x_time)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)  # (B, 256, 64, 70)
        x = center_crop(x, (64, 64))  # (B, 256, 64, 64) - "Pseudo-depth" latent map

        # Spatial Encoder (Bottleneck)
        x = self.e1(x)  # 32
        x = self.e2(x)  # 16
        x = self.e3(x)  # 8
        x = self.conv_global(x)  # 1x1 Vector

        # -----------------------------------------------
        # Step 2: Decoder (Depth Domain Reconstruction)
        # -----------------------------------------------
        x = self.d0(x)  # 8
        x = self.d1(x)  # 16
        x = self.d2(x)  # 32
        x = self.d3(x)  # 64x64 feature map (base channels)

        # -----------------------------------------------
        # Step 3: Injection of Constraints (Depth Domain)
        # -----------------------------------------------
        # We inject constraints here to keep high spatial resolution
        # Concatenate: [Feature(64x64), Well(64x64), Mask(64x64), Horizon(64x64)]

        # Prepare conditions: (B, 2, 70, 70)
        conds = torch.cat([well_log, horizon], dim=1)

        # Crop conditions to match decoder feature size (64x64)
        conds_cropped = center_crop(conds, (64, 64))

        # Concatenate along channel dim
        x_fused = torch.cat([x, conds_cropped], dim=1)  # (B, base+3, 64, 64)

        # Fuse features
        x_fused = self.fusion_conv(x_fused)  # (B, base, 64, 64)

        # -----------------------------------------------
        # Step 4: Final Output
        # -----------------------------------------------
        out = self.out_conv(x_fused)  # (B, 1, 64, 64)

        # Resize to target without introducing constant-value borders.
        out = F.interpolate(out, size=self.target_hw, mode="bilinear", align_corners=False)

        return self.tanh(out)


# -------------------------
# Discriminator (Standard PatchGAN)
# -------------------------
class VelocityGAN_Discriminator_Patch4(nn.Module):
    """
    PatchGAN-style critic:
    - 4 times MaxPool2d(2) => 70->35->17->8->4 (approx patch grid 4x4)
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()

        def block(cin, cout, pool=True):
            layers = [nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True), ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.b1 = block(in_ch, base, pool=True)  # 70->35
        self.b2 = block(base, base * 2, pool=True)  # 35->17
        self.b3 = block(base * 2, base * 4, pool=True)  # 17->8
        self.b4 = block(base * 4, base * 8, pool=True)  # 8->4
        self.b5 = block(base * 8, base * 8, pool=False)  # keep 4->4
        self.head = nn.Conv2d(base * 8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        x = self.b1(v)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.head(x)  # (B,1,4,4)

    def score(self, v: torch.Tensor) -> torch.Tensor:
        p = self.forward(v)
        return p.mean(dim=(1, 2, 3))  # (B,)


# -------------------------
# WGAN-GP losses
# -------------------------
def wgan_gp_discriminator_loss(D: VelocityGAN_Discriminator_Patch4, real: torch.Tensor, fake: torch.Tensor,
                               gp_lambda: float = 10.0) -> torch.Tensor:
    B = real.size(0)
    real_score = D.score(real)
    fake_score = D.score(fake.detach())
    loss_w = (fake_score - real_score).mean()

    # gradient penalty
    eps = torch.rand(B, 1, 1, 1, device=real.device, dtype=real.dtype)
    x_hat = eps * real + (1 - eps) * fake.detach()
    x_hat.requires_grad_(True)
    hat_score = D.score(x_hat).sum()
    grads = torch.autograd.grad(hat_score, x_hat, create_graph=True)[0]
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return loss_w + gp_lambda * gp


def generator_loss(D: VelocityGAN_Discriminator_Patch4, fake: torch.Tensor, real: torch.Tensor, l1_w: float = 50.0,
                   l2_w: float = 100.0) -> torch.Tensor:
    adv = -D.score(fake).mean()
    l1 = F.l1_loss(fake, real)
    l2 = F.mse_loss(fake, real)
    return adv + l1_w * l1 + l2_w * l2


# -------------------------
# Smoke Test (Updated for New Inputs)
# -------------------------
def smoke_test_velocitygan_mc(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)

    print(f"--- Running Smoke Test on {device} ---")

    # 1) Shapes
    B = 2
    T, X, Z = 1000, 70, 70

    # 2) Build Inputs
    # Time Domain Inputs
    migrated_image = torch.randn(B, 1, T, X, device=device).clamp(-1, 1)
    rms_vel = torch.randn(B, 1, T, X, device=device).clamp(-1, 1)

    # Depth Domain Inputs
    # Horizon: 1 for interface, 0 for background (then normalized/masked as needed)
    # Here assuming simple -1 to 1 feature map
    horizon = torch.randn(B, 1, Z, X, device=device).clamp(-1, 1)

    # Well Log & Mask
    # Initialize well log with 0 (or -1 if using specific normalization, but here we use 0 + mask)
    well_log = torch.zeros(B, 1, Z, X, device=device)
    well_mask = torch.zeros(B, 1, Z, X, device=device)

    # Simulate wells at random columns
    well_cols = [10, 35, 60]
    for xc in well_cols:
        # Fill data
        well_log[:, :, :, xc] = torch.randn(B, 1, Z, device=device).clamp(-1, 1)

    # Ground Truth
    depth_vel_gt = torch.randn(B, 1, Z, X, device=device).clamp(-1, 1)

    # 3) Build Models
    # in_ch=2 (Mig, RMS), cond_ch=3 (Well, Mask, Horizon)
    G = VelocityGAN_Generator_MC(in_ch=2, cond_ch=2, base=32, target_hw=(70, 70)).to(device).train()
    D = VelocityGAN_Discriminator_Patch4(in_ch=1, base=32).to(device).train()

    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # 4) Forward Pass
    fake = G(migrated_image, rms_vel, horizon, well_log)

    # 5) Update D
    lossD = wgan_gp_discriminator_loss(D, real=depth_vel_gt, fake=fake, gp_lambda=10.0)
    optD.zero_grad()
    lossD.backward()
    optD.step()

    # 6) Update G
    fake = G(migrated_image, rms_vel, horizon, well_log)  # Regenerate
    lossG = generator_loss(D, fake=fake, real=depth_vel_gt, l1_w=50.0, l2_w=100.0)
    optG.zero_grad()
    lossG.backward()
    optG.step()

    print(f"[OK] Test Passed.")
    print(f"Output shape: {tuple(fake.shape)}")
    print(f"Loss D: {lossD.item():.4f}")
    print(f"Loss G: {lossG.item():.4f}")


if __name__ == "__main__":
    smoke_test_velocitygan_mc()
