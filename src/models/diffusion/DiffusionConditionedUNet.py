import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput

try:
    from diffusers.models import UNet2DConditionModel, UNet2DModel
except ImportError as e:
    raise ImportError("Please install diffusers>=0.25.0: pip install diffusers") from e


class DiffusionConditionedUNet(nn.Module):
    """
    Noise predictor with three condition fusion modes.

    ``forward(z_t, t, cond_embed)`` returns the predicted noise ``eps_t``.
    """

    def __init__(self, latent_channels: int = 16, latent_size: int = 16, cond_channels: int = 64,
                 mode: str = "crossattn", token_pool: int | None = None,
                 block_out_channels: tuple[int, ...] = (128, 256, 256, 512), attention_head_dim: int = 8):
        super().__init__()
        assert mode in {"crossattn", "concat", "adapter"}
        self.mode = mode
        self.token_pool = token_pool
        self.latent_size = latent_size

        n_levels = len(block_out_channels)

        if mode == "crossattn":
            down_block_types = tuple("CrossAttnDownBlock2D" if i < n_levels - 1 else "DownBlock2D"
                                     for i in range(n_levels))
            up_block_types = tuple("UpBlock2D" if i == 0 else "CrossAttnUpBlock2D" for i in range(n_levels))
            self.unet = UNet2DConditionModel(sample_size=latent_size, in_channels=latent_channels,
                                             out_channels=latent_channels, block_out_channels=block_out_channels,
                                             down_block_types=down_block_types, up_block_types=up_block_types,
                                             cross_attention_dim=cond_channels,
                                             attention_head_dim=attention_head_dim)
        elif mode == "concat":
            down_block_types = tuple("DownBlock2D" for _ in range(n_levels))
            up_block_types = tuple("UpBlock2D" for _ in range(n_levels))
            self.unet = UNet2DModel(sample_size=latent_size, in_channels=latent_channels + cond_channels,
                                    out_channels=latent_channels, block_out_channels=block_out_channels,
                                    down_block_types=down_block_types, up_block_types=up_block_types)
        else:
            self.unet = UNet2DConditionModel(sample_size=latent_size, in_channels=latent_channels,
                                             out_channels=latent_channels, block_out_channels=block_out_channels,
                                             down_block_types=tuple("DownBlock2D" for _ in range(n_levels)),
                                             up_block_types=tuple("UpBlock2D" for _ in range(n_levels)))
            self.adapter = _PyramidAdapter(in_channels=cond_channels, level_channels=block_out_channels,
                                           n_levels=n_levels)

    @staticmethod
    def _to_tokens(cond: torch.Tensor, pool: int | None):
        if pool is not None and pool > 1:
            cond = F.avg_pool2d(cond, kernel_size=pool, stride=pool)
        b, c, h, w = cond.shape
        return cond.permute(0, 2, 3, 1).reshape(b, h * w, c)

    def forward(self, z_t: torch.Tensor, t: torch.LongTensor | int | float, cond_embed: torch.Tensor,
                encoder_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "crossattn":
            tokens = self._to_tokens(cond_embed, self.token_pool)
            return self.unet(sample=z_t, timestep=t, encoder_hidden_states=tokens,
                             encoder_attention_mask=encoder_attention_mask, return_dict=True).sample
        elif self.mode == "concat":
            x = torch.cat([z_t, cond_embed], dim=1)
            return self.unet(sample=x, timestep=t, return_dict=True).sample
        else:
            down_residuals, mid_residual = self.adapter(cond_embed)
            return self.unet(sample=z_t, timestep=t, down_block_additional_residuals=down_residuals,
                             mid_block_additional_residual=mid_residual, return_dict=True).sample


class _PyramidAdapter(nn.Module):
    def __init__(self, in_channels: int, level_channels: tuple[int, ...], n_levels: int):
        super().__init__()
        self.n_levels = n_levels
        convs = []
        ch_in = in_channels
        for i in range(n_levels):
            ch_out = level_channels[i]
            convs.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1 if i == 0 else 2, padding=1),
                nn.GroupNorm(num_groups=min(32, ch_out), num_channels=ch_out),
                nn.SiLU(),
                nn.Conv2d(ch_out, ch_out, kernel_size=1),
            ))
            ch_in = ch_out
        self.level_convs = nn.ModuleList(convs)
        mid_ch = level_channels[-1]
        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, mid_ch), num_channels=mid_ch),
            nn.SiLU(),
        )

    def forward(self, cond: torch.Tensor):
        feats = []
        x = cond
        for i in range(self.n_levels):
            x = self.level_convs[i](x)
            feats.append(x)
        mid = self.mid_proj(feats[-1])
        return feats, mid


class LatentConditionalDiffusion(nn.Module):
    """
    Packaged latent diffusion module with a consistent training and sampling interface.

    Shapes:
        x0 / z_t: (B, 16, 16, 16)
        cond: (B, 64, 16, 16)
    """

    def __init__(self, scheduler_type: str = "ode", num_train_timesteps: int = 1000):
        super().__init__()

        if scheduler_type.lower() == "ddpm":
            self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_start=1e-4, beta_end=2e-2,
                                           beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
                                           thresholding=False)
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        self.scheduler_type = scheduler_type.lower()

        self.noise_model = DiffusionConditionedUNet(
            latent_channels=16,
            latent_size=16,
            cond_channels=64,
            mode="crossattn",
            token_pool=None,
            block_out_channels=(128, 128, 256),
            attention_head_dim=8,
        )

    def _extract_alpha_bar(self, timesteps: torch.LongTensor, x_shape, device, dtype):
        """Gather ``alpha_bar_t`` for a batch of timesteps and reshape for broadcasting."""
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        alpha_bar = alphas_cumprod[timesteps]
        return alpha_bar.view(-1, *([1] * (len(x_shape) - 1)))

    def training_loss(self, x0: torch.Tensor, cond, loss_type: str = "mse", p2_k: float = 1.0,
                      p2_gamma: float = 0.5):
        """Return ``dict(loss=..., noise_pred=..., t=..., x0_pred=...)`` for diffusion training."""
        device = x0.device
        dtype = x0.dtype
        B = x0.shape[0]

        max_t = getattr(self.scheduler.config, "num_train_timesteps", None)
        if max_t is None:
            max_t = self.scheduler.alphas_cumprod.shape[0]
        t = torch.randint(0, max_t, (B,), device=device, dtype=torch.long)

        noise = torch.randn_like(x0)
        xt = self.scheduler.add_noise(x0, noise, t)

        noise_pred = self.noise_model(xt, t, cond)

        alpha_bar_t = self._extract_alpha_bar(t, x0.shape, device, dtype)
        x0_pred = (xt - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t + 1e-8)

        if loss_type == "mse":
            loss = F.mse_loss(noise_pred, noise)
        elif loss_type == "p2":
            snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)
            w = (snr ** p2_gamma) / (p2_k + (snr ** p2_gamma))
            loss = (w * (noise_pred - noise) ** 2).mean()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        return {"loss": loss, "noise_pred": noise_pred, "t": t, "x0_pred": x0_pred}

    @torch.no_grad()
    def sample(self, cond, x_size, num_inference_steps: int = 50):
        """
        Unified sampling interface following the standard diffusers loop:
        ``set_timesteps -> scale_model_input -> UNet -> scheduler.step``.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        B = x_size[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        x_t = torch.randn(x_size, device=device, dtype=dtype)

        for t in timesteps:
            t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)

            # Some schedulers scale the current latent before the model forward pass.
            model_in = self.scheduler.scale_model_input(x_t, t)
            eps_pred = self.noise_model(model_in, t_batch, cond)

            out = self.scheduler.step(model_output=eps_pred, timestep=t_int, sample=x_t, return_dict=True)
            if isinstance(out, SchedulerOutput):
                x_t = out.prev_sample
            else:
                x_t = out[0]

        return x_t


if __name__ == "__main__":
    B = 2
    device = "cuda"
    z0 = torch.randn(B, 16, 16, 16).to(device)
    cond = torch.randn(B, 64, 16, 16).to(device)

    def test_conditioned_unet():
        t = torch.tensor([1, 2])
        model = DiffusionConditionedUNet(latent_channels=16, latent_size=16, cond_channels=64, mode="crossattn",
                                         token_pool=None, block_out_channels=(128, 256, 256, 512),
                                         attention_head_dim=8)
        pre = model(z_t=z0, t=t, cond_embed=cond)
        print("pre shape:", pre.shape)

    def test_latent_diffusion():
        ldm = LatentConditionalDiffusion(scheduler_type="ddpm", num_train_timesteps=1000).to(device)

        out = ldm.training_loss(z0, cond, loss_type="mse")
        loss = out["loss"]
        loss.backward()

        z0_hat = ldm.sample(cond, x_size=(B, 16, 16, 16), num_inference_steps=1000)

        print("train loss:", loss.item(), "t: ", out["t"])
        print("sample shape:", z0_hat.shape)

    # test_conditioned_unet()
    test_latent_diffusion()
