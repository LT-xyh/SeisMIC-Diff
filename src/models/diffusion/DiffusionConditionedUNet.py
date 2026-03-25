import torch
import torch.nn as nn

# ========= 条件 UNet（前一条里给过；这里内嵌，做到开箱即用） =========
try:
    from diffusers.models import UNet2DConditionModel, UNet2DModel
except ImportError as e:
    raise ImportError("Please install diffusers>=0.25.0: pip install diffusers") from e


class DiffusionConditionedUNet(nn.Module):
    """
    噪声预测器：支持三种条件融合方式（默认 crossattn）
    forward(z_t, t, cond_embed) -> eps_pred，与调度器对接
    """

    def __init__(self, latent_channels: int = 16, latent_size: int = 16, cond_channels: int = 64,
                 mode: str = "crossattn",  # 'crossattn' | 'concat' | 'adapter'
                 token_pool: int | None = None,  # crossattn 下可选的平均池化降 token
                 block_out_channels: tuple[int, ...] = (128, 256, 256, 512), attention_head_dim: int = 8, ):
        super().__init__()
        assert mode in {"crossattn", "concat", "adapter"}
        self.mode = mode
        self.token_pool = token_pool
        self.latent_size = latent_size

        n_levels = len(block_out_channels)

        if mode == "crossattn":
            down_block_types = tuple("CrossAttnDownBlock2D" if i < n_levels - 1 else "DownBlock2D" for i in
                                     range(n_levels))  # 下采样块, 最后一层使用DownBlock2D
            up_block_types = tuple(
                "UpBlock2D" if i == 0 else "CrossAttnUpBlock2D" for i in range(n_levels))  # 上采样块, 第一层使用UpBlock2D
            self.unet = UNet2DConditionModel(sample_size=latent_size, in_channels=latent_channels,
                                             out_channels=latent_channels, block_out_channels=block_out_channels,
                                             down_block_types=down_block_types, up_block_types=up_block_types,
                                             cross_attention_dim=cond_channels, attention_head_dim=attention_head_dim, )
        elif mode == "concat":
            # down_block_types = tuple("DownBlock2D" if i == 0 else "AttnDownBlock2D" for i in range(n_levels))
            # up_block_types = tuple("AttnUpBlock2D" if i < n_levels - 1 else "UpBlock2D" for i in range(n_levels))
            down_block_types = tuple("DownBlock2D" for _ in range(n_levels))
            up_block_types = tuple("UpBlock2D" for _ in range(n_levels))
            self.unet = UNet2DModel(sample_size=latent_size, in_channels=latent_channels + cond_channels,
                                    out_channels=latent_channels, block_out_channels=block_out_channels,
                                    down_block_types=down_block_types, up_block_types=up_block_types, )
        else:  # adapter
            self.unet = UNet2DConditionModel(sample_size=latent_size, in_channels=latent_channels,
                                             out_channels=latent_channels, block_out_channels=block_out_channels,
                                             down_block_types=tuple("DownBlock2D" for _ in range(n_levels)),
                                             up_block_types=tuple("UpBlock2D" for _ in range(n_levels)), )
            self.adapter = _PyramidAdapter(in_channels=cond_channels, level_channels=block_out_channels,
                                           n_levels=n_levels, )

    @staticmethod
    def _to_tokens(cond: torch.Tensor, pool: int | None):
        if pool is not None and pool > 1:
            cond = F.avg_pool2d(cond, kernel_size=pool, stride=pool)
        b, c, h, w = cond.shape
        return cond.permute(0, 2, 3, 1).reshape(b, h * w, c)

    def forward(self, z_t: torch.Tensor,  # (b,16,16,16)
                t: torch.LongTensor | int | float,  # (b,)
                cond_embed: torch.Tensor,  # (b,64,16,16)
                encoder_attention_mask: torch.Tensor | None = None, ) -> torch.Tensor:
        # z_t = z_t.to(dtype=self.unet.dtype, device=next(self.unet.parameters()).device)
        # cond_embed = cond_embed.to(dtype=self.unet.dtype, device=z_t.device)
        # t = t.to(device=z_t.device)

        if self.mode == "crossattn":
            tokens = self._to_tokens(cond_embed, self.token_pool)  # (b,L,64)
            return self.unet(sample=z_t, timestep=t, encoder_hidden_states=tokens,
                             encoder_attention_mask=encoder_attention_mask, return_dict=True, ).sample
        elif self.mode == "concat":
            x = torch.cat([z_t, cond_embed], dim=1)
            return self.unet(sample=x, timestep=t, return_dict=True).sample
        else:  # adapter
            down_residuals, mid_residual = self.adapter(cond_embed)
            return self.unet(sample=z_t, timestep=t, down_block_additional_residuals=down_residuals,
                             mid_block_additional_residual=mid_residual, return_dict=True, ).sample


class _PyramidAdapter(nn.Module):
    def __init__(self, in_channels: int, level_channels: tuple[int, ...], n_levels: int):
        super().__init__()
        self.n_levels = n_levels
        convs = []
        ch_in = in_channels
        for i in range(n_levels):
            ch_out = level_channels[i]
            convs.append(nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1 if i == 0 else 2, padding=1),
                                       nn.GroupNorm(num_groups=min(32, ch_out), num_channels=ch_out), nn.SiLU(),
                                       nn.Conv2d(ch_out, ch_out, kernel_size=1), ))
            ch_in = ch_out
        self.level_convs = nn.ModuleList(convs)
        mid_ch = level_channels[-1]
        self.mid_proj = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
                                      nn.GroupNorm(num_groups=min(32, mid_ch), num_channels=mid_ch), nn.SiLU(), )

    def forward(self, cond: torch.Tensor):
        feats = []
        x = cond
        for i in range(self.n_levels):
            x = self.level_convs[i](x)
            feats.append(x)
        mid = self.mid_proj(feats[-1])
        return feats, mid


import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput


class LatentConditionalDiffusion(nn.Module):
    """
    一个打包好的扩散过程模块：
      - 训练：random t, add_noise / q_sample, 预测噪声，计算损失 (simple MSE 或 p2-weighting)
      - 采样：统一使用 scheduler.set_timesteps + scheduler.step
              如果 scheduler=DDPMScheduler -> DDPM 随机采样
              如果 scheduler=ProbabilityFlowODEScheduler -> DDIM / Probability-Flow ODE 确定性采样

    适配形状：
      x0 / z_t : (B, 16, 16, 16)
      cond     : (B, 64, 16, 16) 或你的条件字典（由 DiffusionConditionedUNet 自己处理）
    """

    def __init__(self, scheduler_type: str = "ode",  # 'ddpm' | 'ode'
            num_train_timesteps: int = 1000, ):
        super().__init__()

        # ===== 1. 选择 scheduler =====
        if scheduler_type.lower() == "ddpm":
            # HuggingFace 官方 DDPM 调度器
            self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_start=1e-4, beta_end=2e-2,
                beta_schedule="linear", prediction_type="epsilon", clip_sample=False, thresholding=False, )
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        self.scheduler_type = scheduler_type.lower()

        # ===== 2. 条件 U-Net，用来预测噪声 eps_t =====
        self.noise_model = DiffusionConditionedUNet(latent_channels=16, latent_size=16, cond_channels=64,
            mode="crossattn",  # 'crossattn' | 'concat' | 'adapter'
            token_pool=None,  # crossattn 下可选的平均池化降 token
            block_out_channels=(128, 128, 256), attention_head_dim=8, )

    # ---------- 工具：从 scheduler.alphas_cumprod 抽取 alpha_bar_t ----------
    def _extract_alpha_bar(self, timesteps: torch.LongTensor, x_shape, device, dtype):
        """
        根据 batch t（形状 (B,)）从 scheduler.alphas_cumprod 中取出 alpha_bar_t，
        并 reshape 成 (B,1,1,1,...) 方便广播。
        """
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        # timesteps: (B,)
        alpha_bar = alphas_cumprod[timesteps]  # (B,)
        # -> (B,1,1,1)
        return alpha_bar.view(-1, *([1] * (len(x_shape) - 1)))

    # ---------- 训练前向 ----------
    def training_loss(self, x0: torch.Tensor,  # (B,16,16,16) 目标 latent
                      cond,  # 条件，可以是 (B,64,16,16) 或条件字典
                      loss_type: str = "mse",  # 'mse' | 'p2'
                      p2_k: float = 1.0, p2_gamma: float = 0.5):
        """
        返回 dict(loss=..., noise_pred=..., t=..., x0_pred=...)
        """
        device = x0.device
        dtype = x0.dtype
        B = x0.shape[0]

        # 1) 随机采样每个样本的时间步 t ∈ {0, ..., T-1}
        #    对齐 diffusers：使用 scheduler.num_train_timesteps
        max_t = getattr(self.scheduler.config, "num_train_timesteps", None)
        if max_t is None:
            # Fallback: 某些 scheduler 没有 num_train_timesteps，就用 alphas_cumprod 长度
            max_t = self.scheduler.alphas_cumprod.shape[0]
        t = torch.randint(0, max_t, (B,), device=device, dtype=torch.long)

        # 2) 生成噪声，并根据 scheduler 的 forward 公式加噪
        noise = torch.randn_like(x0)
        # 对齐 diffusers：使用 add_noise（内部等价于 x_t = sqrt(ā_t)x0 + sqrt(1-ā_t)eps）
        xt = self.scheduler.add_noise(x0, noise, t)  # (B,16,16,16)

        # 3) 预测噪声 eps_t
        #    注意：模型接受的 t 是 (B,) 形状
        noise_pred = self.noise_model(xt, t, cond)

        # 4) 可选：根据 eps_hat 反推 x0_pred（用于监控/约束）
        alpha_bar_t = self._extract_alpha_bar(t, x0.shape, device, dtype)
        x0_pred = (xt - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t + 1e-8)

        # 5) 损失：DDPM 标准是 MSE(eps_hat, eps)，也可做 p2 reweight
        if loss_type == "mse":
            loss = F.mse_loss(noise_pred, noise)
        elif loss_type == "p2":
            # p2 reweighting: w_t = (SNR_t)^gamma / (k + (SNR_t)^gamma)
            # SNR_t = alpha_bar_t / (1 - alpha_bar_t)
            snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)
            w = (snr ** p2_gamma) / (p2_k + (snr ** p2_gamma))
            loss = (w * (noise_pred - noise) ** 2).mean()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        return {"loss": loss, "noise_pred": noise_pred, "t": t, "x0_pred": x0_pred, }

    # ---------- 采样 ----------
    @torch.no_grad()
    def sample(self, cond,  # 条件，与训练时同格式
               x_size,  # 例如 (B,16,16,16)
               num_inference_steps: int = 50):
        """
        统一采样接口：
          - 如果 scheduler 是 DDPMScheduler -> DDPM 随机采样
          - 如果 scheduler 是 ProbabilityFlowODEScheduler -> 确定性 DDIM / 概率流 ODE 采样

        都是走 diffusers 风格：
          set_timesteps -> [for t in timesteps: scale_model_input -> UNet -> scheduler.step]
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        B = x_size[0]

        # 1) 设置推理时间步（diffusers 习惯：timesteps 为降序）
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # (num_inference_steps,)

        # 2) 初始噪声 x_T ~ N(0,I)
        x_t = torch.randn(x_size, device=device, dtype=dtype)

        # 3) 反向采样：从 T -> 0
        for t in timesteps:
            # diffusers 的 scheduler.step 接受标量 timestep（int 或 0-dim tensor）
            t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)

            # 某些 scheduler 会缩放 model input，例如 Karras, DPM-Solver 等
            model_in = self.scheduler.scale_model_input(x_t, t)

            # 预测噪声 eps_t
            eps_pred = self.noise_model(model_in, t_batch, cond)

            # 单步更新：返回 SchedulerOutput(prev_sample=...)
            out = self.scheduler.step(model_output=eps_pred, timestep=t_int, sample=x_t, return_dict=True, )
            # 对 DDPM / 你的 ODE scheduler，step 都返回 SchedulerOutput
            if isinstance(out, SchedulerOutput):
                x_t = out.prev_sample
            else:
                # 兼容某些 scheduler 可能返回 tuple(prev_sample, ...)
                x_t = out[0]

        # 此时 x_t ~ 近似 x_0
        return x_t


# ================= 使用示例 =================
if __name__ == "__main__":
    B = 2
    device = "cuda"
    z0 = torch.randn(B, 16, 16, 16).to(device)  # 仅用于演示训练前向
    cond = torch.randn(B, 64, 16, 16).to(device)


    def test_conditioned_unet():
        t = torch.tensor([1, 2])
        model = DiffusionConditionedUNet(latent_channels=16, latent_size=16, cond_channels=64, mode="crossattn",
                                         # 'crossattn' | 'concat' | 'adapter'
                                         token_pool=None,  # crossattn 下可选的平均池化降 token
                                         block_out_channels=(128, 256, 256, 512), attention_head_dim=8, )

        pre = model(z_t=z0, t=t, cond_embed=cond)
        print("pre shape:", pre.shape)


    def test_latent_diffusion():
        ldm = LatentConditionalDiffusion(scheduler_type="ddpm", num_train_timesteps=1000).to(device)

        # 训练
        out = ldm.training_loss(z0, cond, loss_type="mse")
        loss = out["loss"]
        loss.backward()

        # 采样
        z0_hat = ldm.sample(cond, x_size=(B, 16, 16, 16), num_inference_steps=1000)

        print("train loss:", loss.item(), "t: ", out["t"])

        print("sample shape:", z0_hat.shape)


    # test_conditioned_unet()
    test_latent_diffusion()
