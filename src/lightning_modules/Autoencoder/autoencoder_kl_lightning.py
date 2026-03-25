import math

import torch
import torch.nn.functional as F
from diffusers.training_utils import EMAModel

from lightning_modules.base_lightning import BaseLightningModule
from models.Autoencoder.AutoencoderKLInterpolation import AutoencoderKLInterpolation


class AutoencoderKLLightning(BaseLightningModule):
    def __init__(self, conf):
        if conf.datasets.use_normalize == '-1_1':
            data_range = 2.0
        else:
            data_range = 1.0
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr, data_range=data_range)
        self.conf = conf
        self.vae = AutoencoderKLInterpolation(latent_channels=conf.autoencoder_conf.latent_channels,
                                              depth_vel_shape=conf.datasets.depth_velocity.shape,
                                              depth_vel_reshape=conf.autoencoder_conf.reshape,
                                              down_block_types=conf.autoencoder_conf.down_block_types,
                                              up_block_types=conf.autoencoder_conf.up_block_types,
                                              block_out_channels=conf.autoencoder_conf.block_out_channels, )
        self.ema = None

        # === KL anneal config ===
        ka = getattr(conf.training, "kl_anneal", None)
        self.ka = {"strategy": (ka.strategy if ka else "linear_epoch"),
                   "warmup_epochs": (ka.warmup_epochs if ka else max(1, int(conf.training.max_epochs * 0.2))),
                   "start": (ka.start if ka else 0.0), "end": (ka.end if ka else conf.training.loss.kl_weight),
                   "cycles": (ka.cycles if ka else 3), "ratio": (ka.ratio if ka else 0.5),
                   "free_bits": (ka.free_bits if ka else 0.0), }
        self.register_buffer("kl_beta", torch.tensor(float(self.ka["start"])))  # 当前 β

        if self.conf.training.use_ema:
            self._ema_parameters = list(self.vae.parameters())
            if self.ema is None:
                self.ema = EMAModel(parameters=self._ema_parameters, use_ema_warmup=True, foreach=True, power=0.75,
                                    device=self.device)

    # —— 每个 epoch 开头刷新一次 β（也可改成按 step 刷新）——
    def on_train_epoch_start(self):
        self.kl_beta.fill_(self._compute_beta_epoch(self.current_epoch))

    def _compute_beta_epoch(self, epoch: int) -> float:
        strat = self.ka["strategy"]
        start, end = float(self.ka["start"]), float(self.ka["end"])
        warm = int(self.ka["warmup_epochs"])
        if strat == "none":
            return end

        if strat == "linear_epoch":
            t = min(1.0, (epoch + 1) / max(1, warm))
            return start + t * (end - start)

        if strat == "cosine_epoch":
            if epoch + 1 <= warm:
                # 0 → 1 的半余弦升温
                t = 0.5 * (1 - math.cos(math.pi * (epoch + 1) / max(1, warm)))
            else:
                t = 1.0
            return start + t * (end - start)

        if strat == "cyclic_epoch":
            cycles = int(self.ka["cycles"])
            rise_ratio = float(self.ka["ratio"])
            total = max(1, self.trainer.max_epochs)
            cycle_len = max(1, total // max(1, cycles))
            pos_in_cycle = (epoch % cycle_len) + 1
            rise_len = max(1, int(cycle_len * rise_ratio))
            if pos_in_cycle <= rise_len:
                t = pos_in_cycle / rise_len
            else:
                t = 1.0
            return start + t * (end - start)

        return end  # fallback

    # —— 你原有的 training_step 里，把 KL 权重替换为 self.kl_beta.item() ——
    def training_step(self, batch, batch_idx):
        depth_velocity = batch.pop('depth_vel')

        posterior = self.vae.encode(depth_velocity)
        latents = posterior.sample()
        reconstructions = self.vae.decode(latents)

        l1_loss = F.l1_loss(reconstructions, depth_velocity)
        mse_loss = F.mse_loss(reconstructions, depth_velocity)

        # KL 原始项
        kl_map = posterior.kl()  # 形状可能是 [B, C, H, W] 或 [B, ...]
        kl_loss_raw = kl_map.mean()

        # ——（可选）free-bits/min-rate：给 KL 一个最小速率阈值，避免过小——
        fb = float(self.ka["free_bits"])
        if fb > 0.0:
            B = kl_map.shape[0]
            kl_per_sample = kl_map.view(B, -1).sum(dim=1)  # 每个样本 KL 总量（nats）
            kl_loss = torch.clamp(kl_per_sample - fb, min=0.0).mean()
        else:
            kl_loss = kl_loss_raw

        beta = float(self.kl_beta.item())
        loss = (
                l1_loss * self.conf.training.loss.l1_weight + mse_loss * self.conf.training.loss.mse_weight + kl_loss * beta)

        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)
        self.log_dict(
            {'train/MAE': l1_loss.detach(), 'train/MSE': mse_loss.detach(), 'train/KL_raw': kl_loss_raw.detach(),
             'train/KL_used': kl_loss.detach(), 'train/beta': beta, }, on_step=True, on_epoch=False, prog_bar=False,
            batch_size=self.conf.training.dataloader.batch_size)

        self.train_metrics.update(reconstructions, depth_velocity)
        if self.conf.training.use_ema:
            self.ema.step(self._ema_params())
        return loss

    def validation_step(self, batch, batch_idx):

        depth_velocity = batch.pop('depth_vel')
        posterior = self.vae.encode(depth_velocity)
        latents = posterior.sample()
        reconstructions = self.vae.decode(latents)

        l1_loss = F.l1_loss(reconstructions, depth_velocity)
        mse_loss = F.mse_loss(reconstructions, depth_velocity)
        kl_loss = posterior.kl().mean()

        beta = float(self.kl_beta.item())  # 验证集同一轮使用同一 β
        loss = (
                l1_loss * self.conf.training.loss.l1_weight + mse_loss * self.conf.training.loss.mse_weight + kl_loss * beta)

        self.log('val/loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)
        self.log_dict(
            {'val/MAE': l1_loss.detach(), 'val/MSE': mse_loss.detach(), 'val/KL': kl_loss.detach(), 'val/beta': beta, },
            on_step=False, on_epoch=True, prog_bar=False, batch_size=self.conf.training.dataloader.batch_size)

        self.val_metrics.update(reconstructions, depth_velocity)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())
        return loss
