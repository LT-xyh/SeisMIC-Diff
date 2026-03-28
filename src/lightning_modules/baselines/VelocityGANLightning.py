from diffusers import EMAModel

from baselines.velocity_gan.VelocityGAN import *
from lightning_modules.base_lightning import BaseLightningModule


class VelocityGANLightning(BaseLightningModule):
    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr, data_range=2.0)
        self.conf = conf
        self.test_save_dir = conf.testing.test_save_dir

        # GAN manual optimization
        self.automatic_optimization = False

        vg = getattr(conf, "velocity_gan", None)

        # ===== model =====
        in_ch = getattr(vg, "in_ch", 2) if vg is not None else 2
        cond_ch = getattr(vg, "cond_ch", 2) if vg is not None else 2
        g_base = getattr(vg, "base_channel", 32) if vg is not None else 32
        d_base = getattr(vg, "d_base_channel", 32) if vg is not None else 32
        self.G = VelocityGAN_Generator_MC(in_ch=in_ch, cond_ch=cond_ch, base=g_base, target_hw=(70, 70))
        self.D = VelocityGAN_Discriminator_Patch4(in_ch=1, base=d_base)

        # ===== loss weights (velocity_gan) =====
        self.gp_lambda = float(getattr(vg, "gp_lambda", 10.0)) if vg is not None else 10.0
        self.l1_w = float(getattr(vg, "l1_w", 50.0)) if vg is not None else 50.0
        self.l2_w = float(getattr(vg, "l2_w", 100.0)) if vg is not None else 100.0

        # NEW: adversarial weight + warmup
        self.adv_w = float(getattr(vg, "adv_w", 1.0)) if vg is not None else 1.0
        self.warmup_epochs = int(getattr(vg, "warmup_epochs", 0)) if vg is not None else 0

        # ===== training controls (training) =====
        self.n_critic = int(getattr(conf.training, "n_critic", 5))
        self.lambda_grad = float(getattr(conf.training, "lambda_grad", 0.0))

        self.grad_clip_val = float(getattr(conf.training, "grad_clip_val", 0.0))
        self.grad_clip_algo = str(getattr(conf.training, "grad_clip_algo", "norm")).lower()

        if self.conf.training.use_ema:
            self._ema_parameters = list(self.G.parameters())
            self.ema = EMAModel(parameters=self._ema_parameters, use_ema_warmup=True, foreach=True, power=0.75,
                                device='cpu')

    def configure_optimizers(self):
        betas = tuple(getattr(self.conf.training, "betas", (0.0, 0.9)))
        lr_g = float(getattr(self.conf.training, "lr", 1e-4))
        lr_d = float(getattr(self.conf.training, "lr_d", lr_g))
        opt_g = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=betas)
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=betas)
        return [opt_g, opt_d]

    def _clip_grads_(self, params):
        if self.grad_clip_val <= 0:
            return
        if self.grad_clip_algo == "value":
            torch.nn.utils.clip_grad_value_(params, self.grad_clip_val)
        else:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip_val)

    def _wgan_gp_loss_fp32(self, real, fake):
        # 强制用 fp32 算 GP，避免 bf16-mixed 下抖动/NaN
        device_type = real.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            loss_d = wgan_gp_discriminator_loss(self.D, real=real.float(), fake=fake.float(),
                                                gp_lambda=self.gp_lambda, )
        return loss_d

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        depth_velocity = batch.pop("depth_vel")
        migrated_image = batch["migrated_image"]
        rms_vel = batch["rms_vel"]
        horizon = batch["horizon"]
        well_log = batch["well_log"]
        del batch

        # =========================================================
        # Warmup: only train G with reconstruction (no D, no adv)
        # =========================================================
        if self.current_epoch < self.warmup_epochs:
            self.toggle_optimizer(opt_g)

            recon = self.G(migrated_image, rms_vel, horizon, well_log)

            mae = F.l1_loss(recon, depth_velocity)
            mse = F.mse_loss(recon, depth_velocity)
            loss_g = self.l1_w * mae + self.l2_w * mse

            if self.lambda_grad > 0:
                gl = self.grad_loss_yx(recon, depth_velocity)
                loss_g = loss_g + self.lambda_grad * gl
                self.log("train/grad_loss", gl.detach(), on_step=True, on_epoch=True, prog_bar=False)

            opt_g.zero_grad(set_to_none=True)
            self.manual_backward(loss_g)
            self._clip_grads_(self.G.parameters())
            opt_g.step()
            self.untoggle_optimizer(opt_g)

            if self.ema is not None:
                self.ema.step(self._ema_params())

            self.log("train/loss_g", loss_g.detach(), on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/mae", mae.detach(), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/mse", mse.detach(), on_step=True, on_epoch=True, prog_bar=False)
            self.train_metrics.update(depth_velocity, recon)
            return loss_g

        # =========================================================
        # Normal GAN training: D step then (every n_critic) G step
        # =========================================================

        # ---- (A) D step ----
        self.toggle_optimizer(opt_d)
        with torch.no_grad():
            recon = self.G(migrated_image, rms_vel, horizon, well_log)

        loss_d = self._wgan_gp_loss_fp32(depth_velocity, recon)

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(loss_d)
        self._clip_grads_(self.D.parameters())
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.log("train/loss_d", loss_d.detach(), on_step=True, on_epoch=True, prog_bar=True)

        # ---- (B) G step ----
        do_g = (batch_idx % self.n_critic == 0)
        if do_g:
            self.toggle_optimizer(opt_g)

            recon = self.G(migrated_image, rms_vel, horizon, well_log)

            # adv + recon
            adv = -self.D.score(recon).mean()
            mae = F.l1_loss(recon, depth_velocity)
            mse = F.mse_loss(recon, depth_velocity)

            loss_g = self.adv_w * adv + self.l1_w * mae + self.l2_w * mse

            if self.lambda_grad > 0:
                gl = self.grad_loss_yx(recon, depth_velocity)
                loss_g = loss_g + self.lambda_grad * gl
                self.log("train/grad_loss", gl.detach(), on_step=True, on_epoch=True, prog_bar=False)

            opt_g.zero_grad(set_to_none=True)
            self.manual_backward(loss_g)
            self._clip_grads_(self.G.parameters())
            opt_g.step()
            self.untoggle_optimizer(opt_g)

            if self.ema is not None:
                self.ema.step(self._ema_params())

            # logs (split components)
            self.log("train/loss_g", loss_g.detach(), on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/adv", adv.detach(), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/mae", mae.detach(), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/mse", mse.detach(), on_step=True, on_epoch=True, prog_bar=False)

        # monitor critic gap
        with torch.no_grad():
            d_real = self.D.score(depth_velocity).mean()
            d_fake = self.D.score(recon).mean()
        self.log("train/d_real", d_real, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/d_fake", d_fake, on_step=True, on_epoch=True, prog_bar=False)
        self.train_metrics.update(depth_velocity, recon)

        return loss_d

    def validation_step(self, batch, batch_idx):
        depth_velocity = batch.pop("depth_vel")
        recon = self.G(batch["migrated_image"], batch["rms_vel"], batch["horizon"], batch["well_log"], )
        del batch

        mse = F.mse_loss(recon, depth_velocity)
        mae = F.l1_loss(recon, depth_velocity)
        self.log("val/mse", mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", mae.detach(), on_step=False, on_epoch=True, prog_bar=True)

        self.val_metrics.update(depth_velocity, recon)
        self._last_val_batch = (depth_velocity.detach(), recon.detach())
        return mse

    def test_step(self, batch, batch_idx):
        depth_velocity = batch.pop("depth_vel")
        well_log = batch.get("well_log", None)
        recon = self.G(batch["migrated_image"], batch["rms_vel"], batch["horizon"], batch["well_log"], )
        del batch

        mse = F.mse_loss(recon, depth_velocity)
        mae = F.l1_loss(recon, depth_velocity)
        self.log("test/mse", mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", mae.detach(), on_step=False, on_epoch=True, prog_bar=True)

        self.test_metrics.update(depth_velocity, recon)
        well_metrics = self.well_match_metrics(recon, depth_velocity, well_log)
        if well_metrics is not None:
            self.log("test/well_mae", well_metrics["well_mae"].detach(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/well_mse", well_metrics["well_mse"].detach(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/well_cc", well_metrics["well_cc"].detach(), on_step=False, on_epoch=True, prog_bar=True)
        self._last_test_batch = (depth_velocity.detach(), recon.detach())
        if batch_idx < 2:
            self.save_batch_torch(batch_idx, recon, save_dir=self.conf.testing.test_save_dir, well_log=well_log)
        return mse
