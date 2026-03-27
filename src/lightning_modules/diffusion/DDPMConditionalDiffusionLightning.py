import torch
import torch.nn.functional as F
from diffusers import EMAModel

from lightning_modules.Autoencoder.autoencoder_kl_lightning import AutoencoderKLLightning
from lightning_modules.base_lightning import BaseLightningModule
from models.conditional_encoder.CondFusionPyramid70 import CondFusionPyramid70
from models.diffusion.DiffusionConditionedUNet import LatentConditionalDiffusion


class DDPMConditionalDiffusionLightning(BaseLightningModule):
    """Lightning wrapper for conditional latent DDPM training and evaluation."""

    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr)
        self.conf = conf

        # Load the frozen autoencoder used to map velocity models into the latent space.
        autoencoder = AutoencoderKLLightning.load_from_checkpoint(conf.autoencoder_conf.autoencoder_checkpoint_path)
        for param in autoencoder.parameters():
            param.requires_grad = False
        self.vae = autoencoder.vae
        del autoencoder

        self.ldm = LatentConditionalDiffusion(
            scheduler_type=self.conf.latent_diffusion.scheduler.scheduler_type,
            num_train_timesteps=self.conf.latent_diffusion.scheduler.num_train_timesteps,
        )

        self.num_val_timesteps = self.conf.latent_diffusion.scheduler.num_val_timesteps
        self.num_test_timesteps = self.conf.latent_diffusion.scheduler.num_test_timesteps

        # Encode the four conditioning modalities into a shared multi-scale representation.
        self.ldm_cond_encoder = CondFusionPyramid70()

        if self.conf.training.use_ema:
            self._ema_parameters = [p for p in self.parameters() if p.requires_grad]
            self.ema = EMAModel(parameters=self._ema_parameters, use_ema_warmup=True, foreach=True, power=0.75)

    def training_step(self, batch, batch_idx):
        depth_velocity = batch.pop("depth_vel")

        ldm_cond_embedding = self.ldm_cond_encoder(batch)["s16"]
        del batch

        posterior = self.vae.encode(depth_velocity)
        latents = posterior.sample()
        ldm_dict = self.ldm.training_loss(x0=latents, cond=ldm_cond_embedding, loss_type="mse")
        with torch.no_grad():
            recon_z = ldm_dict["x0_pred"].detach()
            reconstructions = self.vae.decode(recon_z)

        loss = ldm_dict["loss"]
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)

        if self.conf.training.use_ema:
            self.ema.step(self._ema_params())

        with torch.no_grad():
            mse = F.mse_loss(depth_velocity, reconstructions)
            mae = F.l1_loss(depth_velocity, reconstructions)
        self.log("train/mse", mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", mae.detach(), on_step=False, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            self.train_metrics.update(depth_velocity, reconstructions)

        return loss

    def validation_step(self, batch, batch_idx):
        depth_velocity = batch.pop("depth_vel")

        ldm_cond_embedding = self.ldm_cond_encoder(batch)["s16"]
        del batch

        recon_z = self.ldm.sample(cond=ldm_cond_embedding, x_size=(depth_velocity.shape[0], 16, 16, 16),
                                  num_inference_steps=self.num_val_timesteps)
        with torch.no_grad():
            reconstructions = self.vae.decode(recon_z)

        with torch.no_grad():
            mse = F.mse_loss(depth_velocity, reconstructions)
            mae = F.l1_loss(depth_velocity, reconstructions)
        self.log("val/mse", mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", mae.detach(), on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            self.val_metrics.update(reconstructions, depth_velocity)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        return mse

    def test_step(self, batch, batch_idx):
        depth_velocity = batch.pop("depth_vel")
        well_log = batch.get("well_log", None)

        ldm_cond_embedding = self.ldm_cond_encoder(batch)["s16"]

        recon_z = self.ldm.sample(cond=ldm_cond_embedding, x_size=(depth_velocity.shape[0], 16, 16, 16),
                                  num_inference_steps=self.num_test_timesteps)
        with torch.no_grad():
            reconstructions = self.vae.decode(recon_z)

        with torch.no_grad():
            mse = F.mse_loss(depth_velocity, reconstructions)
            mae = F.l1_loss(depth_velocity, reconstructions)
        self.log("test/mse", mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", mae.detach(), on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            self.test_metrics.update(reconstructions, depth_velocity)
            well_metrics = self.well_match_metrics(reconstructions, depth_velocity, well_log)
            if well_metrics is not None:
                self.log("test/well_mae", well_metrics["well_mae"].detach(), on_step=False, on_epoch=True,
                         prog_bar=True)
                self.log("test/well_mse", well_metrics["well_mse"].detach(), on_step=False, on_epoch=True,
                         prog_bar=True)
                self.log("test/well_cc", well_metrics["well_cc"].detach(), on_step=False, on_epoch=True,
                         prog_bar=True)
        self._last_test_batch = (depth_velocity.detach(), reconstructions.detach())
        if batch_idx < 2:
            # Persist a small sample of predictions plus well-log visualizations for offline inspection.
            self.save_batch_torch(batch_idx, reconstructions, save_dir=f"{self.conf.testing.test_save_dir}",
                                  well_log=well_log)

        return mse
