import torch
from diffusers import EMAModel
from torch.nn import functional as F

from baselines.inversion_net.InversionNet import MultiConstraintInversionNet
from lightning_modules.base_lightning import BaseLightningModule


class InversionNetLightning(BaseLightningModule):
    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr, )
        self.conf = conf
        self.model = MultiConstraintInversionNet(base=conf.inversion_net.base_channel, use_tanh=False)
        self.test_save_dir = conf.testing.test_save_dir
        if self.conf.training.use_ema:
            self._ema_parameters = list(self.model.parameters())
            self.ema = EMAModel(parameters=self._ema_parameters, use_ema_warmup=True, foreach=True, power=0.75,
                                device='cpu')

    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')

        # 2. 模型
        reconstructions = self.model(migrated_image=batch['migrated_image'], rms_vel=batch['rms_vel'],
                                     horizon=batch['horizon'], well_log=batch['well_log'])
        del batch

        # 3. 损失
        loss = F.mse_loss(reconstructions, depth_velocity)
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)

        if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
            self.ema.step(self._ema_params())

        with torch.no_grad():
            self.train_metrics.update(depth_velocity, reconstructions)
        self._last_train_batch = (depth_velocity.detach(), reconstructions.detach())

        return loss

    def validation_step(self, batch, batch_idx):

        depth_velocity = batch.pop('depth_vel')

        # 2. 模型
        reconstructions = self.model(migrated_image=batch['migrated_image'], rms_vel=batch['rms_vel'],
                                     horizon=batch['horizon'], well_log=batch['well_log'])
        del batch

        # 3. 损失
        with torch.no_grad():
            mse = F.mse_loss(depth_velocity, reconstructions)
            mae = F.l1_loss(depth_velocity, reconstructions)
        self.log('val/mse', mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mae', mae.detach(), on_step=False, on_epoch=True, prog_bar=True)

        # 4. 评价指标
        with torch.no_grad():
            self.val_metrics.update(depth_velocity, reconstructions)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        return mse

    def test_step(self, batch, batch_idx):
        depth_velocity = batch.pop('depth_vel')
        well_log = batch.get('well_log', None)

        # 2. 模型
        reconstructions = self.model(migrated_image=batch['migrated_image'], rms_vel=batch['rms_vel'],
                                     horizon=batch['horizon'], well_log=batch['well_log'])
        del batch

        # 3. 损失
        with torch.no_grad():
            mse = F.mse_loss(depth_velocity, reconstructions)
            mae = F.l1_loss(depth_velocity, reconstructions)
        self.log('test/mse', mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mae', mae.detach(), on_step=False, on_epoch=True, prog_bar=True)

        # 4. 评价指标
        with torch.no_grad():
            self.test_metrics.update(depth_velocity, reconstructions)
            well_metrics = self.well_match_metrics(reconstructions, depth_velocity, well_log)
            if well_metrics is not None:
                self.log('test/well_mae', well_metrics['well_mae'].detach(), on_step=False, on_epoch=True,
                         prog_bar=True)
                self.log('test/well_mse', well_metrics['well_mse'].detach(), on_step=False, on_epoch=True,
                         prog_bar=True)
                self.log('test/well_cc', well_metrics['well_cc'].detach(), on_step=False, on_epoch=True, prog_bar=True)
        self._last_test_batch = (depth_velocity.detach(), reconstructions.detach())
        if batch_idx < 2:
            self.save_batch_torch(batch_idx, reconstructions, save_dir=self.conf.testing.test_save_dir,
                                  well_log=well_log)
        return mse
