import torch
from torch.nn import functional as F
from diffusers import EMAModel

from models.baselines.SVInvNet import MultiConstraintSVInvNet
from lightning_modules.base_lightning import BaseLightningModule



class SVInvNetLightning(BaseLightningModule):
    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr, data_range=2.0)
        self.conf = conf
        self.test_save_dir = conf.testing.test_save_dir
        self.model = MultiConstraintSVInvNet(base_ch=conf.sv_inv_net.base_channel,
                                             cond_ch=conf.sv_inv_net.condition_channel, growth=64, use_tanh=True)
        if self.conf.training.use_ema:
            self._ema_parameters = list(self.model.parameters())
            self.ema = EMAModel(parameters=self._ema_parameters, use_ema_warmup=True, foreach=True, power=0.75,
                                device='cpu')

    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')

        # 2. 模型
        reconstructions = self.model(batch.pop('migrated_image'), batch.pop('rms_vel'), batch.pop('horizon'),
                                     batch.pop('well_log'))
        del batch

        # 3. 损失
        loss = F.mse_loss(reconstructions, depth_velocity)
        self.log('train/mse', loss.detach(), on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)

        if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
            self.ema.step(self._ema_params())
        self.train_metrics.update(depth_velocity, reconstructions)
        return loss

    def validation_step(self, batch, batch_idx):

        depth_velocity = batch.pop('depth_vel')

        # 2. 模型
        reconstructions = self.model(batch.pop('migrated_image'), batch.pop('rms_vel'), batch.pop('horizon'),
                                     batch.pop('well_log'))
        del batch

        # 3. 损失
        mse = F.mse_loss(depth_velocity, reconstructions)
        mae = F.l1_loss(depth_velocity, reconstructions)
        self.log('val/mse', mse.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)
        self.log('val/mae', mae.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)

        # 4. 评价指标
        self.val_metrics.update(depth_velocity, reconstructions)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        return mse

    def test_step(self, batch, batch_idx):
        depth_velocity = batch.pop('depth_vel')
        well_log = batch.get('well_log', None)
        reconstructions = self.model(batch.pop('migrated_image'), batch.pop('rms_vel'), batch.pop('horizon'),
                                     batch.pop('well_log'))
        del batch
        # 3. 损失
        mse = F.mse_loss(depth_velocity, reconstructions)
        mae = F.l1_loss(depth_velocity, reconstructions)
        self.log('test/mse', mse.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)
        self.log('test/mae', mae.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)
        # 4. 评价指标
        self.test_metrics.update(depth_velocity, reconstructions)
        well_metrics = self.well_match_metrics(reconstructions, depth_velocity, well_log)
        if well_metrics is not None:
            self.log('test/well_mae', well_metrics['well_mae'].detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
            self.log('test/well_mse', well_metrics['well_mse'].detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
            self.log('test/well_cc', well_metrics['well_cc'].detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        if batch_idx < 2:
            self.save_batch_torch(batch_idx, reconstructions, save_dir=self.conf.testing.test_save_dir,
                                  well_log=well_log)

        return mse


class MAblationSVInvNetLightning(SVInvNetLightning):
    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        rms_vel = torch.zeros_like(batch.pop('rms_vel'))
        horizon = torch.zeros_like(batch.pop('horizon'))
        well_log = torch.zeros_like(batch.pop('well_log'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=rms_vel, horizons=horizon, wells=well_log)
        del batch

        # 3. 损失
        loss = F.mse_loss(reconstructions, depth_velocity)
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)

        # 4. 评价指标
        self.train_metrics.update(depth_velocity, reconstructions)
        self._last_train_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
            self.ema.step(self._ema_params())

        return loss

    def validation_step(self, batch, batch_idx):
        if self.conf.training.use_ema:
            self.ema.store(self._ema_params())
            self.ema.copy_to(self._ema_params())

        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        rms_vel = torch.zeros_like(batch.pop('rms_vel'))
        horizon = torch.zeros_like(batch.pop('horizon'))
        well_log = torch.zeros_like(batch.pop('well_log'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=rms_vel, horizons=horizon, wells=well_log)
        del batch

        # 3. 损失
        loss = F.mse_loss(depth_velocity, reconstructions)
        self.log('val/mse', loss.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)

        # 4. 评价指标
        self.val_metrics.update(depth_velocity, reconstructions)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:
            self.ema.restore(self._ema_params())
        return loss

    def test_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        rms_vel = torch.zeros_like(batch.pop('rms_vel'))
        horizon = torch.zeros_like(batch.pop('horizon'))
        well_log = torch.zeros_like(batch.pop('well_log'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=rms_vel, horizons=horizon, wells=well_log)
        del batch
        # 3. 损失
        loss = F.mse_loss(depth_velocity, reconstructions)
        self.log('test/mse', loss.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)
        # 4. 评价指标
        self.test_metrics.update(depth_velocity, reconstructions)
        # self.save_batch_images(batch_idx, depth_velocity, reconstructions, self.test_save_dir)
        # self.save_batch_torch(batch_idx, reconstructions, self.test_save_dir)
        return loss


class MVAblationSVInvNetLightning(SVInvNetLightning):
    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        horizon = torch.zeros_like(batch.pop('horizon'))
        well_log = torch.zeros_like(batch.pop('well_log'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=batch.pop('rms_vel'), horizons=horizon,
                                     wells=well_log)
        del batch

        # 3. 损失
        loss = F.mse_loss(reconstructions, depth_velocity)
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)

        # 4. 评价指标
        self.train_metrics.update(depth_velocity, reconstructions)
        self._last_train_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
            self.ema.step(self._ema_params())

        return loss

    def validation_step(self, batch, batch_idx):
        if self.conf.training.use_ema:
            self.ema.store(self._ema_params())
            self.ema.copy_to(self._ema_params())

        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        horizon = torch.zeros_like(batch.pop('horizon'))
        well_log = torch.zeros_like(batch.pop('well_log'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=batch.pop('rms_vel'), horizons=horizon,
                                     wells=well_log)
        del batch

        # 3. 损失
        loss = F.mse_loss(depth_velocity, reconstructions)
        self.log('val/mse', loss.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)

        # 4. 评价指标
        self.val_metrics.update(depth_velocity, reconstructions)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:
            self.ema.restore(self._ema_params())
        return loss

    def test_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        horizon = torch.zeros_like(batch.pop('horizon'))
        well_log = torch.zeros_like(batch.pop('well_log'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=batch.pop('rms_vel'), horizons=horizon,
                                     wells=well_log)
        del batch

        # 3. 损失
        loss = F.mse_loss(depth_velocity, reconstructions)
        self.log('test/mse', loss.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)
        # 4. 评价指标
        self.test_metrics.update(depth_velocity, reconstructions)
        # self.save_batch_images(batch_idx, depth_velocity, reconstructions, self.test_save_dir)
        # self.save_batch_torch(batch_idx, reconstructions, self.test_save_dir)
        return loss


class MHAblationSVInvNetLightning(SVInvNetLightning):
    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        rms_vel = torch.zeros_like(batch.pop('rms_vel'))
        well_log = torch.zeros_like(batch.pop('well_log'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=rms_vel, horizons=batch.pop('horizon'),
                                     wells=well_log)
        del batch

        # 3. 损失
        loss = F.mse_loss(reconstructions, depth_velocity)
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)

        # 4. 评价指标
        self.train_metrics.update(depth_velocity, reconstructions)
        self._last_train_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
            self.ema.step(self._ema_params())

        return loss

    def validation_step(self, batch, batch_idx):
        if self.conf.training.use_ema:
            self.ema.store(self._ema_params())
            self.ema.copy_to(self._ema_params())

        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        rms_vel = torch.zeros_like(batch.pop('rms_vel'))
        well_log = torch.zeros_like(batch.pop('well_log'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=rms_vel, horizons=batch.pop('horizon'),
                                     wells=well_log)
        del batch

        # 3. 损失
        loss = F.mse_loss(depth_velocity, reconstructions)
        self.log('val/mse', loss.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)

        # 4. 评价指标
        self.val_metrics.update(depth_velocity, reconstructions)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:
            self.ema.restore(self._ema_params())
        return loss

    def test_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        rms_vel = torch.zeros_like(batch.pop('rms_vel'))
        well_log = torch.zeros_like(batch.pop('well_log'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=rms_vel, horizons=batch.pop('horizon'),
                                     wells=well_log)
        del batch

        # 3. 损失
        loss = F.mse_loss(depth_velocity, reconstructions)
        self.log('test/mse', loss.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)
        # 4. 评价指标
        self.test_metrics.update(depth_velocity, reconstructions)
        # self.save_batch_images(batch_idx, depth_velocity, reconstructions, self.test_save_dir)
        # self.save_batch_torch(batch_idx, reconstructions, self.test_save_dir)
        return loss


class MWAblationSVInvNetLightning(SVInvNetLightning):
    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        rms_vel = torch.zeros_like(batch.pop('rms_vel'))
        horizon = torch.zeros_like(batch.pop('horizon'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=rms_vel, horizons=horizon,
                                     wells=batch.pop('well_log'))
        del batch

        # 3. 损失
        loss = F.mse_loss(reconstructions, depth_velocity)
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)

        # 4. 评价指标
        self.train_metrics.update(depth_velocity, reconstructions)
        self._last_train_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
            self.ema.step(self._ema_params())

        return loss

    def validation_step(self, batch, batch_idx):
        if self.conf.training.use_ema:
            self.ema.store(self._ema_params())
            self.ema.copy_to(self._ema_params())

        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        rms_vel = torch.zeros_like(batch.pop('rms_vel'))
        horizon = torch.zeros_like(batch.pop('horizon'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=rms_vel, horizons=horizon,
                                     wells=batch.pop('well_log'))
        del batch

        # 3. 损失
        loss = F.mse_loss(depth_velocity, reconstructions)
        self.log('val/mse', loss.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)

        # 4. 评价指标
        self.val_metrics.update(depth_velocity, reconstructions)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:
            self.ema.restore(self._ema_params())
        return loss

    def test_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')
        rms_vel = torch.zeros_like(batch.pop('rms_vel'))
        horizon = torch.zeros_like(batch.pop('horizon'))

        # 2. 模型
        reconstructions = self.model(pstm=batch.pop('migrated_image'), vrms=rms_vel, horizons=horizon,
                                     wells=batch.pop('well_log'))
        del batch

        # 3. 损失
        loss = F.mse_loss(depth_velocity, reconstructions)
        self.log('test/mse', loss.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.conf.training.dataloader.batch_size)
        # 4. 评价指标
        self.test_metrics.update(depth_velocity, reconstructions)
        # self.save_batch_images(batch_idx, depth_velocity, reconstructions, self.test_save_dir)
        return loss
