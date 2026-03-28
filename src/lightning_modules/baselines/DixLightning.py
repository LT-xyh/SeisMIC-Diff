import torch.nn.functional as F

from baselines.smooth_dix.dix import SmoothDix
from lightning_modules.base_lightning import BaseLightningModule


class DixLightning(BaseLightningModule):
    def __init__(self, batch_size, lr=2e-5, test_image_save_dir='./logs/dix/'):
        super().__init__(batch_size, lr, data_range=2.0)
        self.dix = SmoothDix()
        self.vmax = 4500.0
        self.vmin = 1500.0
        self.test_image_save_dir = test_image_save_dir

    def test_step(self, batch, batch_idx):
        # 1. 数据
        depth_vel = batch.pop('depth_vel')  # 不做归一化
        well_log = batch.get('well_log', None)
        rms_vel = batch['rms_vel']
        del batch
        recon, _ = self.dix(rms_vel)
        # 3. 损失
        depth_vel = ((depth_vel - self.vmin) / (self.vmax - self.vmin)) * 2 - 1.0  # [-1, 1]
        recon = ((recon - self.vmin) / (self.vmax - self.vmin)) * 2 - 1.0  # [-1, 1]
        if well_log is not None:
            well_log = ((well_log - self.vmin) / (self.vmax - self.vmin)) * 2 - 1.0
        mse = F.mse_loss(depth_vel, recon)
        mae = F.l1_loss(depth_vel, recon)
        self.log('test/mse', mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mae', mae.detach(), on_step=False, on_epoch=True, prog_bar=True)

        # 4. 评价指标
        self.test_metrics.update(recon, depth_vel)
        well_metrics = self.well_match_metrics(recon, depth_vel, well_log)
        if well_metrics is not None:
            self.log('test/well_mae', well_metrics['well_mae'].detach(), on_step=False, on_epoch=True, prog_bar=True)
            self.log('test/well_mse', well_metrics['well_mse'].detach(), on_step=False, on_epoch=True, prog_bar=True)
            self.log('test/well_cc', well_metrics['well_cc'].detach(), on_step=False, on_epoch=True, prog_bar=True)
        self._last_test_batch = (depth_vel.detach(), recon.detach())
        if batch_idx < 2:
            self.save_batch_torch(batch_idx, recon, save_dir=self.test_image_save_dir, well_log=well_log)
        return mse


if __name__ == '__main__':
    for dataset_name in ('FlatVelA', 'FlatVelB', 'CurveVelA', 'CurveVelB', 'CurveFaultA'):
        print(f'\n\n\n{dataset_name}')

