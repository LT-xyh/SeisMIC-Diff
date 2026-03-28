import os

import lightning
import torch
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import Subset, DataLoader

from lightning_modules.baselines.DixLightning import DixLightning
from data.dataset_openfwi import OpenFWI


def base_test_dix(model, test_set, batch_size, log_dir, log_version, fast_run=False):
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True,
                             pin_memory=True, prefetch_factor=3)

    tensorboard_logger = TensorBoardLogger(save_dir=log_dir, name='tensorboard', version=log_version, )
    csv_logger = CSVLogger(save_dir=log_dir, name="csv", version=log_version, )
    trainer = lightning.Trainer(precision='bf16-mixed',  # fp16混合精度训练
                                accelerator="gpu",  # strategy='ddp_spawn',
                                devices=[0],  # 指定要使用的 GPU 编号
                                logger=[tensorboard_logger, csv_logger], log_every_n_steps=512 // batch_size,
                                fast_dev_run=fast_run, )
    trainer.test(model, dataloaders=test_loader)


def test_dix_OpenFWI(batch_size, log_dir, log_version, fast_run=False):
    for dataset in ('FlatVelA', 'FlatVelB', 'CurveVelA', 'CurveVelB'):
        log_version = f'{log_version}_{dataset}'
        test_image_save_dir = os.path.join(log_dir, 'test_torch', log_version)
        model = DixLightning(batch_size=batch_size, test_image_save_dir=test_image_save_dir)
        dataset = OpenFWI(root_dir='data/openfwi', use_data=('depth_vel', 'rms_vel', 'well_log'), datasets=(dataset,),
                          use_normalize=None)
        total_size = len(dataset)
        test_size = int(0.1 * total_size)  # 取最后10%作为测试集（非随机）
        test_idx = list(range(total_size - test_size, total_size))
        test_set = Subset(dataset, test_idx)
        base_test_dix(model, test_set, batch_size, log_dir, log_version, fast_run=fast_run)


def test_dix():
    batch_size = 100
    log_dir = 'logs/baselines/smooth_dix'
    log_version = '260120_'
    test_dix_OpenFWI(batch_size, log_dir, log_version, fast_run=True)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    test_dix()
