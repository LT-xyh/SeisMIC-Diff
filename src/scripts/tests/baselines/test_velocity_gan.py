from datetime import datetime

import torch
from omegaconf import OmegaConf

from lightning_modules.baselines.VelocityGANLightning import VelocityGANLightning
from scripts.tests.basetest import base_test


def test_velocity_gan(dataset_name, fast_run=False):
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d")
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/velocity_gan.yaml')
    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/{dataset_name}'
    conf.testing.ckpt_path = 'ckpt_path/velocity_gan-ssim_0.772.ckpt'
    conf.datasets.dataset_name = [dataset_name, ]

    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/test/{date_str}/{dataset_name}'
    conf.training.logging.log_version = f"test/{date_str}_{dataset_name}"

    model = VelocityGANLightning.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
    base_test(model, conf, fast_run=fast_run)


if __name__ == '__main__':
    for dataset_name in ['CurveVelA', 'FlatVelA', 'FlatVelB', 'CurveVelB']:
        print(f"------------------{dataset_name}------------------")
        test_velocity_gan(dataset_name, fast_run=True)
