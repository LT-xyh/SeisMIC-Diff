from datetime import datetime

import torch
from omegaconf import OmegaConf

from lightning_modules.baselines.SVInvNetLightning import SVInvNetLightning
from scripts.tests.basetest import base_test


def test_sv_inv_net(dataset_name):
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d")
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/sv_inv_net.yaml')
    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/{dataset_name}'
    conf.datasets.dataset_name = [dataset_name, ]

    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/test_{date_str}/{dataset_name}'
    conf.testing.ckpt_path = 'ckpt_path/sv_inv_net_ssim0.732.ckpt'
    conf.training.logging.log_version = f"test/{date_str}_{dataset_name}"
    model = SVInvNetLightning.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
    base_test(model, conf, fast_run=True)


if __name__ == '__main__':
    for dataset_name in ['CurveVelA', 'FlatVelA', 'FlatVelB', 'CurveVelB']:
        test_sv_inv_net(dataset_name)
