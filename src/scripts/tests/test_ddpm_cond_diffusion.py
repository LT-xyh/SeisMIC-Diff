from datetime import datetime

import torch
from omegaconf import OmegaConf

from lightning_modules.diffusion.DDPMConditionalDiffusionLightning import DDPMConditionalDiffusionLightning
from scripts.tests.basetest import base_test


def test_ddpm_cond_diffusion(dataset_name):
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d")
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/ddpm_cond_diffusion.yaml')
    conf.datasets.dataset_name = [dataset_name, ]

    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/test_{date_str}/{dataset_name}'
    conf.testing.ckpt_path = 'ckpt_path/ddpm_diffusion_valssim0.918.ckpt'
    conf.training.logging.log_version = f"test/{date_str}_{dataset_name}"
    model = DDPMConditionalDiffusionLightning.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
    base_test(model, conf, fast_run=True)


if __name__ == '__main__':

    for dataset_name in ['FlatVelA', 'FlatVelB', 'CurveVelA', 'CurveVelB']:
        print(f'\n\n{dataset_name}\n')
        test_ddpm_cond_diffusion(dataset_name)
