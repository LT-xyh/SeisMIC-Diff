from datetime import datetime

import torch
from lightning_modules.diffusion.DDPMConditionalDiffusionLightning import DDPMConditionalDiffusionLightning
from omegaconf import OmegaConf

from scripts.trains.basetrain import base_train


def train_ddpm_cond_diffusion():
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/ddpm_cond_diffusion.yaml')
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d-%H")
    conf.training.logging.log_version = date_str + "base-cond_rms-smooth_All-data"

    model = DDPMConditionalDiffusionLightning(conf)
    base_train(model, conf, fast_run=True, use_lr_finder=False, )


if __name__ == '__main__':
    train_ddpm_cond_diffusion()
