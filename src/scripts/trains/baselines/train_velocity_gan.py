from datetime import datetime

import torch
from omegaconf import OmegaConf

from baselines.velocity_gan.VelocityGANLightning import VelocityGANLightning
from scripts.trains.basetrain import base_train


def train_velocity_gan():
    # 获取当前日期时间
    current_date = datetime.now()
    # 格式化月份和日期为两位数，组成"MMDD"形式的字符串
    date_str = current_date.strftime("%y%m%d-%H")
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/velocity_gan.yaml')

    conf.training.logging.log_version = date_str + "_All_data_-VelocityGAN"
    model = VelocityGANLightning(conf)
    base_train(model, conf, fast_run=True, use_lr_finder=False, )


if __name__ == '__main__':
    train_velocity_gan()
