from datetime import datetime

import torch
from omegaconf import OmegaConf

from baselines.inversion_net.InversionNetLightning import InversionNetLightning
from scripts.trains.basetrain import base_train


def train_inversion_net():
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/inversion_net.yaml')
    # 获取当前日期时间
    current_date = datetime.now()
    # 格式化月份和日期为两位数，组成"MMDD"形式的字符串
    date_str = current_date.strftime("%y%m%d-%H")

    conf.training.logging.log_version = date_str + "_All_data_lr-1e-3"
    model = InversionNetLightning(conf)
    base_train(model, conf, fast_run=False, use_lr_finder=False, )


if __name__ == '__main__':
    train_inversion_net()
