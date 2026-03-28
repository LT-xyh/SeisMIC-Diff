from datetime import datetime

import torch
from omegaconf import OmegaConf

from lightning_modules.baselines.SVInvNetLightning import SVInvNetLightning
from scripts.trains.basetrain import base_train


def train_sv_inv_net():
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/sv_inv_net.yaml')

    # 获取当前日期时间
    current_date = datetime.now()
    # 格式化月份和日期为两位数，组成"MMDD"形式的字符串
    date_str = current_date.strftime("%y%m%d-%H")

    conf.training.logging.log_version = date_str + "_All-Data"
    model = SVInvNetLightning(conf)
    base_train(model, conf, fast_run=True, use_lr_finder=False  , )


if __name__ == '__main__':
    train_sv_inv_net()
