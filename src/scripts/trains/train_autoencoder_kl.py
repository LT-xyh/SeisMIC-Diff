from datetime import datetime

import torch
from omegaconf import OmegaConf

from lightning_modules.Autoencoder.autoencoder_kl_lightning import AutoencoderKLLightning
from scripts.trains.basetrain import base_train


def train_autoencoder_kl():
    torch.set_float32_matmul_precision("medium")  # Allow Tensor Core friendly matmul precision.
    conf = OmegaConf.load("configs/autoencoder_kl.yaml")

    current_date = datetime.now()
    date_str = current_date.strftime("%m%d")
    conf.training.logging.log_version = "all_data" + date_str

    model = AutoencoderKLLightning(conf)
    base_train(model, conf, fast_run=True, use_lr_finder=False)


if __name__ == "__main__":
    train_autoencoder_kl()
