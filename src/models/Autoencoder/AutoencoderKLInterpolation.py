import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from utils.modules import Interpolation


class AutoencoderKLInterpolation(nn.Module):
    """
    使用diffusers库中自带的AutoencoderKL：
    [1, 70, 70] -init_conv-> [16, 64, 64] -AutoencoderKl.encoder-> [16, 16, 16]
    -AutoencoderKl.decoder-> [16, 64, 64] -final_conv-> [1, 70, 70]
    """

    def __init__(self, latent_channels=16, depth_vel_shape=(1, 70, 70), depth_vel_reshape=(16, 64, 64),
                 down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
                 up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
                 block_out_channels=(64, 128, 256), ):
        """
        :param conf:
        :param depth_vel_shape:
        :param depth_vel_reshape:
        """
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=depth_vel_shape[0], out_channels=depth_vel_reshape[0], kernel_size=3, padding=1),
            Interpolation(depth_vel_reshape[1:]), )

        self.autoencoder_kl = AutoencoderKL(in_channels=depth_vel_reshape[0], out_channels=depth_vel_reshape[0],
                                            down_block_types=down_block_types, up_block_types=up_block_types,
                                            block_out_channels=block_out_channels, latent_channels=latent_channels,
                                            force_upcast=False  # 非全精
                                            )

        self.final_conv = nn.Sequential(Interpolation(depth_vel_shape[1:]),
            nn.Conv2d(depth_vel_reshape[0], depth_vel_shape[0], kernel_size=3, padding=1), )

    def forward(self, x):
        kl_weight = 1
        posterior = self.encode(x)
        latents = posterior.sample()
        reconstructions = self.decode(latents)

        recon_loss = F.mse_loss(reconstructions, x, reduction="none")
        kl_loss = posterior.kl().mean()
        loss = recon_loss + kl_loss * kl_weight
        return reconstructions, loss,

    def encode(self, x):
        x = self.init_conv(x)
        posterior = self.autoencoder_kl.encode(x).latent_dist
        return posterior

    def decode(self, latents):
        reconstructions = self.autoencoder_kl.decode(latents, return_dict=True).sample
        reconstructions = self.final_conv(reconstructions)
        return reconstructions


def test_autoencoder_kl_mlp():
    model = AutoencoderKLInterpolation()
    input_tensor = torch.randn(3, 1, 70, 70)  # Batch of 1, 1 channel, 70x70
    output, loss = model(input_tensor)

    assert output.shape == (3, 1, 70, 70), f"Expected output shape (3, 1, 70, 70), got {output.shape}"

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    print(f"Test passed!\noutput shape: {output.shape} loss: {loss.mean()}")


if __name__ == "__main__":
    test_autoencoder_kl_mlp()
