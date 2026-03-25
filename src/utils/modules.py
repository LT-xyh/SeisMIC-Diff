import torch
import torch.nn as nn
from torch.nn import functional as F


class Interpolation(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        x = F.interpolate(
            input=x,
            size=self.shape,
            mode="bilinear",
            align_corners=False,
        )
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.shape)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1,
                              stride=stride) if in_channels != out_channels or stride != 1 else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.proj(x))
