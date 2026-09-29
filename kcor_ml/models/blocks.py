"""Reusable two-dimensional neural-network blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def group_count(channels: int, requested: int) -> int:
    for groups in range(min(channels, requested), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        normalized_groups = group_count(channels, groups)
        self.layers = nn.Sequential(
            nn.GroupNorm(normalized_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(normalized_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.projection = nn.Conv2d(in_channels + skip_channels, out_channels, 1)
        self.residual = ResidualBlock(out_channels, groups)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.residual(self.projection(torch.cat((inputs, skip), dim=1)))


class FixedLinearScaler(nn.Module):
    """Fixed physical scale; keeping it fixed removes a decomposition gauge."""

    def __init__(self, scale: float):
        super().__init__()
        self.register_buffer("fixed_scale", torch.tensor(float(scale)))

    @property
    def scale(self) -> torch.Tensor:
        return self.fixed_scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs / self.scale.clamp_min(torch.finfo(inputs.dtype).tiny)

    def inverse(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.scale
