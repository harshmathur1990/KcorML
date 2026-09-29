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


class LearnedLinearScaler(nn.Module):
    """Positive learnable scale that keeps all public outputs in physical units."""

    def __init__(self, initial_scale: float):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(float(initial_scale)).log())

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs / self.scale.clamp_min(torch.finfo(inputs.dtype).tiny)

    def inverse(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.scale

