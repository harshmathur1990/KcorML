"""Shared multiscale image encoder with learned spatial position features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import Downsample, ResidualBlock


class SharedImageEncoder(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        blocks_per_level: int,
        groups: int,
        positional_grid_size: int,
    ):
        super().__init__()
        # Image intensity plus absolute normalized solar radius.
        self.stem = nn.Conv2d(2, channels[0], 3, padding=1)
        self.position = nn.Parameter(
            torch.zeros(1, channels[0], positional_grid_size, positional_grid_size)
        )
        self.levels = nn.ModuleList(
            nn.Sequential(*(ResidualBlock(width, groups) for _ in range(blocks_per_level)))
            for width in channels
        )
        self.downsamples = nn.ModuleList(
            Downsample(channels[index], channels[index + 1])
            for index in range(len(channels) - 1)
        )

    def forward(self, image: torch.Tensor, radius: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        value = self.stem(torch.cat((image, radius), dim=1))
        position = F.interpolate(self.position, size=value.shape[-2:], mode="bilinear", align_corners=False)
        value = value + position
        for index, level in enumerate(self.levels):
            value = level(value)
            features.append(value)
            if index < len(self.downsamples):
                value = self.downsamples[index](value)
        return features
