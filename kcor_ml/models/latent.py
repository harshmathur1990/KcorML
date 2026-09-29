"""Separate common, dynamic, and frame-specific noise representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class LatentComponents:
    common: torch.Tensor
    dynamic_1: torch.Tensor
    dynamic_2: torch.Tensor
    noise_1: torch.Tensor
    noise_2: torch.Tensor


class LatentSeparator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        pair_channels = channels * 2
        self.common = nn.Conv2d(pair_channels, channels, 1)
        self.dynamic = nn.Conv2d(pair_channels, channels, 1)
        self.noise = nn.Conv2d(channels, channels, 1)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> LatentComponents:
        ordered = torch.cat((first, second), dim=1)
        reversed_order = torch.cat((second, first), dim=1)
        common = 0.5 * (self.common(ordered) + self.common(reversed_order))
        return LatentComponents(
            common=common,
            dynamic_1=self.dynamic(ordered),
            dynamic_2=self.dynamic(reversed_order),
            noise_1=self.noise(first),
            noise_2=self.noise(second),
        )
