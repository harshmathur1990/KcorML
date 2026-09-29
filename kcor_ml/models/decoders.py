"""Independent output decoders for each scientific responsibility."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ResidualBlock, UpsampleBlock


class SkipDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], groups: int, output_channels: int):
        super().__init__()
        reversed_channels = tuple(reversed(channels))
        self.input = nn.Conv2d(reversed_channels[0] * 2, reversed_channels[0], 1)
        self.upsamples = nn.ModuleList(
            UpsampleBlock(reversed_channels[index], reversed_channels[index + 1], reversed_channels[index + 1], groups)
            for index in range(len(reversed_channels) - 1)
        )
        self.output = nn.Conv2d(channels[0], output_channels, 1)

    def forward(self, common: torch.Tensor, specific: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        value = self.input(torch.cat((common, specific), dim=1))
        for layer, skip in zip(self.upsamples, reversed(skips[:-1])):
            value = layer(value, skip)
        return self.output(value)


class CoarseDecoder(nn.Module):
    """Upsample without high-resolution skips to enforce broad output structure."""

    def __init__(self, channels: int, levels: int, groups: int, output_channels: int):
        super().__init__()
        blocks: list[nn.Module] = []
        width = channels
        for _ in range(levels - 1):
            next_width = max(width // 2, 16)
            blocks.extend(
                (
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(width, next_width, 3, padding=1),
                    ResidualBlock(next_width, groups),
                )
            )
            width = next_width
        self.layers = nn.Sequential(*blocks)
        self.output = nn.Conv2d(width, output_channels, 1)

    def forward(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        value = self.output(self.layers(latent))
        if value.shape[-2:] != output_size:
            value = F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
        return value


class NormalizationDecoder(nn.Module):
    def __init__(self, channels: int, levels: int, groups: int, grid_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.decoder = CoarseDecoder(channels, levels, groups, 1)

    def forward(self, common: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        # This is log(B), not B. Positivity is imposed after the gauge is fixed.
        coarse = F.adaptive_avg_pool2d(common, (self.grid_size, self.grid_size))
        shared = self.decoder(coarse, output_size)
        return shared.expand(-1, 2, -1, -1)


class FlatCoronaDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], groups: int):
        super().__init__()
        self.decoder = SkipDecoder(channels, groups, 1)

    def forward(self, common: torch.Tensor, dynamic: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        # This is log(C). The model centers it before exponentiation.
        return self.decoder(common, dynamic, skips)


class NoiseDecoder(nn.Module):
    def __init__(self, channels: int, levels: int, groups: int, grid_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.decoder = CoarseDecoder(channels, levels, groups, 2)

    def forward(
        self, noise_latent: torch.Tensor, output_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse = F.adaptive_avg_pool2d(noise_latent, (self.grid_size, self.grid_size))
        raw_scale, raw_df = self.decoder(coarse, output_size).chunk(2, dim=1)
        return F.softplus(raw_scale), F.softplus(raw_df) + 2.0


class CMEDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], groups: int):
        super().__init__()
        self.decoder = SkipDecoder(channels, groups, 1)

    def forward(self, common: torch.Tensor, dynamic: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.decoder(common, dynamic, skips))
