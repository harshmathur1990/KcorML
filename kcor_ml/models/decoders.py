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
    def __init__(self, channels: int, levels: int, groups: int):
        super().__init__()
        self.decoder = CoarseDecoder(channels, levels, groups, 1)

    def forward(self, common: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        shared = F.softplus(self.decoder(common, output_size)) + 1.0e-6
        return shared.expand(-1, 2, -1, -1)


class FlatCoronaDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], groups: int):
        super().__init__()
        self.decoder = SkipDecoder(channels, groups, 1)

    def forward(self, common: torch.Tensor, dynamic: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        return F.softplus(self.decoder(common, dynamic, skips)) + 1.0e-6


class NoiseDecoder(nn.Module):
    def __init__(self, channels: int, levels: int, groups: int):
        super().__init__()
        self.decoder = CoarseDecoder(channels, levels, groups, 3)

    def forward(
        self, noise_latent: torch.Tensor, output_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        location, raw_scale, raw_df = self.decoder(noise_latent, output_size).chunk(3, dim=1)
        return location, F.softplus(raw_scale), F.softplus(raw_df) + 2.0


class CMEDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], groups: int):
        super().__init__()
        self.decoder = SkipDecoder(channels, groups, 1)

    def forward(self, common: torch.Tensor, dynamic: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.decoder(common, dynamic, skips))
