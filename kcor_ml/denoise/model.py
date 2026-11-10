"""Residual U-Net mapping normalized pB frames to a noise-free estimate."""

from __future__ import annotations

import torch
from torch import nn

from ..models.blocks import Downsample, ResidualBlock, UpsampleBlock


class DenoisingUNet(nn.Module):
    """Predict one clean frame from ``input_frames`` normalized pB frames.

    With one input frame the network starts as the identity map; with several
    (temporal neighbours of the target time) it starts from their masked mean.
    Either way it learns a correction to that baseline.
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        blocks_per_level: int = 2,
        groups: int = 8,
        input_frames: int = 1,
    ):
        super().__init__()
        self.input_frames = input_frames
        # Inputs: normalized pB and validity per frame, plus the radial
        # coordinate, which tells the network where the signal-to-noise changes.
        self.stem = nn.Conv2d(2 * input_frames + 1, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList(
            nn.Sequential(*(ResidualBlock(width, groups) for _ in range(blocks_per_level)))
            for width in channels
        )
        self.downsample = nn.ModuleList(
            Downsample(channels[level], channels[level + 1]) for level in range(len(channels) - 1)
        )
        self.upsample = nn.ModuleList(
            UpsampleBlock(channels[level + 1], channels[level], channels[level], groups)
            for level in reversed(range(len(channels) - 1))
        )
        self.head = nn.Conv2d(channels[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image: torch.Tensor, valid: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """``image`` and ``valid`` are [B, F, H, W], ``radius`` is [B, 1, H, W];
        returns the denoised normalized image [B, 1, H, W]."""
        valid = valid.to(image.dtype)
        image = image * valid
        base = image.sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        features = self.stem(torch.cat((image, valid, radius.to(image.dtype)), dim=1))
        skips = []
        for level, encoder in enumerate(self.encoder):
            features = encoder(features)
            if level < len(self.downsample):
                skips.append(features)
                features = self.downsample[level](features)
        for upsample, skip in zip(self.upsample, reversed(skips)):
            features = upsample(features, skip)
        return base + self.head(features).float()
