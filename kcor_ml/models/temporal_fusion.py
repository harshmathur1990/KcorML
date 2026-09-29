"""Learned bidirectional deformable fusion for consecutive observations."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _base_grid(height: int, width: int, device, dtype) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


class DeformableCrossFusion2d(nn.Module):
    """Predict offsets and gates instead of explicitly registering input images."""

    def __init__(self, channels: int, maximum_offset_pixels: float):
        super().__init__()
        self.maximum_offset_pixels = maximum_offset_pixels
        self.offset = nn.Conv2d(channels * 2, 2, 3, padding=1)
        self.gate = nn.Conv2d(channels * 2, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.output = nn.Conv2d(channels * 2, channels, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def _sample(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = query.shape
        offsets = torch.tanh(self.offset(torch.cat((query, context), dim=1)))
        scale_x = 2.0 * self.maximum_offset_pixels / max(width - 1, 1)
        scale_y = 2.0 * self.maximum_offset_pixels / max(height - 1, 1)
        offsets = offsets.permute(0, 2, 3, 1)
        offsets = offsets * offsets.new_tensor((scale_x, scale_y))
        grid = _base_grid(height, width, query.device, query.dtype).unsqueeze(0)
        return F.grid_sample(context, grid + offsets, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        sampled = self._sample(query, context)
        gate = torch.sigmoid(self.gate(torch.cat((query, sampled), dim=1)))
        merged = torch.cat((query, gate * self.value(sampled)), dim=1)
        return query + self.output(merged)


class MultiscaleTemporalFusion(nn.Module):
    def __init__(self, channels: tuple[int, ...], maximum_offset_pixels: float):
        super().__init__()
        self.fusion = nn.ModuleList(
            DeformableCrossFusion2d(width, maximum_offset_pixels) for width in channels
        )

    def forward(
        self, first: list[torch.Tensor], second: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        fused_first = [layer(a, b) for layer, a, b in zip(self.fusion, first, second)]
        fused_second = [layer(b, a) for layer, a, b in zip(self.fusion, first, second)]
        return fused_first, fused_second
