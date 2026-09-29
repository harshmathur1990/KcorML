"""Regularizers that keep learned responsibilities separated."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def normalization_bandwidth_loss(field: torch.Tensor) -> torch.Tensor:
    horizontal = field[..., :, 2:] - 2.0 * field[..., :, 1:-1] + field[..., :, :-2]
    vertical = field[..., 2:, :] - 2.0 * field[..., 1:-1, :] + field[..., :-2, :]
    return horizontal.abs().mean() + vertical.abs().mean()


def _centered_pooled(features: torch.Tensor) -> torch.Tensor:
    pooled = features.mean(dim=(-2, -1))
    return pooled - pooled.mean(dim=0, keepdim=True)


def noise_independence_loss(common: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    first = F.normalize(_centered_pooled(common), dim=1, eps=1.0e-6)
    second = F.normalize(_centered_pooled(noise), dim=1, eps=1.0e-6)
    return (first * second).sum(dim=1).square().mean()


def common_dynamic_consistency_loss(
    dynamic_1: torch.Tensor, dynamic_2: torch.Tensor
) -> torch.Tensor:
    """Keep the average scale of the two frame-specific pathways comparable."""
    first_scale = dynamic_1.square().mean(dim=(1, 2, 3)).sqrt()
    second_scale = dynamic_2.square().mean(dim=(1, 2, 3)).sqrt()
    return (first_scale - second_scale).abs().mean()

