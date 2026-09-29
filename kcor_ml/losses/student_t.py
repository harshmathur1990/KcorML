"""Numerically stable heteroscedastic Student-t likelihood."""

from __future__ import annotations

import math

import torch


def student_t_nll(
    target: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    degrees_of_freedom: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    scale = scale.clamp_min(torch.finfo(scale.dtype).eps)
    degrees_of_freedom = degrees_of_freedom.clamp_min(2.001)
    standardized = (target - location) / scale
    value = (
        torch.lgamma(0.5 * degrees_of_freedom)
        - torch.lgamma(0.5 * (degrees_of_freedom + 1.0))
        + 0.5 * torch.log(degrees_of_freedom)
        + math.log(math.pi) / 2.0
        + torch.log(scale)
        + 0.5
        * (degrees_of_freedom + 1.0)
        * torch.log1p(standardized.square() / degrees_of_freedom)
    )
    selected = value[mask.bool()]
    if selected.numel() == 0:
        return value.sum() * 0.0
    return selected.mean()

