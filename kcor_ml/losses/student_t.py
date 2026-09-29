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
    # The public model outputs are in physical pB units (typically 1e-8 to
    # 1e-10).  Compute the likelihood in float32 even when the convolutional
    # trunk is under autocast: float16 cannot represent the lower end of this
    # range and large full-frame reductions can overflow.
    target = target.float()
    location = location.float()
    scale = scale.float()
    degrees_of_freedom = degrees_of_freedom.float()
    # eps is a relative precision (~1e-7 for float32), not a minimum positive
    # physical value. Using it here would silently clamp every pB-scale noise
    # estimate. The model already supplies the scientific minimum scale.
    scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)
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
        return value.new_zeros(())
    return selected.mean()
