"""Composite restoration loss with named components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..config import LossConfig
from ..models.outputs import TFHDNOutput
from .separation import (
    common_dynamic_consistency_loss,
    noise_independence_loss,
    normalization_bandwidth_loss,
)
from .student_t import student_t_nll


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    observation: torch.Tensor
    normalization_bandwidth: torch.Tensor
    noise_independence: torch.Tensor
    common_consistency: torch.Tensor
    cme: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "observation": float(self.observation.detach()),
            "normalization_bandwidth": float(self.normalization_bandwidth.detach()),
            "noise_independence": float(self.noise_independence.detach()),
            "common_consistency": float(self.common_consistency.detach()),
            "cme": float(self.cme.detach()),
        }


class CompositeLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        output: TFHDNOutput,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        cme_target: torch.Tensor | None = None,
    ) -> LossBreakdown:
        observation = student_t_nll(
            target,
            output.observation_location,
            output.noise_scale,
            output.noise_df,
            loss_mask,
        )
        bandwidth = normalization_bandwidth_loss(output.auxiliary["normalization_scaled"])
        independence = 0.5 * (
            noise_independence_loss(output.auxiliary["common"], output.auxiliary["noise_1"])
            + noise_independence_loss(output.auxiliary["common"], output.auxiliary["noise_2"])
        )
        consistency = common_dynamic_consistency_loss(
            output.auxiliary["dynamic_1"], output.auxiliary["dynamic_2"]
        )
        if cme_target is None or self.config.cme_weight == 0:
            cme = output.cme_probability.sum() * 0.0
        else:
            cme = F.binary_cross_entropy(output.cme_probability, cme_target.float())
        total = (
            self.config.observation_weight * observation
            + self.config.normalization_bandwidth_weight * bandwidth
            + self.config.noise_independence_weight * independence
            + self.config.common_consistency_weight * consistency
            + self.config.cme_weight * cme
        )
        return LossBreakdown(total, observation, bandwidth, independence, consistency, cme)
