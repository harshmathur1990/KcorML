"""Typed output contract shared by training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TFHDNOutput:
    clean_pb: torch.Tensor
    flat_corona: torch.Tensor
    normalization_field: torch.Tensor
    noise_location: torch.Tensor
    noise_scale: torch.Tensor
    noise_df: torch.Tensor
    cme_probability: torch.Tensor
    auxiliary: dict[str, torch.Tensor]

    @property
    def observation_location(self) -> torch.Tensor:
        return self.clean_pb + self.noise_location

