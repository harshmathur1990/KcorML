"""Construct the configured model and training components.

This module intentionally contains no FSDP or torch.distributed integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .losses import CompositeLoss
from .models import TwoFrameHeteroscedasticDecompositionNet


@dataclass(slots=True)
class TrainingComponents:
    model: TwoFrameHeteroscedasticDecompositionNet
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    loss: CompositeLoss
    device: torch.device


class ModelBuilder:
    def __init__(self, config: ExperimentConfig):
        config.validate()
        self.config = config

    def resolve_device(self) -> torch.device:
        requested = self.config.train.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; set train.device to 'cpu'")
        return torch.device(requested)

    def build_model(self, device: torch.device | None = None) -> TwoFrameHeteroscedasticDecompositionNet:
        device = device or self.resolve_device()
        return TwoFrameHeteroscedasticDecompositionNet(self.config.model).to(device)

    def build(self) -> TrainingComponents:
        device = self.resolve_device()
        model = self.build_model(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.train.epochs,
            eta_min=self.config.train.minimum_learning_rate,
        )
        loss = CompositeLoss(self.config.loss).to(device)
        return TrainingComponents(model, optimizer, scheduler, loss, device)
