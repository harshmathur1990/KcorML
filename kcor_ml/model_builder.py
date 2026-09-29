"""Construct model and training components, optionally wrapped with DDP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .config import ExperimentConfig
from .distributed import DistributedContext
from .losses import CompositeLoss
from .models import TwoFrameHeteroscedasticDecompositionNet


@dataclass(slots=True)
class TrainingComponents:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    loss: CompositeLoss
    device: torch.device
    distributed: DistributedContext


class ModelBuilder:
    def __init__(self, config: ExperimentConfig, distributed: DistributedContext | None = None):
        config.validate()
        self.config = config
        self.distributed = distributed or DistributedContext.initialize(config.train.device)

    def resolve_device(self) -> torch.device:
        return self.distributed.device

    def build_model(self, device: torch.device | None = None) -> nn.Module:
        device = device or self.resolve_device()
        model = TwoFrameHeteroscedasticDecompositionNet(self.config.model).to(device)
        if self.distributed.enabled:
            model = DistributedDataParallel(
                model,
                device_ids=[self.distributed.local_rank],
                output_device=self.distributed.local_rank,
                broadcast_buffers=False,
            )
        return model

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
        return TrainingComponents(model, optimizer, scheduler, loss, device, self.distributed)
