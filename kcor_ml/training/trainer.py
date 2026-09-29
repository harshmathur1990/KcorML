"""Single-device trainer for masked two-frame self-supervision."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
from torch.nn.utils import clip_grad_norm_

from ..config import ExperimentConfig
from ..data.masking import PairMasker
from ..model_builder import TrainingComponents
from .checkpoint import load_checkpoint, save_checkpoint
from .metrics import MeanMetrics


class Trainer:
    def __init__(self, config: ExperimentConfig, components: TrainingComponents):
        self.config = config
        self.components = components
        self.masker = PairMasker(
            config.train.mask_fraction,
            config.train.mask_patch_size,
            config.train.frame_drop_probability,
        )
        self.output_dir = Path(config.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = config.train.mixed_precision and components.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _move_batch(self, batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.components.device
        images = batch["images"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True).bool()
        delta_t = batch["delta_t"].to(device, non_blocking=True)
        return images, valid, delta_t

    def run_epoch(self, loader: Iterator[dict[str, object]], *, training: bool) -> dict[str, float]:
        model = self.components.model
        model.train(training)
        metrics = MeanMetrics()
        context = torch.enable_grad if training else torch.no_grad
        with context():
            for batch in loader:
                target, valid, delta_t = self._move_batch(batch)
                corrupted, hidden = self.masker(target, valid)
                if training:
                    self.components.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    output = model(corrupted, delta_t=delta_t, valid_mask=valid)
                    breakdown = self.components.loss(output, target, hidden & valid)
                if training:
                    self.scaler.scale(breakdown.total).backward()
                    self.scaler.unscale_(self.components.optimizer)
                    clip_grad_norm_(model.parameters(), self.config.train.gradient_clip)
                    self.scaler.step(self.components.optimizer)
                    self.scaler.update()
                metrics.update(breakdown.detached())
        return metrics.compute()

    def fit(self, train_loader, validation_loader) -> dict[str, float]:
        start_epoch = 0
        best = float("inf")
        if self.config.train.resume_checkpoint:
            state = load_checkpoint(
                self.config.train.resume_checkpoint,
                model=self.components.model,
                optimizer=self.components.optimizer,
                scheduler=self.components.scheduler,
                map_location=self.components.device,
            )
            start_epoch = int(state.get("epoch", -1)) + 1
            best = float(state.get("best_validation_loss", best))

        last_validation: dict[str, float] = {}
        for epoch in range(start_epoch, self.config.train.epochs):
            train_metrics = self.run_epoch(train_loader, training=True)
            validation_metrics = self.run_epoch(validation_loader, training=False)
            self.components.scheduler.step()
            last_validation = validation_metrics
            validation_loss = validation_metrics["total"]
            checkpoint_arguments = dict(
                model=self.components.model,
                optimizer=self.components.optimizer,
                scheduler=self.components.scheduler,
                epoch=epoch,
                best_validation_loss=min(best, validation_loss),
                config=self.config.to_dict(),
            )
            save_checkpoint(self.output_dir / "last.pt", **checkpoint_arguments)
            if validation_loss < best:
                best = validation_loss
                checkpoint_arguments["best_validation_loss"] = best
                save_checkpoint(self.output_dir / "best.pt", **checkpoint_arguments)
            print(
                f"epoch={epoch + 1} train={train_metrics['total']:.6g} "
                f"validation={validation_loss:.6g}"
            )
        return last_validation

