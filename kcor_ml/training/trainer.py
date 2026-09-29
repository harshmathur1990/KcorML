"""Single-device trainer for masked two-frame self-supervision."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

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

    def run_epoch(
        self,
        loader: Iterator[dict[str, object]],
        *,
        training: bool,
        epoch: int | None = None,
    ) -> dict[str, float]:
        model = self.components.model
        model.train(training)
        metrics = MeanMetrics()
        phase = "train" if training else "validation"
        total_steps = len(loader) if hasattr(loader, "__len__") else None
        epoch_label = f" epoch {epoch}/{self.config.train.epochs}" if epoch is not None else ""
        progress = tqdm(
            loader,
            total=total_steps,
            desc=f"{phase}{epoch_label}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            disable=not self.config.train.progress_bar,
        )
        context = torch.enable_grad if training else torch.no_grad
        with context():
            for step, batch in enumerate(progress, start=1):
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
                if step == 1 or step % self.config.train.log_every_steps == 0:
                    current = metrics.compute()
                    progress.set_postfix(
                        loss=f"{current['total']:.5g}",
                        observation=f"{current['observation']:.5g}",
                        lr=f"{self.components.optimizer.param_groups[0]['lr']:.3g}",
                        refresh=True,
                    )
        progress.close()
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
            epoch_number = epoch + 1
            train_metrics = self.run_epoch(train_loader, training=True, epoch=epoch_number)
            validation_metrics = self.run_epoch(
                validation_loader, training=False, epoch=epoch_number
            )
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
                f"epoch={epoch_number} train={train_metrics['total']:.6g} "
                f"validation={validation_loss:.6g}",
                flush=True,
            )
        return last_validation
