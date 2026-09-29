"""Single- or multi-device trainer for cross-frame Noise2Noise learning."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from ..config import ExperimentConfig
from ..model_builder import TrainingComponents
from .checkpoint import load_checkpoint, save_checkpoint
from .diagnostics import decomposition_diagnostics
from .metrics import MeanMetrics


class Trainer:
    def __init__(self, config: ExperimentConfig, components: TrainingComponents):
        self.config = config
        self.components = components
        self.output_dir = Path(config.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = config.train.mixed_precision and components.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _move_batch(
        self, batch: dict[str, object]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.components.device
        images = batch["images"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True).bool()
        delta_t = batch["delta_t"].to(device, non_blocking=True)
        radius = batch["radial_coordinate"].to(device, non_blocking=True)
        return images, valid, delta_t, radius

    def _blind_spot_mask(
        self,
        valid: torch.Tensor,
        *,
        training: bool,
        step: int,
        direction: int,
    ) -> torch.Tensor:
        batch, _, height, width = valid.shape
        patch = self.config.train.mask_patch_size
        coarse_h = (height + patch - 1) // patch
        coarse_w = (width + patch - 1) // patch
        if training:
            coarse = torch.rand(
                (batch, 1, coarse_h, coarse_w), device=valid.device
            ) < self.config.train.mask_fraction
        else:
            period = max(2, round(1.0 / self.config.train.mask_fraction))
            y = torch.arange(coarse_h, device=valid.device).reshape(1, 1, -1, 1)
            x = torch.arange(coarse_w, device=valid.device).reshape(1, 1, 1, -1)
            coarse = ((3 * y + 5 * x + step + direction) % period == 0).expand(
                batch, -1, -1, -1
            )
        return coarse.repeat_interleave(patch, -2).repeat_interleave(patch, -1)[
            ..., :height, :width
        ]

    def _cross_frame_view(
        self,
        target: torch.Tensor,
        valid: torch.Tensor,
        direction: int,
        *,
        training: bool,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hide the target and blind source pixels; never form a difference."""
        inputs = target.clone()
        loss_mask = torch.zeros_like(valid)
        component_mask = torch.zeros_like(valid)
        blind = self._blind_spot_mask(
            valid, training=training, step=step, direction=direction
        )[:, 0]
        if direction == 0:
            inputs[:, 1] = 0.0
            selected = blind & valid[:, 0] & valid[:, 1]
            inputs[:, 0] = inputs[:, 0].masked_fill(selected, 0.0)
            loss_mask[:, 1] = selected
            component_mask[:, 1] = valid[:, 1]
        elif direction == 1:
            inputs[:, 0] = 0.0
            selected = blind & valid[:, 0] & valid[:, 1]
            inputs[:, 1] = inputs[:, 1].masked_fill(selected, 0.0)
            loss_mask[:, 0] = selected
            component_mask[:, 0] = valid[:, 0]
        else:
            raise ValueError(f"invalid cross-frame direction: {direction}")
        return inputs, loss_mask, component_mask

    def run_epoch(
        self,
        loader: Iterator[dict[str, object]],
        *,
        training: bool,
        epoch: int | None = None,
        phase_name: str | None = None,
    ) -> dict[str, float]:
        model = self.components.model
        model.train(training)
        metrics = MeanMetrics()
        phase = phase_name or ("train" if training else "validation")
        total_steps = len(loader) if hasattr(loader, "__len__") else None
        epoch_label = f" epoch {epoch}/{self.config.train.epochs}" if epoch is not None else ""
        progress = tqdm(
            loader,
            total=total_steps,
            desc=f"{phase}{epoch_label}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            disable=(not self.config.train.progress_bar) or (not self.components.distributed.is_main),
        )
        context = torch.enable_grad if training else torch.no_grad
        with context():
            for step, batch in enumerate(progress, start=1):
                target, valid, delta_t, radius = self._move_batch(batch)
                # Alternate directions during training to retain the original
                # memory footprint; evaluate both directions deterministically.
                if training:
                    directions = ((step + (epoch or 0)) % 2,)
                else:
                    directions = (0, 1)
                for direction in directions:
                    cross_input, loss_mask, component_mask = self._cross_frame_view(
                        target,
                        valid,
                        direction,
                        training=training,
                        step=step,
                    )
                    if training:
                        self.components.optimizer.zero_grad(set_to_none=True)
                    with self._autocast():
                        output = model(
                            cross_input,
                            delta_t=delta_t,
                            valid_mask=valid,
                            radial_coordinate=radius,
                        )
                        breakdown = self.components.loss(
                            output,
                            target,
                            loss_mask,
                            radial_coordinate=radius,
                            component_mask=component_mask,
                        )
                    if not torch.isfinite(breakdown.total):
                        paths = batch.get("paths", "unknown")
                        raise FloatingPointError(
                            f"non-finite {phase} loss at step={step} "
                            f"direction={direction} paths={paths}"
                        )
                    if training:
                        self.scaler.scale(breakdown.total).backward()
                        self.scaler.unscale_(self.components.optimizer)
                        clip_grad_norm_(model.parameters(), self.config.train.gradient_clip)
                        self.scaler.step(self.components.optimizer)
                        self.scaler.update()
                    values = breakdown.detached()
                    values.update(
                        decomposition_diagnostics(
                            output,
                            component_mask,
                            self.config.model.log_component_limit,
                            radius,
                        )
                    )
                    metrics.update(values)
                if step == 1 or step % self.config.train.log_every_steps == 0:
                    current = metrics.compute()
                    progress.set_postfix(
                        loss=f"{current['total']:.5g}",
                        observation=f"{current['observation']:.5g}",
                        lr=f"{self.components.optimizer.param_groups[0]['lr']:.3g}",
                        refresh=True,
                    )
        progress.close()
        metrics.synchronize(self.components.device)
        return metrics.compute()

    def fit(self, train_loader, validation_loader, full_diagnostic_loader) -> dict[str, float]:
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
            train_sampler = getattr(train_loader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            train_metrics = self.run_epoch(train_loader, training=True, epoch=epoch_number)
            validation_metrics = self.run_epoch(
                validation_loader, training=False, epoch=epoch_number
            )
            full_diagnostic_metrics = self.run_epoch(
                full_diagnostic_loader,
                training=False,
                epoch=epoch_number,
                phase_name="full diagnostic",
            )
            self.components.scheduler.step()
            last_validation = validation_metrics
            validation_loss = validation_metrics["total"]
            if self.components.distributed.is_main:
                checkpoint_eligible = (
                    torch.isfinite(torch.tensor(validation_loss)).item()
                    and full_diagnostic_metrics["outputs_finite"] == 1.0
                    and full_diagnostic_metrics["normalization_radial_log_std"]
                    >= self.config.train.minimum_normalization_log_std
                    and full_diagnostic_metrics["log_saturation_fraction"]
                    <= self.config.train.maximum_log_saturation_fraction
                    and full_diagnostic_metrics["flat_radial_rms"]
                    <= self.config.train.maximum_flat_radial_rms
                    and full_diagnostic_metrics["noise_structure_correlation"]
                    <= self.config.train.maximum_noise_structure_correlation
                )
                improved = checkpoint_eligible and validation_loss < best
                if improved:
                    best = validation_loss
                checkpoint_arguments = dict(
                    model=self.components.model,
                    optimizer=self.components.optimizer,
                    scheduler=self.components.scheduler,
                    epoch=epoch,
                    best_validation_loss=best,
                    config=self.config.to_dict(),
                    diagnostics={
                        **validation_metrics,
                        **{
                            f"full_{name}": value
                            for name, value in full_diagnostic_metrics.items()
                        },
                    },
                )
                save_checkpoint(self.output_dir / "last.pt", **checkpoint_arguments)
                if improved:
                    save_checkpoint(self.output_dir / "best.pt", **checkpoint_arguments)
                print(
                    f"epoch={epoch_number} train={train_metrics['total']:.6g} "
                    f"validation={validation_loss:.6g} "
                    f"full_norm_radial_std={full_diagnostic_metrics['normalization_radial_log_std']:.4g} "
                    f"full_flat_radial_rms={full_diagnostic_metrics['flat_radial_rms']:.4g} "
                    f"full_noise_corr={full_diagnostic_metrics['noise_structure_correlation']:.3g} "
                    f"full_saturation={full_diagnostic_metrics['log_saturation_fraction']:.3g} "
                    f"checkpoint_eligible={checkpoint_eligible}",
                    flush=True,
                )
            self.components.distributed.barrier()
        return last_validation
