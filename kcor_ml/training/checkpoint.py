"""Portable single-process checkpoint save and restore."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model so DDP checkpoints remain single-device loadable."""
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation_loss: float,
    config: dict[str, Any],
    diagnostics: dict[str, float] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "config": config,
            "diagnostics": diagnostics or {},
        },
        temporary,
    )
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    bare_model = unwrap_model(model)
    checkpoint_model = checkpoint.get("config", {}).get("model", {}).get("name")
    expected_model = getattr(bare_model, "architecture_name", None)
    if checkpoint_model and expected_model and checkpoint_model != expected_model:
        raise ValueError(
            f"checkpoint architecture {checkpoint_model!r} is incompatible with "
            f"configured architecture {expected_model!r}"
        )
    bare_model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint
