"""Regularizers that keep learned responsibilities separated."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def normalization_bandwidth_loss(
    field: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    field = field.float()
    horizontal = field[..., :, 2:] - 2.0 * field[..., :, 1:-1] + field[..., :, :-2]
    vertical = field[..., 2:, :] - 2.0 * field[..., 1:-1, :] + field[..., :-2, :]
    if mask is None:
        return horizontal.abs().mean() + vertical.abs().mean()
    valid = mask.bool()
    horizontal_valid = valid[..., :, 2:] & valid[..., :, 1:-1] & valid[..., :, :-2]
    vertical_valid = valid[..., 2:, :] & valid[..., 1:-1, :] & valid[..., :-2, :]
    terms: list[torch.Tensor] = []
    if horizontal_valid.any():
        terms.append(horizontal[horizontal_valid].abs().mean())
    if vertical_valid.any():
        terms.append(vertical[vertical_valid].abs().mean())
    return torch.stack(terms).sum() if terms else field.new_zeros(())


def _centered_pooled(features: torch.Tensor) -> torch.Tensor:
    features = features.float()
    pooled = features.mean(dim=(-2, -1))
    return pooled - pooled.mean(dim=0, keepdim=True)


def noise_independence_loss(common: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    first = F.normalize(_centered_pooled(common), dim=1, eps=1.0e-6)
    second = F.normalize(_centered_pooled(noise), dim=1, eps=1.0e-6)
    return (first * second).sum(dim=1).square().mean()


def common_dynamic_consistency_loss(
    dynamic_1: torch.Tensor, dynamic_2: torch.Tensor
) -> torch.Tensor:
    """Keep the average scale of the two frame-specific pathways comparable."""
    dynamic_1 = dynamic_1.float()
    dynamic_2 = dynamic_2.float()
    first_scale = dynamic_1.square().mean(dim=(1, 2, 3)).sqrt()
    second_scale = dynamic_2.square().mean(dim=(1, 2, 3)).sqrt()
    return (first_scale - second_scale).abs().mean()


def gauge_loss(log_flat_corona: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Keep each flat-corona frame at unit geometric mean."""
    values = log_flat_corona.float()
    weights = mask.bool().to(values.dtype)
    means = (values * weights).sum(dim=(-2, -1)) / weights.sum(dim=(-2, -1)).clamp_min(1.0)
    active = weights.sum(dim=(-2, -1)) > 0
    return means[active].square().mean() if active.any() else values.new_zeros(())


def radial_flatness_loss(
    log_flat_corona: torch.Tensor,
    mask: torch.Tensor,
    radial_bins: int,
    radial_coordinate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Remove only annular mean trends from the learned clean corona.

    No radial statistic from the noisy input is divided into an image.  This is
    a training constraint on log(C), and does not normalize by annular variance
    as NRGF does.
    """
    values = log_flat_corona.float()
    selected = mask.bool()
    if radial_coordinate is None:
        _, _, height, width = values.shape
        y = torch.linspace(-1.0, 1.0, height, device=values.device, dtype=values.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=values.device, dtype=values.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        radius = (grid_x.square() + grid_y.square()).sqrt()
        radius = radius / radius.max().clamp_min(1.0)
        radial_coordinate = radius.expand_as(values)
    else:
        radial_coordinate = radial_coordinate.to(device=values.device, dtype=values.dtype)
        if radial_coordinate.shape != values.shape:
            raise ValueError("radial_coordinate must match log_flat_corona shape")

    sums = values.new_zeros(radial_bins)
    counts = values.new_zeros(radial_bins)
    for batch in range(values.shape[0]):
        for frame in range(values.shape[1]):
            frame_mask = selected[batch, frame].reshape(-1)
            if not frame_mask.any():
                continue
            frame_values = values[batch, frame].reshape(-1)
            frame_radius = radial_coordinate[batch, frame].reshape(-1)
            bins = (frame_radius[frame_mask] * radial_bins).long().clamp(0, radial_bins - 1)
            sums = sums.scatter_add(0, bins, frame_values[frame_mask])
            counts = counts.scatter_add(
                0, bins, torch.ones_like(frame_values[frame_mask])
            )
    active = counts > 0
    if not active.any():
        return values.new_zeros(())
    annular_means = sums[active] / counts[active]
    return annular_means.square().mean()


def noise_structure_correlation_loss(
    log_flat_corona: torch.Tensor,
    log_noise_scale: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize shared fine morphology while permitting broad radial noise."""
    flat = log_flat_corona.float()
    noise = log_noise_scale.float()
    selected = mask.bool()
    losses: list[torch.Tensor] = []
    for batch in range(flat.shape[0]):
        for frame in range(flat.shape[1]):
            valid = selected[batch, frame]
            interior = (
                valid[1:-1, 1:-1]
                & valid[:-2, 1:-1]
                & valid[2:, 1:-1]
                & valid[1:-1, :-2]
                & valid[1:-1, 2:]
            )
            if not interior.any():
                continue

            def laplacian(value: torch.Tensor) -> torch.Tensor:
                return (
                    -4.0 * value[1:-1, 1:-1]
                    + value[:-2, 1:-1]
                    + value[2:, 1:-1]
                    + value[1:-1, :-2]
                    + value[1:-1, 2:]
                )[interior]

            left = laplacian(flat[batch, frame])
            right = laplacian(noise[batch, frame])
            left = left - left.mean()
            right = right - right.mean()
            denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
            correlation = (left * right).sum() / denominator.clamp_min(1.0e-12)
            losses.append(correlation.square())
    return torch.stack(losses).mean() if losses else flat.new_zeros(())
