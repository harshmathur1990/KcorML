"""Numerical and decomposition-collapse diagnostics."""

from __future__ import annotations

import torch

from ..models.outputs import TFHDNOutput


def decomposition_diagnostics(
    output: TFHDNOutput,
    mask: torch.Tensor,
    log_component_limit: float,
    radial_coordinate: torch.Tensor | None = None,
) -> dict[str, float]:
    selected = mask.bool()
    log_normalization = output.auxiliary["log_normalization"].float()
    log_flat = output.auxiliary["log_flat_corona"].float()

    standard_deviations: list[torch.Tensor] = []
    saturation_count = log_flat.new_zeros(())
    selected_count = log_flat.new_zeros(())
    for batch in range(selected.shape[0]):
        for frame in range(selected.shape[1]):
            frame_mask = selected[batch, frame]
            if not frame_mask.any():
                continue
            normalization_values = log_normalization[batch, frame][frame_mask]
            flat_values = log_flat[batch, frame][frame_mask]
            standard_deviations.append(normalization_values.std(unbiased=False))
            saturation_count = saturation_count + (
                flat_values.abs() >= log_component_limit - 1.0e-4
            ).sum()
            saturation_count = saturation_count + (
                normalization_values.abs() >= log_component_limit - 1.0e-4
            ).sum()
            selected_count = selected_count + 2 * frame_mask.sum()

    if standard_deviations:
        normalization_log_std = torch.stack(standard_deviations).mean()
    else:
        normalization_log_std = log_flat.new_zeros(())
    saturation_fraction = saturation_count / selected_count.clamp_min(1)

    radial_log_std = log_flat.new_zeros(())
    if radial_coordinate is not None:
        bins_count = 32
        radius = radial_coordinate.to(device=log_flat.device, dtype=log_flat.dtype)
        sums = log_flat.new_zeros(bins_count)
        counts = log_flat.new_zeros(bins_count)
        for batch in range(selected.shape[0]):
            for frame in range(selected.shape[1]):
                frame_mask = selected[batch, frame].reshape(-1)
                if not frame_mask.any():
                    continue
                bins = (radius[batch, frame].reshape(-1)[frame_mask] * bins_count).long()
                bins = bins.clamp(0, bins_count - 1)
                samples = log_normalization[batch, frame].reshape(-1)[frame_mask]
                sums = sums.scatter_add(0, bins, samples)
                counts = counts.scatter_add(0, bins, torch.ones_like(samples))
        active_bins = counts > 0
        if active_bins.any():
            radial_log_std = (sums[active_bins] / counts[active_bins]).std(unbiased=False)

    finite_tensors = (
        output.clean_pb,
        output.flat_corona,
        output.normalization_field,
        output.noise_scale,
        output.noise_df,
    )
    finite = all(bool(torch.isfinite(value).all()) for value in finite_tensors)
    return {
        "normalization_log_std": float(normalization_log_std.detach()),
        "normalization_radial_log_std": float(radial_log_std.detach()),
        "log_saturation_fraction": float(saturation_fraction.detach()),
        "outputs_finite": float(finite),
    }
