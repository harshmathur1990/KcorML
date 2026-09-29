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
    flat_radial_rms = log_flat.new_zeros(())
    if radial_coordinate is not None:
        bins_count = 32
        radius = radial_coordinate.to(device=log_flat.device, dtype=log_flat.dtype)
        sums = log_flat.new_zeros(bins_count)
        flat_sums = log_flat.new_zeros(bins_count)
        counts = log_flat.new_zeros(bins_count)
        for batch in range(selected.shape[0]):
            for frame in range(selected.shape[1]):
                frame_mask = selected[batch, frame].reshape(-1)
                if not frame_mask.any():
                    continue
                bins = (radius[batch, frame].reshape(-1)[frame_mask] * bins_count).long()
                bins = bins.clamp(0, bins_count - 1)
                samples = log_normalization[batch, frame].reshape(-1)[frame_mask]
                flat_samples = log_flat[batch, frame].reshape(-1)[frame_mask]
                sums = sums.scatter_add(0, bins, samples)
                flat_sums = flat_sums.scatter_add(0, bins, flat_samples)
                counts = counts.scatter_add(0, bins, torch.ones_like(samples))
        active_bins = counts > 0
        if active_bins.any():
            radial_log_std = (sums[active_bins] / counts[active_bins]).std(unbiased=False)
            flat_annular_means = flat_sums[active_bins] / counts[active_bins]
            flat_radial_rms = flat_annular_means.square().mean().sqrt()

    structure_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    log_noise = output.noise_scale.float().clamp_min(1.0e-30).log()
    for batch in range(selected.shape[0]):
        for frame in range(selected.shape[1]):
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
            flat_frame = log_flat[batch, frame]
            noise_frame = log_noise[batch, frame]
            flat_laplacian = (
                -4.0 * flat_frame[1:-1, 1:-1]
                + flat_frame[:-2, 1:-1]
                + flat_frame[2:, 1:-1]
                + flat_frame[1:-1, :-2]
                + flat_frame[1:-1, 2:]
            )[interior]
            noise_laplacian = (
                -4.0 * noise_frame[1:-1, 1:-1]
                + noise_frame[:-2, 1:-1]
                + noise_frame[2:, 1:-1]
                + noise_frame[1:-1, :-2]
                + noise_frame[1:-1, 2:]
            )[interior]
            structure_pairs.append((flat_laplacian, noise_laplacian))
    noise_structure_correlation = log_flat.new_zeros(())
    if structure_pairs:
        flat_structure = torch.cat([pair[0] for pair in structure_pairs])
        noise_structure = torch.cat([pair[1] for pair in structure_pairs])
        flat_structure = flat_structure - flat_structure.mean()
        noise_structure = noise_structure - noise_structure.mean()
        denominator = flat_structure.square().sum().sqrt() * noise_structure.square().sum().sqrt()
        noise_structure_correlation = (
            (flat_structure * noise_structure).sum() / denominator.clamp_min(1.0e-12)
        ).abs()

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
        "flat_radial_rms": float(flat_radial_rms.detach()),
        "noise_structure_correlation": float(noise_structure_correlation.detach()),
        "log_saturation_fraction": float(saturation_fraction.detach()),
        "outputs_finite": float(finite),
    }
