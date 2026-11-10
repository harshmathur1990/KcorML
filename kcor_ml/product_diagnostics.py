"""Scientific sanity checks for an inference FITS product."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _radial_statistics(
    log_normalization: np.ndarray,
    log_flat: np.ndarray,
    valid: np.ndarray,
    radius: np.ndarray | None = None,
    bins_count: int = 32,
) -> tuple[float, float]:
    if radius is None:
        height, width = valid.shape
        y, x = np.indices(valid.shape, dtype=np.float64)
        center_x = (width - 1.0) / 2.0
        center_y = (height - 1.0) / 2.0
        radius = np.hypot(x - center_x, y - center_y)
        radius /= max(radius.max(), 1.0)
    bins = np.clip((radius * bins_count).astype(int), 0, bins_count - 1)
    normalization_means: list[float] = []
    flat_means: list[float] = []
    for index in range(bins_count):
        selected = valid & (bins == index)
        if selected.any():
            normalization_means.append(float(log_normalization[selected].mean()))
            flat_means.append(float(log_flat[selected].mean()))
    return (
        float(np.std(normalization_means)),
        float(np.sqrt(np.mean(np.square(flat_means)))),
    )


def _structure_correlation(first: np.ndarray, second: np.ndarray, valid: np.ndarray) -> float:
    interior = (
        valid[1:-1, 1:-1]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
    )
    if not interior.any():
        return 0.0

    def laplacian(array: np.ndarray) -> np.ndarray:
        return (
            -4.0 * array[1:-1, 1:-1]
            + array[:-2, 1:-1]
            + array[2:, 1:-1]
            + array[1:-1, :-2]
            + array[1:-1, 2:]
        )[interior]

    left = laplacian(first)
    right = laplacian(second)
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(left, right)[0, 1]))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _laplacian_std(array: np.ndarray, valid: np.ndarray) -> float:
    interior = (
        valid[1:-1, 1:-1]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
    )
    if not interior.any():
        return 0.0
    laplacian = (
        -4.0 * array[1:-1, 1:-1]
        + array[:-2, 1:-1]
        + array[2:, 1:-1]
        + array[1:-1, :-2]
        + array[1:-1, 2:]
    )[interior]
    return float(laplacian.std())


def diagnose_product(
    path: str | Path,
    minimum_log_std: float = 1.0e-3,
    maximum_flat_radial_rms: float = 2.0e-2,
    maximum_noise_structure_correlation: float = 0.5,
    maximum_log_saturation_fraction: float = 1.0e-3,
    maximum_residual_crossframe_correlation: float = 0.2,
) -> dict[str, object]:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("product diagnostics require astropy") from exc

    frames: list[dict[str, float | bool]] = []
    residuals: list[np.ndarray] = []
    with fits.open(path, memmap=True) as hdus:
        input_scale = hdus[0].header.get("INPSCALE")
        log_limit = hdus[0].header.get("LOGLIMIT")
        valid = np.asarray(hdus["VALID_MASK"].data, dtype=bool)
        has_observed = "OBSERVED_PB_1" in hdus and "CLEAN_PB_1" in hdus
        for frame in (1, 2):
            normalization = np.asarray(hdus[f"NORM_FIELD_{frame}"].data, dtype=np.float64)
            flat = np.asarray(hdus[f"FLAT_CORONA_{frame}"].data, dtype=np.float64)
            noise = np.asarray(hdus[f"NOISE_SCALE_{frame}"].data, dtype=np.float64)
            selected = valid & np.isfinite(normalization) & np.isfinite(flat) & np.isfinite(noise)
            positive = selected & (normalization > 0) & (flat > 0) & (noise > 0)
            if not positive.any():
                frames.append({"frame": frame, "finite_positive": False})
                continue
            log_normalization = np.log(normalization[positive])
            log_flat = np.log(flat[positive])
            log_normalization_image = np.log(np.clip(normalization, 1.0e-30, None))
            log_flat_image = np.log(np.clip(flat, 1.0e-30, None))
            log_noise_image = np.log(np.clip(noise, 1.0e-30, None))
            radius_name = f"RADIAL_COORD_{frame}"
            radius = (
                np.asarray(hdus[radius_name].data, dtype=np.float64)
                if radius_name in hdus
                else None
            )
            normalization_radial_std, flat_radial_rms = _radial_statistics(
                log_normalization_image, log_flat_image, positive, radius
            )
            noise_structure_correlation = _structure_correlation(
                log_flat_image, log_noise_image, positive
            )
            saturation_fraction = 0.0
            if input_scale is not None and log_limit is not None:
                log_normalization_scaled = np.log(
                    np.clip(normalization[positive] / float(input_scale), 1.0e-30, None)
                )
                saturated = (
                    np.abs(log_normalization_scaled) >= float(log_limit) - 1.0e-4
                ) | (np.abs(log_flat) >= float(log_limit) - 1.0e-4)
                saturation_fraction = float(saturated.mean())
            separation: dict[str, float] = {
                # A flat corona carrying a negligible share of the log signal
                # means the normalization field has absorbed the corona.
                "flat_log_variance_share": float(
                    log_flat.var() / max((log_normalization + log_flat).var(), 1.0e-30)
                ),
                "clean_normalization_log_correlation": _correlation(
                    log_normalization + log_flat, log_normalization
                ),
                "noise_normalization_log_correlation": _correlation(
                    log_noise_image[positive], log_normalization
                ),
            }
            if has_observed:
                observed = np.asarray(hdus[f"OBSERVED_PB_{frame}"].data, dtype=np.float64)
                clean = np.asarray(hdus[f"CLEAN_PB_{frame}"].data, dtype=np.float64)
                residual = observed - clean
                residuals.append(residual)
                standardized = residual[positive] / noise[positive]
                observed_fine = _laplacian_std(observed, positive)
                separation.update(
                    {
                        "standardized_residual_mean": float(standardized.mean()),
                        "standardized_residual_median_abs": float(np.median(np.abs(standardized))),
                        "coverage_1sigma": float((np.abs(standardized) < 1.0).mean()),
                        "coverage_2sigma": float((np.abs(standardized) < 2.0).mean()),
                        "fine_structure_ratio": _laplacian_std(clean, positive)
                        / max(observed_fine, 1.0e-30),
                    }
                )
            frames.append(
                {
                    **separation,
                    "frame": frame,
                    "finite_positive": bool(positive.sum() == valid.sum()),
                    "normalization_log_std": float(log_normalization.std()),
                    "normalization_radial_log_std": normalization_radial_std,
                    "normalization_p01": float(np.percentile(normalization[positive], 1)),
                    "normalization_p99": float(np.percentile(normalization[positive], 99)),
                    "flat_log_std": float(log_flat.std()),
                    "flat_radial_rms": flat_radial_rms,
                    "noise_median": float(np.median(noise[positive])),
                    "noise_structure_correlation": noise_structure_correlation,
                    "log_saturation_fraction": saturation_fraction,
                }
            )

    # The two frames are 15 s apart, so their noise is independent while the
    # corona is nearly identical. A residual shared by both frames is coronal
    # structure the reconstruction failed to keep.
    residual_crossframe_correlation = None
    if len(residuals) == 2:
        residual_crossframe_correlation = _correlation(residuals[0][valid], residuals[1][valid])

    # Gate on the radial statistic, matching training checkpoint eligibility.
    eligible = (
        residual_crossframe_correlation is None
        or residual_crossframe_correlation <= maximum_residual_crossframe_correlation
    ) and all(
        frame.get("finite_positive", False)
        and frame.get("normalization_radial_log_std", 0.0) >= minimum_log_std
        and frame.get("flat_radial_rms", float("inf")) <= maximum_flat_radial_rms
        and frame.get("noise_structure_correlation", float("inf"))
        <= maximum_noise_structure_correlation
        and frame.get("log_saturation_fraction", float("inf"))
        <= maximum_log_saturation_fraction
        for frame in frames
    )
    return {
        "path": str(path),
        "eligible": eligible,
        "residual_crossframe_correlation": residual_crossframe_correlation,
        "frames": frames,
    }
