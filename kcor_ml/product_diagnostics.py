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


def diagnose_product(
    path: str | Path,
    minimum_log_std: float = 1.0e-3,
    maximum_flat_radial_rms: float = 2.0e-2,
    maximum_noise_structure_correlation: float = 0.5,
) -> dict[str, object]:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("product diagnostics require astropy") from exc

    frames: list[dict[str, float | bool]] = []
    with fits.open(path, memmap=True) as hdus:
        valid = np.asarray(hdus["VALID_MASK"].data, dtype=bool)
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
            frames.append(
                {
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
                }
            )

    eligible = all(
        frame.get("finite_positive", False)
        and frame.get("normalization_log_std", 0.0) >= minimum_log_std
        and frame.get("flat_radial_rms", float("inf")) <= maximum_flat_radial_rms
        and frame.get("noise_structure_correlation", float("inf"))
        <= maximum_noise_structure_correlation
        for frame in frames
    )
    return {"path": str(path), "eligible": eligible, "frames": frames}
