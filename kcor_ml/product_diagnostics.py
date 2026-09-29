"""Scientific sanity checks for an inference FITS product."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def diagnose_product(path: str | Path, minimum_log_std: float = 1.0e-3) -> dict[str, object]:
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
            frames.append(
                {
                    "frame": frame,
                    "finite_positive": bool(positive.sum() == valid.sum()),
                    "normalization_log_std": float(log_normalization.std()),
                    "normalization_p01": float(np.percentile(normalization[positive], 1)),
                    "normalization_p99": float(np.percentile(normalization[positive], 99)),
                    "flat_log_std": float(log_flat.std()),
                    "noise_median": float(np.median(noise[positive])),
                }
            )

    eligible = all(
        frame.get("finite_positive", False)
        and frame.get("normalization_log_std", 0.0) >= minimum_log_std
        for frame in frames
    )
    return {"path": str(path), "eligible": eligible, "frames": frames}
