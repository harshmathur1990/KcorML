"""Fixed radial intensity and noise scales estimated from training frames.

K-Cor pB falls by more than two decades across the field. The network input
is divided by a fixed radial brightness scale (invertible; the output is
multiplied back to physical pB), and the loss is measured in units of the
radial noise level so that a reconstruction error larger than the noise costs
the same in the bright inner corona as in the faint outer corona.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter

from ..data.dataset import read_fits_image

PROFILE_BINS = 512


def _binned(values: np.ndarray, radius: np.ndarray, statistic) -> np.ndarray:
    bins = np.clip((radius * PROFILE_BINS).astype(int), 0, PROFILE_BINS - 1)
    order = np.argsort(bins, kind="stable")
    counts = np.bincount(bins, minlength=PROFILE_BINS)
    result = np.full(PROFILE_BINS, np.nan)
    for index, group in enumerate(np.split(values[order], np.cumsum(counts)[:-1])):
        if group.size:
            result[index] = statistic(group)
    return result


def _finalize(per_frame: list[np.ndarray]) -> np.ndarray:
    profile = np.nanmedian(np.stack(per_frame), axis=0)
    known = np.flatnonzero(np.isfinite(profile) & (profile > 0))
    if known.size == 0:
        raise ValueError("no valid pixels found for the radial profile")
    # Bins outside the science annulus are always masked; fill them from the
    # nearest known bin so the profile stays finite and positive.
    profile = np.interp(np.arange(PROFILE_BINS), known, profile[known])
    return median_filter(profile, size=5, mode="nearest").astype(np.float32)


def radial_scale(paths: list[str | Path]) -> np.ndarray:
    """Median |pB| per normalized-radius bin across frames.

    The absolute value keeps the scale positive in the outer corona, where the
    Level-2 median pB is slightly negative and the scale tends to the noise level.
    """
    per_frame = []
    for path in paths:
        image, valid, radius = read_fits_image(path)
        per_frame.append(_binned(np.abs(image[valid]), radius[valid], np.median))
    return _finalize(per_frame)


def radial_noise(pairs: list[tuple[str | Path, str | Path]]) -> np.ndarray:
    """Robust per-frame noise standard deviation per radius bin.

    For frames 15 s apart, (A - B) / sqrt(2) is dominated by independent noise;
    the MAD estimator resists the few pixels where the corona truly changed.
    """

    def robust_std(values: np.ndarray) -> float:
        return 1.4826 * float(np.median(np.abs(values - np.median(values))))

    per_pair = []
    for first, second in pairs:
        image_1, valid_1, radius = read_fits_image(first)
        image_2, valid_2, _ = read_fits_image(second)
        valid = valid_1 & valid_2
        difference = (image_1[valid] - image_2[valid]) / np.sqrt(2.0)
        per_pair.append(_binned(difference, radius[valid], robust_std))
    return _finalize(per_pair)


def scale_at(scale: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    bins = (radius * PROFILE_BINS).long().clamp(0, PROFILE_BINS - 1)
    return scale.to(radius.device)[bins]
