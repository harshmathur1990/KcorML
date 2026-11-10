"""Spatial filter for the persistent fine-scale grain in K-Cor Level-2 pB.

About half of the pixel-scale (1-4 px) variance in a Level-2 frame persists for
10-30 minutes, so temporal methods treat it as signal. It is isotropic, while
coronal structure is radially elongated and CMEs span tens of pixels. The
grain matters only where the corona is faint, so the Gaussian smoothing width
follows the radial signal-to-noise ratio: negligible in the bright inner
corona, up to ``maximum_sigma`` pixels in the faint outer corona.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .profile import PROFILE_BINS

SIGMA_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def sigma_map(
    radius: np.ndarray,
    scale: np.ndarray,
    noise: np.ndarray,
    maximum_sigma: float = 3.0,
    reference_snr: float = 2.5,
) -> np.ndarray:
    """Smoothing width per pixel: ``maximum_sigma * sqrt(reference_snr / snr)``,
    clipped to [0, maximum_sigma]."""
    bins = np.clip((radius * PROFILE_BINS).astype(int), 0, PROFILE_BINS - 1)
    snr = scale[bins] / np.maximum(noise[bins], 1.0e-30)
    return np.clip(maximum_sigma * np.sqrt(reference_snr / np.maximum(snr, 1.0e-6)), 0.0, maximum_sigma)


def grain_filter(image: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Spatially varying Gaussian smoothing of ``image`` (NaN = invalid).

    Normalized convolution keeps occulted and out-of-FOV pixels from leaking
    in; the per-pixel width is interpolated between precomputed levels.
    """
    valid = np.isfinite(image)
    filled = np.where(valid, image, 0.0).astype(np.float64)
    weight = valid.astype(np.float64)
    levels = [s for s in SIGMA_LEVELS if s <= sigmas.max() + 1.0e-9 or s == 0.0]
    if levels[-1] < sigmas.max():
        levels.append(float(sigmas.max()))
    smoothed = []
    for sigma in levels:
        if sigma == 0.0:
            smoothed.append(filled)
        else:
            smoothed.append(gaussian_filter(filled, sigma) / np.maximum(gaussian_filter(weight, sigma), 1.0e-12))
    position = np.interp(sigmas, levels, np.arange(len(levels)))
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(levels) - 1)
    fraction = position - lower
    stack = np.stack(smoothed)
    rows, cols = np.indices(image.shape)
    result = (1.0 - fraction) * stack[lower, rows, cols] + fraction * stack[upper, rows, cols]
    return np.where(valid, result, np.nan).astype(np.float32)
