"""Lazy PyTorch dataset for consecutive K-Cor pB observations."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .records import FramePair


def read_fits_image(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("FITS loading requires astropy; install requirements-ml.txt") from exc
    raw, header = fits.getdata(path, memmap=True, header=True)
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D FITS image, got {array.shape} from {path}")
    # Exact zero plateaus in Level-2 pB mark occulted/out-of-FOV pixels. A
    # continuously calibrated pB measurement may be negative, but an exact
    # zero is not treated as a scientific sample.
    valid = np.isfinite(array) & (array != 0.0)
    height, width = array.shape
    center_x = float(header.get("CRPIX1", (width + 1.0) / 2.0)) - 1.0
    center_y = float(header.get("CRPIX2", (height + 1.0) / 2.0)) - 1.0
    y, x = np.indices(array.shape, dtype=np.float32)
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    corners = np.asarray(
        (
            np.hypot(center_x, center_y),
            np.hypot(width - 1 - center_x, center_y),
            np.hypot(center_x, height - 1 - center_y),
            np.hypot(width - 1 - center_x, height - 1 - center_y),
        ),
        dtype=np.float32,
    )
    radius /= max(float(corners.max()), 1.0)
    return np.where(valid, array, 0.0), valid, radius


class KCorPairDataset(Dataset):
    """Return raw image pairs, validity masks, timestamps, and identifiers."""

    def __init__(self, pairs: list[FramePair], crop_size: int | None = None, training: bool = False):
        self.pairs = pairs
        self.crop_size = crop_size
        self.training = training

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop(
        self, images: torch.Tensor, valid: torch.Tensor, radius: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.crop_size is None:
            return images, valid, radius
        _, height, width = images.shape
        size = self.crop_size
        if height < size or width < size:
            raise ValueError(f"crop size {size} exceeds image shape {(height, width)}")
        if self.training:
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)
        else:
            top, left = (height - size) // 2, (width - size) // 2
        return (
            images[:, top : top + size, left : left + size],
            valid[:, top : top + size, left : left + size],
            radius[:, top : top + size, left : left + size],
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        first, valid_first, radius_first = read_fits_image(pair.first.path)
        second, valid_second, radius_second = read_fits_image(pair.second.path)
        if first.shape != second.shape:
            raise ValueError(f"pair shape changed after indexing: {first.shape} != {second.shape}")
        images = torch.from_numpy(np.stack((first, second), axis=0))
        valid = torch.from_numpy(np.stack((valid_first, valid_second), axis=0))
        radius = torch.from_numpy(np.stack((radius_first, radius_second), axis=0))
        images, valid, radius = self._crop(images, valid, radius)
        return {
            "images": images,
            "valid_mask": valid,
            "radial_coordinate": radius,
            "delta_t": torch.tensor(pair.delta_seconds, dtype=torch.float32),
            "paths": (pair.first.path, pair.second.path),
            "timestamps": (pair.first.observed_at.isoformat(), pair.second.observed_at.isoformat()),
        }
