"""Lazy PyTorch dataset for consecutive K-Cor pB observations."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .records import FramePair


def read_fits_image(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("FITS loading requires astropy; install requirements-ml.txt") from exc
    array = np.asarray(fits.getdata(path, memmap=True), dtype=np.float32)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D FITS image, got {array.shape} from {path}")
    valid = np.isfinite(array)
    return np.where(valid, array, 0.0), valid


class KCorPairDataset(Dataset):
    """Return raw image pairs, validity masks, timestamps, and identifiers."""

    def __init__(self, pairs: list[FramePair], crop_size: int | None = None, training: bool = False):
        self.pairs = pairs
        self.crop_size = crop_size
        self.training = training

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop(self, images: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.crop_size is None:
            return images, valid
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
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        first, valid_first = read_fits_image(pair.first.path)
        second, valid_second = read_fits_image(pair.second.path)
        if first.shape != second.shape:
            raise ValueError(f"pair shape changed after indexing: {first.shape} != {second.shape}")
        images = torch.from_numpy(np.stack((first, second), axis=0))
        valid = torch.from_numpy(np.stack((valid_first, valid_second), axis=0))
        images, valid = self._crop(images, valid)
        return {
            "images": images,
            "valid_mask": valid,
            "delta_t": torch.tensor(pair.delta_seconds, dtype=torch.float32),
            "paths": (pair.first.path, pair.second.path),
            "timestamps": (pair.first.observed_at.isoformat(), pair.second.observed_at.isoformat()),
        }

