"""Lazy PyTorch dataset for consecutive K-Cor pB observations."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .records import FramePair

# Level-2 pB rings within ~2 px of the occulter edge contain diffraction
# artifacts (about 15% negative values at r = 186 px for OC-1018.9"), so the
# science annulus starts slightly outside the nominal occulter radius.
OCCULTER_EDGE_MARGIN_PIXELS = 4.0
# The occulted fill plateau ends ~2 px outside the nominal occulter radius,
# so the same science boundary is plateau edge + 2 px.
PLATEAU_EDGE_MARGIN_PIXELS = 2.0


def fill_plateau_radius(fill_mask: np.ndarray, radius_pixels: np.ndarray) -> float | None:
    """Outer radius of the central occulted fill plateau: the innermost
    one-pixel annulus where fill pixels are no longer the majority."""
    bins = radius_pixels.astype(int).ravel()
    total = np.bincount(bins)
    filled = np.bincount(bins, weights=fill_mask.ravel().astype(float), minlength=total.size)
    fraction = filled / np.maximum(total, 1)
    if fraction[0] < 0.5:
        return None
    return float(np.argmax(fraction < 0.5))


def occulter_radius_pixels(header) -> float | None:
    """Return the occulter radius in pixels from OCCLTRID (e.g. 'OC-1018.9"').

    Newer IDs such as 'OC-T3-19.486' do not encode the radius in arcsec;
    for those this returns None and the fill plateau is used instead.
    """
    occulter_id = header.get("OCCLTRID")
    platescale = header.get("CDELT1")
    if not occulter_id or not platescale:
        return None
    try:
        arcsec = float(str(occulter_id).strip().removeprefix("OC-").rstrip('"'))
    except ValueError:
        return None
    return arcsec / abs(float(platescale))


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
    height, width = array.shape
    # Level-2 pB marks occulted and out-of-FOV pixels with a constant fill
    # value (-7e-9, not zero), also present in the array corners. Calibrated
    # coronal pB may legitimately be negative near the outer edge, so only
    # exact matches of the corner fill value are rejected.
    fill = array[0, 0]
    valid = np.isfinite(array)
    if np.isfinite(fill):
        valid &= array != fill
    center_x = float(header.get("CRPIX1", (width + 1.0) / 2.0)) - 1.0
    center_y = float(header.get("CRPIX2", (height + 1.0) / 2.0)) - 1.0
    y, x = np.indices(array.shape, dtype=np.float32)
    radius_pixels = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    corners = np.asarray(
        (
            np.hypot(center_x, center_y),
            np.hypot(width - 1 - center_x, center_y),
            np.hypot(center_x, height - 1 - center_y),
            np.hypot(width - 1 - center_x, height - 1 - center_y),
        ),
        dtype=np.float32,
    )
    farthest_corner = max(float(corners.max()), 1.0)
    radius = radius_pixels / farthest_corner
    # K-Cor's useful field is circular. Mask the geometric corners even when
    # the FITS array stores small finite fill values there.
    inscribed_radius = max(
        min(center_x, width - 1 - center_x, center_y, height - 1 - center_y),
        1.0,
    )
    valid &= radius_pixels <= inscribed_radius
    occulter = occulter_radius_pixels(header)
    if occulter is not None:
        valid &= radius_pixels >= occulter + OCCULTER_EDGE_MARGIN_PIXELS
    elif np.isfinite(fill):
        plateau = fill_plateau_radius(array == fill, radius_pixels)
        if plateau is not None:
            valid &= radius_pixels >= plateau + PLATEAU_EDGE_MARGIN_PIXELS
    return np.where(valid, array, 0.0), valid, radius


def crop_origin(height: int, width: int, size: int, training: bool, index: int) -> tuple[int, int]:
    if training:
        return random.randint(0, height - size), random.randint(0, width - size)
    # A centered crop lies entirely behind the occulter. Evaluate on a
    # mid-annulus crop (2/3 of the half-width from the center), cycling
    # deterministically through the four cardinal directions.
    dy, dx = ((0, 1), (1, 0), (0, -1), (-1, 0))[index % 4]
    top = (height - size) // 2 + dy * (height // 3)
    left = (width - size) // 2 + dx * (width // 3)
    return min(max(top, 0), height - size), min(max(left, 0), width - size)


class KCorPairDataset(Dataset):
    """Return raw image pairs, validity masks, timestamps, and identifiers."""

    def __init__(self, pairs: list[FramePair], crop_size: int | None = None, training: bool = False):
        self.pairs = pairs
        self.crop_size = crop_size
        self.training = training

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop(
        self, images: torch.Tensor, valid: torch.Tensor, radius: torch.Tensor, index: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.crop_size is None:
            return images, valid, radius
        _, height, width = images.shape
        size = self.crop_size
        if height < size or width < size:
            raise ValueError(f"crop size {size} exceeds image shape {(height, width)}")
        top, left = crop_origin(height, width, size, self.training, index)
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
        images, valid, radius = self._crop(images, valid, radius, index)
        return {
            "images": images,
            "valid_mask": valid,
            "radial_coordinate": radius,
            "delta_t": torch.tensor(pair.delta_seconds, dtype=torch.float32),
            "paths": (pair.first.path, pair.second.path),
            "timestamps": (pair.first.observed_at.isoformat(), pair.second.observed_at.isoformat()),
        }
