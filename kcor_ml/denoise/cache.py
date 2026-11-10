"""Memory-mapped frame cache for fast multi-frame access.

All Level-2 frames are decoded once into one float32 array of shape
[frames, height, width], with invalid (occulted, out-of-FOV) pixels set to NaN.
Crops of neighbouring frames are then cheap reads from the OS page cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..data.dataset import read_fits_image


def _observed_at(path: str) -> datetime:
    from astropy.io import fits

    return datetime.fromisoformat(fits.getheader(path)["DATE-OBS"])


def _load(path: str) -> np.ndarray:
    image, valid, _ = read_fits_image(path)
    return np.where(valid, image, np.nan).astype(np.float32)


def build_cache(paths: list[str], directory: str | Path, workers: int = 16) -> None:
    """Decode frames in parallel and write them sequentially; parallel random
    writes into one large file are very slow on network filesystems. A
    RAM-backed directory such as /dev/shm is fastest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(paths)
    with Pool(workers) as pool:
        times = pool.map(_observed_at, paths)
    order = np.argsort(np.array(times))
    paths = [paths[i] for i in order]
    times = [times[i] for i in order]
    _, _, radius = read_fits_image(paths[0])
    np.save(directory / "radius.npy", radius.astype(np.float32))
    frames = np.lib.format.open_memmap(
        directory / "frames.npy", mode="w+", dtype=np.float32, shape=(len(paths), *radius.shape)
    )
    with Pool(workers) as pool:
        for index, frame in enumerate(pool.imap(_load, paths, chunksize=4)):
            frames[index] = frame
            if (index + 1) % 1000 == 0:
                print(f"cached {index + 1}/{len(paths)} frames", flush=True)
    frames.flush()
    del frames
    index = [{"path": path, "observed_at": time.isoformat()} for path, time in zip(paths, times)]
    (directory / "index.json").write_text(json.dumps(index, indent=1) + "\n")


@dataclass(slots=True)
class FrameCache:
    frames: np.ndarray
    radius: np.ndarray
    paths: list[str]
    times: np.ndarray  # seconds since the first frame

    @classmethod
    def open(cls, directory: str | Path) -> "FrameCache":
        directory = Path(directory)
        index = json.loads((directory / "index.json").read_text())
        times = [datetime.fromisoformat(item["observed_at"]) for item in index]
        seconds = np.array([(time - times[0]).total_seconds() for time in times])
        return cls(
            frames=np.load(directory / "frames.npy", mmap_mode="r"),
            radius=np.load(directory / "radius.npy"),
            paths=[item["path"] for item in index],
            times=seconds,
        )

    def day(self, index: int) -> str:
        return Path(self.paths[index]).name[:8]
