"""Train, evaluate, or apply the multi-frame K-Cor denoiser (one clean frame per raw frame).

    torchrun --standalone --nproc_per_node=4 denoise_temporal.py --config configs/denoise_temporal.json --train
    python denoise_temporal.py --checkpoint artifacts/denoise_temporal/best.pt --evaluate --grain-sigma 3
    python denoise_temporal.py --checkpoint artifacts/denoise_temporal/best.pt --predict-day 20190105 --grain-sigma 3

``--grain-sigma`` adds the SNR-adaptive spatial filter for the persistent
fine-scale grain (see kcor_ml/denoise/grain.py); 0 disables it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from kcor_ml.denoise.grain import grain_filter, sigma_map
from kcor_ml.denoise.profile import PROFILE_BINS
from kcor_ml.denoise.temporal import TemporalConfig, TemporalDenoiser, TemporalTrainer, split_days

# Radial bands in pixels from the image centre (science annulus ~188-504 px).
BANDS = ((188, 250), (250, 350), (350, 504))
KM_PER_PIXEL_PER_FRAME = 5.556 * 725.0 / 15.0  # arcsec/px * km/arcsec / s per frame


def write_fits(destination: Path, source: str, clean: np.ndarray, checkpoint: str, grain_sigma: float) -> None:
    from astropy.io import fits

    header = fits.getheader(source)
    header["DENOISED"] = (True, "multi-frame Noise2Noise U-Net, native cadence")
    header["DNCKPT"] = (Path(checkpoint).parent.name + "/" + Path(checkpoint).name, "denoiser checkpoint")
    header["DNSOURCE"] = (Path(source).name, "input Level-2 pB frame")
    header["DNGRAIN"] = (grain_sigma, "max px of SNR-adaptive grain filter (0 = off)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(clean.astype(np.float32), header).writeto(destination, overwrite=True)


def predict_day(
    denoiser: TemporalDenoiser,
    day: str,
    output_dir: Path,
    checkpoint: str,
    sigmas: np.ndarray | None,
    grain_sigma: float,
    shard: tuple[int, int] = (0, 1),
) -> None:
    centers = [
        index for index in range(len(denoiser.cache.paths)) if day == "all" or denoiser.cache.day(index) == day
    ]
    centers = centers[shard[0] :: shard[1]]
    for count, center in enumerate(centers, start=1):
        source = denoiser.cache.paths[center]
        destination = output_dir / denoiser.cache.day(center) / f"{Path(source).name.split('.')[0]}_denoised.fts"
        if not destination.exists():
            clean = denoiser.denoise(center)
            if sigmas is not None:
                clean = grain_filter(clean, sigmas)
            write_fits(destination, source, clean, checkpoint, grain_sigma)
        if count % 200 == 0 or count == len(centers):
            print(f"{day}: {count}/{len(centers)}", flush=True)


def _neighbour_mean(stack: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


def _highpass_std(image: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> float:
    filled = np.where(valid, image, 0.0)
    return float(np.std((filled - gaussian_filter(filled, 1.5))[mask]))


def evaluate(
    denoiser: TemporalDenoiser,
    segment_starts: list[int],
    sigmas: np.ndarray | None,
    day: str | None = None,
    length: int = 24,
) -> None:
    cache = denoiser.cache
    test_day = day or sorted(split_days(denoiser.config.manifest)["test"])[0]
    day_indices = [index for index in range(len(cache.paths)) if cache.day(index) == test_day]
    k = denoiser.config.neighbours
    height, width = cache.radius.shape
    pixels = cache.radius * float(np.hypot(height - 1, width - 1) / 2.0)
    noise = denoiser.noise[np.clip((cache.radius * PROFILE_BINS).astype(int), 0, PROFILE_BINS - 1)]
    methods = ["original", "running mean of 9", "model"] + (["model + grain filter"] if sigmas is not None else [])

    def outputs(frames: np.ndarray, centers: range) -> dict[str, list[np.ndarray]]:
        """frames: [length + 2k, H, W] with NaN invalid; centers index into frames."""
        result = {name: [] for name in methods}
        for center in centers:
            stack = np.concatenate((frames[center - k : center], frames[center + 1 : center + k + 1]))
            model = denoiser(stack)
            result["original"].append(frames[center])
            result["running mean of 9"].append(_neighbour_mean(frames[center - k : center + k + 1]))
            result["model"].append(model)
            if sigmas is not None:
                result["model + grain filter"].append(grain_filter(model, sigmas))
        return result

    # Blob tests: (radius px, size sigma px, speed px/frame).
    blob_cases = [(r, size, 0.0) for r in (300.0, 420.0) for size in (3.0, 5.0, 10.0)]
    blob_cases += [(r, 5.0, speed) for r in (300.0, 420.0) for speed in (2.0, 4.0)]
    flicker = {name: {band: [] for band in BANDS} for name in methods}
    persistent = {name: {band: [] for band in BANDS} for name in methods}
    injection: dict[tuple, list] = {}
    y, x = np.indices((height, width))
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    direction = np.array([np.sin(np.radians(35)), np.cos(np.radians(35))])
    for start in segment_starts:
        indices = day_indices[start : start + length + 2 * k]
        frames = np.stack([cache.frames[index] for index in indices]).astype(np.float64)
        centers = range(k, k + length)
        clean_outputs = outputs(frames, centers)
        valid = np.isfinite(frames).all(axis=0)
        interior = valid & (gaussian_filter((~valid).astype(float), 4) < 1.0e-3)
        for name, series in clean_outputs.items():
            series = np.stack(series)
            for band in BANDS:
                mask = interior & (pixels >= band[0]) & (pixels < band[1])
                differences = series[1:] - series[:-1]
                flicker[name][band].append(np.mean([np.std(d[mask]) for d in differences]))
                # Fine grain that survives a 6-minute mean is the persistent component.
                persistent[name][band].append(_highpass_std(series.mean(axis=0), valid, mask))

        for case in blob_cases:
            radius, size, speed = case
            blobs = []
            for t in range(frames.shape[0]):
                r = radius + speed * (t - k)
                py, px = cy + r * direction[0], cx + r * direction[1]
                amplitude = 1.5 * noise[int(round(py)), int(round(px))]
                blobs.append(amplitude * np.exp(-((y - py) ** 2 + (x - px) ** 2) / (2 * size**2)))
            blobs = np.stack(blobs)
            injected_outputs = outputs(frames + blobs, range(k, k + 6))
            for name in methods:
                for i, center in enumerate(range(k, k + 6)):
                    response = np.nan_to_num(injected_outputs[name][i] - clean_outputs[name][i])
                    truth = blobs[center]
                    region = truth > 0.01 * truth.max()
                    amplitude = float((response * truth)[region].sum() / (truth * truth)[region].sum())
                    error = float(np.linalg.norm((response - truth)[region]) / np.linalg.norm(truth[region]))
                    injection.setdefault((case, name), []).append((amplitude, error))

    print(f"\nTest day {test_day}, {len(segment_starts)} segments of {length} frames")
    print("\nFRAME-TO-FRAME FLICKER (std of t+1 minus t) / original   |   PERSISTENT FINE GRAIN (high-pass of 6-min mean) / original")
    for band in BANDS:
        flick = "  ".join(
            f"{name}: {np.mean(flicker[name][band]) / np.mean(flicker['original'][band]):.2f}" for name in methods[1:]
        )
        pers = "  ".join(
            f"{name}: {np.mean(persistent[name][band]) / np.mean(persistent['original'][band]):.2f}"
            for name in methods[1:]
        )
        print(f"  {band[0]}-{band[1]} px | {flick} | {pers}")
    print("\nINJECTED BLOB, amplitude 1.5x per-frame noise: kept amplitude (1 = all) / shape error (0 = exact)")
    for case in blob_cases:
        radius, size, speed = case
        line = f"  r={radius:.0f}px size={size:.0f}px speed={speed:.0f}px/frame (~{speed * KM_PER_PIXEL_PER_FRAME:.0f} km/s):"
        for name in methods[1:]:
            amplitude, error = np.mean(injection[(case, name)], axis=0)
            line += f"  {name} {amplitude:.2f}/{error:.2f} |"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/denoise_temporal.json")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--evaluate", action="store_true", help="flicker, grain, and injected-blob tests on the test day")
    modes.add_argument("--predict-day", metavar="YYYYMMDD", help="a UTC date, or 'all' for every cached frame")
    parser.add_argument("--checkpoint")
    parser.add_argument("--grain-sigma", type=float, default=0.0, help="max smoothing px in the faintest corona")
    parser.add_argument("--segments", type=int, nargs="+", default=[100, 600, 1200, 1700])
    parser.add_argument("--output", default="artifacts/denoised_temporal")
    parser.add_argument("--shard", default="0/1", help="i/n: process every n-th frame starting at i")
    parser.add_argument("--cache", help="frame cache directory (default: the training cache)")
    parser.add_argument("--day", help="YYYYMMDD to evaluate (default: the test split day)")
    arguments = parser.parse_args()

    if arguments.train:
        from kcor_ml.distributed import DistributedContext

        config = TemporalConfig.load(arguments.config)
        distributed = DistributedContext.initialize(config.device)
        distributed.seed_everything(config.seed)
        try:
            TemporalTrainer(config, distributed).fit()
        finally:
            distributed.close()
        return
    if not arguments.checkpoint:
        raise SystemExit("--checkpoint is required")
    denoiser = TemporalDenoiser(arguments.checkpoint, cache_dir=arguments.cache)
    sigmas = None
    if arguments.grain_sigma > 0:
        sigmas = sigma_map(
            denoiser.cache.radius, denoiser.scale.cpu().numpy(), denoiser.noise, maximum_sigma=arguments.grain_sigma
        )
    if arguments.evaluate:
        evaluate(denoiser, arguments.segments, sigmas, arguments.day)
    else:
        shard = tuple(int(part) for part in arguments.shard.split("/"))
        predict_day(
            denoiser, arguments.predict_day, Path(arguments.output), arguments.checkpoint, sigmas,
            arguments.grain_sigma, shard,
        )


if __name__ == "__main__":
    main()
