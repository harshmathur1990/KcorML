"""Train, evaluate, or apply the single-image K-Cor pB denoiser.

    torchrun --standalone --nproc_per_node=4 denoise.py --config configs/denoise_n2n.json --train
    python denoise.py --checkpoint artifacts/denoise_n2n/best.pt --evaluate --pairs 0 250 500 1000 1500
    python denoise.py --checkpoint artifacts/denoise_n2n/best.pt --predict data/<frame>.fts.gz --output out.fits
    python denoise.py --checkpoint artifacts/denoise_n2n/best.pt --predict-glob "data/20190105_*" --output artifacts/denoised
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kcor_ml.denoise.train import DenoiseConfig, Denoiser, Noise2NoiseTrainer


def write_product(path: str | Path, source: str | Path, observed, clean, valid) -> None:
    from astropy.io import fits

    header = fits.getheader(source)
    hdus = [fits.PrimaryHDU(header=header)]
    for name, array in (("OBSERVED_PB", observed), ("DENOISED_PB", clean), ("RESIDUAL_PB", observed - clean)):
        hdus.append(fits.ImageHDU(np.where(valid, array, np.nan).astype(np.float32), name=name))
    hdus.append(fits.ImageHDU(valid.astype(np.uint8), name="VALID_MASK"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def predict_many(denoiser: Denoiser, pattern: str, output_dir: Path, checkpoint: str) -> None:
    """Write one denoised Level-2 FITS per input frame into <output_dir>/<YYYYMMDD>/."""
    from glob import glob

    from astropy.io import fits

    paths = sorted(glob(pattern))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    for count, path in enumerate(paths, start=1):
        name = Path(path).name.split(".")[0]
        destination = output_dir / name[:8] / f"{name}_denoised.fts"
        if destination.exists():
            continue
        _, clean, valid = denoiser.denoise_file(path)
        header = fits.getheader(path)
        header["DENOISED"] = (True, "noise removed by single-frame Noise2Noise U-Net")
        header["DNCKPT"] = (Path(checkpoint).parent.name + "/" + Path(checkpoint).name, "denoiser checkpoint")
        header["DNSOURCE"] = (Path(path).name, "input Level-2 pB frame")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(np.where(valid, clean, np.nan).astype(np.float32), header).writeto(destination)
        if count % 200 == 0 or count == len(paths):
            print(f"{count}/{len(paths)} {destination}", flush=True)


def evaluate(denoiser: Denoiser, pair_indices: list[int], output_dir: Path) -> None:
    """Full-frame checks on test pairs: frames are 15 s apart, so after
    denoising their residuals should be uncorrelated noise."""
    from kcor_ml.data.records import PairManifest

    from kcor_ml.data.dataset import read_fits_image
    from kcor_ml.denoise.profile import PROFILE_BINS

    test = PairManifest.load(denoiser.config.manifest).splits["test"]
    # Radial bands in pixels from the image center (K-Cor science annulus ~188-504 px).
    bands = {"inner_188_250": (188, 250), "middle_250_350": (250, 350), "outer_350_504": (350, 504)}
    for index in pair_indices:
        pair = test[index]
        observed_1, clean_1, valid_1 = denoiser.denoise_file(pair.first.path)
        observed_2, clean_2, valid_2 = denoiser.denoise_file(pair.second.path)
        _, _, radius = read_fits_image(pair.first.path)
        valid = valid_1 & valid_2
        residual_1, residual_2 = observed_1 - clean_1, observed_2 - clean_2
        noise = np.ones_like(observed_1)
        if denoiser.noise is not None:
            noise = denoiser.noise[np.clip((radius * PROFILE_BINS).astype(int), 0, PROFILE_BINS - 1)]
        pixels = radius * float(np.hypot(*(np.array(observed_1.shape) - 1) / 2.0))
        result = {"pair": index, "source": Path(pair.first.path).name}
        for name, (inner, outer) in bands.items():
            band = valid & (pixels >= inner) & (pixels < outer)
            # In units of the per-frame noise: identity -> ~2, perfect denoiser -> ~1.
            identity = np.mean(((observed_1 - observed_2) / noise)[band] ** 2)
            denoised = np.mean(((clean_1 - observed_2) / noise)[band] ** 2)
            result[name] = {
                "residual_crossframe_correlation": round(
                    float(np.corrcoef(residual_1[band], residual_2[band])[0, 1]), 4
                ),
                "noise_removed_fraction": round(float((identity - denoised) / (identity / 2.0)), 4),
                "residual_rms_over_noise": round(float(np.sqrt(np.mean((residual_1 / noise)[band] ** 2))), 4),
            }
        print(json.dumps(result), flush=True)
        write_product(output_dir / f"test_{index:06d}.fits", pair.first.path, observed_1, clean_1, valid_1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/denoise_n2n.json")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--evaluate", action="store_true", help="full-frame checks on test pairs")
    modes.add_argument("--predict", metavar="FITS", help="denoise one Level-2 pB frame")
    modes.add_argument("--predict-glob", metavar="PATTERN", help="denoise every matching frame")
    parser.add_argument("--checkpoint")
    parser.add_argument("--pairs", type=int, nargs="+", default=[0, 250, 500, 1000, 1500])
    parser.add_argument("--output")
    arguments = parser.parse_args()

    if arguments.train:
        from kcor_ml.distributed import DistributedContext

        config = DenoiseConfig.load(arguments.config)
        distributed = DistributedContext.initialize(config.device)
        distributed.seed_everything(config.seed)
        try:
            Noise2NoiseTrainer(config, distributed).fit()
        finally:
            distributed.close()
        return

    if not arguments.checkpoint:
        raise SystemExit("--checkpoint is required")
    denoiser = Denoiser(arguments.checkpoint)
    if arguments.evaluate:
        output_dir = Path(arguments.output or Path(arguments.checkpoint).parent / "evaluation")
        evaluate(denoiser, arguments.pairs, output_dir)
    elif arguments.predict_glob:
        predict_many(denoiser, arguments.predict_glob, Path(arguments.output or "artifacts/denoised"), arguments.checkpoint)
    else:
        observed, clean, valid = denoiser.denoise_file(arguments.predict)
        destination = arguments.output or Path(arguments.predict).name.split(".")[0] + "_denoised.fits"
        write_product(destination, arguments.predict, observed, clean, valid)
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
