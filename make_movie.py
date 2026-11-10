"""Side-by-side movie of original (left) and denoised (right) K-Cor pB frames for one day.

Both panels show pB divided by the denoiser's fixed radial brightness scale,
with one grayscale stretch shared by both panels and every frame of the day,
so noise and coronal structure can be compared directly at all heights.

    python make_movie.py --day 20190105 --denoised artifacts/denoised \
        --scale artifacts/denoise_n2n/radial_scale.npy --output artifacts/movies/20190105.mp4
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
import subprocess

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw, ImageFont

from kcor_ml.data.dataset import read_fits_image
from kcor_ml.denoise.profile import PROFILE_BINS

HEADER_PIXELS = 44
INVALID_GRAY = 40
# Isophote colours, faintest to brightest level: yellow, orange, red.
ISOPHOTE_COLOURS = ((255, 230, 0), (255, 140, 0), (230, 0, 0))
MINIMUM_REGION_PIXELS = 50
TRACK_COLOUR = (255, 40, 40)


def to_gray(image: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    gray = np.clip((image - low) / (high - low), 0.0, 1.0) * 255.0
    gray = np.where(valid & np.isfinite(image), gray, INVALID_GRAY)
    # FITS row 0 is the bottom of the image.
    return np.flipud(gray).astype(np.uint8)


def isophote_outlines(image: np.ndarray, valid: np.ndarray, levels: list[float]) -> list[np.ndarray]:
    """Outline pixels of the regions brighter than each level in one frame.

    Uses only this frame's intensity (no background or difference image);
    regions smaller than MINIMUM_REGION_PIXELS are dropped. Returned masks
    are flipped like ``to_gray`` and thickened to 2 px for visibility.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, label

    outlines = []
    for level in levels:
        mask = valid & np.isfinite(image) & (np.nan_to_num(image, nan=-np.inf) > level)
        labels, count = label(mask)
        if count:
            sizes = np.bincount(labels.ravel())
            mask = (sizes >= MINIMUM_REGION_PIXELS)[labels] & mask
        outline = mask & ~binary_erosion(mask) & binary_erosion(valid, iterations=2)
        outlines.append(np.flipud(binary_dilation(outline)))
    return outlines


def tracked_region(image: np.ndarray, valid: np.ndarray, level: float, seed: tuple[int, int],
                   previous: np.ndarray | None) -> np.ndarray:
    """The one connected region brighter than ``level`` that holds the CME.

    It is the region containing the seed pixel, or, once the seed drops
    below the level, the region overlapping the previous frame's region most.
    Uses only this frame's intensity (no background or difference image).
    """
    from scipy.ndimage import label

    labels, count = label(valid & (np.nan_to_num(image, nan=-np.inf) > level))
    chosen = labels[seed]
    if not chosen and previous is not None and count:
        overlap = np.bincount(labels[previous], minlength=count + 1)
        overlap[0] = 0
        chosen = int(np.argmax(overlap))
    return (labels == chosen) & (chosen > 0)


def region_outline(region: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion

    outline = region & ~binary_erosion(region) & binary_erosion(valid, iterations=2)
    return np.flipud(binary_dilation(outline, iterations=2))


def colour_panel(gray: np.ndarray, outlines: list[np.ndarray]) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for outline, colour in zip(outlines, ISOPHOTE_COLOURS):
        rgb[outline] = colour
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day", required=True, help="YYYYMMDD, or a glob such as 2020061[23] or *")
    parser.add_argument("--data", default="data")
    parser.add_argument("--denoised", default="artifacts/denoised")
    parser.add_argument("--scale", required=True, help="radial_scale.npy saved with the denoiser")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=23, help="H.264 quality; lower is larger and sharper")
    parser.add_argument("--label", default="", help="extra text for the title, e.g. the data split")
    parser.add_argument("--right-title", default="Denoised", help="title of the right panel")
    parser.add_argument(
        "--isophotes",
        type=float,
        nargs="*",
        help="draw intensity contours at these percentiles of the denoised brightness (e.g. 70 85 95)",
    )
    parser.add_argument(
        "--track",
        type=float,
        nargs=3,
        metavar=("LEVEL", "PA", "RADIUS"),
        help="draw one contour at LEVEL (pB / radial scale) around the region holding the pixel at position "
        "angle PA (degrees from north through east) and RADIUS (pixels from centre)",
    )
    arguments = parser.parse_args()

    sources = sorted(glob(f"{arguments.data}/{arguments.day}_*_kcor_l2_pb.fts*"))
    pairs = []
    for source in sources:
        name = Path(source).name.split(".")[0]
        denoised = Path(arguments.denoised) / name[:8] / f"{name}_denoised.fts"
        if denoised.exists():
            pairs.append((source, denoised))
    if not pairs:
        raise SystemExit(f"no denoised frames found for {arguments.day}")
    print(f"{arguments.day}: {len(pairs)} of {len(sources)} frames have denoised products", flush=True)

    _, _, radius = read_fits_image(pairs[0][0])
    scale = np.load(arguments.scale)[np.clip((radius * PROFILE_BINS).astype(int), 0, PROFILE_BINS - 1)]
    # One stretch for the whole day, from the denoised frames.
    # Median of per-frame percentiles, so a few bad frames (clouds) cannot set the stretch.
    sample = [fits.getdata(denoised) / scale for _, denoised in pairs[:: max(1, len(pairs) // 40)]]
    low, high = np.median([np.nanpercentile(frame, [1.0, 99.5]) for frame in sample], axis=0)
    levels = []
    if arguments.isophotes:
        levels = list(np.median([np.nanpercentile(frame, arguments.isophotes) for frame in sample], axis=0))
        print("isophote levels (pB / radial scale):", ", ".join(f"{level:.3g}" for level in levels), flush=True)
    colour = bool(levels or arguments.track)
    mode, channels = ("RGB", "rgb24") if colour else ("L", "gray")
    region = None
    if arguments.track:
        header = fits.getheader(pairs[0][0])
        angle = np.radians(arguments.track[1])
        # Solar north is +y and east is -x in K-Cor L2 images.
        seed = (
            int(round(header["CRPIX2"] - 1 + arguments.track[2] * np.cos(angle))),
            int(round(header["CRPIX1"] - 1 - arguments.track[2] * np.sin(angle))),
        )

    height, width = radius.shape
    frame_size = (2 * width, height + HEADER_PIXELS)
    font = ImageFont.load_default(size=22)
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", channels, "-s", f"{frame_size[0]}x{frame_size[1]}",
            "-r", str(arguments.fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(arguments.crf), "-preset", "medium",
            arguments.output,
        ],
        stdin=subprocess.PIPE,
    )
    for count, (source, denoised) in enumerate(pairs, start=1):
        observed, valid, _ = read_fits_image(source)
        clean = fits.getdata(denoised).astype(np.float64)
        canvas = Image.new(mode, frame_size, 0)
        if arguments.track:
            # Track on the denoised frame; the original panel gets the same level and seed.
            region = tracked_region(clean / scale, valid, arguments.track[0], seed, region)
            raw_region = tracked_region(observed / scale, valid, arguments.track[0], seed, region)
        for offset, image in ((0, observed / scale), (width, clean / scale)):
            panel = to_gray(image, valid, low, high)
            if levels:
                panel = colour_panel(panel, isophote_outlines(image, valid, levels))
            if arguments.track:
                outline = region_outline(raw_region if offset == 0 else region, valid)
                if panel.ndim == 2:
                    panel = colour_panel(panel, [])
                panel[outline] = TRACK_COLOUR
            canvas.paste(Image.fromarray(panel), (offset, HEADER_PIXELS))
        draw = ImageDraw.Draw(canvas)
        white = (255, 255, 255) if colour else 255
        timestamp = fits.getheader(source).get("DATE-OBS", Path(source).name[:15])
        draw.text((12, 10), f"Original L2 pB   {timestamp}", fill=white, font=font)
        draw.text((width + 12, 10), f"{arguments.right_title}   {arguments.label}", fill=white, font=font)
        encoder.stdin.write(canvas.tobytes())
        if count % 500 == 0:
            print(f"{arguments.day}: {count}/{len(pairs)} frames", flush=True)
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {arguments.output} ({len(pairs)} frames, stretch {low:.3g}..{high:.3g})", flush=True)


if __name__ == "__main__":
    main()
