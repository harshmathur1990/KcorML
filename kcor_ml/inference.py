"""Model inference and multi-extension FITS product writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .models.outputs import TFHDNOutput


def predict_pair(
    model: torch.nn.Module,
    images: torch.Tensor,
    valid_mask: torch.Tensor,
    delta_t: torch.Tensor,
) -> TFHDNOutput:
    model.eval()
    with torch.inference_mode():
        names = (
            "clean_pb",
            "flat_corona",
            "normalization_field",
            "noise_location",
            "noise_scale",
            "noise_df",
            "cme_probability",
        )
        from_first = images.clone()
        from_first[:, 1] = 0.0
        predicted_second = model(from_first, valid_mask=valid_mask, delta_t=delta_t)
        second_channels = {
            name: getattr(predicted_second, name)[:, 1:2].clone() for name in names
        }
        del predicted_second, from_first

        from_second = images.clone()
        from_second[:, 0] = 0.0
        predicted_first = model(from_second, valid_mask=valid_mask, delta_t=delta_t)

        def combine(name: str) -> torch.Tensor:
            first = getattr(predicted_first, name)[:, 0:1]
            return torch.cat((first, second_channels[name]), dim=1)

        return TFHDNOutput(
            clean_pb=combine("clean_pb"),
            flat_corona=combine("flat_corona"),
            normalization_field=combine("normalization_field"),
            noise_location=combine("noise_location"),
            noise_scale=combine("noise_scale"),
            noise_df=combine("noise_df"),
            cme_probability=combine("cme_probability"),
            auxiliary={},
        )


def _image(output: torch.Tensor, frame: int) -> np.ndarray:
    return output[0, frame].detach().float().cpu().numpy()


def write_fits_product(
    path: str | Path,
    output: TFHDNOutput,
    valid_mask: torch.Tensor,
    *,
    source_paths: tuple[str, str],
    checkpoint: str,
    observed_images: torch.Tensor | None = None,
    overwrite: bool = False,
) -> None:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("FITS output requires astropy; install requirements-ml.txt") from exc

    primary = fits.PrimaryHDU()
    primary.header["MODEL"] = "TF-HDN-V2"
    primary.header["CKPT"] = Path(checkpoint).name
    primary.header["SOURCE1"] = Path(source_paths[0]).name
    primary.header["SOURCE2"] = Path(source_paths[1]).name
    arrays = {
        "CLEAN_PB_1": _image(output.clean_pb, 0),
        "CLEAN_PB_2": _image(output.clean_pb, 1),
        "FLAT_CORONA_1": _image(output.flat_corona, 0),
        "FLAT_CORONA_2": _image(output.flat_corona, 1),
        "NORM_FIELD_1": _image(output.normalization_field, 0),
        "NORM_FIELD_2": _image(output.normalization_field, 1),
        "NOISE_SCALE_1": _image(output.noise_scale, 0),
        "NOISE_SCALE_2": _image(output.noise_scale, 1),
        "CME_PROB_1": _image(output.cme_probability, 0),
        "CME_PROB_2": _image(output.cme_probability, 1),
        "VALID_MASK": valid_mask[0].all(dim=0).detach().cpu().numpy().astype(np.uint8),
    }
    if observed_images is not None:
        arrays = {
            "OBSERVED_PB_1": _image(observed_images, 0),
            "OBSERVED_PB_2": _image(observed_images, 1),
            **arrays,
        }
    hdus = [primary] + [fits.ImageHDU(array, name=name) for name, array in arrays.items()]
    fits.HDUList(hdus).writeto(path, overwrite=overwrite, checksum=True)
