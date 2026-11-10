from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from astropy.io import fits
except ImportError:
    fits = None


def write_product(path, structure_kept):
    rng = np.random.default_rng(1)
    size = 96
    y, x = np.indices((size, size), dtype=np.float64)
    radius = np.hypot(x - (size - 1) / 2, y - (size - 1) / 2)
    valid = (radius > 12) & (radius < 44)
    normalization = 1.0e-7 * (radius.clip(1.0) / 12.0) ** -2
    streamers = 1.0 + 0.3 * np.cos(6 * np.arctan2(y - size / 2, x - size / 2)) * np.sin(radius / 2)
    true_clean = normalization * streamers
    noise = 0.02 * true_clean.mean() * np.ones_like(true_clean)
    flat = streamers if structure_kept else np.ones_like(streamers)
    hdus = [fits.PrimaryHDU()]
    for frame in (1, 2):
        observed = true_clean + noise * rng.standard_normal(true_clean.shape)
        arrays = {
            "OBSERVED_PB": observed,
            "CLEAN_PB": normalization * flat,
            "FLAT_CORONA": flat,
            "NORM_FIELD": normalization,
            "NOISE_SCALE": noise,
            "RADIAL_COORD": radius / radius.max(),
        }
        for name, array in arrays.items():
            hdus.append(fits.ImageHDU(array.astype(np.float32), name=f"{name}_{frame}"))
    hdus.append(fits.ImageHDU(valid.astype(np.uint8), name="VALID_MASK"))
    fits.HDUList(hdus).writeto(path)


@unittest.skipIf(fits is None, "astropy is required")
class ProductDiagnosticsTests(unittest.TestCase):
    def diagnose(self, structure_kept):
        from kcor_ml.product_diagnostics import diagnose_product

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "product.fits"
            write_product(path, structure_kept)
            return diagnose_product(path, maximum_flat_radial_rms=1.0)

    def test_independent_residuals_pass(self):
        result = self.diagnose(structure_kept=True)
        self.assertLess(abs(result["residual_crossframe_correlation"]), 0.2)
        self.assertTrue(result["eligible"])
        self.assertGreater(result["frames"][0]["flat_log_variance_share"], 0.01)

    def test_structure_absorbed_by_residual_fails(self):
        result = self.diagnose(structure_kept=False)
        self.assertGreater(result["residual_crossframe_correlation"], 0.5)
        self.assertFalse(result["eligible"])
        self.assertLess(result["frames"][0]["flat_log_variance_share"], 1.0e-6)
        self.assertGreater(result["frames"][0]["clean_normalization_log_correlation"], 0.999)


if __name__ == "__main__":
    unittest.main()
