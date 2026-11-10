from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from astropy.io import fits
except ImportError:
    fits = None

try:
    import torch
except ImportError:
    torch = None

FILL = np.float32(-7.0e-9)
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def synthetic_level2(size=128, occulter_arcsec=250.0, platescale=5.0, outer_pixels=58):
    """Level-2-like frame: fill inside the occulter, beyond the FOV, and in corners."""
    center = (size - 1) / 2.0
    y, x = np.indices((size, size), dtype=np.float32)
    radius = np.hypot(x - center, y - center)
    # Noise makes faint outer samples negative, as in real frames; they must
    # not be treated as fill.
    noise = np.random.default_rng(0).normal(0.0, 2.0e-8, radius.shape)
    corona = (1.0e-8 * (radius.clip(1.0) / 50.0) ** -3 + noise).astype(np.float32)
    occulter = occulter_arcsec / platescale
    data = np.where((radius <= occulter) | (radius >= outer_pixels), FILL, corona)
    header = fits.Header()
    header["CRPIX1"] = center + 1.0
    header["CRPIX2"] = center + 1.0
    header["CDELT1"] = platescale
    header["OCCLTRID"] = f'OC-{occulter_arcsec:.1f}"'
    return data.astype(np.float32), header, radius, occulter


@unittest.skipIf(fits is None or torch is None, "astropy and PyTorch are required")
class ScienceMaskTests(unittest.TestCase):
    def read(self, data, header):
        from kcor_ml.data.dataset import read_fits_image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.fts"
            fits.PrimaryHDU(data, header).writeto(path)
            return read_fits_image(path)

    def test_occulter_and_outside_fov_are_invalid(self):
        from kcor_ml.data.dataset import OCCULTER_EDGE_MARGIN_PIXELS

        data, header, radius, occulter = synthetic_level2()
        image, valid, _ = self.read(data, header)
        self.assertFalse(valid[radius <= occulter + OCCULTER_EDGE_MARGIN_PIXELS - 0.5].any())
        self.assertFalse(valid[radius >= 58].any())
        self.assertFalse(valid[0, 0])
        self.assertTrue((image[~valid] == 0.0).all())

    def test_science_annulus_is_valid_including_negative_pb(self):
        from kcor_ml.data.dataset import OCCULTER_EDGE_MARGIN_PIXELS

        data, header, radius, occulter = synthetic_level2()
        _, valid, _ = self.read(data, header)
        annulus = (radius >= occulter + OCCULTER_EDGE_MARGIN_PIXELS + 0.5) & (radius < 57)
        self.assertTrue(valid[annulus].all())
        self.assertTrue(valid[annulus & (data < 0)].any())

    def test_occulter_margin_applies_beyond_fill_plateau(self):
        from kcor_ml.data.dataset import OCCULTER_EDGE_MARGIN_PIXELS

        # Real frames end the fill plateau ~2 px outside the nominal occulter.
        data, header, radius, occulter = synthetic_level2()
        data = np.where(radius > occulter, np.abs(data) + 1.0e-8, data).astype(np.float32)
        data[radius >= 58] = FILL
        _, valid, _ = self.read(data, header)
        edge_ring = (radius > occulter + 0.5) & (radius < occulter + OCCULTER_EDGE_MARGIN_PIXELS - 0.5)
        self.assertTrue(edge_ring.any())
        self.assertFalse(valid[edge_ring].any())

    def test_unparseable_occulter_id_falls_back_to_fill_plateau(self):
        from kcor_ml.data.dataset import PLATEAU_EDGE_MARGIN_PIXELS

        data, header, radius, occulter = synthetic_level2()
        header["OCCLTRID"] = "OC-T3-19.486"
        _, valid, _ = self.read(data, header)
        self.assertFalse(valid[radius <= occulter + PLATEAU_EDGE_MARGIN_PIXELS - 0.5].any())
        self.assertTrue(valid[(radius >= occulter + PLATEAU_EDGE_MARGIN_PIXELS + 1.5) & (radius < 57)].all())

    def test_occulter_radius_from_header(self):
        from kcor_ml.data.dataset import occulter_radius_pixels

        header = fits.Header()
        header["OCCLTRID"] = 'OC-1018.9"'
        header["CDELT1"] = 5.556
        self.assertAlmostEqual(occulter_radius_pixels(header), 1018.9 / 5.556)
        self.assertIsNone(occulter_radius_pixels(fits.Header()))

    @unittest.skipUnless(any(DATA_ROOT.glob("*_kcor_l2_pb.fts*")), "K-Cor Level-2 data not present")
    def test_evaluation_crops_cover_science_annulus(self):
        from kcor_ml.data.dataset import KCorPairDataset, read_fits_image

        path = sorted(DATA_ROOT.glob("*_kcor_l2_pb.fts*"))[0]
        image, valid, radius = (torch.from_numpy(a).expand(2, -1, -1) for a in read_fits_image(path))
        dataset = KCorPairDataset([], crop_size=256, training=False)
        for index in range(4):
            _, crop_valid, _ = dataset._crop(image, valid, radius, index)
            # Validation previously used a centered crop with no valid pixels.
            self.assertGreater(crop_valid.float().mean().item(), 0.9)

    @unittest.skipUnless(any(DATA_ROOT.glob("*_kcor_l2_pb.fts*")), "K-Cor Level-2 data not present")
    def test_real_level2_frame(self):
        from kcor_ml.data.dataset import read_fits_image

        path = sorted(DATA_ROOT.glob("*_kcor_l2_pb.fts*"))[0]
        raw, header = fits.getdata(path, header=True)
        _, valid, _ = read_fits_image(path)
        center_y, center_x = header["CRPIX2"] - 1.0, header["CRPIX1"] - 1.0
        y, x = np.indices(valid.shape)
        radius = np.hypot(x - center_x, y - center_y)
        self.assertFalse(valid[radius < 185].any())
        self.assertFalse(valid[radius > 504].any())
        self.assertTrue(valid[(radius > 190) & (radius < 500)].all())
        self.assertFalse((raw[valid] == raw[0, 0]).any())


if __name__ == "__main__":
    unittest.main()
