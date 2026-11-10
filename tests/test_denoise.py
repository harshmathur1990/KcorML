import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DenoiserTests(unittest.TestCase):
    def test_untrained_model_is_identity_on_valid_pixels(self):
        from kcor_ml.denoise.model import DenoisingUNet

        model = DenoisingUNet((8, 16), blocks_per_level=1)
        image = torch.randn(2, 1, 33, 47)
        valid = torch.rand(2, 1, 33, 47) > 0.2
        radius = torch.rand(2, 1, 33, 47)
        output = model(image, valid, radius)
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.allclose(output, image * valid))

    def test_untrained_multiframe_model_is_masked_neighbour_mean(self):
        from kcor_ml.denoise.model import DenoisingUNet

        model = DenoisingUNet((8, 16), blocks_per_level=1, input_frames=4)
        images = torch.randn(2, 4, 20, 20)
        valid = torch.rand(2, 4, 20, 20) > 0.3
        output = model(images, valid, torch.rand(2, 1, 20, 20))
        weights = valid.float()
        expected = (images * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(1.0)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_neighbours_exclude_center_and_respect_gaps_and_days(self):
        import numpy as np

        from kcor_ml.denoise.cache import FrameCache
        from kcor_ml.denoise.temporal import neighbour_indices

        # Day A: frames at 0, 15, 30, 90 s (a gap), day B starts right after.
        times = np.array([0.0, 15.0, 30.0, 90.0, 105.0, 120.0])
        paths = [f"x/2019010{1 if i < 4 else 2}_{i:06d}_kcor_l2_pb.fts" for i in range(6)]
        cache = FrameCache(frames=np.zeros((6, 1, 1)), radius=np.zeros((1, 1)), paths=paths, times=times)
        self.assertEqual(neighbour_indices(cache, 1, 2, 20.0), [None, 0, 2, None])
        # Frame 3 is 60 s after frame 2, beyond 1 * 20 s; frames 4-5 are another day.
        self.assertEqual(neighbour_indices(cache, 3, 2, 20.0), [None, None, None, None])
        self.assertNotIn(1, neighbour_indices(cache, 1, 2, 20.0))
        # Inference may cross midnight UTC: frame 4 (next date) is 15 s after frame 3.
        self.assertEqual(neighbour_indices(cache, 3, 2, 20.0, same_day=False), [None, None, 4, 5])

    def test_moving_features_follow_their_velocity_in_every_frame(self):
        import numpy as np

        from kcor_ml.denoise.temporal import add_moving_features

        class FixedGenerator:
            """One feature at (30, 30), 4 px/frame along +y (axis 0), round, amplitude 2x noise."""

            def integers(self, low, high):
                return 1

            def uniform(self, low, high, size=None):
                values = {(0, 64): np.array([30.0, 30.0]), (0.0, 4.0): 4.0, (3.0, 12.0): np.array([4.0, 4.0]),
                          (0.5, 4.0): 2.0}
                if (low, high) == (0, 2 * np.pi):
                    return np.pi / 2
                return values.get((low, high), 0.0)

        stack = np.zeros((3, 64, 64), dtype=np.float32)
        target = np.zeros((1, 64, 64), dtype=np.float32)
        add_moving_features(stack, target, [-1.0, None, 2.0], np.ones((64, 64)), FixedGenerator(), 4.0)
        peak = lambda image: np.unravel_index(np.argmax(image), image.shape)
        self.assertEqual(peak(target[0]), (30, 30))
        self.assertEqual(peak(stack[0]), (26, 30))
        self.assertEqual(peak(stack[2]), (38, 30))
        self.assertFalse(stack[1].any())
        self.assertAlmostEqual(float(target.max()), 2.0, places=5)

    def test_grain_filter_smooths_by_snr_and_respects_mask(self):
        import numpy as np

        from kcor_ml.denoise.grain import grain_filter, sigma_map

        rng = np.random.default_rng(0)
        image = rng.normal(5.0, 1.0, (64, 64))
        image[:, :8] = np.nan
        # Left half high SNR (no smoothing), right half low SNR (maximum smoothing).
        sigmas = np.zeros((64, 64))
        sigmas[:, 32:] = 3.0
        filtered = grain_filter(image, sigmas)
        self.assertTrue(np.isnan(filtered[:, :8]).all())
        self.assertTrue(np.allclose(filtered[:, 8:32], image[:, 8:32], atol=1e-5))
        self.assertLess(np.std(filtered[8:56, 40:56]), 0.2 * np.std(image[8:56, 40:56]))
        # Masked pixels must not pull values toward zero near the mask edge.
        self.assertAlmostEqual(float(np.nanmean(grain_filter(image, np.full((64, 64), 3.0))[:, 8:12])), 5.0, delta=0.3)

        radius = np.linspace(0, 0.999, 64)[None].repeat(64, 0)
        scale, noise = np.linspace(100.0, 2.5, 512), np.ones(512)
        mapped = sigma_map(radius, scale, noise)
        self.assertLess(mapped[0, 0], 0.5)
        self.assertAlmostEqual(float(mapped[0, -1]), 3.0, places=3)

    def test_noise2noise_views_swap_frames_and_share_mask(self):
        from kcor_ml.denoise.train import noise2noise_views

        images = torch.stack((torch.full((1, 4, 4), 1.0), torch.full((1, 4, 4), 2.0)), dim=1)
        valid = torch.ones(1, 2, 4, 4, dtype=torch.bool)
        valid[0, 1, 0, 0] = False
        batch = {"images": images, "valid_mask": valid, "radial_coordinate": torch.full((1, 2, 4, 4), 0.5)}
        scale = torch.full((512,), 2.0)
        noise = torch.full((512,), 0.5)
        inputs, targets, input_valid, weight, _ = noise2noise_views(
            batch, scale, noise, torch.device("cpu"), augment=False
        )
        self.assertTrue(torch.equal(inputs, torch.cat((images[:, 0:1], images[:, 1:2])) / 2.0))
        self.assertTrue(torch.equal(targets, torch.cat((images[:, 1:2], images[:, 0:1])) / 2.0))
        self.assertFalse(input_valid[1, 0, 0, 0])
        self.assertEqual(float(weight[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(weight[1, 0, 0, 0]), 0.0)
        # Errors in normalized units are converted to noise units: (2 / 0.5)^2.
        self.assertEqual(float(weight[0, 0, 1, 1]), 16.0)

    def test_noise2noise_learns_denoising_on_synthetic_pairs(self):
        from kcor_ml.denoise.model import DenoisingUNet

        torch.manual_seed(0)
        model = DenoisingUNet((16, 32), blocks_per_level=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
        clean = (torch.sin(4 * x) * torch.cos(3 * y)).expand(8, 1, -1, -1)
        valid = torch.ones_like(clean, dtype=torch.bool)
        radius = (x.square() + y.square()).sqrt().expand_as(clean)
        for _ in range(150):
            noisy_a = clean + 0.5 * torch.randn_like(clean)
            noisy_b = clean + 0.5 * torch.randn_like(clean)
            loss = (model(noisy_a, valid, radius) - noisy_b).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        noisy = clean + 0.5 * torch.randn_like(clean)
        with torch.no_grad():
            error = (model(noisy, valid, radius) - clean).square().mean()
        # Input noise variance is 0.25; denoising must cut it substantially.
        self.assertLess(float(error), 0.1)


if __name__ == "__main__":
    unittest.main()
