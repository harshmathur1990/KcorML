import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TFHDNTests(unittest.TestCase):
    def setUp(self):
        from kcor_ml.config import ModelConfig
        from kcor_ml.models import TwoFrameHeteroscedasticDecompositionNet

        config = ModelConfig(
            channels=(8, 16, 32),
            blocks_per_level=1,
            group_norm_groups=4,
            positional_grid_size=8,
        )
        self.model = TwoFrameHeteroscedasticDecompositionNet(config)

    def test_output_contract_and_gradients(self):
        images = torch.randn(2, 2, 64, 64) * 1.0e-8
        valid = torch.ones_like(images, dtype=torch.bool)
        output = self.model(images, valid_mask=valid)
        self.assertEqual(output.clean_pb.shape, images.shape)
        self.assertEqual(output.flat_corona.shape, images.shape)
        self.assertEqual(output.normalization_field.shape, images.shape)
        self.assertTrue(torch.all(output.normalization_field > 0))
        self.assertTrue(torch.all(output.noise_scale > 0))
        log_flat = output.auxiliary["log_flat_corona"]
        self.assertTrue(torch.allclose(log_flat.mean(dim=(-2, -1)), torch.zeros(2, 2), atol=1e-5))
        self.assertFalse(any(name == "scaler.log_scale" for name, _ in self.model.named_parameters()))
        output.observation_location.mean().backward()
        self.assertIsNotNone(self.model.encoder.stem.weight.grad)

    def test_invalid_input_shape_fails(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 1, 64, 64))

    def test_invalid_radial_coordinate_shape_fails(self):
        images = torch.randn(1, 2, 64, 64) * 1.0e-8
        with self.assertRaises(ValueError):
            self.model(images, radial_coordinate=torch.zeros(1, 1, 64, 64))

    def test_swapping_frames_swaps_frame_outputs(self):
        self.model.eval()
        images = torch.randn(1, 2, 64, 64) * 1.0e-8
        with torch.no_grad():
            original = self.model(images)
            swapped = self.model(images.flip(1))
        for name in (
            "clean_pb",
            "flat_corona",
            "normalization_field",
            "noise_location",
            "noise_scale",
            "noise_df",
            "cme_probability",
        ):
            self.assertTrue(
                torch.allclose(getattr(original, name).flip(1), getattr(swapped, name), atol=1e-6),
                name,
            )


if __name__ == "__main__":
    unittest.main()
