import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SeparationLossTests(unittest.TestCase):
    def test_radial_loss_detects_radial_trend_but_not_angular_structure(self):
        from kcor_ml.losses.separation import radial_flatness_loss

        size = 64
        axis = torch.linspace(-1.0, 1.0, size)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        radius = (x.square() + y.square()).sqrt()
        angle_only = x / radius.clamp_min(1.0e-3)
        mask = torch.ones(1, 2, size, size, dtype=torch.bool)
        radial = radius.expand(1, 2, -1, -1)
        angular = angle_only.expand(1, 2, -1, -1)

        radial_loss = radial_flatness_loss(radial, mask, radial_bins=16)
        angular_loss = radial_flatness_loss(angular, mask, radial_bins=16)
        self.assertGreater(float(radial_loss), 100.0 * float(angular_loss) + 1.0e-5)

    def test_noise_structure_loss_detects_copied_morphology(self):
        from kcor_ml.losses.separation import noise_structure_correlation_loss

        torch.manual_seed(4)
        pattern = torch.randn(1, 2, 32, 32)
        mask = torch.ones_like(pattern, dtype=torch.bool)
        copied = noise_structure_correlation_loss(pattern, pattern, mask)
        smooth = noise_structure_correlation_loss(pattern, torch.zeros_like(pattern), mask)
        self.assertGreater(float(copied), 0.99)
        self.assertEqual(float(smooth), 0.0)

    def test_degenerate_losses_have_finite_gradients(self):
        from kcor_ml.losses.separation import (
            common_dynamic_consistency_loss,
            noise_structure_correlation_loss,
        )

        # A noise scale clamped to its minimum is spatially constant; this
        # produced NaN gradients that poisoned DDP weights during training.
        flat = torch.randn(1, 2, 16, 16, requires_grad=True)
        noise = torch.zeros(1, 2, 16, 16, requires_grad=True)
        mask = torch.ones(1, 2, 16, 16, dtype=torch.bool)
        noise_structure_correlation_loss(flat, noise, mask).backward()
        self.assertTrue(torch.isfinite(flat.grad).all())
        self.assertTrue(torch.isfinite(noise.grad).all())

        dynamic = torch.zeros(2, 4, 8, 8, requires_grad=True)
        common_dynamic_consistency_loss(dynamic, torch.randn(2, 4, 8, 8)).backward()
        self.assertTrue(torch.isfinite(dynamic.grad).all())

    def test_radial_derivative_loss_is_resolution_invariant(self):
        from kcor_ml.losses.separation import radial_flatness_loss

        def loss_at(size):
            axis = torch.linspace(-1.0, 1.0, size)
            y, x = torch.meshgrid(axis, axis, indexing="ij")
            radius = (x.square() + y.square()).sqrt()
            radius = radius / radius.max()
            values = (2.0 * radius).expand(1, 2, -1, -1)
            coordinates = radius.expand_as(values)
            mask = ((radius > 0.1) & (radius < 0.9)).expand_as(values)
            return radial_flatness_loss(values, mask, 16, coordinates)

        low_resolution = loss_at(64)
        high_resolution = loss_at(128)
        self.assertTrue(
            torch.allclose(low_resolution, high_resolution, rtol=0.03, atol=0.03),
            (float(low_resolution), float(high_resolution)),
        )


if __name__ == "__main__":
    unittest.main()
