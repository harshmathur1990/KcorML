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


if __name__ == "__main__":
    unittest.main()
