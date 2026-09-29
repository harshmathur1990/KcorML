import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class LossStabilityTests(unittest.TestCase):
    def test_disabled_cme_loss_does_not_reduce_probability_map(self):
        from kcor_ml.config import LossConfig
        from kcor_ml.losses.composite import CompositeLoss
        from kcor_ml.models.outputs import TFHDNOutput

        shape = (1, 2, 512, 512)
        zeros = torch.zeros(shape)
        features = torch.zeros(1, 4, 8, 8)
        output = TFHDNOutput(
            clean_pb=zeros,
            flat_corona=torch.ones(shape),
            normalization_field=torch.ones(shape),
            noise_location=zeros,
            noise_scale=torch.full(shape, 1.0e-8),
            noise_df=torch.full(shape, 4.0),
            cme_probability=torch.ones(shape, dtype=torch.float16),
            auxiliary={
                "common": features,
                "dynamic_1": features,
                "dynamic_2": features,
                "noise_1": features,
                "noise_2": features,
                "normalization_scaled": torch.ones(shape),
                "log_normalization": torch.zeros(shape),
                "log_flat_corona": torch.zeros(shape),
            },
        )
        loss = CompositeLoss(LossConfig(cme_weight=0.0))(
            output, zeros, torch.ones(shape, dtype=torch.bool)
        )
        self.assertTrue(torch.isfinite(loss.total))
        self.assertEqual(float(loss.cme), 0.0)


if __name__ == "__main__":
    unittest.main()
