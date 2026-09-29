import json
from pathlib import Path
import tempfile
import unittest

from kcor_ml.config import ExperimentConfig


class ExperimentConfigTests(unittest.TestCase):
    def test_default_config_round_trip(self):
        config = ExperimentConfig()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config.save(path)
            restored = ExperimentConfig.load(path)
        self.assertEqual(restored.model.channels, config.model.channels)
        self.assertEqual(restored.data.expected_product, "pb2")

    def test_repository_default_config_loads(self):
        config = ExperimentConfig.load("configs/default.json")
        self.assertEqual(config.model.name, "tf_hdn_v21")
        self.assertLessEqual(config.data.max_delta_seconds, 15)

    def test_invalid_split_is_rejected(self):
        payload = ExperimentConfig().to_dict()
        payload["data"]["train_fraction"] = 0.95
        payload["data"]["validation_fraction"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                ExperimentConfig.load(path)


if __name__ == "__main__":
    unittest.main()
