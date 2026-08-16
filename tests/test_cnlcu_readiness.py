from pathlib import Path
import unittest

import yaml

from lnl_toolbox.algorithms.cnlcu import CNLCUConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/reproduction/cifar10_cnlcu_soft_sym20_short.yaml"


class CNLCUReadinessTest(unittest.TestCase):
    def test_short_config_is_full_data_cnlcu_soft_sym20(self) -> None:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        method = CNLCUConfig.from_mapping(config)
        self.assertEqual(config["method"], "cnlcu")
        self.assertEqual(config["execution"]["runner"], "cnlcu")
        self.assertEqual(config["configuration_fidelity"], "engineering")
        self.assertEqual(config["data"]["name"], "cifar10")
        for key in (
            "max_train_samples", "max_validation_samples", "max_test_samples"
        ):
            self.assertNotIn(key, config["data"])
        self.assertEqual(config["noise"]["name"], "symmetric")
        self.assertEqual(config["noise"]["rate"], 0.2)
        self.assertEqual(config["noise"]["validation_targets"], "noisy")
        self.assertEqual(config["model"]["name"], "cifar_cnn8")
        self.assertEqual(config["trainer"]["epochs"], 15)
        self.assertEqual(method.variant, "soft")
        self.assertEqual(method.window_size, 5)
        self.assertEqual(method.rate_at(0), 1.0)
        self.assertEqual(method.rate_at(10), 0.8)


if __name__ == "__main__":
    unittest.main()
