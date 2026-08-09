from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from lnl_toolbox import toolbox
from lnl_toolbox.algorithms.upm import UPMConfig
from lnl_toolbox.catalog import validate_config
from lnl_toolbox.training.adapters import AlternatingRunner


class UPMTrainingTest(unittest.TestCase):
    def test_public_smoke_uses_complete_two_stage_config(self) -> None:
        config = yaml.safe_load(Path("configs/experiment/upm_cifar10_smoke.yaml").read_text())
        parsed = UPMConfig.from_mapping(config)
        self.assertEqual(parsed.stage1.epochs, 2)
        self.assertEqual(parsed.main.epochs, 2)
        self.assertEqual(validate_config(config).name, "upm")

    def test_unified_interface_uses_alternating_runner(self) -> None:
        self.assertIsInstance(toolbox.get("upm"), AlternatingRunner)


if __name__ == "__main__":
    unittest.main()
