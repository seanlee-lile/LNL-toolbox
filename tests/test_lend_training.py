from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from lnl_toolbox import toolbox
from lnl_toolbox.algorithms.lend import LENDConfig
from lnl_toolbox.catalog import validate_config
from lnl_toolbox.training.adapters import GraphStateRunner


class LENDTrainingTest(unittest.TestCase):
    def test_public_smoke_uses_complete_graph_history_config(self) -> None:
        config = yaml.safe_load(Path("configs/experiment/lend_cifar10_smoke.yaml").read_text())
        parsed = LENDConfig.from_mapping(config)
        self.assertEqual(parsed.epochs, 2)
        self.assertEqual(validate_config(config).name, "lend")

    def test_unified_interface_uses_graph_state_runner(self) -> None:
        self.assertIsInstance(toolbox.get("lend"), GraphStateRunner)


if __name__ == "__main__":
    unittest.main()
