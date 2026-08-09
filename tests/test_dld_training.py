from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from lnl_toolbox.algorithms.dld import DLDConfig
from lnl_toolbox.catalog import validate_config


class DLDTrainingTest(unittest.TestCase):
    def test_public_smoke_uses_complete_paper_oriented_config(self) -> None:
        config = yaml.safe_load(Path("configs/experiment/dld_cifar10_smoke.yaml").read_text())
        parsed = DLDConfig.from_mapping(config)
        self.assertEqual(parsed.diffusion["epochs"], 2)
        self.assertEqual(validate_config(config).name, "dld")


if __name__ == "__main__":
    unittest.main()
