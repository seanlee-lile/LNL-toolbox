from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from lnl_toolbox.training.lend_experiment import run_lend_experiment


class LENDTrainingTest(unittest.TestCase):
    def test_smoke_and_resume(self) -> None:
        config = yaml.safe_load(Path("configs/experiment/lend_cifar10_smoke.yaml").read_text())
        with tempfile.TemporaryDirectory() as directory:
            run = run_lend_experiment(config, output_dir=directory)
            self.assertTrue((run / "last.pt").is_file())
            run_lend_experiment(config, resume=run / "last.pt")


if __name__ == "__main__":
    unittest.main()
