from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from lnl_toolbox import toolbox
from lnl_toolbox.training.interfaces import RunResult
from lnl_toolbox.training.dld_experiment import run_dld_experiment


class DLDTrainingTest(unittest.TestCase):
    def test_legacy_smoke_and_resume_remains_compatible(self) -> None:
        config = yaml.safe_load(Path("configs/experiment/dld_cifar10_smoke.yaml").read_text())
        with tempfile.TemporaryDirectory() as directory:
            run = run_dld_experiment(config, output_dir=directory)
            self.assertTrue((run / "last.pt").is_file())
            run_dld_experiment(config, resume=run / "last.pt")

    def test_unified_smoke_and_completed_resume(self) -> None:
        config = yaml.safe_load(Path("configs/experiment/dld_cifar10_smoke.yaml").read_text())
        with tempfile.TemporaryDirectory() as directory:
            result = toolbox.run("dld", config=config, output_dir=directory)
            self.assertIsInstance(result, RunResult)
            self.assertTrue((result.run_dir / "last.pt").is_file())
            self.assertTrue((result.run_dir / "report.json").is_file())
            payload = torch.load(result.run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertIn("checkpoint_v3", payload)
            resumed = toolbox.run("dld", config=config, resume=result.run_dir / "last.pt")
            self.assertEqual(resumed.run_dir, result.run_dir)


if __name__ == "__main__":
    unittest.main()
