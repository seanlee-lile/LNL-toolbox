from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml
import torch

from lnl_toolbox.training.experiment import run_experiment


ROOT = Path(__file__).resolve().parents[1]


class L2RWTrainingTest(unittest.TestCase):
    def test_step_budget_stops_exactly(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/experiment/l2rw_cifar10_smoke.yaml").read_text(encoding="utf-8")
        )
        config["trainer"] = {"epochs": 10, "max_steps": 2, "device": "cpu"}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(config, Path(directory) / "run")
            payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["global_step"], 2)

    def test_smoke_and_completed_resume(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/experiment/l2rw_cifar10_smoke.yaml").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(config, Path(directory) / "run")
            checkpoint = run_dir / "last.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((run_dir / "trusted_validation_manifest.npz").is_file())
            self.assertEqual(run_experiment(config, resume=checkpoint), run_dir)


if __name__ == "__main__": unittest.main()
