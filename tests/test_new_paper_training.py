from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from lnl_toolbox.training.experiment import run_experiment


ROOT = Path(__file__).resolve().parents[1]


class NewPaperTrainingTest(unittest.TestCase):
    def _config(self, name: str) -> dict:
        return yaml.safe_load(
            (ROOT / "configs" / "experiment" / name).read_text(encoding="utf-8")
        )

    def test_all_new_workflows_smoke_and_resume(self) -> None:
        names = (
            "mc_ldce_cifar10_smoke.yaml",
            "cal_cifar10_smoke.yaml",
            "ca2c_cifar10_smoke.yaml",
        )
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                with self.subTest(name=name):
                    config = deepcopy(self._config(name))
                    run_dir = run_experiment(config, Path(directory) / name)
                    checkpoint = run_dir / "last.pt"
                    self.assertTrue(checkpoint.is_file())
                    self.assertTrue((run_dir / "metrics.jsonl").is_file())
                    self.assertEqual(run_experiment(config, resume=checkpoint), run_dir)


if __name__ == "__main__":
    unittest.main()
