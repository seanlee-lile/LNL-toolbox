from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from lnl_toolbox.evaluation.run_comparison import compare_runs, write_report


class RunComparisonTest(unittest.TestCase):
    def _run(self, root: Path, seed: int, accuracy: float, model: str) -> None:
        run = root / f"seed-{seed}"
        run.mkdir()
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "seed": seed,
                    "execution": {"runner": "supervised"},
                    "data": {"name": "cifar10"},
                    "model": {"name": model},
                    "trainer": {"epochs": 10},
                    "loss": {"name": "gce"},
                    "noise": {"name": "symmetric", "rate": 0.4},
                }
            ),
            encoding="utf-8",
        )
        (run / "final_metrics.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "method": "gce",
                    "primary_metric": {"name": "test_accuracy", "value": accuracy},
                    "selection": {"split": "validation"},
                }
            ),
            encoding="utf-8",
        )

    def test_statistics_warnings_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1, 0.7, "resnet18")
            self._run(root, 2, 0.9, "resnet34")
            summary = compare_runs(root)
            row = summary["summaries"][0]
            self.assertEqual(row["n"], 2)
            self.assertAlmostEqual(row["mean"], 0.8)
            self.assertTrue(any("model" in warning for warning in summary["warnings"]))
            paths = write_report(summary, root / "report")
            self.assertTrue(all(path.is_file() for path in paths.values()))
            with paths["csv"].open(encoding="utf-8") as handle:
                self.assertEqual(next(csv.DictReader(handle))["method"], "gce")


if __name__ == "__main__":
    unittest.main()
