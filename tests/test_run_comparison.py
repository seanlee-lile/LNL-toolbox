from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from lnl_toolbox.evaluation.run_comparison import compare_runs, write_report


class RunComparisonTest(unittest.TestCase):
    def _run(
        self,
        root: Path,
        seed: int,
        accuracy: float,
        model: str,
        *,
        method: str = "gce",
        rate: float = 0.4,
        manifest: str | None = None,
        metric: str = "test_accuracy",
        leakage: bool = False,
    ) -> Path:
        run = root / f"{method}-seed-{seed}-rate-{rate}-{metric}"
        run.mkdir()
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "seed": seed,
                    "execution": {"runner": "supervised"},
                    "data": {"name": "cifar10", "augment": True},
                    "model": {"name": model},
                    "trainer": {"epochs": 10},
                    "loss": {"name": method},
                    "noise": {"name": "symmetric", "rate": rate},
                    "evaluation": {"primary": metric},
                }
            ),
            encoding="utf-8",
        )
        (run / "final_metrics.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "method": method,
                    "primary_metric": {"name": metric, "value": accuracy},
                    "selection": {"split": "validation"},
                    "test_selection_leakage": leakage,
                    "metrics": {"noise": {"mapping_hash": manifest}},
                }
            ),
            encoding="utf-8",
        )
        return run

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

    def test_methods_compare_without_warning_and_noise_rate_is_research_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1, 0.7, "resnet18", method="ce", rate=0.2, manifest="same")
            self._run(root, 1, 0.8, "resnet18", method="gce", rate=0.2, manifest="same")
            self._run(root, 1, 0.6, "resnet18", method="ce", rate=0.4, manifest="other")
            summary = compare_runs(
                root,
                group_by=("method", "noise.rate"),
                require_equal=("dataset", "model", "noise.rate"),
            )
            self.assertEqual(len(summary["summaries"]), 3)
            self.assertFalse(any("noise.rate differs" in item for item in summary["warnings"]))
            self.assertFalse(summary["compatibility_findings"])

    def test_manifest_is_compared_across_methods_only_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1, 0.7, "resnet18", method="ce", manifest="a")
            self._run(root, 1, 0.8, "resnet18", method="gce", manifest="b")
            self._run(root, 2, 0.75, "resnet18", method="ce", manifest="c")
            self._run(root, 2, 0.85, "resnet18", method="gce", manifest="c")
            summary = compare_runs(root)
            manifest_findings = [
                item for item in summary["compatibility_findings"]
                if item["field"] == "noise_manifest"
            ]
            self.assertEqual(len(manifest_findings), 1)
            self.assertEqual(manifest_findings[0]["group"]["seed"], 1)

    def test_different_metrics_never_share_an_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1, 0.7, "resnet18", metric="test_accuracy")
            self._run(root, 2, 0.6, "resnet18", metric="test_f1")
            summary = compare_runs(root, group_by=("method", "noise.rate"))
            self.assertEqual(len(summary["summaries"]), 2)
            self.assertEqual(
                {item["metric"] for item in summary["summaries"]},
                {"test_accuracy", "test_f1"},
            )

    def test_strict_leakage_excludes_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaked = self._run(root, 1, 0.9, "resnet18", leakage=True)
            normal = compare_runs(root)
            self.assertEqual(normal["summaries"][0]["n"], 1)
            strict = compare_runs(root, strict=True)
            self.assertFalse(strict["summaries"])
            self.assertEqual(strict["excluded_runs"][0]["run_dir"], str(leaked))


if __name__ == "__main__":
    unittest.main()
