from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from lnl_toolbox.training.reporting import load_metric_events, write_run_report


class ReportingTest(unittest.TestCase):
    def test_existing_flat_metric_rows_remain_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"epoch": 1, "train_loss": 1.0, "validation_accuracy": 0.2})
                + "\n"
                + json.dumps({"event": "final", "test_accuracy": 0.2})
                + "\n",
                encoding="utf-8",
            )
            write_run_report(run_dir, config={"method": "legacy"}, runner="legacy", method="legacy")
            events = load_metric_events(run_dir / "metrics.jsonl")
            self.assertEqual(events[0]["event"], "epoch")
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["events"]["epoch_count"], 1)

    def test_failed_report_records_error_without_claiming_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            write_run_report(
                run_dir,
                config={"method": "broken"},
                status="failed",
                error="intentional test failure",
            )
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["identity"]["status"], "failed")
            self.assertEqual(report["final_metrics"]["status"], "failed")
            self.assertTrue((run_dir / "final_metrics.json").is_file())

    def test_standard_epoch_run_writes_environment_config_and_curves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({
                    "event": "epoch", "epoch": 1, "train_loss": 1.0,
                    "validation_loss": 0.9, "train_accuracy": 0.4,
                    "validation_accuracy": 0.5, "learning_rate": 0.1,
                })
                + "\n",
                encoding="utf-8",
            )
            write_run_report(run_dir, config={"method": "demo"}, status="completed")
            self.assertTrue((run_dir / "resolved_config.yaml").is_file())
            self.assertTrue((run_dir / "environment.json").is_file())
            self.assertTrue((run_dir / "training_curves.svg").is_file())
            artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
            names = {item["path"] for item in artifacts["artifacts"]}
            self.assertIn("training_curves.svg", names)


if __name__ == "__main__":
    unittest.main()
