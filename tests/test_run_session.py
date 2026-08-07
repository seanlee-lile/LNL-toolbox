from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from lnl_toolbox.training.checkpoint import (
    atomic_save,
    read_v3_checkpoint,
    save_v3_checkpoint,
    upgrade_checkpoint_to_v3,
)
from lnl_toolbox.training.reporting import RunSession, load_metric_events, write_toolbox_report


class RunSessionTest(unittest.TestCase):
    def test_lifecycle_events_and_report_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            session = RunSession(
                run_dir,
                config={"method": "demo", "seed": 7, "execution": {"runner": "demo"}},
                runner="demo",
                method="demo",
            )
            session.start_run()
            session.start_phase("train", total_units=2)
            session.log_epoch(1, train_loss=1.0, validation_accuracy=0.5)
            session.end_phase("train", completed_units=1)
            paths = session.finish_run(test_accuracy=0.5)

            events = load_metric_events(run_dir / "metrics.jsonl")
            self.assertEqual([row["seq"] for row in events], list(range(len(events))))
            self.assertTrue(paths["json"].is_file())
            report = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(report["identity"]["method"], "demo")
            self.assertEqual(report["paper_fidelity_status"], "not_audited")

    def test_toolbox_report_aggregates_run_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "one"
            RunSession(first, config={"method": "one"}, runner="one", method="one").finish_run()
            output = root / "summary"
            paths = write_toolbox_report(root, output)
            value = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(value["run_count"], 1)

    def test_v3_checkpoint_round_trip_preserves_identity_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_v3_checkpoint(
                path,
                identity={"runner": "demo", "method": "demo", "seed": 3},
                progress={"phase": "train", "completed_epoch": 2, "global_step": 8},
                component_states={"model": {"weight": torch.ones(1)}},
                config={"method": "demo"},
                rng_state={"torch": torch.get_rng_state()},
                log_sequence=11,
            )
            payload = read_v3_checkpoint(path)
            envelope = payload["checkpoint"]
            self.assertEqual(payload["format_version"], 3)
            self.assertEqual(envelope["identity"]["method"], "demo")
            self.assertEqual(envelope["progress"]["completed_epoch"], 2)
            self.assertEqual(envelope["log_sequence"], 11)

    def test_recovery_truncates_events_ahead_of_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            session = RunSession(run_dir, config={"method": "demo"}, method="demo")
            session.start_run()
            session.log_epoch(1, train_loss=1.0)
            checkpoint = session.save_checkpoint({"model": {"weight": torch.ones(1)}}, run_dir / "last.pt")
            session.log_epoch(2, train_loss=0.5)
            self.assertEqual(session.recover_metrics_from_checkpoint(checkpoint), 2)
            self.assertEqual([row["epoch"] for row in load_metric_events(run_dir / "metrics.jsonl") if "epoch" in row], [1])

    def test_legacy_checkpoint_upgrade_retains_top_level_reader_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_save({"format_version": 1, "model": {"weight": torch.ones(1)}}, path)
            self.assertTrue(
                upgrade_checkpoint_to_v3(
                    path,
                    config={"method": "demo", "seed": 1},
                    runner="demo",
                    method="demo",
                )
            )
            payload = read_v3_checkpoint(path)
            self.assertEqual(payload["format_version"], 1)
            self.assertIn("checkpoint_v3", payload)


if __name__ == "__main__":
    unittest.main()
