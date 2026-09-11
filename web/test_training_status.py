from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from web.training_status import TrainingContext, infer_training_context, training_snapshot


class TrainingStatusTest(unittest.TestCase):
    def _context(self, directory: Path) -> TrainingContext:
        return TrainingContext(directory, None, "run", 12)

    def test_regular_epoch_uses_flushed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "metrics.jsonl").write_text(json.dumps({"event": "epoch", "epoch": 2}) + "\n", encoding="utf-8")
            snapshot = training_snapshot(self._context(path), lines=[], running=True, returncode=None, cancel_requested=False)
        assert snapshot is not None
        self.assertEqual((snapshot.source, snapshot.stage, snapshot.completed_epoch), ("metrics", "main", 2))

    def test_ca2c_phase_is_used_as_display_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "metrics.jsonl").write_text(
                json.dumps({"event": "epoch", "epoch": 1, "phase": "warmup"}) + "\n",
                encoding="utf-8",
            )
            snapshot = training_snapshot(self._context(path), lines=[], running=True, returncode=None, cancel_requested=False)
        self.assertEqual(snapshot.stage, "warmup")

    def test_dividemix_warmup_then_main_uses_latest_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            rows = [{"event": "warmup", "epoch": 10}, {"event": "epoch", "epoch": 1}]
            (path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            snapshot = training_snapshot(self._context(path), lines=[], running=True, returncode=None, cancel_requested=False)
        assert snapshot is not None
        self.assertEqual((snapshot.stage, snapshot.completed_epoch), ("main", 1))

    def test_partial_json_line_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "metrics.jsonl").write_text('{"event":"warmup","epoch":1}\n{"event":', encoding="utf-8")
            snapshot = training_snapshot(self._context(path), lines=[], running=True, returncode=None, cancel_requested=False)
        assert snapshot is not None
        self.assertEqual(snapshot.completed_epoch, 1)

    def test_stdout_structured_event_is_a_fallback(self) -> None:
        snapshot = training_snapshot(
            TrainingContext(None, None, "run", 3),
            lines=['{"event":"epoch","epoch":2}'],
            running=False,
            returncode=0,
            cancel_requested=False,
        )
        assert snapshot is not None
        self.assertEqual((snapshot.source, snapshot.state, snapshot.completed_epoch), ("stdout", "completed", 2))

    def test_positional_config_and_explicit_output_are_inferred(self) -> None:
        context = infer_training_context(
            ["lnl", "run", "configs/experiment/cifar10_clean_smoke.yaml", "--output-dir", "artifacts/web"],
            Path.cwd(),
        )
        assert context is not None
        self.assertEqual(context.run_dir, Path.cwd() / "artifacts" / "web")
        self.assertEqual(context.total_epochs, 2)


if __name__ == "__main__":
    unittest.main()
