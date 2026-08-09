from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from lnl_toolbox.training.adapters import NativeSingleStageRunner, NativeStagedRunner


class NativeRunnerLifecycleTest(unittest.TestCase):
    def test_cwd_five_fold_completed_checkpoint_is_strict_noop(self) -> None:
        spec = SimpleNamespace(name="cwd")
        config = {
            "cwd": {"protocol": "five_fold"},
            "trainer": {"epochs": 200},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            torch.save(
                {
                    "protocol_state": {
                        "protocol": "five_fold",
                        "completed": True,
                        "completed_folds": [0, 1, 2, 3, 4],
                    }
                },
                checkpoint,
            )
            self.assertTrue(
                NativeStagedRunner._is_completed_noop(spec, config, checkpoint)
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["protocol_state"]["completed"] = False
            torch.save(payload, checkpoint)
            self.assertFalse(
                NativeStagedRunner._is_completed_noop(spec, config, checkpoint)
            )

    def test_native_runner_passes_context_and_writes_common_lifecycle_events(self) -> None:
        received: list[object] = []

        def execute(config, output_dir, resume, *, context):
            received.append(context)
            context.session.start_run()
            context.session.start_phase("train", total_units=1)
            context.session.log_epoch(1, loss=0.5)
            context.session.end_phase("train", completed_units=1)
            context.session.emit("final", phase="evaluation", accuracy=0.75)
            (Path(output_dir) / "final_metrics.json").write_text(
                json.dumps({"accuracy": 0.75}), encoding="utf-8"
            )
            return Path(output_dir)

        spec = SimpleNamespace(
            name="supervised",
            supports_resume=True,
            lifecycle="single_stage",
            checkpoint_unit="epoch",
            load=lambda: execute,
        )
        runner = NativeSingleStageRunner(spec, method="gce")
        with tempfile.TemporaryDirectory() as directory:
            result = runner.fit(
                config={"method": "gce", "trainer": {"epochs": 1}},
                output_dir=directory,
            )
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].run_dir, Path(directory).resolve())
            events = received[0].events
            self.assertEqual([event["event"] for event in events], [
                "run_start", "phase_start", "epoch", "phase_end", "final"
            ])
            self.assertEqual(events[2]["unit"], "epoch")
            self.assertEqual(events[2]["completed"], 1)
            self.assertEqual(result.final_metrics["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
