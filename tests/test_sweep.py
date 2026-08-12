from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lnl_toolbox.training.sweep import run_sweep


class _Runner:
    supports_resume = True


class _Service:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def run(self, config, output_dir, resume=None, **kwargs):
        seed = int(config["seed"])
        self.calls.append(seed)
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        if seed == 13:
            raise RuntimeError("deliberate failure")
        (root / "final_metrics.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "completed": True,
                    "primary_metric": {"name": "test_accuracy", "value": 0.5},
                }
            ),
            encoding="utf-8",
        )
        return root


class SweepTest(unittest.TestCase):
    def test_sweep_is_sequential_resumable_and_failure_isolated(self) -> None:
        service = _Service()
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.sweep.resolve_runner", return_value=_Runner()
        ):
            first = run_sweep(
                {"method": "fake", "seed": 1},
                [2, 13, 3],
                output_dir=directory,
                service=service,
            )
            self.assertEqual(service.calls, [2, 13, 3])
            self.assertEqual(first.completed, 2)
            self.assertEqual(first.failed, 1)

            second = run_sweep(
                {"method": "fake", "seed": 1},
                [2, 13, 3],
                output_dir=directory,
                service=service,
            )
            self.assertEqual(service.calls, [2, 13, 3, 13])
            self.assertEqual(second.skipped, 2)

    def test_duplicate_seeds_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            run_sweep({}, [1, 1])


if __name__ == "__main__":
    unittest.main()
