from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lnl_toolbox.training.results import discover_run_results, load_metric_history
from lnl_toolbox.training.sweep import plan_sweep, run_sweep, sweep_status


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
    @staticmethod
    def _matrix_config() -> dict:
        return {
            "seed": 0,
            "method": "fake",
            "noise": {"rate": 0.1},
            "optimizer": {"lr": 0.1},
        }

    def test_matrix_planning_is_cartesian_and_deterministic(self) -> None:
        base = self._matrix_config()
        single = plan_sweep(base, [1, 2], matrix={"noise.rate": [0.2, 0.4]})
        self.assertEqual(len(single.runs), 4)
        matrix = {
            "optimizer.lr": [0.1, 0.01, 0.001],
            "noise.rate": [0.2, 0.4],
        }
        first = plan_sweep(base, [1, 2], matrix=matrix)
        second = plan_sweep(base, [1, 2], matrix=dict(reversed(tuple(matrix.items()))))
        self.assertEqual(first.runs, second.runs)
        self.assertEqual(len(first.runs), 12)
        self.assertEqual(first.runs[0].seed, 1)
        self.assertEqual(first.runs[1].seed, 2)
        self.assertEqual(
            first.runs[0].overrides,
            (("noise.rate", 0.2), ("optimizer.lr", 0.1)),
        )

    def test_batch_lr_matrix_and_optional_seeds_have_expected_size(self) -> None:
        base = {
            **self._matrix_config(),
            "loader": {"batch_size": 128},
        }
        matrix = {
            "loader.batch_size": [256, 512],
            "optimizer.lr": [0.01, 0.001],
        }
        self.assertEqual(len(plan_sweep(base, [base["seed"]], matrix=matrix).runs), 4)
        self.assertEqual(len(plan_sweep(base, [1, 2, 3], matrix=matrix).runs), 12)

    def test_invalid_matrix_override_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            with self.assertRaisesRegex(ValueError, "Unknown config path"):
                run_sweep(
                    self._matrix_config(),
                    [1],
                    matrix={"optimizer.lrr": [0.1]},
                    output_dir=output,
                    service=_Service(),
                )
            self.assertFalse(output.exists())

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

    def test_matrix_manifest_identity_resume_and_status(self) -> None:
        service = _Service()
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.sweep.resolve_runner", return_value=_Runner()
        ):
            first = run_sweep(
                self._matrix_config(),
                [2, 13],
                matrix={"noise.rate": [0.2, 0.4]},
                output_dir=directory,
                service=service,
            )
            self.assertEqual(first.completed, 2)
            self.assertEqual(first.failed, 2)
            self.assertTrue(
                all("config_hash" in item and "overrides" in item for item in first.manifest["runs"])
            )
            status = sweep_status(directory)
            self.assertEqual(status["counts"]["completed"], 2)
            self.assertEqual(status["counts"]["failed"], 2)
            self.assertEqual(len(status["failed_runs"]), 2)
            self.assertEqual(len(status["runs"]), 4)
            self.assertIn("run_dir", status["runs"][0])

            second = run_sweep(
                self._matrix_config(),
                [2, 13],
                matrix={"noise.rate": [0.2, 0.4]},
                output_dir=directory,
                service=service,
            )
            self.assertEqual(second.skipped, 2)

    def test_result_discovery_reads_partial_history_and_reports_old_damage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "run-a"
            first.mkdir()
            (first / "metrics.jsonl").write_text(
                '\n'.join([
                    json.dumps({"epoch": 1, "validation_accuracy": 0.4}),
                    "not-json",
                    json.dumps({"epoch": 2, "validation_accuracy": 0.5}),
                    '{"epoch": 3',
                ]),
                encoding="utf-8",
            )
            rows, errors = load_metric_history(first)
            self.assertEqual([row["epoch"] for row in rows], [1, 2])
            self.assertEqual(len(errors), 1)
            discovered = discover_run_results(root)
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0]["status"], "incomplete")
            self.assertEqual(discovered[0]["current_epoch"], 2)
            self.assertIn("validation_accuracy", discovered[0]["metric_names"])


if __name__ == "__main__":
    unittest.main()
