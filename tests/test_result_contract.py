from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from lnl_toolbox.training.results import finalize_result, is_completed_result


class ResultContractTest(unittest.TestCase):
    def test_finalizer_preserves_runner_metrics_and_adds_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "seed-3"
            root.mkdir()
            (root / "final_metrics.json").write_text(
                json.dumps({"test_accuracy": 0.75, "best_epoch": 4}),
                encoding="utf-8",
            )
            result = finalize_result(
                root,
                {
                    "seed": 3,
                    "method": "gce",
                    "evaluation": {"primary": "accuracy", "selection_split": "validation"},
                },
                runner="supervised",
                recipe="example",
            )
            self.assertEqual(result["primary_metric"], {"name": "test_accuracy", "value": 0.75})
            self.assertEqual(result["metrics"]["test_accuracy"], 0.75)
            self.assertEqual(result["selection"]["best_epoch"], 4)
            self.assertFalse(result["test_selection_leakage"])
            self.assertTrue(is_completed_result(root))

    def test_test_selection_is_explicitly_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text(
                json.dumps({"test_accuracy": 0.5}) + "\n", encoding="utf-8"
            )
            result = finalize_result(
                root,
                {"evaluation": {"selection_split": "test"}},
                runner="supervised",
            )
            self.assertTrue(result["test_selection_leakage"])


if __name__ == "__main__":
    unittest.main()
