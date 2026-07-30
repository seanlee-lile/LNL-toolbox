import json
import math
import tempfile
import unittest
from pathlib import Path

from lnl_toolbox.evaluation.curve_comparison import (
    compare_curves,
    load_metrics_jsonl,
    write_curve_comparison,
)


class CurveComparisonTest(unittest.TestCase):
    def test_compare_and_write_outputs(self) -> None:
        rows = [
            {
                "event": "epoch",
                "epoch": 1,
                "validation_accuracy": 0.5,
            },
            {
                "event": "epoch",
                "epoch": 2,
                "validation_accuracy": 0.7,
            },
        ]
        summary = compare_curves(
            rows,
            {"validation_accuracy": [0.4, 0.8]},
        )
        self.assertEqual(summary["overlap_epochs"], 2)
        self.assertAlmostEqual(summary["mean_absolute_error"], 0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(len(load_metrics_jsonl(path)), 2)
            outputs = write_curve_comparison(
                rows,
                {"validation_accuracy": [0.4, 0.8]},
                directory,
            )
            self.assertTrue(all(
                path.is_file() for path in outputs.values()
            ))
            self.assertEqual(
                json.loads(outputs["summary"].read_text("utf-8"))[
                    "overlap_epochs"
                ],
                2,
            )

    def test_invalid_curve_values_and_json_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_curves(
                [{"validation_accuracy": math.nan}],
                {"validation_accuracy": [0.5]},
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("{invalid", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_metrics_jsonl(path)


if __name__ == "__main__":
    unittest.main()
