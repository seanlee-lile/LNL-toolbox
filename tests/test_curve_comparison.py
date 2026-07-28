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
        rows = [{"event": "epoch", "epoch": 1, "validation_accuracy": 0.5},
                {"event": "epoch", "epoch": 2, "validation_accuracy": 0.7}]
        summary = compare_curves(rows, {"validation_accuracy": [0.4, 0.8]})
        self.assertEqual(summary["overlap_epochs"], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("\n".join(__import__("json").dumps(row) for row in rows), encoding="utf-8")
            self.assertEqual(len(load_metrics_jsonl(path)), 2)
            outputs = write_curve_comparison(rows, {"validation_accuracy": [0.4, 0.8]}, directory)
            self.assertTrue(all(path.is_file() for path in outputs.values()))


if __name__ == "__main__":
    unittest.main()
