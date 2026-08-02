import json
import math
import re
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
            {
                "epoch": [1, 2],
                "validation_accuracy": [0.4, 0.8],
            },
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
                {
                    "epoch": [1, 2],
                    "validation_accuracy": [0.4, 0.8],
                },
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
                [{"epoch": 1, "validation_accuracy": math.nan}],
                {"epoch": [1], "validation_accuracy": [0.5]},
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("{invalid", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_metrics_jsonl(path)

    def test_epoch_contract_rejects_missing_duplicate_and_no_overlap(self):
        with self.assertRaisesRegex(KeyError, "missing epoch"):
            compare_curves(
                [{"validation_accuracy": 0.5}],
                {"epoch": [1], "validation_accuracy": [0.5]},
            )
        with self.assertRaisesRegex(ValueError, "duplicate epoch 1"):
            compare_curves(
                [
                    {"epoch": 1, "validation_accuracy": 0.5},
                    {"epoch": 1, "validation_accuracy": 0.6},
                ],
                {"epoch": [1], "validation_accuracy": [0.5]},
            )
        with self.assertRaisesRegex(ValueError, "finite integers"):
            compare_curves(
                [{"epoch": 1.0, "validation_accuracy": 0.5}],
                {"epoch": [1], "validation_accuracy": [0.5]},
            )
        with self.assertRaisesRegex(ValueError, "no overlapping epochs"):
            compare_curves(
                [{"epoch": 1, "validation_accuracy": 0.5}],
                {"epoch": [2], "validation_accuracy": [0.5]},
            )

    def test_out_of_order_partial_overlap_uses_max_common_epoch(self):
        summary = compare_curves(
            [
                {"epoch": 5, "validation_accuracy": 0.9},
                {"epoch": 1, "validation_accuracy": 0.1},
                {"epoch": 3, "validation_accuracy": 0.7},
            ],
            [
                {"epoch": 4, "validation_accuracy": 0.8},
                {"epoch": 3, "validation_accuracy": 0.6},
                {"epoch": 1, "validation_accuracy": 0.2},
            ],
        )
        self.assertEqual(summary["epochs"], [1, 3])
        self.assertEqual(summary["final_epoch"], 3)
        self.assertEqual(summary["final_reproduced"], 0.7)
        self.assertEqual(summary["final_paper"], 0.6)
        self.assertEqual(len(summary["differences"]), 2)
        self.assertAlmostEqual(summary["differences"][0], -0.1)
        self.assertAlmostEqual(summary["differences"][1], 0.1)

    def test_outputs_share_coordinates_and_are_byte_deterministic(self):
        ours = [
            {"epoch": 1, "validation_accuracy": 0.0},
            {"epoch": 3, "validation_accuracy": 10.0},
        ]
        paper = [
            {"epoch": 2, "validation_accuracy": 5.0},
            {"epoch": 3, "validation_accuracy": 10.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = write_curve_comparison(ours, paper, directory)
            first_bytes = {
                name: path.read_bytes()
                for name, path in first.items()
            }
            second = write_curve_comparison(ours, paper, directory)
            self.assertEqual(
                first_bytes,
                {
                    name: path.read_bytes()
                    for name, path in second.items()
                },
            )
            svg = first["overlay"].read_text(encoding="utf-8")
            lines = dict(re.findall(
                r'data-series="([^"]+)".*?points="([^"]+)"',
                svg,
            ))
            self.assertEqual(
                lines["reproduction"],
                "80.00,300.00 840.00,80.00",
            )
            self.assertEqual(
                lines["reference"],
                "460.00,190.00 840.00,80.00",
            )
            csv_text = first["difference"].read_text(
                encoding="utf-8",
            )
            self.assertIn("3,10.0,10.0,0.0", csv_text)


if __name__ == "__main__":
    unittest.main()
