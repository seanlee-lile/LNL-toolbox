from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from lnl_toolbox.scratch.formula import (
    composite_parameter_coverage,
    special_audit,
    write_formula_audit,
)


class FormulaAuditReportTest(unittest.TestCase):
    def test_special_audit_reads_callable_bodies_and_has_no_unknown_decisions(self) -> None:
        rows = special_audit()
        self.assertGreaterEqual(len(rows), 40)
        self.assertTrue(all(row["current_kind"] == "special" for row in rows))
        self.assertTrue(all(row["keep_or_change"] in {"KEEP_SPECIAL", "CHANGE_TO_COMPOSITE"} for row in rows))
        self.assertEqual([], [row for row in rows if row["keep_or_change"] == "CHANGE_TO_COMPOSITE"])
        self.assertTrue(all(len(row["body_sha256"]) == 64 for row in rows))
        # Warm-up is ordinary CE and must no longer be registered as a special
        # operation after the canonical composite migration.
        self.assertNotIn("fine_warmup_loss", {row["block_id"] for row in rows})
        converted = {
            "binary_risk", "cal_cores2_adjusted_risk", "cnlcu_soft_score",
            "one_hot_like", "t_revision_importance_ratio",
        }
        self.assertTrue(converted.isdisjoint({row["block_id"] for row in rows}))

    def test_special_and_branch_audits_use_executable_evidence(self) -> None:
        rows = special_audit()
        self.assertEqual([], [row for row in rows if row["keep_or_change"] != "KEEP_SPECIAL"])
        composite_root = Path(__file__).resolve().parents[1] / "formula" / "composites"
        yaml_text = "\n".join(path.read_text(encoding="utf-8") for path in composite_root.glob("*.yaml"))
        # Keep the forbidden marker split so a repository scan cannot mistake
        # this assertion itself for a live branch annotation.
        self.assertNotIn("oracle_" + "branches", yaml_text)
        coverage = composite_parameter_coverage()
        self.assertTrue(any(row["branch_status"] == "EXECUTABLE_FORMULA_VARIANT" for row in coverage))
        legacy_branch_status = "DOCUMENTED_YAML_" + "ORACLE_BRANCH"
        self.assertEqual([], [row for row in coverage if row["branch_status"] == legacy_branch_status])

    def test_composite_parameter_coverage_has_executable_branch_records(self) -> None:
        rows = composite_parameter_coverage()
        self.assertGreaterEqual(len(rows), 15)
        self.assertEqual([], [row for row in rows if row["status"] == "FAIL"])
        controls = {row["block_id"]: row for row in rows if row["branch_status"] == "EXECUTABLE_FORMULA_VARIANT"}
        self.assertTrue({"mean_squared_error", "soft_target_cross_entropy", "masked_mean", "weighted_blend"}.issubset(controls))
        for row in rows:
            self.assertIn(row["status"], {"PASS", "FAIL"})
            self.assertIn("missing_parameters", row)
            if row["branch_status"] == "EXECUTABLE_FORMULA_VARIANT":
                self.assertTrue(row["branch_documentation"])

    def test_write_formula_audit_emits_requested_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_formula_audit(directory)
            destination = Path(directory)
            self.assertTrue(all(path.exists() for path in paths))
            for name in ("special_audit.tsv", "composite_parameter_coverage.tsv"):
                report = destination / name
                self.assertTrue(report.exists())
                with report.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                self.assertTrue(rows, name)


if __name__ == "__main__":
    unittest.main()
