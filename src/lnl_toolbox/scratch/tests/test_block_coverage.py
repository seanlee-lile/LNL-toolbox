from __future__ import annotations

import csv
from pathlib import Path
import unittest

from lnl_toolbox.scratch.registry import BLOCKS


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT.parents[2] / "artifacts" / "scratch_block_audit"


class ScratchBlockCoverageTest(unittest.TestCase):
    def test_audit_artifacts_exist_and_have_required_columns(self) -> None:
        inventory = (AUDIT / "block_inventory.tsv").read_text(encoding="utf-8").splitlines()
        coverage = (AUDIT / "block_coverage_matrix.tsv").read_text(encoding="utf-8").splitlines()
        self.assertIn("block_id", inventory[0])
        self.assertIn("operation", inventory[0])
        self.assertIn("status", coverage[0])
        self.assertGreater(len(inventory), 200)

    def test_inventory_semantics_are_audited_and_recipe_refs_are_recursive(self) -> None:
        with (AUDIT / "block_inventory.tsv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertGreater(len(rows), 200)
        for row in rows:
            for field in ("state_mutation", "detach", "reduction", "rounding", "stable_tiebreak"):
                self.assertTrue(row[field], f"missing audit value for {row['block_id']}:{field}")
                self.assertNotEqual("unknown", row[field], f"placeholder audit value for {row['block_id']}:{field}")
        by_id = {row["block_id"]: row for row in rows}
        # These are all nested under epoch/batch loop steps; a top-level-only
        # scanner would leave the references empty.
        self.assertIn("recipes/papers/coteaching.yaml", by_id["select_lowest_scores"]["recipe_refs"])
        self.assertIn("recipes/papers/cnlcu.yaml", by_id["select_lowest_scores"]["recipe_refs"])
        self.assertIn("recipes/papers/gce.yaml", by_id["gather_by_label"]["recipe_refs"])
        self.assertIn("recipes/papers/jocor.yaml", by_id["linear_rate_schedule"]["recipe_refs"])
        self.assertIn(".steps[", by_id["select_lowest_scores"]["recipe_refs"])

    def test_lend_masking_is_strict_fail_not_legacy_success(self) -> None:
        report = (AUDIT / "block_dedup_report.md").read_text(encoding="utf-8")
        self.assertIn("LEND masking", report)
        self.assertIn("PAPER_SPECIFIC_ONLY", report)
        self.assertIn("strict masking test must report FAIL", report)
        self.assertTrue({"lend_build_neighbor_graph", "lend_normalize_neighbor_graph", "lend_dilute_labels"}.issubset(BLOCKS))

    def test_canonical_palette_has_separate_contracts(self) -> None:
        self.assertEqual(BLOCKS["gather_by_label"].requires, ("values", "labels"))
        self.assertEqual(BLOCKS["select_by_indices"].provides, ("save_as",))
        self.assertEqual(BLOCKS["mean_by_indices"].provides, ("save_as",))
        self.assertEqual(BLOCKS["select_lowest_scores"].provides, ("save_as",))
        self.assertNotIn("selected_mask", BLOCKS["select_lowest_scores"].provides)

    def test_scratch_ui_has_no_selector_id_whitelist(self) -> None:
        source = (ROOT / "web" / "scratch.js").read_text(encoding="utf-8")
        self.assertNotIn("small_loss_indices", source)
        self.assertNotIn("select_lowest_scores", source)


if __name__ == "__main__":
    unittest.main()
