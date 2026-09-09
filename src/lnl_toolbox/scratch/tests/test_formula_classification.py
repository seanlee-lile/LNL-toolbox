from __future__ import annotations

import unittest

from lnl_toolbox.scratch.formula import formula_classification, formula_closure, get_formula, list_composite_formulas
from lnl_toolbox.scratch.registry import list_blocks


class FormulaClassificationTest(unittest.TestCase):
    def test_formula_operations_have_one_explicit_kind(self) -> None:
        allowed = {"primitive", "special", "composite"}
        rows = formula_classification()
        self.assertTrue(rows)
        self.assertEqual([row for row in rows if row["formula_kind"] == "UNKNOWN"], [])
        for definition in list_blocks():
            if definition.formula is not None:
                self.assertIn(definition.formula_kind, allowed, definition.id)

    def test_composite_registry_is_machine_readable(self) -> None:
        definitions = list_composite_formulas()
        self.assertGreaterEqual(len(definitions), 12)
        for definition in list_blocks():
            if definition.formula_kind == "composite":
                self.assertTrue(definition.formula_ref, definition.id)
                # Historical example formulas are definitions too; the new
                # composite directory covers canonical mathematical blocks.
                self.assertTrue(get_formula(definition.formula_ref), definition.id)

    def test_recursive_formula_closure_has_no_broken_composites(self) -> None:
        rows = formula_closure()
        self.assertTrue(rows)
        bad = [row for row in rows if row["status"] in {"BROKEN_COMPOSITE", "MISSING_DEFINITION"}]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
