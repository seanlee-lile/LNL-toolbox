import unittest

from lnl_toolbox.scratch import BLOCKS, describe_block, get_block
from lnl_toolbox.scratch.registry import list_blocks
from lnl_toolbox.scratch.formula import FormulaValidationError, get_formula, register_formula, unregister_formula


def _formula(formula_id="user/registry_formula"):
    return {
        "id": formula_id,
        "name": "Registry Formula",
        "inputs": {"x": {}},
        "steps": [{"id": "out", "block": "detach", "bindings": {"input": "x"}}],
        "outputs": {"value": {"source": "out"}},
    }


class FormulaRegistryTest(unittest.TestCase):
    def tearDown(self):
        for formula_id in ("user/registry_formula", "user/nested_formula", "user/cycle_a", "user/cycle_b"):
            try:
                unregister_formula(formula_id)
            except KeyError:
                pass

    def test_formula_becomes_normal_block(self):
        register_formula(_formula())
        self.assertIs(get_block("formula/user/registry_formula"), BLOCKS["formula__user__registry_formula"])
        metadata = describe_block("formula/user/registry_formula")
        self.assertEqual(metadata["formula_kind"], "composite")
        self.assertIn("save_as", metadata["provides"])

    def test_nested_formula_is_allowed(self):
        register_formula(_formula())
        register_formula({
            "id": "user/nested_formula",
            "name": "Nested Formula",
            "inputs": {"x": {}},
            "steps": [{"id": "out", "block": "formula/user/registry_formula", "bindings": {"x": "x"}}],
            "outputs": {"value": {"source": "out"}},
        })
        self.assertEqual(get_formula("user/nested_formula").name, "Nested Formula")

    def test_builtin_formula_count(self):
        self.assertGreaterEqual(len([item for item in BLOCKS if item.startswith("formula__builtin__")]), 5)

    def test_formula_blocks_have_explicit_kinds_and_no_legacy_flag(self):
        allowed = {"primitive", "special", "composite"}
        for block in list_blocks():
            if block.formula:
                self.assertIn(block.formula_kind, allowed, block.id)
            self.assertNotIn("formula" + "_safe", block.describe())

    def test_nested_cycle_is_rejected(self):
        register_formula(_formula("user/cycle_a"))
        register_formula({
            "id": "user/cycle_b", "name": "Cycle B", "inputs": {"x": {}},
            "steps": [{"id": "out", "block": "formula/user/cycle_a", "bindings": {"x": "x"}}],
            "outputs": {"value": {"source": "out"}},
        })
        with self.assertRaises(FormulaValidationError):
            register_formula({
                "id": "user/cycle_a", "name": "Cycle A", "inputs": {"x": {}},
                "steps": [{"id": "out", "block": "formula/user/cycle_b", "bindings": {"x": "x"}}],
                "outputs": {"value": {"source": "out"}},
            }, replace=True)
