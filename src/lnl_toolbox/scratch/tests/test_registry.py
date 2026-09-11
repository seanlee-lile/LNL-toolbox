from __future__ import annotations

import unittest

from lnl_toolbox.scratch.registry import BlockDefinition, describe_block, get_block, list_blocks, register_block


class ScratchRegistryTest(unittest.TestCase):
    def test_builtin_metadata_is_available(self) -> None:
        definition = get_block("set_value")
        self.assertEqual(definition.kind, "action")
        self.assertIn("save_as", definition.params)
        self.assertEqual(describe_block("set_value")["category"], "Runtime")
        self.assertIn("repeat_n", {value.id for value in list_blocks("Control")})

    def test_backward_exposes_model_annotation_for_dual_model_recipes(self) -> None:
        definition = get_block("backward")
        self.assertIn("loss", definition.params)
        self.assertIn("model", definition.params)
        self.assertIn("explicitly named model", definition.description)

    def test_duplicate_id_is_rejected(self) -> None:
        existing = get_block("set_value")
        duplicate = BlockDefinition(
            existing.id,
            existing.name,
            existing.category,
            existing.description,
            existing.kind,
            existing.params,
            existing.requires,
            existing.provides,
            existing.execute,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            register_block(duplicate)


if __name__ == "__main__":
    unittest.main()
