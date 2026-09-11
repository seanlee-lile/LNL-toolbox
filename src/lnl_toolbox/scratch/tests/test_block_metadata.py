from __future__ import annotations

import unittest

from lnl_toolbox.scratch.registry import list_blocks


class ScratchBlockMetadataTest(unittest.TestCase):
    def test_ids_and_placement_are_valid(self) -> None:
        blocks = list_blocks()
        ids = [definition.id for definition in blocks]
        self.assertEqual(len(ids), len(set(ids)))
        for definition in blocks:
            self.assertTrue(definition.placement)
            self.assertTrue(set(definition.placement) <= {"top", "epoch", "batch", "any"})
            self.assertIn(definition.stage, {"data", "setup", "train", "evaluate"})

    def test_beginner_metadata_is_ui_usable(self) -> None:
        for definition in list_blocks():
            if definition.beginner_visible:
                self.assertTrue(definition.ui_group, definition.id)
            if definition.paper is not None:
                self.assertTrue(definition.formula, definition.id)
                self.assertTrue(definition.formula_ref, definition.id)

    def test_describe_contains_v2_fields(self) -> None:
        for definition in list_blocks():
            payload = definition.describe()
            for key in (
                "placement",
                "stage",
                "ui_group",
                "formula",
                "formula_ref",
                "paper",
                "beginner_visible",
            ):
                self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
