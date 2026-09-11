from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from lnl_toolbox.scratch import load_recipe, save_recipe


class ScratchRecipeRoundTripTest(unittest.TestCase):
    def test_yaml_round_trip_preserves_recipe(self) -> None:
        recipe = {
            "schema_version": 1,
            "name": "roundtrip",
            "description": "中文说明",
            "steps": [{"block": "set_value", "params": {"value": 3, "save_as": "answer"}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.yaml"
            save_recipe(recipe, path)
            self.assertEqual(load_recipe(path), recipe)


if __name__ == "__main__":
    unittest.main()
