from __future__ import annotations

import json
from pathlib import Path
import unittest

from lnl_toolbox.scratch import execute_recipe, load_recipe, validate_recipe


ROOT = Path(__file__).resolve().parents[1]


class ScratchCatalogRecipeTest(unittest.TestCase):
    def test_every_catalog_method_has_one_valid_recipe(self) -> None:
        catalog = json.loads((ROOT.parent / "paper_catalog.json").read_text(encoding="utf-8"))
        expected = {item["id"] for item in catalog}
        recipes = {}
        for path in (ROOT / "recipes" / "papers").glob("*.yaml"):
            recipe = load_recipe(path)
            validate_recipe(recipe)
            recipes[recipe["name"]] = path
        self.assertEqual(set(recipes), expected)
        self.assertEqual(len(recipes), 26)

    def test_every_catalog_recipe_executes_synthetic_path(self) -> None:
        for path in sorted((ROOT / "recipes" / "papers").glob("*.yaml")):
            recipe = load_recipe(path)
            context = execute_recipe(recipe)
            self.assertIn("labels", context, path.name)


if __name__ == "__main__":
    unittest.main()
