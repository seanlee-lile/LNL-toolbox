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

    def test_legacy_catalog_recipes_execute_synthetic_path(self) -> None:
        for path in sorted((ROOT / "recipes" / "papers").glob("*.yaml")):
            if path.name in {"gce.yaml", "coteaching.yaml", "apl.yaml", "binary_risk.yaml"}:
                # These are formal CIFAR reproduction recipes. Their
                # execution intentionally requires the user's local dataset
                # and is covered by static validation plus runtime-limited
                # execution tests.
                continue
            recipe = load_recipe(path)
            context = execute_recipe(recipe, runtime_limits={"fixture": True})
            self.assertIn("labels", context, path.name)


if __name__ == "__main__":
    unittest.main()
