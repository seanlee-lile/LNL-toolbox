from __future__ import annotations

import unittest

from lnl_toolbox.catalog import discover_recipes, load_recipe_config, recipe_by_id, validate_config
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner


class DivideMixCliTest(unittest.TestCase):
    def test_recipe_validates_and_epoch_override_only_changes_main(self):
        config = load_recipe_config(recipe_by_id("cifar10-dividemix-smoke"))
        self.assertEqual(resolve_runner(config).name, "dividemix")
        self.assertEqual(validate_config(config).name, "dividemix")
        warmup = config["dividemix"]["warmup"]["epochs"]
        apply_epoch_override(config, 7)
        self.assertEqual(config["dividemix"]["training"]["epochs"], 7)
        self.assertEqual(config["dividemix"]["warmup"]["epochs"], warmup)

    def test_formal_recipe_is_discoverable_and_epoch_override_is_main_only(self):
        recipe_ids = {item.id for item in discover_recipes()}
        self.assertIn("cifar10-dividemix-smoke", recipe_ids)
        self.assertIn("cifar10-dividemix-sym20", recipe_ids)
        config = load_recipe_config(recipe_by_id("cifar10-dividemix-sym20"))
        self.assertEqual(resolve_runner(config).name, "dividemix")
        self.assertEqual(validate_config(config).name, "dividemix")
        self.assertEqual(config["dividemix"]["warmup"]["epochs"], 10)
        self.assertEqual(config["dividemix"]["training"]["epochs"], 300)
        apply_epoch_override(config, 1)
        self.assertEqual(config["dividemix"]["warmup"]["epochs"], 10)
        self.assertEqual(config["dividemix"]["training"]["epochs"], 1)


if __name__ == "__main__": unittest.main()
