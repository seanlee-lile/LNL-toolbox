from __future__ import annotations

import unittest

from lnl_toolbox.catalog import load_papers, load_recipe_config, recipe_by_id, validate_config
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

    def test_catalog_matches_runnable_complete_workflow(self):
        paper = next(value for value in load_papers() if value.id == "dividemix")
        self.assertEqual(paper.implementation_status, "user_ready")
        self.assertEqual(paper.availability, "runnable")


if __name__ == "__main__": unittest.main()
