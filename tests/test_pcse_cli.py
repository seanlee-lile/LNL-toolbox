from pathlib import Path
import os
import unittest
from unittest import mock

from lnl_toolbox.algorithms.pcse import PCSEConfig
from lnl_toolbox.catalog import (
    discover_recipes,
    load_recipe_config,
    paper_by_id,
    recipe_by_id,
    validate_config,
)


class PCSECliTest(unittest.TestCase):
    def test_real_cifar_recipe_is_registered_and_valid(self) -> None:
        recipe = recipe_by_id("cifar10-pcse-reproduction")
        self.assertEqual(recipe.profile, "reproduction")
        self.assertEqual(recipe.runner, "pcse")
        self.assertEqual(recipe.configuration_fidelity, "engineering")
        self.assertEqual(recipe.availability, "conditional")
        config = load_recipe_config(recipe)
        parsed = PCSEConfig.from_mapping(config)
        self.assertEqual(parsed.pretraining.mode, "external_checkpoint")
        self.assertEqual(parsed.pretraining.source["adapter"], "upm_main_best")
        self.assertEqual(config["data"]["name"], "cifar10")
        self.assertNotIn("max_train_samples", config["data"])
        self.assertEqual(
            [(item.name, item.pooling) for item in parsed.feature_layers],
            [("layer3", "global_average"), ("layer4", "global_average")],
        )
        self.assertEqual(parsed.transition_backend, "paper_volmin")

    def test_real_cifar_recipe_is_hidden_without_conditional_flag(self) -> None:
        public_ids = {item.id for item in discover_recipes()}
        all_ids = {
            item.id for item in discover_recipes(include_conditional=True)
        }
        self.assertNotIn("cifar10-pcse-reproduction", public_ids)
        self.assertIn("cifar10-pcse-reproduction", all_ids)

    def test_real_cifar_preflight_rejects_missing_source_environment(self) -> None:
        config = load_recipe_config(recipe_by_id("cifar10-pcse-reproduction"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LNL_PCSE_SOURCE_RUN", None)
            with self.assertRaisesRegex(
                ValueError, "source environment variable is not set"
            ):
                validate_config(config)

    def test_paper_catalog_does_not_claim_numerical_reproduction(self) -> None:
        paper = paper_by_id("pcse")
        recipe_ids = {item.recipe_id for item in paper.configs}
        self.assertIn("cifar10-pcse-reproduction", recipe_ids)
        self.assertEqual(paper.reproduction_status, "not_run")
        real_config = next(
            item
            for item in paper.configs
            if item.recipe_id == "cifar10-pcse-reproduction"
        )
        self.assertEqual(real_config.availability, "conditional")


if __name__ == "__main__":
    unittest.main()
