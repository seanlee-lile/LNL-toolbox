from __future__ import annotations

from pathlib import Path
import unittest

from lnl_toolbox.scratch import load_recipe, validate_recipe


PAPERS = Path(__file__).parents[1] / "recipes" / "papers"
PARTICLES = {
    "load_dataset", "inspect_dataset_semantics", "create_dataset_split",
    "select_label_source", "apply_noise", "build_noise_manifest",
    "configure_preprocessing", "configure_views", "assign_data_roles",
    "configure_loader", "build_prepared_data", "build_loaders",
}


class DataRecipeProtocolTest(unittest.TestCase):
    def test_all_paper_data_entries_are_particle_sequences(self) -> None:
        for path in PAPERS.glob("*.yaml"):
            recipe = load_recipe(path)
            validate_recipe(recipe)
            ids = [str(step["block"]) for step in recipe["steps"]]
            self.assertFalse(any(block.startswith("prepare_") for block in ids), path.name)
            self.assertTrue(PARTICLES.issubset(ids), path.name)

    def test_zero_validation_recipes_have_no_validation_role_or_epoch_eval(self) -> None:
        for name in ("cal.yaml", "fine.yaml"):
            recipe = load_recipe(PAPERS / name)
            data_step = next(step for step in recipe["steps"] if step["block"] == "create_dataset_split")
            role_step = next(step for step in recipe["steps"] if step["block"] == "assign_data_roles")
            self.assertEqual(data_step["params"]["validation_size"], 0)
            self.assertNotIn("clean_validation", role_step["params"]["roles"])
            self.assertNotIn("noisy_validation", role_step["params"]["roles"])
            epoch = next(step for step in recipe["steps"] if step["block"] == "epoch_loop")
            self.assertFalse(any(
                step.get("block") == "evaluate_accuracy" and
                step.get("params", {}).get("loader") == "validation_loader"
                for step in epoch["steps"]
            ))
            final = recipe["steps"][-1]
            self.assertEqual(final["block"], "evaluate_accuracy")
            self.assertEqual(final["params"]["loader"], "test_loader")
            self.assertTrue(final["params"].get("final"))


if __name__ == "__main__":
    unittest.main()
