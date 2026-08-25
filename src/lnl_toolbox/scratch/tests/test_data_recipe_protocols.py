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


if __name__ == "__main__":
    unittest.main()
