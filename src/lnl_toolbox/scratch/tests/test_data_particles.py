from __future__ import annotations

import unittest

from lnl_toolbox.scratch import ScratchContext, execute_recipe, validate_recipe


def _recipe() -> dict:
    return {
        "schema_version": 1,
        "name": "scratch-data-particles",
        "steps": [
            {"block": "load_dataset", "params": {"dataset": "synthetic", "options": {"samples": 8, "features": 3, "classes": 2}}},
            {"block": "inspect_dataset_semantics"},
            {"block": "create_dataset_split", "params": {"validation_size": 2, "split_seed": 7}},
            {"block": "select_label_source", "params": {"train": "observed", "validation": "clean", "test": "clean"}},
            {"block": "apply_noise", "params": {"name": "symmetric", "rate": 0.2, "seed": 11}},
            {"block": "build_noise_manifest"},
            {"block": "configure_preprocessing", "params": {"preprocessing": "gce2018", "augment": True}},
            {"block": "configure_views", "params": {"views": ["weak", "strong"]}},
            {"block": "assign_data_roles", "params": {"roles": ["train", "clean_validation", "test"]}},
            {"block": "configure_loader", "params": {"batch_size": 2, "num_workers": 0}},
            {"block": "build_prepared_data"},
            {"block": "build_loaders"},
        ],
    }


class DataParticleTest(unittest.TestCase):
    def test_composition_keeps_formal_slots_explicit(self) -> None:
        recipe = _recipe()
        validate_recipe(recipe)
        context = execute_recipe(recipe, runtime_limits={"fixture": True})
        plan = context["data_plan"]
        self.assertEqual(plan["data"]["name"], "synthetic")
        self.assertEqual(plan["split"]["validation_size"], 2)
        self.assertEqual(plan["labels"]["train"], "observed")
        self.assertEqual(plan["noise"]["rate"], 0.2)
        self.assertEqual(plan["views"], ["weak", "strong"])
        self.assertEqual(plan["roles"], ["train", "clean_validation", "test"])
        self.assertIn("train_loader", context)
        self.assertIn("validation_loader", context)
        self.assertIn("test_loader", context)

    def test_clean_training_label_is_rejected(self) -> None:
        recipe = _recipe()
        recipe["steps"][3] = {"block": "select_label_source", "params": {"train": "clean"}}
        with self.assertRaisesRegex(Exception, "clean labels"):
            execute_recipe(recipe, runtime_limits={"fixture": True})

    def test_particles_are_scratch_native(self) -> None:
        from pathlib import Path

        source = Path(__file__).parents[1].joinpath("blocks", "data.py").read_text(encoding="utf-8")
        self.assertNotIn("from lnl_toolbox.training.data_service", source)
        self.assertNotIn("from lnl_toolbox.data import", source)


if __name__ == "__main__":
    unittest.main()
