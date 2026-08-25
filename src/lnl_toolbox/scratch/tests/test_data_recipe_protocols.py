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
        giant = {"prepare_gce_cifar10", "prepare_cdr_cifar10", "prepare_dual_t_cifar10",
                 "prepare_pdl_cifar10", "prepare_volminnet_cifar10", "prepare_t_revision_cifar10",
                 "prepare_cwd_cifar10", "prepare_loss_correction_cifar10", "prepare_jocor_cifar10",
                 "prepare_apl_cifar10", "prepare_binary_risk_data", "prepare_importance_reweighting_binary",
                 "prepare_coteaching_cifar10", "prepare_cnlcu_cifar10", "prepare_formal_cifar",
                 "prepare_pcse_cifar10", "prepare_cal_cifar10"}
        for path in PAPERS.glob("*.yaml"):
            recipe = load_recipe(path)
            validate_recipe(recipe)
            ids = [str(step["block"]) for step in recipe["steps"]]
            self.assertFalse(giant.intersection(ids), path.name)
            self.assertTrue(PARTICLES.issubset(ids), path.name)


if __name__ == "__main__":
    unittest.main()
