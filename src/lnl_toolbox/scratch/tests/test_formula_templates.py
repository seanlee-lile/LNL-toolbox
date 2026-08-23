from __future__ import annotations

from pathlib import Path
import unittest

from lnl_toolbox.scratch import execute_recipe, load_recipe, validate_recipe
from lnl_toolbox.scratch.registry import get_block


ROOT = Path(__file__).resolve().parents[1]


def _batch_blocks(recipe: dict) -> list[str]:
    epoch = next(step for step in recipe["steps"] if step["block"] == "epoch_loop")
    batch = next(step for step in epoch["steps"] if step["block"] == "batch_loop")
    return [step["block"] for step in batch["steps"]]


class ScratchFormulaTemplateTest(unittest.TestCase):
    def test_gce_is_expanded_to_formula_blocks(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "gce.yaml")
        validate_recipe(recipe)
        epoch = next(step for step in recipe["steps"] if step["block"] == "epoch_loop")
        self.assertEqual(epoch["params"]["epochs"], 120)
        self.assertEqual(
            _batch_blocks(recipe),
            [
                "get_batch",
                "move_batch_to_device",
                "zero_grad",
                "forward",
                "softmax_probability",
                "gather_target_probability",
                "gce_q_formula",
                "mean_loss",
                "backward",
                "optimizer_step",
            ],
        )
        self.assertNotIn("gce_loss", _batch_blocks(recipe))

    def test_coteaching_exposes_peer_selection_and_cross_update(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "coteaching.yaml")
        validate_recipe(recipe)
        epoch = next(step for step in recipe["steps"] if step["block"] == "epoch_loop")
        self.assertEqual(epoch["steps"][0]["block"], "remember_rate_formula")
        self.assertEqual(
            _batch_blocks(recipe)[7:],
            [
                "small_loss_indices",
                "small_loss_indices",
                "cross_select_loss_a_from_b",
                "cross_select_loss_b_from_a",
                "backward",
                "optimizer_step",
                "backward",
                "optimizer_step",
            ],
        )

    def test_formula_metadata_is_complete(self) -> None:
        for block_id in (
            "softmax_probability",
            "gather_target_probability",
            "gce_q_formula",
            "remember_rate_formula",
            "small_loss_indices",
            "cross_select_loss_a_from_b",
            "cross_select_loss_b_from_a",
        ):
            definition = get_block(block_id)
            self.assertTrue(definition.formula, block_id)
            self.assertTrue(definition.formula_ref, block_id)
            self.assertTrue(definition.paper, block_id)

    def test_formula_templates_execute_smoke(self) -> None:
        for name, context_key in (
            (ROOT / "recipes" / "examples" / "gce_formula_smoke.yaml", "model"),
            (ROOT / "recipes" / "papers" / "coteaching.yaml", "model_a"),
        ):
            recipe = load_recipe(name)
            context = execute_recipe(recipe)
            self.assertIn(context_key, context)


if __name__ == "__main__":
    unittest.main()
