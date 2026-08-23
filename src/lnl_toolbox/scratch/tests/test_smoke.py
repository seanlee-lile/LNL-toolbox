from __future__ import annotations

from pathlib import Path
import unittest

import torch

from lnl_toolbox.scratch import ScratchContext, execute_recipe, load_recipe, validate_recipe


ROOT = Path(__file__).resolve().parents[1]


class ScratchSmokeTest(unittest.TestCase):
    def test_ce_and_gce_recipes_execute_one_epoch(self) -> None:
        for filename in ("ce_synthetic.yaml", "gce_small_loss_synthetic.yaml"):
            recipe = load_recipe(ROOT / "recipes" / "examples" / filename)
            validate_recipe(recipe)
            context = execute_recipe(recipe)
            self.assertIn("model", context)
            self.assertTrue(context["loss"].ndim == 0)

    def test_peer_exchange_uses_string_slots(self) -> None:
        recipe = {
            "schema_version": 1,
            "name": "peer",
            "steps": [{
                "block": "peer_exchange",
                "params": {"loss_a": "a", "loss_b": "b", "keep_rate": 0.5},
            }],
        }
        context = execute_recipe(recipe, ScratchContext({"a": torch.tensor([3.0, 1.0, 2.0, 4.0]), "b": torch.tensor([1.0, 4.0, 2.0, 3.0])}))
        self.assertEqual(context["selected_a"].tolist(), [1, 2])
        self.assertEqual(context["selected_b"].tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
