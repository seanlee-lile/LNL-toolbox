from __future__ import annotations

from pathlib import Path
import unittest

import torch

from lnl_toolbox.scratch import ScratchContext, load_recipe, validate_recipe
from lnl_toolbox.scratch.blocks.forward import softmax
from lnl_toolbox.scratch.blocks.losses import (
    affine_transform,
    clamp_min,
    elementwise_power,
    gather_by_label,
)
from lnl_toolbox.scratch.blocks.selection import (
    linear_rate_schedule,
    mean_by_indices,
    select_by_indices,
    select_lowest_scores,
)
from lnl_toolbox.scratch.registry import BLOCKS


ROOT = Path(__file__).resolve().parents[1]


class ScratchBlockDedupTest(unittest.TestCase):
    def test_gather_by_label_is_pure_and_does_not_clamp(self) -> None:
        values = torch.tensor([[-1.0e-20, 0.5], [0.25, -2.0e-20]])
        labels = torch.tensor([0, 1])
        ctx = ScratchContext({"values": values, "labels": labels})
        gather_by_label(ctx, values="values", labels="labels", save_as="gathered")
        self.assertTrue(torch.equal(ctx["gathered"], torch.tensor([-1.0e-20, -2.0e-20])))

    def test_select_and_mean_are_distinct_operations(self) -> None:
        ctx = ScratchContext({"values": torch.tensor([1.0, 4.0, 9.0]), "indices": torch.tensor([2, 0])})
        select_by_indices(ctx, values="values", indices="indices", save_as="selected")
        mean_by_indices(ctx, values="values", indices="indices", save_as="mean")
        self.assertTrue(torch.equal(ctx["selected"], torch.tensor([9.0, 1.0])))
        self.assertEqual(float(ctx["mean"]), 5.0)
        self.assertNotEqual(ctx["selected"].ndim, 0)

    def test_lowest_scores_only_publishes_indices_and_honours_minimum_count(self) -> None:
        ctx = ScratchContext({
            "scores": torch.tensor([1.0, 1.0, 0.5, 4.0]),
            "fraction": 0.0,
            "sample_indices": torch.tensor([9, 3, 7, 8]),
        })
        select_lowest_scores(
            ctx, scores="scores", keep_fraction="fraction",
            stable_sample_indices="sample_indices", rounding="floor",
            minimum_count=0, save_as="selected",
        )
        self.assertEqual(ctx["selected"].numel(), 0)
        self.assertNotIn("selected_mask", ctx)
        ctx["fraction"] = 0.5
        select_lowest_scores(
            ctx, scores="scores", keep_fraction="fraction",
            stable_sample_indices="sample_indices", rounding="floor",
            minimum_count=1, save_as="selected_min_one",
        )
        self.assertEqual(ctx["selected_min_one"].tolist(), [2, 1])

    def test_gce_public_composition_matches_formula(self) -> None:
        logits = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, 0.0]], requires_grad=True)
        labels = torch.tensor([2, 1])
        ctx = ScratchContext({"logits": logits, "labels": labels})
        softmax(ctx)
        gather_by_label(ctx, values="probabilities", labels="labels", save_as="target_probability")
        clamp_min(ctx, input="target_probability", minimum=1.0e-12, save_as="clamped")
        elementwise_power(ctx, input="clamped", q=0.7, save_as="powered")
        affine_transform(ctx, input="powered", scale=-1.0 / 0.7, bias=1.0 / 0.7)
        expected = (1.0 - ctx["target_probability"].clamp_min(1.0e-12).pow(0.7)) / 0.7
        self.assertTrue(torch.allclose(ctx["loss_per_sample"], expected))
        ctx["loss_per_sample"].mean().backward()
        self.assertIsNotNone(logits.grad)

    def test_common_masked_and_weighted_cross_entropy(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]], requires_grad=True)
        labels = torch.tensor([0, 1, 0])
        ctx = ScratchContext({"logits": logits, "labels": labels, "mask": torch.tensor([True, False, True]), "pseudo": labels, "weights": torch.tensor([1.0, 0.5, 2.0])})
        BLOCKS["masked_cross_entropy"].execute(ctx, logits="logits", labels="labels", mask="mask", save_as="masked")
        BLOCKS["weighted_pseudo_label_cross_entropy"].execute(ctx, logits="logits", pseudo_labels="pseudo", weights="weights", save_as="weighted")
        expected_masked = torch.nn.functional.cross_entropy(logits[[0, 2]], labels[[0, 2]])
        expected_weighted = (torch.nn.functional.cross_entropy(logits, labels, reduction="none") * ctx["weights"]).mean()
        self.assertTrue(torch.allclose(ctx["masked"], expected_masked))
        self.assertTrue(torch.allclose(ctx["weighted"], expected_weighted))

    def test_linear_schedule_replaces_both_formal_rate_shapes(self) -> None:
        for epoch, expected in ((0, 1.0), (5, 0.75), (10, 0.5), (20, 0.5)):
            ctx = ScratchContext({"epoch": epoch})
            linear_rate_schedule(ctx, start=1.0, end=0.5, warmup_epochs=10)
            self.assertAlmostEqual(ctx["keep_rate"], expected)

    def test_deleted_duplicate_ids_are_not_registered(self) -> None:
        deleted = {
            "remember_" + "rate_formula", "jocor_" + "keep_rate_formula",
            "small_" + "loss_indices", "jocor_small_" + "loss_indices",
            "mean_selected_" + "loss", "cross_select_loss_a_" + "from_b",
            "cross_select_loss_b_" + "from_a", "gather_target_" + "probability",
            "softmax_" + "probability", "gce_" + "q_formula", "peer_" + "exchange",
            "track_best_" + "peer_models", "restore_best_" + "peer_models",
            "upm_" + "soft_target_loss",
            "small_loss",
            "l2rw_" + "initialize_epsilon", "l2rw_" + "virtual_weighted_loss",
            "t_revision_" + "weighted_objective",
            "jocor_" + "agreement", "jocor_" + "symmetric_kl",
            "volminnet_" + "objective", "cnlcu_" + "soft_influence",
        }
        self.assertTrue(deleted.isdisjoint(BLOCKS))

    def test_paper_recipes_use_common_ids(self) -> None:
        for name in ("gce.yaml", "coteaching.yaml", "cnlcu.yaml", "jocor.yaml", "t_revision.yaml"):
            recipe = load_recipe(ROOT / "recipes" / "papers" / name)
            validate_recipe(recipe)
            text = str(recipe)
            for old in ("remember_" + "rate_formula", "jocor_" + "keep_rate_formula", "small_" + "loss_indices", "jocor_small_" + "loss_indices", "gather_target_" + "probability", "softmax_" + "probability", "track_best_" + "peer_models", "restore_best_" + "peer_models"):
                self.assertNotIn(old, text, name)


if __name__ == "__main__":
    unittest.main()
