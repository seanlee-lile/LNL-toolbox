"""Focused operation and masked-chain checks for the six synchronized migrations."""

from __future__ import annotations

import unittest

import torch

import lnl_toolbox.scratch.blocks  # noqa: F401
from lnl_toolbox.scratch.context import ScratchContext
from lnl_toolbox.scratch.registry import BLOCKS


def run(block_id: str, ctx: ScratchContext, **params) -> ScratchContext:
    BLOCKS[block_id].execute(ctx, **params)
    return ctx


class ScratchOperationMigrationTest(unittest.TestCase):
    def test_generic_best_state_tracks_and_restores_two_modules(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        auxiliary = torch.nn.Linear(2, 2, bias=False)
        original_model = model.weight.detach().clone()
        original_aux = auxiliary.weight.detach().clone()
        ctx = ScratchContext(model=model, auxiliary=auxiliary, metric=0.5, epoch=3)
        run("track_best_state", ctx, model="model", auxiliary="auxiliary", metric="metric", metric_name="metric", mode="max", save_as="best")
        self.assertEqual(ctx["best_metric"], 0.5)
        with torch.no_grad():
            model.weight.add_(10.0)
            auxiliary.weight.sub_(10.0)
        run("restore_best_state", ctx, model="model", auxiliary="auxiliary", state="best")
        self.assertTrue(torch.equal(model.weight, original_model))
        self.assertTrue(torch.equal(auxiliary.weight, original_aux))

    def test_step_milestone_update_is_public_optimization_operation(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ctx = ScratchContext(optimizer=optimizer, global_step=9)
        run("step_milestone_update", ctx, optimizer="optimizer", milestones=[10], gamma=0.1, global_step="global_step")
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_weight_projection_and_normalization_are_public(self) -> None:
        ctx = ScratchContext(gradient=torch.tensor([-2.0, 1.0, 0.0]))
        run("nonnegative_projection", ctx, input="gradient", negate=False, save_as="raw")
        run("normalize_nonnegative_weights", ctx, weights="raw", save_as="weights")
        self.assertTrue(torch.equal(ctx["raw"], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(ctx["weights"], torch.tensor([0.0, 1.0, 0.0])))

    def test_transition_composition_is_public_and_row_stochastic(self) -> None:
        first = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        second = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        ctx = ScratchContext(first=first, second=second)
        run("compose_transition", ctx, first="first", second="second", save_as="composed")
        self.assertTrue(torch.allclose(ctx["composed"].sum(dim=1), torch.ones(2)))
        self.assertTrue(torch.all(ctx["composed"] >= 0))

    def test_transition_nll_is_explicitly_composable(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        labels = torch.tensor([0, 1])
        matrix = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        ctx = ScratchContext(logits=logits, labels=labels, matrix=matrix)
        run("softmax", ctx, logits="logits", save_as="probabilities")
        run("apply_transition", ctx, probabilities="probabilities", transition="matrix", save_as="noisy_probabilities")
        run("gather_by_label", ctx, values="noisy_probabilities", labels="labels", save_as="target_probability")
        run("negative_log", ctx, input="target_probability", save_as="losses")
        expected = -torch.log((torch.softmax(logits, -1) @ matrix).gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12))
        self.assertTrue(torch.allclose(ctx["losses"], expected))

    def test_l2rw_epsilon_and_virtual_loss_use_public_tensor_ops(self) -> None:
        losses = torch.tensor([2.0, 3.0], requires_grad=True)
        ctx = ScratchContext(losses=losses)
        run("zeros_like", ctx, input="losses", requires_grad=True, save_as="epsilon")
        run("elementwise_multiply", ctx, left="epsilon", right="losses", save_as="weighted")
        run("sum_values", ctx, input="weighted", save_as="virtual")
        self.assertEqual(tuple(ctx["epsilon"].shape), (2,))
        self.assertEqual(float(ctx["virtual"]), 0.0)
        self.assertTrue(ctx["epsilon"].requires_grad)

    def test_canonical_gather_select_and_reduction_are_separate(self) -> None:
        ctx = ScratchContext(values=torch.tensor([[1.0, 2.0], [3.0, 4.0]]), labels=torch.tensor([1, 0]), indices=torch.tensor([1]))
        run("gather_by_label", ctx, values="values", labels="labels", save_as="gathered")
        run("select_by_indices", ctx, values="gathered", indices="indices", save_as="selected")
        run("mean_by_indices", ctx, values="values", indices="indices", save_as="mean")
        self.assertTrue(torch.equal(ctx["gathered"], torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.equal(ctx["selected"], torch.tensor([3.0])))
        self.assertEqual(float(ctx["mean"]), 2.0)
        self.assertNotIn("selected_mask", BLOCKS["select_lowest_scores"].provides)

    def test_lowest_selection_minimum_count_is_explicit(self) -> None:
        ctx = ScratchContext(scores=torch.tensor([3.0, 1.0]), keep=torch.tensor(0.0), indices=torch.tensor([9, 4]))
        run("select_lowest_scores", ctx, scores="scores", keep_fraction="keep", stable_sample_indices="indices", rounding="floor", minimum_count=0, save_as="selected")
        self.assertEqual(ctx["selected"].numel(), 0)

    def test_dld_masked_chain_uses_public_operations(self) -> None:
        ctx = ScratchContext(
            labels=torch.tensor([0, 2]),
            predicted=torch.tensor([[1.0, 2.0], [3.0, 5.0]]),
            target=torch.tensor([[0.0, 1.0], [2.0, 4.0]]),
        )
        run("mean_squared_error", ctx, predicted="predicted", target="target", save_as="direction_loss")
        run("one_hot", ctx, labels="labels", num_classes=3, save_as="one_hot_labels")
        ctx["noise_loss"] = ctx["direction_loss"]
        run("weighted_sum", ctx, terms=["direction_loss", "noise_loss"], weights=[1.0, 1.0], save_as="objective")
        self.assertTrue(torch.isfinite(ctx["direction_loss"]))
        self.assertNotIn("dld_direction_loss", BLOCKS)
        self.assertNotIn("dld_objective_composition", BLOCKS)

    def test_dividemix_masked_chain(self) -> None:
        ctx = ScratchContext(probabilities=torch.tensor([[0.8, 0.2], [0.3, 0.7]]), clean_probability=torch.tensor([0.8, 0.7]), labels=torch.tensor([0, 1]))
        run("threshold_mask", ctx, values="clean_probability", threshold=0.5, comparison="ge", save_as="clean_mask")
        # The threshold operation is explicitly followed by the public mask/index conversion.
        run("mask_to_indices", ctx, mask="clean_mask", save_as="clean_indices")
        run("sharpen_distribution", ctx, input="probabilities", temperature=0.5, save_as="sharpened")
        self.assertEqual(ctx["clean_indices"].tolist(), [0, 1])
        self.assertEqual(tuple(ctx["sharpened"].shape), (2, 2))
        self.assertNotIn("dividemix_supervised_loss", BLOCKS)

    def test_pdl_masked_transition_chain(self) -> None:
        class Transition:
            def transition_for(self, _dataset, indices, device=None, dtype=None):
                matrix = torch.tensor([[0.8, 0.2], [0.1, 0.9]], device=device, dtype=dtype)
                return matrix.unsqueeze(0).expand(indices.numel(), -1, -1)

        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        ctx = ScratchContext(logits=logits, labels=torch.tensor([0, 1]), indices=torch.tensor([0, 1]), transition=Transition())
        run("transition_corrected_risk", ctx, logits="logits", labels="labels", indices="indices", transition="transition", save_as="risk")
        self.assertEqual(tuple(ctx["risk"].shape), (2,))
        self.assertTrue(torch.isfinite(ctx["risk"]).all())
        self.assertNotIn("pdl_corrected_loss", BLOCKS)
        self.assertNotIn("pdl_revision_loss", BLOCKS)

    def test_pdl_revision_is_composed_publicly(self) -> None:
        class Transition:
            def transition_for(self, _dataset, indices, device=None, dtype=None):
                return torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(indices.numel(), -1, -1)

        model = torch.nn.Linear(2, 2, bias=False)
        model.T_revision = torch.nn.Linear(2, 2, bias=False)
        model.T_revision.weight.data.zero_()
        ctx = ScratchContext(model=model, transition=Transition(), indices=torch.tensor([0, 1]))
        run("compose_revision_transition", ctx, transition="transition", model="model", indices="indices", save_as="revised")
        self.assertTrue(torch.allclose(ctx["revised"], torch.eye(2).expand(2, -1, -1)))

    def test_cal_masked_schedule_and_composition(self) -> None:
        ctx = ScratchContext(epoch=40, adjusted=torch.tensor(2.0), covariance=torch.tensor(0.5))
        run("piecewise_rate_schedule", ctx, epoch="epoch", default=1.0, milestones=[10, 40], values=[0.0, 1.0], save_as="confidence")
        run("weighted_sum", ctx, terms=["adjusted", "covariance"], weights=[1.0, -1.0], save_as="loss")
        self.assertEqual(ctx["confidence"], 1.0)
        self.assertEqual(float(ctx["loss"]), 1.5)
        self.assertNotIn("cal_confidence_schedule", BLOCKS)
        self.assertNotIn("subtract_objectives", BLOCKS)

    def test_lend_masked_graph_state_chain(self) -> None:
        ctx = ScratchContext(features=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), labels=torch.tensor([0, 1, 0]), indices=torch.tensor([0, 1, 2]), epoch=0)
        run("pairwise_similarity", ctx, features="features", metric="cosine", save_as="similarity")
        run("topk_neighborhood", ctx, similarity="similarity", k=1, save_as="adjacency")
        run("normalize_graph", ctx, adjacency="adjacency", save_as="graph")
        run("propagate_labels", ctx, graph="graph", labels="labels", num_classes=2, steps=2, save_as="diluted")
        run("create_indexed_history", ctx, size=3, width=2, save_as="history")
        run("indexed_ema", ctx, state="history", indices="indices", values="diluted", save_as="history_values")
        # The core public graph operations are executable without any lend_* symbol.
        self.assertEqual(tuple(ctx["diluted"].shape), (3, 2))
        self.assertFalse(any(key.startswith("lend_") for key in BLOCKS))

    def test_cdr_masked_parameter_chain(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        model.weight.data.copy_(torch.tensor([[2.0, 1.0], [1.0, 3.0]]))
        model.weight.grad = torch.tensor([[1.0, 0.5], [0.25, 0.1]])
        ctx = ScratchContext(model=model)
        run("parameter_criticality_mask", ctx, model="model", noise_rate=0.5, save_as="masks")
        before = model.weight.grad.clone()
        run("masked_gradient_update", ctx, model="model", masks="masks", scale=0.5, l1_decay=0.0)
        self.assertTrue((model.weight.grad.abs() <= before.abs()).all())
        self.assertNotIn("cdr_criticality_score", BLOCKS)
        self.assertNotIn("cdr_masked_gradient_update", BLOCKS)


if __name__ == "__main__":
    unittest.main()
