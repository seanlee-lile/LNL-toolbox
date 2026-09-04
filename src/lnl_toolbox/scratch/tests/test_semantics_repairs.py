"""Regression tests for the paper-semantics repairs in Scratch recipes."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

import lnl_toolbox.scratch.blocks  # noqa: F401
from lnl_toolbox.scratch import ScratchContext, load_recipe
from lnl_toolbox.scratch.native_stats import (
    CALProxyArtifact,
    FINERegularizer,
    PartTransitionEstimator,
    PosteriorSnapshot,
    FeatureSnapshot,
    cnlcu_soft_score,
)
from lnl_toolbox.scratch.registry import BLOCKS


ROOT = Path(__file__).resolve().parents[1]


def _walk(steps):
    for step in steps:
        yield step
        yield from _walk(step.get("steps", []))


def _run(block_id: str, context: ScratchContext, **params) -> ScratchContext:
    BLOCKS[block_id].execute(context, **params)
    return context


class SemanticsRepairTest(unittest.TestCase):
    def test_l2rw_recipe_uses_paper_meta_gradient_and_objective(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "l2rw.yaml")
        blocks = list(_walk(recipe["steps"]))
        self.assertNotIn("parameter_squared_norm", [step["block"] for step in blocks])
        projection = next(step for step in blocks if step["block"] == "nonnegative_projection")
        self.assertTrue(projection["params"]["negate"])
        final_sum = [step for step in blocks if step["block"] == "sum_values" and step.get("params", {}).get("save_as") == "loss"]
        self.assertTrue(final_sum)

    def test_apl_recipe_has_paper_rce_constant_and_no_gradient_clip(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "apl.yaml")
        blocks = list(_walk(recipe["steps"]))
        rce = next(step for step in blocks if step["block"] == "rce_loss")
        self.assertEqual(rce["params"]["log_zero"], -4.0)
        self.assertNotIn("clip_grad_norm", [step["block"] for step in blocks])

    def test_fine_rejected_regularizer_uses_seeded_complementary_label(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], requires_grad=True)
        observed = torch.tensor([0, 1])
        pseudo = torch.tensor([1, 0])  # Eq. (3)/(4) must not use this input.
        seed = 17
        regularizer = FINERegularizer(beta=0.001, gamma=0.1, probability_floor=1e-7, seed=seed)
        actual = regularizer(logits, observed, rejected_mask=torch.ones(2, dtype=torch.bool), pseudo_labels=pseudo)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        draws = torch.randint(2, (2,), generator=generator)
        complementary = draws + (draws >= observed).long()
        self.assertTrue(torch.all(complementary != observed))
        probabilities = torch.softmax(logits, dim=1)
        p_complementary = probabilities.gather(1, complementary[:, None]).squeeze(1)
        machine_unlearning = torch.log_softmax(logits, dim=1).gather(1, observed[:, None]).squeeze(1) / 3
        negative_learning = -torch.log((1.0 - p_complementary).clamp_min(1e-7)) / 3
        expected = 0.001 * machine_unlearning.mean() + 0.1 * negative_learning.mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_fine_warmup_is_plain_cross_entropy(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        context = ScratchContext(logits=logits, labels=labels)
        _run("fine_warmup_loss", context, logits="logits", labels="labels", save_as="loss")
        torch.testing.assert_close(context["loss"], torch.nn.functional.cross_entropy(logits, labels))

    def test_cnlcu_soft_score_matches_paper_eq7(self) -> None:
        robust = torch.tensor([0.7, 1.2])
        history_length = torch.tensor([3.0, 3.0])
        effective_count = torch.tensor([1.0, 5.0])
        score, bonus = cnlcu_soft_score(robust, history_length, effective_count, 0.1)
        expected_bonus = 0.1 * (history_length + 0.1 * torch.log(2.0 * history_length) / history_length.square()) / (effective_count - 0.1)
        torch.testing.assert_close(bonus, expected_bonus)
        torch.testing.assert_close(score, robust - expected_bonus)

    def test_cal_reference_transition_means_use_retained_proxy_pairs(self) -> None:
        """CAL Eq. (8) must use empirical proxy→observed transitions."""
        from lnl_toolbox.scratch.native_stats import _reference_transition_means

        proxy = CALProxyArtifact(
            indices=np.array([10, 20, 30, 40]),
            targets=np.array([0, 0, 1, 1]),
            retained=np.array([True, True, True, False]),
            dataset="toy", lower=-1.0, upper=1.0,
        )
        # The training arrays are deliberately out of order to exercise the
        # stable-index alignment rather than relying on row position.
        actual = _reference_transition_means(
            proxy,
            np.array([40, 10, 30, 20]),
            np.array([1, 1, 0, 1]),
            2,
        )
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        torch.testing.assert_close(actual, expected)

    def test_cal_cores2_uses_square_root_noisy_prior(self) -> None:
        logits = torch.tensor([[1.5, 0.5], [0.2, 1.2]])
        labels = torch.tensor([0, 1])
        prior = torch.tensor([0.8, 0.2])
        context = ScratchContext(logits=logits, labels=labels, prior=prior, confidence=1.0)
        _run("cal_cores2_adjusted_risk", context, logits="logits", labels="labels", noisy_prior="prior", confidence_weight="confidence", save_as="risk")
        probabilities = torch.softmax(logits, dim=1)
        observed = -torch.log(probabilities + 1e-8).gather(1, labels[:, None]).squeeze(1)
        all_losses = -torch.log(probabilities + 1e-5)
        root_prior = prior.sqrt(); root_prior = root_prior / root_prior.sum()
        expected = (observed - (all_losses * root_prior).sum(1)).mean()
        torch.testing.assert_close(context["risk"], expected)

    def test_cal_recipe_uses_paper_proxy_thresholds(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "cal.yaml")
        step = next(step for step in _walk(recipe["steps"]) if step.get("block") == "cal_materialize_proxy_artifact")
        self.assertEqual(step["params"]["lower_threshold"], -8.3)
        self.assertEqual(step["params"]["upper_threshold"], -8.3)

    def test_pdl_transition_uses_per_sample_coefficients(self) -> None:
        features = FeatureSnapshot(
            np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]), np.array([20, 10]), "toy", "train"
        )
        posterior = PosteriorSnapshot(
            np.array([[0.6, 0.4], [0.2, 0.8]]), np.array([0, 1]), np.array([20, 10]), "toy", "train"
        )
        parts = np.eye(2)
        coefficients = np.array([[1.0, 0.0], [0.0, 1.0]])
        representation_indices = np.array([10, 20])
        matrices = np.array(
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.3, 0.7], [0.6, 0.4]],
            ]
        )
        artifact = PartTransitionEstimator(2, 2, representation_seed=1).estimate_from_shared_representation(
            features,
            posterior,
            representation_parts=parts,
            representation_coefficients=coefficients,
            representation_indices=representation_indices,
            part_matrices=matrices,
        )
        actual = artifact.transition_for(None, torch.tensor([20, 10]), dtype=torch.float64)
        expected = torch.as_tensor(np.stack([matrices[1], matrices[0]]), dtype=torch.float64)
        torch.testing.assert_close(actual, expected)

    def test_ca2c_candidate_memory_accumulates_counts(self) -> None:
        context = ScratchContext(indices=torch.tensor([0, 2]), epoch=4)
        _run("create_indexed_state", context, size=3, width=3, dtype="float32", initial_value=0.0, save_as="memory")
        first = torch.tensor([[True, False, True], [False, True, False]])
        second = torch.tensor([[False, True, True], [True, False, False]])
        context["values"] = first
        _run("indexed_accumulate", context, state="memory", indices="indices", values="values", save_as="memory")
        context["values"] = second
        _run("indexed_accumulate", context, state="memory", indices="indices", values="values", save_as="memory")
        _run("indexed_read", context, state="memory", indices="indices", save_as="read")
        torch.testing.assert_close(context["read"], torch.tensor([[1.0, 1.0, 2.0], [1.0, 1.0, 0.0]]))

    def test_top_k_and_selection_are_stable_and_ceil_is_available(self) -> None:
        context = ScratchContext(scores=torch.tensor([1.0, 1.0, 0.0]), fraction=torch.tensor(0.5), indices=torch.tensor([20, 10, 30]))
        _run("select_lowest_scores", context, scores="scores", keep_fraction="fraction", stable_sample_indices="indices", rounding="ceil", minimum_count=0, save_as="selected")
        self.assertEqual(context["selected"].numel(), 2)
        self.assertEqual(context["selected"].tolist(), [2, 1])
        context = ScratchContext(scores=torch.tensor([[2.0, 2.0, 1.0]]))
        _run("top_k_mask", context, scores="scores", k=1, save_as="mask")
        self.assertEqual(context["mask"].tolist(), [[True, False, False]])

    def test_t_revision_raw_additive_mode_does_not_project_or_normalize(self) -> None:
        context = ScratchContext(transition=torch.eye(2))
        _run("create_additive_transition_revision", context, transition="transition", project_nonnegative=False, normalize=False, save_as="revision")
        with torch.no_grad():
            context["revision"].delta.copy_(torch.tensor([[-2.0, 0.0], [0.0, 1.0]]))
        torch.testing.assert_close(context["revision"].matrix(), torch.tensor([[-1.0, 0.0], [0.0, 2.0]]))

    def test_dld_pre_correction_uses_two_view_posteriors(self) -> None:
        weak = torch.tensor([[0.90, 0.10], [0.55, 0.45], [0.20, 0.80]])
        strong = torch.tensor([[0.80, 0.20], [0.10, 0.90], [0.30, 0.70]])
        context = ScratchContext(weak=weak, strong=strong, labels=torch.tensor([0, 0, 1]))
        _run(
            "dld_pre_correct_labels", context,
            weak_probabilities="weak", strong_probabilities="strong",
            noisy_labels="labels", y0_as="y0", yn_as="yn",
            direction_as="direction", partition_as="partition",
            divergence_as="divergence",
        )
        torch.testing.assert_close(context["direction"], context["yn"] - context["y0"])
        self.assertEqual(tuple(context["y0"].shape), (3, 2))
        self.assertTrue(torch.isfinite(context["divergence"]).all())
        changed = ScratchContext(weak=weak, strong=weak, labels=torch.tensor([0, 0, 1]))
        _run(
            "dld_pre_correct_labels", changed,
            weak_probabilities="weak", strong_probabilities="strong",
            noisy_labels="labels", y0_as="y0", yn_as="yn",
            direction_as="direction", partition_as="partition",
            divergence_as="divergence",
        )
        self.assertFalse(torch.equal(context["direction"], changed["direction"]))

    def test_dividemix_loss_snapshot_covers_the_whole_loader(self) -> None:
        model = torch.nn.Linear(2, 2)
        rows = [
            {"inputs": torch.tensor([1.0, 0.0]), "targets": 0, "indices": 4},
            {"inputs": torch.tensor([0.0, 1.0]), "targets": 1, "indices": 2},
            {"inputs": torch.tensor([1.0, 1.0]), "targets": 0, "indices": 7},
        ]
        loader = torch.utils.data.DataLoader(rows, batch_size=2, shuffle=False)
        context = ScratchContext(model=model, loader=loader, device="cpu")
        _run(
            "collect_loss_snapshot", context,
            model="model", loader="loader", device="device",
            losses_as="losses", targets_as="targets", indices_as="snapshot_indices",
            save_as="snapshot",
        )
        self.assertEqual(context["snapshot_indices"].tolist(), [2, 4, 7])
        self.assertEqual(context["losses"].numel(), 3)
        self.assertEqual(context["snapshot"].global_indices.tolist(), [2, 4, 7])

    def test_dividemix_reads_peer_probability_from_indexed_state(self) -> None:
        recipe = load_recipe(ROOT / "recipes" / "papers" / "dividemix.yaml")
        epoch = next(step for step in recipe["steps"] if step.get("block") == "epoch_loop")
        batch = next(step for step in epoch["steps"] if step.get("block") == "batch_loop")
        first_peer_read = next(step for step in batch["steps"] if step.get("block") == "indexed_read")
        self.assertEqual(first_peer_read["params"]["state"], "clean_probability_state_b")
        detach = next(step for step in batch["steps"] if step.get("block") == "detach")
        self.assertEqual(detach["params"]["input"], "clean_probability")


if __name__ == "__main__":
    unittest.main()
