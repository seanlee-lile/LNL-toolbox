from __future__ import annotations

import math
import unittest

import torch
from torch.nn import functional as F

from lnl_toolbox.algorithms.t_revision import t_revision_reweight_objective


class TRevisionObjectiveTest(unittest.TestCase):
    def test_matches_vectorized_equation_three_by_hand(self) -> None:
        logits = torch.tensor([[1.2, -0.4], [0.1, 0.8]], requires_grad=True)
        targets = torch.tensor([0, 1])
        transition = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
        result = t_revision_reweight_objective(
            logits, targets, transition, denominator_floor=1e-12
        )
        clean = torch.softmax(logits, dim=1)
        noisy = clean @ transition
        expected_weights = torch.stack([clean[0, 0] / noisy[0, 0], clean[1, 1] / noisy[1, 1]])
        expected = (expected_weights * F.cross_entropy(logits, targets, reduction="none")).mean()
        torch.testing.assert_close(result.objective, expected)
        self.assertAlmostEqual(result.weight_mean, float(expected_weights.mean()))

    def test_non_commuting_direction_is_clean_probability_times_transition(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
        targets = torch.tensor([1])
        transition = torch.tensor([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.4, 0.1, 0.5],
        ])
        result = t_revision_reweight_objective(
            logits, targets, transition, denominator_floor=0.0
        )
        clean = torch.softmax(logits, 1)
        correct = clean[0, 1] / (clean @ transition)[0, 1]
        reverse = clean[0, 1] / (clean @ transition.t())[0, 1]
        self.assertAlmostEqual(result.weight_mean, float(correct))
        self.assertNotAlmostEqual(result.weight_mean, float(reverse))

    def test_identity_transition_reduces_to_cross_entropy(self) -> None:
        logits = torch.randn(5, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0, 2])
        result = t_revision_reweight_objective(
            logits, targets, torch.eye(3), denominator_floor=0.0
        )
        torch.testing.assert_close(result.objective, F.cross_entropy(logits, targets))
        self.assertAlmostEqual(result.weight_min, 1.0)
        self.assertAlmostEqual(result.weight_max, 1.0)

    def test_classifier_and_transition_receive_gradients(self) -> None:
        logits = torch.tensor([[1.0, 0.2], [-0.4, 0.8]], requires_grad=True)
        transition = torch.tensor([[0.85, 0.15], [0.25, 0.75]], requires_grad=True)
        result = t_revision_reweight_objective(
            logits, torch.tensor([0, 1]), transition, denominator_floor=1e-12
        )
        result.objective.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(transition.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(transition.grad.abs().sum()), 0.0)

    def test_ratio_is_numerator_over_denominator_not_reversed(self) -> None:
        logits = torch.tensor([[1.5, -0.2]])
        transition = torch.tensor([[0.6, 0.4], [0.2, 0.8]])
        result = t_revision_reweight_objective(
            logits, torch.tensor([0]), transition, denominator_floor=0.0
        )
        self.assertNotAlmostEqual(result.weight_mean, 1.0 / result.weight_mean)

    def test_invalid_denominator_and_nonfinite_values_fail(self) -> None:
        logits = torch.tensor([[0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "strictly greater"):
            t_revision_reweight_objective(
                logits, torch.tensor([0]), torch.zeros(2, 2), denominator_floor=0.0
            )
        with self.assertRaisesRegex(ValueError, "transition must be finite"):
            t_revision_reweight_objective(
                logits,
                torch.tensor([0]),
                torch.tensor([[math.nan, 0.0], [0.0, 1.0]]),
                denominator_floor=0.0,
            )
        with self.assertRaisesRegex(ValueError, "logits must be finite"):
            t_revision_reweight_objective(
                torch.tensor([[math.inf, 0.0]]),
                torch.tensor([0]),
                torch.eye(2),
                denominator_floor=0.0,
            )

    def test_raw_negative_entries_are_not_clipped_when_denominator_is_valid(self) -> None:
        logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
        transition = torch.tensor([[1.1, -0.1], [0.2, 0.8]], requires_grad=True)
        result = t_revision_reweight_objective(
            logits, torch.tensor([0]), transition, denominator_floor=0.0
        )
        self.assertTrue(torch.isfinite(result.objective))
        result.objective.backward()
        self.assertIsNotNone(transition.grad)


if __name__ == "__main__":
    unittest.main()
