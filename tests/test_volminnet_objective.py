from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from lnl_toolbox.algorithms.volminnet import volminnet_objective


class VolMinNetObjectiveTest(unittest.TestCase):
    def test_asymmetric_row_direction_and_hand_calculation(self) -> None:
        logits = torch.tensor([[1.1, -0.2, 0.4]], dtype=torch.float64, requires_grad=True)
        transition = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.25, 0.15, 0.6]],
            dtype=torch.float64,
            requires_grad=True,
        )
        targets = torch.tensor([1])
        objective, metrics = volminnet_objective(logits, targets, transition, lambda_volume=0.03)
        clean = torch.softmax(logits, 1)
        expected_nll = -torch.log((clean @ transition)[0, 1])
        expected = expected_nll + 0.03 * torch.logdet(transition)
        self.assertTrue(torch.allclose(objective, expected))
        self.assertAlmostEqual(metrics["classification_loss"], float(expected_nll.detach()))
        self.assertFalse(torch.allclose(clean @ transition, clean @ transition.T))

    def test_classifier_and_transition_receive_finite_gradients(self) -> None:
        logits = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        raw = torch.tensor(
            [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.15, 0.1, 0.75]],
            dtype=torch.float64,
            requires_grad=True,
        )
        objective, _ = volminnet_objective(
            logits, torch.tensor([0, 1, 2, 1, 0]), raw, lambda_volume=1e-4
        )
        objective.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))

    def test_positive_logdet_sign_is_not_ignored(self) -> None:
        negative = torch.tensor(
            [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]], dtype=torch.float64
        )
        self.assertLess(float(torch.linalg.slogdet(negative).sign), 0.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            volminnet_objective(torch.randn(2, 3), torch.tensor([0, 1]), negative, lambda_volume=1e-4)

    def test_singular_and_non_finite_fail(self) -> None:
        singular = torch.full((3, 3), 1.0 / 3.0, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "positive"):
            volminnet_objective(torch.randn(2, 3), torch.tensor([0, 1]), singular, lambda_volume=1e-4)
        invalid = torch.eye(3, dtype=torch.float64)
        invalid[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            volminnet_objective(torch.randn(2, 3), torch.tensor([0, 1]), invalid, lambda_volume=1e-4)

    def test_objective_uses_log_probabilities_not_cross_entropy_on_probabilities(self) -> None:
        logits = torch.tensor([[0.2, 0.4, -0.1]], dtype=torch.float64)
        transition = torch.tensor(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float64
        )
        objective, _ = volminnet_objective(logits, torch.tensor([1]), transition, lambda_volume=0.1)
        wrong = F.cross_entropy(torch.softmax(logits, 1) @ transition, torch.tensor([1]))
        wrong = wrong + 0.1 * torch.logdet(transition)
        self.assertFalse(torch.allclose(objective, wrong))
