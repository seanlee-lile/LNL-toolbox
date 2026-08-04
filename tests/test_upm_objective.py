import unittest

import torch

from lnl_toolbox.algorithms.upm import (
    ObservedNoisyProbabilityLookup,
    predict_true_posterior,
    soft_target_cross_entropy,
    update_confusing_probability,
)
from lnl_toolbox.noise import PosteriorSnapshot


class UPMObjectiveTest(unittest.TestCase):
    def test_eq8_hand_calculation_and_boundaries(self) -> None:
        clean = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.6, 0.0]])
        targets = torch.tensor([1, 0])
        psi = torch.tensor([0.3, 0.4])
        eta = torch.tensor([0.25, 0.0])
        actual = predict_true_posterior(clean, targets, psi, eta)
        factor = torch.tensor([[0.075, 0.825, 0.075], [1.0, 0.0, 0.0]])
        expected = clean * factor
        expected /= expected.sum(1, keepdim=True)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual[1], torch.tensor([1.0, 0.0, 0.0]))

        eta_one = predict_true_posterior(
            clean[:1], targets[:1], psi[:1], torch.ones(1)
        )
        torch.testing.assert_close(eta_one, clean[:1])

    def test_eq8_rejects_zero_normalizer(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            predict_true_posterior(
                torch.tensor([[1.0, 0.0]]), torch.tensor([1]),
                torch.tensor([0.0]), torch.tensor([0.0]),
            )

    def test_eq11_hand_calculation_clamp_and_detach(self) -> None:
        eta = torch.tensor([0.2, 0.9], requires_grad=True)
        q = torch.tensor([[0.25, 0.75], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        psi = torch.tensor([0.6, 0.4])
        actual = update_confusing_probability(
            eta, q, targets, psi, learning_rate=0.1, epsilon=1e-4
        )
        one_hot = torch.nn.functional.one_hot(targets, 2).float()
        bracket = torch.ones_like(q) + (
            psi * eta.detach() - eta.detach() - 1
        )[:, None] * one_hot
        expected = (
            eta.detach()
            + 0.1 * (bracket * q).sum(1) / (eta.detach() + 1e-4)
        ).clamp(0, 1)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(bool(((actual >= 0) & (actual <= 1)).all()))

    def test_observed_class_psi_is_not_max_probability(self) -> None:
        snapshot = PosteriorSnapshot(
            noisy_probabilities=[[0.9, 0.1], [0.2, 0.8]],
            noisy_targets=[1, 0], global_indices=[20, 10],
            dataset="tiny", split="train",
        )
        lookup = ObservedNoisyProbabilityLookup(snapshot, "cpu")
        result = lookup.resolve(
            torch.tensor([10, 20]), torch.tensor([0, 1]), dtype=torch.float32
        )
        torch.testing.assert_close(result, torch.tensor([0.2, 0.1]))

    def test_soft_target_ce_value_and_gradient(self) -> None:
        logits = torch.tensor([[1.0, -1.0], [0.0, 2.0]], requires_grad=True)
        q = torch.tensor([[0.25, 0.75], [0.8, 0.2]])
        values = soft_target_cross_entropy(logits, q)
        expected = -(q * torch.log_softmax(logits, 1)).sum(1)
        torch.testing.assert_close(values, expected)
        values.mean().backward()
        torch.testing.assert_close(logits.grad, (torch.softmax(logits.detach(), 1) - q) / 2)


if __name__ == "__main__":
    unittest.main()
