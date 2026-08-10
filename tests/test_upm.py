from __future__ import annotations

import unittest

import torch

from lnl_toolbox.algorithms.upm import predict_true_posterior, soft_target_cross_entropy


class UPMMathTest(unittest.TestCase):
    def test_eq8_is_normalized_and_eta_zero_keeps_noisy_label(self) -> None:
        logits = torch.zeros(2, 3, requires_grad=True)
        labels = torch.tensor([1, 2])
        q = predict_true_posterior(
            logits.softmax(1), labels, torch.tensor([0.8, 0.7]), torch.zeros(2)
        )
        torch.testing.assert_close(q.sum(1), torch.ones(2))
        self.assertEqual(int(q[0].argmax()), 1)
        loss = soft_target_cross_entropy(logits, q).mean()
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_eta_changes_posterior_towards_snapshot_probability(self) -> None:
        logits = torch.tensor([[4.0, 0.0, 0.0]])
        q = predict_true_posterior(
            logits.softmax(1), torch.tensor([1]), torch.tensor([0.9]), torch.tensor([1.0])
        )
        self.assertGreater(float(q[0, 0]), float(q[0, 1]))


if __name__ == "__main__":
    unittest.main()
