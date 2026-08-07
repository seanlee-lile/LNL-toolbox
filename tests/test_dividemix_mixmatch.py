from __future__ import annotations

import unittest

import numpy as np
import torch

from lnl_toolbox.algorithms.dividemix import dividemix_objective, mixmatch_mixup, unsupervised_weight


class DivideMixMixMatchTest(unittest.TestCase):
    def test_mixup_uses_one_scalar_and_shared_permutation(self):
        x = (torch.tensor([[0.0], [1.0]]), torch.tensor([[2.0], [3.0]]))
        u = (torch.tensor([[4.0], [5.0]]), torch.tensor([[6.0], [7.0]]))
        y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        q = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
        order = torch.arange(7, -1, -1)
        result = mixmatch_mixup(x, u, y, q, alpha=4.0, rng=np.random.default_rng(4), permutation=order)
        self.assertGreaterEqual(result.mix_lambda, 0.5)
        expected = result.mix_lambda * torch.cat(x + u) + (1 - result.mix_lambda) * torch.cat(x + u)[order]
        self.assertTrue(torch.allclose(result.inputs, expected))
        self.assertEqual(result.labeled_count, 4)

    def test_objective_matches_manual_terms_and_keeps_autograd(self):
        logits_x = torch.tensor([[1.0, 0.0]], requires_grad=True)
        logits_u = torch.tensor([[0.0, 1.0]], requires_grad=True)
        targets_x = torch.tensor([[0.75, 0.25]])
        targets_u = torch.tensor([[0.2, 0.8]])
        all_logits = torch.cat((logits_x, logits_u))
        objective, metrics = dividemix_objective(logits_x, targets_x, logits_u, targets_u, all_logits, lambda_u=2.0, lambda_r=1.0)
        objective.backward()
        self.assertIsNotNone(logits_x.grad)
        self.assertIsNotNone(logits_u.grad)
        self.assertAlmostEqual(float(objective.detach()), metrics["objective"], places=6)

    def test_fractional_ramp_up(self):
        self.assertEqual(unsupervised_weight(25.0, 1.0, 1, 16), 0.0)
        self.assertAlmostEqual(unsupervised_weight(25.0, 9.0, 1, 16), 12.5)
        self.assertEqual(unsupervised_weight(25.0, 30.0, 1, 16), 25.0)


if __name__ == "__main__":
    unittest.main()
