import unittest

import numpy as np
import torch
from torch.nn import functional as F

from lnl_toolbox.losses import (
    ActivePassiveLoss,
    CrossEntropyLoss,
    GeneralizedCrossEntropyLoss,
    MeanAbsoluteErrorLoss,
    NormalizedCrossEntropyLoss,
    ReverseCrossEntropyLoss,
    cross_entropy,
    generalized_cross_entropy,
)
from lnl_toolbox.losses.torch_losses import loss_for_all_targets
from lnl_toolbox.plugins.builtin import build_builtin_loss


class LossTest(unittest.TestCase):
    def test_loss_for_all_targets_reuses_per_sample_loss_contract(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
        values = loss_for_all_targets(CrossEntropyLoss(), logits)
        expected = torch.stack(
            [
                CrossEntropyLoss()(logits, torch.zeros(2, dtype=torch.long)),
                CrossEntropyLoss()(logits, torch.ones(2, dtype=torch.long)),
                CrossEntropyLoss()(logits, torch.full((2,), 2, dtype=torch.long)),
            ],
            dim=1,
        )
        torch.testing.assert_close(values, expected)
    def test_gce_approaches_ce(self) -> None:
        probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
        targets = np.array([0, 1])
        np.testing.assert_allclose(
            generalized_cross_entropy(probabilities, targets, q=1e-7),
            cross_entropy(probabilities, targets),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_torch_ce_matches_pytorch_per_sample(self) -> None:
        logits = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 1.5, 0.0]])
        targets = torch.tensor([0, 2])
        actual = CrossEntropyLoss()(logits, targets)
        expected = F.cross_entropy(logits, targets, reduction="none")
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (2,))

    def test_torch_gce_limits(self) -> None:
        logits = torch.tensor([[1.2, -0.3, 0.1], [0.2, 0.8, -1.0]], dtype=torch.float64)
        targets = torch.tensor([0, 1])
        ce = CrossEntropyLoss()(logits, targets)
        torch.testing.assert_close(
            GeneralizedCrossEntropyLoss(q=1e-7)(logits, targets), ce, rtol=1e-6, atol=1e-7
        )
        p_y = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        torch.testing.assert_close(GeneralizedCrossEntropyLoss(q=1.0)(logits, targets), 1.0 - p_y)

    def test_torch_gce_matches_paper_below_previous_epsilon_threshold(self) -> None:
        q = 0.1
        logits = torch.tensor([[0.0, -30.0]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1])
        actual = GeneralizedCrossEntropyLoss(q=q)(logits, targets)
        actual_gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]

        p_y = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        expected = (1.0 - p_y.pow(q)) / q
        expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]

        self.assertLess(float(p_y.item()), 1e-8)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertGreater(float(actual_gradient.abs().max().item()), 0.0)

    def test_normalized_ce_sums_to_one_across_hypothetical_targets(self) -> None:
        logits = torch.tensor([[1.3, -0.4, 0.2]], dtype=torch.float64).repeat(3, 1)
        targets = torch.arange(3)
        values = NormalizedCrossEntropyLoss()(logits, targets)
        torch.testing.assert_close(values.sum(), torch.tensor(1.0, dtype=torch.float64))

    def test_mae_rce_and_apl_match_manual_formulas(self) -> None:
        logits = torch.tensor([[1.0, 0.0], [-0.2, 0.6]], dtype=torch.float64)
        targets = torch.tensor([0, 1])
        p_y = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        mae = MeanAbsoluteErrorLoss()(logits, targets)
        rce = ReverseCrossEntropyLoss(log_zero=-4.0)(logits, targets)
        torch.testing.assert_close(mae, 2.0 * (1.0 - p_y))
        torch.testing.assert_close(rce, 4.0 * (1.0 - p_y))
        active = NormalizedCrossEntropyLoss()
        apl = ActivePassiveLoss(active, ReverseCrossEntropyLoss(), alpha=2.0, beta=0.5)
        torch.testing.assert_close(
            apl(logits, targets), 2.0 * active(logits, targets) + 0.5 * rce
        )

    def test_extreme_logits_have_finite_values_and_gradients(self) -> None:
        configs = [
            {"name": "ce"},
            {"name": "gce", "q": 0.7},
            {"name": "nce"},
            {"name": "mae"},
            {"name": "rce"},
            {"name": "apl"},
        ]
        for config in configs:
            with self.subTest(name=config["name"]):
                logits = torch.tensor(
                    [[10000.0, -10000.0, 0.0], [-10000.0, 10000.0, 0.0]],
                    requires_grad=True,
                )
                values = build_builtin_loss(config)(logits, torch.tensor([1, 1]))
                self.assertTrue(torch.isfinite(values).all())
                values.mean().backward()
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_loss_parameters_and_inputs_fail(self) -> None:
        for q in (0.0, 1.1):
            with self.subTest(q=q), self.assertRaises(ValueError):
                GeneralizedCrossEntropyLoss(q=q)
        with self.assertRaises(ValueError):
            ReverseCrossEntropyLoss(log_zero=0.0)
        with self.assertRaises(ValueError):
            NormalizedCrossEntropyLoss(eps=float("nan"))
        for alpha, beta in ((0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)):
            with self.subTest(alpha=alpha, beta=beta), self.assertRaises(ValueError):
                ActivePassiveLoss(
                    NormalizedCrossEntropyLoss(), MeanAbsoluteErrorLoss(), alpha, beta
                )
        with self.assertRaises(TypeError):
            ActivePassiveLoss(CrossEntropyLoss(), MeanAbsoluteErrorLoss())
        with self.assertRaises(TypeError):
            ActivePassiveLoss(NormalizedCrossEntropyLoss(), CrossEntropyLoss())
        with self.assertRaises(ValueError):
            CrossEntropyLoss()(torch.randn(2, 3, 1), torch.tensor([0, 1]))
        with self.assertRaises(TypeError):
            CrossEntropyLoss()(torch.randn(2, 3), torch.tensor([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()

