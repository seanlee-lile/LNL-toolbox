from __future__ import annotations

from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.func import functional_call

from lnl_toolbox.algorithms.l2rw import meta_gradient, meta_reweight


class L2RWTest(unittest.TestCase):
    def _fixture(self):
        model = nn.Linear(1, 2, bias=False)
        with torch.no_grad(): model.weight.copy_(torch.tensor([[0.2], [-0.1]]))
        train_x = torch.tensor([[1.0], [-1.0]])
        train_y = torch.tensor([0, 0])
        trusted_x = torch.tensor([[1.0], [-1.0]])
        trusted_y = torch.tensor([0, 1])
        return model, train_x, train_y, trusted_x, trusted_y

    def test_weights_are_nonnegative_normalized_and_favor_useful_sample(self) -> None:
        model, train_x, train_y, trusted_x, trusted_y = self._fixture()
        result = meta_reweight(
            model, train_x, train_y, trusted_x, trusted_y,
            virtual_learning_rate=0.1,
        )
        self.assertTrue(bool((result.sample_weights >= 0).all()))
        self.assertAlmostEqual(float(result.sample_weights.sum()), 1.0, places=6)
        self.assertGreater(float(result.sample_weights[0]), float(result.sample_weights[1]))

    def test_virtual_step_does_not_mutate_model(self) -> None:
        model, train_x, train_y, trusted_x, trusted_y = self._fixture()
        before = deepcopy(model.state_dict())
        gradient = meta_gradient(
            model, train_x, train_y, trusted_x, trusted_y,
            virtual_learning_rate=0.1,
        )
        self.assertTrue(torch.isfinite(gradient).all())
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_meta_gradient_matches_finite_difference(self) -> None:
        model, train_x, train_y, trusted_x, trusted_y = self._fixture()
        analytic = meta_gradient(
            model, train_x, train_y, trusted_x, trusted_y,
            virtual_learning_rate=0.1,
        )
        parameters = dict(model.named_parameters())
        train_losses = torch.nn.functional.cross_entropy(
            model(train_x), train_y, reduction="none"
        )

        def trusted_loss(epsilon: torch.Tensor) -> float:
            virtual_loss = torch.sum(epsilon * train_losses)
            gradients = torch.autograd.grad(
                virtual_loss, tuple(parameters.values()), retain_graph=True
            )
            virtual = {
                name: parameter - 0.1 * gradient
                for (name, parameter), gradient in zip(parameters.items(), gradients)
            }
            logits = functional_call(model, virtual, (trusted_x,), strict=True)
            return float(torch.nn.functional.cross_entropy(logits, trusted_y))

        step = 1e-3
        positive = torch.tensor([step, 0.0], requires_grad=True)
        negative = torch.tensor([-step, 0.0], requires_grad=True)
        finite = (trusted_loss(positive) - trusted_loss(negative)) / (2.0 * step)
        self.assertAlmostEqual(float(analytic[0]), finite, places=4)

    def test_zero_meta_signal_keeps_all_weights_zero(self) -> None:
        model, train_x, train_y, _, _ = self._fixture()
        result = meta_reweight(
            model, train_x, train_y, torch.zeros(2, 1), torch.tensor([0, 1]),
            virtual_learning_rate=0.1,
        )
        self.assertEqual(result.sample_weights.tolist(), [0.0, 0.0])

    def test_invalid_virtual_rate_fails(self) -> None:
        model, train_x, train_y, trusted_x, trusted_y = self._fixture()
        with self.assertRaises(ValueError):
            meta_reweight(model, train_x, train_y, trusted_x, trusted_y, virtual_learning_rate=0.0)


if __name__ == "__main__": unittest.main()
