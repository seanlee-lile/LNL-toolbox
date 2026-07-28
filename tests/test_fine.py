import unittest

import torch

from lnl_toolbox.algorithms.fine import FINERegularizer
from lnl_toolbox.selectors.sed import SEDSelector
from lnl_toolbox.selectors.base import SelectionInput


class FINETest(unittest.TestCase):
    def test_regularizer_and_rng_state_round_trip(self) -> None:
        regularizer = FINERegularizer(beta=0.001, gamma=0.1, seed=4)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0])
        value = regularizer(logits, targets, selected_mask=torch.ones(4, dtype=torch.bool))
        value.backward()
        state = regularizer.state_dict()
        restored = FINERegularizer(beta=0.001, gamma=0.1, seed=4)
        restored.load_state_dict(state)
        self.assertTrue(torch.isfinite(value))

    def test_sed_keeps_at_least_one_sample(self) -> None:
        result = SEDSelector(threshold=0.0).select(SelectionInput(torch.tensor([0.2, 0.4]), torch.tensor([3, 4])))
        self.assertTrue(bool(result.selected_mask.any()))
        self.assertTrue(torch.equal(result.rejected_mask, ~result.selected_mask))

    def test_regularizer_uses_rejected_subset_and_empty_subset_is_zero(self) -> None:
        regularizer = FINERegularizer(beta=0.001, gamma=0.1, seed=4)
        logits = torch.randn(3, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2])
        rejected = torch.tensor([False, True, True])
        value = regularizer(logits, targets, rejected_mask=rejected)
        value.backward()
        self.assertEqual(float(logits.grad[0].abs().sum()), 0.0)

        empty_logits = torch.randn(2, 3, requires_grad=True)
        empty_value = regularizer(
            empty_logits,
            torch.tensor([0, 1]),
            rejected_mask=torch.zeros(2, dtype=torch.bool),
        )
        empty_value.backward()
        self.assertEqual(float(empty_value), 0.0)


if __name__ == "__main__":
    unittest.main()
