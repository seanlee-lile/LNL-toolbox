import unittest

import torch

from lnl_toolbox.algorithms.fine import FINERegularizer
from lnl_toolbox.selectors.sed import (
    SEDSelector,
    SelfAdaptiveClassSelector,
    SelfAdaptiveConfidenceReweighting,
)
from lnl_toolbox.selectors.base import SelectionInput


class FINETest(unittest.TestCase):
    def test_regularizer_matches_official_two_terms_and_state_round_trip(self) -> None:
        regularizer = FINERegularizer(beta=0.001, gamma=0.1, seed=4)
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        targets = torch.tensor([0, 0])
        pseudo = torch.tensor([0, 1])
        value = regularizer(
            logits,
            targets,
            rejected_mask=torch.ones(2, dtype=torch.bool),
            pseudo_labels=pseudo,
        )
        probabilities = torch.softmax(logits, dim=1)
        expected = (
            0.001 * -torch.log(1.0 + 1.0e-7 - probabilities[1, 0])
            + 0.1 * torch.log_softmax(logits, dim=1)[1, 0]
        )
        self.assertTrue(torch.allclose(value, expected))
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
        value = regularizer(
            logits,
            targets,
            rejected_mask=rejected,
            pseudo_labels=torch.tensor([1, 0, 0]),
        )
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

    def test_scs_and_scr_are_stateful_and_finite(self) -> None:
        probabilities = torch.tensor([
            [0.9, 0.1],
            [0.4, 0.6],
            [0.8, 0.2],
            [0.3, 0.7],
        ])
        targets = torch.tensor([0, 1, 1, 0])
        scs = SelfAdaptiveClassSelector(2, momentum=0.0, quantile=None)
        mask = scs.select_epoch(probabilities, targets)
        self.assertEqual(mask.shape, targets.shape)
        restored_scs = SelfAdaptiveClassSelector(2, momentum=0.0, quantile=None)
        restored_scs.load_state_dict(scs.state_dict())
        self.assertTrue(torch.equal(restored_scs.local, scs.local))

        scr = SelfAdaptiveConfidenceReweighting(2, momentum=0.0)
        weights = scr.weights(probabilities)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(bool(((weights > 0.0) & (weights <= 1.0)).all()))
        restored_scr = SelfAdaptiveConfidenceReweighting(2, momentum=0.0)
        restored_scr.load_state_dict(scr.state_dict())
        self.assertTrue(torch.equal(restored_scr.mean, scr.mean))


if __name__ == "__main__":
    unittest.main()
