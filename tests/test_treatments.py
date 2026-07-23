from __future__ import annotations

import unittest

import torch

from lnl_toolbox.selectors import SelectionInput, SelectionResult
from lnl_toolbox.treatments import (
    ContributionResult,
    ReductionSpec,
    SelectorContributionAdapter,
    reduce_per_sample_loss,
    validate_contribution_result,
)


class SampleTreatmentTest(unittest.TestCase):
    def test_selector_adapter_preserves_mask_metrics_and_adds_identity_weights(self):
        class FixedSelector:
            def select(self, selection_input):
                return SelectionResult(
                    selected_mask=torch.tensor([False, True, True]),
                    metrics={"selected_samples": 2.0, "selected_ratio": 2.0 / 3.0},
                )

        result = SelectorContributionAdapter(FixedSelector()).resolve(
            SelectionInput(
                scores=torch.tensor([0.7, 0.2, 0.3]),
                sample_indices=torch.tensor([8, 2, 5]),
            )
        )

        self.assertEqual(result.selected_mask.tolist(), [False, True, True])
        self.assertTrue(torch.equal(result.sample_weights, torch.ones(3)))
        self.assertEqual(result.metrics["selected_samples"], 2.0)
        self.assertEqual(result.metrics["selected_ratio"], 2.0 / 3.0)

    def test_weight_sum_mean_matches_selected_hard_mask_mean(self):
        losses = torch.tensor([1.0, 2.0, 7.0, 9.0], requires_grad=True)
        contribution = ContributionResult(
            selected_mask=torch.tensor([True, False, True, False]),
            sample_weights=torch.ones(4),
        )

        objective = reduce_per_sample_loss(losses, contribution)

        self.assertEqual(objective.item(), losses[[0, 2]].mean().item())
        objective.backward()
        self.assertTrue(torch.equal(
            losses.grad, torch.tensor([0.5, 0.0, 0.5, 0.0])
        ))

    def test_continuous_weights_and_reduction_modes(self):
        losses = torch.tensor([1.0, 3.0, 8.0])
        contribution = ContributionResult(
            selected_mask=torch.tensor([True, True, False]),
            sample_weights=torch.tensor([1.0, 3.0, 100.0]),
        )

        self.assertEqual(
            reduce_per_sample_loss(losses, contribution).item(),
            2.5,
        )
        self.assertAlmostEqual(
            reduce_per_sample_loss(
                losses, contribution, ReductionSpec("batch_mean")
            ).item(),
            10.0 / 3.0,
            places=6,
        )
        self.assertEqual(
            reduce_per_sample_loss(
                losses, contribution, ReductionSpec("sum")
            ).item(),
            10.0,
        )

    def test_reduction_spec_rejects_unknown_normalization(self):
        with self.assertRaisesRegex(ValueError, "normalization"):
            ReductionSpec("mean")

    def test_contribution_validation_rejects_invalid_weights_and_empty_result(self):
        cases = (
            (
                ContributionResult(
                    selected_mask=torch.tensor([True, False]),
                    sample_weights=torch.tensor([1.0, -1.0]),
                ),
                "non-negative",
            ),
            (
                ContributionResult(
                    selected_mask=torch.tensor([True, False]),
                    sample_weights=torch.tensor([float("nan"), 1.0]),
                ),
                "finite",
            ),
            (
                ContributionResult(
                    selected_mask=torch.tensor([False, False]),
                    sample_weights=torch.ones(2),
                ),
                "positive contribution",
            ),
            (
                ContributionResult(
                    selected_mask=torch.tensor([True, True]),
                    sample_weights=torch.zeros(2),
                ),
                "positive contribution",
            ),
        )
        for result, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                validate_contribution_result(
                    result, batch_size=2, device=torch.device("cpu")
                )

    def test_reducer_rejects_nonfinite_loss_without_fallback(self):
        contribution = ContributionResult(
            selected_mask=torch.tensor([True, False]),
            sample_weights=torch.ones(2),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            reduce_per_sample_loss(
                torch.tensor([float("nan"), 2.0]), contribution
            )


if __name__ == "__main__":
    unittest.main()
