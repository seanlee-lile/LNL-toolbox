from __future__ import annotations

import unittest

import torch

from lnl_toolbox.estimators import (
    DivideMixGMMCleanProbabilityEstimator,
    DivideMixGMMLossInput,
    ReliabilityResult,
    ReliabilityToSelectionInputAdapter,
)
from lnl_toolbox.selectors import SmallLossSelector
from lnl_toolbox.treatments import SelectorContributionAdapter


class ReliabilitySelectionAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = ReliabilityToSelectionInputAdapter()

    def test_dataset_result_is_aligned_to_requested_batch_order(self):
        result = ReliabilityResult(
            sample_indices=torch.tensor([90, 7, 41, 300]),
            scores=torch.tensor([0.2, 0.9, 0.5, 0.8]),
        )
        expected = torch.tensor([300, 7, 90])

        selection_input = self.adapter.adapt(
            result,
            expected_sample_indices=expected,
            metadata={"epoch": 2},
        )

        self.assertTrue(torch.equal(
            selection_input.sample_indices, expected
        ))
        self.assertTrue(torch.equal(
            selection_input.scores, torch.tensor([-0.8, -0.9, -0.2])
        ))
        self.assertEqual(selection_input.metadata, {"epoch": 2})

    def test_result_and_expected_permutations_preserve_index_score_mapping(self):
        indices = torch.tensor([101, -5, 800, 42])
        scores = torch.tensor([0.1, 0.7, 0.4, 0.9])
        first = self.adapter.adapt(
            ReliabilityResult(indices, scores),
            expected_sample_indices=torch.tensor([42, 101, 800]),
        )
        permutation = torch.tensor([2, 0, 3, 1])
        second = self.adapter.adapt(
            ReliabilityResult(
                indices[permutation],
                scores[permutation],
            ),
            expected_sample_indices=torch.tensor([800, 42, 101]),
        )

        first_by_index = {
            int(index): float(score)
            for index, score in zip(
                first.sample_indices.tolist(), first.scores.tolist()
            )
        }
        second_by_index = {
            int(index): float(score)
            for index, score in zip(
                second.sample_indices.tolist(), second.scores.tolist()
            )
        }
        self.assertEqual(first_by_index, second_by_index)

    def test_missing_and_duplicate_expected_indices_are_rejected(self):
        result = ReliabilityResult(
            sample_indices=torch.tensor([2, 8, 50]),
            scores=torch.tensor([0.2, 0.8, 0.5]),
        )

        with self.assertRaisesRegex(ValueError, "absent"):
            self.adapter.adapt(
                result,
                expected_sample_indices=torch.tensor([8, 99]),
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            self.adapter.adapt(
                result,
                expected_sample_indices=torch.tensor([8, 8]),
            )

    def test_invalid_expected_index_contract_is_rejected(self):
        result = ReliabilityResult(
            sample_indices=torch.tensor([2, 8]),
            scores=torch.tensor([0.2, 0.8]),
        )
        cases = (
            (torch.tensor([[2, 8]]), "one-dimensional"),
            (torch.tensor([], dtype=torch.long), "must not be empty"),
            (torch.tensor([2.0, 8.0]), "integer dtype"),
        )
        for expected, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                self.adapter.adapt(
                    result,
                    expected_sample_indices=expected,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_expected_indices_must_share_result_device(self):
        result = ReliabilityResult(
            sample_indices=torch.tensor([2, 8], device="cuda"),
            scores=torch.tensor([0.2, 0.8], device="cuda"),
        )

        with self.assertRaisesRegex(ValueError, "result device"):
            self.adapter.adapt(
                result,
                expected_sample_indices=torch.tensor([8]),
            )

    def test_invalid_reliability_result_is_rejected(self):
        cases = (
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([1, 2]),
                    scores=torch.tensor([0.1, float("nan")]),
                ),
                "finite",
            ),
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([1, 2]),
                    scores=torch.tensor([0.1, float("inf")]),
                ),
                "finite",
            ),
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([1, 2]),
                    scores=torch.tensor([0.1, 0.2], requires_grad=True),
                ),
                "detached",
            ),
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([1, 1]),
                    scores=torch.tensor([0.1, 0.2]),
                ),
                "unique",
            ),
        )
        for result, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                self.adapter.adapt(
                    result,
                    expected_sample_indices=torch.tensor([1]),
                )

    def test_high_reliability_is_preferred_by_small_loss_selector(self):
        result = ReliabilityResult(
            sample_indices=torch.tensor([70, 10, 90, 30]),
            scores=torch.tensor([0.1, 0.95, 0.4, 0.8]),
        )
        expected = torch.tensor([90, 70, 30, 10])
        selection_input = self.adapter.adapt(
            result,
            expected_sample_indices=expected,
        )

        selection = SmallLossSelector(0.5).select(selection_input)

        selected_indices = expected[selection.selected_mask]
        self.assertEqual(set(selected_indices.tolist()), {10, 30})

    def test_output_can_continue_through_existing_contribution_adapter(self):
        result = ReliabilityResult(
            sample_indices=torch.tensor([20, 4, 80, 9]),
            scores=torch.tensor([0.3, 0.9, 0.1, 0.7]),
        )
        selection_input = self.adapter.adapt(
            result,
            expected_sample_indices=torch.tensor([9, 80, 4, 20]),
        )

        contribution = SelectorContributionAdapter(
            SmallLossSelector(0.5)
        ).resolve(selection_input)

        self.assertEqual(
            contribution.selected_mask.tolist(),
            [True, False, True, False],
        )
        self.assertTrue(torch.equal(
            contribution.sample_weights,
            torch.ones(4),
        ))

    def test_adapter_does_not_modify_result_or_add_treatment_behavior(self):
        indices = torch.tensor([6, 2, 99])
        scores = torch.tensor([0.6, 0.2, 0.9])
        result = ReliabilityResult(
            sample_indices=indices,
            scores=scores,
            metrics={"score_mean": float(scores.mean().item())},
        )
        original_indices = indices.clone()
        original_scores = scores.clone()
        original_metrics = dict(result.metrics)

        selection_input = self.adapter.adapt(
            result,
            expected_sample_indices=torch.tensor([99, 6]),
        )

        self.assertTrue(torch.equal(result.sample_indices, original_indices))
        self.assertTrue(torch.equal(result.scores, original_scores))
        self.assertEqual(result.metrics, original_metrics)
        for absent_attribute in (
            "selected_mask",
            "sample_weights",
            "threshold",
            "labels",
            "split",
        ):
            self.assertFalse(hasattr(selection_input, absent_attribute))

    def test_dividemix_component_output_becomes_batch_ranking_input_only(self):
        estimator_input = DivideMixGMMLossInput(
            per_sample_losses=torch.tensor(
                [0.10, 0.13, 0.16, 1.80, 2.00, 2.20]
            ),
            sample_indices=torch.tensor([40, 10, 70, 20, 90, 30]),
        )
        reliability = DivideMixGMMCleanProbabilityEstimator(
            random_seed=17
        ).estimate(estimator_input)
        expected = torch.tensor([90, 10, 30])

        selection_input = self.adapter.adapt(
            reliability,
            expected_sample_indices=expected,
        )

        self.assertTrue(torch.equal(
            selection_input.sample_indices, expected
        ))
        expected_scores = -torch.stack([
            reliability.scores[
                reliability.sample_indices == index
            ].squeeze(0)
            for index in expected
        ])
        self.assertTrue(torch.equal(
            selection_input.scores, expected_scores
        ))


if __name__ == "__main__":
    unittest.main()
