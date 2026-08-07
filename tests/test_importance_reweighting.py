from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

import torch

from lnl_toolbox.treatments import (
    BinaryRCNImportanceWeightProvider,
    BinaryRCNWeightInput,
    ReductionSpec,
    SupervisedWeightInput,
    WeightContributionAdapter,
    WeightResult,
    reduce_per_sample_loss,
)


class ImportanceReweightingTest(unittest.TestCase):
    def test_binary_rcn_requires_explicit_posterior_input(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        with self.assertRaisesRegex(TypeError, "BinaryRCNWeightInput"):
            provider.compute(SupervisedWeightInput(
                logits=torch.tensor([[2.0, 0.0]]),
                noisy_targets=torch.tensor([0]),
                sample_indices=torch.tensor([3]),
                per_sample_loss=torch.tensor([0.1]),
            ))

    def test_binary_asymmetric_rcn_weights_match_manual_formula(self):
        provider = BinaryRCNImportanceWeightProvider(
            rho_positive=0.2,
            rho_negative=0.1,
        )
        result = provider.compute(BinaryRCNWeightInput(
            posterior_probabilities=torch.tensor([
                [0.8, 0.2],
                [0.3, 0.7],
            ]),
            observed_targets=torch.tensor([0, 1]),
        ))

        expected = torch.tensor([
            (0.8 - 0.2) / (0.7 * 0.8),
            (0.7 - 0.1) / (0.7 * 0.7),
        ])
        self.assertTrue(torch.allclose(result.sample_weights, expected))
        self.assertEqual(
            set(result.metrics),
            {"weight_mean", "weight_min", "weight_max", "zero_weight_ratio"},
        )
        for value in result.metrics.values():
            self.assertIs(type(value), float)
            self.assertTrue(math.isfinite(value))

    def test_rate_direction_and_observed_label_probability_are_correct(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        posterior = torch.tensor([
            [0.8, 0.2],
            [0.8, 0.2],
        ])
        weights = provider.compute(BinaryRCNWeightInput(
            posterior_probabilities=posterior,
            observed_targets=torch.tensor([0, 1]),
        )).sample_weights

        expected_target_zero = (0.8 - 0.2) / (0.7 * 0.8)
        expected_target_one = (0.2 - 0.1) / (0.7 * 0.2)
        self.assertAlmostEqual(weights[0].item(), expected_target_zero, places=6)
        self.assertAlmostEqual(weights[1].item(), expected_target_one, places=6)
        self.assertNotAlmostEqual(weights[1].item(), expected_target_zero, places=6)

    def test_zero_noise_rates_produce_unit_weights_for_nonzero_q(self):
        provider = BinaryRCNImportanceWeightProvider(0.0, 0.0)
        weights = provider.compute(BinaryRCNWeightInput(
            posterior_probabilities=torch.tensor([
                [0.9, 0.1],
                [0.3, 0.7],
            ]),
            observed_targets=torch.tensor([0, 1]),
        )).sample_weights
        self.assertTrue(torch.equal(weights, torch.ones(2)))

    def test_zero_observed_probability_is_assigned_zero_without_nonfinite_value(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        result = provider.compute(BinaryRCNWeightInput(
            posterior_probabilities=torch.tensor([
                [0.0, 1.0],
                [0.3, 0.7],
            ]),
            observed_targets=torch.tensor([0, 1]),
        ))
        self.assertEqual(result.sample_weights[0].item(), 0.0)
        self.assertTrue(bool(torch.isfinite(result.sample_weights).all().item()))
        self.assertEqual(result.metrics["zero_weight_ratio"], 0.5)

    def test_invalid_noise_rates_are_rejected(self):
        invalid = (
            (-0.1, 0.1),
            (1.0, 0.0),
            (0.1, -0.1),
            (0.0, 1.0),
            (0.6, 0.4),
            (float("nan"), 0.1),
        )
        for rho_positive, rho_negative in invalid:
            with self.subTest(
                rho_positive=rho_positive,
                rho_negative=rho_negative,
            ), self.assertRaises((TypeError, ValueError)):
                BinaryRCNImportanceWeightProvider(
                    rho_positive,
                    rho_negative,
                )

    def test_invalid_posterior_and_targets_are_rejected(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        cases = (
            (
                torch.tensor([0.4, 0.6]),
                torch.tensor([0]),
                "shape",
            ),
            (
                torch.tensor([[1, 0]]),
                torch.tensor([0]),
                "floating-point",
            ),
            (
                torch.tensor([[float("nan"), float("nan")]]),
                torch.tensor([0]),
                "finite",
            ),
            (
                torch.tensor([[1.1, -0.1]]),
                torch.tensor([0]),
                r"\[0, 1\]",
            ),
            (
                torch.tensor([[0.4, 0.4]]),
                torch.tensor([0]),
                "sum to one",
            ),
            (
                torch.tensor([[0.4, 0.6]]),
                torch.tensor([0.0]),
                "integer",
            ),
            (
                torch.tensor([[0.4, 0.6]]),
                torch.tensor([2]),
                "binary",
            ),
        )
        for posterior, targets, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                provider.compute(BinaryRCNWeightInput(posterior, targets))

        with self.assertRaisesRegex(ValueError, "same device"):
            provider.compute(BinaryRCNWeightInput(
                torch.tensor([[0.4, 0.6]]),
                torch.empty(1, dtype=torch.long, device="meta"),
            ))

    def test_negative_weights_fail_but_roundoff_scale_negative_is_clamped(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        with self.assertRaisesRegex(ValueError, "negative importance weight"):
            provider.compute(BinaryRCNWeightInput(
                posterior_probabilities=torch.tensor(
                    [[0.1, 0.9]], dtype=torch.float64
                ),
                observed_targets=torch.tensor([0]),
            ))

        result = provider.compute(BinaryRCNWeightInput(
            posterior_probabilities=torch.tensor(
                [[0.2 - 1e-8, 0.8 + 1e-8]], dtype=torch.float64
            ),
            observed_targets=torch.tensor([0]),
        ))
        self.assertEqual(result.sample_weights.item(), 0.0)

    def test_output_weights_are_detached(self):
        posterior = torch.tensor(
            [[0.8, 0.2], [0.3, 0.7]],
            requires_grad=True,
        )
        result = BinaryRCNImportanceWeightProvider(0.2, 0.1).compute(
            BinaryRCNWeightInput(posterior, torch.tensor([0, 1]))
        )
        self.assertIs(result.sample_weights.requires_grad, False)

    def test_weight_contribution_adapter_uses_all_true_mask_and_metrics(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        contribution = WeightContributionAdapter(provider).resolve(
            BinaryRCNWeightInput(
            posterior_probabilities=torch.tensor([
                [0.8, 0.2],
                [0.3, 0.7],
            ]),
            observed_targets=torch.tensor([0, 1]),
        ))

        self.assertTrue(torch.equal(
            contribution.selected_mask,
            torch.ones(2, dtype=torch.bool),
        ))
        self.assertEqual(
            set(contribution.metrics),
            {"weight_mean", "weight_min", "weight_max", "zero_weight_ratio"},
        )

    def test_weight_contribution_adapter_rejects_non_float_or_nonfinite_metrics(self):
        class InvalidMetricProvider:
            def __init__(self, value):
                self.value = value

            def compute(self, weight_input):
                return WeightResult(
                    sample_weights=torch.ones(1),
                    metrics={"invalid": self.value},
                )

        weight_input = BinaryRCNWeightInput(
            posterior_probabilities=torch.tensor([[0.4, 0.6]]),
            observed_targets=torch.tensor([1]),
        )
        with self.assertRaisesRegex(TypeError, "Python float"):
            WeightContributionAdapter(InvalidMetricProvider(1)).resolve(
                weight_input
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            WeightContributionAdapter(
                InvalidMetricProvider(float("nan"))
            ).resolve(weight_input)

    def test_batch_mean_weighted_loss_value_and_gradient_match_paper_objective(self):
        class FixedWeightProvider:
            def compute(self, weight_input):
                return WeightResult(
                    sample_weights=torch.tensor([2.0, 4.0]),
                    metrics={"weight_mean": 3.0},
                )

        contribution = WeightContributionAdapter(
            FixedWeightProvider()
        ).resolve(BinaryRCNWeightInput(
            posterior_probabilities=torch.tensor([
                [0.8, 0.2],
                [0.3, 0.7],
            ]),
            observed_targets=torch.tensor([0, 1]),
        ))
        losses = torch.tensor([1.0, 3.0], requires_grad=True)

        objective = reduce_per_sample_loss(
            losses,
            contribution,
            ReductionSpec("batch_mean"),
        )
        weight_sum_mean = reduce_per_sample_loss(
            losses,
            contribution,
            ReductionSpec("weight_sum_mean"),
        )

        self.assertEqual(objective.item(), 7.0)
        self.assertNotEqual(objective.item(), weight_sum_mean.item())
        objective.backward()
        self.assertTrue(torch.equal(losses.grad, torch.tensor([1.0, 2.0])))

    def test_adapter_accepts_a_non_posterior_provider_input(self):
        @dataclass(frozen=True)
        class DummyWeightInput:
            scores: torch.Tensor

        class DummyWeightProvider:
            def compute(self, weight_input):
                weights = weight_input.scores.detach()
                return WeightResult(
                    sample_weights=weights,
                    metrics={
                        "weight_mean": float(weights.mean().item()),
                    },
                )

        dummy_input = DummyWeightInput(
            scores=torch.tensor([0.25, 0.75], requires_grad=True)
        )
        contribution = WeightContributionAdapter(
            DummyWeightProvider()
        ).resolve(dummy_input)

        self.assertTrue(torch.equal(
            contribution.selected_mask,
            torch.tensor([True, True]),
        ))
        self.assertTrue(torch.equal(
            contribution.sample_weights,
            torch.tensor([0.25, 0.75]),
        ))
        self.assertIs(contribution.sample_weights.requires_grad, False)
        self.assertEqual(contribution.metrics, {"weight_mean": 0.5})


if __name__ == "__main__":
    unittest.main()
