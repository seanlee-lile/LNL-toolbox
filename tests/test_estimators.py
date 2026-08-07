from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import unittest

import torch

from lnl_toolbox.estimators import (
    ReliabilityEstimator,
    ReliabilityResult,
    StatisticResult,
    validate_reliability_result,
    validate_statistic_result,
)


@dataclass(frozen=True)
class DummyReliabilityInput:
    values: torch.Tensor
    sample_indices: torch.Tensor


class DummyStatelessEstimator:
    def estimate(
        self, estimator_input: DummyReliabilityInput
    ) -> ReliabilityResult:
        return ReliabilityResult(
            sample_indices=estimator_input.sample_indices,
            scores=estimator_input.values.detach(),
            metrics={"score_mean": float(estimator_input.values.mean().item())},
        )


class EstimatorContractTest(unittest.TestCase):
    def test_stateless_estimator_satisfies_protocol_without_state_methods(self):
        estimator = DummyStatelessEstimator()

        self.assertIsInstance(estimator, ReliabilityEstimator)
        self.assertFalse(hasattr(estimator, "state_dict"))
        result = estimator.estimate(
            DummyReliabilityInput(
                values=torch.tensor([0.2, 0.9]),
                sample_indices=torch.tensor([4, 1]),
            )
        )
        indices, scores = validate_reliability_result(result)
        self.assertEqual(indices.tolist(), [4, 1])
        self.assertEqual(scores.tolist(), result.scores.tolist())

    def test_larger_scores_have_the_documented_more_reliable_direction(self):
        result = ReliabilityResult(
            sample_indices=torch.tensor([8, 3, 5]),
            scores=torch.tensor([0.2, 0.9, 0.5]),
        )

        _, scores = validate_reliability_result(result)

        ranked_indices = result.sample_indices[torch.argsort(
            scores, descending=True
        )]
        self.assertEqual(ranked_indices.tolist(), [3, 5, 8])

    def test_reliability_result_is_frozen(self):
        result = ReliabilityResult(
            sample_indices=torch.tensor([0]),
            scores=torch.tensor([1.0]),
        )

        with self.assertRaises(FrozenInstanceError):
            result.scores = torch.tensor([0.0])

    def test_expected_indices_require_exact_order_and_values(self):
        result = ReliabilityResult(
            sample_indices=torch.tensor([7, 2]),
            scores=torch.tensor([0.8, 0.1]),
        )
        validate_reliability_result(
            result, expected_sample_indices=torch.tensor([7, 2])
        )

        with self.assertRaisesRegex(ValueError, "expected order"):
            validate_reliability_result(
                result, expected_sample_indices=torch.tensor([2, 7])
            )

    def test_reliability_validation_rejects_invalid_indices(self):
        cases = (
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([[0, 1]]),
                    scores=torch.tensor([0.1, 0.2]),
                ),
                "one-dimensional",
            ),
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([], dtype=torch.long),
                    scores=torch.tensor([]),
                ),
                "must not be empty",
            ),
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([0.0, 1.0]),
                    scores=torch.tensor([0.1, 0.2]),
                ),
                "integer dtype",
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
                validate_reliability_result(result)

    def test_reliability_validation_rejects_invalid_scores(self):
        cases = (
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([0, 1]),
                    scores=torch.tensor([[0.1, 0.2]]),
                ),
                "one-dimensional",
            ),
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([0, 1]),
                    scores=torch.tensor([0.1]),
                ),
                "one-to-one",
            ),
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([0, 1]),
                    scores=torch.tensor([0, 1]),
                ),
                "floating-point",
            ),
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([0, 1]),
                    scores=torch.tensor([0.1, float("nan")]),
                ),
                "finite",
            ),
            (
                ReliabilityResult(
                    sample_indices=torch.tensor([0, 1]),
                    scores=torch.tensor([0.1, 0.2], requires_grad=True),
                ),
                "detached",
            ),
        )
        for result, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                validate_reliability_result(result)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_reliability_validation_rejects_device_mismatch(self):
        result = ReliabilityResult(
            sample_indices=torch.tensor([0, 1]),
            scores=torch.tensor([0.1, 0.2], device="cuda"),
        )

        with self.assertRaisesRegex(ValueError, "same device"):
            validate_reliability_result(result)

    def test_reliability_metrics_require_finite_python_floats(self):
        cases = (
            ({"count": 2}, TypeError, "Python float"),
            ({"mean": float("inf")}, ValueError, "finite"),
            ({1: 0.5}, TypeError, "names"),
        )
        for metrics, error_type, message in cases:
            with self.subTest(metrics=metrics), self.assertRaisesRegex(
                error_type, message
            ):
                validate_reliability_result(ReliabilityResult(
                    sample_indices=torch.tensor([0]),
                    scores=torch.tensor([0.5]),
                    metrics=metrics,
                ))

    def test_statistic_result_preserves_arbitrary_typed_payload(self):
        @dataclass(frozen=True)
        class CentroidStatistics:
            centroids: torch.Tensor
            class_counts: tuple[int, ...]

        payload = CentroidStatistics(
            centroids=torch.tensor([[1.0, 2.0]]),
            class_counts=(3,),
        )
        result = StatisticResult(
            statistics=payload,
            metrics={"classes": 1.0},
        )

        self.assertIs(validate_statistic_result(result), payload)
        self.assertFalse(hasattr(result, "fit"))
        self.assertFalse(hasattr(result, "compute"))
        self.assertFalse(hasattr(result, "state_dict"))

    def test_statistic_validation_does_not_inspect_payload(self):
        opaque_payload = object()

        self.assertIs(
            validate_statistic_result(StatisticResult(opaque_payload)),
            opaque_payload,
        )

    def test_statistic_validation_checks_only_container_and_metrics(self):
        with self.assertRaisesRegex(TypeError, "StatisticResult"):
            validate_statistic_result({"statistics": object()})
        with self.assertRaisesRegex(TypeError, "Python float"):
            validate_statistic_result(StatisticResult(
                statistics=object(),
                metrics={"count": 1},
            ))


if __name__ == "__main__":
    unittest.main()
