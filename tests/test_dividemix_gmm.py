from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from lnl_toolbox.estimators import (
    DivideMixGMMCleanProbabilityEstimator,
    DivideMixGMMLossInput,
    ReliabilityEstimator,
    validate_reliability_result,
)


def _separated_input(
    *,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> DivideMixGMMLossInput:
    return DivideMixGMMLossInput(
        per_sample_losses=torch.tensor(
            [0.10, 0.13, 0.16, 1.80, 2.00, 2.20],
            device=device,
            requires_grad=requires_grad,
        ),
        sample_indices=torch.tensor(
            [40, 10, 70, 20, 90, 30],
            device=device,
        ),
    )


class DivideMixGMMTest(unittest.TestCase):
    def test_estimator_import_is_safe_without_sklearn_until_estimate(self):
        repository_root = Path(__file__).resolve().parents[1]
        script = r'''
import builtins

real_import = builtins.__import__

def block_sklearn(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "sklearn" or name.startswith("sklearn."):
        raise ModuleNotFoundError("blocked sklearn for isolated test")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_sklearn

import lnl_toolbox.estimators
from lnl_toolbox.estimators import (
    DivideMixGMMCleanProbabilityEstimator,
    DivideMixGMMLossInput,
    ReliabilityEstimator,
    ReliabilityResult,
    StatisticResult,
)
import torch

assert ReliabilityEstimator is not None
assert ReliabilityResult is not None
assert StatisticResult is not None
estimator = DivideMixGMMCleanProbabilityEstimator()
estimator_input = DivideMixGMMLossInput(
    per_sample_losses=torch.tensor([0.1, 1.0]),
    sample_indices=torch.tensor([0, 1]),
)
try:
    estimator.estimate(estimator_input)
except ImportError as error:
    assert type(error) is ImportError
    message = str(error)
    assert "DivideMix GMM" in message
    assert "optional training dependency" in message
    assert 'python -m pip install -e ".[train]"' in message
else:
    raise AssertionError("estimate unexpectedly succeeded without sklearn")

print("lazy-import-ok")
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository_root / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertEqual(completed.stdout.strip(), "lazy-import-ok")
        self.assertNotIn("ModuleNotFoundError", completed.stderr)

    def test_pyproject_has_one_parseable_optional_dependency_table(self):
        repository_root = Path(__file__).resolve().parents[1]
        pyproject_path = repository_root / "pyproject.toml"
        source = pyproject_path.read_text(encoding="utf-8")

        self.assertEqual(
            source.count("[project.optional-dependencies]"),
            1,
        )
        parsed = tomllib.loads(source)
        optional_dependencies = parsed["project"]["optional-dependencies"]
        self.assertIn("train", optional_dependencies)
        self.assertIn(
            "scikit-learn>=1.7,<2",
            optional_dependencies["train"],
        )

    def test_separated_clusters_return_lower_loss_clean_probabilities(self):
        estimator = DivideMixGMMCleanProbabilityEstimator(random_seed=17)

        result = estimator.estimate(_separated_input())

        self.assertIsInstance(estimator, ReliabilityEstimator)
        self.assertEqual(result.sample_indices.tolist(), [40, 10, 70, 20, 90, 30])
        self.assertGreater(float(result.scores[:3].min()), 0.9)
        self.assertLess(float(result.scores[3:].max()), 0.1)
        self.assertLess(
            result.metrics["clean_component_mean"],
            result.metrics["noisy_component_mean"],
        )
        self.assertGreater(result.metrics["mean_separation"], 1e-6)
        for value in result.metrics.values():
            self.assertIs(type(value), float)
            self.assertTrue(math.isfinite(value))

    def test_output_is_float64_detached_finite_bounded_and_aligned(self):
        estimator_input = _separated_input(requires_grad=True)

        result = DivideMixGMMCleanProbabilityEstimator().estimate(
            estimator_input
        )

        indices, scores = validate_reliability_result(
            result,
            expected_sample_indices=estimator_input.sample_indices,
        )
        self.assertTrue(torch.equal(indices, estimator_input.sample_indices))
        self.assertEqual(scores.dtype, torch.float64)
        self.assertEqual(scores.device, estimator_input.per_sample_losses.device)
        self.assertIs(scores.requires_grad, False)
        self.assertTrue(bool(torch.isfinite(scores).all().item()))
        self.assertTrue(bool(((scores >= 0) & (scores <= 1)).all().item()))
        self.assertFalse(hasattr(result, "selected_mask"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_output_returns_to_cuda_input_device_as_float64(self):
        estimator_input = _separated_input(device="cuda")

        result = DivideMixGMMCleanProbabilityEstimator().estimate(
            estimator_input
        )

        self.assertEqual(result.scores.device.type, "cuda")
        self.assertEqual(result.scores.dtype, torch.float64)
        self.assertTrue(torch.equal(
            result.sample_indices, estimator_input.sample_indices
        ))

    def test_permutation_preserves_index_probability_mapping(self):
        estimator = DivideMixGMMCleanProbabilityEstimator(random_seed=23)
        original_input = _separated_input()
        original = estimator.estimate(original_input)
        permutation = torch.tensor([4, 0, 5, 2, 1, 3])
        permuted_input = DivideMixGMMLossInput(
            per_sample_losses=original_input.per_sample_losses[permutation],
            sample_indices=original_input.sample_indices[permutation],
        )

        permuted = estimator.estimate(permuted_input)

        original_by_index = {
            int(index): float(score)
            for index, score in zip(
                original.sample_indices.tolist(), original.scores.tolist()
            )
        }
        for index, score in zip(
            permuted.sample_indices.tolist(), permuted.scores.tolist()
        ):
            self.assertAlmostEqual(
                float(score), original_by_index[int(index)], places=12
            )
        self.assertTrue(torch.equal(
            permuted.sample_indices, permuted_input.sample_indices
        ))

    def test_fixed_seed_is_reproducible(self):
        estimator_input = _separated_input()

        first = DivideMixGMMCleanProbabilityEstimator(
            random_seed=101
        ).estimate(estimator_input)
        second = DivideMixGMMCleanProbabilityEstimator(
            random_seed=101
        ).estimate(estimator_input)

        self.assertTrue(torch.equal(first.scores, second.scores))
        self.assertEqual(first.metrics, second.metrics)

    def test_constant_and_single_sample_losses_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive range"):
            DivideMixGMMCleanProbabilityEstimator().estimate(
                DivideMixGMMLossInput(
                    per_sample_losses=torch.ones(4),
                    sample_indices=torch.arange(4),
                )
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            DivideMixGMMCleanProbabilityEstimator().estimate(
                DivideMixGMMLossInput(
                    per_sample_losses=torch.tensor([0.2]),
                    sample_indices=torch.tensor([5]),
                )
            )

    def test_invalid_loss_values_and_empty_input_are_rejected(self):
        cases = (
            (torch.tensor([]), torch.tensor([], dtype=torch.long), "at least two"),
            (
                torch.tensor([0.1, float("nan")]),
                torch.tensor([0, 1]),
                "finite",
            ),
            (
                torch.tensor([0.1, float("inf")]),
                torch.tensor([0, 1]),
                "finite",
            ),
            (
                torch.tensor([1, 2]),
                torch.tensor([0, 1]),
                "floating-point",
            ),
        )
        for losses, indices, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                DivideMixGMMCleanProbabilityEstimator().estimate(
                    DivideMixGMMLossInput(losses, indices)
                )

    def test_invalid_indices_are_rejected(self):
        cases = (
            (torch.tensor([0]), "same one-dimensional shape"),
            (torch.tensor([0.0, 1.0]), "integer dtype"),
            (torch.tensor([1, 1]), "unique"),
        )
        for indices, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                DivideMixGMMCleanProbabilityEstimator().estimate(
                    DivideMixGMMLossInput(
                        per_sample_losses=torch.tensor([0.1, 1.0]),
                        sample_indices=indices,
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_mismatched_input_devices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same device"):
            DivideMixGMMCleanProbabilityEstimator().estimate(
                DivideMixGMMLossInput(
                    per_sample_losses=torch.tensor([0.1, 1.0], device="cuda"),
                    sample_indices=torch.tensor([0, 1]),
                )
            )

    def test_mean_separation_uses_explicit_tolerance(self):
        estimator = DivideMixGMMCleanProbabilityEstimator(
            minimum_mean_separation=1.0
        )

        with self.assertRaisesRegex(ValueError, "sufficiently separated"):
            estimator.estimate(_separated_input())

    def test_nonconverged_fit_fails_explicitly(self):
        estimator = DivideMixGMMCleanProbabilityEstimator(
            max_iter=1,
            tolerance=1e-15,
        )

        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            estimator.estimate(_separated_input())

    def test_constructor_rejects_invalid_configuration(self):
        cases = (
            ({"random_seed": True}, TypeError, "random_seed"),
            ({"max_iter": 0}, ValueError, "positive"),
            ({"tolerance": 0.0}, ValueError, "greater"),
            (
                {"covariance_regularization": -1.0},
                ValueError,
                "at least",
            ),
            (
                {"minimum_mean_separation": float("nan")},
                ValueError,
                "finite",
            ),
            (
                {"minimum_mean_separation": -1.0},
                ValueError,
                "at least",
            ),
        )
        for kwargs, error_type, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                error_type, message
            ):
                DivideMixGMMCleanProbabilityEstimator(**kwargs)


if __name__ == "__main__":
    unittest.main()
