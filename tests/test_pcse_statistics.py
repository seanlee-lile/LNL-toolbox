from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from lnl_toolbox.algorithms.pcse.gda import (
    GDALayer,
    fit_ensemble_weights,
    fit_gda_layers,
)
from lnl_toolbox.algorithms.pcse.statistics import (
    PCSELayerStatistics,
    PCSEStatistics,
    build_coefficient_matrix,
    estimate_pcse_statistics,
    recover_clean_priors,
)
from lnl_toolbox.training.snapshots import FeatureSnapshot


def _snapshots() -> tuple[FeatureSnapshot, FeatureSnapshot]:
    features = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=np.float64,
    )
    targets = np.repeat(np.arange(3, dtype=np.int64), 2)
    indices = np.array([50, 10, 41, 9, 33, 2], dtype=np.int64)
    first = FeatureSnapshot(
        features, targets, indices, "synthetic", "train:hidden1"
    )
    second = FeatureSnapshot(
        features * np.array([2.0, 0.5]),
        targets,
        indices,
        "synthetic",
        "train:hidden2",
    )
    return first, second


class PCSEStatisticsTest(unittest.TestCase):
    def test_identity_transition_recovers_empirical_statistics(self) -> None:
        snapshots = _snapshots()
        statistics = estimate_pcse_statistics(
            snapshots, ("hidden1", "hidden2"), np.eye(3)
        )
        np.testing.assert_allclose(statistics.noisy_priors, np.full(3, 1 / 3))
        np.testing.assert_allclose(statistics.clean_priors, np.full(3, 1 / 3))
        np.testing.assert_allclose(statistics.coefficient_matrix, np.eye(3))
        for layer, snapshot in zip(statistics.layers, snapshots):
            expected_means = np.stack(
                [
                    snapshot.features[snapshot.noisy_targets == class_index].mean(
                        axis=0
                    )
                    for class_index in range(3)
                ]
            )
            expected_second = np.stack(
                [
                    np.einsum("ni,nj->ij", values, values) / len(values)
                    for class_index in range(3)
                    for values in [
                        snapshot.features[
                            snapshot.noisy_targets == class_index
                        ]
                    ]
                ]
            )
            np.testing.assert_allclose(layer.clean_means, expected_means)
            np.testing.assert_allclose(
                layer.clean_second_moments, expected_second
            )
            np.testing.assert_allclose(
                layer.clean_covariances,
                expected_second
                - np.einsum("ci,cj->cij", expected_means, expected_means),
            )

    def test_transition_orientation_and_clean_prior_solve(self) -> None:
        transition = np.array(
            [
                [0.8, 0.2, 0.0],
                [0.1, 0.7, 0.2],
                [0.0, 0.3, 0.7],
            ]
        )
        clean = np.array([0.2, 0.3, 0.5])
        noisy = transition.T @ clean
        recovered = recover_clean_priors(noisy, transition)
        np.testing.assert_allclose(recovered, clean)
        wrong_orientation = np.linalg.solve(transition, noisy)
        self.assertFalse(np.allclose(wrong_orientation, clean))

    def test_oracle_transition_recovers_constructed_clean_mean(self) -> None:
        transition = np.array(
            [
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
            ]
        )
        means = np.array([[-2.0, 0.0], [0.0, 2.0], [2.0, -1.0]])
        clean_priors = np.full(3, 1 / 3)
        noisy_priors = transition.T @ clean_priors
        coefficient = build_coefficient_matrix(clean_priors, transition)
        # Rearranged paper Eq. (14):
        # U_noisy = U_clean Lambda M Lambda_noisy^-1.
        noisy_means = (
            means.T
            @ np.diag(clean_priors)
            @ coefficient
            @ np.diag(1.0 / noisy_priors)
        ).T
        clean_covariances = np.repeat(
            (10.0 * np.eye(2))[None, :, :], 3, axis=0
        )
        clean_second = clean_covariances + np.einsum(
            "ci,cj->cij", means, means
        )
        noisy_second = np.einsum(
            "iab,ij,i,j->jab",
            clean_second,
            coefficient,
            clean_priors,
            1.0 / noisy_priors,
        )
        values: list[np.ndarray] = []
        labels: list[int] = []
        for class_index in range(3):
            covariance = noisy_second[class_index] - np.outer(
                noisy_means[class_index], noisy_means[class_index]
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            self.assertGreater(float(eigenvalues.min()), 0.0)
            for axis in range(2):
                offset = (
                    np.sqrt(2.0 * eigenvalues[axis])
                    * eigenvectors[:, axis]
                )
                values.extend(
                    [
                        noisy_means[class_index] - offset,
                        noisy_means[class_index] + offset,
                    ]
                )
                labels.extend([class_index, class_index])
        features = np.asarray(values)
        targets = np.asarray(labels, dtype=np.int64)
        indices = np.arange(len(targets), dtype=np.int64)[::-1]
        snapshots = (
            FeatureSnapshot(features, targets, indices, "oracle", "train:h1"),
            FeatureSnapshot(
                2.0 * features, targets, indices, "oracle", "train:h2"
            ),
        )
        statistics = estimate_pcse_statistics(
            snapshots, ("h1", "h2"), transition
        )
        np.testing.assert_allclose(
            statistics.layers[0].clean_means, means, atol=1e-12
        )
        np.testing.assert_allclose(
            statistics.layers[0].clean_second_moments,
            clean_second,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            statistics.layers[0].clean_covariances,
            clean_covariances,
            atol=1e-11,
        )

    def test_coefficient_matrix_matches_equation_19(self) -> None:
        transition = np.array(
            [
                [0.8, 0.2, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.25, 0.75],
            ]
        )
        priors = np.array([0.2, 0.3, 0.5])
        expected = np.zeros((3, 3))
        identity = np.eye(3)
        for clean_class in range(3):
            for noisy_class in range(3):
                permutation = identity.copy()
                permutation[[clean_class, noisy_class]] = permutation[
                    [noisy_class, clean_class]
                ]
                expected += (
                    priors[clean_class]
                    * transition[clean_class, noisy_class]
                    * permutation.T
                )
        np.testing.assert_allclose(
            build_coefficient_matrix(priors, transition), expected
        )

    def test_singular_transition_is_rejected(self) -> None:
        transition = np.full((3, 3), 1 / 3)
        with self.assertRaisesRegex(ValueError, "singular"):
            estimate_pcse_statistics(
                _snapshots(), ("hidden1", "hidden2"), transition
            )

    def test_singular_coefficient_matrix_is_rejected(self) -> None:
        with mock.patch(
            "lnl_toolbox.algorithms.pcse.statistics."
            "build_coefficient_matrix",
            return_value=np.ones((3, 3)),
        ):
            with self.assertRaisesRegex(ValueError, "coefficient matrix M"):
                estimate_pcse_statistics(
                    _snapshots(), ("hidden1", "hidden2"), np.eye(3)
                )

    def test_negative_clean_prior_is_rejected(self) -> None:
        transition = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.1, 0.9],
            ]
        )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            recover_clean_priors(
                np.array([0.01, 0.98, 0.01]), transition
            )

    def test_missing_observed_class_is_rejected(self) -> None:
        snapshots = list(_snapshots())
        modified = []
        for snapshot in snapshots:
            modified.append(
                FeatureSnapshot(
                    snapshot.features[:4],
                    np.array([0, 0, 1, 1]),
                    snapshot.global_indices[:4],
                    snapshot.dataset,
                    snapshot.split,
                )
            )
        with self.assertRaisesRegex(ValueError, "missing observed classes"):
            estimate_pcse_statistics(
                modified, ("hidden1", "hidden2"), np.eye(3)
            )

    def test_non_psd_shared_covariance_is_rejected(self) -> None:
        layer = PCSELayerStatistics(
            name="hidden",
            noisy_means=np.zeros((3, 2)),
            noisy_second_moments=np.zeros((3, 2, 2)),
            clean_means=np.zeros((3, 2)),
            clean_second_moments=np.zeros((3, 2, 2)),
            clean_covariances=np.repeat(
                np.diag([-1.0, 1.0])[None, :, :], 3, axis=0
            ),
        )
        statistics = PCSEStatistics(
            noisy_priors=np.full(3, 1 / 3),
            clean_priors=np.full(3, 1 / 3),
            coefficient_matrix=np.eye(3),
            transition_condition=1.0,
            coefficient_condition=1.0,
            layers=(layer, layer.__class__(
                name="hidden2",
                noisy_means=layer.noisy_means,
                noisy_second_moments=layer.noisy_second_moments,
                clean_means=layer.clean_means,
                clean_second_moments=layer.clean_second_moments,
                clean_covariances=layer.clean_covariances,
            )),
        )
        with self.assertRaisesRegex(ValueError, "non-PSD"):
            fit_gda_layers(statistics, covariance_ridge=2.0)

    def test_gda_scores_match_hand_calculation(self) -> None:
        means = np.eye(3, dtype=np.float64)
        gda = GDALayer(
            name="hidden",
            clean_priors=np.full(3, 1 / 3),
            means=means,
            shared_covariance=np.eye(3),
            covariance_ridge=0.0,
        )
        point = np.array([[2.0, 0.5, -1.0]])
        expected = point @ means.T - 0.5 + np.log(1 / 3)
        np.testing.assert_allclose(gda.scores(point), expected)

    def test_ensemble_weights_are_positive_simplex(self) -> None:
        probabilities = np.array(
            [
                [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
            ]
        )
        raw, _, losses = fit_ensemble_weights(
            probabilities,
            np.array([0, 1, 2]),
            epochs=5,
            learning_rate=0.1,
        )
        weights = torch.softmax(raw.detach(), dim=0)
        self.assertTrue(bool((weights > 0).all().item()))
        self.assertAlmostEqual(float(weights.sum().item()), 1.0)
        self.assertTrue(all(np.isfinite(losses)))


if __name__ == "__main__":
    unittest.main()
