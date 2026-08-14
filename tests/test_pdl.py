from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from lnl_toolbox.algorithms.instance_transition import InstanceTransitionClassificationAlgorithm
from lnl_toolbox.algorithms.instance_transition import pdl_instance_corrected_losses
from lnl_toolbox.algorithms.transition_risk import (
    forward_instance_corrected_losses,
    instance_importance_reweighted_losses,
)
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.models.cifar_resnet import cifar_resnet34
from lnl_toolbox.noise import (
    PartTransitionArtifact,
    PartTransitionEstimator,
    PosteriorSnapshot,
    fit_part_representation,
    fit_part_transition_matrices,
    generate_pdl_idn,
    select_anchor_candidates,
)
from lnl_toolbox.training.snapshots import FeatureSnapshot
from lnl_toolbox.training.instance_transition_experiment import (
    _official_pdl_split,
    _pdl_official_raw_features,
)


class PDLTest(unittest.TestCase):
    def test_official_raw_features_convert_hwc_to_flattened_chw(self) -> None:
        image = np.zeros((1, 32, 32, 3), dtype=np.float64)
        image[0, 0, 0] = [1.0, 2.0, 3.0]
        actual = _pdl_official_raw_features(image)
        expected = np.transpose(image, (0, 3, 1, 2)).reshape(1, -1)
        np.testing.assert_array_equal(actual, expected)

    def test_official_cifar_resnet34_stem_setting_is_available(self) -> None:
        model = cifar_resnet34(
            10, initialization="torch_default", stem_padding=0
        )
        self.assertEqual(model.stem[0].padding, (0, 0))

    def test_generator_is_deterministic_and_row_stochastic(self) -> None:
        inputs = np.arange(60, dtype=np.float64).reshape(10, 6) / 60.0
        labels = np.arange(10) % 3
        first = generate_pdl_idn(inputs, labels, 3, 0.4, 17, "toy")
        second = generate_pdl_idn(inputs, labels, 3, 0.4, 17, "toy")
        np.testing.assert_array_equal(first.noisy_targets, second.noisy_targets)
        np.testing.assert_allclose(first.per_sample_transition.sum(axis=1), 1.0)
        self.assertTrue((first.per_sample_transition >= 0.0).all())
        self.assertEqual(first.metadata["generator"], "pdl_algorithm_2")

    def test_generator_matches_official_rng_and_softmax_order(self) -> None:
        from scipy import stats
        import torch.nn.functional as F

        inputs = np.arange(30, dtype=np.float64).reshape(5, 6) / 30.0
        labels = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        actual = generate_pdl_idn(inputs, labels, 3, 0.4, 17, "toy")

        np.random.seed(17)
        import torch

        torch.manual_seed(17)
        flip_rate = stats.truncnorm(
            (0.0 - 0.4) / 0.1, (1.0 - 0.4) / 0.1, loc=0.4, scale=0.1,
        ).rvs(labels.shape[0])
        weights = torch.as_tensor(np.random.randn(3, 6, 3), dtype=torch.float32)
        probabilities = []
        for position, clean_class in enumerate(labels):
            scores = torch.as_tensor(inputs[position], dtype=torch.float32).reshape(1, -1).mm(
                weights[int(clean_class)]
            ).squeeze(0)
            scores[int(clean_class)] = -float("inf")
            scores = float(flip_rate[position]) * F.softmax(scores, dim=0)
            scores[int(clean_class)] += 1.0 - float(flip_rate[position])
            probabilities.append(scores)
        reference_probabilities = torch.stack(probabilities).numpy()
        reference_labels = np.asarray([
            np.random.choice([0, 1, 2], p=reference_probabilities[position])
            for position in range(labels.size)
        ])
        np.testing.assert_array_equal(actual.noisy_targets, reference_labels)
        np.testing.assert_allclose(actual.per_sample_transition, reference_probabilities)

    def test_anchor_candidates_break_ties_by_global_index(self) -> None:
        snapshot = PosteriorSnapshot(
            np.array([[0.8, 0.2], [0.8, 0.2], [0.1, 0.9]]),
            np.array([0, 0, 1]), np.array([9, 2, 7]), "toy", "train",
        )
        positions, indices = select_anchor_candidates(snapshot, 2)
        np.testing.assert_array_equal(indices[0], [2, 9])
        np.testing.assert_array_equal(indices[1], [7, 2])
        np.testing.assert_array_equal(snapshot.global_indices[positions], indices)

    def test_part_representation_uses_simplex_coefficients(self) -> None:
        features = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        parts, coefficients = fit_part_representation(features, 2, seed=3, iterations=300)
        self.assertEqual(parts.shape, (2, 2))
        self.assertEqual(coefficients.shape, (3, 2))
        np.testing.assert_allclose(coefficients.sum(axis=1), 1.0)
        self.assertTrue((coefficients >= 0.0).all())
        self.assertTrue((parts >= 0.0).all())
        self.assertTrue(np.isfinite((coefficients @ parts.T - features)).all())

    def test_official_nmf_matches_train_m_global_rng_and_final_normalization(self) -> None:
        features = np.array(
            [[0.9, 0.1, 0.3], [0.2, 0.8, 0.4], [0.6, 0.4, 0.7]],
            dtype=np.float64,
        )
        iterations = 4
        np.random.seed(23)
        weights = np.random.random((features.shape[0], 2))
        latent = np.random.random((2, features.shape[1]))
        for _ in range(iterations):
            error = features - weights @ latent
            if float(np.square(error).sum()) < 1.0e-5:
                break
            latent *= np.divide(
                weights.T @ features,
                (weights.T @ weights) @ latent,
                out=np.zeros_like(latent),
                where=((weights.T @ weights) @ latent) != 0.0,
            )
            weights *= np.divide(
                features @ latent.T,
                (weights @ latent) @ latent.T,
                out=np.zeros_like(weights),
                where=((weights @ latent) @ latent.T) != 0.0,
            )
        weights /= weights.sum(axis=1, keepdims=True)

        np.random.seed(23)
        actual_parts, actual_coefficients = fit_part_representation(
            features, 2, seed=None, iterations=iterations, error_tolerance=1.0e-5
        )
        np.testing.assert_allclose(actual_parts, latent.T)
        np.testing.assert_allclose(actual_coefficients, weights)

    def test_official_basis_reset_matches_init_params_scale(self) -> None:
        from torch.nn import functional as F
        from lnl_toolbox.noise.pdl import fit_pdl_basis_matrices

        coefficients = np.array([
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.2, 0.8], [0.1, 0.9]],
        ])
        targets = coefficients.copy()
        actual = fit_pdl_basis_matrices(
            coefficients, targets, epochs=2, loss_threshold=0.0, seed=29
        )

        torch.manual_seed(29)
        weights = torch.nn.Parameter(torch.empty((2, 2), dtype=torch.float32))
        torch.nn.init.normal_(weights, mean=0.0, std=0.1)
        optimizer = torch.optim.Adam([weights], lr=0.001)
        expected = np.empty((2, 2, 2), dtype=np.float64)
        for class_index in range(2):
            torch.nn.init.normal_(weights, mean=0.0, std=1e-1)
            for _ in range(2):
                loss_total = torch.zeros(())
                for basis_index in range(2):
                    with torch.no_grad():
                        weights.copy_(
                            weights.abs()
                            / weights.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
                        )
                    prediction = (
                        torch.as_tensor(
                            coefficients[class_index, basis_index], dtype=torch.float32
                        )[:, None]
                        * weights
                    ).sum(dim=0)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.mse_loss(
                        prediction,
                        torch.as_tensor(targets[class_index, basis_index], dtype=torch.float32),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_total = loss_total + loss.detach()
            with torch.no_grad():
                expected[:, class_index, :] = (
                    weights.abs()
                    / weights.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
                ).numpy()
        np.testing.assert_allclose(actual, expected)

    def test_official_split_matches_data_py_and_consumes_global_rng(self) -> None:
        np.random.seed(19)
        expected_train = np.random.choice(12, 8, replace=False)
        expected_validation = np.delete(np.arange(12), expected_train)
        expected_next = np.random.random()
        np.random.seed(19)
        actual_train, actual_validation = _official_pdl_split(12, 4, 19)
        np.testing.assert_array_equal(actual_train, expected_train)
        np.testing.assert_array_equal(actual_validation, expected_validation)
        actual_next = np.random.random()
        self.assertEqual(actual_next, expected_next)

    def test_official_anchor_and_basis_shapes(self) -> None:
        from lnl_toolbox.noise.pdl import (
            fit_pdl_basis_matrices,
            select_pdl_anchor_candidates,
        )

        probabilities = np.array([
            [0.99, 0.01], [0.98, 0.02], [0.02, 0.98], [0.01, 0.99],
        ])
        positions = select_pdl_anchor_candidates(probabilities, [97.0, 99.0])
        self.assertEqual(positions.shape, (2, 2))
        coefficients = np.array([
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.2, 0.8], [0.1, 0.9]],
        ])
        targets = np.array([
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.2, 0.8], [0.1, 0.9]],
        ])
        matrices = fit_pdl_basis_matrices(
            coefficients, targets, epochs=3, loss_threshold=0.0, seed=4
        )
        self.assertEqual(matrices.shape, (2, 2, 2))
        np.testing.assert_allclose(matrices.sum(axis=2), 1.0, atol=1e-6)

    def test_eq4_recovers_known_part_matrices(self) -> None:
        part_matrices = np.array([
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.6, 0.4], [0.3, 0.7]],
        ])
        coefficients = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.75, 0.25], [0.25, 0.75]],
        ])
        posteriors = np.empty((2, 2, 2))
        for clean_class in range(2):
            posteriors[clean_class] = np.einsum(
                "kr,rc->kc", coefficients[clean_class], part_matrices[:, clean_class]
            )
        fitted = fit_part_transition_matrices(coefficients, posteriors)
        np.testing.assert_allclose(fitted, part_matrices, atol=1e-10)

    def _artifact(self) -> PartTransitionArtifact:
        return PartTransitionArtifact(
            parts=np.eye(2),
            coefficients=np.array([[1.0, 0.0], [0.25, 0.75]]),
            part_matrices=np.array([
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.6, 0.4], [0.3, 0.7]],
            ]),
            global_indices=np.array([11, 4]),
            feature_snapshot_hash="a" * 64,
            posterior_snapshot_hash="b" * 64,
            anchor_indices=np.array([[4, 11], [11, 4]]),
        )

    def test_artifact_aligns_global_indices_and_round_trips(self) -> None:
        artifact = self._artifact()
        matrices = artifact.transitions_for(torch.tensor([11, 4]), dtype=torch.float64)
        np.testing.assert_allclose(matrices[0].numpy(), artifact.part_matrices[0])
        expected = 0.25 * artifact.part_matrices[0] + 0.75 * artifact.part_matrices[1]
        np.testing.assert_allclose(matrices[1].numpy(), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pdl.npz"
            artifact.save(path)
            restored = PartTransitionArtifact.load(path)
        self.assertEqual(restored.artifact_hash, artifact.artifact_hash)

    def test_official_raw_basis_artifact_clips_only_combined_train_matrix(self) -> None:
        artifact = PartTransitionArtifact(
            parts=np.eye(2),
            coefficients=np.array([[1.0, 0.0]]),
            part_matrices=np.array([[[0.8, 0.3], [-0.1, 0.2]], [[0.2, -0.1], [0.4, 0.6]]]),
            global_indices=np.array([0]),
            feature_snapshot_hash="a" * 64,
            posterior_snapshot_hash="b" * 64,
            anchor_indices=np.array([[0], [0]]),
            metadata={"official_raw_basis": True},
        )
        actual = artifact.transitions_for([0]).numpy()
        np.testing.assert_allclose(actual, [[[0.8, 0.3], [0.0, 0.2]]])

    def test_official_split_estimation_shares_nmf_and_separates_basis_artifacts(self) -> None:
        indices = np.arange(6, dtype=np.int64)
        features = FeatureSnapshot(
            np.array([
                [1.0, 0.0, 0.2], [0.8, 0.2, 0.1], [0.1, 0.9, 0.2],
                [0.2, 0.8, 0.1], [0.6, 0.4, 0.3], [0.3, 0.7, 0.4],
            ]),
            np.array([0, 0, 1, 1, 0, 1]), indices, "toy", "union",
        )
        posteriors = PosteriorSnapshot(
            np.array([
                [0.90, 0.10], [0.75, 0.25], [0.15, 0.85],
                [0.25, 0.75], [0.65, 0.35], [0.35, 0.65],
            ]),
            features.noisy_targets, indices, "toy", "union",
        )
        estimator = PartTransitionEstimator(
            num_parts=2,
            anchor_candidates=2,
            representation_seed=7,
            representation_iterations=2,
            anchor_percentages=(97.0, 99.0),
            basis_epochs=1,
            basis_loss_threshold=0.0,
        )
        parts, coefficients = fit_part_representation(
            features.features, 2, seed=7, iterations=2
        )
        train = FeatureSnapshot(
            features.features[:3], features.noisy_targets[:3], indices[:3], "toy", "train"
        )
        train_posterior = PosteriorSnapshot(
            posteriors.noisy_probabilities[:3], posteriors.noisy_targets[:3], indices[:3],
            "toy", "train",
        )
        validation = FeatureSnapshot(
            features.features[3:], features.noisy_targets[3:], indices[3:], "toy", "validation"
        )
        validation_posterior = PosteriorSnapshot(
            posteriors.noisy_probabilities[3:], posteriors.noisy_targets[3:], indices[3:],
            "toy", "validation",
        )
        train_artifact = estimator.estimate_from_shared_representation(
            train, train_posterior,
            representation_parts=parts,
            representation_coefficients=coefficients,
            representation_indices=indices,
        )
        validation_artifact = estimator.estimate_from_shared_representation(
            validation, validation_posterior,
            representation_parts=parts,
            representation_coefficients=coefficients,
            representation_indices=indices,
        )
        np.testing.assert_allclose(train_artifact.parts, validation_artifact.parts)
        np.testing.assert_allclose(train_artifact.coefficients, coefficients[:3])
        np.testing.assert_allclose(validation_artifact.coefficients, coefficients[3:])
        revision_artifact = validation_artifact.with_part_matrices(
            train_artifact.part_matrices,
            role="revision_validation",
            source_artifact_hash=train_artifact.artifact_hash,
        )
        np.testing.assert_allclose(revision_artifact.parts, validation_artifact.parts)
        np.testing.assert_allclose(
            revision_artifact.coefficients, validation_artifact.coefficients
        )
        np.testing.assert_allclose(
            revision_artifact.part_matrices, train_artifact.part_matrices
        )
        self.assertEqual(revision_artifact.metadata["artifact_role"], "revision_validation")
        self.assertEqual(
            revision_artifact.metadata["source_artifact_hash"], train_artifact.artifact_hash
        )

    def test_forward_and_importance_objectives_match_manual_values(self) -> None:
        logits = torch.tensor([[1.0, -0.5], [-0.2, 0.7]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([0, 1])
        matrices = torch.tensor([
            [[0.8, 0.2], [0.1, 0.9]],
            [[1.0, 0.0], [0.0, 1.0]],
        ], dtype=torch.float64)
        loss = nn.CrossEntropyLoss(reduction="none")
        forward = forward_instance_corrected_losses(logits, targets, matrices, loss)
        clean = torch.softmax(logits, 1)
        noisy = torch.bmm(clean.unsqueeze(1), matrices).squeeze(1)
        expected_forward = -torch.log(noisy.gather(1, targets[:, None]).squeeze(1))
        torch.testing.assert_close(forward, expected_forward)
        importance = instance_importance_reweighted_losses(logits, targets, matrices, loss)
        expected_weight = (
            clean.gather(1, targets[:, None]).squeeze(1)
            / noisy.gather(1, targets[:, None]).squeeze(1)
        ).detach()
        torch.testing.assert_close(importance, loss(logits, targets) * expected_weight)
        (forward.mean() + importance.mean()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_pdl_reweight_objective_matches_official_beta_nll(self) -> None:
        logits = torch.tensor(
            [[1.0, -0.5], [-0.2, 0.7]], dtype=torch.float64, requires_grad=True
        )
        targets = torch.tensor([0, 1])
        matrices = torch.tensor([
            [[0.8, 0.2], [0.1, 0.9]],
            [[1.0, 0.0], [0.0, 1.0]],
        ], dtype=torch.float64)
        clean = torch.softmax(logits, dim=1)
        noisy = torch.bmm(clean.unsqueeze(1), matrices).squeeze(1)
        clean_y = clean.gather(1, targets[:, None]).squeeze(1)
        noisy_y = noisy.gather(1, targets[:, None]).squeeze(1)
        expected = (clean_y / noisy_y) * (-torch.log(clean_y))
        actual = pdl_instance_corrected_losses(logits, targets, matrices)
        torch.testing.assert_close(actual, expected)

        official_logits = logits.detach().clone().requires_grad_(True)
        official_clean = torch.softmax(official_logits, dim=1)
        official_noisy = torch.bmm(
            official_clean.unsqueeze(1), matrices
        ).squeeze(1)
        official_clean_y = official_clean.gather(1, targets[:, None]).squeeze(1)
        official_noisy_y = official_noisy.gather(1, targets[:, None]).squeeze(1)
        official_beta = (official_clean_y / official_noisy_y).detach()
        (official_beta * -torch.log(official_clean_y)).mean().backward()

        correction_logits = logits.detach().clone().requires_grad_(True)
        pdl_instance_corrected_losses(
            correction_logits,
            targets,
            matrices,
            detach_importance_weight=True,
        ).mean().backward()
        torch.testing.assert_close(correction_logits.grad, official_logits.grad)

        revision_logits = logits.detach().clone().requires_grad_(True)
        pdl_instance_corrected_losses(
            revision_logits,
            targets,
            matrices,
            detach_importance_weight=False,
        ).mean().backward()
        self.assertFalse(torch.allclose(revision_logits.grad, official_logits.grad))

    def test_pdl_algorithm_uses_phase_specific_beta_gradients(self) -> None:
        batch = Batch({
            "input": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "target": torch.tensor([0, 1]),
            "index": torch.tensor([11, 4]),
        })
        artifact = self._artifact()
        for correction, expected_detach in (("pdl", True), ("pdl_revision", False)):
            model = nn.Linear(2, 2)
            if correction == "pdl_revision":
                model.T_revision = nn.Linear(2, 2, bias=False)
                nn.init.zeros_(model.T_revision.weight)
            algorithm = InstanceTransitionClassificationAlgorithm(
                model,
                torch.optim.SGD(model.parameters(), lr=0.0),
                nn.CrossEntropyLoss(reduction="none"),
                artifact,
                torch.device("cpu"),
                correction=correction,
            )
            algorithm.setup(ExperimentContext(work_dir=Path(".")))
            with patch(
                "lnl_toolbox.algorithms.instance_transition.pdl_instance_corrected_losses",
                wraps=pdl_instance_corrected_losses,
            ) as corrected_loss:
                algorithm.step(batch, RunState())
            self.assertEqual(
                corrected_loss.call_args.kwargs["detach_importance_weight"],
                expected_detach,
            )

    def test_algorithm_checkpoint_rejects_artifact_change(self) -> None:
        artifact = self._artifact()
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        algorithm = InstanceTransitionClassificationAlgorithm(
            model, optimizer, nn.CrossEntropyLoss(reduction="none"), artifact,
            torch.device("cpu"), correction="forward",
        )
        algorithm.setup(ExperimentContext(work_dir=Path(".")))
        state = RunState()
        algorithm.on_cycle_start(state)
        result = algorithm.step(Batch({
            "input": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "target": torch.tensor([0, 1]),
            "index": torch.tensor([11, 4]),
        }), state)
        self.assertTrue(np.isfinite(result.metrics["loss"]))
        saved = algorithm.state_dict()
        saved["transition_artifact_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "artifact mismatch"):
            algorithm.load_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
