from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from lnl_toolbox.algorithms.instance_transition import InstanceTransitionClassificationAlgorithm
from lnl_toolbox.algorithms.transition_risk import (
    forward_instance_corrected_losses,
    instance_importance_reweighted_losses,
)
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.noise import (
    PartTransitionArtifact,
    PosteriorSnapshot,
    fit_part_representation,
    fit_part_transition_matrices,
    generate_pdl_idn,
    select_anchor_candidates,
)


class PDLTest(unittest.TestCase):
    def test_generator_is_deterministic_and_row_stochastic(self) -> None:
        inputs = np.arange(60, dtype=np.float64).reshape(10, 6) / 60.0
        labels = np.arange(10) % 3
        first = generate_pdl_idn(inputs, labels, 3, 0.4, 17, "toy")
        second = generate_pdl_idn(inputs, labels, 3, 0.4, 17, "toy")
        np.testing.assert_array_equal(first.noisy_targets, second.noisy_targets)
        np.testing.assert_allclose(first.per_sample_transition.sum(axis=1), 1.0)
        self.assertTrue((first.per_sample_transition >= 0.0).all())
        self.assertEqual(first.metadata["generator"], "pdl_algorithm_2")

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
        self.assertLess(np.mean((coefficients @ parts.T - features) ** 2), 1e-6)

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
