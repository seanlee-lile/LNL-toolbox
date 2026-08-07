from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from lnl_toolbox.noise import (
    AnchorTransitionEstimator,
    DualTransitionEstimator,
    KnownTransitionEstimator,
    PosteriorSnapshot,
    TransitionArtifact,
)
from lnl_toolbox.training.snapshots import collect_posterior_snapshot


class TransitionEstimatorTest(unittest.TestCase):
    def test_collector_sorts_indices_and_restores_model_mode(self) -> None:
        records = [
            {"input": torch.tensor([0.0, 2.0]), "target": 1, "index": 20},
            {"input": torch.tensor([3.0, 0.0]), "target": 0, "index": 10},
            {"input": torch.tensor([0.5, 0.5]), "target": 1, "index": 30},
        ]
        model = torch.nn.Identity()
        model.train()

        snapshot = collect_posterior_snapshot(
            model,
            DataLoader(records, batch_size=2, shuffle=False),
            "cpu",
            dataset="fixture",
            split="train",
        )

        self.assertTrue(model.training)
        np.testing.assert_array_equal(snapshot.global_indices, [10, 20, 30])
        np.testing.assert_array_equal(snapshot.noisy_targets, [0, 1, 1])
        expected_logits = torch.stack(
            [records[1]["input"], records[0]["input"], records[2]["input"]]
        )
        expected = torch.softmax(expected_logits, dim=1).numpy()
        np.testing.assert_allclose(snapshot.noisy_probabilities, expected, atol=1e-7)

    def test_collector_rejects_empty_or_duplicate_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one batch"):
            collect_posterior_snapshot(
                torch.nn.Identity(),
                DataLoader([], batch_size=1),
                "cpu",
                dataset="fixture",
                split="train",
            )

        records = [
            {"input": torch.tensor([1.0, 0.0]), "target": 0, "index": 7},
            {"input": torch.tensor([0.0, 1.0]), "target": 1, "index": 7},
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            collect_posterior_snapshot(
                torch.nn.Identity(),
                DataLoader(records, batch_size=2),
                "cpu",
                dataset="fixture",
                split="train",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_collector_accepts_cuda_model_and_returns_cpu_snapshot(self) -> None:
        records = [
            {"input": torch.tensor([2.0, 0.0]), "target": 0, "index": 1},
            {"input": torch.tensor([0.0, 2.0]), "target": 1, "index": 0},
        ]
        snapshot = collect_posterior_snapshot(
            torch.nn.Identity().cuda(),
            DataLoader(records, batch_size=2),
            "cuda",
            dataset="fixture",
            split="train",
        )
        np.testing.assert_array_equal(snapshot.global_indices, [0, 1])
        self.assertEqual(snapshot.noisy_probabilities.dtype, np.float64)

    @staticmethod
    def _dual_t_snapshot() -> PosteriorSnapshot:
        return PosteriorSnapshot(
            noisy_probabilities=np.asarray(
                [
                    [0.90, 0.05, 0.05],
                    [0.10, 0.80, 0.10],
                    [0.05, 0.05, 0.90],
                    [0.60, 0.20, 0.20],
                    [0.20, 0.60, 0.20],
                    [0.20, 0.20, 0.60],
                ]
            ),
            noisy_targets=np.asarray([0, 1, 2, 1, 2, 0]),
            global_indices=np.asarray([30, 10, 50, 20, 40, 60]),
            dataset="fixture",
            split="train",
        )

    def test_dual_t_matches_paper_factorization(self) -> None:
        snapshot = self._dual_t_snapshot()
        artifact = DualTransitionEstimator().estimate(snapshot)
        t_club = np.asarray(
            [
                [0.90, 0.05, 0.05],
                [0.10, 0.80, 0.10],
                [0.05, 0.05, 0.90],
            ]
        )
        t_spade = np.asarray(
            [
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
            ]
        )

        self.assertEqual(artifact.estimator, "dual_t")
        self.assertEqual(artifact.source_snapshot_hash, snapshot.snapshot_hash)
        self.assertEqual(artifact.metadata["composition"], "t_club @ t_spade")
        np.testing.assert_allclose(artifact.metadata["t_club"], t_club)
        np.testing.assert_allclose(artifact.metadata["t_spade"], t_spade)
        np.testing.assert_allclose(artifact.matrix, t_club @ t_spade)
        self.assertEqual(artifact.metadata["anchor_global_indices"], [30, 10, 50])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dual_t.npz"
            artifact.save(path)
            loaded = TransitionArtifact.load(path)
        self.assertEqual(loaded.artifact_hash, artifact.artifact_hash)
        np.testing.assert_allclose(loaded.metadata["t_club"], t_club)
        np.testing.assert_allclose(loaded.metadata["t_spade"], t_spade)

    def test_dual_t_is_invariant_to_snapshot_order(self) -> None:
        snapshot = self._dual_t_snapshot()
        order = np.asarray([4, 1, 5, 0, 3, 2])
        reordered = PosteriorSnapshot(
            noisy_probabilities=snapshot.noisy_probabilities[order],
            noisy_targets=snapshot.noisy_targets[order],
            global_indices=snapshot.global_indices[order],
            dataset=snapshot.dataset,
            split=snapshot.split,
        )

        first = DualTransitionEstimator().estimate(snapshot)
        second = DualTransitionEstimator().estimate(reordered)
        np.testing.assert_allclose(first.matrix, second.matrix)
        np.testing.assert_allclose(first.metadata["t_club"], second.metadata["t_club"])
        np.testing.assert_allclose(first.metadata["t_spade"], second.metadata["t_spade"])
        self.assertEqual(
            first.metadata["anchor_global_indices"],
            second.metadata["anchor_global_indices"],
        )

    def test_dual_t_rejects_empty_intermediate_class(self) -> None:
        snapshot = PosteriorSnapshot(
            noisy_probabilities=np.asarray(
                [[0.70, 0.20, 0.10], [0.20, 0.70, 0.10], [0.60, 0.30, 0.10]]
            ),
            noisy_targets=np.asarray([0, 1, 2]),
            global_indices=np.asarray([0, 1, 2]),
            dataset="fixture",
            split="train",
        )
        with self.assertRaisesRegex(ValueError, "empty intermediate classes: 2"):
            DualTransitionEstimator().estimate(snapshot)

    def test_dual_t_reduces_to_anchor_with_identity_second_factor(self) -> None:
        probabilities = np.asarray(
            [[0.90, 0.10], [0.80, 0.20], [0.20, 0.80], [0.10, 0.90]]
        )
        snapshot = PosteriorSnapshot(
            noisy_probabilities=probabilities,
            noisy_targets=probabilities.argmax(axis=1),
            global_indices=np.asarray([0, 1, 2, 3]),
            dataset="fixture",
            split="train",
        )
        anchor = AnchorTransitionEstimator().estimate(snapshot)
        dual = DualTransitionEstimator().estimate(snapshot)
        np.testing.assert_allclose(dual.metadata["t_spade"], np.eye(2))
        np.testing.assert_allclose(dual.matrix, anchor.matrix)

    def setUp(self) -> None:
        self.matrix = np.array(
            [
                [0.80, 0.10, 0.10],
                [0.10, 0.75, 0.15],
                [0.05, 0.10, 0.85],
            ],
            dtype=np.float64,
        )
        self.probabilities = np.array(
            [
                self.matrix[0],
                self.matrix[1],
                self.matrix[2],
                [0.50, 0.25, 0.25],
                [0.20, 0.50, 0.30],
            ],
            dtype=np.float64,
        )
        self.targets = np.array([0, 1, 2, 0, 1], dtype=np.int64)
        self.indices = np.array([30, 10, 50, 20, 40], dtype=np.int64)

    def snapshot(
        self,
        probabilities: np.ndarray | None = None,
        targets: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> PosteriorSnapshot:
        return PosteriorSnapshot(
            self.probabilities if probabilities is None else probabilities,
            self.targets if targets is None else targets,
            self.indices if indices is None else indices,
            "cifar10",
            "train",
        )

    def test_anchor_estimator_recovers_transition_and_records_identity(self) -> None:
        snapshot = self.snapshot()
        artifact = AnchorTransitionEstimator().estimate(snapshot)
        np.testing.assert_allclose(artifact.matrix, self.matrix)
        self.assertEqual(artifact.estimator, "anchor")
        self.assertEqual(artifact.source_snapshot_hash, snapshot.snapshot_hash)
        self.assertEqual(
            artifact.metadata["anchor_global_indices"], [30, 10, 50]
        )
        self.assertFalse(artifact.matrix.flags.writeable)

    def test_known_estimator_returns_configured_transition(self) -> None:
        snapshot = self.snapshot()
        artifact = KnownTransitionEstimator(self.matrix).estimate(snapshot)
        np.testing.assert_allclose(artifact.matrix, self.matrix)
        self.assertEqual(artifact.estimator, "known")
        self.assertEqual(artifact.metadata["source"], "configuration")

    def test_known_estimator_rejects_wrong_class_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "classes"):
            KnownTransitionEstimator(np.eye(2)).estimate(self.snapshot())

    def test_input_reordering_does_not_change_estimate(self) -> None:
        original = AnchorTransitionEstimator().estimate(self.snapshot())
        order = np.array([4, 2, 0, 3, 1], dtype=np.int64)
        reordered = AnchorTransitionEstimator().estimate(
            self.snapshot(
                self.probabilities[order],
                self.targets[order],
                self.indices[order],
            )
        )
        np.testing.assert_array_equal(reordered.matrix, original.matrix)
        self.assertEqual(
            reordered.metadata["anchor_global_indices"],
            original.metadata["anchor_global_indices"],
        )

    def test_anchor_tie_uses_smallest_global_index(self) -> None:
        probabilities = np.array(
            [
                [0.80, 0.10, 0.10],
                [0.80, 0.15, 0.05],
                [0.10, 0.80, 0.10],
                [0.10, 0.10, 0.80],
            ]
        )
        snapshot = PosteriorSnapshot(
            probabilities,
            np.array([0, 0, 1, 2]),
            np.array([20, 5, 30, 40]),
            "fixture",
            "train",
        )
        artifact = AnchorTransitionEstimator().estimate(snapshot)
        np.testing.assert_array_equal(artifact.matrix[0], probabilities[1])
        self.assertEqual(artifact.metadata["anchor_global_indices"][0], 5)

    def test_snapshot_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            self.snapshot(probabilities=self.probabilities[:, 0])
        with self.assertRaisesRegex(ValueError, "noisy_targets"):
            self.snapshot(targets=self.targets[:-1])
        with self.assertRaisesRegex(ValueError, "global_indices"):
            self.snapshot(indices=np.array([1, 1, 2, 3, 4]))
        invalid = self.probabilities.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.snapshot(probabilities=invalid)
        invalid = self.probabilities.copy()
        invalid[0] = [0.8, 0.3, -0.1]
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.snapshot(probabilities=invalid)
        invalid = self.probabilities.copy()
        invalid[0] = [0.8, 0.2, 0.2]
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self.snapshot(probabilities=invalid)
        with self.assertRaisesRegex(ValueError, "within"):
            self.snapshot(targets=np.array([0, 1, 3, 0, 1]))

    def test_snapshot_hash_binds_probabilities_targets_and_indices(self) -> None:
        snapshot = self.snapshot()
        changed_targets = self.targets.copy()
        changed_targets[0] = 1
        changed_indices = self.indices.copy()
        changed_indices[0] = 31
        self.assertNotEqual(
            snapshot.snapshot_hash,
            self.snapshot(targets=changed_targets).snapshot_hash,
        )
        self.assertNotEqual(
            snapshot.snapshot_hash,
            self.snapshot(indices=changed_indices).snapshot_hash,
        )

    def test_artifact_roundtrip_and_tamper_detection(self) -> None:
        artifact = AnchorTransitionEstimator().estimate(self.snapshot())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transition.npz"
            artifact.save(path)
            loaded = TransitionArtifact.load(path)
            np.testing.assert_array_equal(loaded.matrix, artifact.matrix)
            self.assertEqual(loaded.artifact_hash, artifact.artifact_hash)

            with np.load(path, allow_pickle=False) as data:
                matrix = data["matrix"].copy()
                metadata_json = data["metadata_json"].copy()
            matrix[0] = [0.70, 0.20, 0.10]
            np.savez_compressed(path, matrix=matrix, metadata_json=metadata_json)
            with self.assertRaisesRegex(ValueError, "hash"):
                TransitionArtifact.load(path)

            artifact.save(path)
            with np.load(path, allow_pickle=False) as data:
                matrix = data["matrix"].copy()
                payload = json.loads(str(data["metadata_json"].item()))
            payload["metadata"]["dataset"] = "tampered"
            np.savez_compressed(
                path,
                matrix=matrix,
                metadata_json=np.array(json.dumps(payload)),
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                TransitionArtifact.load(path)

    def test_artifact_rejects_invalid_matrix_and_convention(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            TransitionArtifact(np.eye(3) * 0.5, "anchor")
        with self.assertRaisesRegex(ValueError, "convention"):
            TransitionArtifact(np.eye(3), "anchor", convention="column")

    def test_artifact_tensor_device_and_dtype(self) -> None:
        artifact = AnchorTransitionEstimator().estimate(self.snapshot())
        cpu = artifact.as_tensor(device="cpu", dtype=torch.float64)
        self.assertEqual(cpu.device.type, "cpu")
        self.assertEqual(cpu.dtype, torch.float64)
        np.testing.assert_allclose(cpu.numpy(), artifact.matrix)
        if torch.cuda.is_available():
            cuda = artifact.as_tensor(device="cuda", dtype=torch.float32)
            self.assertEqual(cuda.device.type, "cuda")
            self.assertEqual(cuda.dtype, torch.float32)
            np.testing.assert_allclose(cuda.cpu().numpy(), artifact.matrix, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
