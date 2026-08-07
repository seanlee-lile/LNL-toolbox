from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lnl_toolbox.algorithms.dld import (
    DLDPreCorrectionArtifact,
    PARTITION_CLEAN,
    PARTITION_HARD,
    PARTITION_NOISY,
    construct_y0,
    construct_yn,
    kl_ps_to_pw,
    partition_samples,
    persist_precorrection_atomically,
    weighted_neighbor_distribution,
)


class DLDPreCorrectionTest(unittest.TestCase):
    def test_cosine_knn_uses_stable_indices_and_normalizes(self) -> None:
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        targets = torch.tensor([0, 1, 1])
        indices = torch.tensor([50, 10, 90])
        result = weighted_neighbor_distribution(
            features, features, targets, indices, indices,
            num_classes=2, k=2, metric="cosine_distance", delta=1e-3,
            self_neighbor="include",
        )
        self.assertTrue(torch.allclose(result.probabilities.sum(1), torch.ones(3)))
        self.assertEqual(result.neighbor_indices[0, 0].item(), 50)
        order = torch.tensor([2, 0, 1])
        permuted = weighted_neighbor_distribution(
            features[order], features[order], targets[order], indices[order], indices[order],
            num_classes=2, k=2, metric="cosine_distance", delta=1e-3,
            self_neighbor="include",
        )
        by_index = {int(i): p for i, p in zip(indices, result.probabilities)}
        for index, probability in zip(indices[order], permuted.probabilities):
            self.assertTrue(torch.allclose(probability, by_index[int(index)]))

    def test_kl_direction_and_eq14_eq15(self) -> None:
        pw = torch.tensor([[0.8, 0.2], [0.25, 0.75], [0.6, 0.4]], dtype=torch.float64)
        ps = torch.tensor([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]], dtype=torch.float64)
        expected = (ps * (ps.log() - pw.log())).sum(1)
        self.assertTrue(torch.allclose(kl_ps_to_pw(pw, ps), expected))
        target = torch.tensor([0, 1, 0])
        partition = torch.tensor([PARTITION_CLEAN, PARTITION_NOISY, PARTITION_HARD])
        pws = (pw + ps) / 2
        y0 = construct_y0(pws, target, partition)
        yn = construct_yn(pw, ps, target, partition)
        self.assertTrue(torch.equal(y0[0], torch.tensor([1.0, 0.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(yn[0], torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(yn[1], torch.tensor([0.0, 1.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(yn[2], torch.tensor([0.5, 0.5], dtype=torch.float64)))
        self.assertTrue(torch.allclose(yn - y0, yn - y0))

    def test_hard_zero_denominator_fails(self) -> None:
        p = torch.tensor([[0.5, 0.5]])
        with self.assertRaisesRegex(ValueError, "denominator"):
            construct_yn(p, p, torch.tensor([0]), torch.tensor([PARTITION_HARD]))

    def test_gmm_high_mean_component_is_hard(self) -> None:
        pw = torch.tensor([[0.99, 0.01]] * 8 + [[0.99, 0.01]] * 8, dtype=torch.float64)
        ps = torch.tensor([[0.98, 0.02]] * 8 + [[0.01, 0.99]] * 8, dtype=torch.float64)
        targets = torch.zeros(16, dtype=torch.int64)
        result = partition_samples(pw, ps, targets, random_state=0)
        self.assertLess(result.low_mean, result.high_mean)
        self.assertTrue(torch.equal(result.partition[8:], torch.full((8,), PARTITION_HARD)))

    def test_artifact_atomic_roundtrip_and_hash(self) -> None:
        indices = np.array([9, 2])
        targets = np.array([0, 1])
        pw = np.array([[0.8, 0.2], [0.3, 0.7]])
        ps = np.array([[0.7, 0.3], [0.4, 0.6]])
        pws = (pw + ps) / 2
        partition = np.array([PARTITION_CLEAN, PARTITION_NOISY])
        y0 = np.eye(2)
        yn = np.array([[0.0, 0.0], [0.0, 1.0]])
        artifact = DLDPreCorrectionArtifact(
            indices, targets, pw, ps, pws, np.array([0.1, 0.2]), partition,
            y0, yn, yn - y0, np.ones((2, 3)), {"source": "test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dld_precorrection.npz"
            loaded = persist_precorrection_atomically(artifact, path)
            self.assertEqual(loaded.artifact_hash, artifact.artifact_hash)
            self.assertEqual(loaded.global_indices.tolist(), [2, 9])


if __name__ == "__main__":
    unittest.main()
