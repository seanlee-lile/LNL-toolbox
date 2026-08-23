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
    def test_cosine_similarity_knn_uses_stable_indices_and_normalizes(self) -> None:
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        targets = torch.tensor([0, 1, 1])
        indices = torch.tensor([50, 10, 90])
        result = weighted_neighbor_distribution(
            features, features, targets, indices, indices,
            num_classes=2, k=2, metric="cosine_similarity", delta=1e-3,
            self_neighbor="include",
        )
        self.assertTrue(torch.allclose(result.probabilities.sum(1), torch.ones(3)))
        self.assertEqual(result.neighbor_indices[0, 0].item(), 50)
        order = torch.tensor([2, 0, 1])
        permuted = weighted_neighbor_distribution(
            features[order], features[order], targets[order], indices[order], indices[order],
            num_classes=2, k=2, metric="cosine_similarity", delta=1e-3,
            self_neighbor="include",
        )
        by_index = {int(i): p for i, p in zip(indices, result.probabilities)}
        for index, probability in zip(indices[order], permuted.probabilities):
            self.assertTrue(torch.allclose(probability, by_index[int(index)]))

    def test_cosine_similarity_weights_match_released_formula(self) -> None:
        features = torch.tensor(
            [[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.4, 3.0 ** 0.5 * 0.4]],
            dtype=torch.float64,
        )
        targets = torch.tensor([0, 1, 1, 0])
        indices = torch.tensor([30, 10, 20, 40])
        delta = 1e-6
        result = weighted_neighbor_distribution(
            features[:1], features, targets, indices[:1], indices,
            num_classes=2, k=3, metric="cosine_similarity", delta=delta,
            self_neighbor="include",
        )
        expected_similarity = torch.tensor([[1.0, 0.8, 0.6]], dtype=torch.float64)
        expected_raw = 1.0 / (expected_similarity + delta)
        expected_weights = expected_raw / expected_raw.sum(dim=1, keepdim=True)
        torch.testing.assert_close(result.neighbor_values, expected_similarity)
        torch.testing.assert_close(result.unnormalized_weights, expected_raw)
        torch.testing.assert_close(result.weights, expected_weights)
        torch.testing.assert_close(
            result.probabilities,
            torch.tensor(
                [[expected_weights[0, 0], expected_weights[0, 1:].sum()]],
                dtype=torch.float64,
            ),
        )
        self.assertAlmostEqual(float(result.neighbor_values[0, 0]), 1.0)
        self.assertAlmostEqual(float(result.unnormalized_weights[0, 0]), 1.0, places=5)
        self.assertLess(float(result.unnormalized_weights[0, 0]), 2.0)

    def test_cosine_similarity_invalid_weight_denominator_fails(self) -> None:
        query = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        references = torch.tensor(
            [[-1.0, 0.0], [-0.5, 3.0 ** 0.5 / 2.0]], dtype=torch.float64
        )
        with self.assertRaisesRegex(ValueError, "denominator"):
            weighted_neighbor_distribution(
                query, references, torch.tensor([0, 1]),
                torch.tensor([10]), torch.tensor([20, 30]),
                num_classes=2, k=1, metric="cosine_similarity", delta=1e-6,
            )
        zeros = torch.zeros((2, 2), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "weights"):
            weighted_neighbor_distribution(
                zeros[:1], zeros, torch.tensor([0, 1]),
                torch.tensor([10]), torch.tensor([20, 30]),
                num_classes=2, k=1, metric="cosine_similarity",
                delta=float.fromhex("0x0.0000000000001p-1022"),
            )

    def test_exact_chunking_matches_dense_end_to_end(self) -> None:
        generator = torch.Generator().manual_seed(413)
        weak = torch.rand(23, 7, generator=generator, dtype=torch.float64)
        strong = weak + 0.02 * torch.randn(
            23, 7, generator=generator, dtype=torch.float64
        )
        targets = torch.arange(23, dtype=torch.int64) % 3
        indices = torch.randperm(23, generator=generator, dtype=torch.int64) + 100

        def estimate(features, chunk_size):
            return weighted_neighbor_distribution(
                features,
                features,
                targets,
                indices,
                indices,
                num_classes=3,
                k=6,
                metric="cosine_similarity",
                delta=1e-6,
                self_neighbor="include",
                query_chunk_size=chunk_size,
            )

        dense_w, dense_s = estimate(weak, None), estimate(strong, None)
        chunk_w, chunk_s = estimate(weak, 4), estimate(strong, 4)
        for dense, chunked in ((dense_w, chunk_w), (dense_s, chunk_s)):
            self.assertTrue(torch.equal(dense.neighbor_indices, chunked.neighbor_indices))
            torch.testing.assert_close(
                dense.neighbor_values, chunked.neighbor_values, rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(
                dense.unnormalized_weights, chunked.unnormalized_weights,
                rtol=1e-12, atol=1e-12,
            )
            torch.testing.assert_close(
                dense.weights, chunked.weights, rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(
                dense.probabilities, chunked.probabilities, rtol=1e-12, atol=1e-12
            )

        dense_partition = partition_samples(
            dense_w.probabilities,
            dense_s.probabilities,
            targets,
            random_state=7,
            minimum_mean_separation=0.0,
        )
        chunk_partition = partition_samples(
            chunk_w.probabilities,
            chunk_s.probabilities,
            targets,
            random_state=7,
            minimum_mean_separation=0.0,
        )
        self.assertTrue(
            torch.equal(dense_partition.partition, chunk_partition.partition)
        )
        dense_y0 = construct_y0(
            dense_partition.p_ws, targets, dense_partition.partition
        )
        chunk_y0 = construct_y0(
            chunk_partition.p_ws, targets, chunk_partition.partition
        )
        dense_yn = construct_yn(
            dense_w.probabilities,
            dense_s.probabilities,
            targets,
            dense_partition.partition,
        )
        chunk_yn = construct_yn(
            chunk_w.probabilities,
            chunk_s.probabilities,
            targets,
            chunk_partition.partition,
        )
        torch.testing.assert_close(dense_y0, chunk_y0, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(dense_yn, chunk_yn, rtol=1e-12, atol=1e-12)

    def test_chunking_keeps_self_exclusion_and_rejects_invalid_size(self) -> None:
        features = torch.eye(4, dtype=torch.float64)
        targets = torch.tensor([0, 1, 0, 1])
        indices = torch.tensor([40, 10, 30, 20])
        dense = weighted_neighbor_distribution(
            features, features, targets, indices, indices,
            num_classes=2, k=2, metric="cosine_distance", delta=1e-6,
            self_neighbor="exclude",
        )
        chunked = weighted_neighbor_distribution(
            features, features, targets, indices, indices,
            num_classes=2, k=2, metric="cosine_distance", delta=1e-6,
            self_neighbor="exclude", query_chunk_size=1,
        )
        self.assertTrue(torch.equal(dense.neighbor_indices, chunked.neighbor_indices))
        for row, index in zip(chunked.neighbor_indices, indices):
            self.assertNotIn(int(index), row.tolist())
        with self.assertRaisesRegex(ValueError, "query_chunk_size"):
            weighted_neighbor_distribution(
                features, features, targets, indices, indices,
                num_classes=2, k=2, metric="cosine_distance", delta=1e-6,
                query_chunk_size=0,
            )

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
