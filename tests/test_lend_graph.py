import unittest

import torch

from lnl_toolbox.algorithms.lend import (
    build_lend_similarity,
    dilute_labels,
    normalize_lend_graph,
)


class LENDGraphTest(unittest.TestCase):
    def test_equation_one_and_normalization_hand_calculation(self):
        features = torch.tensor([[1., 1.], [2., 1.], [1., 2.]])
        indices = torch.tensor([10, 20, 30])
        adjacency = build_lend_similarity(
            features, indices, k=2, gamma=1.0,
            metric="inner_product", normalize_features=False,
        )
        expected = torch.tensor([[0., 3., 3.], [3., 0., 4.], [3., 4., 0.]])
        torch.testing.assert_close(adjacency, expected)
        product = expected.T @ expected
        degree = product.sum(1)
        manual = degree.rsqrt()[:, None] * product * degree.rsqrt()[None, :]
        graph = normalize_lend_graph(adjacency)
        torch.testing.assert_close(graph, manual)
        torch.testing.assert_close(graph, graph.T)

    def test_positive_clamp_directed_knn_and_stable_tie(self):
        features = torch.tensor([[1., 0.], [1., 1.], [-1., 0.]])
        indices = torch.tensor([30, 10, 20])
        adjacency = build_lend_similarity(
            features, indices, k=1, gamma=2.0,
            metric="inner_product", normalize_features=False,
        )
        self.assertTrue(torch.equal(adjacency.diag(), torch.zeros(3)))
        self.assertTrue(bool((adjacency >= 0).all()))
        self.assertEqual(adjacency[0, 1].item(), 1.0)
        self.assertEqual(adjacency[2].sum().item(), 0.0)

    def test_permutation_equivariance(self):
        features = torch.tensor([[1., 1.], [2., 1.], [1., 2.], [2., 2.]])
        indices = torch.tensor([90, 10, 40, 20])
        original = build_lend_similarity(features, indices, k=2, gamma=1.0,
            metric="inner_product", normalize_features=False)
        order = torch.tensor([2, 0, 3, 1])
        permuted = build_lend_similarity(features[order], indices[order], k=2,
            gamma=1.0, metric="inner_product", normalize_features=False)
        torch.testing.assert_close(permuted, original[order][:, order])

    def test_one_zero_degree_node_uses_zero_inverse(self):
        adjacency = torch.tensor([
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
        ])
        graph = normalize_lend_graph(adjacency)
        torch.testing.assert_close(graph, torch.diag(torch.tensor([1., 1., 0., 0.])))
        self.assertTrue(bool(torch.isfinite(graph).all()))

    def test_multiple_zero_degree_nodes_are_finite_and_deterministic(self):
        adjacency = torch.zeros(4, 4)
        first = normalize_lend_graph(adjacency)
        second = normalize_lend_graph(adjacency)
        torch.testing.assert_close(first, torch.zeros_like(adjacency))
        torch.testing.assert_close(second, first)
        torch.testing.assert_close(first, first.T)

    def test_zero_degree_dilution_retains_noisy_argmax(self):
        adjacency = torch.tensor([[0., 1., 0.], [1., 0., 0.], [1., 0., 0.]])
        graph = normalize_lend_graph(adjacency)
        noisy = torch.eye(3)
        first = dilute_labels(noisy, graph, alpha=0.99, steps=10)
        second = dilute_labels(noisy, graph, alpha=0.99, steps=10)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[2], (1.0 - 0.99) ** 10 * noisy[2])
        self.assertEqual(first[2].argmax().item(), 2)
        self.assertTrue(bool(torch.isfinite(first).all()))

    def test_non_finite_and_negative_adjacency_remain_invalid(self):
        for value in (float("nan"), float("inf"), -1.0):
            adjacency = torch.zeros(3, 3)
            adjacency[0, 1] = value
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                normalize_lend_graph(adjacency)

    def test_sparse_batch_256_can_have_valid_zero_degree_nodes(self):
        adjacency = torch.zeros(256, 256)
        for row in range(256):
            for offset in range(1, 9):
                adjacency[row, (row + offset) % 128] = 1.0
        self.assertTrue(bool(((adjacency.T @ adjacency).sum(1) == 0).any()))
        graph = normalize_lend_graph(adjacency)
        self.assertTrue(bool(torch.isfinite(graph).all()))
        torch.testing.assert_close(graph[128:], torch.zeros_like(graph[128:]))
        torch.testing.assert_close(graph[:, 128:], torch.zeros_like(graph[:, 128:]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_matches_cpu(self):
        features = torch.tensor([[1., 1.], [2., 1.], [1., 2.]])
        indices = torch.tensor([1, 2, 3])
        cpu = build_lend_similarity(features, indices, k=2, gamma=1.0,
            metric="inner_product", normalize_features=False)
        cuda = build_lend_similarity(features.cuda(), indices.cuda(), k=2,
            gamma=1.0, metric="inner_product", normalize_features=False)
        torch.testing.assert_close(cuda.cpu(), cpu)


if __name__ == "__main__": unittest.main()
