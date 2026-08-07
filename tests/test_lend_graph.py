import unittest

import torch

from lnl_toolbox.algorithms.lend import build_lend_similarity, normalize_lend_graph


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

    def test_zero_degree_fails(self):
        with self.assertRaisesRegex(ValueError, "non-positive degree"):
            normalize_lend_graph(torch.zeros(3, 3))

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
