import unittest

import torch

from lnl_toolbox.algorithms.lend import dilute_labels, select_lend_samples


class LENDDilutionTest(unittest.TestCase):
    def test_single_and_multiple_step_equation(self):
        labels = torch.eye(2)
        graph = torch.tensor([[0.5, 0.25], [0.25, 0.5]])
        one = dilute_labels(labels, graph, alpha=0.8, steps=1)
        expected = 0.8 * graph @ labels + 0.2 * labels
        torch.testing.assert_close(one, expected)
        two = dilute_labels(labels, graph, alpha=0.8, steps=2)
        torch.testing.assert_close(two, 0.8 * graph @ expected + 0.2 * expected)
        self.assertFalse(two.requires_grad)

    def test_does_not_renormalize_rows(self):
        labels = torch.eye(2)
        graph = torch.tensor([[2., 0.], [0., 0.5]])
        result = dilute_labels(labels, graph, alpha=0.5, steps=1)
        self.assertFalse(torch.allclose(result.sum(1), torch.ones(2)))

    def test_selection_uses_noisy_label_and_first_index_tie(self):
        noisy = torch.tensor([0, 1, 1])
        history = torch.tensor([[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]])
        self.assertEqual(select_lend_samples(noisy, history).tolist(), [True, True, False])


if __name__ == "__main__": unittest.main()
