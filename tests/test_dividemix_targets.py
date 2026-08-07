from __future__ import annotations

import unittest

import torch
from torch import nn

from lnl_toolbox.algorithms.dividemix import co_guess, co_refine, sharpen


class FixedModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("values", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs):
        return self.values.expand(inputs.shape[0], -1)


class DivideMixTargetsTest(unittest.TestCase):
    def test_sharpen_is_normalized_and_detached(self):
        result = sharpen(torch.tensor([[0.25, 0.75]], requires_grad=True), 0.5)
        self.assertTrue(torch.allclose(result.sum(1), torch.ones(1)))
        self.assertFalse(result.requires_grad)
        self.assertGreater(float(result[0, 1]), 0.75)

    def test_co_refinement_has_paper_endpoints(self):
        model = FixedModel([0.0, 2.0])
        inputs = (torch.zeros(2, 1), torch.ones(2, 1))
        targets = torch.tensor([0, 1])
        result = co_refine(model, inputs, targets, torch.tensor([1.0, 0.0]), 1.0)
        self.assertTrue(torch.equal(result[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(result[1], torch.softmax(torch.tensor([0.0, 2.0]), 0)))
        self.assertFalse(result.requires_grad)

    def test_co_guess_averages_both_peers_and_all_views(self):
        model_a = FixedModel([3.0, 0.0])
        model_b = FixedModel([0.0, 1.0])
        views = (torch.zeros(2, 1), torch.ones(2, 1))
        result = co_guess(model_a, model_b, views, 1.0)
        expected = (torch.softmax(torch.tensor([3.0, 0.0]), 0) + torch.softmax(torch.tensor([0.0, 1.0]), 0)) / 2
        self.assertTrue(torch.allclose(result[0], expected))
        self.assertTrue(torch.allclose(result.sum(1), torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
