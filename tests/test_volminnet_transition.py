from __future__ import annotations

import math
import unittest

import torch

from lnl_toolbox.algorithms.volminnet import VolMinTransition


class VolMinTransitionTest(unittest.TestCase):
    def test_paper_initialization_and_row_contract(self) -> None:
        transition = VolMinTransition(10)
        matrix = transition.matrix()
        self.assertEqual(tuple(matrix.shape), (10, 10))
        self.assertTrue(torch.allclose(matrix.sum(1), torch.ones(10, dtype=torch.float64)))
        self.assertTrue(bool((matrix >= 0).all()))
        self.assertTrue(torch.allclose(torch.diagonal(matrix), torch.full((10,), 0.5, dtype=torch.float64)))
        off = matrix[~torch.eye(10, dtype=torch.bool)]
        self.assertTrue(torch.allclose(off, torch.full_like(off, 1.0 / 18.0)))
        self.assertAlmostEqual(transition.initial_raw_value, math.log(1.0 / 8.0))

    def test_only_off_diagonal_values_are_parameters_and_receive_gradient(self) -> None:
        transition = VolMinTransition(3)
        self.assertEqual(tuple(dict(transition.named_parameters())), ("off_diagonal_logits",))
        transition.matrix()[0, 1].backward()
        self.assertIsNotNone(transition.off_diagonal_logits.grad)
        self.assertTrue(bool(torch.isfinite(transition.off_diagonal_logits.grad).all()))

    def test_diagonal_exceeds_every_same_row_off_diagonal(self) -> None:
        transition = VolMinTransition(5)
        with torch.no_grad():
            transition.off_diagonal_logits.copy_(torch.linspace(-5.0, 5.0, 20))
        matrix = transition.matrix()
        for row in range(5):
            off = torch.cat((matrix[row, :row], matrix[row, row + 1 :]))
            self.assertTrue(bool((matrix[row, row] > off).all()))

    def test_binary_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_classes >= 3"):
            VolMinTransition(2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_contract(self) -> None:
        transition = VolMinTransition(3).cuda()
        matrix = transition.matrix()
        self.assertEqual(matrix.device.type, "cuda")
        matrix[0, 1].backward()
        self.assertTrue(bool(torch.isfinite(transition.off_diagonal_logits.grad).all()))
