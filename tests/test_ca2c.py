from __future__ import annotations

import unittest

import torch

from lnl_toolbox.models.ca2c_cnn import CA2CSevenCNN

from lnl_toolbox.algorithms.ca2c import (
    CandidateMemory, cross_guidance, negative_label_objective,
    partial_label_objective,
)


class CA2CTest(unittest.TestCase):
    def test_official_seven_cnn_shape(self) -> None:
        model = CA2CSevenCNN(100)
        logits, features = model.forward_with_features(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 100))
        self.assertEqual(tuple(features.shape), (2, 128))

    def test_cross_guidance_shapes_and_detach(self) -> None:
        p = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        n = torch.tensor([[1.0, 3.0, 2.0]], requires_grad=True)
        candidate, complement = cross_guidance(p, n, 1)
        self.assertEqual(candidate.tolist(), [[False, True, False]])
        self.assertEqual(complement.tolist(), [[False, True, True]])
        self.assertFalse(candidate.requires_grad)

    def test_cross_guidance_uses_p_topk_complement_under_ties(self) -> None:
        p = torch.tensor([[2.0, 1.0, 1.0, 1.0]])
        n = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
        candidate, complement = cross_guidance(p, n, 2)
        self.assertEqual(candidate.tolist(), [[False, True, True, False]])
        self.assertEqual(complement.tolist(), [[False, False, True, True]])

    def test_memory_uses_global_indices(self) -> None:
        memory = CandidateMemory.create(torch.tensor([8, 3, 5]), 3)
        memory.update_(torch.tensor([5, 8]), torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool))
        self.assertEqual(memory.targets(torch.tensor([8])).tolist(), [[0.0, 1.0, 0.0]])
        restored = CandidateMemory.from_state_dict(memory.state_dict())
        self.assertTrue(torch.equal(restored.counts, memory.counts))

    def test_objectives_only_update_target_logits(self) -> None:
        logits = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
        partial = partial_label_objective(
            logits, torch.tensor([[0.5, 0.5, 0.0]]), 0.5
        )
        partial.backward(retain_graph=True)
        self.assertIsNotNone(logits.grad)
        logits.grad.zero_()
        negative = negative_label_objective(
            logits, torch.tensor([[False, False, True]])
        )
        negative.backward()
        self.assertTrue(torch.isfinite(negative))

    def test_negative_objective_sums_classes_per_sample(self) -> None:
        logits = torch.zeros(2, 3, requires_grad=True)
        mask = torch.tensor([[False, True, True], [False, False, True]])
        value = negative_label_objective(logits, mask)
        expected = (2.0 * (-torch.log(torch.tensor(2.0 / 3.0))) + (-torch.log(torch.tensor(2.0 / 3.0)))) / 2.0
        self.assertAlmostEqual(float(value), float(expected), places=6)

    def test_partial_objective_accepts_official_confidence_weight(self) -> None:
        logits = torch.zeros(2, 3, requires_grad=True)
        targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        value = partial_label_objective(
            logits, targets, 0.99, confidence=torch.tensor([1.0, 0.5])
        )
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertIsNotNone(logits.grad)

    def test_memory_fingerprint_changes_after_update(self) -> None:
        memory = CandidateMemory.create(torch.tensor([0, 1]), 3)
        before = memory.fingerprint()
        memory.update_(torch.tensor([0]), torch.tensor([[True, False, False]]))
        self.assertNotEqual(before, memory.fingerprint())

    def test_invalid_k_fails(self) -> None:
        with self.assertRaises(ValueError):
            cross_guidance(torch.zeros(1, 3), torch.zeros(1, 3), 3)


if __name__ == "__main__":
    unittest.main()
