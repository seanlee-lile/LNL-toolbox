"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_ca2c.py ---
import unittest

# --- merged from test_ca2c.py ---
import torch

# --- merged from test_ca2c.py ---
from lnl_toolbox.models.ca2c_cnn import CA2CSevenCNN

# --- merged from test_ca2c.py ---
from lnl_toolbox.algorithms.ca2c import CandidateMemory, cross_guidance, negative_label_objective, partial_label_objective

from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

from lnl_toolbox.training.ca2c_experiment import _ca2c_batch_objectives

# --- merged from test_ca2c.py ---
class _ca2c_CA2CTest(unittest.TestCase):

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

    def test_memory_confidence_is_class_wise_m_over_max_m(self) -> None:
        memory = CandidateMemory.create(torch.tensor([4]), 3)
        memory.counts[0] = torch.tensor([3.0, 1.0, 0.0])
        torch.testing.assert_close(
            memory.confidence_weights(torch.tensor([4])),
            torch.tensor([[1.0, 1.0 / 3.0, 0.0]]),
        )

    def test_objectives_only_update_target_logits(self) -> None:
        logits = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
        partial = partial_label_objective(logits, torch.tensor([[0.5, 0.5, 0.0]]), 0.5)
        partial.backward(retain_graph=True)
        self.assertIsNotNone(logits.grad)
        logits.grad.zero_()
        negative = negative_label_objective(logits, torch.tensor([[False, False, True]]))
        negative.backward()
        self.assertTrue(torch.isfinite(negative))

    def test_negative_objective_sums_classes_per_sample(self) -> None:
        logits = torch.zeros(2, 3, requires_grad=True)
        mask = torch.tensor([[False, True, True], [False, False, True]])
        value = negative_label_objective(logits, mask)
        expected = (2.0 * -torch.log(torch.tensor(2.0 / 3.0)) + -torch.log(torch.tensor(2.0 / 3.0))) / 2.0
        self.assertAlmostEqual(float(value), float(expected), places=6)

    def test_partial_objective_accepts_official_confidence_weight(self) -> None:
        logits = torch.zeros(2, 3, requires_grad=True)
        targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        value = partial_label_objective(logits, targets, 0.99, confidence_weights=torch.tensor([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0]]))
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertIsNotNone(logits.grad)

    def test_partial_objective_applies_weights_per_class(self) -> None:
        logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
        soft_targets = torch.tensor([[0.5, 0.5, 0.0]])
        confidence = torch.tensor([[1.0, 1.0 / 3.0, 0.0]])
        actual = partial_label_objective(
            logits,
            soft_targets,
            0.0,
            confidence_weights=confidence,
        )
        expected = -(0.5 * torch.log(torch.tensor(0.6)) + (0.5 / 3.0) * torch.log(torch.tensor(0.3)))
        torch.testing.assert_close(actual, expected)

    def test_warmup_is_ce_only_and_does_not_update_memory(self) -> None:
        p_logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
        n_logits = torch.tensor([[0.0, 2.0, -1.0]], requires_grad=True)
        targets = torch.tensor([0])
        memory = CandidateMemory.create(torch.tensor([7]), 3)
        p_loss, n_loss = _ca2c_batch_objectives(
            p_logits,
            n_logits,
            targets,
            torch.tensor([7]),
            CrossEntropyLoss(),
            memory,
            candidate_k=1,
            hard_weight=0.99,
            robust=False,
        )
        torch.testing.assert_close(
            p_loss, torch.nn.functional.cross_entropy(p_logits, targets)
        )
        torch.testing.assert_close(
            n_loss, torch.nn.functional.cross_entropy(n_logits, targets)
        )
        self.assertEqual(int(memory.counts.sum()), 0)

    def test_robust_batch_uses_only_paper_objectives(self) -> None:
        p_logits = torch.tensor([[2.0, 0.0, -1.0]])
        n_logits = torch.tensor([[0.0, 2.0, -1.0]])
        memory = CandidateMemory.create(torch.tensor([7]), 3)
        p_loss, n_loss = _ca2c_batch_objectives(
            p_logits,
            n_logits,
            torch.tensor([0]),
            torch.tensor([7]),
            CrossEntropyLoss(),
            memory,
            candidate_k=1,
            hard_weight=0.99,
            robust=True,
        )
        candidates, complements = cross_guidance(p_logits, n_logits, 1)
        expected_p = partial_label_objective(
            p_logits,
            memory.targets(torch.tensor([7])),
            0.99,
            confidence_weights=memory.confidence_weights(torch.tensor([7])),
        )
        expected_n = negative_label_objective(n_logits, complements)
        torch.testing.assert_close(p_loss, expected_p)
        torch.testing.assert_close(n_loss, expected_n)
        self.assertEqual(int(memory.counts.sum()), int(candidates.sum()))

    def test_memory_fingerprint_changes_after_update(self) -> None:
        memory = CandidateMemory.create(torch.tensor([0, 1]), 3)
        before = memory.fingerprint()
        memory.update_(torch.tensor([0]), torch.tensor([[True, False, False]]))
        self.assertNotEqual(before, memory.fingerprint())

    def test_invalid_k_fails(self) -> None:
        with self.assertRaises(ValueError):
            cross_guidance(torch.zeros(1, 3), torch.zeros(1, 3), 3)
