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

    def test_warmup_preserves_ce_and_accumulates_both_peers(self) -> None:
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
        torch.testing.assert_close(memory.counts, torch.tensor([[1., 1., 0.]]))

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

    def test_400_warmup_epochs_preserve_history_then_only_n_adds_candidates(self) -> None:
        p = torch.tensor([[3., 2., 1.]])
        n = torch.tensor([[1., 3., 2.]])
        indices = torch.tensor([7])
        memory = CandidateMemory.create(indices, 3)
        for _ in range(400):
            _ca2c_batch_objectives(p, n, torch.tensor([0]), indices, CrossEntropyLoss(), memory,
                                  candidate_k=2, hard_weight=.99, robust=False)
        torch.testing.assert_close(memory.counts, torch.tensor([[400., 800., 400.]]))
        _ca2c_batch_objectives(p, n, torch.tensor([0]), indices, CrossEntropyLoss(), memory,
                              candidate_k=2, hard_weight=.99, robust=True)
        torch.testing.assert_close(memory.counts, torch.tensor([[400., 801., 401.]]))

    def test_robust_mixture_matches_loss_and_gradients_for_both_peers(self) -> None:
        inputs = [torch.tensor([[2., 0., -1.], [0., 2., 1.]], requires_grad=True),
                  torch.tensor([[0., 2., -1.], [2., 0., 1.]], requires_grad=True),
                  torch.tensor([[0., 1., 2.], [2., 1., 0.]], requires_grad=True),
                  torch.tensor([[1., 0., 2.], [1., 2., 0.]], requires_grad=True)]
        p, n, ps, ns = inputs
        indices, targets = torch.tensor([7, 3]), torch.tensor([0, 1])
        for weight in [.2, .8]:
            with self.subTest(weight=weight):
                memory = CandidateMemory.create(indices, 3)
                actual = _ca2c_batch_objectives(p, n, targets, indices, CrossEntropyLoss(), memory,
                            candidate_k=1, hard_weight=.99, robust=True, robust_weight=weight,
                            p_strong_logits=ps, n_strong_logits=ns)
                _, complements = cross_guidance(p, n, 1)
                expected = (
                    weight * partial_label_objective(p, memory.targets(indices), .99, confidence_weights=memory.confidence_weights(indices)) + (1-weight) * torch.nn.functional.cross_entropy(ps, p.detach().argmax(1)),
                    weight * negative_label_objective(n, complements) + (1-weight) * torch.nn.functional.cross_entropy(ns, n.detach().argmax(1)),
                )
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right)
                actual_grad = torch.autograd.grad(sum(actual), inputs, retain_graph=True)
                expected_grad = torch.autograd.grad(sum(expected), inputs, retain_graph=True)
                for left, right in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(left, right)
                self.assertGreater(float(actual_grad[2].abs().sum()), 0.)
                self.assertGreater(float(actual_grad[3].abs().sum()), 0.)
        memory = CandidateMemory.create(indices, 3)
        losses = _ca2c_batch_objectives(p, n, targets, indices, CrossEntropyLoss(), memory,
                    candidate_k=1, hard_weight=.99, robust=True, robust_weight=0., p_strong_logits=ps, n_strong_logits=ns)
        gradients = torch.autograd.grad(sum(losses), inputs)
        torch.testing.assert_close(gradients[0], torch.zeros_like(p))
        torch.testing.assert_close(gradients[1], torch.zeros_like(n))

    def test_robust_weight_and_missing_views_fail_before_memory_mutation(self) -> None:
        p, n, indices = torch.zeros(1, 3), torch.ones(1, 3), torch.tensor([7])
        for weight in [-.1, 1.1, float('nan'), .8]:
            memory = CandidateMemory.create(indices, 3)
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, 'robust_weight'):
                _ca2c_batch_objectives(p, n, torch.tensor([0]), indices, CrossEntropyLoss(), memory,
                                      candidate_k=1, hard_weight=.99, robust=True, robust_weight=weight)
            self.assertEqual(float(memory.counts.sum()), 0.)


class CA2CRunnerChangesTest(unittest.TestCase):
    def test_web_defaults_and_saved_robust_weight(self) -> None:
        import tempfile
        import yaml
        from pathlib import Path
        from web import command_console
        fields = {f['path']: f for f in command_console._config_schema('ca2c-cifar100-reproduction')['fields']}
        self.assertEqual(fields['optimizer.lr']['value'], .01)
        self.assertEqual(fields['ca2c.robust_weight']['value'], .8)
        self.assertIn('PLL/NL', fields['ca2c.robust_weight']['note'])
        self.assertIn('强增强一致性', fields['ca2c.robust_weight']['note'])
        self.assertIn('lambda 单独控制', fields['ca2c.robust_weight']['note'])
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            saved = command_console._save_config({'recipe': 'ca2c-cifar100-reproduction', 'path': str(Path(directory) / 'edited.yaml'),
                       'patches': [{'path': 'ca2c.robust_weight', 'value': .2}], 'acknowledge_paper_impact': True})
            resolved = yaml.safe_load((command_console.ROOT / saved['path']).read_text(encoding='utf-8'))
            self.assertEqual(resolved['ca2c']['robust_weight'], .2)
            self.assertEqual(resolved['ca2c']['lambda'], .99)

    def test_real_two_epoch_training_uses_strong_views_and_resumes_memory(self) -> None:
        import tempfile
        import numpy as np
        import yaml
        from pathlib import Path
        from unittest.mock import patch
        from lnl_toolbox.data.cifar import CifarData
        from lnl_toolbox.training import ca2c_experiment as runner
        from lnl_toolbox.training.checkpoint import read_checkpoint
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / 'configs/experiment/ca2c_cifar100_reproduction.yaml').read_text(encoding='utf-8'))
        # Controlled image data and a tiny classifier exercise warmup -> robust
        # without running the 700-epoch paper experiment.
        config['data'].update(name='cifar10', root='unused-controlled-source', validation_size=0)
        config['loader'].update(batch_size=10, num_workers=0, pin_memory=False, drop_last=False)
        config['model'] = {'name': 'tiny_cnn', 'width': 4}
        config['ca2c'].update(warmup_epochs=1, robust_weight=.2)
        config['trainer'].update(epochs=2, device='cpu')
        config['scheduler']['t_max'] = 2
        rng = np.random.default_rng(13)
        def source(split, count):
            labels = np.arange(count) % 10
            return CifarData(rng.integers(0, 256, (count, 32, 32, 3), dtype=np.uint8), labels,
                             tuple(str(i) for i in range(10)), split, 'cifar10')
        train, test = source('train', 40), source('test', 10)
        real_objectives, real_save = runner._ca2c_batch_objectives, runner.atomic_save
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=lambda root, split: train if split == 'train' else test):
            complete = Path(directory) / 'complete'
            with patch.object(runner, '_ca2c_batch_objectives', wraps=real_objectives) as spy:
                runner.run_ca2c_experiment(config, complete)
            warmup_calls = [c for c in spy.call_args_list if not c.kwargs['robust']]
            robust_calls = [c for c in spy.call_args_list if c.kwargs['robust']]
            self.assertEqual(len(warmup_calls), 4)
            self.assertEqual(len(robust_calls), 4)
            for call in robust_calls:
                self.assertEqual(call.kwargs['robust_weight'], .2)
                self.assertTrue(call.kwargs['p_strong_logits'].requires_grad)
                self.assertTrue(call.kwargs['n_strong_logits'].requires_grad)
            interrupted = Path(directory) / 'interrupted'
            def interrupt(payload, path):
                real_save(payload, path)
                if payload['completed_epoch'] == 0:
                    raise RuntimeError('controlled warmup interruption')
            with patch.object(runner, 'atomic_save', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'warmup interruption'):
                runner.run_ca2c_experiment(config, interrupted)
            warmup = read_checkpoint(interrupted / 'last.pt', 'cpu')
            self.assertTrue(torch.equal(warmup['candidate_memory']['counts'].sum(1), torch.full((40,), 4.)))
            runner.run_ca2c_experiment(config, resume=interrupted / 'last.pt')
            reference = read_checkpoint(complete / 'last.pt', 'cpu')
            restored = read_checkpoint(interrupted / 'last.pt', 'cpu')
            torch.testing.assert_close(reference['candidate_memory']['counts'], restored['candidate_memory']['counts'], rtol=0, atol=0)
            self.assertTrue(torch.equal(restored['candidate_memory']['counts'].sum(1), torch.full((40,), 6.)))
            for peer in ['p_model', 'n_model']:
                for name, value in reference[peer].items():
                    torch.testing.assert_close(value, restored[peer][name], rtol=0, atol=0)
            self.assertEqual(restored['completed_epoch'], 1)
