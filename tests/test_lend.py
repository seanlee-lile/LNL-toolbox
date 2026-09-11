"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_lend.py ---
import unittest

# --- merged from test_lend.py ---
import torch

# --- merged from test_lend.py ---
from lnl_toolbox.selectors.lend import LENDSelector, dilute_labels as selector_dilute_labels

# --- merged from test_lend.py ---
class _lend_LENDMathTest(unittest.TestCase):

    def test_batch_local_dilution_is_a_distribution(self) -> None:
        features = torch.eye(4)
        targets = torch.tensor([0, 1, 2, 1])
        diluted = selector_dilute_labels(features, targets, neighbors=2, num_classes=3, diffusion_steps=2)
        self.assertEqual(tuple(diluted.shape), (4, 3))
        torch.testing.assert_close(diluted.sum(1), torch.ones(4))

    def test_selector_returns_complementary_mask(self) -> None:
        result = LENDSelector(neighbors=2, num_classes=3).select(features=torch.eye(4), noisy_targets=torch.tensor([0, 1, 2, 1]))
        torch.testing.assert_close(result.rejected_mask, ~result.selected_mask)

# --- merged from test_lend_algorithm.py ---
import copy

# --- merged from test_lend_algorithm.py ---
import unittest

# --- merged from test_lend_algorithm.py ---
from unittest.mock import patch

# --- merged from test_lend_algorithm.py ---
import torch

# --- merged from test_lend_algorithm.py ---
from torch import nn

# --- merged from test_lend_algorithm.py ---
from lnl_toolbox.algorithms.lend import LENDAlgorithm, LENDConfig

# --- merged from test_lend_algorithm.py ---
from lnl_toolbox.algorithms.lend.graph import build_lend_similarity

# --- merged from test_lend_algorithm.py ---
from lnl_toolbox.core import Batch, ExperimentContext, RunState

# --- merged from test_lend_algorithm.py ---
from lnl_toolbox.models.cifar_resnet import cifar_resnet18

# --- merged from test_lend_algorithm.py ---
from lnl_toolbox.models.feature_output import FeatureOutput

# --- merged from test_lend_algorithm.py ---
class _lend_algorithm_FeatureModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 3)
        self.classifier = nn.Linear(3, 2)
        self.feature_calls = 0

    def forward_with_features(self, inputs):
        self.feature_calls += 1
        features = torch.relu(self.encoder(inputs)) + 0.1
        return FeatureOutput(self.classifier(features), features)

    def forward(self, inputs):
        features = torch.relu(self.encoder(inputs)) + 0.1
        return self.classifier(features)

# --- merged from test_lend_algorithm.py ---
def _lend_algorithm__config(beta=1.0):
    return {'method': 'lend', 'data': {'name': 'cifar10', 'max_train_samples': 3}, 'model': {'name': 'tiny_cnn'}, 'loader': {'batch_size': 3, 'drop_last': False}, 'loss': {'name': 'ce'}, 'lend': {'graph': {'k': 2, 'gamma': 1.0, 'metric': 'inner_product', 'normalize_features': False}, 'dilution': {'alpha': 0.99, 'policy': 'fixed_steps', 'steps': 1}, 'history': {'beta': beta, 'first_observation': 'current'}, 'selection': {'rule': 'noisy_equals_diluted_argmax', 'reduction': 'batch_mean', 'empty_batch': 'skip_update'}, 'training': {'epochs': 2}}}

# --- merged from test_lend_algorithm.py ---
class _lend_algorithm_LENDAlgorithmTest(unittest.TestCase):

    def _algorithm(self, beta=1.0, lr=0.0):
        torch.manual_seed(3)
        model = _lend_algorithm_FeatureModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        algorithm = LENDAlgorithm(model=model, optimizer=optimizer, loss=nn.CrossEntropyLoss(reduction='none'), device=torch.device('cpu'), method_config=LENDConfig.from_mapping(_lend_algorithm__config(beta)), canonical_global_indices=torch.tensor([10, 20, 30]), num_classes=2)
        algorithm.setup(ExperimentContext(__import__('pathlib').Path('.')))
        return algorithm

    def test_exact_batch_mean_formula_uses_four_sample_denominator(self):
        config = _lend_algorithm__config()
        config['loader']['batch_size'] = 4
        model = _lend_algorithm_FeatureModel()
        algorithm = LENDAlgorithm(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.0), loss=nn.CrossEntropyLoss(reduction='none'), device=torch.device('cpu'), method_config=LENDConfig.from_mapping(config), canonical_global_indices=torch.tensor([10, 20, 30, 40]), num_classes=2)
        algorithm.setup(ExperimentContext(__import__('pathlib').Path('.')))
        algorithm.private_state.history.values[:] = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        algorithm.private_state.history.initialized[:] = True
        logits = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0]], requires_grad=True)
        losses = nn.CrossEntropyLoss(reduction='none')(logits, torch.zeros(4, dtype=torch.long))
        featured = FeatureOutput(logits, torch.ones(4, 3))
        with patch('lnl_toolbox.algorithms.lend.algorithm.forward_with_features', return_value=featured):
            result = algorithm.step(Batch({'input': torch.zeros(4, 2), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.tensor([10, 20, 30, 40])}), RunState(cycle=0))
        self.assertEqual(result.metadata['selected_mask'].tolist(), [True, True, False, True])
        self.assertAlmostEqual(result.metrics['selected_train_loss_sum'], ((losses[0] + losses[1] + losses[3]) / 4).item(), places=6)

    def test_one_forward_batch_mean_and_noisy_targets(self):
        algorithm = self._algorithm(lr=0.0)
        history = algorithm.private_state.history
        history.values[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        history.initialized[:] = True
        inputs = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        targets = torch.zeros(3, dtype=torch.long)
        before = copy.deepcopy(algorithm.model.state_dict())
        expected_model = _lend_algorithm_FeatureModel()
        expected_model.load_state_dict(before)
        expected = nn.CrossEntropyLoss(reduction='none')(expected_model(inputs), targets)[torch.tensor([True, False, True])].sum() / 3
        expected.backward()
        result = algorithm.step(Batch({'input': inputs, 'target': targets, 'index': torch.tensor([10, 20, 30])}), RunState(cycle=0))
        self.assertEqual(algorithm.model.feature_calls, 1)
        self.assertEqual(result.metadata['selected_mask'].tolist(), [True, False, True])
        self.assertAlmostEqual(result.metrics['selected_train_loss_sum'], expected.item(), places=6)
        for parameter, expected_parameter in zip(algorithm.model.parameters(), expected_model.parameters()):
            torch.testing.assert_close(parameter.grad, expected_parameter.grad)

    def test_all_selected_batch_mean_equals_ce_mean(self):
        algorithm = self._algorithm(lr=0.0)
        history = algorithm.private_state.history
        history.values[:] = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        history.initialized[:] = True
        inputs = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        targets = torch.zeros(3, dtype=torch.long)
        expected = nn.CrossEntropyLoss()(algorithm.model(inputs), targets)
        result = algorithm.step(Batch({'input': inputs, 'target': targets, 'index': torch.tensor([10, 20, 30])}), RunState(cycle=0))
        self.assertEqual(result.metadata['selected_mask'].tolist(), [True] * 3)
        self.assertAlmostEqual(result.metrics['selected_train_loss_sum'], expected.item(), places=6)

    def test_partial_selection_uses_batch_size_not_selected_count(self):
        algorithm = self._algorithm(lr=0.0)
        history = algorithm.private_state.history
        history.values[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        history.initialized[:] = True
        inputs = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        targets = torch.zeros(3, dtype=torch.long)
        losses = nn.CrossEntropyLoss(reduction='none')(algorithm.model(inputs), targets)
        result = algorithm.step(Batch({'input': inputs, 'target': targets, 'index': torch.tensor([10, 20, 30])}), RunState(cycle=0))
        self.assertAlmostEqual(result.metrics['selected_train_loss_sum'], ((losses[0] + losses[2]) / 3).item(), places=6)
        self.assertNotAlmostEqual(result.metrics['selected_train_loss_sum'], ((losses[0] + losses[2]) / 2).item(), places=6)

    def test_empty_selection_skips_optimizer_but_commits_history(self):
        algorithm = self._algorithm(lr=0.1)
        history = algorithm.private_state.history
        history.values[:] = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        history.initialized[:] = True
        before = copy.deepcopy(algorithm.model.state_dict())
        state = RunState(cycle=0)
        result = algorithm.step(Batch({'input': torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]), 'target': torch.zeros(3, dtype=torch.long), 'index': torch.tensor([10, 20, 30])}), state)
        self.assertEqual(result.metrics['selected_samples'], 0.0)
        self.assertEqual(algorithm.private_state.optimizer_steps, 0)
        self.assertTrue(bool((history.last_updated_epoch == 0).all()))
        for name, value in algorithm.model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_state_roundtrip_and_config_drift(self):
        source = self._algorithm()
        state = source.state_dict()
        restored = self._algorithm()
        restored.load_state_dict(state)
        changed = self._algorithm(beta=0.5)
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            changed.load_state_dict(state)
        legacy = copy.deepcopy(state)
        legacy['method_config']['zero_degree_policy'] = 'error'
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            restored.load_state_dict(legacy)
        legacy = copy.deepcopy(state)
        legacy['method_config']['reduction'] = 'paper_sum'
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            restored.load_state_dict(legacy)

    def test_resnet18_batch_256_step_keeps_finite_nonzero_features(self):
        torch.manual_seed(19)
        model = cifar_resnet18(num_classes=10, base_width=64)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)
        config = _lend_algorithm__config(beta=0.9)
        config['data']['max_train_samples'] = 256
        config['loader']['batch_size'] = 256
        config['lend']['graph']['k'] = 8
        algorithm = LENDAlgorithm(model=model, optimizer=optimizer, loss=nn.CrossEntropyLoss(reduction='none'), device=torch.device('cpu'), method_config=LENDConfig.from_mapping(config), canonical_global_indices=torch.arange(256), num_classes=10)
        algorithm.setup(ExperimentContext(__import__('pathlib').Path('.')))
        inputs = torch.randn(256, 3, 32, 32)
        targets = torch.arange(256) % 10
        before = torch.cat([value.detach().flatten() for value in model.parameters()])
        classifier_weight_before = model.classifier.weight.detach().clone()
        classifier_before = classifier_weight_before.norm()
        result = algorithm.step(Batch({'input': inputs, 'target': targets, 'index': torch.arange(256)}), RunState(cycle=0))
        after = torch.cat([value.detach().flatten() for value in model.parameters()])
        self.assertTrue(bool(torch.isfinite(after).all()))
        self.assertLess(float((after - before).norm() / before.norm()), 0.1)
        self.assertLess(float((model.classifier.weight.detach() - classifier_weight_before).norm() / classifier_before), 1.0)
        model.train()
        with torch.no_grad():
            features = model.forward_with_features(inputs).features
        self.assertGreater(float(features.norm(dim=1).median()), 0.0)
        adjacency = build_lend_similarity(features.detach(), torch.arange(256), k=8, gamma=1.0, metric='inner_product', normalize_features=False)
        self.assertGreater(int(torch.count_nonzero(adjacency)), 0)
        self.assertTrue(torch.isfinite(torch.tensor(result.metrics['selected_train_loss_sum'])))

    def test_zero_degree_graph_completes_algorithm_step(self):
        algorithm = self._algorithm(lr=0.0)
        featured = FeatureOutput(torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]], requires_grad=True), torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]))
        batch = Batch({'input': torch.zeros(3, 2), 'target': torch.tensor([0, 1, 0]), 'index': torch.tensor([10, 20, 30])})
        with patch('lnl_toolbox.algorithms.lend.algorithm.forward_with_features', return_value=featured):
            result = algorithm.step(batch, RunState(cycle=0))
        self.assertEqual(result.metrics['graph_degree_min'], 0.0)
        self.assertEqual(algorithm.private_state.history.initialized.tolist(), [True] * 3)
        self.assertEqual(result.metadata['diluted_labels'][2].argmax().item(), 0)

    def test_optimizer_failure_does_not_commit_history(self):
        algorithm = self._algorithm(beta=0.9, lr=0.1)
        history = algorithm.private_state.history
        before = history.state_dict()
        batch = Batch({'input': torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]), 'target': torch.zeros(3, dtype=torch.long), 'index': torch.tensor([10, 20, 30])})
        with patch.object(algorithm.optimizer, 'step', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                algorithm.step(batch, RunState(cycle=0))
        self.assertTrue(torch.equal(history.values, before['values']))
        self.assertTrue(torch.equal(history.initialized, before['initialized']))
        self.assertTrue(torch.equal(history.last_updated_epoch, before['last_updated_epoch']))

    def test_clean_target_payload_is_ignored(self):
        left, right = (self._algorithm(beta=0.9, lr=0.0), self._algorithm(beta=0.9, lr=0.0))
        inputs = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        common = {'input': inputs, 'target': torch.tensor([0, 1, 0]), 'index': torch.tensor([10, 20, 30])}
        left.step(Batch({**common, 'clean_target': torch.tensor([0, 0, 0])}), RunState(cycle=0))
        right.step(Batch({**common, 'clean_target': torch.tensor([1, 1, 1])}), RunState(cycle=0))
        for name, value in left.model.state_dict().items():
            torch.testing.assert_close(value, right.model.state_dict()[name])

# --- merged from test_lend_dilution.py ---
import unittest

# --- merged from test_lend_dilution.py ---
import torch

# --- merged from test_lend_dilution.py ---
from lnl_toolbox.algorithms.lend import dilute_labels, select_lend_samples

# --- merged from test_lend_dilution.py ---
class _lend_dilution_LENDDilutionTest(unittest.TestCase):

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
        graph = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
        result = dilute_labels(labels, graph, alpha=0.5, steps=1)
        self.assertFalse(torch.allclose(result.sum(1), torch.ones(2)))

    def test_selection_uses_noisy_label_and_first_index_tie(self):
        noisy = torch.tensor([0, 1, 1])
        history = torch.tensor([[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]])
        self.assertEqual(select_lend_samples(noisy, history).tolist(), [True, True, False])

# --- merged from test_lend_graph.py ---
import unittest

# --- merged from test_lend_graph.py ---
import torch

# --- merged from test_lend_graph.py ---
from lnl_toolbox.algorithms.lend import build_lend_similarity, dilute_labels, normalize_lend_graph

# --- merged from test_lend_graph.py ---
class _lend_graph_LENDGraphTest(unittest.TestCase):

    def test_equation_one_and_normalization_hand_calculation(self):
        features = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        indices = torch.tensor([10, 20, 30])
        adjacency = build_lend_similarity(features, indices, k=2, gamma=1.0, metric='inner_product', normalize_features=False)
        expected = torch.tensor([[0.0, 3.0, 3.0], [3.0, 0.0, 4.0], [3.0, 4.0, 0.0]])
        torch.testing.assert_close(adjacency, expected)
        product = expected.T @ expected
        degree = product.sum(1)
        manual = degree.rsqrt()[:, None] * product * degree.rsqrt()[None, :]
        graph = normalize_lend_graph(adjacency)
        torch.testing.assert_close(graph, manual)
        torch.testing.assert_close(graph, graph.T)

    def test_positive_clamp_directed_knn_and_stable_tie(self):
        features = torch.tensor([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
        indices = torch.tensor([30, 10, 20])
        adjacency = build_lend_similarity(features, indices, k=1, gamma=2.0, metric='inner_product', normalize_features=False)
        self.assertTrue(torch.equal(adjacency.diag(), torch.zeros(3)))
        self.assertTrue(bool((adjacency >= 0).all()))
        self.assertEqual(adjacency[0, 1].item(), 1.0)
        self.assertEqual(adjacency[2].sum().item(), 0.0)

    def test_permutation_equivariance(self):
        features = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
        indices = torch.tensor([90, 10, 40, 20])
        original = build_lend_similarity(features, indices, k=2, gamma=1.0, metric='inner_product', normalize_features=False)
        order = torch.tensor([2, 0, 3, 1])
        permuted = build_lend_similarity(features[order], indices[order], k=2, gamma=1.0, metric='inner_product', normalize_features=False)
        torch.testing.assert_close(permuted, original[order][:, order])

    def test_one_zero_degree_node_uses_zero_inverse(self):
        adjacency = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        graph = normalize_lend_graph(adjacency)
        torch.testing.assert_close(graph, torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0])))
        self.assertTrue(bool(torch.isfinite(graph).all()))

    def test_multiple_zero_degree_nodes_are_finite_and_deterministic(self):
        adjacency = torch.zeros(4, 4)
        first = normalize_lend_graph(adjacency)
        second = normalize_lend_graph(adjacency)
        torch.testing.assert_close(first, torch.zeros_like(adjacency))
        torch.testing.assert_close(second, first)
        torch.testing.assert_close(first, first.T)

    def test_zero_degree_dilution_retains_noisy_argmax(self):
        adjacency = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        graph = normalize_lend_graph(adjacency)
        noisy = torch.eye(3)
        first = dilute_labels(noisy, graph, alpha=0.99, steps=10)
        second = dilute_labels(noisy, graph, alpha=0.99, steps=10)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[2], (1.0 - 0.99) ** 10 * noisy[2])
        self.assertEqual(first[2].argmax().item(), 2)
        self.assertTrue(bool(torch.isfinite(first).all()))

    def test_non_finite_and_negative_adjacency_remain_invalid(self):
        for value in (float('nan'), float('inf'), -1.0):
            adjacency = torch.zeros(3, 3)
            adjacency[0, 1] = value
            with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
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

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_matches_cpu(self):
        features = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        indices = torch.tensor([1, 2, 3])
        cpu = build_lend_similarity(features, indices, k=2, gamma=1.0, metric='inner_product', normalize_features=False)
        cuda = build_lend_similarity(features.cuda(), indices.cuda(), k=2, gamma=1.0, metric='inner_product', normalize_features=False)
        torch.testing.assert_close(cuda.cpu(), cpu)

# --- merged from test_lend_history.py ---
import unittest

# --- merged from test_lend_history.py ---
import torch

# --- merged from test_lend_history.py ---
from lnl_toolbox.algorithms.lend.history import LENDLabelHistory

# --- merged from test_lend_history.py ---
class _lend_history_LENDHistoryTest(unittest.TestCase):

    def test_arbitrary_indices_first_observation_and_equation_five(self):
        history = LENDLabelHistory(torch.tensor([40, 10, 90]), 2)
        indices = torch.tensor([90, 10])
        current = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
        first = history.propose(indices, current, epoch=0, beta=0.9)
        torch.testing.assert_close(first.values, current)
        history.commit(first)
        next_value = torch.tensor([[0.6, 0.4], [0.1, 0.9]])
        second = history.propose(indices.flip(0), next_value, epoch=1, beta=0.75)
        expected = 0.25 * next_value + 0.75 * current.flip(0)
        torch.testing.assert_close(second.values, expected)
        history.commit(second)
        untouched = torch.searchsorted(history.canonical_sample_indices, torch.tensor([40]))
        self.assertFalse(history.initialized[untouched].item())

    def test_duplicate_same_epoch_missing_index_and_mapping_drift_fail(self):
        history = LENDLabelHistory(torch.tensor([2, 7]), 2)
        proposal = history.propose(torch.tensor([2]), torch.tensor([[1.0, 0.0]]), epoch=0, beta=0.9)
        history.commit(proposal)
        with self.assertRaisesRegex(ValueError, 'twice'):
            history.propose(torch.tensor([2]), torch.tensor([[1.0, 0.0]]), epoch=0, beta=0.9)
        with self.assertRaisesRegex(ValueError, 'missing'):
            history.propose(torch.tensor([8]), torch.tensor([[1.0, 0.0]]), epoch=1, beta=0.9)
        with self.assertRaisesRegex(ValueError, 'beta'):
            history.propose(torch.tensor([7]), torch.tensor([[1.0, 0.0]]), epoch=1, beta=1.1)
        state = history.state_dict()
        other = LENDLabelHistory(torch.tensor([2, 8]), 2)
        with self.assertRaisesRegex(ValueError, 'mapping changed'):
            other.load_state_dict(state)

    def test_checkpoint_roundtrip_is_deep(self):
        history = LENDLabelHistory(torch.tensor([5, 1]), 3)
        proposal = history.propose(torch.tensor([5]), torch.tensor([[0.2, 0.3, 0.4]]), epoch=0, beta=0.9)
        history.commit(proposal)
        state = history.state_dict()
        restored = LENDLabelHistory(torch.tensor([1, 5]), 3)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.values, history.values))
        state['values'].zero_()
        self.assertFalse(torch.equal(restored.values, state['values']))

# --- merged from test_lend_workflow.py ---
import copy

# --- merged from test_lend_workflow.py ---
import hashlib

# --- merged from test_lend_workflow.py ---
import json

# --- merged from test_lend_workflow.py ---
import tempfile

# --- merged from test_lend_workflow.py ---
import unittest

# --- merged from test_lend_workflow.py ---
from pathlib import Path

# --- merged from test_lend_workflow.py ---
from unittest.mock import patch

# --- merged from test_lend_workflow.py ---
import numpy as np

# --- merged from test_lend_workflow.py ---
import torch

# --- merged from test_lend_workflow.py ---
from lnl_toolbox.data.cifar import CifarData

# --- merged from test_lend_workflow.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_lend_workflow.py ---
def _lend_workflow__cifar(size, split, classes=10):
    rng = np.random.default_rng(401 + size + classes)
    images = rng.integers(1, 255, size=(size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(images, labels, tuple(map(str, range(classes))), split, f'cifar{classes}')

# --- merged from test_lend_workflow.py ---
def _lend_workflow__config(epochs=2, dataset='cifar10'):
    classes = 10 if dataset == 'cifar10' else 100
    return {'method': 'lend', 'execution': {'runner': 'lend'}, 'seed': 7, 'data': {'name': dataset, 'root': 'unused', 'validation_size': classes, 'max_train_samples': classes * 2, 'max_validation_samples': classes, 'max_test_samples': classes, 'augment': False}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 17, 'validation_targets': 'noisy'}, 'loss': {'name': 'ce'}, 'model': {'name': 'tiny_cnn', 'width': 2}, 'optimizer': {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0}, 'scheduler': {'name': 'none'}, 'loader': {'batch_size': classes, 'num_workers': 0, 'pin_memory': False, 'drop_last': False}, 'lend': {'graph': {'k': classes - 1, 'gamma': 1.0, 'metric': 'inner_product', 'normalize_features': False}, 'dilution': {'alpha': 0.99, 'policy': 'fixed_steps', 'steps': 2}, 'history': {'beta': 0.9, 'first_observation': 'current'}, 'selection': {'rule': 'noisy_equals_diluted_argmax', 'reduction': 'batch_mean', 'empty_batch': 'skip_update'}, 'training': {'epochs': epochs}}, 'evaluation': {'selection_split': 'validation', 'primary': 'accuracy'}, 'trainer': {'device': 'cpu'}}

# --- merged from test_lend_workflow.py ---
def _lend_workflow__sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# --- merged from test_lend_workflow.py ---
def _lend_workflow__nested_equal(test, left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _lend_workflow__nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            _lend_workflow__nested_equal(test, a, b)
    else:
        test.assertEqual(left, right)

# --- merged from test_lend_workflow.py ---
class _lend_workflow_LENDWorkflowTest(unittest.TestCase):

    def _run_case(self, dataset='cifar10'):
        classes = 10 if dataset == 'cifar10' else 100
        train, test = (_lend_workflow__cifar(classes * 4, 'train', classes), _lend_workflow__cifar(classes * 2, 'test', classes))
        loader_name = 'load_cifar10' if dataset == 'cifar10' else 'load_cifar100'
        with tempfile.TemporaryDirectory() as directory, patch(f'lnl_toolbox.data.sources.{loader_name}', side_effect=lambda _root, split: train if split == 'train' else test):
            run_dir = run_experiment(_lend_workflow__config(1, dataset), Path(directory) / 'resumed')
            first = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(first['completed_epoch'], 0)
            manifest_hash = _lend_workflow__sha(run_dir / 'noise_manifest.npz')
            manifest_mtime = (run_dir / 'noise_manifest.npz').stat().st_mtime_ns
            run_experiment(_lend_workflow__config(2, dataset), resume=run_dir / 'last.pt')
            resumed = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            uninterrupted_dir = run_experiment(_lend_workflow__config(2, dataset), Path(directory) / 'fresh')
            uninterrupted = torch.load(uninterrupted_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(resumed['completed_epoch'], 1)
            self.assertEqual(resumed['run_state']['phase'], 'completed')
            self.assertEqual(resumed['run_state']['step'], 4)
            _lend_workflow__nested_equal(self, resumed['model'], uninterrupted['model'])
            _lend_workflow__nested_equal(self, resumed['optimizer'], uninterrupted['optimizer'])
            _lend_workflow__nested_equal(self, resumed['algorithm_private_state'], uninterrupted['algorithm_private_state'])
            epoch_rows = lambda path: [json.loads(line) for line in path.read_text().splitlines() if json.loads(line)['event'] == 'epoch']
            self.assertEqual(epoch_rows(run_dir / 'metrics.jsonl'), epoch_rows(uninterrupted_dir / 'metrics.jsonl'))
            self.assertEqual(_lend_workflow__sha(run_dir / 'noise_manifest.npz'), manifest_hash)
            self.assertEqual((run_dir / 'noise_manifest.npz').stat().st_mtime_ns, manifest_mtime)
            tracked = {name: (_lend_workflow__sha(run_dir / name), (run_dir / name).stat().st_mtime_ns) for name in ('last.pt', 'metrics.jsonl', 'noise_manifest.npz')}
            run_experiment(_lend_workflow__config(2, dataset), resume=run_dir / 'last.pt')
            self.assertEqual(tracked, {name: (_lend_workflow__sha(run_dir / name), (run_dir / name).stat().st_mtime_ns) for name in tracked})
            final = json.loads((run_dir / 'final_metrics.json').read_text())
            self.assertIn('clean_test_accuracy', final)
            self.assertEqual(final['fidelity'], 'paper_oriented')

    def test_cifar10_fresh_resume_and_completed_noop(self):
        self._run_case()

    def test_cifar100_lightweight_workflow(self):
        self._run_case('cifar100')

    def test_resume_rejects_graph_drift(self):
        train, test = (_lend_workflow__cifar(40, 'train'), _lend_workflow__cifar(20, 'test'))
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=lambda _root, split: train if split == 'train' else test):
            run_dir = run_experiment(_lend_workflow__config(1), Path(directory) / 'run')
            changed = copy.deepcopy(_lend_workflow__config(2))
            changed['lend']['graph']['gamma'] = 2.0
            with self.assertRaisesRegex(ValueError, 'LEND settings'):
                run_experiment(changed, resume=run_dir / 'last.pt')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_lightweight_workflow(self):
        train, test = (_lend_workflow__cifar(40, 'train'), _lend_workflow__cifar(20, 'test'))
        config = _lend_workflow__config(1)
        config['trainer']['device'] = 'cuda'
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=lambda _root, split: train if split == 'train' else test):
            run_dir = run_experiment(config, Path(directory) / 'cuda')
            final = json.loads((run_dir / 'final_metrics.json').read_text())
            self.assertGreater(final['max_cuda_memory_mb'], 0.0)
