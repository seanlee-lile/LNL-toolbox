from __future__ import annotations
import unittest
import torch
from lnl_toolbox.selectors.lend import KNNLabelPropagationSelector, propagate_knn_labels

class _knn_KNNPropagationTest(unittest.TestCase):

    def test_batch_local_dilution_is_a_distribution(self) -> None:
        features = torch.eye(4)
        targets = torch.tensor([0, 1, 2, 1])
        diluted = propagate_knn_labels(features, targets, neighbors=2, num_classes=3, diffusion_steps=2)
        self.assertEqual(tuple(diluted.shape), (4, 3))
        torch.testing.assert_close(diluted.sum(1), torch.ones(4))

    def test_selector_returns_complementary_mask(self) -> None:
        result = KNNLabelPropagationSelector(neighbors=2, num_classes=3).select(features=torch.eye(4), noisy_targets=torch.tensor([0, 1, 2, 1]))
        torch.testing.assert_close(result.rejected_mask, ~result.selected_mask)

import copy
from unittest.mock import patch
from torch import nn
from lnl_toolbox.algorithms.lend import LENDAlgorithm, LENDConfig
from lnl_toolbox.algorithms.lend.graph import build_lend_similarity
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.models.cifar_resnet import cifar_resnet18
from lnl_toolbox.models.feature_output import FeatureOutput

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

def _lend_algorithm__config(beta=1.0):
    return {'method': 'lend', 'data': {'name': 'cifar10', 'max_train_samples': 3}, 'model': {'name': 'tiny_cnn'}, 'loader': {'batch_size': 3, 'drop_last': False}, 'loss': {'name': 'ce'}, 'lend': {'graph': {'k': 2, 'gamma': 1.0, 'metric': 'inner_product', 'normalize_features': False}, 'dilution': {'alpha': 0.99, 'policy': 'fixed_steps', 'steps': 1}, 'history': {'beta': beta, 'first_observation': 'current'}, 'selection': {'rule': 'noisy_equals_diluted_argmax', 'reduction': 'batch_mean', 'empty_batch': 'skip_update'}, 'training': {'epochs': 2}}}

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
        self.assertAlmostEqual(result.metrics['selected_train_loss_sum'], (losses[0] + losses[1] + losses[3]).item(), places=6)
        self.assertAlmostEqual(result.metrics['selected_train_objective'], ((losses[0] + losses[1] + losses[3]) / 4).item(), places=6)

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
        self.assertAlmostEqual(result.metrics['selected_train_loss_sum'], expected.item() * 3, places=6)
        self.assertAlmostEqual(result.metrics['selected_train_objective'], expected.item(), places=6)
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
        self.assertAlmostEqual(result.metrics['selected_train_loss_sum'], expected.item() * 3, places=6)
        self.assertAlmostEqual(result.metrics['selected_train_objective'], expected.item(), places=6)

    def test_partial_selection_uses_batch_size_not_selected_count(self):
        algorithm = self._algorithm(lr=0.0)
        history = algorithm.private_state.history
        history.values[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        history.initialized[:] = True
        inputs = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        targets = torch.zeros(3, dtype=torch.long)
        losses = nn.CrossEntropyLoss(reduction='none')(algorithm.model(inputs), targets)
        result = algorithm.step(Batch({'input': inputs, 'target': targets, 'index': torch.tensor([10, 20, 30])}), RunState(cycle=0))
        self.assertAlmostEqual(result.metrics['selected_train_loss_sum'], (losses[0] + losses[2]).item(), places=6)
        self.assertAlmostEqual(result.metrics['selected_train_objective'], ((losses[0] + losses[2]) / 3).item(), places=6)
        self.assertNotAlmostEqual(result.metrics['selected_train_objective'], ((losses[0] + losses[2]) / 2).item(), places=6)

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
        from pathlib import Path
        import yaml
        formal = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/experiment/lend_cifar10_reproduction.yaml').read_text(encoding='utf-8'))
        config['lend']['dilution'] = formal['lend']['dilution']
        algorithm = LENDAlgorithm(model=model, optimizer=optimizer, loss=nn.CrossEntropyLoss(reduction='none'), device=torch.device('cpu'), method_config=LENDConfig.from_mapping(config), canonical_global_indices=torch.arange(256), num_classes=10)
        algorithm.setup(ExperimentContext(__import__('pathlib').Path('.')))
        inputs = torch.randn(256, 3, 32, 32)
        targets = torch.arange(256) % 10
        before = torch.cat([value.detach().flatten() for value in model.parameters()])
        classifier_weight_before = model.classifier.weight.detach().clone()
        classifier_before = classifier_weight_before.norm()
        result = algorithm.step(Batch({'input': inputs, 'target': targets, 'index': torch.arange(256)}), RunState(cycle=0))
        self.assertEqual(result.metrics['dilution_converged'], 1.)
        self.assertLessEqual(result.metrics['dilution_residual'], formal['lend']['dilution']['tolerance'])
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
        left, right = (self._algorithm(beta=0.9, lr=0.1), self._algorithm(beta=0.9, lr=0.1))
        for algorithm in (left, right):
            algorithm.private_state.history.values[:] = torch.tensor([[1.0, 0.0]] * 3)
            algorithm.private_state.history.initialized[:] = True
        before = copy.deepcopy(left.model.state_dict())
        inputs = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        common = {'input': inputs, 'target': torch.tensor([0, 1, 0]), 'index': torch.tensor([10, 20, 30])}
        first = left.step(Batch({**common, 'clean_target': torch.tensor([0, 0, 0])}), RunState(cycle=0))
        second = right.step(Batch({**common, 'clean_target': torch.tensor([1, 1, 1])}), RunState(cycle=0))
        self.assertGreater(first.metrics['selected_samples'], 0)
        self.assertEqual(left.private_state.optimizer_steps, 1)
        self.assertTrue(any(not torch.equal(value, before[name])
                            for name, value in left.model.state_dict().items()))
        for name, value in left.model.state_dict().items():
            torch.testing.assert_close(value, right.model.state_dict()[name])
        for parameter, other in zip(left.model.parameters(), right.model.parameters()):
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, other.grad)
            torch.testing.assert_close(left.optimizer.state[parameter]['momentum_buffer'],
                                       right.optimizer.state[other]['momentum_buffer'])
        self.assertEqual(first.metrics, second.metrics)
        for key in first.metadata:
            torch.testing.assert_close(first.metadata[key], second.metadata[key])
        for key in ('values', 'initialized', 'last_updated_epoch'):
            torch.testing.assert_close(getattr(left.private_state.history, key),
                                       getattr(right.private_state.history, key))


from lnl_toolbox.algorithms.lend import dilute_labels, select_lend_samples

class LENDConvergenceAndScratchTest(unittest.TestCase):
    def test_until_converged_matches_known_fixed_point_and_reports_residual(self):
        from lnl_toolbox.algorithms.lend.dilution import dilute_labels
        labels = torch.eye(2, dtype=torch.float64)
        graph = torch.full((2, 2), .5, dtype=torch.float64)
        diagnostics = {}
        result = dilute_labels(labels, graph, alpha=.99, policy='until_converged', tolerance=1e-8,
                               max_steps=1000, diagnostics=diagnostics)
        torch.testing.assert_close(result, torch.full_like(labels, .5), atol=1e-8, rtol=0)
        self.assertEqual(diagnostics['converged'], 1.)
        self.assertEqual(diagnostics['initial_classes'], 2.)
        self.assertEqual(diagnostics['final_classes'], 2.)
        self.assertEqual(diagnostics['dominant_class_fraction'], .5)
        self.assertLessEqual(diagnostics['residual'], 1e-8)
        self.assertLess(diagnostics['iterations'], 10)
        self.assertFalse(result.requires_grad)
        torch.testing.assert_close(labels, torch.eye(2, dtype=torch.float64))

    def test_cap_is_a_failure_not_silent_fixed_steps(self):
        from lnl_toolbox.algorithms.lend.dilution import dilute_labels
        labels = torch.eye(2, dtype=torch.float64)
        graph = torch.tensor([[.99, .01], [.01, .99]], dtype=torch.float64)
        diagnostics = {}
        with self.assertRaisesRegex(RuntimeError, 'did not converge after 2 iterations.*residual=.*tolerance='):
            dilute_labels(labels, graph, alpha=.99, policy='until_converged', tolerance=1e-8,
                          max_steps=2, diagnostics=diagnostics)
        self.assertEqual(diagnostics['converged'], 0.)
        self.assertEqual(diagnostics['iterations'], 2.)
        self.assertGreater(diagnostics['residual'], 1e-8)
        self.assertTrue(torch.isfinite(dilute_labels(labels, graph, alpha=.99, steps=2)).all())

    def test_invalid_convergence_settings_are_rejected(self):
        from lnl_toolbox.algorithms.lend.dilution import dilute_labels
        for options in ({'tolerance': 0.}, {'tolerance': float('nan')}, {'max_steps': 0},
                        {'max_steps': True}, {'max_steps': 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                dilute_labels(torch.eye(2), torch.eye(2), alpha=.99, policy='until_converged', **options)

    def test_scratch_and_formal_share_exact_dilution_without_row_normalization(self):
        from lnl_toolbox.algorithms.lend.dilution import dilute_labels
        from lnl_toolbox.algorithms.lend.graph import normalize_lend_graph
        from lnl_toolbox.scratch import ScratchContext
        from lnl_toolbox.scratch.blocks.graph import propagate_labels
        adjacency = torch.tensor([[0., 2., 1.], [1., 0., .5], [3., 1., 0.]], dtype=torch.float64)
        graph = normalize_lend_graph(adjacency)
        targets = torch.tensor([0, 1, 0])
        for policy in ('fixed_steps', 'until_converged'):
            context = ScratchContext(graph=graph, labels=targets)
            propagate_labels(context, num_classes=2, alpha=.99, policy=policy, steps=7,
                             tolerance=1e-6, max_steps=1000)
            diagnostics = {}
            expected = dilute_labels(torch.nn.functional.one_hot(targets, 2).to(graph), graph,
                                     alpha=.99, policy=policy, steps=7, tolerance=1e-6,
                                     max_steps=1000, diagnostics=diagnostics)
            torch.testing.assert_close(context['diluted_labels'], expected, atol=0, rtol=0)
            self.assertEqual(context['diluted_labels_diagnostics'], diagnostics)
            self.assertFalse(torch.allclose(expected.sum(1), torch.ones(3, dtype=torch.float64)))

    def test_scratch_empty_selection_preserves_parameters_and_existing_sgd_momentum(self):
        from lnl_toolbox.scratch import ScratchContext
        from lnl_toolbox.scratch.executor import execute_steps
        import yaml
        from pathlib import Path
        recipe = yaml.safe_load((Path(__file__).resolve().parents[1] / 'src/lnl_toolbox/scratch/recipes/papers/lend.yaml').read_text(encoding='utf-8'))
        batch_steps = next(step for step in next(s for s in recipe['steps'] if s['block'] == 'epoch_loop')['steps']
                           if step['block'] == 'batch_loop')['steps']
        condition = next(step for step in batch_steps if step['block'] == 'if_mask_any')
        self.assertLess(next(i for i, s in enumerate(batch_steps) if s['block'] == 'indexed_ema'), batch_steps.index(condition))
        self.assertEqual(next(s for s in batch_steps if s['block'] == 'pairwise_similarity')['params']['metric'], 'inner_product')
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005)
        inputs = torch.ones(3, 2)
        model(inputs).square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before_model, before_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        context = ScratchContext(model=model, optimizer=optimizer, lend_selected=torch.zeros(3, dtype=torch.bool),
                                 loss=model(inputs).square().sum() * 0.)
        execute_steps([condition], context)
        _lend_workflow__nested_equal(self, model.state_dict(), before_model)
        _lend_workflow__nested_equal(self, optimizer.state_dict(), before_optimizer)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        context.update(lend_selected=torch.ones(3, dtype=torch.bool), loss=model(inputs).square().sum())
        execute_steps([condition], context)
        self.assertTrue(any(not torch.equal(value, before_model[name]) for name, value in model.state_dict().items()))

    def test_web_can_save_and_parse_both_policies_with_correct_default_sources(self):
        from web import command_console
        import tempfile, yaml
        from pathlib import Path
        fields = {f['path']: f for f in command_console._config_schema('lend-cifar10-reproduction')['fields']}
        self.assertTrue(fields['lend.dilution.policy']['editable'])
        self.assertEqual([choice['value'] for choice in fields['lend.dilution.policy']['choices']], ['until_converged', 'fixed_steps'])
        for policy in ('fixed_steps', 'until_converged'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
                saved = command_console._save_config({'recipe': 'lend-cifar10-reproduction',
                         'path': str(Path(directory) / 'edited.yaml'),
                         'patches': [{'path': 'lend.dilution.policy', 'value': policy}], 'acknowledge_paper_impact': True})
                config = yaml.safe_load((command_console.ROOT / saved['path']).read_text(encoding='utf-8'))
                parsed = LENDConfig.from_mapping(config)
                self.assertEqual(parsed.dilution_policy, policy)
                self.assertEqual(parsed.dilution_max_steps, fields['lend.dilution.max_steps']['value'])


    def test_algorithm_records_convergence_before_selection_and_update(self):
        algorithm = _lend_algorithm_LENDAlgorithmTest()._algorithm(beta=.9, lr=.01)
        config = _lend_algorithm__config(beta=.9)
        config['lend']['dilution'] = {'alpha': .99, 'policy': 'until_converged', 'tolerance': 1e-5, 'max_steps': 1000}
        algorithm.method_config = LENDConfig.from_mapping(config)
        result = algorithm.step(Batch({'input': torch.tensor([[1., 1.], [2., 1.], [1., 2.]]),
                                      'target': torch.tensor([0, 1, 0]), 'index': torch.tensor([10, 20, 30])}), RunState())
        self.assertEqual(result.metrics['dilution_converged'], 1.)
        self.assertLessEqual(result.metrics['dilution_residual'], 1e-5)
        self.assertGreaterEqual(result.metrics['dilution_iterations'], 1.)

    def test_controlled_converged_training_resume_matches_uninterrupted(self):
        import tempfile
        from pathlib import Path
        train, test = _lend_workflow__cifar(60, 'train'), _lend_workflow__cifar(20, 'test')
        config = _lend_workflow__config(1)
        config['lend']['dilution'] = {'alpha': .99, 'policy': 'until_converged', 'tolerance': 1e-5, 'max_steps': 1000}
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10',
                                                             side_effect=lambda root, split: train if split == 'train' else test):
            run_dir = run_experiment(config, Path(directory) / 'resumed')
            config['lend']['training']['epochs'] = 2
            run_experiment(config, resume=run_dir / 'last.pt')
            fresh = run_experiment(config, Path(directory) / 'fresh')
            resumed = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            continuous = torch.load(fresh / 'last.pt', map_location='cpu', weights_only=False)
            for key in ('model', 'optimizer', 'algorithm_private_state'):
                _lend_workflow__nested_equal(self, resumed[key], continuous[key])


class _lend_dilution_LENDDilutionTest(unittest.TestCase):

    def test_numeric_convergence_also_reports_class_collapse(self):
        labels = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], dtype=torch.float64)
        graph = torch.full((3, 3), 1 / 3, dtype=torch.float64)
        diagnostics = {}
        dilute_labels(labels, graph, alpha=.99, policy='until_converged',
                      tolerance=1e-8, diagnostics=diagnostics)
        self.assertEqual(diagnostics['converged'], 1.)
        self.assertEqual(diagnostics['initial_classes'], 2.)
        self.assertEqual(diagnostics['final_classes'], 1.)
        self.assertEqual(diagnostics['dominant_class_fraction'], 1.)

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


from lnl_toolbox.algorithms.lend import build_lend_similarity, normalize_lend_graph

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


from lnl_toolbox.algorithms.lend.history import LENDLabelHistory

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


import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.training.experiment import run_experiment

def _lend_workflow__cifar(size, split, classes=10):
    rng = np.random.default_rng(401 + size + classes)
    images = rng.integers(1, 255, size=(size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(images, labels, tuple(map(str, range(classes))), split, f'cifar{classes}')

def _lend_workflow__config(epochs=2, dataset='cifar10'):
    classes = 10 if dataset == 'cifar10' else 100
    return {'method': 'lend', 'execution': {'runner': 'lend'}, 'seed': 7, 'data': {'name': dataset, 'root': 'unused', 'validation_size': classes, 'max_train_samples': classes * 2, 'max_validation_samples': classes, 'max_test_samples': classes, 'augment': False}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 17, 'validation_targets': 'noisy'}, 'loss': {'name': 'ce'}, 'model': {'name': 'tiny_cnn', 'width': 2}, 'optimizer': {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0}, 'scheduler': {'name': 'none'}, 'loader': {'batch_size': classes, 'num_workers': 0, 'pin_memory': False, 'drop_last': False}, 'lend': {'graph': {'k': classes - 1, 'gamma': 1.0, 'metric': 'inner_product', 'normalize_features': False}, 'dilution': {'alpha': 0.99, 'policy': 'fixed_steps', 'steps': 2}, 'history': {'beta': 0.9, 'first_observation': 'current'}, 'selection': {'rule': 'noisy_equals_diluted_argmax', 'reduction': 'batch_mean', 'empty_batch': 'skip_update'}, 'training': {'epochs': epochs}}, 'evaluation': {'selection_split': 'validation', 'primary': 'accuracy'}, 'trainer': {'device': 'cpu'}}

def _lend_workflow__sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

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

def _author_release_reference(features, current, k):
    """Independent CPU translation of release code/main.py:42-69 (no imports/I/O)."""
    square = (features ** 2).sum(1, keepdim=True)
    nearest = (square + square.t() - 2 * features @ features.t()).topk(k=k, largest=False).indices
    adjacency = ((features @ features.t()) / features.size(0)).clamp(min=0) ** 3
    eye = torch.eye(features.size(0), device=features.device, dtype=features.dtype)
    adjacency = adjacency * (1 - eye)
    mask = torch.ones_like(adjacency)
    for row in range(nearest.size(0)):
        mask[row][nearest[row]] = 0
    adjacency *= 1 - mask
    graph = adjacency @ adjacency.t()
    diagonal = (1 / graph.sum(1, keepdim=True).sqrt()) * eye
    graph = diagonal @ graph @ diagonal
    result = (eye - .99 * graph).inverse() @ current
    if bool((result != result).any()):
        result = current
    return adjacency, graph, nearest, result.softmax(1)


class LENDAuthorReleaseTest(unittest.TestCase):
    def _config(self, variant='author_release'):
        config = _lend_algorithm__config(beta=.9)
        config['lend']['implementation_variant'] = variant
        return config

    def _algorithm(self, variant='author_release', lr=.01):
        torch.manual_seed(37)
        model = _lend_algorithm_FeatureModel()
        algorithm = LENDAlgorithm(
            model=model, optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=.9),
            loss=nn.CrossEntropyLoss(reduction='none'), device=torch.device('cpu'),
            method_config=LENDConfig.from_mapping(self._config(variant)),
            canonical_global_indices=torch.tensor([10, 20, 30]), num_classes=2,
        )
        return algorithm

    def _batch(self, order=None):
        inputs = torch.tensor([[1., 1.], [2., 1.], [1., 2.]])
        targets = torch.tensor([0, 1, 0])
        indices = torch.tensor([10, 20, 30])
        order = torch.arange(3) if order is None else order
        return Batch({'input': inputs[order], 'target': targets[order], 'index': indices[order]})

    def test_variant_defaults_unknown_and_catalog(self):
        from lnl_toolbox.catalog import validate_config
        paper = _lend_algorithm__config()
        explicit = copy.deepcopy(paper)
        explicit['lend']['implementation_variant'] = 'paper_equation'
        self.assertEqual(LENDConfig.from_mapping(paper), LENDConfig.from_mapping(explicit))
        config = _lend_workflow__config()
        config['lend']['implementation_variant'] = 'author_release'
        self.assertEqual(validate_config(config, check_data=False).name, 'lend')
        config['lend']['implementation_variant'] = 'typo'
        with self.assertRaisesRegex(ValueError, 'implementation_variant'):
            validate_config(config, check_data=False)

    def test_effective_constants_and_inactive_controls(self):
        config = self._config()
        config['lend']['selection']['reduction'] = 'sum'
        method = LENDConfig.from_mapping(config)
        effective = method.implementation_metadata()
        self.assertEqual(effective['gamma'], 3)
        self.assertEqual(effective['metric'], 'squared_euclidean')
        self.assertEqual(effective['orientation'], 'AAT')
        self.assertEqual(effective['lambda'], .99)
        self.assertEqual(effective['softmax_scale'], 1.)
        self.assertEqual(effective['reduction'], 'batch_mean')
        self.assertEqual(effective['k'], method.k)
        self.assertEqual(effective['beta'], method.beta)
        self.assertIn('lend.selection', effective['inactive_config_fields'])

    def test_graph_and_solve_match_author_reference(self):
        from lnl_toolbox.algorithms.lend.graph import build_author_release_graph
        from lnl_toolbox.algorithms.lend.dilution import author_release_propagation
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                features = torch.tensor([[1., 0.], [1., .3], [.5, 1.], [.2, 2.], [1.2, .5]], dtype=dtype)
                current = torch.eye(5, dtype=dtype)[:, :2] + .1
                a, w, neighbors, expected = _author_release_reference(features, current, 3)
                actual_a, actual_w, actual_neighbors = build_author_release_graph(features, torch.arange(5), k=3)
                torch.testing.assert_close(actual_neighbors, neighbors)
                torch.testing.assert_close(actual_a, a, rtol=0, atol=0)
                torch.testing.assert_close(actual_w, w, rtol=0, atol=0)
                torch.testing.assert_close(author_release_propagation(current, actual_w), expected, rtol=1e-4, atol=1e-5)
                self.assertFalse(torch.allclose(a.T @ a, a @ a.T))
                self.assertTrue(bool((neighbors == torch.arange(5)[:, None]).any(1).all()))
                self.assertTrue(bool((a.diag() == 0).all()))

    def test_topk_duplicate_ties_match_same_device_reference(self):
        from lnl_toolbox.algorithms.lend.graph import build_author_release_graph
        features = torch.tensor([[1., 1.], [1., 1.], [2., .5], [2., .5]])
        expected = _author_release_reference(features, torch.eye(4), 2)
        actual = build_author_release_graph(features, torch.tensor([40, 30, 20, 10]), k=2)
        for left, right in zip(actual, expected[:3]):
            torch.testing.assert_close(left, right, equal_nan=True)

    def test_zero_degree_whole_batch_fallback_then_softmax(self):
        from lnl_toolbox.algorithms.lend.graph import build_author_release_graph
        from lnl_toolbox.algorithms.lend.dilution import author_release_propagation
        for features in (torch.zeros(3, 2), torch.tensor([[0., 0.], [1., 1.], [1., 2.]])):
            with self.subTest(features=features):
                current = torch.tensor([[1., 0.], [.2, .8], [0., 1.]])
                graph = build_author_release_graph(features, torch.arange(3), k=2)[1]
                diagnostics = {}
                with patch('torch.linalg.solve', side_effect=AssertionError('known NaN graph must not enter solve')):
                    actual = author_release_propagation(current, graph, diagnostics=diagnostics)
                torch.testing.assert_close(actual, _author_release_reference(features, current, 2)[3])
                torch.testing.assert_close(actual, current.softmax(1))
                self.assertEqual(diagnostics['fallback'], 1.)

    def test_solver_nan_fallback_and_exceptions_not_swallowed(self):
        from lnl_toolbox.algorithms.lend.dilution import author_release_propagation
        current = torch.eye(2)
        with patch('torch.linalg.solve', return_value=torch.full((2, 2), float('nan'))):
            torch.testing.assert_close(author_release_propagation(current, torch.eye(2)), current.softmax(1))
        with patch('torch.linalg.solve', side_effect=RuntimeError('solver failure')):
            with self.assertRaisesRegex(RuntimeError, 'solver failure'):
                author_release_propagation(current, torch.eye(2))
        with self.assertRaisesRegex(ValueError, 'infinity'):
            author_release_propagation(current, torch.full((2, 2), float('inf')))

    def test_history_zero_based_boundary_and_direct_storage(self):
        from lnl_toolbox.algorithms.lend.history import LENDLabelHistory
        indices = torch.tensor([20, 10, 30])
        history = LENDLabelHistory(indices, 2)
        onehot = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        previous = torch.tensor([[.1, .9], [.8, .2], [.3, .7]])
        history.commit(history.propose_author(indices, previous, epoch=8))
        torch.testing.assert_close(history.author_input(indices, onehot, epoch=9, beta=.9), onehot)
        history.commit(history.propose_author(indices, previous, epoch=9))
        torch.testing.assert_close(history.author_input(indices, onehot, epoch=10, beta=.9), .9 * onehot + .1 * previous)
        changed = previous.flip(1)
        history.commit(history.propose_author(indices, changed, epoch=10))
        torch.testing.assert_close(history.author_input(indices, onehot, epoch=11, beta=.9), .9 * onehot + .1 * changed)
        with self.assertRaisesRegex(ValueError, 'twice'):
            history.author_input(indices, onehot, epoch=10, beta=.9)

    def test_history_missing_or_stale_at_boundary_rejected(self):
        from lnl_toolbox.algorithms.lend.history import LENDLabelHistory
        history = LENDLabelHistory(torch.tensor([10, 20, 30]), 2)
        for epoch in (10, 11):
            with self.assertRaisesRegex(ValueError, 'previous-epoch'):
                history.author_input(torch.tensor([10, 20, 30]), torch.ones(3, 2), epoch=epoch, beta=.9)

    def test_soft_weight_loss_gradient_and_noisy_target_guard(self):
        algorithm = self._algorithm(lr=0.)
        batch = self._batch()
        logits = torch.tensor([[2., 0.], [1., 3.], [.5, 0.]], requires_grad=True)
        diluted = torch.tensor([[.3, .7], [.8, .2], [.5, .5]])
        expected_logits = logits.detach().clone().requires_grad_()
        expected_weights = diluted.gather(1, batch.payload['target'][:, None]).squeeze(1)
        expected_sum = (expected_weights * nn.CrossEntropyLoss(reduction='none')(expected_logits, batch.payload['target'])).sum()
        (expected_sum / 3).backward()
        # Retain the fixed loss/logits while using a real finite graph.
        with patch('lnl_toolbox.algorithms.lend.algorithm.forward_with_features', return_value=FeatureOutput(logits, torch.ones(3, 2))), patch('lnl_toolbox.algorithms.lend.algorithm.author_release_propagation', side_effect=lambda c, g, diagnostics: (diagnostics.update(fallback=0.) or diluted)):
            result = algorithm.step(batch, RunState(cycle=0))
        torch.testing.assert_close(result.metadata['sample_weights'], expected_weights)
        self.assertFalse(result.metadata['sample_weights'].requires_grad)
        self.assertNotIn('selected_samples', result.metrics)
        self.assertNotIn('selected_mask', result.metadata)
        self.assertAlmostEqual(result.metrics['weighted_loss_sum'], expected_sum.item(), places=6)
        self.assertAlmostEqual(result.metrics['weighted_objective'], expected_sum.item() / 3, places=6)
        torch.testing.assert_close(logits.grad, expected_logits.grad)

    def test_full_step_matches_reference_including_history_and_optimizer(self):
        algorithm = self._algorithm()
        reference_model = copy.deepcopy(algorithm.model)
        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=.01, momentum=.9)
        batch = self._batch(torch.tensor([2, 0, 1]))
        previous = torch.tensor([[.2, .8], [.8, .2], [.4, .6]])
        algorithm.private_state.history.commit(algorithm.private_state.history.propose_author(batch.payload['index'], previous, epoch=9))
        featured = reference_model.forward_with_features(batch.payload['input'])
        logits, features = featured.logits, featured.features.detach()
        onehot = torch.nn.functional.one_hot(batch.payload['target'], 2).float()
        diluted = _author_release_reference(features, .9 * onehot + .1 * previous, 2)[3].detach()
        weights = diluted.gather(1, batch.payload['target'][:, None]).squeeze(1)
        expected = (weights * nn.CrossEntropyLoss(reduction='none')(logits, batch.payload['target'])).mean()
        expected.backward()
        reference_optimizer.step()
        result = algorithm.step(batch, RunState(cycle=10))
        self.assertEqual(algorithm.model.feature_calls, 1)
        self.assertAlmostEqual(result.metrics['weighted_objective'], expected.item(), places=5)
        for left, right in zip(algorithm.model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        rows = torch.searchsorted(algorithm.private_state.history.canonical_sample_indices, batch.payload['index'])
        torch.testing.assert_close(algorithm.private_state.history.values[rows], diluted, rtol=1e-4, atol=1e-5)

    def test_optimizer_failure_does_not_commit_author_history(self):
        algorithm = self._algorithm()
        before = copy.deepcopy(algorithm.private_state.state_dict())
        with patch.object(algorithm.optimizer, 'step', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                algorithm.step(self._batch(), RunState(cycle=0))
        _lend_workflow__nested_equal(self, before, algorithm.private_state.state_dict())

    def test_clean_labels_ignored_and_diagnostics_do_not_consume_rng(self):
        a, b = self._algorithm(), self._algorithm()
        batch = self._batch()
        before = torch.get_rng_state().clone()
        result_a = a.step(batch, RunState(cycle=0))
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        batch.payload['clean_target'] = 1 - batch.payload['target']
        result_b = b.step(batch, RunState(cycle=0))
        _lend_workflow__nested_equal(self, a.state_dict(), b.state_dict())
        self.assertEqual(result_a.metrics, result_b.metrics)

    def test_same_variant_checkpoint_resume_across_history_boundary(self):
        from lnl_toolbox.training.checkpoint import save_checkpoint, load_checkpoint
        continuous = self._algorithm()
        state = RunState()
        for epoch in range(10):
            state.cycle = epoch
            continuous.step(self._batch(), state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            save_checkpoint(path, continuous, state, 9, self._config())
            restored = self._algorithm()
            restored_state, completed, _ = load_checkpoint(path, restored, torch.device('cpu'))
            self.assertEqual(completed, 9)
            for epoch in (10, 11):
                state.cycle = restored_state.cycle = epoch
                order = torch.tensor([2, 0, 1]) if epoch == 10 else torch.tensor([1, 2, 0])
                continuous.step(self._batch(order), state)
                restored.step(self._batch(order), restored_state)
            _lend_workflow__nested_equal(self, continuous.state_dict(), restored.state_dict())

    def test_legacy_paper_and_cross_variant_identity(self):
        paper = self._algorithm('paper_equation')
        legacy = copy.deepcopy(paper.state_dict())
        legacy['method_config'].pop('implementation_variant')
        saved_copy = copy.deepcopy(legacy)
        restored = self._algorithm('paper_equation')
        restored.load_state_dict(legacy)
        _lend_workflow__nested_equal(self, saved_copy, legacy)
        for source, target in ((legacy, self._algorithm()), (paper.state_dict(), self._algorithm()), (self._algorithm().state_dict(), self._algorithm('paper_equation'))):
            with self.subTest(target=target.method_config.implementation_variant):
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    target.load_state_dict(source)
        legacy['method_config'].pop('gamma')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            restored.load_state_dict(legacy)

    def test_resume_config_variant_and_other_drift(self):
        from lnl_toolbox.training.lend_experiment import _validate_resume_config
        old = _lend_workflow__config()
        explicit = copy.deepcopy(old)
        explicit['lend']['implementation_variant'] = 'paper_equation'
        _validate_resume_config(explicit, old)
        author = copy.deepcopy(old)
        author['lend']['implementation_variant'] = 'author_release'
        for current, saved in ((author, old), (old, author)):
            with self.assertRaisesRegex(ValueError, 'LEND settings'):
                _validate_resume_config(current, saved)
        changed = copy.deepcopy(author)
        changed['optimizer']['lr'] *= 2
        with self.assertRaisesRegex(ValueError, 'optimizer'):
            _validate_resume_config(changed, author)

    def test_dry_run_effective_semantics_and_original_paper_plan(self):
        from lnl_toolbox.training.planning import lend_plan
        config = _lend_workflow__config()
        old_plan = lend_plan(config, 'lend', ('lend', 'training', 'epochs'))
        config['lend']['implementation_variant'] = 'paper_equation'
        self.assertEqual(old_plan, lend_plan(config, 'lend', ('lend', 'training', 'epochs')))
        config['lend']['implementation_variant'] = 'author_release'
        text = str(lend_plan(config, 'lend', ('lend', 'training', 'epochs')))
        for value in ('author_release', 'squared_euclidean', 'gamma=3', 'AAT', 'lambda=0.99', 'softmax_scale=1', 'epoch 10', 'continuous noisy_class_probability', 'inactive inherited controls'):
            self.assertIn(value, text)
        self.assertNotIn('until_converged', text)

    def test_epoch_weight_statistics_use_global_not_batch_medians(self):
        from lnl_toolbox.training.lend_experiment import _weight_diagnostics
        weights = torch.tensor([0., 0., .2, .3, .9])
        targets = torch.tensor([0, 1, 0, 1, 1])
        stats = _weight_diagnostics(weights, targets, 3)
        self.assertAlmostEqual(stats['effective_weight_sum'], weights.double().sum().item())
        self.assertAlmostEqual(stats['sample_weight_median'], .2)
        torch.testing.assert_close(torch.tensor(stats['classwise_weight_sum']), torch.tensor([.2, 1.2, 0.]))

    def test_explicit_paper_matches_legacy_step_exactly(self):
        from dataclasses import replace
        explicit = self._algorithm('paper_equation')
        implicit = self._algorithm('paper_equation')
        implicit.method_config = LENDConfig.from_mapping(_lend_algorithm__config(beta=.9))
        for epoch in (0, 1):
            left = explicit.step(self._batch(), RunState(cycle=epoch))
            right = implicit.step(self._batch(), RunState(cycle=epoch))
            self.assertEqual(left.metrics, right.metrics)
            _lend_workflow__nested_equal(self, left.metadata, right.metadata)
            _lend_workflow__nested_equal(self, explicit.state_dict(), implicit.state_dict())
        # All-zero soft weights must still execute the author's optimizer
        # step (e.g. momentum); argmax agreement must not decide admission.
        author = self._algorithm()
        author.method_config = replace(author.method_config, reduction='sum')
        diluted = torch.tensor([[0., 1.], [1., 0.], [0., 1.]])
        with patch('lnl_toolbox.algorithms.lend.algorithm.author_release_propagation', side_effect=lambda c, g, diagnostics: (diagnostics.update(fallback=0.) or diluted)):
            result = author.step(self._batch(), RunState(cycle=0))
        self.assertEqual(author.private_state.optimizer_steps, 1)
        self.assertEqual(result.metrics['weighted_objective'], 0.)
        self.assertEqual(author.private_state.empty_selection_batches, 0)

    def test_runner_pairing_metadata_diagnostics_and_selection_unchanged(self):
        import contextlib
        import io
        import yaml
        from lnl_toolbox.training import lend_experiment
        train, test = _lend_workflow__cifar(40, 'train'), _lend_workflow__cifar(20, 'test')
        initial_models, batches = [], []
        build_model = lend_experiment.build_model
        step = LENDAlgorithm.step

        def capture_model(*args, **kwargs):
            model = build_model(*args, **kwargs)
            initial_models.append(copy.deepcopy(model.state_dict()))
            return model

        def capture_step(algorithm, batch, state):
            batches.append((state.cycle, batch.payload['index'].tolist()))
            return step(algorithm, batch, state)

        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=lambda _root, split: train if split == 'train' else test), patch.object(lend_experiment, 'build_model', side_effect=capture_model), patch.object(LENDAlgorithm, 'step', capture_step), contextlib.redirect_stdout(io.StringIO()):
            paper_config = _lend_workflow__config(2)
            author_config = copy.deepcopy(paper_config)
            author_config['lend']['implementation_variant'] = 'author_release'
            paper_dir = run_experiment(paper_config, Path(directory) / 'paper')
            paper_batches = copy.deepcopy(batches)
            batches.clear()
            author_dir = run_experiment(author_config, Path(directory) / 'author')
            self.assertEqual(paper_batches, batches)
            _lend_workflow__nested_equal(self, initial_models[0], initial_models[1])
            left = torch.load(paper_dir / 'last.pt', map_location='cpu', weights_only=False)
            right = torch.load(author_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(left['noise'], right['noise'])
            self.assertEqual(left['selection_split'], right['selection_split'])
            self.assertEqual(left['config']['data'], right['config']['data'])
            self.assertEqual(left['config']['optimizer'], right['config']['optimizer'])
            self.assertEqual(left['config']['scheduler'], right['config']['scheduler'])
            resolved = yaml.safe_load((author_dir / 'resolved_config.yaml').read_text(encoding='utf-8'))
            self.assertEqual(resolved['lend']['implementation_variant'], 'author_release')
            self.assertEqual(resolved['lend']['implementation_metadata']['gamma'], 3)
            self.assertEqual(right['algorithm_private_state']['method_config']['implementation_variant'], 'author_release')
            rows = [json.loads(line) for line in (author_dir / 'metrics.jsonl').read_text().splitlines()]
            epochs = [row for row in rows if row['event'] == 'epoch']
            for row in epochs:
                self.assertEqual(row['implementation_variant'], 'author_release')
                self.assertNotIn('selected_samples', row)
                self.assertNotIn('oracle_selection_precision', row)
                self.assertIn('sample_weight_median', row)
                self.assertIn('propagation_fallback_batches', row)
                self.assertAlmostEqual(sum(row['classwise_weight_sum']), row['effective_weight_sum'])
                self.assertAlmostEqual(row['effective_weight_sum'], row['sample_weight_mean'] * 20)
            final = rows[-1]
            self.assertEqual(final['reporting_rule'], 'noisy_validation_selected')
            self.assertEqual(final['reported_epoch'], max(epochs, key=lambda row: row['validation_accuracy'])['epoch'])
            # Resume is still the same runner; same-variant restoration never
            # changes reporting/selection or rewrites the old checkpoint in place.
            resumed_config = copy.deepcopy(author_config)
            resumed_config['lend']['training']['epochs'] = 3
            run_experiment(resumed_config, resume=author_dir / 'last.pt')
            restored = torch.load(author_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(restored['completed_epoch'], 2)
            self.assertEqual(restored['config']['lend']['implementation_variant'], 'author_release')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_author_graph_and_propagation_parity(self):
        from lnl_toolbox.algorithms.lend.graph import build_author_release_graph
        from lnl_toolbox.algorithms.lend.dilution import author_release_propagation
        features = torch.tensor([[1., 0.], [1., .3], [.5, 1.], [.2, 2.], [1.2, .5]], device='cuda')
        current = torch.eye(5, device='cuda')
        reference = _author_release_reference(features, current, 3)
        actual = build_author_release_graph(features, torch.arange(5, device='cuda'), k=3)
        for left, right in zip(actual, reference[:3]):
            torch.testing.assert_close(left, right)
        torch.testing.assert_close(author_release_propagation(current, actual[1]), reference[3], rtol=1e-4, atol=1e-5)


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
