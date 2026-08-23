"""Merged unit tests; source modules were consolidated without changing assertions."""

# --- merged from test_coteaching.py ---
import unittest

# --- merged from test_coteaching.py ---
import numpy as np

# --- merged from test_coteaching.py ---
from lnl_toolbox.algorithms import coteaching_exchange, remember_rate

# --- merged from test_coteaching.py ---
from lnl_toolbox.algorithms.coteaching import coteaching_exchange as package_coteaching_exchange

# --- merged from test_coteaching.py ---
from lnl_toolbox.algorithms.coteaching import remember_rate as package_remember_rate

# --- merged from test_coteaching.py ---
class _coteaching_CoTeachingTest(unittest.TestCase):

    def test_peer_exchange(self) -> None:
        losses_a = np.array([0.1, 0.8, 0.2, 0.7])
        losses_b = np.array([0.9, 0.1, 0.8, 0.2])
        update_a, update_b = coteaching_exchange(losses_a, losses_b, keep_rate=0.5)
        np.testing.assert_array_equal(update_a, np.array([1, 3]))
        np.testing.assert_array_equal(update_b, np.array([0, 2]))
        self.assertAlmostEqual(remember_rate(10, 0.4, 10), 0.6)

    def test_legacy_package_imports_preserve_behavior(self) -> None:
        losses_a = np.array([0.1, 0.8, 0.2, 0.7])
        losses_b = np.array([0.9, 0.1, 0.8, 0.2])
        update_a, update_b = package_coteaching_exchange(losses_a, losses_b, keep_rate=0.5)
        np.testing.assert_array_equal(update_a, np.array([1, 3]))
        np.testing.assert_array_equal(update_b, np.array([0, 2]))
        self.assertAlmostEqual(package_remember_rate(10, 0.4, 10), 0.6)

# --- merged from test_coteaching_algorithm.py ---
import copy

# --- merged from test_coteaching_algorithm.py ---
import math

# --- merged from test_coteaching_algorithm.py ---
import unittest

# --- merged from test_coteaching_algorithm.py ---
from pathlib import Path

# --- merged from test_coteaching_algorithm.py ---
import torch

# --- merged from test_coteaching_algorithm.py ---
from lnl_toolbox.algorithms.coteaching import CoTeachingAlgorithm, CoTeachingConfig, determine_keep_count, stable_small_loss_mask

# --- merged from test_coteaching_algorithm.py ---
from lnl_toolbox.core import Batch, ExperimentContext, RunState

# --- merged from test_coteaching_algorithm.py ---
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

# --- merged from test_coteaching_algorithm.py ---
from lnl_toolbox.training.coteaching_experiment import _build_peer_models

# --- merged from test_coteaching_algorithm.py ---
def _coteaching_algorithm__method_config(*, noise_rate=0.5, gradual_epochs=1):
    return {'method': 'coteaching', 'noise': {'name': 'symmetric', 'rate': noise_rate}, 'coteaching': {'model_count': 2, 'noise_rate': noise_rate, 'initialization': {'peer_seed_offset': 1}, 'remember_schedule': {'name': 'linear', 'start': 1.0, 'end': 1.0 - noise_rate, 'gradual_epochs': gradual_epochs}, 'selection': {'count_rule': 'floor', 'tie_break': 'stable_sample_index'}}, 'evaluation': {'selection_split': 'validation', 'primary': 'mean_peer_accuracy', 'ensemble': 'mean_probabilities'}}

# --- merged from test_coteaching_algorithm.py ---
class _coteaching_algorithm__IndexedLogits(torch.nn.Module):

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs):
        return self.logits[inputs.reshape(-1).long()]

# --- merged from test_coteaching_algorithm.py ---
class _coteaching_algorithm__TracingIndexedLogits(_coteaching_algorithm__IndexedLogits):

    def __init__(self, logits, name, events):
        super().__init__(logits)
        self.name = name
        self.events = events

    def forward(self, inputs):
        self.events.append(f'forward_{self.name}')
        return super().forward(inputs)

# --- merged from test_coteaching_algorithm.py ---
class _coteaching_algorithm__TracingSGD(torch.optim.SGD):

    def __init__(self, params, name, events):
        super().__init__(params, lr=0.1)
        self.name = name
        self.events = events

    def step(self, closure=None):
        self.events.append(f'step_{self.name}')
        return super().step(closure)

# --- merged from test_coteaching_algorithm.py ---
def _coteaching_algorithm__cross_update_algorithm():
    model_a = _coteaching_algorithm__IndexedLogits([[5, -5], [5, -5], [-5, 5], [-5, 5]])
    model_b = _coteaching_algorithm__IndexedLogits([[-5, 5], [-5, 5], [5, -5], [5, -5]])
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    algorithm = CoTeachingAlgorithm(model_a=model_a, model_b=model_b, optimizer_a=optimizer_a, optimizer_b=optimizer_b, scheduler_a=None, scheduler_b=None, loss=CrossEntropyLoss(), device=torch.device('cpu'), method_config=CoTeachingConfig.from_mapping(_coteaching_algorithm__method_config()))
    algorithm.setup(ExperimentContext(Path.cwd()))
    algorithm.on_cycle_start(RunState(cycle=1))
    return algorithm

# --- merged from test_coteaching_algorithm.py ---
class _coteaching_algorithm_CoTeachingSelectionTest(unittest.TestCase):

    def test_zero_based_schedule(self):
        config = CoTeachingConfig.from_mapping(_coteaching_algorithm__method_config(noise_rate=0.4, gradual_epochs=10))
        expected = {0: 1.0, 1: 0.96, 5: 0.8, 10: 0.6, 14: 0.6}
        for epoch, rate in expected.items():
            with self.subTest(epoch=epoch):
                self.assertAlmostEqual(config.rate_at(epoch), rate)

    def test_floor_count_and_minimum_one(self):
        self.assertEqual(determine_keep_count(7, 0.5), 3)
        self.assertEqual(determine_keep_count(2, 0.01), 1)

    def test_ties_use_stable_global_index_not_batch_position(self):
        losses = torch.tensor([0.2, 0.2, 0.1, 0.2])
        indices = torch.tensor([30, 10, 40, 20])
        mask = stable_small_loss_mask(losses, indices, 3)
        self.assertEqual(set(indices[mask].tolist()), {10, 20, 40})
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = stable_small_loss_mask(losses[permutation], indices[permutation], 3)
        self.assertEqual(set(indices[permutation][permuted].tolist()), {10, 20, 40})

    def test_invalid_selection_inputs_fail(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite'):
                stable_small_loss_mask(torch.tensor([0.1, value]), torch.tensor([0, 1]), 1)
        with self.assertRaisesRegex(ValueError, 'unique'):
            stable_small_loss_mask(torch.tensor([0.1, 0.2]), torch.tensor([1, 1]), 1)

# --- merged from test_coteaching_algorithm.py ---
class _coteaching_algorithm_CoTeachingAlgorithmTest(unittest.TestCase):

    def test_peer_cross_update_uses_other_models_selection(self):
        algorithm = _coteaching_algorithm__cross_update_algorithm()
        before_a = algorithm.model_a.logits.detach().clone()
        before_b = algorithm.model_b.logits.detach().clone()
        state = RunState(cycle=1)
        result = algorithm.step(Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.arange(4)}), state)
        self.assertEqual(result.metadata['selected_by_a_indices'].tolist(), [0, 1])
        self.assertEqual(result.metadata['selected_by_b_indices'].tolist(), [2, 3])
        torch.testing.assert_close(algorithm.model_a.logits[:2], before_a[:2])
        self.assertFalse(torch.equal(algorithm.model_a.logits[2:], before_a[2:]))
        self.assertFalse(torch.equal(algorithm.model_b.logits[:2], before_b[:2]))
        torch.testing.assert_close(algorithm.model_b.logits[2:], before_b[2:])
        self.assertEqual(state.step, 1)
        self.assertEqual(algorithm.private_state.optimizer_steps_a, 1)
        self.assertEqual(algorithm.private_state.optimizer_steps_b, 1)
        self.assertIsNot(algorithm.optimizer_a, algorithm.optimizer_b)

    def test_both_forwards_happen_before_either_optimizer_step(self):
        events = []
        logits_a = [[5, -5], [5, -5], [-5, 5], [-5, 5]]
        logits_b = [[-5, 5], [-5, 5], [5, -5], [5, -5]]
        model_a = _coteaching_algorithm__TracingIndexedLogits(logits_a, 'a', events)
        model_b = _coteaching_algorithm__TracingIndexedLogits(logits_b, 'b', events)
        algorithm = CoTeachingAlgorithm(model_a=model_a, model_b=model_b, optimizer_a=_coteaching_algorithm__TracingSGD(model_a.parameters(), 'a', events), optimizer_b=_coteaching_algorithm__TracingSGD(model_b.parameters(), 'b', events), scheduler_a=None, scheduler_b=None, loss=CrossEntropyLoss(), device=torch.device('cpu'), method_config=CoTeachingConfig.from_mapping(_coteaching_algorithm__method_config()))
        algorithm.setup(ExperimentContext(Path.cwd()))
        algorithm.step(Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.arange(4)}), RunState(cycle=1))
        self.assertEqual(events, ['forward_a', 'forward_b', 'step_a', 'step_b'])

    def test_clean_label_oracle_field_cannot_change_selection_or_update(self):
        baseline = _coteaching_algorithm__cross_update_algorithm()
        with_oracle = _coteaching_algorithm__cross_update_algorithm()
        payload = {'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.arange(4)}
        first = baseline.step(Batch(payload), RunState(cycle=1))
        second = with_oracle.step(Batch({**payload, 'clean_target': torch.ones(4, dtype=torch.long)}), RunState(cycle=1))
        self.assertEqual(first.metadata['selected_by_a_indices'].tolist(), second.metadata['selected_by_a_indices'].tolist())
        self.assertEqual(first.metadata['selected_by_b_indices'].tolist(), second.metadata['selected_by_b_indices'].tolist())
        for left, right in zip(baseline.model_a.parameters(), with_oracle.model_a.parameters()):
            torch.testing.assert_close(left, right)
        for left, right in zip(baseline.model_b.parameters(), with_oracle.model_b.parameters()):
            torch.testing.assert_close(left, right)

    def test_schedulers_are_distinct_and_roundtrip(self):
        algorithm = _coteaching_algorithm__cross_update_algorithm()
        algorithm.scheduler_a = torch.optim.lr_scheduler.StepLR(algorithm.optimizer_a, step_size=1, gamma=0.5)
        algorithm.scheduler_b = torch.optim.lr_scheduler.StepLR(algorithm.optimizer_b, step_size=1, gamma=0.5)
        self.assertIsNot(algorithm.scheduler_a, algorithm.scheduler_b)
        algorithm.optimizer_a.step()
        algorithm.optimizer_b.step()
        algorithm.step_schedulers()
        saved = copy.deepcopy(algorithm.state_dict())
        restored = _coteaching_algorithm__cross_update_algorithm()
        restored.scheduler_a = torch.optim.lr_scheduler.StepLR(restored.optimizer_a, step_size=1, gamma=0.5)
        restored.scheduler_b = torch.optim.lr_scheduler.StepLR(restored.optimizer_b, step_size=1, gamma=0.5)
        restored.load_state_dict(saved)
        self.assertEqual(restored.scheduler_a.state_dict(), algorithm.scheduler_a.state_dict())
        self.assertEqual(restored.scheduler_b.state_dict(), algorithm.scheduler_b.state_dict())

    def test_peer_initialization_is_distinct_and_reproducible(self):
        model_config = {'name': 'tiny_cnn', 'width': 4}
        first = _build_peer_models(model_config, 10, 13, 1)
        second = _build_peer_models(model_config, 10, 13, 1)
        self.assertEqual(tuple(first[0].state_dict()), tuple(first[1].state_dict()))
        self.assertTrue(any((not torch.equal(left, right) for left, right in zip(first[0].parameters(), first[1].parameters()))))
        for left_peer, right_peer in zip(first, second):
            for left, right in zip(left_peer.state_dict().values(), right_peer.state_dict().values()):
                torch.testing.assert_close(left, right)

    def test_checkpoint_roundtrip_keeps_fixed_peer_identity_and_optimizer_state(self):
        algorithm = _coteaching_algorithm__cross_update_algorithm()
        algorithm.step(Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.arange(4)}), RunState(cycle=1))
        saved = copy.deepcopy(algorithm.state_dict())
        restored = _coteaching_algorithm__cross_update_algorithm()
        restored.load_state_dict(saved)
        self.assertEqual(restored.private_state.optimizer_steps_a, 1)
        self.assertEqual(restored.private_state.optimizer_steps_b, 1)
        for original, loaded in zip(algorithm.model_a.parameters(), restored.model_a.parameters()):
            torch.testing.assert_close(original, loaded)
        for original, loaded in zip(algorithm.model_b.parameters(), restored.model_b.parameters()):
            torch.testing.assert_close(original, loaded)
        swapped = copy.deepcopy(saved)
        swapped['peer_identity'] = ('b', 'a')
        with self.assertRaisesRegex(ValueError, 'peer identity'):
            restored.load_state_dict(swapped)

    def test_configuration_rejects_single_model_composition_and_invalid_values(self):
        for key in ('selector', 'parameter_update', 'weight_provider', 'target_provider', 'objective_consumer'):
            values = _coteaching_algorithm__method_config()
            values[key] = {'name': 'anything'}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                CoTeachingConfig.from_mapping(values)
        values = _coteaching_algorithm__method_config()
        values['coteaching']['model_count'] = 1
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            CoTeachingConfig.from_mapping(values)
        values = _coteaching_algorithm__method_config()
        values['coteaching']['noise_rate'] = math.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            CoTeachingConfig.from_mapping(values)

# --- merged from test_coteaching_workflow.py ---
import copy

# --- merged from test_coteaching_workflow.py ---
from contextlib import redirect_stdout

# --- merged from test_coteaching_workflow.py ---
import hashlib

# --- merged from test_coteaching_workflow.py ---
import io

# --- merged from test_coteaching_workflow.py ---
import json

# --- merged from test_coteaching_workflow.py ---
import tempfile

# --- merged from test_coteaching_workflow.py ---
import unittest

# --- merged from test_coteaching_workflow.py ---
from pathlib import Path

# --- merged from test_coteaching_workflow.py ---
from unittest.mock import patch

# --- merged from test_coteaching_workflow.py ---
import numpy as np

# --- merged from test_coteaching_workflow.py ---
import torch

# --- merged from test_coteaching_workflow.py ---
from lnl_toolbox.data.cifar import CifarData

# --- merged from test_coteaching_workflow.py ---
from lnl_toolbox.catalog import load_recipe_config, paper_by_id, recipe_by_id, validate_config

# --- merged from test_coteaching_workflow.py ---
from lnl_toolbox.cli import main as cli_main

# --- merged from test_coteaching_workflow.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_coteaching_workflow.py ---
def _coteaching_workflow__cifar(size, split):
    labels = np.arange(size, dtype=np.int64) % 10
    images = np.zeros((size, 32, 32, 3), dtype=np.uint8)
    return CifarData(images, labels, tuple(map(str, range(10))), split, 'cifar10')

# --- merged from test_coteaching_workflow.py ---
def _coteaching_workflow__config(epochs=2):
    return {'method': 'coteaching', 'seed': 7, 'data': {'name': 'cifar10', 'root': 'unused', 'validation_size': 10, 'max_train_samples': 20, 'max_validation_samples': 10, 'max_test_samples': 10, 'augment': False}, 'noise': {'name': 'symmetric', 'rate': 0.4, 'seed': 17, 'validation_targets': 'noisy'}, 'loss': {'name': 'ce'}, 'model': {'name': 'tiny_cnn', 'width': 4}, 'optimizer': {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0}, 'scheduler': {'name': 'none'}, 'coteaching': {'model_count': 2, 'noise_rate': 0.4, 'initialization': {'peer_seed_offset': 1}, 'remember_schedule': {'name': 'linear', 'start': 1.0, 'end': 0.6, 'gradual_epochs': 2}, 'selection': {'count_rule': 'floor', 'tie_break': 'stable_sample_index'}}, 'loader': {'batch_size': 10, 'num_workers': 0, 'pin_memory': False}, 'evaluation': {'selection_split': 'validation', 'primary': 'mean_peer_accuracy', 'ensemble': 'mean_probabilities'}, 'trainer': {'epochs': epochs, 'device': 'cpu'}}

# --- merged from test_coteaching_workflow.py ---
def _coteaching_workflow__sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# --- merged from test_coteaching_workflow.py ---
class _coteaching_workflow_CoTeachingWorkflowTest(unittest.TestCase):

    def setUp(self):
        self.train_data = _coteaching_workflow__cifar(40, 'train')
        self.test_data = _coteaching_workflow__cifar(20, 'test')

    def _load_data(self, _root, split):
        return self.train_data if split == 'train' else self.test_data

    def test_fresh_resume_and_completed_resume_preserve_dual_state(self):
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load_data):
            run_dir = run_experiment(_coteaching_workflow__config(2), Path(directory) / 'run')
            first = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            manifest = run_dir / 'noise_manifest.npz'
            manifest_hash = _coteaching_workflow__sha256(manifest)
            manifest_mtime = manifest.stat().st_mtime_ns
            rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            epochs = [row for row in rows if row['event'] == 'epoch']
            self.assertEqual(len(epochs), 2)
            self.assertEqual(set(first['model']), {'a', 'b'})
            self.assertEqual(set(first['optimizer']), {'a', 'b'})
            self.assertEqual(set(first['algorithm_private_state']['schedulers']), {'a', 'b'})
            self.assertEqual(first['algorithm_private_state']['method_identity'], 'coteaching')
            self.assertEqual(first['run_state']['step'], 4)
            self.assertEqual(first['algorithm_private_state']['coteaching_state']['optimizer_steps_a'], 4)
            self.assertEqual(first['algorithm_private_state']['coteaching_state']['optimizer_steps_b'], 4)
            self.assertTrue((run_dir / 'best.pt').is_file())
            final = json.loads((run_dir / 'final_metrics.json').read_text(encoding='utf-8'))
            for key in ('test_accuracy_a', 'test_accuracy_b', 'test_mean_peer_accuracy', 'test_accuracy_ensemble'):
                self.assertIn(key, final)
            run_experiment(_coteaching_workflow__config(3), resume=run_dir / 'last.pt')
            resumed = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(resumed['completed_epoch'], 2)
            self.assertEqual(resumed['run_state']['step'], 6)
            self.assertEqual(_coteaching_workflow__sha256(manifest), manifest_hash)
            self.assertEqual(manifest.stat().st_mtime_ns, manifest_mtime)
            self.assertAlmostEqual(json.loads((run_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()[-2])['remember_rate'], 0.6)
            checkpoint_hash = _coteaching_workflow__sha256(run_dir / 'last.pt')
            checkpoint_mtime = (run_dir / 'last.pt').stat().st_mtime_ns
            metrics_hash = _coteaching_workflow__sha256(run_dir / 'metrics.jsonl')
            run_experiment(_coteaching_workflow__config(3), resume=run_dir / 'last.pt')
            self.assertEqual(_coteaching_workflow__sha256(run_dir / 'last.pt'), checkpoint_hash)
            self.assertEqual((run_dir / 'last.pt').stat().st_mtime_ns, checkpoint_mtime)
            self.assertEqual(_coteaching_workflow__sha256(run_dir / 'metrics.jsonl'), metrics_hash)

    def test_resume_rejects_method_configuration_drift(self):
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load_data):
            run_dir = run_experiment(_coteaching_workflow__config(1), Path(directory) / 'run')
            changed = _coteaching_workflow__config(2)
            changed['coteaching']['initialization']['peer_seed_offset'] = 2
            with self.assertRaisesRegex(ValueError, 'Co-teaching settings'):
                run_experiment(changed, resume=run_dir / 'last.pt')

    def test_cli_dispatch_is_lazy_and_does_not_change_supervised_default(self):
        with patch('lnl_toolbox.training.coteaching_experiment.run_coteaching_experiment', return_value=Path('coteaching-run')) as run:
            self.assertEqual(run_experiment(_coteaching_workflow__config()), Path('coteaching-run'))
            run.assert_called_once()

    def test_full_run_recipe_uses_engineering_cifar10_protocol(self):
        recipe = recipe_by_id('cifar10-coteaching-reproduction')
        self.assertEqual(recipe.profile, 'reproduction')
        self.assertEqual(recipe.method, 'coteaching')
        self.assertEqual(recipe.runner, 'coteaching')
        self.assertEqual(recipe.configuration_fidelity, 'engineering')
        config = load_recipe_config(recipe)
        self.assertEqual(validate_config(config).name, 'coteaching')
        self.assertEqual(config['method'], 'coteaching')
        self.assertEqual(config['execution']['runner'], 'coteaching')
        self.assertEqual(config['data']['name'], 'cifar10')
        for key in ('max_train_samples', 'max_validation_samples', 'max_test_samples'):
            self.assertNotIn(key, config['data'])
        self.assertEqual(config['noise']['name'], 'symmetric')
        self.assertEqual(config['noise']['rate'], 0.2)
        self.assertEqual(config['noise']['sampling'], 'transition')
        self.assertEqual(config['noise']['validation_targets'], 'noisy')
        self.assertEqual(config['model'], {'name': 'cifar_cnn8'})
        self.assertEqual(config['trainer']['epochs'], 200)
        self.assertEqual(config['loader']['batch_size'], 128)
        self.assertEqual(config['optimizer']['name'], 'adam')
        self.assertEqual(config['optimizer']['lr'], 0.001)
        self.assertEqual(config['coteaching']['model_count'], 2)
        self.assertEqual(config['coteaching']['noise_rate'], 0.2)
        self.assertEqual(config['coteaching']['remember_schedule']['gradual_epochs'], 10)
        paper = paper_by_id('coteaching')
        exposed = {item.recipe_id: item for item in paper.configs}
        self.assertEqual(exposed[recipe.id].configuration_fidelity, 'engineering')

    def test_coteaching_dry_run_reports_method_specific_settings(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli_main.main(['run', '--recipe', 'cifar10-coteaching-reproduction', '--dry-run', '--no-check-data'])
        self.assertEqual(result, 0)
        text = output.getvalue()
        for expected in ('Co-teaching networks: 2', 'Co-teaching batch size: 128', 'Co-teaching optimizer: adam', 'Co-teaching learning rate: 0.001', 'Co-teaching Tk / gradual epochs: 10', 'Co-teaching tau / noise rate: 0.2'):
            self.assertIn(expected, text)
