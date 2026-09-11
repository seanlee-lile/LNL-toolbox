"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_upm.py ---
import unittest

# --- merged from test_upm.py ---
import torch

# --- merged from test_upm.py ---
from lnl_toolbox.algorithms.upm import predict_true_posterior, soft_target_cross_entropy

# --- merged from test_upm.py ---
class _upm_UPMMathTest(unittest.TestCase):

    def test_eq8_is_normalized_and_eta_zero_keeps_noisy_label(self) -> None:
        logits = torch.zeros(2, 3, requires_grad=True)
        labels = torch.tensor([1, 2])
        q = predict_true_posterior(logits.softmax(1), labels, torch.tensor([0.8, 0.7]), torch.zeros(2))
        torch.testing.assert_close(q.sum(1), torch.ones(2))
        self.assertEqual(int(q[0].argmax()), 1)
        loss = soft_target_cross_entropy(logits, q).mean()
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_eta_changes_posterior_towards_snapshot_probability(self) -> None:
        logits = torch.tensor([[4.0, 0.0, 0.0]])
        q = predict_true_posterior(logits.softmax(1), torch.tensor([1]), torch.tensor([0.9]), torch.tensor([1.0]))
        self.assertGreater(float(q[0, 0]), float(q[0, 1]))

# --- merged from test_upm_algorithm.py ---
import unittest

# --- merged from test_upm_algorithm.py ---
import torch

# --- merged from test_upm_algorithm.py ---
from lnl_toolbox.algorithms.upm import ConfusingProbabilityState, UPMConfig, UPMTargetProvider, predict_true_posterior

# --- merged from test_upm_algorithm.py ---
from lnl_toolbox.core import TargetInput

# --- merged from test_upm_algorithm.py ---
from lnl_toolbox.noise import PosteriorSnapshot

# --- merged from test_upm_algorithm.py ---
def _upm_algorithm__config():
    stage = {'epochs': 1, 'model': {'name': 'tiny_cnn'}, 'optimizer': {'name': 'sgd', 'lr': 0.01}, 'scheduler': {'name': 'none'}}
    return {'method': 'upm', 'execution': {'runner': 'upm'}, 'noise': {'validation_targets': 'noisy'}, 'evaluation': {'selection_split': 'validation'}, 'upm': {'stage1': {**stage, 'best_metric': 'noisy_validation_accuracy'}, 'psi': {'source': 'stage1_best', 'split': 'train', 'augmentation': False}, 'main': {**stage, 'initialization': 'fresh'}, 'confusing_probability': {'initial_value': 0.01, 'learning_rate': 0.1, 'epsilon': 0.0001, 'update_start_epoch': 0, 'update_interval_epochs': 1}}}

# --- merged from test_upm_algorithm.py ---
class _upm_algorithm_UPMTargetProviderTest(unittest.TestCase):

    def test_same_fixed_q_is_returned_after_eta_update(self) -> None:
        snapshot = PosteriorSnapshot([[0.8, 0.2], [0.3, 0.7]], [0, 1], [4, 9], 'tiny', 'train')
        eta = ConfusingProbabilityState(torch.tensor([4, 9]), 0.01)
        provider = UPMTargetProvider(snapshot=snapshot, eta_state=eta, config=UPMConfig.from_mapping(_upm_algorithm__config()), device=torch.device('cpu'))
        logits = torch.tensor([[0.4, -0.1], [-0.2, 0.6]], requires_grad=True)
        result = provider.resolve(TargetInput(logits.detach(), torch.tensor([0, 1]), torch.tensor([4, 9]), {'epoch': 0}))
        expected = predict_true_posterior(torch.softmax(logits.detach(), 1), torch.tensor([0, 1]), torch.tensor([0.8, 0.7]), torch.full((2,), 0.01))
        torch.testing.assert_close(result.targets, expected)
        torch.testing.assert_close(provider.last_q, expected)
        self.assertFalse(result.targets.requires_grad)
        self.assertFalse(torch.equal(eta.eta, torch.full((2,), 0.01, dtype=torch.float64)))

    def test_schedule_skips_eta_but_still_produces_q(self) -> None:
        config = _upm_algorithm__config()
        config['upm']['confusing_probability']['update_start_epoch'] = 2
        snapshot = PosteriorSnapshot([[0.6, 0.4]], [0], [5], 'tiny', 'train')
        eta = ConfusingProbabilityState(torch.tensor([5]), 0.1)
        provider = UPMTargetProvider(snapshot=snapshot, eta_state=eta, config=UPMConfig.from_mapping(config), device=torch.device('cpu'))
        provider.resolve(TargetInput(torch.tensor([[0.2, -0.2]]), torch.tensor([0]), torch.tensor([5]), {'epoch': 1}))
        self.assertEqual(int(eta.update_count.item()), 0)

# --- merged from test_upm_objective.py ---
import unittest

# --- merged from test_upm_objective.py ---
import torch

# --- merged from test_upm_objective.py ---
from lnl_toolbox.algorithms.upm import ObservedNoisyProbabilityLookup, predict_true_posterior, soft_target_cross_entropy, update_confusing_probability

# --- merged from test_upm_objective.py ---
from lnl_toolbox.noise import PosteriorSnapshot

# --- merged from test_upm_objective.py ---
class _upm_objective_UPMObjectiveTest(unittest.TestCase):

    def test_eq8_hand_calculation_and_boundaries(self) -> None:
        clean = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.6, 0.0]])
        targets = torch.tensor([1, 0])
        psi = torch.tensor([0.3, 0.4])
        eta = torch.tensor([0.25, 0.0])
        actual = predict_true_posterior(clean, targets, psi, eta)
        factor = torch.tensor([[0.075, 0.825, 0.075], [1.0, 0.0, 0.0]])
        expected = clean * factor
        expected /= expected.sum(1, keepdim=True)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual[1], torch.tensor([1.0, 0.0, 0.0]))
        eta_one = predict_true_posterior(clean[:1], targets[:1], psi[:1], torch.ones(1))
        torch.testing.assert_close(eta_one, clean[:1])

    def test_eq8_rejects_zero_normalizer(self) -> None:
        with self.assertRaisesRegex(ValueError, 'positive'):
            predict_true_posterior(torch.tensor([[1.0, 0.0]]), torch.tensor([1]), torch.tensor([0.0]), torch.tensor([0.0]))

    def test_eq11_hand_calculation_clamp_and_detach(self) -> None:
        eta = torch.tensor([0.2, 0.9], requires_grad=True)
        q = torch.tensor([[0.25, 0.75], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        psi = torch.tensor([0.6, 0.4])
        actual = update_confusing_probability(eta, q, targets, psi, learning_rate=0.1, epsilon=0.0001)
        one_hot = torch.nn.functional.one_hot(targets, 2).float()
        bracket = torch.ones_like(q) + (psi * eta.detach() - eta.detach() - 1)[:, None] * one_hot
        expected = (eta.detach() + 0.1 * (bracket * q).sum(1) / (eta.detach() + 0.0001)).clamp(0, 1)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(bool(((actual >= 0) & (actual <= 1)).all()))

    def test_observed_class_psi_is_not_max_probability(self) -> None:
        snapshot = PosteriorSnapshot(noisy_probabilities=[[0.9, 0.1], [0.2, 0.8]], noisy_targets=[1, 0], global_indices=[20, 10], dataset='tiny', split='train')
        lookup = ObservedNoisyProbabilityLookup(snapshot, 'cpu')
        result = lookup.resolve(torch.tensor([10, 20]), torch.tensor([0, 1]), dtype=torch.float32)
        torch.testing.assert_close(result, torch.tensor([0.2, 0.1]))

    def test_soft_target_ce_value_and_gradient(self) -> None:
        logits = torch.tensor([[1.0, -1.0], [0.0, 2.0]], requires_grad=True)
        q = torch.tensor([[0.25, 0.75], [0.8, 0.2]])
        values = soft_target_cross_entropy(logits, q)
        expected = -(q * torch.log_softmax(logits, 1)).sum(1)
        torch.testing.assert_close(values, expected)
        values.mean().backward()
        torch.testing.assert_close(logits.grad, (torch.softmax(logits.detach(), 1) - q) / 2)

# --- merged from test_upm_state.py ---
import copy

# --- merged from test_upm_state.py ---
import unittest

# --- merged from test_upm_state.py ---
import torch

# --- merged from test_upm_state.py ---
from lnl_toolbox.algorithms.upm import ConfusingProbabilityState, UPMPhase, UPMState

# --- merged from test_upm_state.py ---
class _upm_state_UPMConfusingStateTest(unittest.TestCase):

    def test_noncontiguous_permutation_and_only_batch_updates(self) -> None:
        state = ConfusingProbabilityState(torch.tensor([30, 2, 11, 7]), 0.01)
        before = state.eta.clone()
        indices = torch.tensor([11, 2])
        state.update(indices, torch.tensor([[0.2, 0.8], [0.7, 0.3]]), torch.tensor([1, 0]), torch.tensor([0.6, 0.7]), learning_rate=0.1, epsilon=0.0001)
        rows = state.resolve_rows(indices)
        untouched = torch.ones(4, dtype=torch.bool)
        untouched[rows] = False
        torch.testing.assert_close(state.eta[untouched], before[untouched], rtol=0, atol=0)
        self.assertTrue(bool((state.update_count[rows] == 1).all()))
        self.assertTrue(bool((state.update_count[untouched] == 0).all()))

    def test_duplicate_and_missing_indices_fail(self) -> None:
        state = ConfusingProbabilityState(torch.tensor([1, 5]), 0.1)
        with self.assertRaisesRegex(ValueError, 'unique'):
            state.resolve_rows(torch.tensor([1, 1]))
        with self.assertRaises(KeyError):
            state.resolve_rows(torch.tensor([9]))

    def test_roundtrip_and_mapping_drift(self) -> None:
        state = ConfusingProbabilityState(torch.tensor([8, 2, 5]), 0.2)
        saved = state.state_dict()
        restored = ConfusingProbabilityState(torch.tensor([2, 5, 8]), 0.2)
        restored.load_state_dict(saved)
        torch.testing.assert_close(restored.eta, state.eta)
        changed = copy.deepcopy(saved)
        changed['canonical_sample_indices'][0] = 3
        with self.assertRaisesRegex(ValueError, 'mapping'):
            restored.load_state_dict(changed)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_state(self) -> None:
        state = ConfusingProbabilityState(torch.tensor([1, 4]), 0.1, device='cuda')
        self.assertEqual(state.eta.device.type, 'cuda')
        saved = state.state_dict()
        self.assertEqual(saved['eta'].device.type, 'cpu')

# --- merged from test_upm_state.py ---
class _upm_state_UPMPhaseStateTest(unittest.TestCase):

    def test_illegal_phase_transition_fails(self) -> None:
        state = UPMState()
        with self.assertRaisesRegex(ValueError, 'illegal'):
            state.advance(UPMPhase.PSI_READY)

# --- merged from test_upm_workflow.py ---
import hashlib

# --- merged from test_upm_workflow.py ---
import json

# --- merged from test_upm_workflow.py ---
from pathlib import Path

# --- merged from test_upm_workflow.py ---
import tempfile

# --- merged from test_upm_workflow.py ---
import unittest

# --- merged from test_upm_workflow.py ---
from unittest.mock import patch

# --- merged from test_upm_workflow.py ---
import numpy as np

# --- merged from test_upm_workflow.py ---
import torch

# --- merged from test_upm_workflow.py ---
import yaml

# --- merged from test_upm_workflow.py ---
from lnl_toolbox.data.cifar import CifarData

# --- merged from test_upm_workflow.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_upm_workflow.py ---
from lnl_toolbox.training.upm_experiment import UPMWorkflow

# --- merged from test_upm_workflow.py ---
def _upm_workflow__cifar(size: int, split: str, classes: int=10) -> CifarData:
    rng = np.random.default_rng(31 if split == 'train' else 32)
    images = rng.integers(0, 256, (size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(images, labels, tuple(map(str, range(classes))), split, f'cifar{classes}')

# --- merged from test_upm_workflow.py ---
def _upm_workflow__config(main_epochs: int=2, *, stage1_epochs: int=1, dataset: str='cifar10') -> dict:
    classes = 10 if dataset == 'cifar10' else 100
    stage = {'epochs': stage1_epochs, 'model': {'name': 'tiny_cnn', 'width': 2}, 'optimizer': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.0}, 'scheduler': {'name': 'none'}}
    return {'method': 'upm', 'execution': {'runner': 'upm'}, 'seed': 9, 'data': {'name': dataset, 'root': 'unused', 'validation_size': classes, 'max_train_samples': classes * 2, 'max_validation_samples': classes, 'max_test_samples': classes, 'augment': False}, 'loader': {'batch_size': classes, 'num_workers': 0, 'pin_memory': False}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 11, 'validation_targets': 'noisy', 'manifest_filename': 'noise_manifest.npz'}, 'upm': {'stage1': {**stage, 'best_metric': 'noisy_validation_accuracy'}, 'psi': {'source': 'stage1_best', 'split': 'train', 'augmentation': False}, 'main': {**stage, 'epochs': main_epochs, 'initialization': 'fresh'}, 'confusing_probability': {'initial_value': 0.01, 'learning_rate': 0.1, 'epsilon': 0.0001, 'update_start_epoch': 0, 'update_interval_epochs': 1}}, 'evaluation': {'selection_split': 'validation', 'primary': 'accuracy'}, 'trainer': {'device': 'cpu'}}

# --- merged from test_upm_workflow.py ---
def _upm_workflow__sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

# --- merged from test_upm_workflow.py ---
class _upm_workflow_UPMWorkflowTest(unittest.TestCase):

    def setUp(self) -> None:
        self.train = _upm_workflow__cifar(40, 'train')
        self.test = _upm_workflow__cifar(20, 'test')

    def _load(self, _root, split):
        return self.train if split == 'train' else self.test

    def test_fresh_resume_extension_and_completed_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load):
            run_dir = run_experiment(_upm_workflow__config(2), Path(directory) / 'run')
            payload = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(payload['upm_state']['phase'], 'completed')
            self.assertEqual(payload['upm_state']['main_completed_epochs'], 2)
            self.assertEqual(payload['upm_state']['main_global_step'], 4)
            self.assertIsNotNone(payload['best_main_model_state'])
            self.assertIsNotNone(payload['best_eta_state'])
            names = ('resolved_config.yaml', 'environment.json', 'noise_manifest.npz', 'stage1_best.pt', 'psi_snapshot.npz', 'eta_initial.npz', 'eta_best.npz', 'eta_last.npz', 'best.pt', 'last.pt', 'metrics.jsonl', 'final_metrics.json')
            for name in names:
                self.assertTrue((run_dir / name).is_file(), name)
            artifact_hashes = {name: _upm_workflow__sha(run_dir / name) for name in ('psi_snapshot.npz', 'eta_initial.npz', 'noise_manifest.npz')}
            run_experiment(_upm_workflow__config(3), resume=run_dir / 'last.pt')
            extended = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(extended['upm_state']['main_completed_epochs'], 3)
            self.assertEqual(extended['upm_state']['main_global_step'], 6)
            resolved = yaml.safe_load((run_dir / 'resolved_config.yaml').read_text())
            self.assertEqual(resolved['upm']['main']['epochs'], 3)
            for name, digest in artifact_hashes.items():
                self.assertEqual(_upm_workflow__sha(run_dir / name), digest)
            watched = {name: (_upm_workflow__sha(run_dir / name), (run_dir / name).stat().st_mtime_ns) for name in names}
            run_experiment(resolved, resume=run_dir / 'last.pt')
            for name, expected in watched.items():
                self.assertEqual((_upm_workflow__sha(run_dir / name), (run_dir / name).stat().st_mtime_ns), expected)

    def test_snapshot_corruption_and_identity_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load):
            run_dir = run_experiment(_upm_workflow__config(1), Path(directory) / 'run')
            changed = _upm_workflow__config(2)
            changed['upm']['confusing_probability']['learning_rate'] = 0.2
            with self.assertRaisesRegex(ValueError, 'identity'):
                run_experiment(changed, resume=run_dir / 'last.pt')
            (run_dir / 'psi_snapshot.npz').write_bytes(b'damaged')
            with self.assertRaisesRegex(ValueError, 'hash'):
                run_experiment(_upm_workflow__config(2), resume=run_dir / 'last.pt')

    def test_stage1_and_main_interrupted_resume(self) -> None:
        original_stage1 = UPMWorkflow.train_stage1
        original_main = UPMWorkflow.train_main

        def one_stage1(owner):
            return original_stage1(owner, max_epochs=1)

        def one_main(owner):
            return original_main(owner, max_epochs=1)
        config = _upm_workflow__config(2, stage1_epochs=2)
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load):
            stage1_dir = Path(directory) / 'stage1'
            with patch.object(UPMWorkflow, 'train_stage1', one_stage1):
                run_experiment(config, stage1_dir)
            stage1_payload = torch.load(stage1_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(stage1_payload['upm_state']['phase'], 'stage1_training')
            self.assertEqual(stage1_payload['upm_state']['stage1_completed_epochs'], 1)
            run_experiment(config, resume=stage1_dir / 'last.pt')
            self.assertEqual(torch.load(stage1_dir / 'last.pt', map_location='cpu', weights_only=False)['upm_state']['phase'], 'completed')
            main_dir = Path(directory) / 'main'
            with patch.object(UPMWorkflow, 'train_main', one_main):
                run_experiment(_upm_workflow__config(2), main_dir)
            main_payload = torch.load(main_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(main_payload['upm_state']['phase'], 'main_training')
            self.assertEqual(main_payload['upm_state']['main_completed_epochs'], 1)
            run_experiment(_upm_workflow__config(2), resume=main_dir / 'last.pt')
            self.assertEqual(torch.load(main_dir / 'last.pt', map_location='cpu', weights_only=False)['upm_state']['phase'], 'completed')

    def test_uninterrupted_matches_completed_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load):
            direct = run_experiment(_upm_workflow__config(2), Path(directory) / 'direct')
            resumed = run_experiment(_upm_workflow__config(1), Path(directory) / 'resumed')
            run_experiment(_upm_workflow__config(2), resume=resumed / 'last.pt')
            left = torch.load(direct / 'last.pt', map_location='cpu', weights_only=False)
            right = torch.load(resumed / 'last.pt', map_location='cpu', weights_only=False)
            for owner in ('main',):
                for key, value in left[owner]['model'].items():
                    self.assertTrue(torch.equal(value, right[owner]['model'][key]), key)
            torch.testing.assert_close(left['eta_state']['eta'], right['eta_state']['eta'], rtol=0, atol=0)
            torch.testing.assert_close(left['eta_state']['update_count'], right['eta_state']['update_count'], rtol=0, atol=0)

    def test_clean_test_does_not_select_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load), patch('lnl_toolbox.training.upm_experiment.evaluate_classification', side_effect=[{'loss': 1.0, 'accuracy': 0.4, 'samples': 10.0}, {'loss': 1.2, 'accuracy': 0.3, 'samples': 10.0}, {'loss': 999.0, 'accuracy': 0.0, 'samples': 10.0}]):
            run_dir = run_experiment(_upm_workflow__config(1), Path(directory) / 'run')
            final = json.loads((run_dir / 'final_metrics.json').read_text())
            self.assertEqual(final['best_noisy_validation_accuracy'], 0.3)
            self.assertEqual(final['clean_test_accuracy'], 0.0)
            self.assertFalse(final['test_selection_leakage'])

    def test_cifar100_lightweight(self) -> None:
        train = _upm_workflow__cifar(200, 'train', 100)
        test = _upm_workflow__cifar(100, 'test', 100)
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar100', side_effect=lambda _root, split: train if split == 'train' else test):
            run_dir = run_experiment(_upm_workflow__config(1, dataset='cifar100'), Path(directory) / 'run')
            self.assertTrue((run_dir / 'final_metrics.json').is_file())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cifar10_cuda_workflow(self) -> None:
        config = _upm_workflow__config(1)
        config['trainer']['device'] = 'cuda'
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load):
            run_dir = run_experiment(config, Path(directory) / 'run')
            payload = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(payload['upm_state']['phase'], 'completed')
            self.assertEqual(payload['upm_state']['main_completed_epochs'], 1)
            self.assertTrue((run_dir / 'final_metrics.json').is_file())
