"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_l2rw.py ---
from copy import deepcopy

# --- merged from test_l2rw.py ---
import json

# --- merged from test_l2rw.py ---
import unittest

# --- merged from test_l2rw.py ---
import torch

# --- merged from test_l2rw.py ---
from torch import nn

# --- merged from test_l2rw.py ---
from torch.func import functional_call

# --- merged from test_l2rw.py ---
from lnl_toolbox.algorithms.l2rw import meta_gradient, meta_reweight

# --- merged from test_l2rw.py ---
class _l2rw_L2RWTest(unittest.TestCase):

    def _fixture(self):
        model = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.2], [-0.1]]))
        train_x = torch.tensor([[1.0], [-1.0]])
        train_y = torch.tensor([0, 0])
        trusted_x = torch.tensor([[1.0], [-1.0]])
        trusted_y = torch.tensor([0, 1])
        return (model, train_x, train_y, trusted_x, trusted_y)

    def test_weights_are_nonnegative_normalized_and_favor_useful_sample(self) -> None:
        model, train_x, train_y, trusted_x, trusted_y = self._fixture()
        result = meta_reweight(model, train_x, train_y, trusted_x, trusted_y, virtual_learning_rate=0.1)
        self.assertTrue(bool((result.sample_weights >= 0).all()))
        self.assertAlmostEqual(float(result.sample_weights.sum()), 1.0, places=6)
        self.assertGreater(float(result.sample_weights[0]), float(result.sample_weights[1]))

    def test_virtual_step_does_not_mutate_model(self) -> None:
        model, train_x, train_y, trusted_x, trusted_y = self._fixture()
        before = deepcopy(model.state_dict())
        gradient = meta_gradient(model, train_x, train_y, trusted_x, trusted_y, virtual_learning_rate=0.1)
        self.assertTrue(torch.isfinite(gradient).all())
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_meta_gradient_matches_finite_difference(self) -> None:
        model, train_x, train_y, trusted_x, trusted_y = self._fixture()
        analytic = meta_gradient(model, train_x, train_y, trusted_x, trusted_y, virtual_learning_rate=0.1)
        parameters = dict(model.named_parameters())
        train_losses = torch.nn.functional.cross_entropy(model(train_x), train_y, reduction='none')

        def trusted_loss(epsilon: torch.Tensor) -> float:
            virtual_loss = torch.sum(epsilon * train_losses)
            gradients = torch.autograd.grad(virtual_loss, tuple(parameters.values()), retain_graph=True)
            virtual = {name: parameter - 0.1 * gradient for (name, parameter), gradient in zip(parameters.items(), gradients)}
            logits = functional_call(model, virtual, (trusted_x,), strict=True)
            return float(torch.nn.functional.cross_entropy(logits, trusted_y))
        step = 0.001
        positive = torch.tensor([step, 0.0], requires_grad=True)
        negative = torch.tensor([-step, 0.0], requires_grad=True)
        finite = (trusted_loss(positive) - trusted_loss(negative)) / (2.0 * step)
        self.assertAlmostEqual(float(analytic[0]), finite, places=4)

    def test_zero_meta_signal_keeps_all_weights_zero(self) -> None:
        model, train_x, train_y, _, _ = self._fixture()
        result = meta_reweight(model, train_x, train_y, torch.zeros(2, 1), torch.tensor([0, 1]), virtual_learning_rate=0.1)
        self.assertEqual(result.sample_weights.tolist(), [0.0, 0.0])

    def test_invalid_virtual_rate_fails(self) -> None:
        model, train_x, train_y, trusted_x, trusted_y = self._fixture()
        with self.assertRaises(ValueError):
            meta_reweight(model, train_x, train_y, trusted_x, trusted_y, virtual_learning_rate=0.0)

# --- merged from test_l2rw_training.py ---
from pathlib import Path

# --- merged from test_l2rw_training.py ---
import tempfile

# --- merged from test_l2rw_training.py ---
import unittest

# --- merged from test_l2rw_training.py ---
import numpy as np

# --- merged from test_l2rw_training.py ---
import yaml

# --- merged from test_l2rw_training.py ---
import torch

# --- merged from test_l2rw_training.py ---
from torch import nn

# --- merged from test_l2rw_training.py ---
from lnl_toolbox.algorithms.l2rw import meta_gradient

# --- merged from test_l2rw_training.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_l2rw_training.py ---
from lnl_toolbox.training.l2rw_experiment import _official_flip_labels, _official_l2rw_transform, _official_partition, _official_train_subsets

# --- merged from test_l2rw_training.py ---
_l2rw_training_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_l2rw_training.py ---
class _l2rw_training_L2RWTrainingTest(unittest.TestCase):

    def test_official_meta_gradient_matches_source_hessian_vector_product(self) -> None:
        torch.manual_seed(5)
        model = nn.Linear(3, 2)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 0, 1])
        trusted_inputs = torch.randn(2, 3)
        trusted_targets = torch.tensor([0, 1])
        actual = meta_gradient(model, inputs, targets, trusted_inputs, trusted_targets, virtual_learning_rate=1.0, weight_decay=0.2, implementation='official')
        parameters = tuple(model.parameters())
        epsilon = torch.zeros(4, requires_grad=True)
        train_loss = torch.sum(epsilon * torch.nn.functional.cross_entropy(model(inputs), targets, reduction='none'))
        train_gradients = torch.autograd.grad(train_loss, parameters, create_graph=True)
        trusted_loss = torch.nn.functional.cross_entropy(model(trusted_inputs), trusted_targets)
        trusted_gradients = torch.autograd.grad(trusted_loss, parameters, retain_graph=True)
        expected = torch.autograd.grad(train_gradients, epsilon, grad_outputs=trusted_gradients)[0]
        torch.testing.assert_close(actual, expected)
        without_decay = meta_gradient(model, inputs, targets, trusted_inputs, trusted_targets,
            virtual_learning_rate=1.0, weight_decay=0., implementation='official')
        torch.testing.assert_close(actual, without_decay, rtol=0, atol=0)

    def test_official_preprocessing_maps_zero_and_max_pixels_to_minus_one_and_one(self) -> None:
        import pickle
        from PIL import Image
        transform = _official_l2rw_transform(False, False)
        pickle.loads(pickle.dumps(transform))
        np.testing.assert_allclose(transform(Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))).numpy(), -1.0)
        np.testing.assert_allclose(transform(Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))).numpy(), 1.0)

    def test_formal_config_selects_official_l2rw_resnet(self) -> None:
        config = yaml.safe_load((_l2rw_training_ROOT / 'configs/experiment/l2rw_cifar10_reproduction.yaml').read_text(encoding='utf-8'))
        self.assertEqual(config['model']['name'], 'l2rw_resnet32')
        self.assertEqual(config['seed'], 1234)
        for section in ('data', 'noise', 'trusted_validation'):
            self.assertEqual(config[section]['seed'], config['seed'], section)
        from lnl_toolbox.models.cifar_resnet import L2RWResNet32
        from lnl_toolbox.training.reproduction_data import build_reproduction_model
        model = build_reproduction_model(config['model'], config['data'], 10)
        self.assertIsInstance(model, L2RWResNet32)
        self.assertEqual(len(model.stages), 15)
        self.assertEqual(tuple(model(torch.randn(2, 3, 32, 32)).shape), (2, 10))

    def test_official_meta_resnet_uses_batch_statistics_without_running_state(self) -> None:
        from lnl_toolbox.models.cifar_resnet import L2RWResNet32, share_l2rw_meta_parameters
        model = L2RWResNet32(10, 16, meta_batch_statistics=True)
        self.assertTrue(model.meta_batch_statistics)
        self.assertFalse(any(('running_mean' in key for key in model.state_dict())))
        self.assertFalse(any(('running_var' in key for key in model.state_dict())))
        self.assertTrue(all(('weight' not in name for name, _ in model.named_parameters() if '.bn' in name or '_bn' in name)))
        weighted = L2RWResNet32(10, 16, meta_batch_statistics=False)
        share_l2rw_meta_parameters(weighted, model)
        weighted_parameters = dict(weighted.named_parameters())
        for name, parameter in model.named_parameters():
            self.assertIs(parameter, weighted_parameters[name])

    def test_official_train_split_excludes_clean_meta_examples(self) -> None:
        positions = np.arange(12, dtype=np.int64)
        noisy, clean = _official_train_subsets(positions, num_clean=3, seed=7)
        self.assertEqual(noisy.size, 9)
        self.assertEqual(clean.size, 3)
        self.assertEqual(np.intersect1d(noisy, clean).size, 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([noisy, clean])), positions)

    def test_official_partition_matches_random_state_shuffle(self) -> None:
        actual = _official_partition(12, [9, 3], seed=7)
        expected = np.arange(12, dtype=np.int64)
        np.random.RandomState(7).shuffle(expected)
        np.testing.assert_array_equal(actual[0], expected[:9])
        np.testing.assert_array_equal(actual[1], expected[9:])

    def test_official_flip_labels_matches_source_operation(self) -> None:
        labels = np.arange(10, dtype=np.int64)
        actual_noisy, actual_clean_mask, actual_order = _official_flip_labels(labels, num_classes=10, rate=0.4, seed=7)
        random = np.random.RandomState(8)
        replacements = np.floor(random.uniform(0.0, 9.0, [4])).astype(np.int64)
        expected_noisy = np.concatenate([replacements, labels[4:]])
        expected_clean_mask = np.concatenate([np.zeros(4, dtype=np.int64), np.ones(6, dtype=np.int64)])
        order = np.arange(10, dtype=np.int64)
        random.shuffle(order)
        np.testing.assert_array_equal(actual_noisy, expected_noisy[order])
        np.testing.assert_array_equal(actual_clean_mask, expected_clean_mask[order])
        np.testing.assert_array_equal(actual_order, order)
        np.testing.assert_array_equal(labels[actual_order], labels[order])

    def test_step_budget_stops_exactly(self) -> None:
        config = yaml.safe_load((_l2rw_training_ROOT / 'configs/experiment/l2rw_cifar10_smoke.yaml').read_text(encoding='utf-8'))
        config['trainer'] = {'epochs': 10, 'max_steps': 2, 'device': 'cpu'}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(config, Path(directory) / 'run')
            payload = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(payload['global_step'], 2)

    def test_official_meta_path_runs_end_to_end_for_short_budget(self) -> None:
        config = yaml.safe_load((_l2rw_training_ROOT / 'configs/experiment/l2rw_cifar10_smoke.yaml').read_text(encoding='utf-8'))
        config['meta'] = {'virtual_learning_rate': 1.0, 'implementation': 'official'}
        config['trainer'] = {'epochs': 10, 'max_steps': 2, 'device': 'cpu'}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(config, Path(directory) / 'run')
            payload = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(payload['global_step'], 2)

    def test_official_schedule_switches_before_boundary_step(self) -> None:
        from lnl_toolbox.training.l2rw_experiment import _step_learning_rate

        milestones = [40000, 60000]
        self.assertAlmostEqual(_step_learning_rate(0.1, milestones, 0.1, None, 40000), 0.01)
        self.assertAlmostEqual(_step_learning_rate(0.1, milestones, 0.1, [0.03, 0.007], 39999), 0.1)
        self.assertAlmostEqual(_step_learning_rate(0.1, milestones, 0.1, [0.03, 0.007], 40000), 0.03)
        self.assertAlmostEqual(_step_learning_rate(0.1, milestones, 0.1, [0.03, 0.007], 60000), 0.007)

    def test_smoke_and_completed_resume(self) -> None:
        config = yaml.safe_load((_l2rw_training_ROOT / 'configs/experiment/l2rw_cifar10_smoke.yaml').read_text(encoding='utf-8'))
        config['trainer']['epochs'] = 2
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(config, Path(directory) / 'run')
            checkpoint = run_dir / 'last.pt'
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((run_dir / 'trusted_validation_manifest.npz').is_file())
            rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows], ['epoch', 'epoch', 'final'])
            self.assertTrue(all(not {'test_loss', 'test_accuracy'}.intersection(row) for row in rows[:2]))
            self.assertIn('test_accuracy', rows[-1])
            self.assertEqual(run_experiment(config, resume=checkpoint), run_dir)
            resumed = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['event'] for row in resumed], ['epoch', 'epoch', 'final'])


class L2RWPreparedDataIsolationTest(unittest.TestCase):
    def setUp(self):
        from lnl_toolbox.data.cifar import CifarData
        rng = np.random.default_rng(17)
        names = tuple(map(str, range(10)))
        self.train = CifarData(rng.integers(0, 256, (40, 32, 32, 3), dtype=np.uint8),
                               np.arange(40) % 10, names, 'train', 'cifar10')
        self.test = CifarData(rng.integers(0, 256, (20, 32, 32, 3), dtype=np.uint8),
                              np.arange(20) % 10, names, 'test', 'cifar10')
        self.config = yaml.safe_load((_l2rw_training_ROOT / 'configs/experiment/l2rw_cifar10_reproduction.yaml').read_text(encoding='utf-8'))
        self.config['data'].update(root='unused-controlled-source', num_val=10, num_clean=10)
        self.config['loader'].update(batch_size=5, num_workers=0, pin_memory=False)
        self.config['trusted_validation']['batch_size'] = 5

    def load(self, _root, split):
        return self.train if split == 'train' else self.test

    def prepare(self, directory, config):
        from lnl_toolbox.training.data_service import prepare_experiment_data
        from lnl_toolbox.training.runners import resolve_data_requirements
        return prepare_experiment_data(config, requirements=resolve_data_requirements(config, expected_runner='l2rw'),
                                       run_dir=Path(directory), seed=config['seed'])

    def assert_isolated(self, prepared, train_count, clean_count):
        from lnl_toolbox.data import DataRole
        train, clean, validation = map(set, (prepared.train_indices, prepared.trusted_indices, prepared.validation_indices))
        self.assertEqual(len(train), train_count)
        self.assertEqual(len(clean), clean_count)
        self.assertEqual(len(validation), 10)
        self.assertFalse(train & clean)
        self.assertFalse(train & validation)
        self.assertFalse(clean & validation)
        self.assertEqual(set(prepared.manifest.global_indices), train)
        self.assertEqual(prepared.requirements.manifest_scope, 'effective_train')
        trusted = prepared.datasets[DataRole.TRUSTED_VALIDATION]
        for position, index in enumerate(prepared.trusted_indices):
            self.assertEqual(int(trusted[position]['target']), int(self.train.labels[index]))

    def test_real_data_service_balances_trusted_and_matches_author_noise(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self.load):
            prepared = self.prepare(directory, self.config)
            self.assert_isolated(prepared, 20, 10)
            pool, validation = _official_partition(40, [30, 10], seed=self.config['seed'])
            np.testing.assert_array_equal(prepared.validation_indices, validation)
            self.assertEqual(set(prepared.train_indices) | set(prepared.trusted_indices), set(pool))
            np.testing.assert_array_equal(np.bincount(self.train.labels[prepared.trusted_indices], minlength=10),
                                          np.ones(10, dtype=np.int64))
            # Recover the pre-noise order; the author's permutation depends on
            # the seed and sample count, not the class labels.
            _, _, order = _official_flip_labels(np.zeros(20, dtype=np.int64), num_classes=10,
                                               rate=.4, seed=self.config['seed'])
            noisy_indices = prepared.train_indices[np.argsort(order)]
            noisy_labels, _, order = _official_flip_labels(self.train.labels[noisy_indices], num_classes=10,
                                                          rate=.4, seed=self.config['seed'])
            np.testing.assert_array_equal(prepared.train_indices, noisy_indices[order])
            np.testing.assert_array_equal(prepared.manifest.global_indices, noisy_indices[order])
            np.testing.assert_array_equal(prepared.manifest.clean_targets, self.train.labels[noisy_indices[order]])
            np.testing.assert_array_equal(prepared.manifest.noisy_targets, noisy_labels)
            repeated = self.prepare(Path(directory) / 'repeat', self.config)
            np.testing.assert_array_equal(prepared.train_indices, repeated.train_indices)
            np.testing.assert_array_equal(prepared.trusted_indices, repeated.trusted_indices)
            np.testing.assert_array_equal(prepared.manifest.noisy_targets, repeated.manifest.noisy_targets)

    def test_isolation_with_clean_noise_zero_rate_and_sample_limit(self):
        from unittest.mock import patch
        for name, rate, limit, clean_count in [('clean', 0., None, 10), ('official_uniform_flip', 0., None, 10),
                                              ('official_uniform_flip', .4, 12, 4)]:
            config = deepcopy(self.config)
            config['noise'].update(name=name, rate=rate)
            config['data']['num_clean'] = clean_count
            if limit is not None:
                config['data']['max_train_samples'] = limit
            with self.subTest(name=name, rate=rate, limit=limit), tempfile.TemporaryDirectory() as directory, \
                    patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self.load):
                prepared = self.prepare(directory, config)
                self.assert_isolated(prepared, (limit or 30) - clean_count, clean_count)
                if rate == 0.:
                    np.testing.assert_array_equal(prepared.manifest.noisy_targets, prepared.manifest.clean_targets)

    def test_controlled_two_step_training_and_completed_resume_preserve_isolation(self):
        from unittest.mock import patch
        from lnl_toolbox.noise import NoiseManifest
        from lnl_toolbox.data.trusted import TrustedSupervisionManifest
        config = deepcopy(self.config)
        config['model'] = {'name': 'tiny_cnn', 'width': 4}
        config['trainer'] = {'epochs': 2, 'max_steps': 2, 'device': 'cpu'}
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self.load):
            run_dir = run_experiment(config, Path(directory) / 'run')
            checkpoint = run_dir / 'last.pt'
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(payload['global_step'], 2)
            noisy = NoiseManifest.load(run_dir / 'noise_manifest.npz')
            trusted = TrustedSupervisionManifest.load(run_dir / 'trusted_validation_manifest.npz')
            self.assertEqual(len(noisy.global_indices), 20)
            self.assertEqual(len(trusted.global_indices), 10)
            self.assertFalse(set(noisy.global_indices) & set(trusted.global_indices))
            before = checkpoint.read_bytes()
            self.assertEqual(run_experiment(config, resume=checkpoint), run_dir)
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_metadata_and_scratch_do_not_claim_complete_uniform_preset(self):
        registry = yaml.safe_load((_l2rw_training_ROOT / 'web/lnl_parameter_metadata_registry_revised.yaml').read_text(encoding='utf-8'))
        comparison = registry['methods']['l2rw']['paper_comparison']
        self.assertEqual(comparison['alignment_status'], 'official_meta_algorithm_mixed_experiment_defaults')
