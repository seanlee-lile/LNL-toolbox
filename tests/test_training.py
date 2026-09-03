"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_torch_training.py ---
import tempfile

# --- merged from test_torch_training.py ---
import unittest

# --- merged from test_torch_training.py ---
from pathlib import Path

# --- merged from test_torch_training.py ---
from unittest.mock import patch

# --- merged from test_torch_training.py ---
import numpy as np

# --- merged from test_torch_training.py ---
import torch
from torchvision import transforms

# --- merged from test_torch_training.py ---
from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm

# --- merged from test_torch_training.py ---
from lnl_toolbox.algorithms.dss import DSSObjective

# --- merged from test_torch_training.py ---
from lnl_toolbox.core import Batch, ExperimentContext, RunState

# --- merged from test_torch_training.py ---
from lnl_toolbox.core import SoftTargetResult

# --- merged from test_torch_training.py ---
from lnl_toolbox.data.cifar import CifarData, default_data_root

# --- merged from test_torch_training.py ---
from lnl_toolbox.data.noisy_dataset import NoisyTargetDataset

# --- merged from test_torch_training.py ---
from lnl_toolbox.data.torch_cifar import TorchCifarDataset, build_cifar_transform, cifar_pixel_mean, stratified_split, train_validation_split

# --- merged from test_torch_training.py ---
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

# --- merged from test_torch_training.py ---
from lnl_toolbox.models import TinyCNN

# --- merged from test_torch_training.py ---
from lnl_toolbox.models.feature_output import FeatureOutput, forward_with_features

# --- merged from test_torch_training.py ---
from lnl_toolbox.plugins.builtin import build_builtin_loss

# --- merged from test_torch_training.py ---
from lnl_toolbox.runtime import resolve_device

# --- merged from test_torch_training.py ---
from lnl_toolbox.selectors import AllSelector, SelectionResult, SmallLossSelector

# --- merged from test_torch_training.py ---
from lnl_toolbox.training.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint

# --- merged from test_torch_training.py ---
from lnl_toolbox.training.experiment import build_model, build_scheduler

# --- merged from test_torch_training.py ---
class _torch_training_TorchTrainingTest(unittest.TestCase):

    def test_objective_consumer_is_opt_in_and_owns_backward(self):
        base_model = torch.nn.Linear(2, 2, bias=False)
        dss_model = torch.nn.Linear(2, 2, bias=False)
        dss_model.load_state_dict(base_model.state_dict())
        base = SupervisedClassificationAlgorithm(base_model, torch.optim.SGD(base_model.parameters(), lr=0.1), CrossEntropyLoss(), torch.device('cpu'))
        dss_objective = DSSObjective(2, 2, 1, warmup_epochs=1, mda=False, ccs=False)
        dss = SupervisedClassificationAlgorithm(dss_model, torch.optim.SGD(dss_model.parameters(), lr=0.1), CrossEntropyLoss(), torch.device('cpu'), objective_consumer=dss_objective)
        for algorithm in (base, dss):
            algorithm.setup(ExperimentContext(Path.cwd()))
            algorithm.on_cycle_start(RunState(cycle=0))
        batch = Batch({'input': torch.eye(2), 'target': torch.tensor([0, 1]), 'index': torch.tensor([0, 1])})
        base_result = base.step(batch, RunState(cycle=0))
        dss_result = dss.step(batch, RunState(cycle=0))
        torch.testing.assert_close(base_model.weight, dss_model.weight)
        self.assertNotIn('optimization_loss', base_result.metrics)
        self.assertIn('optimization_loss', dss_result.metrics)

    def test_objective_consumer_rejects_existing_treatment_composition(self):
        model = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, 'non-all selector'):
            SupervisedClassificationAlgorithm(model, torch.optim.SGD(model.parameters(), lr=0.1), CrossEntropyLoss(), torch.device('cpu'), selector=SmallLossSelector(0.5), objective_consumer=DSSObjective(2, 2, 1, warmup_epochs=1))

    @staticmethod
    def _target_algorithm(provider):
        model = torch.nn.Linear(2, 2, bias=False)
        algorithm = SupervisedClassificationAlgorithm(model, torch.optim.SGD(model.parameters(), lr=0.1), CrossEntropyLoss(), torch.device('cpu'), target_provider=provider)
        algorithm.setup(ExperimentContext(Path.cwd()))
        return algorithm

    @staticmethod
    def _target_batch():
        return Batch({'input': torch.eye(2), 'target': torch.tensor([0, 1]), 'index': torch.tensor([9, 3])})

    def test_repository_data_path(self):
        self.assertEqual(default_data_root().name, 'data')
        cifar_batch = default_data_root() / 'cifar10' / 'data_batch_1'
        if not cifar_batch.is_file():
            self.skipTest('official CIFAR-10 data is not available in this checkout')
        self.assertTrue(cifar_batch.is_file())

    def test_dataset_shape_dtype_target_and_stable_index(self):
        data = CifarData(np.zeros((3, 32, 32, 3), dtype=np.uint8), np.array([2, 1, 0]), ('a', 'b', 'c'), 'train', 'fixture')
        sample = TorchCifarDataset(data, [2])[0]
        self.assertEqual(sample['input'].shape, (3, 32, 32))
        self.assertEqual(sample['input'].dtype, torch.float32)
        self.assertEqual(sample['target'], 0)
        self.assertEqual(sample['index'], 2)

    def test_dataset_injects_noisy_target_by_global_index_without_clean_label_leak(self):
        data = CifarData(np.zeros((3, 32, 32, 3), dtype=np.uint8), np.array([2, 1, 0]), ('a', 'b', 'c'), 'train', 'fixture')
        clean_dataset = TorchCifarDataset(data, [2])
        clean_sample = clean_dataset[0]
        sample = NoisyTargetDataset(clean_dataset, global_indices=np.array([0, 1, 2]), noisy_targets=np.array([1, 2, 2]))[0]
        self.assertEqual(set(sample), {'input', 'target', 'index'})
        self.assertTrue(torch.equal(sample['input'], clean_sample['input']))
        self.assertEqual(clean_sample['target'], 0)
        self.assertEqual(sample['target'], 2)
        self.assertEqual(sample['index'], clean_sample['index'])
        self.assertEqual(sample['index'], 2)

    def test_dataset_rejects_misaligned_noisy_targets(self):
        data = CifarData(np.zeros((3, 32, 32, 3), dtype=np.uint8), np.array([2, 1, 0]), ('a', 'b', 'c'), 'train', 'fixture')
        dataset = TorchCifarDataset(data, [0, 2])
        with self.assertRaisesRegex(ValueError, 'matching one-dimensional'):
            NoisyTargetDataset(dataset, [0, 2], [1])
        with self.assertRaisesRegex(ValueError, 'no noisy target'):
            NoisyTargetDataset(dataset, [0], [1])

    def test_stratified_split_is_reproducible(self):
        labels = np.repeat(np.arange(10), 100)
        first = stratified_split(labels, 100, 9)
        second = stratified_split(labels, 100, 9)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(np.all(np.bincount(labels[first[1]]) == 10))

    def test_legacy_random_split_matches_choice_then_complement(self):
        labels = np.arange(20) % 3
        train, validation = train_validation_split(labels, 4, 7, strategy='random', rng='numpy_legacy')
        random = np.random.RandomState(7)
        expected_train = random.choice(20, 16, replace=False)
        expected_validation = np.delete(np.arange(20), expected_train)
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(validation, expected_validation)

    def test_tinycnn_shape(self):
        self.assertEqual(TinyCNN(10, 8)(torch.randn(4, 3, 32, 32)).shape, (4, 10))

    def test_cifar_resnet34_shape_and_stage_depths(self):
        model = build_model({'name': 'resnet34', 'base_width': 8}, num_classes=10)
        self.assertEqual(tuple((len(layer) for layer in (model.layer1, model.layer2, model.layer3, model.layer4))), (3, 4, 6, 3))
        self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 10))

    def test_cifar_resnet34_accepts_dataset_bound_input_channels(self):
        model = build_model(
            {"name": "resnet34", "base_width": 8, "input_channels": 1},
            num_classes=10,
        )
        self.assertEqual(model.stem[0].in_channels, 1)
        self.assertEqual(model(torch.randn(2, 1, 28, 28)).shape, (2, 10))

    def test_cifar_resnet34_can_use_torch_default_initialization(self):
        torch.manual_seed(123)
        reference = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        torch.manual_seed(123)
        model = build_model({'name': 'resnet34', 'base_width': 8, 'initialization': 'torch_default'}, num_classes=10)
        torch.testing.assert_close(model.stem[0].weight, reference.weight)

    def test_cifar_resnet50_is_explicit_and_preserves_feature_contract(self):
        model = build_model({'name': 'resnet50', 'base_width': 8, 'stem_padding': 0, 'initialization': 'torch_default'}, num_classes=10)
        self.assertEqual(tuple((len(layer) for layer in (model.layer1, model.layer2, model.layer3, model.layer4))), (3, 4, 6, 3))
        self.assertEqual(model.stem[0].padding, (0, 0))
        model.eval()
        inputs = torch.randn(2, 3, 32, 32)
        ordinary = model(inputs)
        featured = forward_with_features(model, inputs)
        torch.testing.assert_close(featured.logits, ordinary)
        self.assertEqual(featured.features.shape, (2, 256))

    def test_cifar_cnn8_shape_and_reference_channels(self):
        model = build_model({'name': 'cifar_cnn8'}, num_classes=10)
        convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        self.assertEqual([module.out_channels for module in convolutions], [64, 64, 128, 128, 196, 196])
        self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 10))

    def test_cnlcu_cnn9_matches_appendix_channels_and_shape(self):
        model = build_model({'name': 'cnlcu_cnn9'}, num_classes=10)
        convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        self.assertEqual([module.out_channels for module in convolutions], [128, 128, 128, 256, 256, 256, 512, 256, 128])
        self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 10))

    def test_linear_after_scheduler_keeps_then_decays_learning_rate(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = build_scheduler(optimizer, {'name': 'linear_after', 'start_epoch': 2, 'end_epoch': 5}, epochs=5)
        values = [optimizer.param_groups[0]['lr']]
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            values.append(optimizer.param_groups[0]['lr'])
        self.assertEqual(values[:3], [0.001, 0.001, 0.001])
        self.assertAlmostEqual(values[3], 0.001 * 2 / 3)
        self.assertAlmostEqual(values[4], 0.001 / 3)
        self.assertEqual(values[5], 0.0)

    def test_existing_models_expose_compatible_feature_output(self):
        models = (TinyCNN(10, 8), build_model({'name': 'resnet18', 'base_width': 8}, num_classes=10), build_model({'name': 'cifar_cnn8'}, num_classes=10))
        inputs = torch.randn(2, 3, 32, 32)
        for model in models:
            with self.subTest(model=type(model).__name__):
                model.eval()
                keys = tuple(model.state_dict())
                parameter_count = sum((parameter.numel() for parameter in model.parameters()))
                ordinary = model(inputs)
                feature_output = forward_with_features(model, inputs)
                self.assertIsInstance(feature_output, FeatureOutput)
                torch.testing.assert_close(feature_output.logits, ordinary)
                self.assertEqual(feature_output.features.shape[0], 2)
                self.assertEqual(tuple(model.state_dict()), keys)
                self.assertEqual(sum((parameter.numel() for parameter in model.parameters())), parameter_count)
                feature_output.logits.sum().backward()
                self.assertTrue(any((parameter.grad is not None for parameter in model.parameters())))

    def test_feature_output_rejects_invalid_tensor_contract(self):
        with self.assertRaisesRegex(ValueError, 'shape'):
            FeatureOutput(torch.zeros(2), torch.zeros(2, 3))
        with self.assertRaisesRegex(ValueError, 'batch size'):
            FeatureOutput(torch.zeros(2, 3), torch.zeros(3, 4))

    def test_ordinary_forward_does_not_construct_feature_output(self):
        cases = ((TinyCNN(10, 8), 'lnl_toolbox.models.tiny_cnn.FeatureOutput'), (build_model({'name': 'resnet18', 'base_width': 8}, num_classes=10), 'lnl_toolbox.models.cifar_resnet.FeatureOutput'), (build_model({'name': 'cifar_cnn8'}, num_classes=10), 'lnl_toolbox.models.cifar_cnn.FeatureOutput'), (build_model({'name': 'resnet50', 'base_width': 8}, num_classes=10), 'lnl_toolbox.models.cifar_resnet.FeatureOutput'))
        inputs = torch.randn(2, 3, 32, 32)
        for model, symbol in cases:
            with self.subTest(model=type(model).__name__):
                model.eval()
                with patch(symbol, side_effect=AssertionError('ordinary forward constructed FeatureOutput')):
                    self.assertEqual(model(inputs).shape, (2, 10))

    def test_feature_forward_updates_batchnorm_only_once(self):
        model = build_model({'name': 'resnet18', 'base_width': 8}, num_classes=10)
        model.train()
        batch_norm = next((module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)))
        before = int(batch_norm.num_batches_tracked.item())
        output = forward_with_features(model, torch.randn(2, 3, 32, 32))
        self.assertEqual(output.logits.shape, (2, 10))
        self.assertEqual(int(batch_norm.num_batches_tracked.item()), before + 1)

    def test_ordinary_forward_does_not_add_nonfinite_rejection(self):
        model = TinyCNN(10, 8).eval()
        inputs = torch.zeros(1, 3, 32, 32)
        inputs[0, 0, 0, 0] = torch.nan
        output = model(inputs)
        self.assertTrue(bool(torch.isnan(output).any()))
        with self.assertRaisesRegex(ValueError, 'finite'):
            forward_with_features(model, inputs)

    def test_gce2018_preprocessing_subtracts_training_pixel_mean(self):
        images = np.full((2, 32, 32, 3), 128, dtype=np.uint8)
        mean = cifar_pixel_mean(images)
        transform = build_cifar_transform(False, preprocessing='gce2018', pixel_mean=mean)
        data = CifarData(images, np.array([0, 1]), ('a', 'b'), 'train', 'fixture')
        sample = TorchCifarDataset(data, [0], transform=transform)[0]
        torch.testing.assert_close(sample['input'], torch.zeros_like(sample['input']))

    def test_tensor_only_preprocessing_is_unnormalized_and_unaugmented(self):
        transform = build_cifar_transform(
            False,
            augment=False,
            preprocessing='tensor_only',
        )
        self.assertEqual(len(transform.transforms), 1)
        self.assertIsInstance(transform.transforms[0], transforms.ToTensor)
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        output = transform(image)
        torch.testing.assert_close(
            output,
            torch.full_like(output, 128.0 / 255.0),
        )

    def test_tensor_only_preprocessing_rejects_augmentation(self):
        with self.assertRaisesRegex(ValueError, 'does not support augmentation'):
            build_cifar_transform(
                True,
                augment=True,
                preprocessing='tensor_only',
            )

    def test_standard_preprocessing_accepts_explicit_normalization(self):
        transform = build_cifar_transform(False, normalization_mean=(0.5, 0.5, 0.5), normalization_std=(0.25, 0.25, 0.25))
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        output = transform(image)
        expected = (128.0 / 255.0 - 0.5) / 0.25
        torch.testing.assert_close(output, torch.full_like(output, expected))
        with self.assertRaisesRegex(ValueError, 'provided together'):
            build_cifar_transform(False, normalization_mean=(0.5, 0.5, 0.5))
        with self.assertRaisesRegex(ValueError, 'three values'):
            build_cifar_transform(False, normalization_mean=(0.5, 0.5), normalization_std=(0.25, 0.25))
        with self.assertRaisesRegex(ValueError, 'positive'):
            build_cifar_transform(False, normalization_mean=(0.5, 0.5, 0.5), normalization_std=(0.25, 0.0, 0.25))

    def test_training_step_changes_parameters(self):
        model = TinyCNN(10, 8)
        algorithm = SupervisedClassificationAlgorithm(model, torch.optim.SGD(model.parameters(), lr=0.01), CrossEntropyLoss(), torch.device('cpu'))
        algorithm.setup(ExperimentContext(Path.cwd()))
        before = model.classifier.weight.detach().clone()
        result = algorithm.step(Batch({'input': torch.randn(4, 3, 32, 32), 'target': torch.tensor([0, 1, 2, 3]), 'index': torch.arange(4)}), RunState())
        self.assertTrue(np.isfinite(result.metrics['loss']))
        self.assertEqual(result.metrics['selected_ratio'], 1.0)
        self.assertFalse(torch.equal(before, model.classifier.weight))

    def test_target_sample_indices_exact_match_passes(self):

        class Provider:

            def resolve(self, target_input):
                return SoftTargetResult(targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), sample_indices=target_input.sample_indices)
        result = self._target_algorithm(Provider()).step(self._target_batch(), RunState())
        self.assertTrue(np.isfinite(result.metrics['loss']))

    def test_target_sample_indices_permutation_rejected(self):

        class Provider:

            def resolve(self, target_input):
                return SoftTargetResult(targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), sample_indices=target_input.sample_indices.flip(0))
        with self.assertRaisesRegex(ValueError, 'batch-order aligned'):
            self._target_algorithm(Provider()).step(self._target_batch(), RunState())

    def test_target_sample_indices_duplicate_rejected(self):

        class Provider:

            def resolve(self, target_input):
                return SoftTargetResult(targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), sample_indices=torch.tensor([9, 9]))
        with self.assertRaisesRegex(ValueError, 'unique'):
            self._target_algorithm(Provider()).step(self._target_batch(), RunState())

    def test_target_sample_indices_length_mismatch_rejected(self):

        class Provider:

            def resolve(self, target_input):
                return SoftTargetResult(targets=torch.tensor([[1.0, 0.0]]), sample_indices=torch.tensor([9]))
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            self._target_algorithm(Provider()).step(self._target_batch(), RunState())

    def test_each_p0_loss_completes_one_training_step(self):
        configs = [{'name': 'ce'}, {'name': 'gce'}, {'name': 'nce'}, {'name': 'mae'}, {'name': 'rce'}, {'name': 'apl'}]
        for config in configs:
            with self.subTest(name=config['name']):
                model = TinyCNN(10, 8)
                algorithm = SupervisedClassificationAlgorithm(model, torch.optim.SGD(model.parameters(), lr=0.01), build_builtin_loss(config), torch.device('cpu'))
                algorithm.setup(ExperimentContext(Path.cwd()))
                before = model.classifier.weight.detach().clone()
                result = algorithm.step(Batch({'input': torch.randn(4, 3, 32, 32), 'target': torch.tensor([0, 1, 2, 3]), 'index': torch.arange(4)}), RunState())
                self.assertTrue(np.isfinite(result.metrics['loss']))
                self.assertFalse(torch.equal(before, model.classifier.weight))

    def test_training_rejects_scalar_loss(self):
        model = TinyCNN(10, 8)
        algorithm = SupervisedClassificationAlgorithm(model, torch.optim.SGD(model.parameters(), lr=0.01), torch.nn.CrossEntropyLoss(), torch.device('cpu'))
        with self.assertRaises(ValueError):
            algorithm.step(Batch({'input': torch.randn(2, 3, 32, 32), 'target': torch.tensor([0, 1]), 'index': torch.arange(2)}), RunState())

    def test_selector_receives_detached_scores_and_only_selected_loss_backpropagates(self):

        class FirstSampleSelector:

            def __init__(self) -> None:
                self.scores_require_grad = None
                self.sample_indices = None
                self.epoch = None

            def select(self, selection_input):
                self.scores_require_grad = selection_input.scores.requires_grad
                self.sample_indices = selection_input.sample_indices.detach().clone()
                self.epoch = selection_input.metadata.get('epoch')
                return SelectionResult(selected_mask=torch.tensor([True, False]), metrics={'selected_samples': 1.0, 'selected_ratio': 0.5})
        model = torch.nn.Linear(2, 2, bias=False)
        selector = FirstSampleSelector()
        algorithm = SupervisedClassificationAlgorithm(model, torch.optim.SGD(model.parameters(), lr=0.1), CrossEntropyLoss(), torch.device('cpu'), selector=selector)
        algorithm.setup(ExperimentContext(Path.cwd()))
        before = model.weight.detach().clone()
        result = algorithm.step(Batch({'input': torch.eye(2), 'target': torch.tensor([0, 1]), 'index': torch.tensor([9, 3])}), RunState(cycle=4))
        self.assertIs(selector.scores_require_grad, False)
        self.assertTrue(torch.equal(selector.sample_indices, torch.tensor([9, 3])))
        self.assertEqual(selector.epoch, 4)
        self.assertFalse(torch.equal(before[:, 0], model.weight[:, 0]))
        self.assertTrue(torch.equal(before[:, 1], model.weight[:, 1]))
        self.assertEqual(result.metrics['selected_samples'], 1.0)
        self.assertEqual(result.metrics['selected_ratio'], 0.5)

    def test_small_loss_selector_completes_training_step(self):
        model = TinyCNN(10, 8)
        algorithm = SupervisedClassificationAlgorithm(model, torch.optim.SGD(model.parameters(), lr=0.01), CrossEntropyLoss(), torch.device('cpu'), selector=SmallLossSelector(keep_rate=0.5))
        algorithm.setup(ExperimentContext(Path.cwd()))
        result = algorithm.step(Batch({'input': torch.randn(4, 3, 32, 32), 'target': torch.tensor([0, 1, 2, 3]), 'index': torch.tensor([100, 20, 80, 40])}), RunState())
        self.assertEqual(result.metrics['selected_samples'], 2.0)
        self.assertEqual(result.metrics['selected_ratio'], 0.5)

    def test_small_loss_reducer_matches_selected_mean_and_parameter_update(self):
        selected_model = torch.nn.Linear(2, 2, bias=False)
        reference_model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            selected_model.weight.copy_(torch.tensor([[0.4, -0.2], [-0.1, 0.3]]))
        reference_model.load_state_dict(selected_model.state_dict())
        inputs = torch.eye(2)
        targets = torch.tensor([0, 0])
        algorithm = SupervisedClassificationAlgorithm(selected_model, torch.optim.SGD(selected_model.parameters(), lr=0.1), CrossEntropyLoss(), torch.device('cpu'), selector=SmallLossSelector(keep_rate=0.5))
        algorithm.setup(ExperimentContext(Path.cwd()))
        result = algorithm.step(Batch({'input': inputs, 'target': targets, 'index': torch.tensor([9, 3])}), RunState())
        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_losses = CrossEntropyLoss()(reference_model(inputs), targets)
        reference_loss = reference_losses[reference_losses.argmin()].mean()
        reference_loss.backward()
        reference_optimizer.step()
        self.assertAlmostEqual(result.metrics['loss'], reference_loss.item(), places=7)
        self.assertEqual(result.metrics['selected_samples'], 1.0)
        self.assertEqual(result.metrics['selected_ratio'], 0.5)
        for selected, reference in zip(selected_model.parameters(), reference_model.parameters()):
            self.assertTrue(torch.equal(selected, reference))

    def test_all_selector_matches_full_batch_mean_and_parameter_update(self):
        torch.manual_seed(23)
        selected_model = torch.nn.Linear(3, 2)
        reference_model = torch.nn.Linear(3, 2)
        reference_model.load_state_dict(selected_model.state_dict())
        inputs = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0], [-1.0, 1.0, 0.5]])
        targets = torch.tensor([0, 1, 0])
        selected_optimizer = torch.optim.SGD(selected_model.parameters(), lr=0.05)
        algorithm = SupervisedClassificationAlgorithm(selected_model, selected_optimizer, CrossEntropyLoss(), torch.device('cpu'), selector=AllSelector())
        algorithm.setup(ExperimentContext(Path.cwd()))
        result = algorithm.step(Batch({'input': inputs, 'target': targets, 'index': torch.tensor([50, 10, 30])}), RunState())
        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.05)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = CrossEntropyLoss()(reference_model(inputs), targets).mean()
        reference_loss.backward()
        reference_optimizer.step()
        self.assertAlmostEqual(result.metrics['loss'], reference_loss.item(), places=7)
        self.assertEqual(result.metrics['selected_ratio'], 1.0)
        for selected, reference in zip(selected_model.parameters(), reference_model.parameters()):
            self.assertTrue(torch.allclose(selected, reference, atol=0.0, rtol=0.0))

    def test_device_selection(self):
        self.assertEqual(resolve_device('cpu').type, 'cpu')
        expected = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(resolve_device('auto').type, expected)
        if torch.cuda.is_available():
            self.assertEqual(resolve_device('cuda').type, 'cuda')

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyCNN(10, 8)
            optimizer = torch.optim.AdamW(model.parameters())
            algorithm = SupervisedClassificationAlgorithm(model, optimizer, CrossEntropyLoss(), torch.device('cpu'))
            algorithm.setup(ExperimentContext(Path(directory)))
            state = RunState(cycle=2, step=17)
            path = Path(directory) / 'last.pt'
            saved = model.classifier.weight.detach().clone()
            save_checkpoint(path, algorithm, state, 2, {'seed': 1})
            with torch.no_grad():
                model.classifier.weight.zero_()
            restored, epoch, _ = load_checkpoint(path, algorithm, torch.device('cpu'))
            self.assertTrue(torch.equal(saved, model.classifier.weight))
            self.assertEqual(restored.step, 17)
            self.assertEqual(epoch, 2)

    def test_restore_cpu_rng_state_on_cpu(self):
        original = torch.get_rng_state()
        expected = torch.Generator().manual_seed(1234).get_state()
        try:
            restore_rng_state({'torch': expected.clone()})
            self.assertTrue(torch.equal(torch.get_rng_state(), expected))
        finally:
            torch.set_rng_state(original)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_restore_cpu_rng_state_loaded_on_cuda_device(self):
        original = torch.get_rng_state()
        expected = torch.Generator().manual_seed(5678).get_state()
        try:
            restore_rng_state({'torch': expected.to('cuda')})
            self.assertTrue(torch.equal(torch.get_rng_state(), expected))
        finally:
            torch.set_rng_state(original)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_restore_cuda_rng_state_list_normalizes_devices(self):
        original = torch.cuda.get_rng_state_all()
        try:
            torch.cuda.manual_seed_all(9012)
            expected = torch.cuda.get_rng_state_all()
            torch.cuda.set_rng_state_all(original)
            restore_rng_state({'cuda': [value.to('cuda') for value in expected]})
            actual = torch.cuda.get_rng_state_all()
            self.assertEqual(len(actual), len(expected))
            for actual_state, expected_state in zip(actual, expected):
                self.assertEqual(actual_state.device.type, 'cpu')
                self.assertTrue(torch.equal(actual_state, expected_state))
        finally:
            torch.cuda.set_rng_state_all(original)

    def test_restore_rng_state_rejects_invalid_tensor_contract(self):
        with self.assertRaisesRegex(TypeError, 'torch.uint8'):
            restore_rng_state({'torch': torch.zeros(8, dtype=torch.int64)})
        with self.assertRaisesRegex(ValueError, 'non-empty 1D'):
            restore_rng_state({'torch': torch.zeros((2, 4), dtype=torch.uint8)})
        with self.assertRaisesRegex(TypeError, 'list or tuple'):
            restore_rng_state({'cuda': torch.zeros(8, dtype=torch.uint8)})
        with self.assertRaisesRegex(TypeError, 'must be a tensor'):
            restore_rng_state({'cuda': [b'not a tensor']})

    def test_legacy_checkpoint_layouts_load_without_fabricated_best_metric(self):
        for layout in ('top-level', 'nested'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                model = TinyCNN(10, 8)
                optimizer = torch.optim.AdamW(model.parameters())
                algorithm = SupervisedClassificationAlgorithm(model, optimizer, CrossEntropyLoss(), torch.device('cpu'))
                state = RunState(cycle=1, step=7)
                payload = {'format_version': 1, 'run_state': {'cycle': state.cycle, 'step': state.step, 'phase': state.phase, 'metrics': state.metrics, 'metadata': state.metadata}, 'completed_epoch': 1, 'config': {'loss': {'name': 'ce'}}}
                if layout == 'top-level':
                    payload.update(algorithm.state_dict())
                else:
                    payload['algorithm'] = algorithm.state_dict()
                path = Path(directory) / 'legacy.pt'
                torch.save(payload, path)
                restored, epoch, loaded = load_checkpoint(path, algorithm, torch.device('cpu'))
                self.assertEqual(restored.step, 7)
                self.assertEqual(epoch, 1)
                self.assertEqual(loaded['best_validation_accuracy'], float('-inf'))
                self.assertTrue(loaded['_compatibility_warnings'])

    def test_checkpoint_requires_scheduler_state_when_scheduler_is_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyCNN(10, 8)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            algorithm = SupervisedClassificationAlgorithm(model, optimizer, CrossEntropyLoss(), torch.device('cpu'))
            path = Path(directory) / 'legacy.pt'
            torch.save({'algorithm': algorithm.state_dict(), 'run_state': {'cycle': 0, 'step': 1, 'phase': 'train', 'metrics': {}, 'metadata': {}}, 'completed_epoch': 0, 'config': {}}, path)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            with self.assertRaisesRegex(ValueError, 'scheduler'):
                load_checkpoint(path, algorithm, torch.device('cpu'), scheduler=scheduler)

# --- merged from test_training_progress.py ---
import io

# --- merged from test_training_progress.py ---
import tempfile

# --- merged from test_training_progress.py ---
import unittest

# --- merged from test_training_progress.py ---
from pathlib import Path

# --- merged from test_training_progress.py ---
from lnl_toolbox.training.progress import TerminalTrainingProgress, standardize_epoch_row, write_training_curves_svg

# --- merged from test_training_progress.py ---
class _training_progress_TrainingProgressTest(unittest.TestCase):

    def test_terminal_progress_reports_interval_and_final_batch(self) -> None:
        stream = io.StringIO()
        progress = TerminalTrainingProgress(epoch=2, total_epochs=5, total_batches=5, update_interval=2, stream=stream, force=True)
        for batch in range(1, 6):
            progress.update(batch, loss=1.0 / batch, accuracy=batch / 10.0)
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn('Epoch 002/005', lines[0])
        self.assertIn('0002/0005', lines[0])
        self.assertIn('0005/0005', lines[-1])

    def test_disabled_terminal_progress_is_silent(self) -> None:
        stream = io.StringIO()
        progress = TerminalTrainingProgress(epoch=1, total_epochs=1, total_batches=1, enabled=False, stream=stream)
        progress.update(1, loss=1.0, accuracy=0.5)
        self.assertEqual(stream.getvalue(), '')

    def test_non_interactive_terminal_progress_is_silent(self) -> None:
        stream = io.StringIO()
        progress = TerminalTrainingProgress(epoch=1, total_epochs=1, total_batches=1, stream=stream)
        progress.update(1, loss=1.0, accuracy=0.5)
        self.assertEqual(stream.getvalue(), '')

    def test_svg_contains_all_curves_and_is_overwritten(self) -> None:
        rows = [{'train_loss': 2.0, 'validation_loss': 2.1, 'train_accuracy': 0.2, 'validation_accuracy': 0.18, 'learning_rate': 0.01}, {'train_loss': 1.5, 'validation_loss': 1.7, 'train_accuracy': 0.4, 'validation_accuracy': 0.35, 'learning_rate': 0.001}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_curves.svg'
            returned = write_training_curves_svg(rows[:1], path)
            write_training_curves_svg(rows, path)
            text = path.read_text(encoding='utf-8')
        self.assertEqual(returned, path)
        self.assertIn('<svg', text)
        self.assertIn('Training progress', text)
        self.assertIn('epochs: 2', text)
        self.assertGreaterEqual(text.count('<polyline'), 5)

    def test_progress_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            TerminalTrainingProgress(epoch=0, total_epochs=1, total_batches=1)
        with self.assertRaises(ValueError):
            write_training_curves_svg([], Path('unused.svg'))

    def test_train_only_epoch_and_svg_are_supported_explicitly(self) -> None:
        row = standardize_epoch_row(
            {
                'epoch': 1,
                'train_loss': 1.0,
                'train_accuracy': 0.5,
                'learning_rate': 0.01,
            },
            require_validation=False,
        )
        self.assertNotIn('validation_loss', row)
        with tempfile.TemporaryDirectory() as directory:
            path = write_training_curves_svg([row], Path(directory) / 'train-only.svg')
            text = path.read_text(encoding='utf-8')
        self.assertIn('Training progress', text)
        self.assertNotIn('>validation</text>', text)
        self.assertEqual(text.count('<polyline'), 3)

    def test_partial_or_mixed_validation_metrics_are_rejected(self) -> None:
        base = {'train_loss': 1.0, 'train_accuracy': 0.5, 'learning_rate': 0.01}
        with self.assertRaisesRegex(ValueError, 'provided together'):
            standardize_epoch_row(
                {**base, 'validation_loss': 1.1}, require_validation=False
            )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'every epoch or none'):
                write_training_curves_svg(
                    [
                        {**base, 'validation_loss': 1.1, 'validation_accuracy': 0.4},
                        base,
                    ],
                    Path(directory) / 'mixed.svg',
                )

# --- merged from test_clean_baseline.py ---
import tempfile

# --- merged from test_clean_baseline.py ---
import unittest

# --- merged from test_clean_baseline.py ---
from pathlib import Path

# --- merged from test_clean_baseline.py ---
import torch

# --- merged from test_clean_baseline.py ---
from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm

# --- merged from test_clean_baseline.py ---
from lnl_toolbox.core import ExperimentContext, RunState

# --- merged from test_clean_baseline.py ---
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

# --- merged from test_clean_baseline.py ---
from lnl_toolbox.training.clean_baseline import build_clean_model, build_clean_optimizer, build_clean_scheduler, run_clean_experiment

# --- merged from test_clean_baseline.py ---
from lnl_toolbox.training.checkpoint import load_checkpoint, save_checkpoint

# --- merged from test_clean_baseline.py ---
class _clean_baseline_CleanBaselineTest(unittest.TestCase):

    def test_clean_runner_rejects_noise_manifest_configuration(self):
        with self.assertRaisesRegex(ValueError, 'does not accept noise configuration'):
            run_clean_experiment({'noise': {'manifest': 'noise.npz'}})

    def test_supported_models_produce_class_logits(self):
        inputs = torch.randn(2, 3, 32, 32)
        for name in ('tiny_cnn', 'resnet18', 'preact_resnet18'):
            config = {'name': name, 'width': 8, 'base_width': 8}
            with self.subTest(name=name):
                self.assertEqual(build_clean_model(config, 10)(inputs).shape, (2, 10))

    def test_component_builders_reject_unknown_names(self):
        with self.assertRaises(ValueError):
            build_clean_model({'name': 'unknown'}, 10)
        model = build_clean_model({'name': 'tiny_cnn', 'width': 8}, 10)
        with self.assertRaises(ValueError):
            build_clean_optimizer(model, {'name': 'unknown', 'lr': 0.1})

    def test_scheduler_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            model = build_clean_model({'name': 'tiny_cnn', 'width': 8}, 10)
            optimizer = build_clean_optimizer(model, {'name': 'sgd', 'lr': 0.1, 'momentum': 0.9})
            scheduler = build_clean_scheduler(optimizer, {'name': 'cosine', 't_max': 5}, 5)
            optimizer.step()
            scheduler.step()
            saved_lr = optimizer.param_groups[0]['lr']
            path = Path(directory) / 'last.pt'
            algorithm = SupervisedClassificationAlgorithm(model, optimizer, CrossEntropyLoss(), torch.device('cpu'))
            save_checkpoint(path, algorithm, RunState(cycle=1, step=9), 1, {}, scheduler=scheduler, best_epoch=0, best_validation_accuracy=0.5)
            new_model = build_clean_model({'name': 'tiny_cnn', 'width': 8}, 10)
            new_optimizer = build_clean_optimizer(new_model, {'name': 'sgd', 'lr': 0.1, 'momentum': 0.9})
            new_scheduler = build_clean_scheduler(new_optimizer, {'name': 'cosine', 't_max': 5}, 5)
            new_algorithm = SupervisedClassificationAlgorithm(new_model, new_optimizer, CrossEntropyLoss(), torch.device('cpu'))
            new_algorithm.setup(ExperimentContext(Path(directory)))
            state, _, restored = load_checkpoint(path, new_algorithm, torch.device('cpu'), scheduler=new_scheduler)
            self.assertAlmostEqual(new_optimizer.param_groups[0]['lr'], saved_lr)
            self.assertEqual(new_scheduler.last_epoch, scheduler.last_epoch)
            self.assertEqual(state.step, 9)
            self.assertEqual(restored['best_validation_accuracy'], 0.5)

    def test_multistep_scheduler(self):
        model = build_clean_model({'name': 'tiny_cnn', 'width': 8}, 10)
        optimizer = build_clean_optimizer(model, {'name': 'sgd', 'lr': 0.1})
        scheduler = build_clean_scheduler(optimizer, {'name': 'multistep', 'milestones': [2, 4], 'gamma': 0.1}, 5)
        self.assertIsNotNone(scheduler)

# --- merged from test_noisy_ce_baseline.py ---
import json

# --- merged from test_noisy_ce_baseline.py ---
import os

# --- merged from test_noisy_ce_baseline.py ---
from copy import deepcopy

# --- merged from test_noisy_ce_baseline.py ---
import tempfile

# --- merged from test_noisy_ce_baseline.py ---
import unittest

# --- merged from test_noisy_ce_baseline.py ---
from pathlib import Path

# --- merged from test_noisy_ce_baseline.py ---
from unittest.mock import patch

# --- merged from test_noisy_ce_baseline.py ---
import numpy as np

# --- merged from test_noisy_ce_baseline.py ---
import torch

# --- merged from test_noisy_ce_baseline.py ---
from lnl_toolbox.data.cifar import CifarData

from lnl_toolbox.data.contracts import (
    DataSpec,
    RawDatasetSplit,
    UnsupportedDatasetSplitError,
)

from lnl_toolbox.data.multiclass_synthetic import generate_synthetic_multiclass

# --- merged from test_noisy_ce_baseline.py ---
from lnl_toolbox.noise.manifest import NoiseManifest

# --- merged from test_noisy_ce_baseline.py ---
from lnl_toolbox.training.experiment import _validate_resume_config, run_experiment

# --- merged from test_noisy_ce_baseline.py ---
from lnl_toolbox.training.noisy_labels import prepare_noise_manifest

from lnl_toolbox.training.data_service import DATASETS

# --- merged from test_noisy_ce_baseline.py ---
def _noisy_ce_baseline__cifar(size: int, split: str) -> CifarData:
    labels = np.arange(size, dtype=np.int64) % 10
    images = np.zeros((size, 32, 32, 3), dtype=np.uint8)
    return CifarData(images, labels, tuple(map(str, range(10))), split, 'cifar10')

# --- merged from test_noisy_ce_baseline.py ---
def _noisy_ce_baseline__config(epochs: int=1) -> dict:
    return {'seed': 7, 'data': {'name': 'cifar10', 'root': 'unused', 'validation_size': 10, 'max_train_samples': 20, 'max_validation_samples': 10, 'max_test_samples': 10, 'augment': False}, 'noise': {'name': 'symmetric', 'rate': 0.4, 'seed': 17}, 'loss': {'name': 'ce'}, 'loader': {'batch_size': 10, 'num_workers': 0, 'pin_memory': False}, 'model': {'name': 'tiny_cnn', 'width': 4}, 'optimizer': {'name': 'adamw', 'lr': 0.001}, 'scheduler': {'name': 'cosine', 't_max': 2}, 'trainer': {'epochs': epochs, 'device': 'cpu'}}


class _noisy_ce_baseline__GenericTabularAdapter:
    name = 'supervised_generic_tabular_fixture'
    aliases: tuple[str, ...] = ()

    def validate(self, spec: DataSpec) -> None:
        if int(spec.options.get('num_classes', 0)) != 4:
            raise ValueError('supervised fixture requires four classes')

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        if split == 'validation':
            raise UnsupportedDatasetSplitError('fixture intentionally uses a train-derived validation split')
        sizes = {'train': 40, 'test': 12}
        if split not in sizes:
            raise ValueError(f'unsupported fixture split: {split}')
        generated = generate_synthetic_multiclass(
            sizes[split], 6, 4, seed + (0 if split == 'train' else 100),
            start_index=0, split=split,
        )
        return RawDatasetSplit(
            generated.features,
            generated.labels,
            generated.global_indices,
            self.name,
            split,
            4,
            clean_targets=generated.labels,
            source='test_fixture',
        )

# --- merged from test_noisy_ce_baseline.py ---
class _noisy_ce_baseline_NoisyCeBaselineTest(unittest.TestCase):

    def test_relative_output_root_is_resolved_before_manifest_metadata(self) -> None:
        config = _noisy_ce_baseline__config()
        config['output_root'] = 'runs'
        train_data = _noisy_ce_baseline__cifar(40, 'train')
        test_data = _noisy_ce_baseline__cifar(20, 'test')

        def load_data(_root, split):
            return train_data if split == 'train' else test_data
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=load_data), patch('lnl_toolbox.training.experiment.evaluate_classification', return_value={'loss': 1.0, 'accuracy': 0.25, 'samples': 10.0}):
            previous = Path.cwd()
            try:
                os.chdir(directory)
                run_dir = run_experiment(config)
            finally:
                os.chdir(previous)
            self.assertTrue(run_dir.is_absolute())
            self.assertEqual(run_dir.parent, (Path(directory) / 'runs').resolve())
            self.assertTrue((run_dir / 'noise_manifest.npz').is_file())
            self.assertTrue((run_dir / 'noise_summary.json').is_file())

    def test_unconnected_transition_estimator_configuration_is_rejected(self) -> None:
        config = _noisy_ce_baseline__config()
        config['transition_estimator'] = {'name': 'anchor'}
        with self.assertRaisesRegex(ValueError, "field 'transition_estimator'.*pipeline.transition_estimator"):
            run_experiment(config)

    def test_noisy_training_writes_manifest_metadata_and_clean_evaluation_sets(self) -> None:
        train_data = _noisy_ce_baseline__cifar(40, 'train')
        test_data = _noisy_ce_baseline__cifar(20, 'test')
        observed_evaluation_targets = []

        def load_data(_root, split):
            return train_data if split == 'train' else test_data

        def evaluate(model, loader, criterion, device):
            sample = loader.dataset[0]
            observed_evaluation_targets.append((sample['index'], sample['target']))
            return {'loss': 1.0, 'accuracy': 0.25, 'samples': float(len(loader.dataset))}
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=load_data), patch('lnl_toolbox.training.experiment.evaluate_classification', side_effect=evaluate):
            run_dir = run_experiment(_noisy_ce_baseline__config(), directory)
            manifest = NoiseManifest.load(run_dir / 'noise_manifest.npz')
            checkpoint = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            final = json.loads((run_dir / 'final_metrics.json').read_text(encoding='utf-8'))
            epoch_rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(line).get('event') == 'epoch']
        self.assertEqual(set(checkpoint['noise']), {'mode', 'manifest_path', 'manifest_version', 'manifest_sha256', 'mapping_hash', 'dataset', 'split', 'dataset_fingerprint', 'noise_type', 'requested_rate', 'seed', 'num_classes', 'manifest_actual_rate', 'effective_train_subset_actual_rate', 'validation_targets', 'effective_validation_subset_actual_rate', 'has_transition_matrix', 'has_per_sample_transition'})
        self.assertEqual(checkpoint['noise']['mapping_hash'], manifest.mapping_hash)
        self.assertEqual(final['noise'], checkpoint['noise'])
        self.assertEqual(checkpoint['config']['selector'], {'name': 'all'})
        self.assertEqual(epoch_rows[0]['selected_ratio'], 1.0)
        self.assertEqual(manifest.global_indices.size, 30)
        self.assertEqual(len(observed_evaluation_targets), 2)
        for index, target in observed_evaluation_targets:
            self.assertEqual(target, index % 10)

    def test_paper_mode_can_apply_one_manifest_to_train_and_validation(self) -> None:
        config = _noisy_ce_baseline__config()
        config['noise']['validation_targets'] = 'noisy'
        train_data = _noisy_ce_baseline__cifar(40, 'train')
        test_data = _noisy_ce_baseline__cifar(20, 'test')
        observed = []

        def load_data(_root, split):
            return train_data if split == 'train' else test_data

        def evaluate(model, loader, criterion, device):
            sample = loader.dataset[0]
            observed.append((sample['index'], sample['target']))
            return {'loss': 1.0, 'accuracy': 0.25, 'samples': float(len(loader.dataset))}
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=load_data), patch('lnl_toolbox.training.experiment.evaluate_classification', side_effect=evaluate):
            run_dir = run_experiment(config, directory)
            manifest = NoiseManifest.load(run_dir / 'noise_manifest.npz')
            summary = json.loads((run_dir / 'noise_summary.json').read_text(encoding='utf-8'))
        mapping = dict(zip(manifest.global_indices, manifest.noisy_targets))
        validation_index, validation_target = observed[0]
        test_index, test_target = observed[1]
        self.assertEqual(manifest.global_indices.size, 40)
        self.assertEqual(validation_target, mapping[validation_index])
        self.assertEqual(test_target, test_index % 10)
        self.assertEqual(summary['validation_targets'], 'noisy')
        self.assertIsNotNone(summary['effective_validation_subset_actual_rate'])

    def test_resume_reuses_manifest_and_rejects_hash_mismatch(self) -> None:
        config = _noisy_ce_baseline__config()
        indices = np.arange(20, dtype=np.int64)
        targets = indices % 10
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            manifest, path = prepare_noise_manifest(config, dataset='cifar10', clean_targets=targets, global_indices=indices, num_classes=10, run_dir=run_dir)
            metadata = {'manifest_path': path.name, 'mapping_hash': manifest.mapping_hash, 'dataset': manifest.dataset, 'split': manifest.split, 'dataset_fingerprint': manifest.dataset_fingerprint, 'noise_type': manifest.noise_type, 'requested_rate': manifest.requested_rate, 'seed': manifest.seed, 'num_classes': manifest.num_classes, 'manifest_actual_rate': manifest.actual_rate}
            loaded, _ = prepare_noise_manifest(config, dataset='cifar10', clean_targets=targets, global_indices=indices, num_classes=10, run_dir=run_dir, checkpoint_payload={'noise': metadata})
            self.assertEqual(loaded.mapping_hash, manifest.mapping_hash)
            metadata['mapping_hash'] = 'tampered'
            with self.assertRaisesRegex(ValueError, 'mapping hash'):
                prepare_noise_manifest(config, dataset='cifar10', clean_targets=targets, global_indices=indices, num_classes=10, run_dir=run_dir, checkpoint_payload={'noise': metadata})

    def test_resume_requires_existing_manifest(self) -> None:
        config = _noisy_ce_baseline__config()
        with tempfile.TemporaryDirectory() as directory:
            metadata = {'manifest_path': 'noise_manifest.npz', 'mapping_hash': 'missing'}
            with self.assertRaises(FileNotFoundError):
                prepare_noise_manifest(config, dataset='cifar10', clean_targets=np.arange(10), global_indices=np.arange(10), num_classes=10, run_dir=Path(directory), checkpoint_payload={'noise': metadata})

    def test_noisy_runner_accepts_configured_gce_loss(self) -> None:
        config = _noisy_ce_baseline__config()
        config['loss'] = {'name': 'gce', 'q': 0.7}
        train_data = _noisy_ce_baseline__cifar(40, 'train')
        test_data = _noisy_ce_baseline__cifar(20, 'test')

        def load_data(_root, split):
            return train_data if split == 'train' else test_data
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=load_data), patch('lnl_toolbox.training.experiment.evaluate_classification', return_value={'loss': 1.0, 'accuracy': 0.25, 'samples': 10.0}):
            run_dir = run_experiment(config, directory)
            checkpoint = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(checkpoint['loss'], {'name': 'gce', 'q': 0.7})

    def test_gce_and_apl_train_on_generic_tabular_data(self) -> None:
        base = {
            'seed': 7,
            'data': {
                'name': 'synthetic_multiclass',
                'num_classes': 4,
                'dimension': 6,
                'train_size': 32,
                'validation_size': 12,
                'test_size': 12,
            },
            'noise': {'name': 'symmetric', 'rate': 0.25, 'seed': 17},
            'loader': {'batch_size': 8, 'num_workers': 0},
            'model': {'name': 'feature_mlp', 'hidden_width': 8},
            'optimizer': {'name': 'adamw', 'lr': 0.001},
            'scheduler': {'name': 'none'},
            'trainer': {'epochs': 1, 'device': 'cpu'},
        }
        losses = (
            {'name': 'gce', 'q': 0.7},
            {
                'name': 'apl',
                'alpha': 1.0,
                'beta': 1.0,
                'active': {'name': 'nce'},
                'passive': {'name': 'rce', 'log_zero': -4.0},
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            for loss in losses:
                with self.subTest(loss=loss['name']):
                    config = deepcopy(base)
                    config['loss'] = loss
                    run_dir = run_experiment(config, Path(directory) / loss['name'])
                    checkpoint = torch.load(
                        run_dir / 'last.pt', map_location='cpu', weights_only=False
                    )
                    final = json.loads(
                        (run_dir / 'final_metrics.json').read_text(encoding='utf-8')
                    )
                    self.assertEqual(checkpoint['loss'], loss)
                    self.assertEqual(final['completed_epochs'], 1)
                    self.assertTrue(np.isfinite(final['test_loss']))

    def test_cdr_and_dss_train_on_generic_tabular_data(self) -> None:
        try:
            DATASETS.get(_noisy_ce_baseline__GenericTabularAdapter.name)
        except ValueError:
            DATASETS.add(_noisy_ce_baseline__GenericTabularAdapter())
        base = {
            'seed': 7,
            'data': {
                'name': _noisy_ce_baseline__GenericTabularAdapter.name,
                'root': 'unused',
                'num_classes': 4,
                'validation_size': 8,
            },
            'noise': {
                'name': 'symmetric',
                'rate': 0.25,
                'seed': 17,
                'validation_targets': 'noisy',
            },
            'loss': {'name': 'ce'},
            'selector': {'name': 'all'},
            'parameter_update': {'name': 'standard'},
            'loader': {'batch_size': 8, 'num_workers': 0},
            'model': {'name': 'feature_mlp', 'hidden_width': 8},
            'optimizer': {'name': 'adamw', 'lr': 0.001},
            'scheduler': {'name': 'none'},
            'trainer': {'epochs': 1, 'device': 'cpu'},
            'execution': {'runner': 'supervised'},
        }
        methods = {
            'cdr': {
                'optimizer': {
                    'name': 'sgd',
                    'lr': 0.01,
                    'momentum': 0.0,
                    'weight_decay': 0.0,
                },
                'parameter_update': {
                    'name': 'cdr',
                    'noise_rate': 0.25,
                    'l1_decay': 0.001,
                    'critical_scope': 'all_trainable',
                    'compatibility_mode': 'paper',
                },
            },
            'dss': {
                'pipeline': {
                    'name': 'standard_noisy_erm',
                    'objective_consumer': {
                        'name': 'dss',
                        # DSS state is addressed by stable source index, so its
                        # capacity covers the complete 40-sample train namespace.
                        'num_samples': 40,
                        'num_classes': 4,
                        'warmup_epochs': 1,
                        'alpha': 0.1,
                        'prior_decay': 0.99,
                        'mda': True,
                        'ccs': True,
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            for method, additions in methods.items():
                with self.subTest(method=method):
                    config = deepcopy(base)
                    config.update(additions)
                    run_dir = run_experiment(config, Path(directory) / method)
                    final = json.loads(
                        (run_dir / 'final_metrics.json').read_text(encoding='utf-8')
                    )
                    self.assertEqual(final['completed_epochs'], 1)
                    self.assertTrue(np.isfinite(final['test_loss']))

    def test_small_loss_selector_is_applied_and_resume_config_is_checked(self) -> None:
        config = _noisy_ce_baseline__config()
        config['selector'] = {'name': 'small_loss', 'keep_rate': 0.5}
        train_data = _noisy_ce_baseline__cifar(40, 'train')
        test_data = _noisy_ce_baseline__cifar(20, 'test')

        def load_data(_root, split):
            return train_data if split == 'train' else test_data
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=load_data), patch('lnl_toolbox.training.experiment.evaluate_classification', return_value={'loss': 1.0, 'accuracy': 0.25, 'samples': 10.0}):
            run_dir = run_experiment(config, directory)
            checkpoint = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        epoch_row = next((row for row in rows if row.get('event') == 'epoch'))
        self.assertEqual(epoch_row['selected_samples'], 10.0)
        self.assertEqual(epoch_row['selected_ratio'], 0.5)
        self.assertIn('train_all_sample_loss', epoch_row)
        self.assertEqual(checkpoint['config']['selector'], config['selector'])
        changed = _noisy_ce_baseline__config()
        changed['selector'] = {'name': 'small_loss', 'keep_rate': 0.75}
        with self.assertRaisesRegex(ValueError, 'selector'):
            _validate_resume_config(changed, config)

    def test_resume_rejects_every_linear_schedule_configuration_change(self) -> None:
        saved = _noisy_ce_baseline__config()
        saved['selector'] = {'name': 'small_loss', 'keep_rate': {'name': 'linear', 'start': 1.0, 'end': 0.6, 'warmup_epochs': 10}}
        _validate_resume_config(deepcopy(saved), saved)
        changes = {'name': 'constant', 'start': 0.9, 'end': 0.5, 'warmup_epochs': 11}
        for key, value in changes.items():
            with self.subTest(key=key):
                current = deepcopy(saved)
                current['selector']['keep_rate'][key] = value
                with self.assertRaisesRegex(ValueError, 'selector'):
                    _validate_resume_config(current, saved)

    def test_resume_rejects_preprocessing_or_validation_target_change(self) -> None:
        saved = _noisy_ce_baseline__config()
        saved['data']['preprocessing'] = 'gce2018'
        saved['noise']['validation_targets'] = 'noisy'
        changed = deepcopy(saved)
        changed['data']['preprocessing'] = 'standard'
        with self.assertRaisesRegex(ValueError, 'data.preprocessing'):
            _validate_resume_config(changed, saved)
        changed = deepcopy(saved)
        changed['noise']['validation_targets'] = 'clean'
        with self.assertRaisesRegex(ValueError, 'noise.validation_targets'):
            _validate_resume_config(changed, saved)
        saved['data']['validation_split'] = {'strategy': 'random', 'rng': 'numpy_legacy'}
        saved['data']['normalization'] = {'mean': [0.1, 0.2, 0.3], 'std': [0.4, 0.5, 0.6]}
        changed = deepcopy(saved)
        changed['data']['validation_split']['strategy'] = 'stratified'
        with self.assertRaisesRegex(ValueError, 'data.validation_split'):
            _validate_resume_config(changed, saved)
        changed = deepcopy(saved)
        changed['data']['normalization']['std'][0] = 0.7
        with self.assertRaisesRegex(ValueError, 'data.normalization'):
            _validate_resume_config(changed, saved)

    def test_test_set_cannot_select_checkpoint_without_explicit_opt_in(self):
        config = _noisy_ce_baseline__config()
        config['evaluation'] = {'selection_split': 'test'}
        with self.assertRaisesRegex(ValueError, 'test selection requires'):
            run_experiment(config)

# --- merged from test_new_paper_training.py ---
from copy import deepcopy

# --- merged from test_new_paper_training.py ---
from pathlib import Path

# --- merged from test_new_paper_training.py ---
import tempfile

# --- merged from test_new_paper_training.py ---
import unittest

# --- merged from test_new_paper_training.py ---
import yaml

# --- merged from test_new_paper_training.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_new_paper_training.py ---
_new_paper_training_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_new_paper_training.py ---
class _new_paper_training_NewPaperTrainingTest(unittest.TestCase):

    def _config(self, name: str) -> dict:
        return yaml.safe_load((_new_paper_training_ROOT / 'configs' / 'experiment' / name).read_text(encoding='utf-8'))

    def test_all_new_workflows_smoke_and_resume(self) -> None:
        names = ('mc_ldce_cifar10_smoke.yaml', 'cal_cifar10_smoke.yaml', 'ca2c_cifar10_smoke.yaml')
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                with self.subTest(name=name):
                    config = deepcopy(self._config(name))
                    run_dir = run_experiment(config, Path(directory) / name)
                    checkpoint = run_dir / 'last.pt'
                    self.assertTrue(checkpoint.is_file())
                    self.assertTrue((run_dir / 'metrics.jsonl').is_file())
                    self.assertEqual(run_experiment(config, resume=checkpoint), run_dir)

# --- merged from test_update_policy.py ---
import unittest

# --- merged from test_update_policy.py ---
import torch

# --- merged from test_update_policy.py ---
from lnl_toolbox.algorithms.update_policy import ParameterUpdateInput, ParameterUpdateResult, StandardUpdatePolicy, restore_update_policy, serialize_update_policy

# --- merged from test_update_policy.py ---
from lnl_toolbox.core import RunState

# --- merged from test_update_policy.py ---
class _update_policy_ParameterUpdatePolicyTest(unittest.TestCase):

    def test_standard_policy_matches_ordinary_torch_update(self) -> None:
        policy_model = torch.nn.Linear(3, 2)
        reference_model = torch.nn.Linear(3, 2)
        reference_model.load_state_dict(policy_model.state_dict())
        inputs = torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]])
        targets = torch.tensor([0, 1])
        policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.1, momentum=0.9)
        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1, momentum=0.9)
        policy_objective = torch.nn.functional.cross_entropy(policy_model(inputs), targets)
        result = StandardUpdatePolicy().update(ParameterUpdateInput(objective=policy_objective, model=policy_model, optimizer=policy_optimizer, run_state=RunState()))
        reference_optimizer.zero_grad(set_to_none=True)
        reference_objective = torch.nn.functional.cross_entropy(reference_model(inputs), targets)
        reference_objective.backward()
        reference_optimizer.step()
        self.assertEqual(result.metrics, {})
        for actual, expected in zip(policy_model.parameters(), reference_model.parameters()):
            self.assertTrue(torch.equal(actual, expected))
        policy_state = policy_optimizer.state_dict()
        reference_state = reference_optimizer.state_dict()
        self.assertEqual(policy_state['param_groups'], reference_state['param_groups'])
        for actual, expected in zip(policy_state['state'].values(), reference_state['state'].values()):
            self.assertEqual(set(actual), set(expected))
            for key in actual:
                if torch.is_tensor(actual[key]):
                    self.assertTrue(torch.equal(actual[key], expected[key]))
                else:
                    self.assertEqual(actual[key], expected[key])

    def test_update_input_requires_a_finite_scalar_with_gradients(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        state = RunState()
        with self.assertRaisesRegex(ValueError, 'scalar'):
            ParameterUpdateInput(torch.ones(2, requires_grad=True), model, optimizer, state)
        with self.assertRaisesRegex(ValueError, 'require gradients'):
            ParameterUpdateInput(torch.tensor(1.0), model, optimizer, state)
        with self.assertRaisesRegex(ValueError, 'finite'):
            ParameterUpdateInput(torch.tensor(float('nan'), requires_grad=True), model, optimizer, state)

    def test_update_result_rejects_nonfinite_metrics(self) -> None:
        self.assertEqual(ParameterUpdateResult({'steps': 1}).metrics, {'steps': 1.0})
        with self.assertRaisesRegex(ValueError, 'finite'):
            ParameterUpdateResult({'bad': float('inf')})

    def test_policy_checkpoint_identity_is_strict(self) -> None:
        policy = StandardUpdatePolicy()
        payload = serialize_update_policy(policy)
        self.assertEqual(payload, {'name': 'standard', 'state': {}})
        restore_update_policy(policy, payload)
        restore_update_policy(policy, None)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            restore_update_policy(policy, {'name': 'other', 'state': {}})
        with self.assertRaisesRegex(TypeError, 'mapping'):
            restore_update_policy(policy, {'name': 'standard', 'state': []})

    def test_standard_policy_gradient_clipping_matches_torch(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        policy = StandardUpdatePolicy(max_grad_norm=0.25)
        objective = (model(torch.ones(1, 2)) - 10.0).square().mean()
        request = ParameterUpdateInput(objective, model, optimizer, RunState())
        policy.update(request)
        self.assertAlmostEqual(float(model.weight.grad.norm()), 0.25, places=5)
        self.assertEqual(policy.state_dict(), {'max_grad_norm': 0.25})
