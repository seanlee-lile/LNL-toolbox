"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_pcse_pretrained_adapter.py ---
import os

# --- merged from test_pcse_pretrained_adapter.py ---
from pathlib import Path

# --- merged from test_pcse_pretrained_adapter.py ---
import tempfile

# --- merged from test_pcse_pretrained_adapter.py ---
import unittest

# --- merged from test_pcse_pretrained_adapter.py ---
from unittest import mock

# --- merged from test_pcse_pretrained_adapter.py ---
import numpy as np

# --- merged from test_pcse_pretrained_adapter.py ---
import torch

# --- merged from test_pcse_pretrained_adapter.py ---
from lnl_toolbox.algorithms.pcse.config import PCSEFeatureLayerConfig

# --- merged from test_pcse_pretrained_adapter.py ---
from lnl_toolbox.algorithms.pcse.features import collect_pcse_features

# --- merged from test_pcse_pretrained_adapter.py ---
from lnl_toolbox.data.noisy_dataset import NoisyTargetDataset

# --- merged from test_pcse_pretrained_adapter.py ---
from lnl_toolbox.data.torch_cifar import TorchCifarDataset

# --- merged from test_pcse_pretrained_adapter.py ---
from lnl_toolbox.noise.generators import generate_symmetric

# --- merged from test_pcse_pretrained_adapter.py ---
from lnl_toolbox.noise.manifest import fingerprint_labels

# --- merged from test_pcse_pretrained_adapter.py ---
from lnl_toolbox.training.checkpoint import atomic_save

# --- merged from test_pcse_pretrained_adapter.py ---
from lnl_toolbox.training.experiment import build_model

# --- merged from test_pcse_pretrained_adapter.py ---
from lnl_toolbox.training.noisy_labels import file_sha256

# --- merged from test_pcse_pretrained_adapter.py ---
from lnl_toolbox.training.pcse_pretrained import (
    load_pretrained_classifier_source,
    load_upm_main_best_source,
)

# --- merged from test_pcse_pretrained_adapter.py ---
class _pcse_pretrained_adapter__CifarFixture:

    def __init__(self, size: int) -> None:
        self.images = np.zeros((size, 32, 32, 3), dtype=np.uint8)
        self.labels = np.arange(size, dtype=np.int64) % 10

    def __len__(self) -> int:
        return int(self.labels.size)

# --- merged from test_pcse_pretrained_adapter.py ---
def _pcse_pretrained_adapter__source(directory: Path) -> tuple[dict, torch.nn.Module]:
    model_config = {'name': 'resnet18', 'base_width': 16}
    model = build_model(model_config, 10)
    labels = np.arange(20, dtype=np.int64) % 10
    manifest = generate_symmetric(labels, 10, 0.4, 1, 'cifar10', sampling='per_class')
    manifest.global_indices = np.arange(labels.size, dtype=np.int64)
    manifest.dataset_fingerprint = fingerprint_labels(labels)
    manifest_path = directory / 'noise_manifest.npz'
    manifest.save(manifest_path)
    noise = {'dataset': 'cifar10', 'num_classes': 10, 'mapping_hash': manifest.mapping_hash, 'dataset_fingerprint': manifest.dataset_fingerprint, 'manifest_sha256': file_sha256(manifest_path)}
    payload = {'method': 'upm', 'checkpoint_role': 'main_best', 'config': {'upm': {'main': {'model': model_config}}}, 'noise': noise, 'upm_state': {'main_completed_epochs': 3, 'main_global_step': 9, 'main_best_epoch': 1, 'main_best_validation_accuracy': 0.5}, 'best_main_model_state': model.state_dict()}
    checkpoint_path = directory / 'best.pt'
    atomic_save(payload, checkpoint_path)
    config = {'adapter': 'upm_main_best', 'run_directory_env': 'PCSE_TEST_SOURCE', 'checkpoint_sha256': file_sha256(checkpoint_path), 'manifest_sha256': file_sha256(manifest_path), 'mapping_hash': manifest.mapping_hash, 'dataset_fingerprint': manifest.dataset_fingerprint, 'model': model_config}
    return (config, model)


def _pcse_pretrained_adapter__standard_source(
    directory: Path, adapter: str
) -> tuple[dict, torch.nn.Module]:
    model_config = {'name': 'resnet18', 'base_width': 16}
    model = build_model(model_config, 10)
    labels = np.arange(20, dtype=np.int64) % 10
    manifest = generate_symmetric(
        labels, 10, 0.4, 1, 'cifar10', sampling='per_class'
    )
    manifest.global_indices = np.arange(labels.size, dtype=np.int64)
    manifest.dataset_fingerprint = fingerprint_labels(labels)
    manifest_path = directory / 'noise_manifest.npz'
    manifest.save(manifest_path)
    noise = {
        'dataset': 'cifar10',
        'num_classes': 10,
        'mapping_hash': manifest.mapping_hash,
        'dataset_fingerprint': manifest.dataset_fingerprint,
        'manifest_sha256': file_sha256(manifest_path),
    }
    if adapter == 'supervised_best':
        config = {
            'execution': {'runner': 'supervised'},
            'loss': {'name': 'ce'},
            'model': model_config,
        }
        state = model.state_dict()
    elif adapter == 'coteaching_peer_a_best':
        config = {'method': 'coteaching', 'model': model_config}
        peer_b = build_model(model_config, 10)
        state = {'a': model.state_dict(), 'b': peer_b.state_dict()}
    else:
        raise AssertionError(f'unsupported test adapter: {adapter}')
    payload = {
        'config': config,
        'noise': noise,
        'model': state,
        'completed_epoch': 2,
        'run_state': {'step': 9},
        'best_epoch': 1,
        'best_validation_accuracy': 0.5,
    }
    checkpoint_path = directory / 'best.pt'
    atomic_save(payload, checkpoint_path)
    source_config = {
        'adapter': adapter,
        'run_directory_env': 'PCSE_TEST_SOURCE',
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'manifest_sha256': file_sha256(manifest_path),
        'mapping_hash': manifest.mapping_hash,
        'dataset_fingerprint': manifest.dataset_fingerprint,
        'model': model_config,
    }
    return source_config, model

# --- merged from test_pcse_pretrained_adapter.py ---
class _pcse_pretrained_adapter_PCSEPretrainedAdapterTest(unittest.TestCase):

    def test_supported_standard_classifier_sources_are_loaded_strictly(self) -> None:
        for adapter, method, role in (
            ('supervised_best', 'ce', 'best'),
            ('coteaching_peer_a_best', 'coteaching', 'peer_a_best'),
        ):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as directory:
                config, expected_model = _pcse_pretrained_adapter__standard_source(
                    Path(directory), adapter
                )
                target_model = build_model(config['model'], 10)
                with mock.patch.dict(os.environ, {'PCSE_TEST_SOURCE': directory}):
                    source = load_pretrained_classifier_source(
                        config, target_model, num_classes=10
                    )
                self.assertEqual(source.adapter, adapter)
                self.assertEqual(source.source_method, method)
                self.assertEqual(source.checkpoint_role, role)
                for actual, expected in zip(
                    target_model.state_dict().values(),
                    expected_model.state_dict().values(),
                ):
                    torch.testing.assert_close(actual, expected)
                source.assert_unchanged()

    def test_standard_sources_reject_wrong_method_or_ambiguous_peer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _pcse_pretrained_adapter__standard_source(
                root, 'supervised_best'
            )
            payload = torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
            payload['config']['loss']['name'] = 'gce'
            atomic_save(payload, root / 'best.pt')
            invalid = {**config, 'checkpoint_sha256': file_sha256(root / 'best.pt')}
            with mock.patch.dict(os.environ, {'PCSE_TEST_SOURCE': directory}):
                with self.assertRaisesRegex(ValueError, 'must use CE'):
                    load_pretrained_classifier_source(
                        invalid, build_model(config['model'], 10), num_classes=10
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _pcse_pretrained_adapter__standard_source(
                root, 'coteaching_peer_a_best'
            )
            payload = torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
            payload['model'] = {'b': payload['model']['b']}
            atomic_save(payload, root / 'best.pt')
            invalid = {**config, 'checkpoint_sha256': file_sha256(root / 'best.pt')}
            with mock.patch.dict(os.environ, {'PCSE_TEST_SOURCE': directory}):
                with self.assertRaisesRegex(ValueError, 'peer A'):
                    load_pretrained_classifier_source(
                        invalid, build_model(config['model'], 10), num_classes=10
                    )

    def test_valid_source_and_identity_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _pcse_pretrained_adapter__source(root)
            with mock.patch.dict(os.environ, {'PCSE_TEST_SOURCE': directory}):
                source = load_upm_main_best_source(config, build_model(config['model'], 10), num_classes=10)
                self.assertEqual(source.checkpoint.sha256, config['checkpoint_sha256'])
                source.assert_unchanged()
                original = (root / 'noise_manifest.npz').read_bytes()
                (root / 'noise_manifest.npz').write_bytes(original + b'changed')
                with self.assertRaisesRegex(RuntimeError, 'immutable source changed'):
                    source.assert_unchanged()
                (root / 'noise_manifest.npz').write_bytes(original)
                for key, message in (('checkpoint_sha256', 'checkpoint SHA-256'), ('manifest_sha256', 'manifest SHA-256'), ('mapping_hash', 'mapping hash'), ('dataset_fingerprint', 'dataset fingerprint')):
                    invalid = {**config, key: 'f' * 64}
                    with self.assertRaisesRegex(ValueError, message):
                        load_upm_main_best_source(invalid, build_model(config['model'], 10), num_classes=10)

    def test_wrong_architecture_classes_role_and_state_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _pcse_pretrained_adapter__source(root)
            with mock.patch.dict(os.environ, {'PCSE_TEST_SOURCE': directory}):
                with self.assertRaisesRegex(ValueError, 'num_classes'):
                    load_upm_main_best_source(config, build_model(config['model'], 10), num_classes=3)
                wrong = {**config, 'model': {'name': 'resnet34', 'base_width': 16}}
                with self.assertRaisesRegex(ValueError, 'architecture'):
                    load_upm_main_best_source(wrong, build_model(wrong['model'], 10), num_classes=10)
                payload = torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
                payload['checkpoint_role'] = 'last'
                atomic_save(payload, root / 'best.pt')
                role_config = {**config, 'checkpoint_sha256': file_sha256(root / 'best.pt')}
                with self.assertRaisesRegex(ValueError, 'main_best'):
                    load_upm_main_best_source(role_config, build_model(config['model'], 10), num_classes=10)
                payload['checkpoint_role'] = 'main_best'
                payload['best_main_model_state'] = {'not_a_real_parameter': torch.ones(1)}
                atomic_save(payload, root / 'best.pt')
                state_config = {**config, 'checkpoint_sha256': file_sha256(root / 'best.pt')}
                with self.assertRaisesRegex(ValueError, 'state_dict'):
                    load_upm_main_best_source(state_config, build_model(config['model'], 10), num_classes=10)

    def test_real_resnet_layers_are_gap_pooled_and_logits_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _pcse_pretrained_adapter__source(root)
            with mock.patch.dict(os.environ, {'PCSE_TEST_SOURCE': directory}):
                model = build_model(config['model'], 10)
                load_upm_main_best_source(config, model, num_classes=10)
            data = _pcse_pretrained_adapter__CifarFixture(20)
            dataset = NoisyTargetDataset(TorchCifarDataset(data), np.arange(20, dtype=np.int64), np.arange(20, dtype=np.int64) % 10)
            loader = torch.utils.data.DataLoader(dataset, batch_size=5)
            result = collect_pcse_features(model, loader, 'cpu', dataset='cifar10', split='train', layers=(PCSEFeatureLayerConfig('layer3', 'global_average'), PCSEFeatureLayerConfig('layer4', 'global_average')))
            self.assertEqual(result.snapshots[0].features.shape, (20, 64))
            self.assertEqual(result.snapshots[1].features.shape, (20, 128))
            with self.assertRaisesRegex(ValueError, 'model logits output'):
                collect_pcse_features(model, loader, 'cpu', dataset='cifar10', split='train', layers=(PCSEFeatureLayerConfig('classifier'), PCSEFeatureLayerConfig('layer4')))

# --- merged from test_pcse_statistics.py ---
import unittest

# --- merged from test_pcse_statistics.py ---
from unittest import mock

# --- merged from test_pcse_statistics.py ---
import numpy as np

# --- merged from test_pcse_statistics.py ---
import torch

# --- merged from test_pcse_statistics.py ---
from lnl_toolbox.algorithms.pcse.gda import GDALayer, fit_ensemble_weights, fit_gda_layers

# --- merged from test_pcse_statistics.py ---
from lnl_toolbox.algorithms.pcse.statistics import PCSELayerStatistics, PCSEStatistics, build_coefficient_matrix, estimate_pcse_statistics, recover_clean_priors

# --- merged from test_pcse_statistics.py ---
from lnl_toolbox.training.snapshots import FeatureSnapshot

# --- merged from test_pcse_statistics.py ---
def _pcse_statistics__snapshots() -> tuple[FeatureSnapshot, FeatureSnapshot]:
    features = np.array([[-2.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, 2.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    targets = np.repeat(np.arange(3, dtype=np.int64), 2)
    indices = np.array([50, 10, 41, 9, 33, 2], dtype=np.int64)
    first = FeatureSnapshot(features, targets, indices, 'synthetic', 'train:hidden1')
    second = FeatureSnapshot(features * np.array([2.0, 0.5]), targets, indices, 'synthetic', 'train:hidden2')
    return (first, second)

# --- merged from test_pcse_statistics.py ---
class _pcse_statistics_PCSEStatisticsTest(unittest.TestCase):

    def test_identity_transition_recovers_empirical_statistics(self) -> None:
        snapshots = _pcse_statistics__snapshots()
        statistics = estimate_pcse_statistics(snapshots, ('hidden1', 'hidden2'), np.eye(3))
        np.testing.assert_allclose(statistics.noisy_priors, np.full(3, 1 / 3))
        np.testing.assert_allclose(statistics.clean_priors, np.full(3, 1 / 3))
        np.testing.assert_allclose(statistics.coefficient_matrix, np.eye(3))
        for layer, snapshot in zip(statistics.layers, snapshots):
            expected_means = np.stack([snapshot.features[snapshot.noisy_targets == class_index].mean(axis=0) for class_index in range(3)])
            expected_second = np.stack([np.einsum('ni,nj->ij', values, values) / len(values) for class_index in range(3) for values in [snapshot.features[snapshot.noisy_targets == class_index]]])
            np.testing.assert_allclose(layer.clean_means, expected_means)
            np.testing.assert_allclose(layer.clean_second_moments, expected_second)
            np.testing.assert_allclose(layer.clean_covariances, expected_second - np.einsum('ci,cj->cij', expected_means, expected_means))

    def test_transition_orientation_and_clean_prior_solve(self) -> None:
        transition = np.array([[0.8, 0.2, 0.0], [0.1, 0.7, 0.2], [0.0, 0.3, 0.7]])
        clean = np.array([0.2, 0.3, 0.5])
        noisy = transition.T @ clean
        recovered = recover_clean_priors(noisy, transition)
        np.testing.assert_allclose(recovered, clean)
        wrong_orientation = np.linalg.solve(transition, noisy)
        self.assertFalse(np.allclose(wrong_orientation, clean))

    def test_oracle_transition_recovers_constructed_clean_mean(self) -> None:
        transition = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
        means = np.array([[-2.0, 0.0], [0.0, 2.0], [2.0, -1.0]])
        clean_priors = np.full(3, 1 / 3)
        noisy_priors = transition.T @ clean_priors
        coefficient = build_coefficient_matrix(clean_priors, transition)
        noisy_means = (means.T @ np.diag(clean_priors) @ coefficient @ np.diag(1.0 / noisy_priors)).T
        clean_covariances = np.repeat((10.0 * np.eye(2))[None, :, :], 3, axis=0)
        clean_second = clean_covariances + np.einsum('ci,cj->cij', means, means)
        noisy_second = np.einsum('iab,ij,i,j->jab', clean_second, coefficient, clean_priors, 1.0 / noisy_priors)
        values: list[np.ndarray] = []
        labels: list[int] = []
        for class_index in range(3):
            covariance = noisy_second[class_index] - np.outer(noisy_means[class_index], noisy_means[class_index])
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            self.assertGreater(float(eigenvalues.min()), 0.0)
            for axis in range(2):
                offset = np.sqrt(2.0 * eigenvalues[axis]) * eigenvectors[:, axis]
                values.extend([noisy_means[class_index] - offset, noisy_means[class_index] + offset])
                labels.extend([class_index, class_index])
        features = np.asarray(values)
        targets = np.asarray(labels, dtype=np.int64)
        indices = np.arange(len(targets), dtype=np.int64)[::-1]
        snapshots = (FeatureSnapshot(features, targets, indices, 'oracle', 'train:h1'), FeatureSnapshot(2.0 * features, targets, indices, 'oracle', 'train:h2'))
        statistics = estimate_pcse_statistics(snapshots, ('h1', 'h2'), transition)
        np.testing.assert_allclose(statistics.layers[0].clean_means, means, atol=1e-12)
        np.testing.assert_allclose(statistics.layers[0].clean_second_moments, clean_second, atol=1e-11)
        np.testing.assert_allclose(statistics.layers[0].clean_covariances, clean_covariances, atol=1e-11)

    def test_coefficient_matrix_matches_equation_19(self) -> None:
        transition = np.array([[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.25, 0.75]])
        priors = np.array([0.2, 0.3, 0.5])
        expected = np.zeros((3, 3))
        identity = np.eye(3)
        for clean_class in range(3):
            for noisy_class in range(3):
                permutation = identity.copy()
                permutation[[clean_class, noisy_class]] = permutation[[noisy_class, clean_class]]
                expected += priors[clean_class] * transition[clean_class, noisy_class] * permutation.T
        np.testing.assert_allclose(build_coefficient_matrix(priors, transition), expected)

    def test_singular_transition_is_rejected(self) -> None:
        transition = np.full((3, 3), 1 / 3)
        with self.assertRaisesRegex(ValueError, 'singular'):
            estimate_pcse_statistics(_pcse_statistics__snapshots(), ('hidden1', 'hidden2'), transition)

    def test_singular_coefficient_matrix_is_rejected(self) -> None:
        with mock.patch('lnl_toolbox.algorithms.pcse.statistics.build_coefficient_matrix', return_value=np.ones((3, 3))):
            with self.assertRaisesRegex(ValueError, 'coefficient matrix M'):
                estimate_pcse_statistics(_pcse_statistics__snapshots(), ('hidden1', 'hidden2'), np.eye(3))

    def test_negative_clean_prior_is_rejected(self) -> None:
        transition = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]])
        with self.assertRaisesRegex(ValueError, 'strictly positive'):
            recover_clean_priors(np.array([0.01, 0.98, 0.01]), transition)

    def test_missing_observed_class_is_rejected(self) -> None:
        snapshots = list(_pcse_statistics__snapshots())
        modified = []
        for snapshot in snapshots:
            modified.append(FeatureSnapshot(snapshot.features[:4], np.array([0, 0, 1, 1]), snapshot.global_indices[:4], snapshot.dataset, snapshot.split))
        with self.assertRaisesRegex(ValueError, 'missing observed classes'):
            estimate_pcse_statistics(modified, ('hidden1', 'hidden2'), np.eye(3))

    def test_non_psd_shared_covariance_is_rejected(self) -> None:
        layer = PCSELayerStatistics(name='hidden', noisy_means=np.zeros((3, 2)), noisy_second_moments=np.zeros((3, 2, 2)), clean_means=np.zeros((3, 2)), clean_second_moments=np.zeros((3, 2, 2)), clean_covariances=np.repeat(np.diag([-1.0, 1.0])[None, :, :], 3, axis=0))
        statistics = PCSEStatistics(noisy_priors=np.full(3, 1 / 3), clean_priors=np.full(3, 1 / 3), coefficient_matrix=np.eye(3), transition_condition=1.0, coefficient_condition=1.0, layers=(layer, layer.__class__(name='hidden2', noisy_means=layer.noisy_means, noisy_second_moments=layer.noisy_second_moments, clean_means=layer.clean_means, clean_second_moments=layer.clean_second_moments, clean_covariances=layer.clean_covariances)))
        with self.assertRaisesRegex(ValueError, 'non-PSD'):
            fit_gda_layers(statistics, covariance_ridge=2.0)

    def test_gda_scores_match_hand_calculation(self) -> None:
        means = np.eye(3, dtype=np.float64)
        gda = GDALayer(name='hidden', clean_priors=np.full(3, 1 / 3), means=means, shared_covariance=np.eye(3), covariance_ridge=0.0)
        point = np.array([[2.0, 0.5, -1.0]])
        expected = point @ means.T - 0.5 + np.log(1 / 3)
        np.testing.assert_allclose(gda.scores(point), expected)

    def test_ensemble_weights_are_positive_simplex(self) -> None:
        probabilities = np.array([[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]])
        raw, _, losses = fit_ensemble_weights(probabilities, np.array([0, 1, 2]), epochs=5, learning_rate=0.1)
        weights = torch.softmax(raw.detach(), dim=0)
        self.assertTrue(bool((weights > 0).all().item()))
        self.assertAlmostEqual(float(weights.sum().item()), 1.0)
        self.assertTrue(all(np.isfinite(losses)))

# --- merged from test_pcse_volmin.py ---
from copy import deepcopy

# --- merged from test_pcse_volmin.py ---
import json

# --- merged from test_pcse_volmin.py ---
from pathlib import Path

# --- merged from test_pcse_volmin.py ---
import tempfile

# --- merged from test_pcse_volmin.py ---
import unittest

# --- merged from test_pcse_volmin.py ---
from unittest import mock

# --- merged from test_pcse_volmin.py ---
import numpy as np

# --- merged from test_pcse_volmin.py ---
import torch

# --- merged from test_pcse_volmin.py ---
from torch import nn

# --- merged from test_pcse_volmin.py ---
from torch.utils.data import DataLoader

# --- merged from test_pcse_volmin.py ---
import yaml

# --- merged from test_pcse_volmin.py ---
from lnl_toolbox.algorithms.pcse import PCSEAlgorithm, PCSEPhase

# --- merged from test_pcse_volmin.py ---
from lnl_toolbox.algorithms.pcse.volmin import DiagonallyDominantTransition, build_volmin_optimizer, validate_trainable_transition, volmin_objective

# --- merged from test_pcse_volmin.py ---
from lnl_toolbox.data.multiclass_synthetic import MulticlassTensorDataset, generate_synthetic_multiclass

# --- merged from test_pcse_volmin.py ---
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

# --- merged from test_pcse_volmin.py ---
from lnl_toolbox.noise.transition import TransitionArtifact

# --- merged from test_pcse_volmin.py ---
from lnl_toolbox.training.checkpoint import read_checkpoint

# --- merged from test_pcse_volmin.py ---
from lnl_toolbox.training.pcse_experiment import _PCSEMultilayerPerceptron, run_pcse_experiment

# --- merged from test_pcse_volmin.py ---
_pcse_volmin_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_pcse_volmin.py ---
_pcse_volmin_VOLMIN_CONFIG = _pcse_volmin_ROOT / 'configs/experiment/pcse_multiclass_volmin_smoke.yaml'

# --- merged from test_pcse_volmin.py ---
def _pcse_volmin__config() -> dict:
    return yaml.safe_load(_pcse_volmin_VOLMIN_CONFIG.read_text(encoding='utf-8'))

# --- merged from test_pcse_volmin.py ---
def _pcse_volmin__algorithm(run_dir: Path) -> PCSEAlgorithm:
    config = _pcse_volmin__config()
    config['pretraining_stage']['epochs'] = 1
    config['transition_stage']['epochs'] = 2
    config['ensemble_stage']['epochs'] = 1
    train = generate_synthetic_multiclass(90, 6, 3, 501, start_index=0, split='train')
    validation = generate_synthetic_multiclass(30, 6, 3, 502, start_index=90, split='validation')
    test = generate_synthetic_multiclass(30, 6, 3, 503, start_index=120, split='test')
    train_loader = DataLoader(MulticlassTensorDataset(train), batch_size=30, shuffle=False)
    validation_loader = DataLoader(MulticlassTensorDataset(validation), batch_size=30, shuffle=False)
    torch.manual_seed(47)
    model = _PCSEMultilayerPerceptron(6, 12, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return PCSEAlgorithm(model=model, optimizer=optimizer, scheduler=None, loss=CrossEntropyLoss(), train_loader=train_loader, statistics_loader=train_loader, noisy_validation_loader=validation_loader, clean_test_loader=DataLoader(MulticlassTensorDataset(test), batch_size=30, shuffle=False), device=torch.device('cpu'), run_dir=run_dir, config=config, dataset='synthetic_multiclass', num_classes=3, noise_metadata={})

# --- merged from test_pcse_volmin.py ---
class _pcse_volmin_PCSEVolMinPrimitiveTest(unittest.TestCase):

    def test_parameterization_is_safe_deterministic_and_row_stochastic(self) -> None:
        first = DiagonallyDominantTransition(3, initial_flip_mass=0.1, max_flip_mass=0.49, temperature=1.0, seed=7)
        second = DiagonallyDominantTransition(3, initial_flip_mass=0.1, max_flip_mass=0.49, temperature=1.0, seed=7)
        transition = first.matrix()
        torch.testing.assert_close(transition, second.matrix())
        self.assertTrue(bool((transition >= 0.0).all()))
        torch.testing.assert_close(transition.sum(dim=1), torch.ones(3, dtype=torch.float64))
        self.assertTrue(bool((torch.diagonal(transition) > 0.5).all()))
        sign, _ = torch.linalg.slogdet(transition)
        self.assertGreater(float(sign), 0.0)
        validate_trainable_transition(transition, determinant_tolerance=1e-08, condition_limit=1000000.0)

    def test_asymmetric_clean_to_noisy_direction_and_objective_sign(self) -> None:
        clean = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float64)
        transition = torch.tensor([[0.75, 0.2, 0.05], [0.1, 0.8, 0.1], [0.15, 0.05, 0.8]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1, 2])
        objective, metrics = volmin_objective(torch.log(clean), targets, transition, lambda_volume=0.2, determinant_tolerance=1e-08, condition_limit=1000000.0)
        noisy = clean @ transition
        expected_nll = -torch.log(noisy[torch.arange(2), targets]).mean()
        expected = expected_nll + 0.2 * torch.logdet(transition)
        torch.testing.assert_close(objective, expected)
        self.assertAlmostEqual(metrics['classification_loss'], float(expected_nll), places=10)

    def test_model_and_transition_receive_finite_gradients_and_update(self) -> None:
        torch.manual_seed(13)
        model = nn.Linear(4, 3)
        transition_model = DiagonallyDominantTransition(3, initial_flip_mass=0.05, max_flip_mass=0.49, temperature=1.0, seed=17)
        optimizer = build_volmin_optimizer(model, transition_model, {'name': 'adamw', 'model_lr': 0.01, 'transition_lr': 0.02, 'weight_decay': 0.0})
        model_before = model.weight.detach().clone()
        transition_before = transition_model.matrix().detach().clone()
        objective, _ = volmin_objective(model(torch.randn(8, 4)).to(torch.float64), torch.tensor([0, 1, 2, 0, 1, 2, 1, 0]), transition_model.matrix(), lambda_volume=0.001, determinant_tolerance=1e-08, condition_limit=1000000.0)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        parameters = list(model.parameters()) + list(transition_model.parameters())
        self.assertTrue(all((parameter.grad is not None for parameter in parameters)))
        self.assertTrue(all((bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)))
        optimizer.step()
        self.assertFalse(torch.equal(model_before, model.weight.detach()))
        self.assertFalse(torch.equal(transition_before, transition_model.matrix().detach()))

    def test_illegal_singular_or_wrong_sign_transition_fails(self) -> None:
        singular = torch.tensor([[0.6, 0.2, 0.2], [0.6, 0.2, 0.2], [0.1, 0.1, 0.8]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'diagonally dominant|singular'):
            validate_trainable_transition(singular, determinant_tolerance=1e-08, condition_limit=1000000.0)
        negative_sign = torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'diagonally dominant|determinant'):
            validate_trainable_transition(negative_sign, determinant_tolerance=1e-08, condition_limit=1000000.0)

# --- merged from test_pcse_volmin.py ---
class _pcse_volmin_PCSEVolMinWorkflowTest(unittest.TestCase):

    def test_transition_training_interruption_resume_and_model_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = _pcse_volmin__algorithm(run_dir)
            first.train_pretraining()
            source_hash = first.state.pretrained_checkpoint_sha256
            first.start_transition_training()
            first.train_transition(max_epochs=1)
            self.assertEqual(first.state.phase, PCSEPhase.TRANSITION_TRAINING)
            self.assertEqual(first.state.transition_completed_epochs, 1)
            initial_step = first.state.transition_global_step
            interrupted_payload = read_checkpoint(run_dir / 'last.pt', 'cpu')
            self.assertIsInstance(interrupted_payload['transition_model'], dict)
            self.assertIsInstance(interrupted_payload['transition_optimizer'], dict)
            self.assertIn('rng_state', interrupted_payload)
            first.close()
            resumed = _pcse_volmin__algorithm(run_dir)
            resumed.resume(run_dir / 'last.pt')
            self.assertEqual(resumed.state.transition_completed_epochs, 1)
            self.assertTrue(bool(resumed.transition_optimizer.state_dict()['state']))
            resumed.train_transition()
            self.assertEqual(resumed.state.phase, PCSEPhase.TRANSITION_READY)
            self.assertGreater(resumed.state.transition_global_step, initial_step)
            self.assertEqual(resumed.transition.metadata['source_pretrained_checkpoint_sha256'], source_hash)
            self.assertEqual(resumed.transition.metadata['feature_model_checkpoint_sha256'], resumed.state.volmin_final_checkpoint_sha256)
            self.assertNotEqual(resumed.state.feature_model_checkpoint_sha256, source_hash)
            statistics = resumed.estimate_statistics()
            self.assertEqual(statistics.provenance['feature_model_checkpoint_sha256'], resumed.state.volmin_final_checkpoint_sha256)
            resumed.close()

    def test_publish_validation_failure_preserves_formal_files_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            algorithm = _pcse_volmin__algorithm(run_dir)
            algorithm.train_pretraining()
            algorithm.start_transition_training()
            final_path = run_dir / 'volmin_final.pt'
            transition_path = run_dir / 'transition_artifact.npz'
            final_path.write_bytes(b'existing-final')
            transition_path.write_bytes(b'existing-transition')
            previous_final = final_path.read_bytes()
            previous_transition = transition_path.read_bytes()
            before_last = (run_dir / 'last.pt').read_bytes()
            with mock.patch.object(TransitionArtifact, 'load', side_effect=ValueError('injected transition validation')):
                with self.assertRaisesRegex(ValueError, 'injected'):
                    algorithm.train_transition()
            self.assertEqual(algorithm.state.phase, PCSEPhase.TRANSITION_TRAINING)
            self.assertEqual(algorithm.state.volmin_final_checkpoint_sha256, '')
            self.assertEqual(algorithm.state.transition_artifact_hash, '')
            self.assertEqual(final_path.read_bytes(), previous_final)
            self.assertEqual(transition_path.read_bytes(), previous_transition)
            self.assertNotEqual((run_dir / 'last.pt').read_bytes(), before_last)
            algorithm.close()

    def test_tiny_workflow_completed_resume_keeps_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'run'
            config = _pcse_volmin__config()
            run_pcse_experiment(config, output_dir=run_dir)
            final = json.loads((run_dir / 'final_metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(final['transition_backend'], 'paper_volmin')
            noise_summary = json.loads((run_dir / 'noise_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(noise_summary['validation_targets'], 'noisy')
            self.assertIsNone(noise_summary['effective_validation_subset_actual_rate'])
            names = ('volmin_final.pt', 'transition_artifact.npz', 'pcse_statistics.npz', 'pcse_gda.npz', 'pcse_ensemble.npz')
            before = {name: ((run_dir / name).read_bytes(), (run_dir / name).stat().st_mtime_ns) for name in names}
            run_pcse_experiment(config, resume=run_dir / 'last.pt')
            after = {name: ((run_dir / name).read_bytes(), (run_dir / name).stat().st_mtime_ns) for name in names}
            self.assertEqual(before, after)
            mismatch = deepcopy(config)
            mismatch['transition_stage']['lambda_volume'] *= 2.0
            with self.assertRaisesRegex(ValueError, 'method settings'):
                run_pcse_experiment(mismatch, resume=run_dir / 'last.pt')

# --- merged from test_pcse_workflow.py ---
from copy import deepcopy

# --- merged from test_pcse_workflow.py ---
import json

# --- merged from test_pcse_workflow.py ---
from pathlib import Path

# --- merged from test_pcse_workflow.py ---
import tempfile

# --- merged from test_pcse_workflow.py ---
import unittest

# --- merged from test_pcse_workflow.py ---
from unittest import mock

# --- merged from test_pcse_workflow.py ---
import numpy as np

# --- merged from test_pcse_workflow.py ---
import torch

# --- merged from test_pcse_workflow.py ---
from torch import nn

# --- merged from test_pcse_workflow.py ---
from torch.utils.data import DataLoader, Dataset

# --- merged from test_pcse_workflow.py ---
import yaml

# --- merged from test_pcse_workflow.py ---
from lnl_toolbox.algorithms.pcse import PCSEAlgorithm, PCSEConfig, PCSEPhase, PCSEState

# --- merged from test_pcse_workflow.py ---
from lnl_toolbox.algorithms.pcse.config import PCSEFeatureLayerConfig

# --- merged from test_pcse_workflow.py ---
from lnl_toolbox.algorithms.pcse.features import collect_pcse_features

# --- merged from test_pcse_workflow.py ---
from lnl_toolbox.algorithms.pcse.artifacts import PCSEEnsembleArtifact, persist_npz_atomically

# --- merged from test_pcse_workflow.py ---
from lnl_toolbox.data.multiclass_synthetic import MulticlassTensorDataset, generate_synthetic_multiclass

# --- merged from test_pcse_workflow.py ---
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

# --- merged from test_pcse_workflow.py ---
from lnl_toolbox.training.pcse_experiment import _PCSEMultilayerPerceptron, run_pcse_experiment

# --- merged from test_pcse_workflow.py ---
_pcse_workflow_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_pcse_workflow.py ---
_pcse_workflow_SMOKE_CONFIG = _pcse_workflow_ROOT / 'configs/experiment/pcse_multiclass_smoke.yaml'

# --- merged from test_pcse_workflow.py ---
def _pcse_workflow__load_smoke_config() -> dict:
    return yaml.safe_load(_pcse_workflow_SMOKE_CONFIG.read_text(encoding='utf-8'))

# --- merged from test_pcse_workflow.py ---
def _pcse_workflow__pretraining_algorithm(run_dir: Path, *, pretraining_epochs: int=2, fixed_model: bool=False) -> PCSEAlgorithm:
    config = _pcse_workflow__load_smoke_config()
    config['pretraining_stage']['epochs'] = pretraining_epochs
    torch.manual_seed(19)
    data = generate_synthetic_multiclass(90, 6, 3, 111, start_index=0, split='train')
    validation = generate_synthetic_multiclass(30, 6, 3, 112, start_index=90, split='validation')
    test = generate_synthetic_multiclass(30, 6, 3, 113, start_index=120, split='test')
    train_set = MulticlassTensorDataset(data)
    validation_set = MulticlassTensorDataset(validation)
    test_set = MulticlassTensorDataset(test)
    train_loader = DataLoader(train_set, batch_size=30, shuffle=False)
    validation_loader = DataLoader(validation_set, batch_size=30, shuffle=False)
    model = _pcse_workflow__FixedPCSEModel() if fixed_model else _PCSEMultilayerPerceptron(6, 12, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0 if fixed_model else 0.01)
    return PCSEAlgorithm(model=model, optimizer=optimizer, scheduler=None, loss=CrossEntropyLoss(), train_loader=train_loader, statistics_loader=train_loader, noisy_validation_loader=validation_loader, clean_test_loader=DataLoader(test_set, batch_size=30, shuffle=False), device=torch.device('cpu'), run_dir=run_dir, config=config, dataset='synthetic_multiclass', num_classes=3, noise_metadata={})

# --- merged from test_pcse_workflow.py ---
class _pcse_workflow__FeatureDataset(Dataset):

    def __init__(self, clean_values: torch.Tensor) -> None:
        self.clean_values = clean_values
        self.inputs = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
        self.targets = torch.tensor([0, 1, 2])
        self.indices = torch.tensor([30, 10, 20])

    def __len__(self) -> int:
        return 3

    def __getitem__(self, item: int):
        return {'input': self.inputs[item], 'target': self.targets[item], 'index': self.indices[item], 'clean_target': self.clean_values[item]}

# --- merged from test_pcse_workflow.py ---
class _pcse_workflow__HookModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.hidden1 = nn.Identity()
        self.hidden2 = nn.Linear(3, 3, bias=False)
        self.classifier = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.hidden2.weight.copy_(torch.eye(3))
            self.classifier.weight.copy_(torch.eye(3))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.hidden2(self.hidden1(values)))

# --- merged from test_pcse_workflow.py ---
class _pcse_workflow__FixedPCSEModel(nn.Module):
    """Near-oracle noisy-posterior fixture without reading clean targets."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden1 = nn.Identity()
        self.hidden2 = nn.Linear(6, 6, bias=False)
        self.classifier = nn.Linear(6, 3, bias=False)
        with torch.no_grad():
            self.hidden2.weight.copy_(torch.eye(6))
            self.classifier.weight.zero_()
            self.classifier.weight[:, :3].copy_(50.0 * torch.eye(3))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.hidden2(self.hidden1(values)))

# --- merged from test_pcse_workflow.py ---
class _pcse_workflow_PCSEWorkflowTest(unittest.TestCase):

    def test_config_requires_multilayer_and_valid_backend(self) -> None:
        config = _pcse_workflow__load_smoke_config()
        parsed = PCSEConfig.from_mapping(config)
        self.assertEqual(parsed.transition_backend, 'dual_t')
        self.assertEqual(tuple((layer.name for layer in parsed.feature_layers)), ('hidden1', 'hidden2'))
        single = deepcopy(config)
        single['feature_stage']['layers'] = [{'name': 'hidden1'}]
        with self.assertRaisesRegex(ValueError, 'at least two'):
            PCSEConfig.from_mapping(single)
        binary = deepcopy(config)
        binary['data']['num_classes'] = 2
        with self.assertRaisesRegex(ValueError, 'num_classes >= 3'):
            PCSEConfig.from_mapping(binary)
        paper_backend = deepcopy(config)
        paper_backend['transition_stage'] = {'name': 'paper_volmin', 'epochs': 2, 'lambda_volume': 0.001, 'optimizer': {'name': 'adamw', 'model_lr': 0.001, 'transition_lr': 0.01, 'weight_decay': 0.0}, 'scheduler': {'name': 'none'}, 'parameterization': {'name': 'diagonal_dominant', 'initial_flip_mass': 0.05, 'max_flip_mass': 0.49, 'temperature': 1.0, 'seed': 3}, 'determinant_tolerance': 1e-06, 'condition_limit': 1000000.0}
        parsed_paper = PCSEConfig.from_mapping(paper_backend)
        self.assertEqual(parsed_paper.transition_backend, 'paper_volmin')
        self.assertEqual(parsed_paper.transition_backend_config['lambda_volume'], 0.001)

    def test_external_checkpoint_config_is_strict_and_train_mode_unchanged(self) -> None:
        config = _pcse_workflow__load_smoke_config()
        self.assertEqual(PCSEConfig.from_mapping(config).pretraining.mode, 'train')
        external = deepcopy(config)
        external['pretraining_stage']['mode'] = 'external_checkpoint'
        external['pretraining_stage']['epochs'] = 0
        model = dict(external['pretraining_stage']['model'])
        external['pretraining_stage']['source'] = {'adapter': 'upm_main_best', 'run_directory_env': 'LNL_PCSE_SOURCE_RUN', 'checkpoint_sha256': 'a' * 64, 'manifest_sha256': 'b' * 64, 'mapping_hash': 'c' * 64, 'dataset_fingerprint': 'd' * 64, 'model': model}
        parsed = PCSEConfig.from_mapping(external)
        self.assertEqual(parsed.pretraining.mode, 'external_checkpoint')
        invalid = deepcopy(external)
        invalid['pretraining_stage']['source']['adapter'] = 'generic'
        with self.assertRaisesRegex(ValueError, 'upm_main_best'):
            PCSEConfig.from_mapping(invalid)

    def test_external_adoption_persists_provenance_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _pcse_workflow__load_smoke_config()
            config['pretraining_stage']['mode'] = 'external_checkpoint'
            config['pretraining_stage']['epochs'] = 0
            model = dict(config['pretraining_stage']['model'])
            config['pretraining_stage']['source'] = {'adapter': 'upm_main_best', 'run_directory_env': 'PCSE_TEST_SOURCE', 'checkpoint_sha256': 'a' * 64, 'manifest_sha256': 'b' * 64, 'mapping_hash': 'c' * 64, 'dataset_fingerprint': 'd' * 64, 'model': model}
            source = {'adapter': 'upm_main_best', 'checkpoint': {'sha256': 'a' * 64}}
            algorithm = _pcse_workflow__pretraining_algorithm(Path(directory))
            algorithm.config = config
            algorithm.method_config = PCSEConfig.from_mapping(config)
            algorithm.external_source_provenance = source
            algorithm.adopt_external_pretrained(completed_epochs=3, global_step=9, best_epoch=1, validation_accuracy=0.5, validation_loss=1.0)
            self.assertEqual(algorithm.state.phase, PCSEPhase.PRETRAINED)
            checkpoint = torch.load(Path(directory) / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['external_source_provenance'], source)
            algorithm.close()

    def test_phase_machine_rejects_illegal_transition(self) -> None:
        state = PCSEState()
        with self.assertRaisesRegex(ValueError, 'illegal PCSE phase'):
            state.advance(PCSEPhase.TRANSITION_READY)

    def test_feature_extraction_preserves_layer_order_and_stable_indices(self) -> None:
        model = _pcse_workflow__HookModel()
        loader = DataLoader(_pcse_workflow__FeatureDataset(torch.tensor([2, 2, 2])), batch_size=2, shuffle=False)
        result = collect_pcse_features(model, loader, 'cpu', dataset='synthetic', split='train', layers=(PCSEFeatureLayerConfig('hidden2', 'global_average'), PCSEFeatureLayerConfig('hidden1', 'global_average')))
        self.assertEqual(result.layer_names, ('hidden2', 'hidden1'))
        np.testing.assert_array_equal(result.snapshots[0].global_indices, np.array([10, 20, 30]))
        np.testing.assert_array_equal(result.snapshots[0].global_indices, result.snapshots[1].global_indices)
        self.assertEqual(len(model.hidden1._forward_hooks), 0)
        self.assertEqual(len(model.hidden2._forward_hooks), 0)

    def test_logits_cannot_be_used_as_hidden_features(self) -> None:
        model = _pcse_workflow__HookModel()
        loader = DataLoader(_pcse_workflow__FeatureDataset(torch.tensor([0, 1, 2])), batch_size=3)
        with self.assertRaisesRegex(ValueError, 'model logits output'):
            collect_pcse_features(model, loader, 'cpu', dataset='synthetic', split='train', layers=(PCSEFeatureLayerConfig('classifier', 'global_average'), PCSEFeatureLayerConfig('hidden2', 'global_average')))
        self.assertEqual(len(model.classifier._forward_hooks), 0)

    def test_feature_extraction_does_not_read_clean_label_field(self) -> None:
        model = _pcse_workflow__HookModel()
        layers = (PCSEFeatureLayerConfig('hidden1', 'global_average'), PCSEFeatureLayerConfig('hidden2', 'global_average'))
        first = collect_pcse_features(model, DataLoader(_pcse_workflow__FeatureDataset(torch.tensor([0, 1, 2])), batch_size=3), 'cpu', dataset='synthetic', split='train', layers=layers)
        second = collect_pcse_features(model, DataLoader(_pcse_workflow__FeatureDataset(torch.tensor([2, 2, 2])), batch_size=3), 'cpu', dataset='synthetic', split='train', layers=layers)
        for left, right in zip(first.snapshots, second.snapshots):
            np.testing.assert_array_equal(left.features, right.features)
            np.testing.assert_array_equal(left.noisy_targets, right.noisy_targets)

    def test_pretraining_interruption_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _pcse_workflow__pretraining_algorithm(Path(directory))
            first.train_pretraining(max_epochs=1)
            self.assertEqual(first.state.phase, PCSEPhase.PRETRAINING)
            self.assertEqual(first.state.pretraining_completed_epochs, 1)
            first.close()
            resumed = _pcse_workflow__pretraining_algorithm(Path(directory))
            resumed.resume(Path(directory) / 'last.pt')
            self.assertEqual(resumed.state.pretraining_completed_epochs, 1)
            initial_step = resumed.state.pretraining_global_step
            resumed.train_pretraining()
            self.assertEqual(resumed.state.phase, PCSEPhase.PRETRAINED)
            self.assertEqual(resumed.state.pretraining_completed_epochs, 2)
            self.assertGreater(resumed.state.pretraining_global_step, initial_step)
            resumed.close()

    def test_transition_persistence_failure_does_not_advance_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            algorithm = _pcse_workflow__pretraining_algorithm(run_dir, pretraining_epochs=1, fixed_model=True)
            algorithm.train_pretraining()
            before_last = (run_dir / 'last.pt').read_bytes()
            with mock.patch('lnl_toolbox.algorithms.pcse.algorithm._persist_transition', side_effect=OSError('injected transition save failure')):
                with self.assertRaisesRegex(OSError, 'injected'):
                    algorithm.estimate_transition()
            self.assertEqual(algorithm.state.phase, PCSEPhase.PRETRAINED)
            self.assertEqual(algorithm.state.posterior_snapshot_hash, '')
            self.assertEqual(algorithm.state.transition_artifact_hash, '')
            self.assertEqual((run_dir / 'last.pt').read_bytes(), before_last)
            algorithm.close()

    def test_atomic_artifact_failure_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'ensemble.npz'
            original = PCSEEnsembleArtifact(('h1', 'h2'), np.array([0.5, 0.5]), {'version': 1})
            persist_npz_atomically(original, destination, PCSEEnsembleArtifact.load)
            before = destination.read_bytes()
            replacement = PCSEEnsembleArtifact(('h1', 'h2'), np.array([0.6, 0.4]), {'version': 2})
            with self.assertRaisesRegex(ValueError, 'injected validation'):
                persist_npz_atomically(replacement, destination, lambda _path: (_ for _ in ()).throw(ValueError('injected validation failure')))
            self.assertEqual(destination.read_bytes(), before)

    def test_ensemble_interruption_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = _pcse_workflow__pretraining_algorithm(run_dir, pretraining_epochs=1, fixed_model=True)
            first.train_pretraining()
            first.estimate_transition()
            first.estimate_statistics()
            first.build_gda()
            first.start_ensemble_training()
            first.train_ensemble(max_epochs=1)
            self.assertEqual(first.state.phase, PCSEPhase.ENSEMBLE_TRAINING)
            self.assertEqual(first.state.ensemble_completed_epochs, 1)
            first.close()
            resumed = _pcse_workflow__pretraining_algorithm(run_dir, pretraining_epochs=1, fixed_model=True)
            resumed.resume(run_dir / 'last.pt')
            self.assertEqual(resumed.state.ensemble_completed_epochs, 1)
            resumed.train_ensemble()
            self.assertEqual(resumed.state.phase, PCSEPhase.COMPLETED)
            self.assertEqual(resumed.state.ensemble_completed_epochs, 3)
            resumed.close()

    def test_two_layer_workflow_completed_resume_and_artifact_damage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'run'
            result = run_pcse_experiment(_pcse_workflow__load_smoke_config(), output_dir=run_dir)
            self.assertEqual(result, run_dir.resolve())
            final = json.loads((run_dir / 'final_metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(final['method'], 'pcse')
            self.assertEqual(final['transition_backend'], 'dual_t')
            self.assertEqual(len(final['ensemble_weights']), 2)
            self.assertTrue(all((value > 0 for value in final['ensemble_weights'])))
            self.assertAlmostEqual(sum(final['ensemble_weights']), 1.0)
            with np.load(run_dir / 'posterior_snapshot.npz', allow_pickle=False) as snapshot:
                posterior = snapshot['noisy_probabilities']
            self.assertEqual(posterior.shape[1], 3)
            self.assertTrue(np.isfinite(posterior).all())
            np.testing.assert_allclose(posterior.sum(axis=1), 1.0, rtol=1e-06, atol=1e-08)
            artifact_names = ('posterior_snapshot.npz', 'transition_artifact.npz', 'pcse_statistics.npz', 'pcse_gda.npz', 'pcse_ensemble.npz')
            before = {name: ((run_dir / name).read_bytes(), (run_dir / name).stat().st_mtime_ns) for name in artifact_names}
            run_pcse_experiment(_pcse_workflow__load_smoke_config(), resume=run_dir / 'last.pt')
            after = {name: ((run_dir / name).read_bytes(), (run_dir / name).stat().st_mtime_ns) for name in artifact_names}
            self.assertEqual(before, after)
            transition_path = run_dir / 'transition_artifact.npz'
            valid_transition = transition_path.read_bytes()
            transition_path.write_bytes(b'damaged')
            with self.assertRaises((ValueError, OSError)):
                run_pcse_experiment(_pcse_workflow__load_smoke_config(), resume=run_dir / 'last.pt')
            transition_path.write_bytes(valid_transition)
            mismatch = _pcse_workflow__load_smoke_config()
            mismatch['feature_stage']['layers'].reverse()
            with self.assertRaisesRegex(ValueError, 'method settings'):
                run_pcse_experiment(mismatch, resume=run_dir / 'last.pt')
