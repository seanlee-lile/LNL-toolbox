"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_importance_reweighting.py ---
from dataclasses import dataclass

# --- merged from test_importance_reweighting.py ---
import math

# --- merged from test_importance_reweighting.py ---
import unittest

# --- merged from test_importance_reweighting.py ---
import torch

# --- merged from test_importance_reweighting.py ---
from lnl_toolbox.treatments import BinaryRCNImportanceWeightProvider, BinaryRCNWeightInput, ReductionSpec, SupervisedWeightInput, WeightContributionAdapter, WeightResult, reduce_per_sample_loss

# --- merged from test_importance_reweighting.py ---
class _importance_reweighting_ImportanceReweightingTest(unittest.TestCase):

    def test_binary_rcn_requires_explicit_posterior_input(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        with self.assertRaisesRegex(TypeError, 'BinaryRCNWeightInput'):
            provider.compute(SupervisedWeightInput(logits=torch.tensor([[2.0, 0.0]]), noisy_targets=torch.tensor([0]), sample_indices=torch.tensor([3]), per_sample_loss=torch.tensor([0.1])))

    def test_binary_asymmetric_rcn_weights_match_manual_formula(self):
        provider = BinaryRCNImportanceWeightProvider(rho_positive=0.2, rho_negative=0.1)
        result = provider.compute(BinaryRCNWeightInput(posterior_probabilities=torch.tensor([[0.8, 0.2], [0.3, 0.7]]), observed_targets=torch.tensor([0, 1])))
        expected = torch.tensor([(0.8 - 0.2) / (0.7 * 0.8), (0.7 - 0.1) / (0.7 * 0.7)])
        self.assertTrue(torch.allclose(result.sample_weights, expected))
        self.assertEqual(set(result.metrics), {'weight_mean', 'weight_min', 'weight_max', 'zero_weight_ratio'})
        for value in result.metrics.values():
            self.assertIs(type(value), float)
            self.assertTrue(math.isfinite(value))

    def test_rate_direction_and_observed_label_probability_are_correct(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        posterior = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
        weights = provider.compute(BinaryRCNWeightInput(posterior_probabilities=posterior, observed_targets=torch.tensor([0, 1]))).sample_weights
        expected_target_zero = (0.8 - 0.2) / (0.7 * 0.8)
        expected_target_one = (0.2 - 0.1) / (0.7 * 0.2)
        self.assertAlmostEqual(weights[0].item(), expected_target_zero, places=6)
        self.assertAlmostEqual(weights[1].item(), expected_target_one, places=6)
        self.assertNotAlmostEqual(weights[1].item(), expected_target_zero, places=6)

    def test_zero_noise_rates_produce_unit_weights_for_nonzero_q(self):
        provider = BinaryRCNImportanceWeightProvider(0.0, 0.0)
        weights = provider.compute(BinaryRCNWeightInput(posterior_probabilities=torch.tensor([[0.9, 0.1], [0.3, 0.7]]), observed_targets=torch.tensor([0, 1]))).sample_weights
        self.assertTrue(torch.equal(weights, torch.ones(2)))

    def test_zero_observed_probability_is_assigned_zero_without_nonfinite_value(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        result = provider.compute(BinaryRCNWeightInput(posterior_probabilities=torch.tensor([[0.0, 1.0], [0.3, 0.7]]), observed_targets=torch.tensor([0, 1])))
        self.assertEqual(result.sample_weights[0].item(), 0.0)
        self.assertTrue(bool(torch.isfinite(result.sample_weights).all().item()))
        self.assertEqual(result.metrics['zero_weight_ratio'], 0.5)

    def test_invalid_noise_rates_are_rejected(self):
        invalid = ((-0.1, 0.1), (1.0, 0.0), (0.1, -0.1), (0.0, 1.0), (0.6, 0.4), (float('nan'), 0.1))
        for rho_positive, rho_negative in invalid:
            with self.subTest(rho_positive=rho_positive, rho_negative=rho_negative), self.assertRaises((TypeError, ValueError)):
                BinaryRCNImportanceWeightProvider(rho_positive, rho_negative)

    def test_invalid_posterior_and_targets_are_rejected(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        cases = ((torch.tensor([0.4, 0.6]), torch.tensor([0]), 'shape'), (torch.tensor([[1, 0]]), torch.tensor([0]), 'floating-point'), (torch.tensor([[float('nan'), float('nan')]]), torch.tensor([0]), 'finite'), (torch.tensor([[1.1, -0.1]]), torch.tensor([0]), '\\[0, 1\\]'), (torch.tensor([[0.4, 0.4]]), torch.tensor([0]), 'sum to one'), (torch.tensor([[0.4, 0.6]]), torch.tensor([0.0]), 'integer'), (torch.tensor([[0.4, 0.6]]), torch.tensor([2]), 'binary'))
        for posterior, targets, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                provider.compute(BinaryRCNWeightInput(posterior, targets))
        with self.assertRaisesRegex(ValueError, 'same device'):
            provider.compute(BinaryRCNWeightInput(torch.tensor([[0.4, 0.6]]), torch.empty(1, dtype=torch.long, device='meta')))

    def test_negative_weights_fail_but_roundoff_scale_negative_is_clamped(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        with self.assertRaisesRegex(ValueError, 'negative importance weight'):
            provider.compute(BinaryRCNWeightInput(posterior_probabilities=torch.tensor([[0.1, 0.9]], dtype=torch.float64), observed_targets=torch.tensor([0])))
        result = provider.compute(BinaryRCNWeightInput(posterior_probabilities=torch.tensor([[0.2 - 1e-08, 0.8 + 1e-08]], dtype=torch.float64), observed_targets=torch.tensor([0])))
        self.assertEqual(result.sample_weights.item(), 0.0)

    def test_output_weights_are_detached(self):
        posterior = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
        result = BinaryRCNImportanceWeightProvider(0.2, 0.1).compute(BinaryRCNWeightInput(posterior, torch.tensor([0, 1])))
        self.assertIs(result.sample_weights.requires_grad, False)

    def test_weight_contribution_adapter_uses_all_true_mask_and_metrics(self):
        provider = BinaryRCNImportanceWeightProvider(0.2, 0.1)
        contribution = WeightContributionAdapter(provider).resolve(BinaryRCNWeightInput(posterior_probabilities=torch.tensor([[0.8, 0.2], [0.3, 0.7]]), observed_targets=torch.tensor([0, 1])))
        self.assertTrue(torch.equal(contribution.selected_mask, torch.ones(2, dtype=torch.bool)))
        self.assertEqual(set(contribution.metrics), {'weight_mean', 'weight_min', 'weight_max', 'zero_weight_ratio'})

    def test_weight_contribution_adapter_rejects_non_float_or_nonfinite_metrics(self):

        class InvalidMetricProvider:

            def __init__(self, value):
                self.value = value

            def compute(self, weight_input):
                return WeightResult(sample_weights=torch.ones(1), metrics={'invalid': self.value})
        weight_input = BinaryRCNWeightInput(posterior_probabilities=torch.tensor([[0.4, 0.6]]), observed_targets=torch.tensor([1]))
        with self.assertRaisesRegex(TypeError, 'Python float'):
            WeightContributionAdapter(InvalidMetricProvider(1)).resolve(weight_input)
        with self.assertRaisesRegex(ValueError, 'finite'):
            WeightContributionAdapter(InvalidMetricProvider(float('nan'))).resolve(weight_input)

    def test_batch_mean_weighted_loss_value_and_gradient_match_paper_objective(self):

        class FixedWeightProvider:

            def compute(self, weight_input):
                return WeightResult(sample_weights=torch.tensor([2.0, 4.0]), metrics={'weight_mean': 3.0})
        contribution = WeightContributionAdapter(FixedWeightProvider()).resolve(BinaryRCNWeightInput(posterior_probabilities=torch.tensor([[0.8, 0.2], [0.3, 0.7]]), observed_targets=torch.tensor([0, 1])))
        losses = torch.tensor([1.0, 3.0], requires_grad=True)
        objective = reduce_per_sample_loss(losses, contribution, ReductionSpec('batch_mean'))
        weight_sum_mean = reduce_per_sample_loss(losses, contribution, ReductionSpec('weight_sum_mean'))
        self.assertEqual(objective.item(), 7.0)
        self.assertNotEqual(objective.item(), weight_sum_mean.item())
        objective.backward()
        self.assertTrue(torch.equal(losses.grad, torch.tensor([1.0, 2.0])))

    def test_adapter_accepts_a_non_posterior_provider_input(self):

        @dataclass(frozen=True)
        class DummyWeightInput:
            scores: torch.Tensor

        class DummyWeightProvider:

            def compute(self, weight_input):
                weights = weight_input.scores.detach()
                return WeightResult(sample_weights=weights, metrics={'weight_mean': float(weights.mean().item())})
        dummy_input = DummyWeightInput(scores=torch.tensor([0.25, 0.75], requires_grad=True))
        contribution = WeightContributionAdapter(DummyWeightProvider()).resolve(dummy_input)
        self.assertTrue(torch.equal(contribution.selected_mask, torch.tensor([True, True])))
        self.assertTrue(torch.equal(contribution.sample_weights, torch.tensor([0.25, 0.75])))
        self.assertIs(contribution.sample_weights.requires_grad, False)
        self.assertEqual(contribution.metrics, {'weight_mean': 0.5})

# --- merged from test_importance_reweighting_kliep.py ---
from copy import deepcopy

# --- merged from test_importance_reweighting_kliep.py ---
from pathlib import Path

# --- merged from test_importance_reweighting_kliep.py ---
import tempfile

# --- merged from test_importance_reweighting_kliep.py ---
import unittest

# --- merged from test_importance_reweighting_kliep.py ---
import numpy as np

# --- merged from test_importance_reweighting_kliep.py ---
import torch

# --- merged from test_importance_reweighting_kliep.py ---
import yaml

# --- merged from test_importance_reweighting_kliep.py ---
from lnl_toolbox.algorithms.importance_reweighting import ImportanceReweightingConfig, KLIEPBinaryNoisyPosteriorEstimator, PaperRawMinNoiseRateEstimator

# --- merged from test_importance_reweighting_kliep.py ---
from lnl_toolbox.data.binary_synthetic import generate_synthetic_binary_high_dim

# --- merged from test_importance_reweighting_kliep.py ---
from lnl_toolbox.training.checkpoint import read_checkpoint

# --- merged from test_importance_reweighting_kliep.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_importance_reweighting_kliep.py ---
_importance_reweighting_kliep_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'experiment' / 'importance_reweighting_binary_high_dim_smoke.yaml'

# --- merged from test_importance_reweighting_kliep.py ---
def _importance_reweighting_kliep_load_config() -> dict:
    return yaml.safe_load(_importance_reweighting_kliep_CONFIG.read_text(encoding='utf-8'))

# --- merged from test_importance_reweighting_kliep.py ---
def _importance_reweighting_kliep_build_estimator() -> KLIEPBinaryNoisyPosteriorEstimator:
    return KLIEPBinaryNoisyPosteriorEstimator(bandwidth=4.0, max_centers=16, max_iterations=150, learning_rate=0.02, tolerance=1e-07, epsilon=1e-12, seed=37)

# --- merged from test_importance_reweighting_kliep.py ---
class _importance_reweighting_kliep_KLIEPPosteriorTest(unittest.TestCase):

    def test_high_dimensional_data_is_balanced_and_deterministic(self) -> None:
        first = generate_synthetic_binary_high_dim(40, 20, 5, split='train')
        second = generate_synthetic_binary_high_dim(40, 20, 5, split='train')
        self.assertEqual(first.features.shape, (40, 20))
        self.assertEqual(first.dataset, 'synthetic_binary_high_dim')
        np.testing.assert_array_equal(np.bincount(first.labels), [20, 20])
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.global_indices, second.global_indices)

    def test_posterior_is_valid_deterministic_and_index_aligned(self) -> None:
        data = generate_synthetic_binary_high_dim(80, 20, 9, start_index=100, split='train')
        estimator = _importance_reweighting_kliep_build_estimator()
        first = estimator.fit_predict(data.features, data.labels, data.global_indices, dataset=data.dataset, split=data.split)
        second = estimator.fit_predict(data.features, data.labels, data.global_indices, dataset=data.dataset, split=data.split)
        self.assertEqual(first.noisy_probabilities.shape, (80, 2))
        self.assertTrue(np.isfinite(first.noisy_probabilities).all())
        self.assertTrue((first.noisy_probabilities >= 0.0).all())
        np.testing.assert_allclose(first.noisy_probabilities.sum(axis=1), np.ones(80), rtol=1e-08, atol=1e-10)
        np.testing.assert_array_equal(first.global_indices, np.sort(data.global_indices))
        np.testing.assert_array_equal(first.noisy_probabilities, second.noisy_probabilities)

    def test_input_permutation_preserves_stable_index_mapping(self) -> None:
        data = generate_synthetic_binary_high_dim(64, 20, 11, start_index=17, split='train')
        permutation = np.random.default_rng(99).permutation(64)
        estimator = _importance_reweighting_kliep_build_estimator()
        original = estimator.fit_predict(data.features, data.labels, data.global_indices, dataset=data.dataset, split=data.split)
        permuted = estimator.fit_predict(data.features[permutation], data.labels[permutation], data.global_indices[permutation], dataset=data.dataset, split=data.split)
        np.testing.assert_array_equal(original.global_indices, permuted.global_indices)
        np.testing.assert_allclose(original.noisy_probabilities, permuted.noisy_probabilities, rtol=0.0, atol=0.0)

    def test_density_ratio_satisfies_empirical_normalization(self) -> None:
        data = generate_synthetic_binary_high_dim(60, 20, 13, split='train')
        estimator = _importance_reweighting_kliep_build_estimator()
        ordered = np.argsort(data.global_indices, kind='stable')
        values = data.features[ordered]
        targets = data.labels[ordered]
        fit = estimator._fit_ratio(values[targets == 0], values, class_index=0)
        ratio = estimator._kernel(values, fit.centers) @ fit.coefficients
        self.assertAlmostEqual(float(ratio.mean()), 1.0, places=10)
        self.assertTrue(np.isfinite(ratio).all())
        self.assertTrue((ratio >= 0.0).all())

    def test_invalid_shapes_labels_and_indices_are_rejected(self) -> None:
        data = generate_synthetic_binary_high_dim(20, 20, 17, split='train')
        estimator = _importance_reweighting_kliep_build_estimator()
        cases = ((data.features[:, :2], data.labels, data.global_indices, 'D > 2'), (data.features, np.zeros(20, dtype=np.int64), data.global_indices, 'both binary classes'), (data.features, np.where(data.labels == 1, 2, 0), data.global_indices, 'only 0 and 1'), (data.features, data.labels, np.zeros(20, dtype=np.int64), 'unique'))
        for features, targets, indices, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    estimator.fit_predict(features, targets, indices, dataset=data.dataset, split=data.split)

    def test_raw_min_rates_from_kliep_snapshot_are_legal(self) -> None:
        data = generate_synthetic_binary_high_dim(80, 20, 23, split='train')
        snapshot = _importance_reweighting_kliep_build_estimator().fit_predict(data.features, data.labels, data.global_indices, dataset=data.dataset, split=data.split)
        rates = PaperRawMinNoiseRateEstimator().estimate(snapshot)
        self.assertGreaterEqual(rates.rho_positive, 0.0)
        self.assertGreaterEqual(rates.rho_negative, 0.0)
        self.assertLess(rates.rho_positive + rates.rho_negative, 1.0)

# --- merged from test_importance_reweighting_kliep.py ---
class _importance_reweighting_kliep_KLIEPWorkflowTest(unittest.TestCase):

    def test_config_requires_matching_high_dimension(self) -> None:
        config = _importance_reweighting_kliep_load_config()
        parsed = ImportanceReweightingConfig.from_mapping(config)
        self.assertEqual(parsed.data['dimension'], 20)
        self.assertEqual(parsed.posterior_stage['name'], 'kliep')
        invalid = deepcopy(config)
        invalid['model']['in_features'] = 19
        with self.assertRaisesRegex(ValueError, 'in_features'):
            ImportanceReweightingConfig.from_mapping(invalid)
        invalid = deepcopy(config)
        invalid['num_classes'] = 10
        with self.assertRaisesRegex(ValueError, 'num_classes'):
            ImportanceReweightingConfig.from_mapping(invalid)

    def test_high_dimensional_smoke_and_resume_preserve_artifacts(self) -> None:
        config = _importance_reweighting_kliep_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            result = run_experiment(config, run_dir)
            snapshot = result / 'posterior_snapshot.npz'
            rates = result / 'noise_rate_artifact.npz'
            snapshot_before = snapshot.read_bytes()
            rates_before = rates.read_bytes()
            first = read_checkpoint(result / 'last.pt', 'cpu')
            self.assertEqual(first['method_state']['final_completed_epochs'], 2)
            self.assertEqual(first['method_state']['final_global_step'], 8)
            self.assertEqual(first['posterior_backend_identity']['name'], 'kliep')
            self.assertEqual(first['posterior_backend_identity']['feature_dimension'], 20)
            self.assertEqual(first['posterior_backend_hash'], first['method_state']['posterior_backend_hash'])
            config['trainer']['epochs'] = 3
            run_experiment(config, resume=result / 'last.pt')
            final = read_checkpoint(result / 'last.pt', 'cpu')
            self.assertEqual(final['method_state']['final_completed_epochs'], 3)
            self.assertEqual(final['method_state']['final_global_step'], 12)
            self.assertEqual(snapshot.read_bytes(), snapshot_before)
            self.assertEqual(rates.read_bytes(), rates_before)
            metrics = yaml.safe_load((result / 'final_metrics.json').read_text(encoding='utf-8'))
            for name in ('test_accuracy', 'test_loss', 'rho_positive_hat', 'rho_negative_hat'):
                self.assertTrue(np.isfinite(metrics[name]))
            self.assertEqual(metrics['reduction'], 'batch_mean')

    def test_resume_rejects_backend_and_parameter_mismatch(self) -> None:
        config = _importance_reweighting_kliep_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            run_experiment(config, run_dir)
            checkpoint = run_dir / 'last.pt'
            changed = deepcopy(config)
            changed['posterior_stage']['bandwidth'] = 3.5
            changed['trainer']['epochs'] = 3
            with self.assertRaisesRegex(ValueError, 'backend identity'):
                run_experiment(changed, resume=checkpoint)
            changed = deepcopy(config)
            changed['posterior_stage']['name'] = 'kde'
            changed['posterior_stage'].pop('max_centers')
            changed['posterior_stage'].pop('max_iterations')
            changed['posterior_stage'].pop('learning_rate')
            changed['posterior_stage'].pop('tolerance')
            changed['posterior_stage'].pop('epsilon')
            changed['posterior_stage'].pop('seed')
            changed['trainer']['epochs'] = 3
            with self.assertRaisesRegex(ValueError, 'backend identity'):
                run_experiment(changed, resume=checkpoint)
            changed = deepcopy(config)
            changed['data']['dimension'] = 19
            changed['model']['in_features'] = 19
            changed['trainer']['epochs'] = 3
            with self.assertRaisesRegex(ValueError, 'data_manifest|data identity'):
                run_experiment(changed, resume=checkpoint)

    def test_resume_rejects_checkpoint_backend_hash_corruption(self) -> None:
        config = _importance_reweighting_kliep_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            run_experiment(config, run_dir)
            payload = read_checkpoint(run_dir / 'last.pt', 'cpu')
            payload['posterior_backend_hash'] = 'f' * 64
            bad = run_dir / 'bad-backend.pt'
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, 'backend hash'):
                run_experiment(config, resume=bad)

# --- merged from test_importance_reweighting_method.py ---
from copy import deepcopy

# --- merged from test_importance_reweighting_method.py ---
from pathlib import Path

# --- merged from test_importance_reweighting_method.py ---
import tempfile

# --- merged from test_importance_reweighting_method.py ---
import unittest

# --- merged from test_importance_reweighting_method.py ---
from unittest import mock

# --- merged from test_importance_reweighting_method.py ---
import numpy as np

# --- merged from test_importance_reweighting_method.py ---
import torch

# --- merged from test_importance_reweighting_method.py ---
import yaml

# --- merged from test_importance_reweighting_method.py ---
from lnl_toolbox.algorithms.importance_reweighting import ImportanceReweightingAlgorithm, ImportanceReweightingConfig, ImportanceReweightingPhase, ImportanceReweightingState, IndexedBinaryRCNWeightProvider, KDEBinaryNoisyPosteriorEstimator, NoiseRateArtifact, PaperRawMinNoiseRateEstimator, validate_binary_posterior_snapshot

# --- merged from test_importance_reweighting_method.py ---
from lnl_toolbox.data.binary_synthetic import BinaryTensorDataset, SyntheticBinaryData, generate_synthetic_binary_2d, validate_zero_one_labels

# --- merged from test_importance_reweighting_method.py ---
from lnl_toolbox.noise.binary_rcn import generate_binary_asymmetric_rcn, validate_binary_rcn_manifest

# --- merged from test_importance_reweighting_method.py ---
from lnl_toolbox.noise.manifest import NoiseManifest

# --- merged from test_importance_reweighting_method.py ---
from lnl_toolbox.noise.estimators import PosteriorSnapshot

# --- merged from test_importance_reweighting_method.py ---
from lnl_toolbox.training.checkpoint import read_checkpoint

# --- merged from test_importance_reweighting_method.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_importance_reweighting_method.py ---
from lnl_toolbox.treatments import SupervisedWeightInput

# --- merged from test_importance_reweighting_method.py ---
_importance_reweighting_method_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_importance_reweighting_method.py ---
_importance_reweighting_method_CONFIG = _importance_reweighting_method_ROOT / 'configs/experiment/importance_reweighting_binary_smoke.yaml'

# --- merged from test_importance_reweighting_method.py ---
_importance_reweighting_method_UCI_CONFIG = _importance_reweighting_method_ROOT / 'configs/reproduction/uci_heart_importance_reweighting.yaml'

# --- merged from test_importance_reweighting_method.py ---
def _importance_reweighting_method_load_config() -> dict:
    return yaml.safe_load(_importance_reweighting_method_CONFIG.read_text(encoding='utf-8'))

# --- merged from test_importance_reweighting_method.py ---
class _importance_reweighting_method_BinaryMethodBoundaryTest(unittest.TestCase):

    def test_uci_config_has_strict_binary_schema(self) -> None:
        config = yaml.safe_load(_importance_reweighting_method_UCI_CONFIG.read_text(encoding='utf-8'))
        parsed = ImportanceReweightingConfig.from_mapping(config)
        self.assertEqual(parsed.data['name'], 'uci_statlog_heart')
        self.assertEqual(parsed.data['dimension'], 13)
        self.assertEqual(tuple(parsed.data['preprocessing']['label_values']), ('1', '2'))
        self.assertEqual(parsed.posterior_stage['name'], 'kliep')

    def test_config_rejects_multiclass_and_wrong_model_output(self) -> None:
        for classes in (3, 10):
            config = _importance_reweighting_method_load_config()
            config['num_classes'] = classes
            with self.assertRaisesRegex(ValueError, 'num_classes'):
                ImportanceReweightingConfig.from_mapping(config)
        config = _importance_reweighting_method_load_config()
        config['model']['num_classes'] = 3
        with self.assertRaisesRegex(ValueError, '2 classes'):
            ImportanceReweightingConfig.from_mapping(config)

    def test_label_validation_rejects_nonbinary_and_single_class(self) -> None:
        with self.assertRaisesRegex(ValueError, 'only 0 and 1'):
            validate_zero_one_labels(np.array([0, 2]), owner='test')
        with self.assertRaisesRegex(ValueError, 'both binary classes'):
            validate_zero_one_labels(np.zeros(3, dtype=np.int64), owner='test', require_both_classes=True)

    def test_dataset_requires_two_features(self) -> None:
        with self.assertRaisesRegex(ValueError, '\\[N, 2\\]'):
            SyntheticBinaryData(np.zeros((4, 3)), np.array([0, 1, 0, 1]), np.arange(4), 'train')

    def test_training_dataset_exposes_no_clean_truth(self) -> None:
        data = generate_synthetic_binary_2d(4, 3, split='train')
        item = BinaryTensorDataset(data, 1 - data.labels)[0]
        self.assertEqual(set(item), {'input', 'target', 'index'})
        self.assertNotIn('clean_target', item)

    def test_manifest_binary_shape_and_identity(self) -> None:
        data = generate_synthetic_binary_2d(20, 1, split='train')
        manifest = generate_binary_asymmetric_rcn(data.labels, data.global_indices, rho_positive=0.2, rho_negative=0.1, seed=2)
        validate_binary_rcn_manifest(manifest)
        manifest.transition_matrix = np.eye(3)
        with self.assertRaisesRegex(ValueError, '\\[2, 2\\]'):
            validate_binary_rcn_manifest(manifest)
        three_class = NoiseManifest(dataset='fixture', noise_type='binary_asymmetric_rcn', seed=1, requested_rate=0.1, clean_targets=np.array([0, 1, 2]), noisy_targets=np.array([0, 1, 2]), transition_matrix=np.eye(3), metadata={'rho_positive': 0.2, 'rho_negative': 0.1, 'label_convention': 'zero_one'}, num_classes=3)
        with self.assertRaisesRegex(ValueError, 'num_classes'):
            validate_binary_rcn_manifest(three_class)

    def test_three_class_snapshot_is_rejected(self) -> None:
        snapshot = PosteriorSnapshot(np.full((3, 3), 1 / 3), np.array([0, 1, 2]), np.arange(3), 'fixture', 'train')
        with self.assertRaisesRegex(ValueError, '\\[N, 2\\]'):
            validate_binary_posterior_snapshot(snapshot)

# --- merged from test_importance_reweighting_method.py ---
class _importance_reweighting_method_EstimationAndWeightLookupTest(unittest.TestCase):

    def setUp(self) -> None:
        self.snapshot = PosteriorSnapshot(noisy_probabilities=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]), noisy_targets=np.array([0, 1, 0]), global_indices=np.array([10, 30, 20]), dataset='synthetic_binary_2d', split='train')
        self.rates = PaperRawMinNoiseRateEstimator().estimate(self.snapshot)

    def test_raw_min_uses_paper_direction(self) -> None:
        self.assertAlmostEqual(self.rates.rho_positive, 0.3)
        self.assertAlmostEqual(self.rates.rho_negative, 0.2)
        self.assertEqual(self.rates.positive_extreme_global_index, 30)
        self.assertEqual(self.rates.negative_extreme_global_index, 10)

    def test_lookup_uses_stable_index_and_detached_posterior(self) -> None:
        provider = IndexedBinaryRCNWeightProvider(self.snapshot, self.rates)
        result = provider.compute(SupervisedWeightInput(logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), noisy_targets=torch.tensor([0, 1]), sample_indices=torch.tensor([20, 30]), per_sample_loss=torch.ones(2)))
        self.assertEqual(result.sample_weights.shape, (2,))
        self.assertIs(result.sample_weights.requires_grad, False)
        self.assertTrue(torch.isfinite(result.sample_weights).all())

    def test_lookup_rejects_bad_logits_missing_duplicate_and_misalignment(self) -> None:
        provider = IndexedBinaryRCNWeightProvider(self.snapshot, self.rates)
        base = dict(logits=torch.zeros(2, 2), noisy_targets=torch.tensor([0, 1]), sample_indices=torch.tensor([10, 30]), per_sample_loss=torch.ones(2))
        for shape in ((2, 1), (2, 3)):
            values = dict(base)
            values['logits'] = torch.zeros(shape)
            with self.assertRaisesRegex(ValueError, '\\[B, 2\\]'):
                provider.compute(SupervisedWeightInput(**values))
        values = dict(base)
        values['sample_indices'] = torch.tensor([10, 99])
        with self.assertRaisesRegex(ValueError, 'missing'):
            provider.compute(SupervisedWeightInput(**values))
        values['sample_indices'] = torch.tensor([10, 10])
        with self.assertRaisesRegex(ValueError, 'unique'):
            provider.compute(SupervisedWeightInput(**values))
        values['sample_indices'] = torch.tensor([10, 30])
        values['noisy_targets'] = torch.tensor([1, 1])
        with self.assertRaisesRegex(ValueError, 'align'):
            provider.compute(SupervisedWeightInput(**values))

    def test_rate_artifact_rejects_bad_sum_and_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, 'sum'):
            NoiseRateArtifact(0.6, 0.4, 1, 2, 'a' * 64, 'synthetic_binary_2d', 'train')
        wrong = NoiseRateArtifact(0.2, 0.1, 1, 2, 'b' * 64, 'synthetic_binary_2d', 'train')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            IndexedBinaryRCNWeightProvider(self.snapshot, wrong)

    def test_kde_snapshot_is_binary_and_index_aligned(self) -> None:
        data = generate_synthetic_binary_2d(30, 7, split='train')
        snapshot = KDEBinaryNoisyPosteriorEstimator(0.2).fit_predict(data.features, data.labels, data.global_indices, dataset=data.dataset, split='train')
        self.assertEqual(snapshot.noisy_probabilities.shape, (30, 2))
        np.testing.assert_array_equal(snapshot.global_indices, np.sort(data.global_indices))

# --- merged from test_importance_reweighting_method.py ---
class _importance_reweighting_method_StateAndSmokeTest(unittest.TestCase):

    @staticmethod
    def _assert_no_pending_artifacts(run_dir: Path) -> None:
        pending = list(run_dir.glob('.*.pending.npz'))
        if pending:
            raise AssertionError(f'temporary artifacts were not cleaned: {pending}')

    def test_method_forces_paper_batch_mean_reduction(self) -> None:
        self.assertEqual(ImportanceReweightingAlgorithm.FINAL_REDUCTION.normalization, 'batch_mean')
        self.assertNotEqual(ImportanceReweightingAlgorithm.FINAL_REDUCTION.normalization, 'weight_sum_mean')

    def test_phase_transitions_and_extension(self) -> None:
        state = ImportanceReweightingState()
        with self.assertRaisesRegex(ValueError, 'illegal'):
            state.advance(ImportanceReweightingPhase.RATE_READY)
        state.posterior_snapshot_hash = 'a' * 64
        state.advance(ImportanceReweightingPhase.POSTERIOR_READY)
        state.noise_rate_artifact_hash = 'b' * 64
        state.advance(ImportanceReweightingPhase.RATE_READY)
        state.advance(ImportanceReweightingPhase.FINAL_TRAINING)
        state.final_completed_epochs = 2
        state.best_final_epoch = 0
        state.advance(ImportanceReweightingPhase.COMPLETED)
        state.reopen_final_training(3)
        self.assertEqual(state.phase, ImportanceReweightingPhase.FINAL_TRAINING)

    def test_tiny_smoke_and_resume_preserve_artifacts(self) -> None:
        config = _importance_reweighting_method_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            result = run_experiment(config, run_dir)
            snapshot = result / 'posterior_snapshot.npz'
            rates = result / 'noise_rate_artifact.npz'
            before_snapshot = snapshot.read_bytes()
            before_rates = rates.read_bytes()
            first = read_checkpoint(result / 'last.pt', 'cpu')
            self.assertEqual(first['method_state']['final_completed_epochs'], 2)
            self.assertEqual(first['method_state']['final_global_step'], 8)
            self.assertEqual(first['method_state']['phase'], 'completed')
            self.assertEqual(read_checkpoint(result / 'last.pt', 'cpu')['config']['num_classes'], 2)
            config['trainer']['epochs'] = 3
            run_experiment(config, resume=result / 'last.pt')
            final = read_checkpoint(result / 'last.pt', 'cpu')
            self.assertEqual(final['method_state']['final_completed_epochs'], 3)
            self.assertEqual(final['method_state']['final_global_step'], 12)
            self.assertEqual(snapshot.read_bytes(), before_snapshot)
            self.assertEqual(rates.read_bytes(), before_rates)
            self.assertEqual(final['posterior_snapshot_hash'], first['posterior_snapshot_hash'])
            self.assertEqual(final['noise_rate_artifact_hash'], first['noise_rate_artifact_hash'])
            self.assertEqual(final['config']['trainer']['epochs'], 3)
            metrics = yaml.safe_load((result / 'final_metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(metrics['reduction'], 'batch_mean')
            self.assertTrue(np.isfinite(metrics['test_loss']))

    def test_resume_rejects_class_and_convention_corruption(self) -> None:
        config = _importance_reweighting_method_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            run_experiment(config, run_dir)
            original = read_checkpoint(run_dir / 'last.pt', 'cpu')
            for field, value, pattern in (('num_classes', 3, 'num_classes'), ('label_convention', 'minus_plus', 'convention')):
                corrupted = deepcopy(original)
                corrupted[field] = value
                path = run_dir / f'bad-{field}.pt'
                torch.save(corrupted, path)
                with self.assertRaisesRegex(ValueError, pattern):
                    run_experiment(config, resume=path)

    def test_resume_rejects_rate_artifact_provenance_mismatch(self) -> None:
        config = _importance_reweighting_method_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            run_experiment(config, run_dir)
            payload = read_checkpoint(run_dir / 'last.pt', 'cpu')
            original = NoiseRateArtifact.load(run_dir / 'noise_rate_artifact.npz')
            wrong = NoiseRateArtifact(rho_positive=original.rho_positive, rho_negative=original.rho_negative, positive_extreme_global_index=original.positive_extreme_global_index, negative_extreme_global_index=original.negative_extreme_global_index, source_snapshot_hash='f' * 64, dataset=original.dataset, split=original.split)
            wrong.save(run_dir / 'noise_rate_artifact.npz')
            payload['noise_rate_artifact_hash'] = wrong.artifact_hash
            payload['method_state']['noise_rate_artifact_hash'] = wrong.artifact_hash
            bad = run_dir / 'bad-provenance.pt'
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, 'source snapshot'):
                run_experiment(config, resume=bad)

    def test_posterior_temporary_write_failure_preserves_formal_path(self) -> None:
        config = _importance_reweighting_method_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            run_dir.mkdir()
            formal = run_dir / 'posterior_snapshot.npz'
            formal.write_bytes(b'previous-valid-placeholder')
            with mock.patch.object(PosteriorSnapshot, 'save', side_effect=OSError('simulated posterior write failure')):
                with self.assertRaisesRegex(OSError, 'write failure'):
                    run_experiment(config, run_dir)
            self.assertEqual(formal.read_bytes(), b'previous-valid-placeholder')
            self.assertFalse((run_dir / 'last.pt').exists())
            self._assert_no_pending_artifacts(run_dir)

    def test_posterior_temporary_reload_failure_preserves_formal_path(self) -> None:
        config = _importance_reweighting_method_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            run_dir.mkdir()
            formal = run_dir / 'posterior_snapshot.npz'
            formal.write_bytes(b'previous-valid-placeholder')
            with mock.patch.object(PosteriorSnapshot, 'load', side_effect=ValueError('simulated posterior validation failure')):
                with self.assertRaisesRegex(ValueError, 'validation failure'):
                    run_experiment(config, run_dir)
            self.assertEqual(formal.read_bytes(), b'previous-valid-placeholder')
            self.assertFalse((run_dir / 'last.pt').exists())
            self._assert_no_pending_artifacts(run_dir)

    def test_rate_temporary_write_failure_keeps_posterior_ready_checkpoint(self) -> None:
        config = _importance_reweighting_method_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            run_dir.mkdir()
            formal = run_dir / 'noise_rate_artifact.npz'
            formal.write_bytes(b'previous-valid-placeholder')
            with mock.patch.object(NoiseRateArtifact, 'save', side_effect=OSError('simulated rate write failure')):
                with self.assertRaisesRegex(OSError, 'write failure'):
                    run_experiment(config, run_dir)
            checkpoint = read_checkpoint(run_dir / 'last.pt', 'cpu')
            self.assertEqual(checkpoint['method_state']['phase'], 'posterior_ready')
            self.assertEqual(checkpoint['method_state']['noise_rate_artifact_hash'], '')
            self.assertEqual(formal.read_bytes(), b'previous-valid-placeholder')
            self._assert_no_pending_artifacts(run_dir)

    def test_rate_temporary_reload_failure_keeps_posterior_ready_checkpoint(self) -> None:
        config = _importance_reweighting_method_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            run_dir.mkdir()
            formal = run_dir / 'noise_rate_artifact.npz'
            formal.write_bytes(b'previous-valid-placeholder')
            with mock.patch.object(NoiseRateArtifact, 'load', side_effect=ValueError('simulated rate validation failure')):
                with self.assertRaisesRegex(ValueError, 'validation failure'):
                    run_experiment(config, run_dir)
            checkpoint = read_checkpoint(run_dir / 'last.pt', 'cpu')
            self.assertEqual(checkpoint['method_state']['phase'], 'posterior_ready')
            self.assertEqual(checkpoint['method_state']['noise_rate_artifact_hash'], '')
            self.assertEqual(formal.read_bytes(), b'previous-valid-placeholder')
            self._assert_no_pending_artifacts(run_dir)

    def test_checkpoint_is_written_only_after_formal_artifact_is_valid(self) -> None:
        config = _importance_reweighting_method_load_config()
        observations: list[tuple[str, bool]] = []
        original = ImportanceReweightingAlgorithm._save_last

        def record_then_save(owner: ImportanceReweightingAlgorithm) -> None:
            phase = owner.state.phase.value
            if phase == 'posterior_ready':
                loaded = PosteriorSnapshot.load(owner.snapshot_path)
                valid = loaded.snapshot_hash == owner.state.posterior_snapshot_hash
            elif phase == 'rate_ready':
                loaded_rate = NoiseRateArtifact.load(owner.rate_path)
                valid = loaded_rate.artifact_hash == owner.state.noise_rate_artifact_hash
            else:
                valid = True
            observations.append((phase, valid))
            original(owner)
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(ImportanceReweightingAlgorithm, '_save_last', record_then_save):
                run_experiment(config, Path(temporary) / 'run')
        self.assertIn(('posterior_ready', True), observations)
        self.assertIn(('rate_ready', True), observations)

    def test_resume_rejects_corrupted_formal_snapshot_without_rebuilding(self) -> None:
        config = _importance_reweighting_method_load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'run'
            run_experiment(config, run_dir)
            snapshot = run_dir / 'posterior_snapshot.npz'
            snapshot.write_bytes(b'corrupted')
            corrupted = snapshot.read_bytes()
            with self.assertRaises(ValueError):
                run_experiment(config, resume=run_dir / 'last.pt')
            self.assertEqual(snapshot.read_bytes(), corrupted)

# --- merged from test_importance_reweighting_uci.py ---
from copy import deepcopy

# --- merged from test_importance_reweighting_uci.py ---
import hashlib

# --- merged from test_importance_reweighting_uci.py ---
import json

# --- merged from test_importance_reweighting_uci.py ---
from pathlib import Path

# --- merged from test_importance_reweighting_uci.py ---
import tempfile

# --- merged from test_importance_reweighting_uci.py ---
import unittest

# --- merged from test_importance_reweighting_uci.py ---
import numpy as np

# --- merged from test_importance_reweighting_uci.py ---
import yaml

# --- merged from test_importance_reweighting_uci.py ---
from lnl_toolbox.data.binary_benchmarks import BinaryBenchmarkTensorDataset

# --- merged from test_importance_reweighting_uci.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_importance_reweighting_uci.py ---
_importance_reweighting_uci_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_importance_reweighting_uci.py ---
_importance_reweighting_uci_CONFIG = _importance_reweighting_uci_ROOT / 'configs/reproduction/uci_heart_importance_reweighting.yaml'

# --- merged from test_importance_reweighting_uci.py ---
def _importance_reweighting_uci__heart_fixture(path: Path, samples: int=60) -> str:
    rows: list[str] = []
    for index in range(samples):
        label = 1 if index % 2 == 0 else 2
        center = -1.0 if label == 1 else 1.0
        features = [center + 0.03 * ((index + column) % 7) for column in range(13)]
        rows.append(' '.join([*(f'{value:.6f}' for value in features), str(label)]))
    payload = ('\n'.join(rows) + '\n').encode('utf-8')
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

# --- merged from test_importance_reweighting_uci.py ---
def _importance_reweighting_uci__config(source: Path, sha256: str, epochs: int) -> dict:
    config = yaml.safe_load(_importance_reweighting_uci_CONFIG.read_text(encoding='utf-8'))
    config['data']['path'] = str(source)
    config['data']['sha256'] = sha256
    config['data']['expected_samples'] = 60
    config['posterior_stage'].update({'max_centers': 8, 'max_iterations': 20, 'learning_rate': 0.01})
    config['loader']['batch_size'] = 12
    config['trainer'].update({'epochs': epochs, 'device': 'cpu'})
    return config

# --- merged from test_importance_reweighting_uci.py ---
class _importance_reweighting_uci_ImportanceReweightingUCIAdapterTest(unittest.TestCase):

    def test_training_view_never_exposes_clean_target(self) -> None:
        from lnl_toolbox.data.binary_benchmarks import BinaryBenchmark
        benchmark = BinaryBenchmark(np.zeros((4, 13), dtype=np.float32), np.asarray([0, 1, 0, 1]), 'fixture', global_indices=np.asarray([5, 7, 11, 13]))
        item = BinaryBenchmarkTensorDataset(benchmark, np.asarray([1, 1, 0, 0]))[0]
        self.assertEqual(set(item), {'input', 'target', 'index'})

    def test_real_adapter_resume_preserves_data_and_method_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'heart.dat'
            sha256 = _importance_reweighting_uci__heart_fixture(source)
            run_dir = root / 'run'
            run_experiment(_importance_reweighting_uci__config(source, sha256, 2), output_dir=run_dir)
            artifact_names = ('noise_manifest.npz', 'posterior_snapshot.npz', 'noise_rate_artifact.npz', 'preprocessing_state.json', 'split_manifest.json')
            before = {name: (hashlib.sha256((run_dir / name).read_bytes()).hexdigest(), (run_dir / name).stat().st_mtime_ns) for name in artifact_names}
            run_experiment(_importance_reweighting_uci__config(source, sha256, 3), resume=run_dir / 'last.pt')
            after = {name: (hashlib.sha256((run_dir / name).read_bytes()).hexdigest(), (run_dir / name).stat().st_mtime_ns) for name in artifact_names}
            self.assertEqual(before, after)
            final = json.loads((run_dir / 'final_metrics.json').read_text())
            self.assertEqual(final['completed_epochs'], 3)
            self.assertEqual(final['posterior']['value_count'], 72)
            self.assertEqual(final['posterior']['finite_count'], 72)
            self.assertLess(final['posterior']['row_sum_max_error'], 1e-10)
            self.assertEqual(final['weights']['negative_count'], 0)
            self.assertEqual(final['weights']['nonfinite_count'], 0)
            self.assertGreater(final['weights']['ess'], 0.0)
            self.assertFalse(final['optimization']['optimizer_stall'])

    def test_raw_hash_and_resume_split_identity_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'heart.dat'
            sha256 = _importance_reweighting_uci__heart_fixture(source)
            bad = _importance_reweighting_uci__config(source, '0' * 64, 1)
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                run_experiment(bad, output_dir=root / 'bad')
            run_dir = root / 'run'
            run_experiment(_importance_reweighting_uci__config(source, sha256, 1), output_dir=run_dir)
            changed = _importance_reweighting_uci__config(source, sha256, 2)
            changed['data']['split']['seed'] += 1
            with self.assertRaisesRegex(ValueError, 'identity mismatch|configuration mismatch|noise manifest|data_manifest'):
                run_experiment(changed, resume=run_dir / 'last.pt')
