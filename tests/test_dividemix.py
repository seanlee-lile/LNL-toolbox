from __future__ import annotations
from copy import deepcopy
import unittest
import numpy as np
import torch
from torch import nn
from unittest.mock import patch
from lnl_toolbox.algorithms.dividemix import DivideMixAlgorithm, DivideMixConfig, DivideMixPhase, DivideMixState, append_loss_history, build_co_divide, history_input
from lnl_toolbox.estimators import ReliabilityResult

def _dividemix_algorithm_config():
    return {'method': 'dividemix', 'execution': {'runner': 'dividemix'}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'validation_targets': 'noisy'}, 'evaluation': {'selection_split': 'validation'}, 'dividemix': {'fidelity': 'official_cifar_v1', 'warmup': {'epochs': 1}, 'gmm': {'threshold': 0.5, 'loss_history': {'name': 'official_auto', 'window_epochs': 5}}, 'mixmatch': {'augmentations': 2, 'temperature': 0.5, 'mixup_alpha': 4.0, 'mixup_lambda_scope': 'minibatch'}, 'objective': {'lambda_u': 25.0, 'lambda_r': 1.0, 'rampup_epochs': 16}, 'training': {'epochs': 2}, 'inference': {'ensemble': 'official_logits_sum'}}}

class _dividemix_algorithm_DivideMixAlgorithmTest(unittest.TestCase):

    def make_algorithm(self):
        torch.manual_seed(1)
        a = nn.Linear(2, 2)
        torch.manual_seed(2)
        b = nn.Linear(2, 2)
        oa, ob = (torch.optim.SGD(a.parameters(), 0.1), torch.optim.SGD(b.parameters(), 0.1))
        return DivideMixAlgorithm(model_a=a, model_b=b, optimizer_a=oa, optimizer_b=ob, scheduler_a=None, scheduler_b=None, config=DivideMixConfig.from_mapping(_dividemix_algorithm_config()), device=torch.device('cpu'))

    def test_optimizer_ownership_and_peer_isolation(self):
        algorithm = self.make_algorithm()
        before_b = deepcopy(algorithm.model_b.state_dict())
        views_x = (torch.randn(2, 2), torch.randn(2, 2))
        views_u = (torch.randn(2, 2), torch.randn(2, 2))
        algorithm.train_peer_step('a', views_x, views_u, torch.tensor([0, 1]), torch.tensor([0.8, 0.7]), epoch=0, batch_index=0, num_batches=1, rng=np.random.default_rng(1))
        for name, value in before_b.items():
            self.assertTrue(torch.equal(value, algorithm.model_b.state_dict()[name]))
        self.assertEqual(algorithm.state.optimizer_steps_a, 1)
        self.assertEqual(algorithm.state.optimizer_steps_b, 0)

    def test_phase_machine_rejects_illegal_transition(self):
        state = DivideMixState()
        with self.assertRaisesRegex(ValueError, 'illegal'):
            state.transition(DivideMixPhase.TRAIN_NETWORK_A)

    def test_config_rejects_self_pipeline_composition(self):
        value = _dividemix_algorithm_config()
        value['selector'] = {'name': 'small_loss'}
        with self.assertRaisesRegex(ValueError, 'owns its complete pipeline'):
            DivideMixConfig.from_mapping(value)

    def test_history_average_aligns_by_stable_index(self):
        history = []
        append_loss_history(history, torch.tensor([20, 10]), torch.tensor([4.0, 2.0]), 5)
        append_loss_history(history, torch.tensor([10, 20]), torch.tensor([2.0, 6.0]), 5)
        averaged = history_input(history, torch.tensor([20, 10]), use_average=True)
        self.assertTrue(torch.allclose(averaged, torch.tensor([1.0, 0.0], dtype=torch.float64)))

    def test_co_divide_uses_cross_not_self_probabilities(self):

        class Estimator:
            calls = 0

            def __init__(self, **_kwargs):
                pass

            def estimate(self, value):
                scores = torch.tensor([0.9, 0.1, 0.9, 0.1], dtype=torch.float64)
                if self.calls == 1:
                    scores = 1.0 - scores
                self.calls += 1
                return ReliabilityResult(value.sample_indices, scores, {'clean_component_mean': 0.2, 'noisy_component_mean': 0.8, 'gmm_iterations': 1.0})
        indices = torch.tensor([3, 8, 20, 44])
        history_a, history_b = ([], [])
        append_loss_history(history_a, indices, torch.tensor([1.0, 2.0, 3.0, 4.0]), 5)
        append_loss_history(history_b, indices, torch.tensor([4.0, 3.0, 2.0, 1.0]), 5)
        with patch('lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator', Estimator):
            result = build_co_divide(indices, history_a, history_b, DivideMixConfig.from_mapping(_dividemix_algorithm_config()), 0.2)
        self.assertTrue(torch.equal(result.labeled_for_a, torch.tensor([False, True, False, True])))
        self.assertTrue(torch.equal(result.labeled_for_b, torch.tensor([True, False, True, False])))

    def test_real_gmm_receives_history_average_without_rescaling(self):
        from sklearn.mixture import GaussianMixture
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        history = []
        append_loss_history(history, indices, torch.tensor([0., .1, .2, .3, .7, .8, .9, 1.]), 5)
        append_loss_history(history, indices, torch.tensor([.3, .2, .1, 0., 1., .9, .8, .7]), 5)
        config = DivideMixConfig.from_mapping(_dividemix_algorithm_config())
        original_fit = GaussianMixture.fit
        for rate in (.9, .2, None):
            captured = []
            def capture_fit(estimator, values, *args, **kwargs):
                captured.append(values.copy())
                return original_fit(estimator, values, *args, **kwargs)
            expected = history_input(history, indices, use_average=rate == .9)
            with self.subTest(rate=rate), patch.object(GaussianMixture, 'fit', capture_fit):
                result = build_co_divide(indices, history, history, config, rate)
            self.assertEqual(len(captured), 2)
            for values in captured:
                np.testing.assert_allclose(values[:, 0], expected.numpy(), atol=0, rtol=0)
            torch.testing.assert_close(result.normalized_loss_a, expected)
            self.assertEqual(result.metrics['loss_history_depth'], 2. if rate == .9 else 1.)
            self.assertEqual(result.labeled_for_a.tolist(), [True] * 4 + [False] * 4)

import math
import os
from pathlib import Path
import subprocess
import sys


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from lnl_toolbox.estimators import DivideMixGMMCleanProbabilityEstimator, DivideMixGMMLossInput, ReliabilityEstimator, validate_reliability_result

def _dividemix_gmm__separated_input(*, device: torch.device | str='cpu', requires_grad: bool=False) -> DivideMixGMMLossInput:
    return DivideMixGMMLossInput(per_sample_losses=torch.tensor([0.1, 0.13, 0.16, 1.8, 2.0, 2.2], device=device, requires_grad=requires_grad), sample_indices=torch.tensor([40, 10, 70, 20, 90, 30], device=device))

class _dividemix_gmm_DivideMixGMMTest(unittest.TestCase):

    def test_estimator_import_is_safe_without_sklearn_until_estimate(self):
        repository_root = Path(__file__).resolve().parents[1]
        script = '\nimport builtins\n\nreal_import = builtins.__import__\n\ndef block_sklearn(name, globals=None, locals=None, fromlist=(), level=0):\n    if name == "sklearn" or name.startswith("sklearn."):\n        raise ModuleNotFoundError("blocked sklearn for isolated test")\n    return real_import(name, globals, locals, fromlist, level)\n\nbuiltins.__import__ = block_sklearn\n\nimport lnl_toolbox.estimators\nfrom lnl_toolbox.estimators import (\n    DivideMixGMMCleanProbabilityEstimator,\n    DivideMixGMMLossInput,\n    ReliabilityEstimator,\n    ReliabilityResult,\n    StatisticResult,\n)\nimport torch\n\nassert ReliabilityEstimator is not None\nassert ReliabilityResult is not None\nassert StatisticResult is not None\nestimator = DivideMixGMMCleanProbabilityEstimator()\nestimator_input = DivideMixGMMLossInput(\n    per_sample_losses=torch.tensor([0.1, 1.0]),\n    sample_indices=torch.tensor([0, 1]),\n)\ntry:\n    estimator.estimate(estimator_input)\nexcept ImportError as error:\n    assert type(error) is ImportError\n    message = str(error)\n    assert "DivideMix GMM" in message\n    assert "optional training dependency" in message\n    assert \'python -m pip install -e ".[train]"\' in message\nelse:\n    raise AssertionError("estimate unexpectedly succeeded without sklearn")\n\nprint("lazy-import-ok")\n'
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(repository_root / 'src')
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        completed = subprocess.run([sys.executable, '-c', script], cwd=repository_root, env=environment, check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, msg=f'stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}')
        self.assertEqual(completed.stdout.strip(), 'lazy-import-ok')
        self.assertNotIn('ModuleNotFoundError', completed.stderr)


    def test_separated_clusters_return_lower_loss_clean_probabilities(self):
        estimator = DivideMixGMMCleanProbabilityEstimator(random_seed=17)
        result = estimator.estimate(_dividemix_gmm__separated_input())
        self.assertIsInstance(estimator, ReliabilityEstimator)
        self.assertEqual(result.sample_indices.tolist(), [40, 10, 70, 20, 90, 30])
        self.assertGreater(float(result.scores[:3].min()), 0.9)
        self.assertLess(float(result.scores[3:].max()), 0.1)
        self.assertLess(result.metrics['clean_component_mean'], result.metrics['noisy_component_mean'])
        self.assertGreater(result.metrics['mean_separation'], 1e-06)
        for value in result.metrics.values():
            self.assertIs(type(value), float)
            self.assertTrue(math.isfinite(value))

    def test_output_is_float64_detached_finite_bounded_and_aligned(self):
        estimator_input = _dividemix_gmm__separated_input(requires_grad=True)
        result = DivideMixGMMCleanProbabilityEstimator().estimate(estimator_input)
        indices, scores = validate_reliability_result(result, expected_sample_indices=estimator_input.sample_indices)
        self.assertTrue(torch.equal(indices, estimator_input.sample_indices))
        self.assertEqual(scores.dtype, torch.float64)
        self.assertEqual(scores.device, estimator_input.per_sample_losses.device)
        self.assertIs(scores.requires_grad, False)
        self.assertTrue(bool(torch.isfinite(scores).all().item()))
        self.assertTrue(bool(((scores >= 0) & (scores <= 1)).all().item()))
        self.assertFalse(hasattr(result, 'selected_mask'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_output_returns_to_cuda_input_device_as_float64(self):
        estimator_input = _dividemix_gmm__separated_input(device='cuda')
        result = DivideMixGMMCleanProbabilityEstimator().estimate(estimator_input)
        self.assertEqual(result.scores.device.type, 'cuda')
        self.assertEqual(result.scores.dtype, torch.float64)
        self.assertTrue(torch.equal(result.sample_indices, estimator_input.sample_indices))

    def test_permutation_preserves_index_probability_mapping(self):
        estimator = DivideMixGMMCleanProbabilityEstimator(random_seed=23)
        original_input = _dividemix_gmm__separated_input()
        original = estimator.estimate(original_input)
        permutation = torch.tensor([4, 0, 5, 2, 1, 3])
        permuted_input = DivideMixGMMLossInput(per_sample_losses=original_input.per_sample_losses[permutation], sample_indices=original_input.sample_indices[permutation])
        permuted = estimator.estimate(permuted_input)
        original_by_index = {int(index): float(score) for index, score in zip(original.sample_indices.tolist(), original.scores.tolist())}
        for index, score in zip(permuted.sample_indices.tolist(), permuted.scores.tolist()):
            self.assertAlmostEqual(float(score), original_by_index[int(index)], places=12)
        self.assertTrue(torch.equal(permuted.sample_indices, permuted_input.sample_indices))

    def test_fixed_seed_is_reproducible(self):
        estimator_input = _dividemix_gmm__separated_input()
        first = DivideMixGMMCleanProbabilityEstimator(random_seed=101).estimate(estimator_input)
        second = DivideMixGMMCleanProbabilityEstimator(random_seed=101).estimate(estimator_input)
        self.assertTrue(torch.equal(first.scores, second.scores))
        self.assertEqual(first.metrics, second.metrics)

    def test_constant_and_single_sample_losses_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive range'):
            DivideMixGMMCleanProbabilityEstimator().estimate(DivideMixGMMLossInput(per_sample_losses=torch.ones(4), sample_indices=torch.arange(4)))
        with self.assertRaisesRegex(ValueError, 'at least two'):
            DivideMixGMMCleanProbabilityEstimator().estimate(DivideMixGMMLossInput(per_sample_losses=torch.tensor([0.2]), sample_indices=torch.tensor([5])))

    def test_pre_normalized_input_and_raw_input_have_distinct_fit_scales(self):
        from sklearn.mixture import GaussianMixture
        losses = torch.tensor([.1, .12, .15, .8, .85, .9], dtype=torch.float64)
        indices = torch.arange(6)
        original_fit = GaussianMixture.fit
        for already_normalized in (True, False):
            captured = []
            def capture_fit(estimator, values, *args, **kwargs):
                captured.append(values.copy())
                return original_fit(estimator, values, *args, **kwargs)
            with self.subTest(already_normalized=already_normalized), patch.object(GaussianMixture, 'fit', capture_fit):
                DivideMixGMMCleanProbabilityEstimator().estimate(
                    DivideMixGMMLossInput(losses, indices, already_normalized=already_normalized))
            expected = losses if already_normalized else (losses - losses.min()) / (losses.max() - losses.min())
            np.testing.assert_allclose(captured[0][:, 0], expected.numpy(), atol=0, rtol=0)

    def test_pre_normalized_input_rejects_out_of_range_and_non_boolean_flag(self):
        estimator = DivideMixGMMCleanProbabilityEstimator()
        for losses in (torch.tensor([-.1, .8]), torch.tensor([.1, 1.1])):
            with self.assertRaisesRegex(ValueError, 'within'):
                estimator.estimate(DivideMixGMMLossInput(losses, torch.arange(2), already_normalized=True))
        with self.assertRaisesRegex(TypeError, 'boolean'):
            estimator.estimate(DivideMixGMMLossInput(torch.tensor([.1, .8]), torch.arange(2), already_normalized='yes'))

    def test_invalid_loss_values_and_empty_input_are_rejected(self):
        cases = ((torch.tensor([]), torch.tensor([], dtype=torch.long), 'at least two'), (torch.tensor([0.1, float('nan')]), torch.tensor([0, 1]), 'finite'), (torch.tensor([0.1, float('inf')]), torch.tensor([0, 1]), 'finite'), (torch.tensor([1, 2]), torch.tensor([0, 1]), 'floating-point'))
        for losses, indices, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                DivideMixGMMCleanProbabilityEstimator().estimate(DivideMixGMMLossInput(losses, indices))

    def test_invalid_indices_are_rejected(self):
        cases = ((torch.tensor([0]), 'same one-dimensional shape'), (torch.tensor([0.0, 1.0]), 'integer dtype'), (torch.tensor([1, 1]), 'unique'))
        for indices, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                DivideMixGMMCleanProbabilityEstimator().estimate(DivideMixGMMLossInput(per_sample_losses=torch.tensor([0.1, 1.0]), sample_indices=indices))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_mismatched_input_devices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'same device'):
            DivideMixGMMCleanProbabilityEstimator().estimate(DivideMixGMMLossInput(per_sample_losses=torch.tensor([0.1, 1.0], device='cuda'), sample_indices=torch.tensor([0, 1])))

    def test_mean_separation_uses_explicit_tolerance(self):
        estimator = DivideMixGMMCleanProbabilityEstimator(minimum_mean_separation=1.0)
        with self.assertRaisesRegex(ValueError, 'sufficiently separated'):
            estimator.estimate(_dividemix_gmm__separated_input())

    def test_nonconverged_fit_fails_explicitly(self):
        estimator = DivideMixGMMCleanProbabilityEstimator(max_iter=1, tolerance=1e-15)
        with self.assertRaisesRegex(RuntimeError, 'did not converge'):
            estimator.estimate(_dividemix_gmm__separated_input())

    def test_constructor_rejects_invalid_configuration(self):
        cases = (({'random_seed': True}, TypeError, 'random_seed'), ({'max_iter': 0}, ValueError, 'positive'), ({'tolerance': 0.0}, ValueError, 'greater'), ({'covariance_regularization': -1.0}, ValueError, 'at least'), ({'minimum_mean_separation': float('nan')}, ValueError, 'finite'), ({'minimum_mean_separation': -1.0}, ValueError, 'at least'))
        for kwargs, error_type, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(error_type, message):
                DivideMixGMMCleanProbabilityEstimator(**kwargs)


from lnl_toolbox.algorithms.dividemix import dividemix_objective, mixmatch_mixup, unsupervised_weight

class _dividemix_mixmatch_DivideMixMixMatchTest(unittest.TestCase):

    def test_mixup_uses_one_scalar_and_shared_permutation(self):
        x = (torch.tensor([[0.0], [1.0]]), torch.tensor([[2.0], [3.0]]))
        u = (torch.tensor([[4.0], [5.0]]), torch.tensor([[6.0], [7.0]]))
        y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        q = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
        order = torch.arange(7, -1, -1)
        result = mixmatch_mixup(x, u, y, q, alpha=4.0, rng=np.random.default_rng(4), permutation=order)
        self.assertGreaterEqual(result.mix_lambda, 0.5)
        expected = result.mix_lambda * torch.cat(x + u) + (1 - result.mix_lambda) * torch.cat(x + u)[order]
        self.assertTrue(torch.allclose(result.inputs, expected))
        self.assertEqual(result.labeled_count, 4)

    def test_objective_matches_manual_terms_and_keeps_autograd(self):
        import numpy as np

        logits_x = torch.tensor([[1.0, 0.0]], dtype=torch.float64, requires_grad=True)
        logits_u = torch.tensor([[0.2, 0.7]], dtype=torch.float64, requires_grad=True)
        targets_x = torch.tensor([[0.75, 0.25]])
        targets_u = torch.tensor([[0.2, 0.8]])
        all_logits = torch.cat((logits_x, logits_u))
        objective, metrics = dividemix_objective(logits_x, targets_x, logits_u, targets_u, all_logits, lambda_u=2.0, lambda_r=1.0)
        objective.backward()
        x_targets = targets_x.numpy().astype(np.float64)
        u_targets = targets_u.numpy().astype(np.float64)

        def reference(values):
            exponentials = np.exp(values - values.max(axis=1, keepdims=True))
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
            supervised = -(x_targets * np.log(probabilities[:1])).sum(axis=1).mean()
            unsupervised = ((probabilities[1:] - u_targets) ** 2).mean()
            mean_prediction = probabilities.mean(axis=0)
            regularizer = (0.5 * np.log(0.5 / mean_prediction)).sum()
            return supervised, unsupervised, regularizer

        values = np.array([[1.0, 0.0], [0.2, 0.7]])
        terms = reference(values)
        expected = terms[0] + 2.0 * terms[1] + terms[2]
        self.assertAlmostEqual(objective.item(), expected, places=10)
        for name, value in zip(('loss_x', 'loss_u', 'loss_regularizer'), terms):
            self.assertAlmostEqual(metrics[name], value, places=10)
        gradient = np.empty_like(values)
        for index in np.ndindex(values.shape):
            plus, minus = values.copy(), values.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            a, b = reference(plus), reference(minus)
            gradient[index] = ((a[0] + 2 * a[1] + a[2]) - (b[0] + 2 * b[1] + b[2])) / 2e-5
        np.testing.assert_allclose(torch.cat((logits_x.grad, logits_u.grad)).numpy(),
                                   gradient, rtol=1e-6, atol=1e-8)

    def test_fractional_ramp_up(self):
        self.assertEqual(unsupervised_weight(25.0, 1.0, 1, 16), 0.0)
        self.assertAlmostEqual(unsupervised_weight(25.0, 9.0, 1, 16), 12.5)
        self.assertEqual(unsupervised_weight(25.0, 30.0, 1, 16), 25.0)


from lnl_toolbox.algorithms.dividemix import co_guess, co_refine, sharpen

class _dividemix_targets_FixedModel(nn.Module):

    def __init__(self, logits):
        super().__init__()
        self.register_buffer('values', torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs):
        return self.values.expand(inputs.shape[0], -1)

class _dividemix_targets_DivideMixTargetsTest(unittest.TestCase):

    def test_sharpen_is_normalized_and_detached(self):
        result = sharpen(torch.tensor([[0.25, 0.75]], requires_grad=True), 0.5)
        self.assertTrue(torch.allclose(result.sum(1), torch.ones(1)))
        self.assertFalse(result.requires_grad)
        self.assertGreater(float(result[0, 1]), 0.75)

    def test_co_refinement_has_paper_endpoints(self):
        model = _dividemix_targets_FixedModel([0.0, 2.0])
        inputs = (torch.zeros(2, 1), torch.ones(2, 1))
        targets = torch.tensor([0, 1])
        result = co_refine(model, inputs, targets, torch.tensor([1.0, 0.0]), 1.0)
        self.assertTrue(torch.equal(result[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(result[1], torch.softmax(torch.tensor([0.0, 2.0]), 0)))
        self.assertFalse(result.requires_grad)

    def test_co_guess_averages_both_peers_and_all_views(self):
        model_a = _dividemix_targets_FixedModel([3.0, 0.0])
        model_b = _dividemix_targets_FixedModel([0.0, 1.0])
        views = (torch.zeros(2, 1), torch.ones(2, 1))
        result = co_guess(model_a, model_b, views, 1.0)
        expected = (torch.softmax(torch.tensor([3.0, 0.0]), 0) + torch.softmax(torch.tensor([0.0, 1.0]), 0)) / 2
        self.assertTrue(torch.allclose(result[0], expected))
        self.assertTrue(torch.allclose(result.sum(1), torch.ones(2)))

import hashlib
import json
import tempfile
from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.catalog import load_recipe_config, recipe_by_id
from lnl_toolbox.training.experiment import run_experiment

def _dividemix_workflow__data(size, split):
    rng = np.random.default_rng(12 if split == 'train' else 13)
    images = rng.integers(0, 256, (size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % 10
    return CifarData(images, labels, tuple(map(str, range(10))), split, 'cifar10')

def _dividemix_workflow__data100(size, split):
    rng = np.random.default_rng(22 if split == 'train' else 23)
    images = rng.integers(0, 256, (size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % 100
    return CifarData(images, labels, tuple(map(str, range(100))), split, 'cifar100')

def _dividemix_workflow__config(epochs=1):
    return {'method': 'dividemix', 'seed': 4, 'data': {'name': 'cifar10', 'root': 'unused', 'validation_size': 10, 'max_train_samples': 20, 'max_validation_samples': 10, 'max_test_samples': 10, 'augment': False}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 8, 'validation_targets': 'noisy'}, 'model': {'name': 'tiny_cnn', 'width': 4}, 'optimizer': {'name': 'adam', 'lr': 0.001}, 'scheduler': {'name': 'none'}, 'loader': {'batch_size': 5, 'num_workers': 0, 'pin_memory': False}, 'evaluation': {'selection_split': 'validation', 'primary': 'ensemble_accuracy'}, 'dividemix': {'fidelity': 'official_cifar_v1', 'initialization': {'peer_seed_offset': 1}, 'warmup': {'epochs': 1, 'confidence_penalty_weight': 1.0}, 'gmm': {'threshold': 0.5, 'loss_history': {'name': 'official_auto', 'window_epochs': 5}}, 'mixmatch': {'augmentations': 2, 'temperature': 0.5, 'mixup_alpha': 4.0, 'mixup_lambda_scope': 'minibatch'}, 'objective': {'lambda_u': 25.0, 'lambda_r': 1.0, 'rampup_epochs': 16}, 'training': {'epochs': epochs}, 'inference': {'ensemble': 'official_logits_sum'}}, 'trainer': {'device': 'cpu'}, 'execution': {'runner': 'dividemix'}}

class _dividemix_workflow__FakeEstimator:

    def __init__(self, **_kwargs):
        pass

    def estimate(self, value):
        count = value.sample_indices.numel()
        scores = torch.where(torch.arange(count) % 2 == 0, 0.9, 0.1).to(torch.float64)
        return ReliabilityResult(value.sample_indices.detach(), scores, {'gmm_iterations': 1.0, 'clean_component_mean': 0.25, 'noisy_component_mean': 0.75})

def _dividemix_workflow__hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class _dividemix_workflow_DivideMixWorkflowTest(unittest.TestCase):

    def setUp(self):
        self.train, self.test = (_dividemix_workflow__data(40, 'train'), _dividemix_workflow__data(20, 'test'))

    def load(self, _root, split):
        return self.train if split == 'train' else self.test


    def test_author_style_network_matches_explicit_forward_and_has_gradients(self):
        from lnl_toolbox.models.cifar_resnet import preact_resnet18
        import torch.nn.functional as F
        config = load_recipe_config(recipe_by_id('cifar10-dividemix-sym20'))
        model = preact_resnet18(base_width=4, variant=config['model']['variant']).eval()
        self.assertIsInstance(model.stem, nn.Sequential)
        self.assertIsInstance(model.final_bn, nn.Identity)
        inputs = torch.randn(2, 3, 32, 32)
        actual = model(inputs)
        output = F.relu(model.stem[1](model.stem[0](inputs)))
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
            self.assertEqual(len(layer), 2)
            for block in layer:
                activated = F.relu(block.bn1(output))
                shortcut = activated if block.shortcut is None else block.shortcut(activated)
                output = block.conv2(F.relu(block.bn2(block.conv1(activated)))) + shortcut
        expected = model.classifier(F.avg_pool2d(output, 4).flatten(1))
        # Fixed and adaptive global pooling can sum float32 values in a
        # different order; compare numerical agreement, not bitwise identity.
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        actual.square().sum().backward()
        self.assertGreater(float(model.stem[0].weight.grad.abs().sum()), 0.)

    def test_prepared_test_images_use_author_normalization(self):
        from lnl_toolbox.data import DataRole
        from lnl_toolbox.training.data_service import prepare_experiment_data
        from lnl_toolbox.training.runners import resolve_data_requirements
        config = _dividemix_workflow__config()
        normalization = load_recipe_config(recipe_by_id('cifar10-dividemix-sym20'))['data']['normalization']
        config['data']['normalization'] = normalization
        self.test.images.fill(0)
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self.load):
            prepared = prepare_experiment_data(config, requirements=resolve_data_requirements(config, expected_runner='dividemix'),
                                               run_dir=Path(directory), seed=config['seed'])
            actual = next(iter(prepared.loader(DataRole.TEST, shuffle=False)))['input']
        expected = (-torch.tensor(normalization['mean']) / torch.tensor(normalization['std'])).view(1, 3, 1, 1)
        torch.testing.assert_close(actual, expected.expand_as(actual))

    def test_fresh_extension_and_completed_noop(self):
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self.load), patch('lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator', _dividemix_workflow__FakeEstimator):
            run_dir = run_experiment(_dividemix_workflow__config(1), Path(directory) / 'run')
            payload = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(payload['algorithm']['dividemix_state']['phase'], 'completed')
            self.assertEqual(payload['algorithm']['dividemix_state']['main_completed_epochs'], 1)
            self.assertEqual(set(payload['algorithm']['model']), {'a', 'b'})
            first_artifact = run_dir / 'dividemix_epoch_0001.npz'
            first_hash, first_mtime = (_dividemix_workflow__hash(first_artifact), first_artifact.stat().st_mtime_ns)
            run_experiment(_dividemix_workflow__config(2), resume=run_dir / 'last.pt')
            resumed = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(resumed['algorithm']['dividemix_state']['main_completed_epochs'], 2)
            self.assertEqual(_dividemix_workflow__hash(first_artifact), first_hash)
            self.assertEqual(first_artifact.stat().st_mtime_ns, first_mtime)
            checkpoint_hash, metrics_hash = (_dividemix_workflow__hash(run_dir / 'last.pt'), _dividemix_workflow__hash(run_dir / 'metrics.jsonl'))
            run_experiment(_dividemix_workflow__config(2), resume=run_dir / 'last.pt')
            self.assertEqual(_dividemix_workflow__hash(run_dir / 'last.pt'), checkpoint_hash)
            self.assertEqual(_dividemix_workflow__hash(run_dir / 'metrics.jsonl'), metrics_hash)
            final = json.loads((run_dir / 'final_metrics.json').read_text())
            self.assertEqual(final['ensemble'], 'official_logits_sum')

    def test_corrupt_ready_artifact_fails_instead_of_refitting(self):
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self.load), patch('lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator', _dividemix_workflow__FakeEstimator):
            run_dir = run_experiment(_dividemix_workflow__config(1), Path(directory) / 'run')
            payload = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            payload['algorithm']['dividemix_state']['phase'] = 'co_divide_ready'
            atomic = run_dir / 'last.pt'
            torch.save(payload, atomic)
            (run_dir / payload['algorithm']['dividemix_state']['current_artifact']).write_bytes(b'broken')
            with self.assertRaises(Exception):
                run_experiment(_dividemix_workflow__config(2), resume=atomic)

    def test_network_a_ready_resume_does_not_repeat_a(self):
        import lnl_toolbox.training.dividemix_experiment as workflow
        original = workflow._train_peer_epoch
        failed = {'value': False}

        def interrupt_b(*args, **kwargs):
            peer = args[1]
            if peer == 'b' and (not failed['value']):
                failed['value'] = True
                raise RuntimeError('controlled B interruption')
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'run'
            with patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self.load), patch('lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator', _dividemix_workflow__FakeEstimator), patch('lnl_toolbox.training.dividemix_experiment._train_peer_epoch', side_effect=interrupt_b):
                with self.assertRaisesRegex(RuntimeError, 'controlled B'):
                    run_experiment(_dividemix_workflow__config(1), run_dir)
            interrupted = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(interrupted['algorithm']['dividemix_state']['phase'], 'network_a_ready')
            steps_a = interrupted['algorithm']['dividemix_state']['optimizer_steps_a']
            with patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self.load), patch('lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator', _dividemix_workflow__FakeEstimator):
                run_experiment(_dividemix_workflow__config(1), resume=run_dir / 'last.pt')
            resumed = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(resumed['algorithm']['dividemix_state']['optimizer_steps_a'], steps_a)

    def test_cifar100_lightweight_workflow(self):
        train, test = (_dividemix_workflow__data100(220, 'train'), _dividemix_workflow__data100(100, 'test'))
        value = _dividemix_workflow__config(1)
        value['data'].update({'name': 'cifar100', 'validation_size': 100, 'max_train_samples': 100, 'max_validation_samples': 100, 'max_test_samples': 100})
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar100', side_effect=lambda _root, split: train if split == 'train' else test), patch('lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator', _dividemix_workflow__FakeEstimator):
            run_dir = run_experiment(value, Path(directory) / 'run')
            final = json.loads((run_dir / 'final_metrics.json').read_text())
            self.assertEqual(final['completed_epochs'], 1)
