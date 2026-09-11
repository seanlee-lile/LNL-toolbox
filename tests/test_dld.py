"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_dld.py ---
import unittest

# --- merged from test_dld.py ---
from unittest.mock import patch

# --- merged from test_dld.py ---
import numpy as np

# --- merged from test_dld.py ---
import torch

# --- merged from test_dld.py ---
from lnl_toolbox.algorithms.dld import DLDAlgorithm, DLDLabelPredictor, DLDPreCorrectionArtifact, DirectionalDiffusionSchedule

# --- merged from test_dld.py ---
from lnl_toolbox.models.directional_diffusion import DirectionalDiffusion

# --- merged from test_dld.py ---
def _dld__algorithm_fixture() -> DLDAlgorithm:
    torch.manual_seed(29)
    direction = DLDLabelPredictor(2, 3, hidden_dim=8, time_dim=4)
    noise = DLDLabelPredictor(2, 3, hidden_dim=8, time_dim=4)
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    y0 = np.eye(2, dtype=np.float64)
    yn = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    artifact = DLDPreCorrectionArtifact(np.array([7, 3]), np.array([0, 1]), probabilities, probabilities, probabilities, np.zeros(2), np.array([0, 1]), y0, yn, yn - y0, np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), {'test': True})
    return DLDAlgorithm(direction_model=direction, noise_model=noise, direction_optimizer=torch.optim.Adam(direction.parameters(), lr=0.01), noise_optimizer=torch.optim.Adam(noise.parameters(), lr=0.01), direction_scheduler=None, noise_scheduler=None, schedule=DirectionalDiffusionSchedule.average(5), artifact=artifact, device='cpu', ema_decay=0.9)

# --- merged from test_dld.py ---
class _dld_DLDModelTest(unittest.TestCase):

    def test_read_only_telemetry_preserves_losses_and_parameter_updates(self) -> None:
        reference = _dld__algorithm_fixture()
        observed = _dld__algorithm_fixture()
        indices = torch.tensor([3, 7], dtype=torch.int64)
        torch.manual_seed(101)
        with patch.object(DLDAlgorithm, '_parameter_norm', return_value=0.0), patch.object(DLDAlgorithm, '_tensor_rms', return_value=0.0):
            expected = reference.train_step(indices)
        torch.manual_seed(101)
        actual = observed.train_step(indices)
        self.assertEqual(actual['direction_loss'], expected['direction_loss'])
        self.assertEqual(actual['noise_loss'], expected['noise_loss'])
        for expected_model, actual_model in ((reference.direction_model, observed.direction_model), (reference.noise_model, observed.noise_model)):
            for left, right in zip(expected_model.parameters(), actual_model.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        for name in ('direction_parameter_norm', 'noise_parameter_norm', 'predicted_direction_rms', 'predicted_noise_rms', 'target_direction_rms', 'target_noise_rms'):
            self.assertTrue(np.isfinite(actual[name]), name)

    def test_cosine_schedule_matches_official_state_shape(self) -> None:
        model = DirectionalDiffusion(3, 5, num_timesteps=32, hidden_width=16, time_dim=8, schedule='cosine', image_base_width=2)
        self.assertEqual(model.schedule, 'cosine')
        self.assertEqual(tuple(model.alpha_cumsum.shape), (32,))
        self.assertEqual(tuple(model.beta2_cumsum.shape), (32,))
        self.assertTrue(torch.isfinite(model.alpha_cumsum).all())
        self.assertTrue(torch.isfinite(model.beta_cumsum).all())
        self.assertGreater(float(model.alpha_cumsum[-1]), float(model.alpha_cumsum[0]))

    def test_residual_and_noise_paths_are_independent(self) -> None:
        torch.manual_seed(4)
        model = DirectionalDiffusion(3, 5, num_timesteps=12, hidden_width=16, time_dim=8)
        y_input = torch.zeros(4, 3)
        y0 = torch.softmax(torch.randn(4, 3), dim=1)
        features = torch.randn(4, 5)
        timesteps = torch.tensor([0, 1, 10, 11])
        predicted_residual, predicted_noise, residual, noise, y_t = model.forward_t(y_input, y0, features, timesteps, noise=torch.zeros_like(y0))
        self.assertEqual(predicted_residual.shape, (4, 3))
        self.assertEqual(predicted_noise.shape, (4, 3))
        self.assertTrue(torch.isfinite(y_t).all())
        residual_loss = (predicted_residual - residual).square().mean()
        residual_loss.backward()
        self.assertTrue(any((parameter.grad is not None for parameter in model.residual_model.parameters())))
        self.assertTrue(all((parameter.grad is None for parameter in model.noise_model.parameters())))

    def test_sampling_is_deterministic_and_normalized(self) -> None:
        torch.manual_seed(9)
        model = DirectionalDiffusion(4, 6, num_timesteps=16, hidden_width=16, time_dim=8)
        features = torch.randn(3, 6)
        first = model.sample(features, sampling_timesteps=5)
        second = model.sample(features, sampling_timesteps=5)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first.sum(dim=1), torch.ones(3), atol=1e-05, rtol=1e-05)
        self.assertTrue(torch.isfinite(first).all())

    def test_image_condition_is_trainable_and_used_by_sampling(self) -> None:
        torch.manual_seed(12)
        model = DirectionalDiffusion(3, 5, num_timesteps=12, hidden_width=16, time_dim=8, image_base_width=2)
        y_input = torch.zeros(4, 3)
        y0 = torch.softmax(torch.randn(4, 3), dim=1)
        features = torch.randn(4, 5)
        images = torch.randn(4, 3, 32, 32)
        timesteps = torch.tensor([0, 1, 10, 11])
        predicted_residual, _, residual, _, _ = model.forward_t(y_input, y0, features, timesteps, noise=torch.zeros_like(y0), images=images)
        (predicted_residual - residual).square().mean().backward()
        self.assertTrue(any((parameter.grad is not None for parameter in model.residual_image_encoder.parameters())))
        self.assertTrue(all((parameter.grad is None for parameter in model.noise_image_encoder.parameters())))
        model.eval()
        probabilities = model.sample(features, images=images, sampling_timesteps=4)
        self.assertEqual(tuple(probabilities.shape), (4, 3))
        self.assertTrue(torch.isfinite(probabilities).all())

# --- merged from test_dld_diffusion.py ---
import unittest

# --- merged from test_dld_diffusion.py ---
import numpy as np

# --- merged from test_dld_diffusion.py ---
import torch

# --- merged from test_dld_diffusion.py ---
from lnl_toolbox.algorithms.dld import DLDAlgorithm, DLDLabelPredictor, DLDPreCorrectionArtifact, DirectionalDiffusionSchedule, construct_direction, dld_objective, sample_forward_state

# --- merged from test_dld_diffusion.py ---
def _dld_diffusion__artifact() -> DLDPreCorrectionArtifact:
    pw = np.array([[0.8, 0.2], [0.2, 0.8]])
    y0 = np.eye(2)
    yn = np.array([[0.0, 0.0], [1.0, 0.0]])
    return DLDPreCorrectionArtifact(np.array([7, 3]), np.array([0, 1]), pw, pw, pw, np.array([0.0, 0.0]), np.array([0, 1]), y0, yn, yn - y0, np.array([[1.0, 2.0], [3.0, 4.0]]), {'test': True})

# --- merged from test_dld_diffusion.py ---
class _dld_diffusion_DLDDiffusionTest(unittest.TestCase):

    def test_schedule_endpoint_and_forward_equation(self) -> None:
        schedule = DirectionalDiffusionSchedule.average(4)
        self.assertAlmostEqual(float(schedule.alpha_bar[-1]), 1.0)
        self.assertAlmostEqual(float(schedule.beta_bar[-1]), 1.0)
        y0 = torch.tensor([[1.0, 0.0]])
        yn = torch.tensor([[0.0, 1.0]])
        epsilon = torch.tensor([[2.0, -2.0]])
        result = sample_forward_state(y0, construct_direction(y0, yn), torch.tensor([3]), epsilon, schedule)
        self.assertTrue(torch.allclose(result, yn + epsilon))

    def test_predictors_and_optimizers_are_independent_and_update(self) -> None:
        direction = DLDLabelPredictor(2, 2, hidden_dim=8, time_dim=4)
        noise = DLDLabelPredictor(2, 2, hidden_dim=8, time_dim=4)
        direction_optimizer = torch.optim.Adam(direction.parameters(), lr=0.01)
        noise_optimizer = torch.optim.Adam(noise.parameters(), lr=0.01)
        algorithm = DLDAlgorithm(direction_model=direction, noise_model=noise, direction_optimizer=direction_optimizer, noise_optimizer=noise_optimizer, direction_scheduler=None, noise_scheduler=None, schedule=DirectionalDiffusionSchedule.average(5), artifact=_dld_diffusion__artifact(), device='cpu', ema_decay=0.9)
        self.assertFalse({id(p) for p in direction.parameters()} & {id(p) for p in noise.parameters()})
        before_d = [p.detach().clone() for p in direction.parameters()]
        before_n = [p.detach().clone() for p in noise.parameters()]
        metrics = algorithm.train_step(torch.tensor([3, 7]))
        self.assertTrue(any((not torch.equal(a, b) for a, b in zip(before_d, direction.parameters()))))
        self.assertTrue(any((not torch.equal(a, b) for a, b in zip(before_n, noise.parameters()))))
        self.assertTrue(all((np.isfinite(value) for value in metrics.values())))

    def test_objective_rejects_nonfinite(self) -> None:
        good = torch.zeros(1, 2)
        with self.assertRaisesRegex(ValueError, 'finite'):
            dld_objective(good + float('nan'), good, good, good)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_predictor_cuda(self) -> None:
        model = DLDLabelPredictor(3, 4).cuda()
        result = model(torch.zeros(2, 3, device='cuda'), torch.zeros(2, 3, device='cuda'), torch.zeros(2, 4, device='cuda'), torch.tensor([0, 1], device='cuda'))
        self.assertEqual(tuple(result.shape), (2, 3))

# --- merged from test_dld_precorrection.py ---
from pathlib import Path

# --- merged from test_dld_precorrection.py ---
import tempfile

# --- merged from test_dld_precorrection.py ---
import unittest

# --- merged from test_dld_precorrection.py ---
import numpy as np

# --- merged from test_dld_precorrection.py ---
import torch

# --- merged from test_dld_precorrection.py ---
from lnl_toolbox.algorithms.dld import DLDPreCorrectionArtifact, PARTITION_CLEAN, PARTITION_HARD, PARTITION_NOISY, construct_y0, construct_yn, kl_ps_to_pw, partition_samples, persist_precorrection_atomically, weighted_neighbor_distribution

# --- merged from test_dld_precorrection.py ---
class _dld_precorrection_DLDPreCorrectionTest(unittest.TestCase):

    def test_cosine_similarity_knn_uses_stable_indices_and_normalizes(self) -> None:
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        targets = torch.tensor([0, 1, 1])
        indices = torch.tensor([50, 10, 90])
        result = weighted_neighbor_distribution(features, features, targets, indices, indices, num_classes=2, k=2, metric='cosine_similarity', delta=0.001, self_neighbor='include')
        self.assertTrue(torch.allclose(result.probabilities.sum(1), torch.ones(3)))
        self.assertEqual(result.neighbor_indices[0, 0].item(), 50)
        order = torch.tensor([2, 0, 1])
        permuted = weighted_neighbor_distribution(features[order], features[order], targets[order], indices[order], indices[order], num_classes=2, k=2, metric='cosine_similarity', delta=0.001, self_neighbor='include')
        by_index = {int(i): p for i, p in zip(indices, result.probabilities)}
        for index, probability in zip(indices[order], permuted.probabilities):
            self.assertTrue(torch.allclose(probability, by_index[int(index)]))

    def test_cosine_similarity_uses_inverse_distance_weights(self) -> None:
        features = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.4, 3.0 ** 0.5 * 0.4]], dtype=torch.float64)
        targets = torch.tensor([0, 1, 1, 0])
        indices = torch.tensor([30, 10, 20, 40])
        delta = 1e-06
        result = weighted_neighbor_distribution(features[:1], features, targets, indices[:1], indices, num_classes=2, k=3, metric='cosine_similarity', delta=delta, self_neighbor='include')
        expected_similarity = torch.tensor([[1.0, 0.8, 0.6]], dtype=torch.float64)
        expected_raw = 1.0 / (1.0 - expected_similarity + delta)
        expected_weights = expected_raw / expected_raw.sum(dim=1, keepdim=True)
        torch.testing.assert_close(result.neighbor_values, expected_similarity)
        torch.testing.assert_close(result.unnormalized_weights, expected_raw)
        torch.testing.assert_close(result.weights, expected_weights)
        torch.testing.assert_close(result.probabilities, torch.tensor([[expected_weights[0, 0], expected_weights[0, 1:].sum()]], dtype=torch.float64))
        self.assertGreater(
            float(result.unnormalized_weights[0, 0]),
            float(result.unnormalized_weights[0, 1]),
        )
        self.assertGreater(
            float(result.unnormalized_weights[0, 1]),
            float(result.unnormalized_weights[0, 2]),
        )

    def test_cosine_similarity_negative_values_are_valid_distances(self) -> None:
        query = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        references = torch.tensor([[-1.0, 0.0], [-0.5, 3.0 ** 0.5 / 2.0]], dtype=torch.float64)
        result = weighted_neighbor_distribution(query, references, torch.tensor([0, 1]), torch.tensor([10]), torch.tensor([20, 30]), num_classes=2, k=1, metric='cosine_similarity', delta=1e-06)
        self.assertTrue(torch.isfinite(result.weights).all())
        zeros = torch.zeros((2, 2), dtype=torch.float64)
        zero_result = weighted_neighbor_distribution(zeros[:1], zeros, torch.tensor([0, 1]), torch.tensor([10]), torch.tensor([20, 30]), num_classes=2, k=1, metric='cosine_similarity', delta=float.fromhex('0x0.0000000000001p-1022'))
        self.assertTrue(torch.isfinite(zero_result.weights).all())

    def test_exact_chunking_matches_dense_end_to_end(self) -> None:
        generator = torch.Generator().manual_seed(413)
        weak = torch.rand(23, 7, generator=generator, dtype=torch.float64)
        strong = weak + 0.02 * torch.randn(23, 7, generator=generator, dtype=torch.float64)
        targets = torch.arange(23, dtype=torch.int64) % 3
        indices = torch.randperm(23, generator=generator, dtype=torch.int64) + 100

        def estimate(features, chunk_size):
            return weighted_neighbor_distribution(features, features, targets, indices, indices, num_classes=3, k=6, metric='cosine_similarity', delta=1e-06, self_neighbor='include', query_chunk_size=chunk_size)
        dense_w, dense_s = (estimate(weak, None), estimate(strong, None))
        chunk_w, chunk_s = (estimate(weak, 4), estimate(strong, 4))
        for dense, chunked in ((dense_w, chunk_w), (dense_s, chunk_s)):
            self.assertTrue(torch.equal(dense.neighbor_indices, chunked.neighbor_indices))
            torch.testing.assert_close(dense.neighbor_values, chunked.neighbor_values, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(dense.unnormalized_weights, chunked.unnormalized_weights, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(dense.weights, chunked.weights, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(dense.probabilities, chunked.probabilities, rtol=1e-12, atol=1e-12)
        dense_partition = partition_samples(dense_w.probabilities, dense_s.probabilities, targets, random_state=7, minimum_mean_separation=0.0)
        chunk_partition = partition_samples(chunk_w.probabilities, chunk_s.probabilities, targets, random_state=7, minimum_mean_separation=0.0)
        self.assertTrue(torch.equal(dense_partition.partition, chunk_partition.partition))
        dense_y0 = construct_y0(dense_partition.p_ws, targets, dense_partition.partition)
        chunk_y0 = construct_y0(chunk_partition.p_ws, targets, chunk_partition.partition)
        dense_yn = construct_yn(dense_w.probabilities, dense_s.probabilities, targets, dense_partition.partition)
        chunk_yn = construct_yn(chunk_w.probabilities, chunk_s.probabilities, targets, chunk_partition.partition)
        torch.testing.assert_close(dense_y0, chunk_y0, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(dense_yn, chunk_yn, rtol=1e-12, atol=1e-12)

    def test_chunking_keeps_self_exclusion_and_rejects_invalid_size(self) -> None:
        features = torch.eye(4, dtype=torch.float64)
        targets = torch.tensor([0, 1, 0, 1])
        indices = torch.tensor([40, 10, 30, 20])
        dense = weighted_neighbor_distribution(features, features, targets, indices, indices, num_classes=2, k=2, metric='cosine_distance', delta=1e-06, self_neighbor='exclude')
        chunked = weighted_neighbor_distribution(features, features, targets, indices, indices, num_classes=2, k=2, metric='cosine_distance', delta=1e-06, self_neighbor='exclude', query_chunk_size=1)
        self.assertTrue(torch.equal(dense.neighbor_indices, chunked.neighbor_indices))
        for row, index in zip(chunked.neighbor_indices, indices):
            self.assertNotIn(int(index), row.tolist())
        with self.assertRaisesRegex(ValueError, 'query_chunk_size'):
            weighted_neighbor_distribution(features, features, targets, indices, indices, num_classes=2, k=2, metric='cosine_distance', delta=1e-06, query_chunk_size=0)

    def test_kl_direction_and_eq14_eq15(self) -> None:
        pw = torch.tensor([[0.8, 0.2], [0.25, 0.75], [0.6, 0.4]], dtype=torch.float64)
        ps = torch.tensor([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]], dtype=torch.float64)
        expected = (ps * (ps.log() - pw.log())).sum(1)
        self.assertTrue(torch.allclose(kl_ps_to_pw(pw, ps), expected))
        target = torch.tensor([0, 1, 0])
        partition = torch.tensor([PARTITION_CLEAN, PARTITION_NOISY, PARTITION_HARD])
        pws = (pw + ps) / 2
        y0 = construct_y0(pws, target, partition)
        yn = construct_yn(pw, ps, target, partition)
        self.assertTrue(torch.equal(y0[0], torch.tensor([1.0, 0.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(yn[0], torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(yn[1], torch.tensor([0.0, 1.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(yn[2], torch.tensor([0.5, 0.5], dtype=torch.float64)))
        self.assertTrue(torch.allclose(yn - y0, yn - y0))

    def test_hard_zero_denominator_fails(self) -> None:
        p = torch.tensor([[0.5, 0.5]])
        with self.assertRaisesRegex(ValueError, 'denominator'):
            construct_yn(p, p, torch.tensor([0]), torch.tensor([PARTITION_HARD]))

    def test_gmm_high_mean_component_is_hard(self) -> None:
        pw = torch.tensor([[0.99, 0.01]] * 8 + [[0.99, 0.01]] * 8, dtype=torch.float64)
        ps = torch.tensor([[0.98, 0.02]] * 8 + [[0.01, 0.99]] * 8, dtype=torch.float64)
        targets = torch.zeros(16, dtype=torch.int64)
        result = partition_samples(pw, ps, targets, random_state=0)
        self.assertLess(result.low_mean, result.high_mean)
        self.assertTrue(torch.equal(result.partition[8:], torch.full((8,), PARTITION_HARD)))

    def test_artifact_atomic_roundtrip_and_hash(self) -> None:
        indices = np.array([9, 2])
        targets = np.array([0, 1])
        pw = np.array([[0.8, 0.2], [0.3, 0.7]])
        ps = np.array([[0.7, 0.3], [0.4, 0.6]])
        pws = (pw + ps) / 2
        partition = np.array([PARTITION_CLEAN, PARTITION_NOISY])
        y0 = np.eye(2)
        yn = np.array([[0.0, 0.0], [0.0, 1.0]])
        artifact = DLDPreCorrectionArtifact(indices, targets, pw, ps, pws, np.array([0.1, 0.2]), partition, y0, yn, yn - y0, np.ones((2, 3)), {'source': 'test'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dld_precorrection.npz'
            loaded = persist_precorrection_atomically(artifact, path)
            self.assertEqual(loaded.artifact_hash, artifact.artifact_hash)
            self.assertEqual(loaded.global_indices.tolist(), [2, 9])

# --- merged from test_dld_sampling.py ---
import unittest

# --- merged from test_dld_sampling.py ---
import torch

# --- merged from test_dld_sampling.py ---
from torch import nn

# --- merged from test_dld_sampling.py ---
from lnl_toolbox.algorithms.dld import DirectionalDiffusionSchedule, accelerated_timesteps, sample_labels

# --- merged from test_dld_sampling.py ---
class _dld_sampling__Constant(nn.Module):

    def __init__(self, classes: int, value: float) -> None:
        super().__init__()
        self.num_classes = classes
        self.value = value

    def forward(self, y_t, y_n, features, timestep):
        return torch.full_like(y_t, self.value)

# --- merged from test_dld_sampling.py ---
class _dld_sampling_DLDSamplingTest(unittest.TestCase):

    def test_five_step_sequence_and_manual_reverse(self) -> None:
        sequence = accelerated_timesteps(20, 5)
        self.assertEqual(len(sequence), 5)
        self.assertEqual(sequence[-1][1], -1)
        features = torch.ones(2, 3)
        result = sample_labels(_dld_sampling__Constant(2, 1.0), _dld_sampling__Constant(2, 0.0), features, DirectionalDiffusionSchedule.average(20), inference_steps=5)
        self.assertTrue(torch.allclose(result, -torch.ones(2, 2), atol=1e-06))

    def test_sampling_is_deterministic_and_restores_train_mode(self) -> None:
        torch.manual_seed(4)
        direction = _dld_sampling__Constant(3, 0.25)
        noise = _dld_sampling__Constant(3, -0.25)
        features = torch.randn(4, 2)
        schedule = DirectionalDiffusionSchedule.average(10)
        first = sample_labels(direction, noise, features, schedule, inference_steps=5)
        second = sample_labels(direction, noise, features, schedule, inference_steps=5)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(direction.training)
        self.assertEqual(tuple(first.shape), (4, 3))

# --- merged from test_dld_workflow.py ---
from copy import deepcopy

# --- merged from test_dld_workflow.py ---
import json

# --- merged from test_dld_workflow.py ---
from pathlib import Path

# --- merged from test_dld_workflow.py ---
import tempfile

# --- merged from test_dld_workflow.py ---
import unittest

# --- merged from test_dld_workflow.py ---
from unittest.mock import patch

# --- merged from test_dld_workflow.py ---
import numpy as np

# --- merged from test_dld_workflow.py ---
import torch

# --- merged from test_dld_workflow.py ---
from lnl_toolbox.algorithms.dld.precorrection import DLDPartitionResult

# --- merged from test_dld_workflow.py ---
from lnl_toolbox.algorithms.dld.state import DLDPhase, DLDState

# --- merged from test_dld_workflow.py ---
from lnl_toolbox.data.cifar import CifarData

# --- merged from test_dld_workflow.py ---
from lnl_toolbox.training.checkpoint import read_checkpoint

# --- merged from test_dld_workflow.py ---
from lnl_toolbox.training.dld_experiment import run_dld_experiment

# --- merged from test_dld_workflow.py ---
def _dld_workflow__data(split: str, samples_per_class: int, classes: int=10) -> CifarData:
    labels = np.repeat(np.arange(classes, dtype=np.int64), samples_per_class)
    rng = np.random.default_rng(91 if split == 'train' else 92)
    images = rng.integers(0, 256, (labels.size, 32, 32, 3), dtype=np.uint8)
    dataset = 'cifar10' if classes == 10 else 'cifar100'
    return CifarData(images, labels, tuple((str(i) for i in range(classes))), split, dataset)

# --- merged from test_dld_workflow.py ---
def _dld_workflow__config(epochs: int) -> dict:
    return {'method': 'dld', 'execution': {'runner': 'dld'}, 'seed': 5, 'data': {'name': 'cifar10', 'root': 'unused', 'validation_size': 10, 'max_train_samples': 20, 'max_validation_samples': 10, 'max_test_samples': 10}, 'loader': {'batch_size': 10, 'num_workers': 0, 'pin_memory': False}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 8, 'validation_targets': 'noisy'}, 'dld': {'fidelity': {'name': 'paper_oriented_v2_cosine_similarity', 'hard_y0': 'averaged_views', 'direction_endpoint': 'estimated_yn', 'direction': 'yn_minus_y0', 'neighbor_metric': 'cosine_similarity', 'neighbor_weighting': 'inverse_neighbor_value', 'self_neighbor': 'include', 'divergence': 'kl_ps_to_pw', 'divergence_softmax': False, 'hard_yn_zero_denominator': 'fail', 'schedule': 'average', 'inference_initialization': 'zero', 'inference_steps': 5}, 'feature_extractor': {'source': 'repository_frozen_model', 'model': {'name': 'tiny_cnn', 'width': 4}}, 'precorrection': {'k_neighbors': 3, 'delta': 1e-06, 'gmm_components': 2, 'gmm_seed': 0, 'minimum_mean_separation': 0.0, 'query_chunk_size': 2}, 'diffusion': {'timesteps': 5, 'epochs': epochs, 'model': {'independent_predictors': True, 'hidden_dim': 8, 'time_dim': 4}, 'optimizer': {'direction': {'name': 'adam', 'lr': 0.001}, 'noise': {'name': 'adam', 'lr': 0.001}}, 'scheduler': {'direction': {'name': 'none'}, 'noise': {'name': 'none'}}, 'ema': {'enabled': True, 'decay': 0.9}}, 'inference': {'steps': 5, 'deterministic': True, 'initialization': 'zero'}}, 'evaluation': {'selection_split': 'validation', 'primary': 'accuracy'}, 'trainer': {'device': 'cpu'}, 'output_root': 'unused'}

# --- merged from test_dld_workflow.py ---
def _dld_workflow__partition(p_w, p_s, targets, **kwargs):
    p_ws = ((p_w + p_s) / 2).detach()
    partition = torch.arange(p_w.shape[0], device=p_w.device) % 2
    divergence = torch.linspace(0.01, 1.0, p_w.shape[0], device=p_w.device, dtype=p_w.dtype)
    return DLDPartitionResult(divergence, partition, p_ws, 0.1, 0.9)

# --- merged from test_dld_workflow.py ---
class _dld_workflow_DLDWorkflowTest(unittest.TestCase):

    def test_state_rejects_illegal_transition(self) -> None:
        with self.assertRaisesRegex(ValueError, 'illegal'):
            DLDState().advance(DLDPhase.DIFFUSION_TRAINING)

    def test_fresh_resume_extension_and_completed_noop(self) -> None:
        train, test = (_dld_workflow__data('train', 4), _dld_workflow__data('test', 2))
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=lambda root, split: train if split == 'train' else test), patch('lnl_toolbox.training.dld_experiment.partition_samples', side_effect=_dld_workflow__partition):
            run = Path(directory) / 'run'
            run_dld_experiment(_dld_workflow__config(1), run)
            first = read_checkpoint(run / 'last.pt', 'cpu')
            self.assertEqual(first['dld_state']['phase'], 'completed')
            artifact = run / 'dld_precorrection.npz'
            hash_before = first['dld_state']['precorrection_artifact_hash']
            mtime_before = artifact.stat().st_mtime_ns
            run_dld_experiment(_dld_workflow__config(2), resume=run / 'last.pt')
            second = read_checkpoint(run / 'last.pt', 'cpu')
            self.assertEqual(second['dld_state']['completed_epochs'], 2)
            self.assertGreater(second['dld_state']['global_step'], first['dld_state']['global_step'])
            self.assertEqual(second['dld_state']['precorrection_artifact_hash'], hash_before)
            self.assertEqual(artifact.stat().st_mtime_ns, mtime_before)
            self.assertIn('direction_model', second['algorithm'])
            self.assertIn('noise_model', second['algorithm'])
            drifted = _dld_workflow__config(2)
            drifted['dld']['precorrection']['k_neighbors'] = 4
            with self.assertRaisesRegex(ValueError, 'identity'):
                run_dld_experiment(drifted, resume=run / 'last.pt')
            with self.assertRaisesRegex(ValueError, 'cannot be reduced'):
                run_dld_experiment(_dld_workflow__config(1), resume=run / 'last.pt')
            files = {name: (run / name).stat().st_mtime_ns for name in ('last.pt', 'best.pt', 'metrics.jsonl', 'dld_precorrection.npz')}
            run_dld_experiment(_dld_workflow__config(2), resume=run / 'last.pt')
            self.assertEqual(files, {name: (run / name).stat().st_mtime_ns for name in files})
            final = json.loads((run / 'final_metrics.json').read_text())
            self.assertFalse(final['test_selection_leakage'])
            for name in ('test_reverse_output_min', 'test_reverse_output_max', 'test_reverse_output_std', 'test_reverse_prediction_class_count', 'test_reverse_prediction_entropy'):
                self.assertTrue(np.isfinite(final[name]), name)
            epoch_rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if '"event": "diffusion_epoch"' in line]
            self.assertTrue(epoch_rows)
            for name in ('direction_parameter_norm', 'noise_parameter_norm', 'predicted_direction_rms', 'predicted_noise_rms', 'validation_reverse_output_std', 'validation_reverse_prediction_class_count'):
                self.assertTrue(np.isfinite(epoch_rows[-1][name]), name)
            for name in ('last.pt', 'best.pt', 'metrics.jsonl', 'final_metrics.json', 'noise_manifest.npz'):
                self.assertTrue((run / name).is_file(), name)

    def test_cifar100_dispatch_is_supported(self) -> None:
        config = _dld_workflow__config(1)
        config['data']['name'] = 'cifar100'
        config['data'].update({'validation_size': 100, 'max_train_samples': 100, 'max_validation_samples': 100, 'max_test_samples': 100})
        config['loader']['batch_size'] = 100
        train, test = (_dld_workflow__data('train', 2, 100), _dld_workflow__data('test', 1, 100))
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar100', side_effect=lambda root, split: train if split == 'train' else test), patch('lnl_toolbox.training.dld_experiment.partition_samples', side_effect=_dld_workflow__partition):
            run = run_dld_experiment(config, Path(directory) / 'cifar100')
            final = json.loads((run / 'final_metrics.json').read_text())
            self.assertEqual(final['completed_epochs'], 1)

# --- merged from test_dld_readiness.py ---
import os

# --- merged from test_dld_readiness.py ---
from pathlib import Path

# --- merged from test_dld_readiness.py ---
import tempfile

# --- merged from test_dld_readiness.py ---
import unittest

# --- merged from test_dld_readiness.py ---
from unittest import mock

# --- merged from test_dld_readiness.py ---
import numpy as np

# --- merged from test_dld_readiness.py ---
import torch

# --- merged from test_dld_readiness.py ---
import yaml

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.algorithms.dld import DLDConfig

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.noise.generators import generate_symmetric

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.noise.manifest import fingerprint_labels

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.training.checkpoint import atomic_save

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.training.dld_pretrained import load_torchvision_resnet34_imagenet1k_v1_source, load_upm_main_best_feature_source

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.training.experiment import build_model

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.training.noisy_labels import file_sha256

# --- merged from test_dld_readiness.py ---
_dld_readiness_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_dld_readiness.py ---
def _dld_readiness__source(directory: Path) -> tuple[dict, torch.nn.Module]:
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
    return ({'adapter': 'upm_main_best', 'run_directory_env': 'DLD_TEST_SOURCE', 'checkpoint_sha256': file_sha256(checkpoint_path), 'manifest_sha256': file_sha256(manifest_path), 'mapping_hash': manifest.mapping_hash, 'dataset_fingerprint': manifest.dataset_fingerprint, 'model': model_config}, model)

# --- merged from test_dld_readiness.py ---
class _dld_readiness_DLDReadinessTest(unittest.TestCase):

    def test_formal_config_uses_current_contract_and_full_budget(self) -> None:
        path = _dld_readiness_ROOT / 'configs/experiment/dld_cifar10_reproduction.yaml'
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        parsed = DLDConfig.from_mapping(config)
        self.assertEqual(config['configuration_fidelity'], 'paper_oriented')
        self.assertEqual(config['loader']['batch_size'], 200)
        self.assertEqual(parsed.precorrection['k_neighbors'], 50)
        self.assertEqual(parsed.diffusion['timesteps'], 1000)
        self.assertEqual(parsed.epochs, 200)
        self.assertEqual(parsed.feature_extractor['source'], 'external_checkpoint')
        self.assertEqual(parsed.feature_extractor['external']['adapter'], 'torchvision_resnet34_imagenet1k_v1')

    def test_torchvision_resnet34_source_is_pretrained_frozen_and_explicit(self) -> None:
        from torchvision.models import resnet34

        with tempfile.TemporaryDirectory() as directory:
            cached = Path(directory) / 'resnet34-b627a593.pth'
            cached.write_bytes(b'official-cache-fixture')
            model = resnet34(weights=None)
            config = {
                'adapter': 'torchvision_resnet34_imagenet1k_v1',
                'weights': 'IMAGENET1K_V1',
                'input_contract': 'cifar10_standard_normalized',
            }
            with mock.patch(
                'lnl_toolbox.training.dld_pretrained._cached_torchvision_weight',
                return_value=cached,
            ), mock.patch('torchvision.models.resnet34', return_value=model) as factory:
                source = load_torchvision_resnet34_imagenet1k_v1_source(config)
            self.assertEqual(factory.call_args.kwargs['weights'].name, 'IMAGENET1K_V1')
            self.assertEqual(source.provenance['feature_dimension'], 512)
            self.assertEqual(source.provenance['input_contract'], 'cifar10_standard_normalized')
            self.assertTrue(source.provenance['frozen'])
            self.assertTrue(all(not parameter.requires_grad for parameter in source.model.parameters()))
            output = source.model.forward_with_features(torch.zeros(1, 3, 32, 32))
            self.assertEqual(tuple(output.features.shape), (1, 512))

    def test_torchvision_source_does_not_download_missing_weights(self) -> None:
        config = {
            'adapter': 'torchvision_resnet34_imagenet1k_v1',
            'weights': 'IMAGENET1K_V1',
            'input_contract': 'cifar10_standard_normalized',
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            'torch.hub.get_dir', return_value=directory
        ):
            with self.assertRaisesRegex(FileNotFoundError, 'not cached'):
                load_torchvision_resnet34_imagenet1k_v1_source(config)

    def test_real_short_config_is_full_data_external_sym20(self) -> None:
        path = _dld_readiness_ROOT / 'configs' / 'reproduction' / 'cifar10_dld_sym20_short.yaml'
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        parsed = DLDConfig.from_mapping(config)
        self.assertEqual(config['method'], 'dld')
        self.assertEqual(config['data']['name'], 'cifar10')
        for name in ('max_train_samples', 'max_validation_samples', 'max_test_samples'):
            self.assertNotIn(name, config['data'])
        self.assertEqual(config['noise']['rate'], 0.2)
        self.assertEqual(parsed.fidelity['name'], 'paper_oriented_v2_cosine_similarity')
        self.assertEqual(parsed.fidelity['neighbor_metric'], 'cosine_similarity')
        self.assertEqual(parsed.fidelity['neighbor_weighting'], 'inverse_neighbor_value')
        self.assertEqual(parsed.feature_extractor['source'], 'external_checkpoint')
        self.assertEqual(parsed.precorrection['query_chunk_size'], 64)
        self.assertEqual(parsed.epochs, 15)
        legacy = yaml.safe_load(path.read_text(encoding='utf-8'))
        legacy['dld']['fidelity']['name'] = 'paper_oriented_v1'
        legacy['dld']['fidelity']['neighbor_metric'] = 'cosine_distance'
        legacy['dld']['fidelity'].pop('neighbor_weighting')
        with self.assertRaisesRegex(ValueError, 'fidelity'):
            DLDConfig.from_mapping(legacy)

    def test_external_source_is_strict_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _dld_readiness__source(root)
            with mock.patch.dict(os.environ, {'DLD_TEST_SOURCE': directory}):
                source = load_upm_main_best_feature_source(config, build_model(config['model'], 10), num_classes=10)
                self.assertEqual(source.provenance['adapter'], 'upm_main_best')
                source.assert_unchanged()
                original = (root / 'noise_manifest.npz').read_bytes()
                (root / 'noise_manifest.npz').write_bytes(original + b'changed')
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    source.assert_unchanged()

    def test_external_source_rejects_identity_and_role_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _dld_readiness__source(root)
            with mock.patch.dict(os.environ, {'DLD_TEST_SOURCE': directory}):
                invalid = {**config, 'checkpoint_sha256': 'f' * 64}
                with self.assertRaisesRegex(ValueError, 'checkpoint SHA-256'):
                    load_upm_main_best_feature_source(invalid, build_model(config['model'], 10), num_classes=10)
                payload = torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
                payload['checkpoint_role'] = 'last'
                atomic_save(payload, root / 'best.pt')
                role_config = {**config, 'checkpoint_sha256': file_sha256(root / 'best.pt')}
                with self.assertRaisesRegex(ValueError, 'main_best'):
                    load_upm_main_best_feature_source(role_config, build_model(config['model'], 10), num_classes=10)
