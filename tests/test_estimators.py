"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_estimators.py ---
from dataclasses import FrozenInstanceError, dataclass

# --- merged from test_estimators.py ---
import unittest

# --- merged from test_estimators.py ---
import torch

# --- merged from test_estimators.py ---
from lnl_toolbox.estimators import ReliabilityEstimator, ReliabilityResult, StatisticResult, validate_reliability_result, validate_statistic_result

# --- merged from test_estimators.py ---
@dataclass(frozen=True)
class _estimators_DummyReliabilityInput:
    values: torch.Tensor
    sample_indices: torch.Tensor

# --- merged from test_estimators.py ---
class _estimators_DummyStatelessEstimator:

    def estimate(self, estimator_input: _estimators_DummyReliabilityInput) -> ReliabilityResult:
        return ReliabilityResult(sample_indices=estimator_input.sample_indices, scores=estimator_input.values.detach(), metrics={'score_mean': float(estimator_input.values.mean().item())})

# --- merged from test_estimators.py ---
class _estimators_EstimatorContractTest(unittest.TestCase):

    def test_stateless_estimator_satisfies_protocol_without_state_methods(self):
        estimator = _estimators_DummyStatelessEstimator()
        self.assertIsInstance(estimator, ReliabilityEstimator)
        self.assertFalse(hasattr(estimator, 'state_dict'))
        result = estimator.estimate(_estimators_DummyReliabilityInput(values=torch.tensor([0.2, 0.9]), sample_indices=torch.tensor([4, 1])))
        indices, scores = validate_reliability_result(result)
        self.assertEqual(indices.tolist(), [4, 1])
        self.assertEqual(scores.tolist(), result.scores.tolist())

    def test_larger_scores_have_the_documented_more_reliable_direction(self):
        result = ReliabilityResult(sample_indices=torch.tensor([8, 3, 5]), scores=torch.tensor([0.2, 0.9, 0.5]))
        _, scores = validate_reliability_result(result)
        ranked_indices = result.sample_indices[torch.argsort(scores, descending=True)]
        self.assertEqual(ranked_indices.tolist(), [3, 5, 8])

    def test_reliability_result_is_frozen(self):
        result = ReliabilityResult(sample_indices=torch.tensor([0]), scores=torch.tensor([1.0]))
        with self.assertRaises(FrozenInstanceError):
            result.scores = torch.tensor([0.0])

    def test_expected_indices_require_exact_order_and_values(self):
        result = ReliabilityResult(sample_indices=torch.tensor([7, 2]), scores=torch.tensor([0.8, 0.1]))
        validate_reliability_result(result, expected_sample_indices=torch.tensor([7, 2]))
        with self.assertRaisesRegex(ValueError, 'expected order'):
            validate_reliability_result(result, expected_sample_indices=torch.tensor([2, 7]))

    def test_reliability_validation_rejects_invalid_indices(self):
        cases = ((ReliabilityResult(sample_indices=torch.tensor([[0, 1]]), scores=torch.tensor([0.1, 0.2])), 'one-dimensional'), (ReliabilityResult(sample_indices=torch.tensor([], dtype=torch.long), scores=torch.tensor([])), 'must not be empty'), (ReliabilityResult(sample_indices=torch.tensor([0.0, 1.0]), scores=torch.tensor([0.1, 0.2])), 'integer dtype'), (ReliabilityResult(sample_indices=torch.tensor([1, 1]), scores=torch.tensor([0.1, 0.2])), 'unique'))
        for result, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_reliability_result(result)

    def test_reliability_validation_rejects_invalid_scores(self):
        cases = ((ReliabilityResult(sample_indices=torch.tensor([0, 1]), scores=torch.tensor([[0.1, 0.2]])), 'one-dimensional'), (ReliabilityResult(sample_indices=torch.tensor([0, 1]), scores=torch.tensor([0.1])), 'one-to-one'), (ReliabilityResult(sample_indices=torch.tensor([0, 1]), scores=torch.tensor([0, 1])), 'floating-point'), (ReliabilityResult(sample_indices=torch.tensor([0, 1]), scores=torch.tensor([0.1, float('nan')])), 'finite'), (ReliabilityResult(sample_indices=torch.tensor([0, 1]), scores=torch.tensor([0.1, 0.2], requires_grad=True)), 'detached'))
        for result, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_reliability_result(result)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_reliability_validation_rejects_device_mismatch(self):
        result = ReliabilityResult(sample_indices=torch.tensor([0, 1]), scores=torch.tensor([0.1, 0.2], device='cuda'))
        with self.assertRaisesRegex(ValueError, 'same device'):
            validate_reliability_result(result)

    def test_reliability_metrics_require_finite_python_floats(self):
        cases = (({'count': 2}, TypeError, 'Python float'), ({'mean': float('inf')}, ValueError, 'finite'), ({1: 0.5}, TypeError, 'names'))
        for metrics, error_type, message in cases:
            with self.subTest(metrics=metrics), self.assertRaisesRegex(error_type, message):
                validate_reliability_result(ReliabilityResult(sample_indices=torch.tensor([0]), scores=torch.tensor([0.5]), metrics=metrics))

    def test_statistic_result_preserves_arbitrary_typed_payload(self):

        @dataclass(frozen=True)
        class CentroidStatistics:
            centroids: torch.Tensor
            class_counts: tuple[int, ...]
        payload = CentroidStatistics(centroids=torch.tensor([[1.0, 2.0]]), class_counts=(3,))
        result = StatisticResult(statistics=payload, metrics={'classes': 1.0})
        self.assertIs(validate_statistic_result(result), payload)
        self.assertFalse(hasattr(result, 'fit'))
        self.assertFalse(hasattr(result, 'compute'))
        self.assertFalse(hasattr(result, 'state_dict'))

    def test_statistic_validation_does_not_inspect_payload(self):
        opaque_payload = object()
        self.assertIs(validate_statistic_result(StatisticResult(opaque_payload)), opaque_payload)

    def test_statistic_validation_checks_only_container_and_metrics(self):
        with self.assertRaisesRegex(TypeError, 'StatisticResult'):
            validate_statistic_result({'statistics': object()})
        with self.assertRaisesRegex(TypeError, 'Python float'):
            validate_statistic_result(StatisticResult(statistics=object(), metrics={'count': 1}))

# --- merged from test_transition_estimators.py ---
import json

# --- merged from test_transition_estimators.py ---
from pathlib import Path

# --- merged from test_transition_estimators.py ---
import tempfile

# --- merged from test_transition_estimators.py ---
import unittest

# --- merged from test_transition_estimators.py ---
import numpy as np

# --- merged from test_transition_estimators.py ---
import torch

# --- merged from test_transition_estimators.py ---
from torch.utils.data import DataLoader

# --- merged from test_transition_estimators.py ---
from lnl_toolbox.noise import AnchorTransitionEstimator, DualTransitionEstimator, KnownTransitionEstimator, PosteriorSnapshot, TransitionArtifact

# --- merged from test_transition_estimators.py ---
from lnl_toolbox.training.snapshots import collect_posterior_snapshot

# --- merged from test_transition_estimators.py ---
class _transition_estimators_TransitionEstimatorTest(unittest.TestCase):

    def test_collector_sorts_indices_and_restores_model_mode(self) -> None:
        records = [{'input': torch.tensor([0.0, 2.0]), 'target': 1, 'index': 20}, {'input': torch.tensor([3.0, 0.0]), 'target': 0, 'index': 10}, {'input': torch.tensor([0.5, 0.5]), 'target': 1, 'index': 30}]
        model = torch.nn.Identity()
        model.train()
        snapshot = collect_posterior_snapshot(model, DataLoader(records, batch_size=2, shuffle=False), 'cpu', dataset='fixture', split='train')
        self.assertTrue(model.training)
        np.testing.assert_array_equal(snapshot.global_indices, [10, 20, 30])
        np.testing.assert_array_equal(snapshot.noisy_targets, [0, 1, 1])
        expected_logits = torch.stack([records[1]['input'], records[0]['input'], records[2]['input']])
        expected = torch.softmax(expected_logits, dim=1).numpy()
        np.testing.assert_allclose(snapshot.noisy_probabilities, expected, atol=1e-07)

    def test_collector_rejects_empty_or_duplicate_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, 'at least one batch'):
            collect_posterior_snapshot(torch.nn.Identity(), DataLoader([], batch_size=1), 'cpu', dataset='fixture', split='train')
        records = [{'input': torch.tensor([1.0, 0.0]), 'target': 0, 'index': 7}, {'input': torch.tensor([0.0, 1.0]), 'target': 1, 'index': 7}]
        with self.assertRaisesRegex(ValueError, 'unique'):
            collect_posterior_snapshot(torch.nn.Identity(), DataLoader(records, batch_size=2), 'cpu', dataset='fixture', split='train')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_collector_accepts_cuda_model_and_returns_cpu_snapshot(self) -> None:
        records = [{'input': torch.tensor([2.0, 0.0]), 'target': 0, 'index': 1}, {'input': torch.tensor([0.0, 2.0]), 'target': 1, 'index': 0}]
        snapshot = collect_posterior_snapshot(torch.nn.Identity().cuda(), DataLoader(records, batch_size=2), 'cuda', dataset='fixture', split='train')
        np.testing.assert_array_equal(snapshot.global_indices, [0, 1])
        self.assertEqual(snapshot.noisy_probabilities.dtype, np.float64)

    @staticmethod
    def _dual_t_snapshot() -> PosteriorSnapshot:
        return PosteriorSnapshot(noisy_probabilities=np.asarray([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]), noisy_targets=np.asarray([0, 1, 2, 1, 2, 0]), global_indices=np.asarray([30, 10, 50, 20, 40, 60]), dataset='fixture', split='train')

    def test_dual_t_matches_paper_factorization(self) -> None:
        snapshot = self._dual_t_snapshot()
        artifact = DualTransitionEstimator().estimate(snapshot)
        t_club = np.asarray([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
        t_spade = np.asarray([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
        self.assertEqual(artifact.estimator, 'dual_t')
        self.assertEqual(artifact.source_snapshot_hash, snapshot.snapshot_hash)
        self.assertEqual(artifact.metadata['composition'], 't_club @ t_spade')
        np.testing.assert_allclose(artifact.metadata['t_club'], t_club)
        np.testing.assert_allclose(artifact.metadata['t_spade'], t_spade)
        np.testing.assert_allclose(artifact.matrix, t_club @ t_spade)
        self.assertEqual(artifact.metadata['anchor_global_indices'], [30, 10, 50])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dual_t.npz'
            artifact.save(path)
            loaded = TransitionArtifact.load(path)
        self.assertEqual(loaded.artifact_hash, artifact.artifact_hash)
        np.testing.assert_allclose(loaded.metadata['t_club'], t_club)
        np.testing.assert_allclose(loaded.metadata['t_spade'], t_spade)

    def test_dual_t_is_invariant_to_snapshot_order(self) -> None:
        snapshot = self._dual_t_snapshot()
        order = np.asarray([4, 1, 5, 0, 3, 2])
        reordered = PosteriorSnapshot(noisy_probabilities=snapshot.noisy_probabilities[order], noisy_targets=snapshot.noisy_targets[order], global_indices=snapshot.global_indices[order], dataset=snapshot.dataset, split=snapshot.split)
        first = DualTransitionEstimator().estimate(snapshot)
        second = DualTransitionEstimator().estimate(reordered)
        np.testing.assert_allclose(first.matrix, second.matrix)
        np.testing.assert_allclose(first.metadata['t_club'], second.metadata['t_club'])
        np.testing.assert_allclose(first.metadata['t_spade'], second.metadata['t_spade'])
        self.assertEqual(first.metadata['anchor_global_indices'], second.metadata['anchor_global_indices'])

    def test_dual_t_rejects_empty_intermediate_class(self) -> None:
        snapshot = PosteriorSnapshot(noisy_probabilities=np.asarray([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1]]), noisy_targets=np.asarray([0, 1, 2]), global_indices=np.asarray([0, 1, 2]), dataset='fixture', split='train')
        with self.assertRaisesRegex(ValueError, 'empty intermediate classes: 2'):
            DualTransitionEstimator().estimate(snapshot)

    def test_dual_t_reduces_to_anchor_with_identity_second_factor(self) -> None:
        probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        snapshot = PosteriorSnapshot(noisy_probabilities=probabilities, noisy_targets=probabilities.argmax(axis=1), global_indices=np.asarray([0, 1, 2, 3]), dataset='fixture', split='train')
        anchor = AnchorTransitionEstimator().estimate(snapshot)
        dual = DualTransitionEstimator().estimate(snapshot)
        np.testing.assert_allclose(dual.metadata['t_spade'], np.eye(2))
        np.testing.assert_allclose(dual.matrix, anchor.matrix)

    def setUp(self) -> None:
        self.matrix = np.array([[0.8, 0.1, 0.1], [0.1, 0.75, 0.15], [0.05, 0.1, 0.85]], dtype=np.float64)
        self.probabilities = np.array([self.matrix[0], self.matrix[1], self.matrix[2], [0.5, 0.25, 0.25], [0.2, 0.5, 0.3]], dtype=np.float64)
        self.targets = np.array([0, 1, 2, 0, 1], dtype=np.int64)
        self.indices = np.array([30, 10, 50, 20, 40], dtype=np.int64)

    def snapshot(self, probabilities: np.ndarray | None=None, targets: np.ndarray | None=None, indices: np.ndarray | None=None) -> PosteriorSnapshot:
        return PosteriorSnapshot(self.probabilities if probabilities is None else probabilities, self.targets if targets is None else targets, self.indices if indices is None else indices, 'cifar10', 'train')

    def test_anchor_estimator_recovers_transition_and_records_identity(self) -> None:
        snapshot = self.snapshot()
        artifact = AnchorTransitionEstimator().estimate(snapshot)
        np.testing.assert_allclose(artifact.matrix, self.matrix)
        self.assertEqual(artifact.estimator, 'anchor')
        self.assertEqual(artifact.source_snapshot_hash, snapshot.snapshot_hash)
        self.assertEqual(artifact.metadata['anchor_global_indices'], [30, 10, 50])
        self.assertFalse(artifact.matrix.flags.writeable)

    def test_known_estimator_returns_configured_transition(self) -> None:
        snapshot = self.snapshot()
        artifact = KnownTransitionEstimator(self.matrix).estimate(snapshot)
        np.testing.assert_allclose(artifact.matrix, self.matrix)
        self.assertEqual(artifact.estimator, 'known')
        self.assertEqual(artifact.metadata['source'], 'configuration')

    def test_known_estimator_rejects_wrong_class_count(self) -> None:
        with self.assertRaisesRegex(ValueError, 'classes'):
            KnownTransitionEstimator(np.eye(2)).estimate(self.snapshot())

    def test_input_reordering_does_not_change_estimate(self) -> None:
        original = AnchorTransitionEstimator().estimate(self.snapshot())
        order = np.array([4, 2, 0, 3, 1], dtype=np.int64)
        reordered = AnchorTransitionEstimator().estimate(self.snapshot(self.probabilities[order], self.targets[order], self.indices[order]))
        np.testing.assert_array_equal(reordered.matrix, original.matrix)
        self.assertEqual(reordered.metadata['anchor_global_indices'], original.metadata['anchor_global_indices'])

    def test_anchor_tie_uses_smallest_global_index(self) -> None:
        probabilities = np.array([[0.8, 0.1, 0.1], [0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        snapshot = PosteriorSnapshot(probabilities, np.array([0, 0, 1, 2]), np.array([20, 5, 30, 40]), 'fixture', 'train')
        artifact = AnchorTransitionEstimator().estimate(snapshot)
        np.testing.assert_array_equal(artifact.matrix[0], probabilities[1])
        self.assertEqual(artifact.metadata['anchor_global_indices'][0], 5)

    def test_snapshot_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.snapshot(probabilities=self.probabilities[:, 0])
        with self.assertRaisesRegex(ValueError, 'noisy_targets'):
            self.snapshot(targets=self.targets[:-1])
        with self.assertRaisesRegex(ValueError, 'global_indices'):
            self.snapshot(indices=np.array([1, 1, 2, 3, 4]))
        invalid = self.probabilities.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.snapshot(probabilities=invalid)
        invalid = self.probabilities.copy()
        invalid[0] = [0.8, 0.3, -0.1]
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            self.snapshot(probabilities=invalid)
        invalid = self.probabilities.copy()
        invalid[0] = [0.8, 0.2, 0.2]
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            self.snapshot(probabilities=invalid)
        with self.assertRaisesRegex(ValueError, 'within'):
            self.snapshot(targets=np.array([0, 1, 3, 0, 1]))

    def test_snapshot_hash_binds_probabilities_targets_and_indices(self) -> None:
        snapshot = self.snapshot()
        changed_targets = self.targets.copy()
        changed_targets[0] = 1
        changed_indices = self.indices.copy()
        changed_indices[0] = 31
        self.assertNotEqual(snapshot.snapshot_hash, self.snapshot(targets=changed_targets).snapshot_hash)
        self.assertNotEqual(snapshot.snapshot_hash, self.snapshot(indices=changed_indices).snapshot_hash)

    def test_artifact_roundtrip_and_tamper_detection(self) -> None:
        artifact = AnchorTransitionEstimator().estimate(self.snapshot())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'transition.npz'
            artifact.save(path)
            loaded = TransitionArtifact.load(path)
            np.testing.assert_array_equal(loaded.matrix, artifact.matrix)
            self.assertEqual(loaded.artifact_hash, artifact.artifact_hash)
            with np.load(path, allow_pickle=False) as data:
                matrix = data['matrix'].copy()
                metadata_json = data['metadata_json'].copy()
            matrix[0] = [0.7, 0.2, 0.1]
            np.savez_compressed(path, matrix=matrix, metadata_json=metadata_json)
            with self.assertRaisesRegex(ValueError, 'hash'):
                TransitionArtifact.load(path)
            artifact.save(path)
            with np.load(path, allow_pickle=False) as data:
                matrix = data['matrix'].copy()
                payload = json.loads(str(data['metadata_json'].item()))
            payload['metadata']['dataset'] = 'tampered'
            np.savez_compressed(path, matrix=matrix, metadata_json=np.array(json.dumps(payload)))
            with self.assertRaisesRegex(ValueError, 'hash'):
                TransitionArtifact.load(path)

    def test_artifact_rejects_invalid_matrix_and_convention(self) -> None:
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            TransitionArtifact(np.eye(3) * 0.5, 'anchor')
        with self.assertRaisesRegex(ValueError, 'convention'):
            TransitionArtifact(np.eye(3), 'anchor', convention='column')

    def test_artifact_tensor_device_and_dtype(self) -> None:
        artifact = AnchorTransitionEstimator().estimate(self.snapshot())
        cpu = artifact.as_tensor(device='cpu', dtype=torch.float64)
        self.assertEqual(cpu.device.type, 'cpu')
        self.assertEqual(cpu.dtype, torch.float64)
        np.testing.assert_allclose(cpu.numpy(), artifact.matrix)
        if torch.cuda.is_available():
            cuda = artifact.as_tensor(device='cuda', dtype=torch.float32)
            self.assertEqual(cuda.device.type, 'cuda')
            self.assertEqual(cuda.dtype, torch.float32)
            np.testing.assert_allclose(cuda.cpu().numpy(), artifact.matrix, rtol=1e-06)

# --- merged from test_volmin_training.py ---
import tempfile

# --- merged from test_volmin_training.py ---
import unittest

# --- merged from test_volmin_training.py ---
from pathlib import Path

# --- merged from test_volmin_training.py ---
import yaml

# --- merged from test_volmin_training.py ---
from lnl_toolbox.training.volmin_experiment import run_volmin_experiment

# --- merged from test_volmin_training.py ---
class _volmin_training_VolMinTrainingTest(unittest.TestCase):

    def test_smoke_and_resume(self) -> None:
        config = yaml.safe_load(Path('configs/experiment/volmin_cifar10_smoke.yaml').read_text())
        with tempfile.TemporaryDirectory() as directory:
            run = run_volmin_experiment(config, output_dir=directory)
            self.assertTrue((run / 'last.pt').is_file())
            run_volmin_experiment(config, resume=run / 'last.pt')

# --- merged from test_volminnet_algorithm.py ---
import copy

# --- merged from test_volminnet_algorithm.py ---
import unittest

# --- merged from test_volminnet_algorithm.py ---
import torch

# --- merged from test_volminnet_algorithm.py ---
from torch import nn

# --- merged from test_volminnet_algorithm.py ---
from lnl_toolbox.algorithms.volminnet import VolMinNetAlgorithm, VolMinTransition

# --- merged from test_volminnet_algorithm.py ---
class _volminnet_algorithm_VolMinNetAlgorithmTest(unittest.TestCase):

    def _algorithm(self) -> VolMinNetAlgorithm:
        model = nn.Linear(4, 3)
        transition = VolMinTransition(3)
        optimizer_model = torch.optim.SGD(model.parameters(), lr=0.05)
        optimizer_transition = torch.optim.SGD(transition.parameters(), lr=0.05)
        return VolMinNetAlgorithm(model=model, transition=transition, classifier_optimizer=optimizer_model, transition_optimizer=optimizer_transition, classifier_scheduler=None, transition_scheduler=None, lambda_volume=0.0001, device=torch.device('cpu'))

    def test_joint_step_updates_both_once_with_disjoint_optimizers(self) -> None:
        algorithm = self._algorithm()
        model_before = copy.deepcopy(algorithm.model.state_dict())
        transition_before = algorithm.transition.off_diagonal_logits.detach().clone()
        metrics = algorithm.train_batch({'input': torch.randn(6, 4), 'target': torch.tensor([0, 1, 2, 0, 1, 2])})
        self.assertTrue(any((not torch.equal(value, model_before[name]) for name, value in algorithm.model.state_dict().items())))
        self.assertFalse(torch.equal(algorithm.transition.off_diagonal_logits, transition_before))
        self.assertEqual(algorithm.state.global_step, 1)
        self.assertEqual(algorithm.state.classifier_optimizer_steps, 1)
        self.assertEqual(algorithm.state.transition_optimizer_steps, 1)
        self.assertTrue(all((torch.isfinite(torch.tensor(value)) for value in metrics.values())))

    def test_optimizer_ownership_mismatch_fails(self) -> None:
        model = nn.Linear(4, 3)
        transition = VolMinTransition(3)
        wrong = torch.optim.SGD(transition.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            VolMinNetAlgorithm(model=model, transition=transition, classifier_optimizer=wrong, transition_optimizer=torch.optim.SGD(transition.parameters(), lr=0.1), classifier_scheduler=None, transition_scheduler=None, lambda_volume=0.0001, device=torch.device('cpu'))

    def test_state_roundtrip_keeps_roles_and_progress(self) -> None:
        first = self._algorithm()
        first.train_batch({'input': torch.randn(3, 4), 'target': torch.tensor([0, 1, 2])})
        state = first.state_dict()
        second = self._algorithm()
        second.load_state_dict(state)
        self.assertEqual(second.state.state_dict(), first.state.state_dict())
        self.assertEqual(second.transition.state_dict().keys(), first.transition.state_dict().keys())

# --- merged from test_volminnet_objective.py ---
import unittest

# --- merged from test_volminnet_objective.py ---
import torch

# --- merged from test_volminnet_objective.py ---
from torch.nn import functional as F

# --- merged from test_volminnet_objective.py ---
from lnl_toolbox.algorithms.volminnet import volminnet_objective

# --- merged from test_volminnet_objective.py ---
class _volminnet_objective_VolMinNetObjectiveTest(unittest.TestCase):

    def test_asymmetric_row_direction_and_hand_calculation(self) -> None:
        logits = torch.tensor([[1.1, -0.2, 0.4]], dtype=torch.float64, requires_grad=True)
        transition = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.25, 0.15, 0.6]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1])
        objective, metrics = volminnet_objective(logits, targets, transition, lambda_volume=0.03)
        clean = torch.softmax(logits, 1)
        expected_nll = -torch.log((clean @ transition)[0, 1])
        expected = expected_nll + 0.03 * torch.logdet(transition)
        self.assertTrue(torch.allclose(objective, expected))
        self.assertAlmostEqual(metrics['classification_loss'], float(expected_nll.detach()))
        self.assertFalse(torch.allclose(clean @ transition, clean @ transition.T))

    def test_classifier_and_transition_receive_finite_gradients(self) -> None:
        logits = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        raw = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.15, 0.1, 0.75]], dtype=torch.float64, requires_grad=True)
        objective, _ = volminnet_objective(logits, torch.tensor([0, 1, 2, 1, 0]), raw, lambda_volume=0.0001)
        objective.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))

    def test_positive_logdet_sign_is_not_ignored(self) -> None:
        negative = torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]], dtype=torch.float64)
        self.assertLess(float(torch.linalg.slogdet(negative).sign), 0.0)
        with self.assertRaisesRegex(ValueError, 'positive'):
            volminnet_objective(torch.randn(2, 3), torch.tensor([0, 1]), negative, lambda_volume=0.0001)

    def test_singular_and_non_finite_fail(self) -> None:
        singular = torch.full((3, 3), 1.0 / 3.0, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'positive'):
            volminnet_objective(torch.randn(2, 3), torch.tensor([0, 1]), singular, lambda_volume=0.0001)
        invalid = torch.eye(3, dtype=torch.float64)
        invalid[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            volminnet_objective(torch.randn(2, 3), torch.tensor([0, 1]), invalid, lambda_volume=0.0001)

    def test_objective_uses_log_probabilities_not_cross_entropy_on_probabilities(self) -> None:
        logits = torch.tensor([[0.2, 0.4, -0.1]], dtype=torch.float64)
        transition = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float64)
        objective, _ = volminnet_objective(logits, torch.tensor([1]), transition, lambda_volume=0.1)
        wrong = F.cross_entropy(torch.softmax(logits, 1) @ transition, torch.tensor([1]))
        wrong = wrong + 0.1 * torch.logdet(transition)
        self.assertFalse(torch.allclose(objective, wrong))

# --- merged from test_volminnet_transition.py ---
import math

# --- merged from test_volminnet_transition.py ---
import unittest

# --- merged from test_volminnet_transition.py ---
import torch

# --- merged from test_volminnet_transition.py ---
from lnl_toolbox.algorithms.volminnet import VolMinTransition

# --- merged from test_volminnet_transition.py ---
class _volminnet_transition_VolMinTransitionTest(unittest.TestCase):

    def test_paper_initialization_and_row_contract(self) -> None:
        transition = VolMinTransition(10)
        matrix = transition.matrix()
        self.assertEqual(tuple(matrix.shape), (10, 10))
        self.assertTrue(torch.allclose(matrix.sum(1), torch.ones(10, dtype=torch.float64)))
        self.assertTrue(bool((matrix >= 0).all()))
        self.assertTrue(torch.allclose(torch.diagonal(matrix), torch.full((10,), 0.5, dtype=torch.float64)))
        off = matrix[~torch.eye(10, dtype=torch.bool)]
        self.assertTrue(torch.allclose(off, torch.full_like(off, 1.0 / 18.0)))
        self.assertAlmostEqual(transition.initial_raw_value, math.log(1.0 / 8.0))

    def test_only_off_diagonal_values_are_parameters_and_receive_gradient(self) -> None:
        transition = VolMinTransition(3)
        self.assertEqual(tuple(dict(transition.named_parameters())), ('off_diagonal_logits',))
        transition.matrix()[0, 1].backward()
        self.assertIsNotNone(transition.off_diagonal_logits.grad)
        self.assertTrue(bool(torch.isfinite(transition.off_diagonal_logits.grad).all()))

    def test_diagonal_exceeds_every_same_row_off_diagonal(self) -> None:
        transition = VolMinTransition(5)
        with torch.no_grad():
            transition.off_diagonal_logits.copy_(torch.linspace(-5.0, 5.0, 20))
        matrix = transition.matrix()
        for row in range(5):
            off = torch.cat((matrix[row, :row], matrix[row, row + 1:]))
            self.assertTrue(bool((matrix[row, row] > off).all()))

    def test_binary_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'num_classes >= 3'):
            VolMinTransition(2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_contract(self) -> None:
        transition = VolMinTransition(3).cuda()
        matrix = transition.matrix()
        self.assertEqual(matrix.device.type, 'cuda')
        matrix[0, 1].backward()
        self.assertTrue(bool(torch.isfinite(transition.off_diagonal_logits.grad).all()))

# --- merged from test_volminnet_workflow.py ---
import hashlib

# --- merged from test_volminnet_workflow.py ---
import json

# --- merged from test_volminnet_workflow.py ---
from pathlib import Path

# --- merged from test_volminnet_workflow.py ---
import tempfile

# --- merged from test_volminnet_workflow.py ---
import unittest

# --- merged from test_volminnet_workflow.py ---
from unittest.mock import patch

# --- merged from test_volminnet_workflow.py ---
import numpy as np

# --- merged from test_volminnet_workflow.py ---
import torch

# --- merged from test_volminnet_workflow.py ---
import yaml

# --- merged from test_volminnet_workflow.py ---
from lnl_toolbox.algorithms.volminnet import VolMinNetConfig

# --- merged from test_volminnet_workflow.py ---
from lnl_toolbox.data.cifar import CifarData

from lnl_toolbox.data.contracts import DataSpec, RawDatasetSplit

from lnl_toolbox.data.multiclass_synthetic import generate_synthetic_multiclass

from lnl_toolbox.training.data_service import DATASETS

# --- merged from test_volminnet_workflow.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_volminnet_workflow.py ---
def _volminnet_workflow__cifar(size: int, split: str, classes: int=10) -> CifarData:
    rng = np.random.default_rng(12 if split == 'train' else 13)
    images = rng.integers(0, 256, (size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(images, labels, tuple(map(str, range(classes))), split, f'cifar{classes}')

# --- merged from test_volminnet_workflow.py ---
def _volminnet_workflow__config(epochs: int=2, dataset: str='cifar10') -> dict:
    classes = 10 if dataset == 'cifar10' else 100
    return {'method': 'volminnet', 'execution': {'runner': 'volminnet'}, 'seed': 5, 'data': {'name': dataset, 'root': 'unused', 'num_classes': classes, 'validation_size': classes, 'max_train_samples': classes * 2, 'max_validation_samples': classes, 'max_test_samples': classes, 'augment': False}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 7, 'sampling': 'transition', 'rng': 'default_rng', 'validation_targets': 'noisy', 'manifest_filename': 'noise_manifest.npz'}, 'volminnet': {'fidelity': 'paper_positive_logdet', 'model': {'name': 'tiny_cnn', 'width': 2}, 'transition': {'parameterization': 'fixed_diagonal_sigmoid_offdiag', 'convention': 'clean_to_noisy_row', 'normalization_axis': 'row', 'initialization': {'mode': 'paper'}}, 'objective': {'classification': 'noisy_nll', 'volume': {'mode': 'positive_logdet', 'coefficient': 0.0001}}, 'optimizer': {'classifier': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.0}, 'transition': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.0}}, 'scheduler': {'classifier': {'name': 'none'}, 'transition': {'name': 'none'}}, 'checkpoint_selection': {'split': 'noisy_validation', 'metric': 'loss', 'mode': 'min'}}, 'loader': {'batch_size': classes, 'num_workers': 0, 'pin_memory': False}, 'trainer': {'epochs': epochs, 'device': 'cpu'}}

# --- merged from test_volminnet_workflow.py ---
def _volminnet_workflow__sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _volminnet_workflow__GenericTabularAdapter:
    name = 'volminnet_generic_tabular_fixture'
    aliases: tuple[str, ...] = ()

    def validate(self, spec: DataSpec) -> None:
        if int(spec.options.get('num_classes', 0)) != 4:
            raise ValueError('VolMinNet fixture requires four classes')

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        if split == 'validation':
            raise ValueError('fixture intentionally uses a train-derived validation split')
        sizes = {'train': 48, 'test': 12}
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

# --- merged from test_volminnet_workflow.py ---
class _volminnet_workflow_VolMinNetWorkflowTest(unittest.TestCase):

    def test_generic_four_class_tabular_data_reaches_training(self) -> None:
        try:
            DATASETS.get(_volminnet_workflow__GenericTabularAdapter.name)
        except ValueError:
            DATASETS.add(_volminnet_workflow__GenericTabularAdapter())
        config = _volminnet_workflow__config(1)
        config['data'] = {
            'name': _volminnet_workflow__GenericTabularAdapter.name,
            'root': 'unused',
            'num_classes': 4,
            'validation_size': 8,
        }
        config['volminnet']['model'] = {
            'name': 'feature_mlp',
            'input_dim': 6,
            'hidden_width': 8,
        }
        config['loader']['batch_size'] = 8
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(config, Path(directory) / 'run')
            final = json.loads((run_dir / 'final_metrics.json').read_text())
            self.assertEqual(len(final['learned_transition']), 4)

    def test_formal_config_matches_cifar10_paper_protocol(self) -> None:
        path = Path(__file__).resolve().parents[1] / 'configs/experiment/volminnet_cifar10_reproduction.yaml'
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        parsed = VolMinNetConfig.from_mapping(config)
        self.assertEqual(config['configuration_fidelity'], 'paper_protocol')
        self.assertEqual(config['volminnet']['model']['name'], 'resnet18')
        self.assertEqual(config['loader']['batch_size'], 128)
        self.assertEqual(config['trainer']['epochs'], 150)
        self.assertEqual(parsed.lambda_volume, 0.0001)
        self.assertEqual(parsed.classifier_scheduler['milestones'], [30, 60])

    def setUp(self) -> None:
        self.train = _volminnet_workflow__cifar(40, 'train')
        self.test = _volminnet_workflow__cifar(20, 'test')

    def _load(self, _root, split):
        return self.train if split == 'train' else self.test

    def test_fresh_resume_extension_and_completed_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load):
            run_dir = run_experiment(_volminnet_workflow__config(1), Path(directory) / 'run')
            first = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertTrue(first['algorithm']['volminnet_state']['completed'])
            self.assertEqual(first['algorithm']['volminnet_state']['global_step'], 2)
            self.assertIn('model', first['best_pair'])
            self.assertIn('transition', first['best_pair'])
            for name in ('resolved_config.yaml', 'environment.json', 'noise_manifest.npz', 'metrics.jsonl', 'final_metrics.json', 'last.pt', 'best.pt', 'transition_initial.npz', 'transition_best.npz', 'transition_last.npz'):
                self.assertTrue((run_dir / name).is_file(), name)
            manifest_hash = _volminnet_workflow__sha(run_dir / 'noise_manifest.npz')
            initial_hash = _volminnet_workflow__sha(run_dir / 'transition_initial.npz')
            run_experiment(_volminnet_workflow__config(2), resume=run_dir / 'last.pt')
            resumed = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(resumed['algorithm']['volminnet_state']['global_step'], 4)
            self.assertEqual(_volminnet_workflow__sha(run_dir / 'noise_manifest.npz'), manifest_hash)
            self.assertEqual(_volminnet_workflow__sha(run_dir / 'transition_initial.npz'), initial_hash)
            watched = {name: (_volminnet_workflow__sha(run_dir / name), (run_dir / name).stat().st_mtime_ns) for name in ('last.pt', 'best.pt', 'metrics.jsonl', 'final_metrics.json', 'noise_manifest.npz', 'transition_initial.npz', 'transition_best.npz', 'transition_last.npz')}
            run_experiment(_volminnet_workflow__config(2), resume=run_dir / 'last.pt')
            for name, expected in watched.items():
                self.assertEqual((_volminnet_workflow__sha(run_dir / name), (run_dir / name).stat().st_mtime_ns), expected)

    def test_uninterrupted_matches_epoch_boundary_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load):
            direct = run_experiment(_volminnet_workflow__config(2), Path(directory) / 'direct')
            resumed = run_experiment(_volminnet_workflow__config(1), Path(directory) / 'resumed')
            run_experiment(_volminnet_workflow__config(2), resume=resumed / 'last.pt')
            direct_state = torch.load(direct / 'last.pt', map_location='cpu', weights_only=False)['algorithm']
            resumed_state = torch.load(resumed / 'last.pt', map_location='cpu', weights_only=False)['algorithm']
            for owner in ('model', 'transition'):
                for key in direct_state[owner]:
                    self.assertTrue(torch.equal(direct_state[owner][key], resumed_state[owner][key]), f'{owner}.{key}')

    def test_resume_rejects_method_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load):
            run_dir = run_experiment(_volminnet_workflow__config(1), Path(directory) / 'run')
            changed = _volminnet_workflow__config(2)
            changed['volminnet']['objective']['volume']['coefficient'] = 0.01
            with self.assertRaisesRegex(ValueError, 'method settings'):
                run_experiment(changed, resume=run_dir / 'last.pt')

    def test_cifar100_lightweight_dispatch(self) -> None:
        train = _volminnet_workflow__cifar(200, 'train', 100)
        test = _volminnet_workflow__cifar(100, 'test', 100)
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar100', side_effect=lambda _root, split: train if split == 'train' else test):
            run_dir = run_experiment(_volminnet_workflow__config(1, 'cifar100'), Path(directory) / 'run')
            final = json.loads((run_dir / 'final_metrics.json').read_text())
            self.assertEqual(len(final['learned_transition']), 100)

    def test_test_metrics_do_not_select_best(self) -> None:
        config = _volminnet_workflow__config(1)
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._load), patch('lnl_toolbox.training.volminnet_experiment._evaluate_clean', return_value={'loss': 999.0, 'accuracy': 0.0, 'samples': 10.0}):
            run_dir = run_experiment(config, Path(directory) / 'run')
            payload = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
            self.assertEqual(payload['best_pair']['validation_loss'], payload['algorithm']['volminnet_state']['best_validation_loss'])
