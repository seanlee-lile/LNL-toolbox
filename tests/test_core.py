"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_core.py ---
import tempfile

# --- merged from test_core.py ---
import unittest

# --- merged from test_core.py ---
from pathlib import Path

# --- merged from test_core.py ---
from lnl_toolbox.core import Batch, ExperimentContext, StepResult

# --- merged from test_core.py ---
from lnl_toolbox.engine import run_cycles

# --- merged from test_core.py ---
class _core_DummyAlgorithm:

    def __init__(self) -> None:
        self.closed = False
        self.total = 0

    def setup(self, context: ExperimentContext) -> None:
        self.context = context

    def on_run_start(self, state) -> None:
        state.metadata['started'] = True

    def on_cycle_start(self, state) -> None:
        pass

    def step(self, batch: Batch, state) -> StepResult:
        self.total += batch.payload
        return StepResult(outputs=self.total, metrics={'last_total': float(self.total)})

    def on_cycle_end(self, state) -> StepResult:
        return StepResult(metadata={'cycle': state.cycle})

    def on_run_end(self, state) -> StepResult:
        return StepResult(metrics={'final_total': float(self.total)})

    def state_dict(self):
        return {'total': self.total}

    def load_state_dict(self, state) -> None:
        self.total = state['total']

    def close(self) -> None:
        self.closed = True

# --- merged from test_core.py ---
class _core_CoreLifecycleTest(unittest.TestCase):

    def test_runner_only_depends_on_generic_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = ExperimentContext(Path(directory), config={'demo': True})
            algorithm = _core_DummyAlgorithm()
            state = run_cycles(algorithm, [Batch(2), Batch(3)], 2, context)
        self.assertEqual(state.step, 4)
        self.assertEqual(state.metrics['final_total'], 10.0)
        self.assertTrue(state.metadata['started'])
        self.assertTrue(algorithm.closed)

# --- merged from test_foundations.py ---
from pathlib import Path

# --- merged from test_foundations.py ---
import tempfile

# --- merged from test_foundations.py ---
import unittest

# --- merged from test_foundations.py ---
import numpy as np

# --- merged from test_foundations.py ---
import torch

# --- merged from test_foundations.py ---
from lnl_toolbox.algorithms.multi_model import SmallLossPeerExchange, consistency_loss

# --- merged from test_foundations.py ---
from lnl_toolbox.core import CandidateLabelResult, ComplementaryLabelResult, PseudoLabelResult, SoftTargetResult

# --- merged from test_foundations.py ---
from lnl_toolbox.data import NeighborGraphArtifact, SemiSupervisedBatch

# --- merged from test_foundations.py ---
from lnl_toolbox.noise import TrainableTransitionModel

# --- merged from test_foundations.py ---
from lnl_toolbox.noise.statistics import StatisticArtifact

# --- merged from test_foundations.py ---
from lnl_toolbox.selectors.history import LossHistory

# --- merged from test_foundations.py ---
from lnl_toolbox.training.artifacts import ArtifactStore

# --- merged from test_foundations.py ---
from lnl_toolbox.training.early_stopping import EarlyStopping

# --- merged from test_foundations.py ---
from lnl_toolbox.training.snapshots import FeatureSnapshot

# --- merged from test_foundations.py ---
class _foundations_FoundationTest(unittest.TestCase):

    def test_early_stopping_roundtrip(self) -> None:
        stopping = EarlyStopping(patience=1)
        self.assertTrue(stopping.update({'selection_accuracy': 0.5}))
        self.assertFalse(stopping.update({'selection_accuracy': 0.4}))
        self.assertFalse(stopping.update({'selection_accuracy': 0.3}))
        self.assertTrue(stopping.stopped)
        restored = EarlyStopping(patience=1)
        restored.load_state_dict(stopping.state_dict())
        self.assertEqual(restored.bad_epochs, stopping.bad_epochs)

    def test_history_is_index_aligned(self) -> None:
        history = LossHistory(default=-1.0)
        history.update(torch.tensor([7, 2]), torch.tensor([0.7, 0.2]))
        values = history.lookup(torch.tensor([2, 9, 7]))
        self.assertTrue(torch.allclose(values, torch.tensor([0.2, -1.0, 0.7])))

    def test_target_and_semisupervised_contracts(self) -> None:
        result = SoftTargetResult(targets=torch.tensor([[1.0, 0.0], [0.25, 0.75]]), sample_indices=torch.tensor([3, 5]))
        self.assertEqual(result.targets.shape, (2, 2))
        batch = SemiSupervisedBatch(labeled={'input': torch.ones(1, 2), 'index': torch.tensor([1])}, unlabeled={'input': torch.ones(2, 2), 'index': torch.tensor([2, 4])})
        self.assertEqual(batch.unlabeled_indices.tolist(), [2, 4])

    def test_target_results_execute_runtime_tensor_validation(self) -> None:
        soft = SoftTargetResult(targets=torch.tensor([[0.75, 0.25], [0.1, 0.9]]), sample_indices=torch.tensor([7, 2]), confidence=torch.tensor([0.8, 0.7]), selected_mask=torch.tensor([True, False]))
        pseudo = PseudoLabelResult(labels=torch.tensor([1, 0]), confidence=torch.tensor([0.9, 0.6]), selected_mask=torch.tensor([True, True]), sample_indices=torch.tensor([7, 2]))
        candidates = CandidateLabelResult(candidates=torch.tensor([[True, False], [True, True]]), sample_indices=torch.tensor([7, 2]))
        self.assertEqual(tuple(soft.targets.shape), (2, 2))
        self.assertEqual(pseudo.labels.tolist(), [1, 0])
        self.assertEqual(candidates.candidates.dtype, torch.bool)

    def test_target_results_reject_invalid_probabilities_and_confidence(self) -> None:
        indices = torch.tensor([7, 2])
        invalid_targets = (torch.tensor([[0.8, 0.3], [0.1, 0.9]]), torch.tensor([[1.1, -0.1], [0.1, 0.9]]), torch.tensor([[float('nan'), 0.0], [0.1, 0.9]]), torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.int64))
        for targets in invalid_targets:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                SoftTargetResult(targets=targets, sample_indices=indices)
        with self.assertRaisesRegex(ValueError, 'confidence.*detached'):
            SoftTargetResult(targets=torch.tensor([[0.8, 0.2], [0.1, 0.9]]), sample_indices=indices, confidence=torch.tensor([0.8, 0.7], requires_grad=True))
        with self.assertRaisesRegex(ValueError, 'confidence.*finite'):
            PseudoLabelResult(labels=torch.tensor([0, 1]), confidence=torch.tensor([0.8, float('inf')]), selected_mask=torch.tensor([True, True]), sample_indices=indices)

    def test_target_results_reject_invalid_indices_masks_and_candidates(self) -> None:
        targets = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        with self.assertRaisesRegex(ValueError, 'integer dtype'):
            SoftTargetResult(targets=targets, sample_indices=torch.tensor([1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, 'unique'):
            SoftTargetResult(targets=targets, sample_indices=torch.tensor([4, 4]))
        with self.assertRaisesRegex(ValueError, 'selected_mask'):
            SoftTargetResult(targets=targets, sample_indices=torch.tensor([4, 5]), selected_mask=torch.tensor([1, 0]))
        with self.assertRaisesRegex(ValueError, 'at least one candidate'):
            CandidateLabelResult(candidates=torch.tensor([[False, False], [True, False]]), sample_indices=torch.tensor([4, 5]))
        with self.assertRaisesRegex(ValueError, 'at least one complementary'):
            ComplementaryLabelResult(negatives=torch.tensor([[False, False], [False, True]]), sample_indices=torch.tensor([4, 5]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_target_results_reject_cross_device_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, 'device'):
            SoftTargetResult(targets=torch.tensor([[0.8, 0.2], [0.1, 0.9]], device='cuda'), sample_indices=torch.tensor([4, 5]))

    def test_peer_exchange_and_consistency(self) -> None:
        losses = {'a': torch.tensor([0.1, 0.5]), 'b': torch.tensor([0.4, 0.2])}
        result = SmallLossPeerExchange(0.5).exchange(losses, torch.tensor([4, 9]))
        self.assertEqual(int(result.masks['a'].sum()), 1)
        self.assertTrue(torch.isfinite(consistency_loss(torch.randn(2, 3), torch.randn(2, 3))))

    def test_transition_and_artifact_roundtrips(self) -> None:
        transition = TrainableTransitionModel(3)
        self.assertTrue(torch.allclose(transition().sum(dim=1), torch.ones(3)))
        statistic = StatisticArtifact(np.eye(2), 'test')
        graph = NeighborGraphArtifact(np.array([[0, 1], [1, 0]]), np.zeros((2, 2)), np.array([4, 9]))
        snapshot = FeatureSnapshot(np.ones((2, 3)), np.array([0, 1]), np.array([9, 4]), 'x', 'train')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            statistic.save(root / 'stats.npz')
            graph.save(root / 'graph.npz')
            snapshot.save(root / 'features.npz')
            self.assertEqual(StatisticArtifact.load(root / 'stats.npz').artifact_hash, statistic.artifact_hash)
            self.assertEqual(NeighborGraphArtifact.load(root / 'graph.npz').artifact_hash, graph.artifact_hash)
            self.assertEqual(FeatureSnapshot.load(root / 'features.npz').snapshot_hash, snapshot.snapshot_hash)
            ref = ArtifactStore(root).write('stats', statistic)
            self.assertEqual(ref.artifact_hash, statistic.artifact_hash)
