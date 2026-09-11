"""Merged unit tests; source modules were consolidated without changing assertions."""

# --- merged from test_cwd.py ---
import unittest

# --- merged from test_cwd.py ---
from pathlib import Path

# --- merged from test_cwd.py ---
import numpy as np

# --- merged from test_cwd.py ---
import torch

# --- merged from test_cwd.py ---
from torch import nn

# --- merged from test_cwd.py ---
from torch.utils.data import DataLoader

# --- merged from test_cwd.py ---
from lnl_toolbox.algorithms.cwd import CWDUnbiasedRisk

# --- merged from test_cwd.py ---
from lnl_toolbox.algorithms.cwd import CWDGlobalObjective

# --- merged from test_cwd.py ---
from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm

# --- merged from test_cwd.py ---
from lnl_toolbox.core import Batch, ExperimentContext, RunState

# --- merged from test_cwd.py ---
from lnl_toolbox.estimators.cwd import CWDEstimator

# --- merged from test_cwd.py ---
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

# --- merged from test_cwd.py ---
from lnl_toolbox.training.snapshots import FeatureSnapshot

# --- merged from test_cwd.py ---
from lnl_toolbox.models.feature_output import FeatureOutput

# --- merged from test_cwd.py ---
from lnl_toolbox.models.cifar_resnet import cifar_resnet34

# --- merged from test_cwd.py ---
from lnl_toolbox.training.pipeline import StandardNoisyERMPipeline

# --- merged from test_cwd.py ---
class _cwd__FeatureModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward_with_features(self, inputs: torch.Tensor) -> FeatureOutput:
        return FeatureOutput(self.classifier(inputs), inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs).logits

# --- merged from test_cwd.py ---
class _cwd__ScalarFeatureModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 1)

    def forward_with_features(self, inputs: torch.Tensor) -> FeatureOutput:
        return FeatureOutput(self.classifier(inputs), inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs).logits

# --- merged from test_cwd.py ---
class _cwd_CWDTest(unittest.TestCase):

    def test_estimator_builds_hashed_classwise_artifact(self) -> None:
        snapshot = FeatureSnapshot(np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]), np.asarray([0, 1, 0, 1]), np.arange(4), 'fixture', 'train')
        artifact = CWDEstimator().estimate(snapshot)
        self.assertEqual(artifact.values.shape, (2, 2))
        self.assertTrue(artifact.artifact_hash)
        np.testing.assert_allclose(artifact.values, np.asarray([[0.5, 0.25], [0.0, 0.25]]), atol=1e-10)
        self.assertEqual(artifact.metadata['cwd_equations'], '19,21-30')
        self.assertEqual(len(artifact.metadata['coefficient_matrices']), 2)

    def test_multiclass_identity_recovers_empirical_centroid(self) -> None:
        features = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
        snapshot = FeatureSnapshot(features, np.asarray([0, 1, 2]), np.arange(3), 'fixture', 'train')
        artifact = CWDEstimator(np.eye(3)).estimate(snapshot)
        np.testing.assert_allclose(artifact.values, features / 3.0, atol=1e-10)
        np.testing.assert_allclose(artifact.metadata['class_prior'], np.ones(3) / 3)

    def test_binary_estimator_matches_paper_eq15(self) -> None:
        features = np.arange(20, dtype=np.float64).reshape(10, 2)
        noisy = np.asarray([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
        transition = np.asarray([[0.9, 0.1], [0.3, 0.7]])
        artifact = CWDEstimator(transition).estimate(FeatureSnapshot(features, noisy, np.arange(10), 'fixture', 'train'))
        observed_prior = noisy.mean()
        rho_positive = transition[1, 0]
        rho_negative = transition[0, 1]
        clean_positive = (observed_prior - rho_negative) / (1.0 - rho_positive - rho_negative)
        clean_negative = 1.0 - clean_positive
        omega = 1.0 / (1.0 - 2.0 * clean_positive * rho_positive) + 1.0 / (1.0 - 2.0 * clean_negative * rho_negative) - 1.0
        observed_centroid = (features[noisy == 1].sum(axis=0) - features[noisy == 0].sum(axis=0)) / len(features)
        np.testing.assert_allclose(artifact.values[1] - artifact.values[0], omega * observed_centroid, atol=1e-10)
        self.assertIsNone(artifact.metadata['pinv_rcond'])

    def test_cifar_resnet_cwd_can_disable_classifier_bias(self) -> None:
        model = cifar_resnet34(1, base_width=4, bias=False)
        self.assertIsNone(model.classifier.bias)

    def test_cwd_risk_is_differentiable(self) -> None:
        logits = torch.randn(3, 2, requires_grad=True)
        values = CWDUnbiasedRisk().per_sample_risk(logits=logits, noisy_targets=torch.tensor([0, 1, 0]), base_loss=CrossEntropyLoss())
        values.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_global_objective_matches_identity_artifact_and_is_differentiable(self) -> None:
        snapshot = FeatureSnapshot(np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]), np.asarray([0, 1, 0, 1]), np.arange(4), 'fixture', 'train')
        artifact = CWDEstimator().estimate(snapshot)
        model = _cwd__FeatureModel()
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        output = model.forward_with_features(features)
        value = CWDGlobalObjective(artifact).compute(model=model, logits=output.logits, features=output.features, noisy_targets=torch.tensor([0, 1]), sample_indices=torch.tensor([0, 1]), base_loss=CrossEntropyLoss(), metadata={})
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())

    def test_binary_margin_objective_matches_signed_centroid_risk(self) -> None:
        snapshot = FeatureSnapshot(np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([0, 1]), np.arange(2), 'fixture', 'train')
        artifact = CWDEstimator(np.eye(2)).estimate(snapshot)
        model = _cwd__FeatureModel()
        with torch.no_grad():
            model.classifier.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 5.0]]))
            model.classifier.bias.zero_()
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        output = model.forward_with_features(features)
        value = CWDGlobalObjective(artifact, variant='binary_margin').compute(model=model, logits=output.logits, features=output.features, noisy_targets=torch.tensor([0, 1]), sample_indices=torch.tensor([0, 1]), base_loss=CrossEntropyLoss(), metadata={})
        expected = torch.tensor(1.0) + torch.tensor([2.0, 3.0]).square().mean() - 2.0 * torch.dot(torch.tensor([2.0, 3.0]), torch.tensor([-0.5, 0.5]))
        torch.testing.assert_close(value.detach(), expected)

    def test_binary_scalar_objective_matches_paper_squared_risk(self) -> None:
        snapshot = FeatureSnapshot(np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([0, 1]), np.arange(2), 'fixture', 'train')
        artifact = CWDEstimator(np.eye(2)).estimate(snapshot)
        model = _cwd__ScalarFeatureModel()
        with torch.no_grad():
            model.classifier.weight.copy_(torch.tensor([[2.0, 3.0]]))
            model.classifier.bias.zero_()
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        weight = model.classifier.weight
        margin = features @ weight[0]
        signed_centroid = torch.tensor([-0.5, 0.5])
        expected = torch.tensor(1.0) + margin.square().mean() - 2.0 * torch.dot(weight[0], signed_centroid)
        value = CWDGlobalObjective(artifact, variant='binary_scalar').compute(model=model, logits=model(features), features=features, noisy_targets=torch.tensor([0, 1]), sample_indices=torch.tensor([0, 1]), base_loss=CrossEntropyLoss(), metadata={})
        torch.testing.assert_close(value.detach(), expected)

    def test_dynamic_centroid_matches_static_identity_case(self) -> None:
        snapshot = FeatureSnapshot(np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([0, 1]), np.arange(2), 'fixture', 'train')
        artifact = CWDEstimator(np.eye(2)).estimate(snapshot)
        model = _cwd__ScalarFeatureModel()
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        output = model.forward_with_features(features)
        kwargs = dict(model=model, logits=output.logits, features=features, noisy_targets=torch.tensor([0, 1]), sample_indices=torch.tensor([0, 1]), base_loss=CrossEntropyLoss(), metadata={})
        static = CWDGlobalObjective(artifact, variant='binary_scalar', dynamic_centroid=False).compute(**kwargs)
        dynamic = CWDGlobalObjective(artifact, variant='binary_scalar', dynamic_centroid=True).compute(**kwargs)
        torch.testing.assert_close(static.detach(), dynamic.detach())

    def test_pipeline_prepares_feature_statistic_and_binds_consumer(self) -> None:
        model = _cwd__FeatureModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = DataLoader([{'input': torch.tensor([1.0, 0.0]), 'target': torch.tensor(0), 'index': torch.tensor(0)}, {'input': torch.tensor([0.0, 1.0]), 'target': torch.tensor(1), 'index': torch.tensor(1)}], batch_size=2)
        pipeline = StandardNoisyERMPipeline.from_config({'statistic_estimator': {'name': 'cwd'}, 'objective_consumer': {'name': 'cwd'}})
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            artifacts = pipeline.prepare(model=model, optimizer=optimizer, loader=loader, device=torch.device('cpu'), dataset='fixture', split='train', run_dir=directory)
            self.assertIsNotNone(artifacts.feature_snapshot)
            self.assertIsNotNone(artifacts.statistic)
            self.assertIs(pipeline.objective_consumer.statistic, artifacts.statistic)
            algorithm = SupervisedClassificationAlgorithm(model, optimizer, CrossEntropyLoss(), torch.device('cpu'), objective_consumer=pipeline.objective_consumer)
            algorithm.setup(ExperimentContext(Path(directory)))
            result = algorithm.step(Batch({'input': torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 'target': torch.tensor([0, 1]), 'index': torch.tensor([0, 1])}), RunState(phase='train'))
            self.assertTrue(np.isfinite(result.metrics['loss']))
            checkpoint_state = pipeline.state_dict()
            restored = StandardNoisyERMPipeline.from_config({'statistic_estimator': {'name': 'cwd'}, 'objective_consumer': {'name': 'cwd'}})
            restored.restore_for_resume(directory, checkpoint_state=checkpoint_state, component_states=pipeline.component_state_dict(), dataset='fixture', split='train')
            self.assertEqual(restored.artifacts.statistic.artifact_hash, artifacts.statistic.artifact_hash)

    def test_singular_flip_matrix_fails_explicitly(self) -> None:
        snapshot = FeatureSnapshot(np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([0, 1]), np.arange(2), 'fixture', 'train')
        with self.assertRaisesRegex(ValueError, 'identifiable'):
            CWDEstimator(label_flip_matrix=np.ones((2, 2)) / 2).estimate(snapshot)

# --- merged from test_cwd_training.py ---
import tempfile

# --- merged from test_cwd_training.py ---
import unittest

# --- merged from test_cwd_training.py ---
from unittest.mock import patch

# --- merged from test_cwd_training.py ---
import numpy as np

# --- merged from test_cwd_training.py ---
import torch

# --- merged from test_cwd_training.py ---
from lnl_toolbox.data.cifar import CifarData

# --- merged from test_cwd_training.py ---
from lnl_toolbox.training.cwd_experiment import run_cwd_experiment

# --- merged from test_cwd_training.py ---
def _cwd_training__cifar(split: str, samples_per_class: int) -> CifarData:
    labels = np.repeat(np.asarray([0, 1], dtype=np.int64), samples_per_class)
    images = np.zeros((labels.size, 32, 32, 3), dtype=np.uint8)
    images[:, 0, 0, 0] = labels.astype(np.uint8)
    return CifarData(images, labels, tuple(map(str, range(10))), split, 'cifar10')

# --- merged from test_cwd_training.py ---
class _cwd_training_CWDTrainingTest(unittest.TestCase):

    def test_one_fold_writes_artifacts_and_resumable_checkpoint(self) -> None:
        train = _cwd_training__cifar('train', 6)
        test = _cwd_training__cifar('test', 2)
        config = {'seed': 7, 'data': {'name': 'cifar10_airplane_automobile', 'root': 'unused', 'folds': 2, 'fold_index': 0, 'augment': False}, 'noise': {'rho_positive': 0.0, 'rho_negative': 0.0, 'seed': 7}, 'loader': {'batch_size': 8, 'num_workers': 0, 'pin_memory': False}, 'model': {'name': 'tiny_cnn', 'width': 1}, 'optimizer': {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0}, 'scheduler': {'milestones': [], 'gamma': 0.1}, 'trainer': {'epochs': 1, 'device': 'cpu'}, 'cwd': {'ridge': 1e-08}}
        with tempfile.TemporaryDirectory() as directory:
            with patch('lnl_toolbox.data.sources.load_cifar10', side_effect=lambda _root, split: train if split == 'train' else test):
                result = run_cwd_experiment(config, directory)
            for name in ('last.pt', 'noise_manifest.npz', 'feature_snapshot.npz', 'statistic_artifact.npz', 'metrics.jsonl', 'training_curves.svg', 'resolved_config.yaml'):
                self.assertTrue((result / name).is_file(), name)
            payload = torch.load(result / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(payload['completed_epoch'], 0)
            self.assertTrue(payload['statistic_hash'])
            self.assertTrue(payload['feature_snapshot_hash'])
