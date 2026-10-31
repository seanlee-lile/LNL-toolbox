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

    def test_dynamic_centroid_loss_and_full_model_gradients_match_reference(self) -> None:
        class LearnedFeatures(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 3)
                self.classifier = nn.Linear(3, 1, bias=False)

            def forward_with_features(self, inputs):
                features = self.encoder(inputs).tanh()
                return FeatureOutput(self.classifier(features), features)

        inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.],
                               [0., -1.], [-1., -1.]], dtype=torch.float64)
        targets = torch.tensor([0, 1, 0, 1, 0, 1])
        for rho_positive, rho_negative in ((0., 0.), (0.3, 0.1)):
            with self.subTest(rates=(rho_positive, rho_negative)):
                torch.manual_seed(15)
                model = LearnedFeatures().double()
                output = model.forward_with_features(inputs)
                transition = np.array([[1-rho_negative, rho_negative], [rho_positive, 1-rho_positive]])
                artifact = CWDEstimator(transition).estimate(FeatureSnapshot(
                    output.features.detach().numpy(), targets.numpy(), np.arange(6), 'fixture', 'train'))
                kwargs = dict(model=model, logits=output.logits, features=output.features,
                    noisy_targets=targets, sample_indices=torch.arange(6), base_loss=CrossEntropyLoss(), metadata={})
                actual = CWDGlobalObjective(artifact, variant='binary_scalar', dynamic_centroid=True).compute(**kwargs)
                clean_positive = (targets.double().mean().item()-rho_negative)/(1-rho_positive-rho_negative)
                omega = 1/(1-2*clean_positive*rho_positive) + 1/(1-2*(1-clean_positive)*rho_negative) - 1
                margin = output.logits[:, 0]
                signs = targets.double()*2-1
                expected = 1 + margin.square().mean() - 2*omega*(margin*signs).mean()
                if rho_positive == rho_negative == 0:
                    torch.testing.assert_close(expected, (margin-signs).square().mean())
                torch.testing.assert_close(actual, expected)
                parameters = tuple(model.parameters())
                actual_grad = torch.autograd.grad(actual, parameters, retain_graph=True)
                expected_grad = torch.autograd.grad(expected, parameters, retain_graph=True)
                for got, reference in zip(actual_grad, expected_grad):
                    self.assertTrue(torch.isfinite(got).all())
                    torch.testing.assert_close(got, reference)
                self.assertGreater(actual_grad[0].abs().sum().item(), 0)
                self.assertGreater(actual_grad[-1].abs().sum().item(), 0)
                # A cached centroid has the same value at snapshot time but
                # omits the feature gradient of its label-dependent term.
                static = CWDGlobalObjective(artifact, variant='binary_scalar', dynamic_centroid=False).compute(**kwargs)
                torch.testing.assert_close(static, actual)
                static_grad = torch.autograd.grad(static, parameters)
                self.assertFalse(torch.allclose(static_grad[0], actual_grad[0]))

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
            for row in payload['metrics']:
                self.assertIn('test_loss', row)
                self.assertIn('test_accuracy', row)
                self.assertNotIn('validation_loss', row)
                self.assertNotIn('validation_accuracy', row)

    def test_interrupted_fold_resumes_same_plan_without_validation_metrics(self) -> None:
        import json
        from lnl_toolbox.training.checkpoint import atomic_save as save
        train = _cwd_training__cifar('train', 6)
        test = _cwd_training__cifar('test', 2)
        config = {'seed': 7, 'data': {'name': 'cifar10_airplane_automobile', 'root': 'unused', 'folds': 2,
            'fold_index': 0, 'augment': False}, 'noise': {'rho_positive': 0., 'rho_negative': 0., 'seed': 7},
            'loader': {'batch_size': 3, 'num_workers': 0, 'pin_memory': False}, 'model': {'name': 'tiny_cnn', 'width': 1},
            'optimizer': {'name': 'adam', 'lr': .001, 'weight_decay': 0.}, 'scheduler': {'milestones': [], 'gamma': .1},
            'trainer': {'epochs': 3, 'device': 'cpu'}, 'cwd': {'variant': 'binary_scalar', 'dynamic_centroid': True}}
        with tempfile.TemporaryDirectory() as directory, \
             patch('lnl_toolbox.data.sources.load_cifar10', side_effect=lambda _root, split: train if split == 'train' else test):
            root = Path(directory)
            uninterrupted = run_cwd_experiment(config, root/'complete')

            def interrupt(payload, path):
                save(payload, path)
                raise RuntimeError('test interruption')

            with patch('lnl_toolbox.training.cwd_experiment.atomic_save', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                    run_cwd_experiment(config, root/'interrupted')
            checkpoint = root/'interrupted'/'last.pt'
            before = torch.load(checkpoint, weights_only=False)
            self.assertEqual(before['completed_epoch'], 0)
            run_cwd_experiment(config, resume=checkpoint)
            resumed = torch.load(checkpoint, weights_only=False)
            complete = torch.load(uninterrupted/'last.pt', weights_only=False)
            self.assertEqual(resumed['completed_epoch'], config['trainer']['epochs'] - 1)
            self.assertEqual([row['epoch'] for row in resumed['metrics']], list(range(1, config['trainer']['epochs'] + 1)))
            for name, value in complete['model'].items():
                torch.testing.assert_close(resumed['model'][name], value, rtol=0, atol=0)
            def training_metrics(rows):
                return [{key: value for key, value in row.items() if key != 'epoch_seconds'} for row in rows]

            self.assertEqual(training_metrics(resumed['metrics']), training_metrics(complete['metrics']))
            for row in resumed['metrics'] + complete['metrics']:
                self.assertGreaterEqual(row['epoch_seconds'], 0.)
                self.assertTrue(np.isfinite(row['epoch_seconds']))
            for name, state in complete['loader_rng_state'].items():
                torch.testing.assert_close(resumed['loader_rng_state'][name], state, rtol=0, atol=0)
            rows = [json.loads(line) for line in (checkpoint.parent/'metrics.jsonl').read_text().splitlines()]
            for row in rows:
                self.assertTrue(np.isfinite(row['test_loss']))
                self.assertFalse(any(key.startswith('validation_') for key in row))
            # A legacy checkpoint cannot silently restart the sample order.
            before.pop('loader_rng_state')
            save(before, checkpoint)
            with self.assertRaisesRegex(ValueError, 'data-loader RNG states'):
                run_cwd_experiment(config, resume=checkpoint)


class CWDAdamBeta1ScheduleTest(unittest.TestCase):
    """Optimizer-only and mocked lifecycle checks; no model training."""

    def _apply(self, optimizer, epoch, enabled=False):
        from lnl_toolbox.training.cwd_experiment import _apply_cwd_beta1_schedule
        _apply_cwd_beta1_schedule(optimizer, epoch, enabled=enabled)

    def _optimizer(self):
        return torch.optim.Adam(nn.Linear(2, 1).parameters(), lr=.001)

    def test_disabled_and_default_keep_beta1_point_nine(self):
        optimizer = self._optimizer()
        for epoch in range(200):
            self._apply(optimizer, epoch, False)
            self.assertEqual(optimizer.param_groups[0]['betas'], (.9, .999))
        self._apply(optimizer, 100)
        self.assertEqual(optimizer.param_groups[0]['betas'][0], .9)

    def test_enabled_epoch_79_is_before_switch(self):
        optimizer = self._optimizer()
        for epoch in (0, 1, 78, 79):
            self._apply(optimizer, epoch, True)
            self.assertEqual(optimizer.param_groups[0]['betas'][0], .9)

    def test_enabled_epoch_80_and_later_match_reference(self):
        optimizer = self._optimizer()
        for epoch in (80, 81, 199):
            self._apply(optimizer, epoch, True)
            self.assertEqual(optimizer.param_groups[0]['betas'][0], .1)

    def test_preserves_beta2_and_group_identity(self):
        first, second = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([
            {'params': [first], 'betas': (.9, .95)},
            {'params': [second], 'betas': (.9, .98)},
        ], lr=.001)
        groups = tuple(optimizer.param_groups)
        for epoch in (79, 80, 81):
            self._apply(optimizer, epoch, True)
            for group, original, beta2 in zip(optimizer.param_groups, groups, (.95, .98)):
                self.assertIs(group, original)
                self.assertEqual(group['betas'][1], beta2)

    def test_existing_lr_scheduler_sequences_unchanged(self):
        from lnl_toolbox.training.experiment import build_scheduler
        for config in (
            {'name': 'multistep', 'milestones': [40, 120], 'gamma': .1},
            {'name': 'linear_after', 'start_epoch': 80, 'end_epoch': 200},
        ):
            with self.subTest(scheduler=config['name']):
                original, scheduled = self._optimizer(), self._optimizer()
                before = build_scheduler(original, config, 200)
                after = build_scheduler(scheduled, config, 200)
                for epoch in range(200):
                    self._apply(scheduled, epoch, True)
                    self.assertEqual(original.param_groups[0]['lr'], scheduled.param_groups[0]['lr'])
                    # No gradients: step only establishes the normal scheduler order.
                    original.step()
                    scheduled.step()
                    before.step()
                    after.step()
                self.assertEqual(before.state_dict(), after.state_dict())

    def test_checkpoint_roundtrip_preserves_betas_and_optimizer_state(self):
        from lnl_toolbox.training.checkpoint import atomic_save, read_checkpoint
        optimizer = self._optimizer()
        parameter = optimizer.param_groups[0]['params'][0]
        optimizer.state[parameter] = {
            'step': torch.tensor(7.), 'exp_avg': torch.ones_like(parameter),
            'exp_avg_sq': torch.full_like(parameter, .5),
        }
        optimizer.param_groups[0]['betas'] = (.9, .97)
        self._apply(optimizer, 80, True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'last.pt'
            atomic_save({'optimizer': optimizer.state_dict(), 'completed_epoch': 80}, checkpoint)
            payload = read_checkpoint(checkpoint, 'cpu')
            restored = self._optimizer()
            restored.load_state_dict(payload['optimizer'])
            self.assertEqual(restored.param_groups[0]['betas'], (.1, .97))
            self._apply(restored, 81, True)
            self.assertEqual(restored.param_groups[0]['betas'], (.1, .97))
            # Disabled means no mutation of restored optimizer state either.
            self._apply(restored, 82, False)
            self.assertEqual(restored.param_groups[0]['betas'], (.1, .97))
            restored_parameter = restored.param_groups[0]['params'][0]
            for key, value in optimizer.state[parameter].items():
                torch.testing.assert_close(restored.state[restored_parameter][key], value)

    def test_yaml_parser_and_catalog_preserve_enabled_flag(self):
        import yaml
        from lnl_toolbox.catalog import load_yaml, validate_config
        config = load_yaml(Path(__file__).resolve().parents[1] / 'configs/experiment/cwd_cifar10_reproduction.yaml')
        config['optimizer']['beta1_schedule'] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            parsed = load_yaml(path, check_parameters=True)
            self.assertEqual(validate_config(parsed, check_data=False).name, 'cwd')
            self.assertIs(parsed['optimizer']['beta1_schedule'], True)

    def test_mocked_resume_applies_schedule_before_epoch_and_logs_it(self):
        import io
        import json
        from contextlib import redirect_stdout
        from types import SimpleNamespace
        from unittest.mock import Mock
        from lnl_toolbox.data import DataRole
        from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state

        model = nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        config = {
            'seed': 7, 'data': {'name': 'mocked_binary'}, 'noise': {},
            'model': {'name': 'tiny_cnn'},
            'optimizer': {'name': 'adam', 'lr': .001, 'beta1_schedule': True},
            'scheduler': {'name': 'none'}, 'trainer': {'epochs': 82, 'device': 'cpu'},
            'cwd': {'variant': 'binary_scalar', 'dynamic_centroid': True},
        }
        loaders = {role: SimpleNamespace(generator=torch.Generator().manual_seed(7))
                   for role in (DataRole.TRAIN, DataRole.TRAIN_EVAL, DataRole.TEST)}
        prepared = SimpleNamespace(
            loader_config={}, num_classes=2, manifest=SimpleNamespace(mapping_hash='fixture', metadata={}),
            manifest_path=Path('unused.npz'), available_roles=set(), dataset='mocked_binary',
            loader=lambda role, **kwargs: loaders[role],
        )
        snapshot, artifact = Mock(), Mock()
        snapshot.snapshot_hash, artifact.artifact_hash = 'snapshot', 'statistic'
        observed = []

        def observe_train(current_model, current_optimizer, *args):
            self.assertIs(current_model, model)
            observed.append((current_optimizer, current_optimizer.param_groups[0]['betas']))
            return 1., .5

        def interrupt_after_switch(payload, path):
            atomic_save(payload, path)
            if payload['completed_epoch'] == 80:
                raise RuntimeError('interrupted after beta1 switch')

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'last.pt'
            atomic_save({
                'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'completed_epoch': 78, 'metrics': [], 'rng_state': capture_rng_state(),
                'noise_mapping_hash': 'fixture',
                'loader_rng_state': {name: loaders[role].generator.get_state() for name, role in (
                    ('train', DataRole.TRAIN), ('snapshot', DataRole.TRAIN_EVAL), ('test', DataRole.TEST))},
            }, checkpoint)
            owner = 'lnl_toolbox.training.cwd_experiment.'
            output = io.StringIO()
            with patch(owner + 'prepare_experiment_data', return_value=prepared), \
                 patch(owner + '_build_model', return_value=model), \
                 patch(owner + 'collect_feature_snapshot', return_value=snapshot), \
                 patch(owner + 'CWDEstimator') as estimator, \
                 patch(owner + '_train_epoch', side_effect=observe_train), \
                 patch(owner + '_evaluate_cwd', return_value={'loss': 1., 'accuracy': .5}), \
                 patch(owner + 'write_training_curves_svg'), redirect_stdout(output):
                estimator.return_value.estimate.return_value = artifact
                with patch(owner + 'atomic_save', side_effect=interrupt_after_switch):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted after beta1 switch'):
                        run_cwd_experiment(config, resume=checkpoint, requirements=object())
                interrupted = torch.load(checkpoint, weights_only=False)
                self.assertEqual(interrupted['optimizer']['param_groups'][0]['betas'], (.1, .999))
                # Resume the unfinished run after the switch, without training.
                run_cwd_experiment(config, resume=checkpoint, requirements=object())
                # A completed resume still restores optimizer state without another epoch.
                run_cwd_experiment(config, resume=checkpoint, requirements=object())
            self.assertEqual([betas[0] for _, betas in observed], [.9, .1, .1])
            self.assertIs(observed[0][0], observed[1][0])
            self.assertIsNot(observed[1][0], observed[2][0])
            payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(payload['optimizer']['param_groups'][0]['betas'], (.1, .999))
            self.assertEqual([row['adam_beta1'] for row in payload['metrics']], [.9, .1, .1])
            rows = [json.loads(line) for line in (Path(directory) / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['adam_beta1'] for row in rows], [.9, .1, .1])
            self.assertEqual([json.loads(line)['adam_beta1'] for line in output.getvalue().splitlines()], [.9, .1, .1])
