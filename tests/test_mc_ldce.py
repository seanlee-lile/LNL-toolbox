"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_mc_ldce.py ---
import unittest

# --- merged from test_mc_ldce.py ---
import math

# --- merged from test_mc_ldce.py ---
from pathlib import Path

# --- merged from test_mc_ldce.py ---
import tempfile

# --- merged from test_mc_ldce.py ---
import numpy as np

# --- merged from test_mc_ldce.py ---
import torch

# --- merged from test_mc_ldce.py ---
from torch import nn

# --- merged from test_mc_ldce.py ---
from lnl_toolbox.algorithms.mc_ldce import MCLDCEObjective

# --- merged from test_mc_ldce.py ---
from lnl_toolbox.algorithms.pcse.volmin import PaperVolMinTransition, paper_volmin_objective

# --- merged from test_mc_ldce.py ---
from lnl_toolbox.estimators.mc_ldce import MCLDCEEstimator, build_label_imputation_matrix, estimate_clean_prior

# --- merged from test_mc_ldce.py ---
from lnl_toolbox.noise.transition import TransitionArtifact

# --- merged from test_mc_ldce.py ---
from lnl_toolbox.models.feature_output import forward_with_features

# --- merged from test_mc_ldce.py ---
from lnl_toolbox.models.mc_ldce_cnn import MCLDCECifarCNN

# --- merged from test_mc_ldce.py ---
from lnl_toolbox.training.mc_ldce_experiment import _prepare_classifier, _write_epoch_metrics

# --- merged from test_mc_ldce.py ---
from lnl_toolbox.training.snapshots import FeatureSnapshot

# --- merged from test_mc_ldce.py ---
class _mc_ldce__LinearFeatures(nn.Module):

    def __init__(self, *, bias: bool=False) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2, bias=bias)

    def forward(self, value):
        return self.classifier(value)

# --- merged from test_mc_ldce.py ---
class _mc_ldce_MCLDCETest(unittest.TestCase):

    def test_epoch_metrics_writer_rewrites_complete_history(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.jsonl'
            _write_epoch_metrics([{'event': 'epoch', 'epoch': 1}], path)
            _write_epoch_metrics(
                [{'event': 'epoch', 'epoch': 1}, {'event': 'epoch', 'epoch': 2}],
                path,
            )
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual([row['epoch'] for row in rows], [1, 2])


    def test_mc_uses_paper_sigmoid_off_diagonal_volmin_parameterization(self) -> None:
        transition_model = PaperVolMinTransition(3, initial_weight=-2.0, seed=1)
        matrix = transition_model.matrix().detach().numpy()
        off_diagonal = 1.0 / (1.0 + np.exp(2.0))
        expected = np.full((3, 3), off_diagonal)
        np.fill_diagonal(expected, 1.0)
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(matrix, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        self.assertTrue((np.diag(matrix)[:, None] > matrix - np.diag(np.diag(matrix))).all())

    def test_paper_volmin_objective_is_differentiable(self) -> None:
        transition_model = PaperVolMinTransition(3, initial_weight=-2.0, seed=2)
        logits = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 1])
        objective, metrics = paper_volmin_objective(logits, targets, transition_model.matrix(), lambda_volume=0.0001, determinant_tolerance=1e-08, condition_limit=100000000.0)
        objective.backward()
        self.assertTrue(torch.isfinite(objective))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertIn('volume_regularizer', metrics)
        self.assertTrue(transition_model.off_diagonal_logits.grad is not None)

    def test_paper_cnn_exposes_128_dimensional_features(self) -> None:
        output = forward_with_features(MCLDCECifarCNN(10), torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output.logits.shape), (2, 10))
        self.assertEqual(tuple(output.features.shape), (2, 128))

    def test_joint_stage_keeps_convolutions_trainable_and_dropout_active(self) -> None:
        torch.manual_seed(3)
        model = MCLDCECifarCNN(10)
        inputs = torch.randn(2, 3, 32, 32)
        model.eval()
        before = forward_with_features(model, inputs).features.detach().clone()
        _prepare_classifier(model)
        model.train()
        after = forward_with_features(model, inputs).features.detach()
        self.assertIsNone(model.classifier.bias)
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(model.dropout.training)
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertIn('convolutions.0.weight', trainable)
        self.assertIn('classifier.weight', trainable)


    def test_objective_rejects_classifier_bias(self) -> None:
        snapshot = FeatureSnapshot(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]), np.array([0, 1]), 'fixture', 'train')
        artifact = MCLDCEEstimator().estimate(snapshot, TransitionArtifact(np.eye(2), 'known'))
        model = _mc_ldce__LinearFeatures(bias=True)
        features = torch.tensor(snapshot.features, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, 'bias-free'):
            MCLDCEObjective(artifact).compute(model=model, logits=model(features), features=features, noisy_targets=torch.tensor([0, 1]), sample_indices=torch.tensor([0, 1]), base_loss=nn.CrossEntropyLoss(), metadata={})

    def test_identity_transition_recovers_joint_centroid(self) -> None:
        snapshot = FeatureSnapshot(np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]), np.array([0, 0, 1, 1]), np.array([3, 1, 4, 2]), 'fixture', 'train')
        transition = TransitionArtifact(np.eye(2), 'known')
        artifact = MCLDCEEstimator().estimate(snapshot, transition)
        np.testing.assert_allclose(artifact.values, [[1.0, 0.0], [0.0, 1.5]])
        self.assertEqual(artifact.metadata['transition_artifact_hash'], transition.artifact_hash)

    def test_prior_and_coefficient_are_explicit(self) -> None:
        transition = np.array([[0.8, 0.2], [0.1, 0.9]])
        clean = np.array([0.6, 0.4])
        observed = clean @ transition
        np.testing.assert_allclose(estimate_clean_prior(observed, transition), clean)
        coefficient = build_label_imputation_matrix(clean, transition)
        self.assertEqual(coefficient.shape, (2, 2))
        self.assertEqual(np.linalg.matrix_rank(coefficient), 2)

    def test_nonidentity_centroid_matches_paper_matrix_recovery(self) -> None:
        transition_matrix = np.array([[0.8, 0.2], [0.1, 0.9]])
        features = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 3.0], [1.0, 1.0]])
        noisy_labels = np.array([0, 0, 1, 1, 0])
        snapshot = FeatureSnapshot(features, noisy_labels, np.arange(features.shape[0]), 'fixture', 'train')
        transition = TransitionArtifact(transition_matrix, 'known')
        artifact = MCLDCEEstimator().estimate(snapshot, transition)
        observed_prior = np.bincount(noisy_labels, minlength=2) / noisy_labels.size
        clean_prior = estimate_clean_prior(observed_prior, transition_matrix)
        coefficient = build_label_imputation_matrix(clean_prior, transition_matrix)
        noisy_centroid = features.T @ np.eye(2)[noisy_labels] / noisy_labels.size
        expected = noisy_centroid @ np.linalg.pinv(coefficient, rcond=1e-08)
        np.testing.assert_allclose(artifact.values, expected.T, rtol=1e-10, atol=1e-10)
        self.assertEqual(artifact.metadata['coefficient_convention'], 'M=sum_i pi_i sum_j T_ij K_i_to_j^T')


    def test_identity_risk_and_feature_gradients_equal_squared_loss(self) -> None:
        features = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])
        snapshot = FeatureSnapshot(features.detach().numpy(), targets.numpy(), np.arange(4), 'fixture', 'train')
        artifact = MCLDCEEstimator().estimate(snapshot, TransitionArtifact(np.eye(2), 'known'))
        model = _mc_ldce__LinearFeatures().double()
        risk = MCLDCEObjective(artifact).compute(model=model, logits=model(features), features=features,
            noisy_targets=targets, sample_indices=torch.arange(4), base_loss=nn.CrossEntropyLoss(), metadata={})
        expected = (model(features) - torch.nn.functional.one_hot(targets, 2)).square().sum(1).mean()
        torch.testing.assert_close(risk, expected)
        actual_grad = torch.autograd.grad(risk, (features, model.classifier.weight), retain_graph=True)
        expected_grad = torch.autograd.grad(expected, (features, model.classifier.weight))
        for actual, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, reference)

    def test_minibatch_risks_and_gradients_average_to_global_current_feature_risk(self) -> None:
        targets = torch.tensor([0, 0, 0, 1, 1, 1])
        features = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
        snapshot = FeatureSnapshot(np.zeros((6, 2)), targets.numpy(), np.arange(6), 'fixture', 'train')
        artifact = MCLDCEEstimator().estimate(snapshot, TransitionArtifact([[0.8, 0.2], [0.2, 0.8]], 'known'))
        model = _mc_ldce__LinearFeatures().double()
        objective = MCLDCEObjective(artifact)
        risks = []
        for start, end in ((0, 6), (0, 2), (2, 6)):
            risks.append(objective.compute(model=model, logits=model(features[start:end]), features=features[start:end],
                noisy_targets=targets[start:end], sample_indices=torch.arange(start, end), base_loss=nn.CrossEntropyLoss(), metadata={}))
        combined = (risks[1] * 2 + risks[2] * 4) / 6
        inverse = torch.tensor(np.linalg.pinv(artifact.metadata['coefficient_matrix']), dtype=torch.float64)
        centroid = features.T @ torch.nn.functional.one_hot(targets, 2).double() / 6 @ inverse
        expected = 1 + model(features).square().sum(1).mean() - 2 * (model.classifier.weight * centroid.T).sum()
        torch.testing.assert_close(risks[0], expected)
        torch.testing.assert_close(combined, expected)
        full_grad = torch.autograd.grad(expected, (features, model.classifier.weight), retain_graph=True)
        batch_grad = torch.autograd.grad(combined, (features, model.classifier.weight))
        for actual, reference in zip(batch_grad, full_grad):
            torch.testing.assert_close(actual, reference)

    def test_joint_risk_really_updates_cnn_weights(self) -> None:
        torch.manual_seed(7)
        model = MCLDCECifarCNN(2, classifier_bias=False)
        _prepare_classifier(model)
        model.train()
        inputs, targets = torch.randn(2, 3, 32, 32), torch.tensor([0, 1])
        output = forward_with_features(model, inputs)
        snapshot = FeatureSnapshot(output.features.detach().numpy(), targets.numpy(), np.arange(2), 'fixture', 'train')
        artifact = MCLDCEEstimator().estimate(snapshot, TransitionArtifact(np.eye(2), 'known'))
        before = model.convolutions[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        risk = MCLDCEObjective(artifact).compute(model=model, logits=output.logits, features=output.features,
            noisy_targets=targets, sample_indices=torch.arange(2), base_loss=nn.CrossEntropyLoss(), metadata={})
        optimizer.zero_grad(); risk.backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.convolutions[0].weight))
        self.assertTrue(torch.isfinite(model.convolutions[0].weight.grad).all())

    def test_short_runner_training_resume_and_summary(self) -> None:
        import json
        import yaml
        from types import SimpleNamespace
        from unittest.mock import patch
        from lnl_toolbox.data import DataRole
        from lnl_toolbox.training.mc_ldce_experiment import run_mc_ldce_experiment

        class TinyFeatures(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 2)
                self.classifier = nn.Linear(2, 2, bias=False)

            def forward_with_features(self, inputs):
                features = self.encoder(inputs)
                return self.classifier(features), features

            def forward(self, inputs):
                return self.forward_with_features(inputs)[0]

        config = yaml.safe_load(Path('configs/experiment/mc_ldce_cifar10_reproduction.yaml').read_text(encoding='utf-8'))
        config['trainer'] = {'epochs': 2, 'device': 'cpu'}
        config['transition'] = {'estimator': 'known', 'matrix': [[1., 0.], [0., 1.]]}
        batch = {'input': torch.tensor([[1., 0.], [0., 1.]]),
                 'target': torch.tensor([0, 1]), 'index': torch.arange(2)}
        prepared = SimpleNamespace(
            input_spec=SimpleNamespace(channels=None), num_classes=2, dataset='fixture',
            loader_config={'num_workers': 0},
            available_roles={DataRole.TRAIN, DataRole.TRAIN_EVAL, DataRole.TEST},
            loader=lambda *args, **kwargs: [batch])
        models = []

        def build(*args):
            model = TinyFeatures()
            models.append((model, model.encoder.weight.detach().clone()))
            return model

        with tempfile.TemporaryDirectory() as directory, \
             patch('lnl_toolbox.training.mc_ldce_experiment.prepare_experiment_data', return_value=prepared), \
             patch('lnl_toolbox.training.mc_ldce_experiment.build_reproduction_model', side_effect=build):
            run_dir = run_mc_ldce_experiment(config, output_dir=directory, requirements=object())
            uninterrupted = torch.load(run_dir / 'last.pt', weights_only=False)
            self.assertFalse(torch.equal(models[0][1], uninterrupted['model']['encoder.weight']))
            summary = json.loads((run_dir / 'summary.json').read_text())
            self.assertEqual(summary['epochs_averaged'], 2)
            self.assertAlmostEqual(summary['test_accuracy_mean'],
                sum(row['test_accuracy'] for row in uninterrupted['metrics']) / 2)
            self.assertEqual(len(uninterrupted['optimizer']['param_groups'][0]['params']), 3)

            # Interrupt after epoch 1, then resume with the SAME two-epoch plan.
            from lnl_toolbox.training.checkpoint import atomic_save as save
            interrupted_dir = Path(directory) / 'interrupted'

            def stop_after_save(payload, path):
                save(payload, path)
                raise RuntimeError('test interruption')

            with patch('lnl_toolbox.training.mc_ldce_experiment.atomic_save', side_effect=stop_after_save):
                with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                    run_mc_ldce_experiment(config, output_dir=interrupted_dir, requirements=object())
            run_mc_ldce_experiment(config, resume=interrupted_dir / 'last.pt', requirements=object())
            resumed = torch.load(interrupted_dir / 'last.pt', weights_only=False)
            for name, value in uninterrupted['model'].items():
                torch.testing.assert_close(resumed['model'][name], value, rtol=0, atol=0)
            self.assertEqual(resumed['metrics'], uninterrupted['metrics'])
