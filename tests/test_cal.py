"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_cal.py ---
import tempfile

import json

# --- merged from test_cal.py ---
from pathlib import Path

# --- merged from test_cal.py ---
import unittest

# --- merged from test_cal.py ---
import numpy as np

# --- merged from test_cal.py ---
import torch

# --- merged from test_cal.py ---
from lnl_toolbox.algorithms.cal import cal_covariance_correction, cal_objective, cal_sieve_scores, cal_transition_indicators, cores2_adjusted_losses, cores2_warmup_prior, resolve_confidence_weight

# --- merged from test_cal.py ---
from lnl_toolbox.noise.cal import DROP, KEEP, RELABEL, CALProxyArtifact, build_cal_proxy_artifact

# --- merged from test_cal.py ---
from lnl_toolbox.noise.estimators import PosteriorSnapshot

# --- merged from test_cal.py ---
from lnl_toolbox.training.cal_experiment import _assert_finite_warmup_gradients, _build_warmup_scheduler, _reference_transition_means, run_cal_experiment

# --- merged from test_cal.py ---
from lnl_toolbox.training.experiment import build_alpha_scaled_scheduler

# --- merged from test_cal.py ---
class _cal_CALTest(unittest.TestCase):

    def test_cores2_warmup_uses_sqrt_prior_without_cifar10_class_multiplier(self) -> None:
        prior = torch.tensor([0.8, 0.2, 0.0], dtype=torch.float64)
        original = prior.clone()
        weights = cores2_warmup_prior(prior)
        torch.testing.assert_close(weights, torch.tensor([2 / 3, 1 / 3, 0.0], dtype=torch.float64))
        torch.testing.assert_close(prior, original)
        balanced = torch.full((10,), 0.1)
        torch.testing.assert_close(cores2_warmup_prior(balanced), balanced)
        logits = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float64)
        probabilities = logits.softmax(dim=1)
        expected = -torch.log(probabilities[:, 0] + 1e-8) - 2.0 * (
            -torch.log(probabilities + 1e-5) * weights
        ).sum(dim=1)
        actual = cores2_adjusted_losses(logits, torch.tensor([0]), weights, 2.0)
        torch.testing.assert_close(actual, expected)

    def test_cores2_warmup_rejects_invalid_prior(self) -> None:
        for prior in (torch.zeros(2), torch.tensor([-1.0, 2.0]), torch.tensor([float('nan'), 1.0])):
            with self.assertRaises(ValueError):
                cores2_warmup_prior(prior)

    def test_sieve_scores_match_author_formula_not_confidence_adjusted_loss(self) -> None:
        logits = torch.tensor([[20.0] + [0.0] * 9, [20.0] + [0.0] * 9], dtype=torch.float64)
        targets = torch.tensor([0, 1])
        probabilities = logits.softmax(dim=1)
        official = -torch.log(probabilities.gather(1, targets[:, None]).squeeze(1) + 1e-8)
        official -= (-torch.log(probabilities + 1e-8)).mean(dim=1)
        scores = cal_sieve_scores(logits, targets)
        torch.testing.assert_close(scores, official)
        adjusted = cores2_adjusted_losses(logits, targets, torch.full((10,), 0.1), 2.0)
        self.assertFalse(torch.allclose(scores, adjusted))
        snapshot = PosteriorSnapshot(probabilities.numpy(), targets.numpy(), np.array([0, 1]), 'fixture', 'train')
        artifact = build_cal_proxy_artifact(snapshot, scores.numpy(), lower_threshold=-8.0, upper_threshold=-8.0)
        self.assertEqual(artifact.sample_status.tolist(), [KEEP, RELABEL])
        self.assertEqual(artifact.proxy_targets.tolist(), [0, 0])
        boundary = build_cal_proxy_artifact(snapshot, np.full(2, -8.0), lower_threshold=-8.0, upper_threshold=-8.0)
        self.assertEqual(boundary.sample_status.tolist(), [DROP, DROP])

    def test_clean_identity_vector_is_not_a_training_objective_input(self) -> None:
        logits = torch.tensor([[1.5, -0.5], [-0.25, 1.0]])
        noisy = torch.tensor([0, 1])
        proxy = torch.tensor([0, 1])
        retained = torch.tensor([True, True])
        noisy_prior = torch.tensor([0.5, 0.5])
        proxy_prior = torch.tensor([0.5, 0.5])
        means = torch.zeros(2, 2)
        transition = torch.eye(2)
        clean_vectors = (torch.tensor([0, 1]), torch.tensor([1, 0]))
        values = []
        for _clean_identity_only in clean_vectors:
            loss, _ = cal_objective(
                logits,
                noisy,
                proxy,
                retained,
                noisy_prior,
                proxy_prior,
                means,
                transition,
                confidence_weight=1.0,
            )
            values.append(loss)
        torch.testing.assert_close(values[0], values[1])

    def _snapshot(self) -> PosteriorSnapshot:
        return PosteriorSnapshot(np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]), np.array([0, 0, 1]), np.array([8, 3, 5]), 'fixture', 'train')

    def test_proxy_status_and_global_index_lookup(self) -> None:
        artifact = build_cal_proxy_artifact(self._snapshot(), np.array([-2.0, 2.0, 0.0]), lower_threshold=-1.0, upper_threshold=1.0)
        targets, retained, status = artifact.lookup(torch.tensor([8, 5, 3]))
        self.assertEqual(status.tolist(), [KEEP, DROP, RELABEL])
        self.assertEqual(retained.tolist(), [True, False, True])
        self.assertEqual(targets.tolist(), [0, 1, 1])

    def test_artifact_round_trip(self) -> None:
        artifact = build_cal_proxy_artifact(self._snapshot(), np.array([-2.0, 2.0, 0.0]), lower_threshold=-1.0, upper_threshold=1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'proxy.npz'
            artifact.save(path)
            loaded = CALProxyArtifact.load(path)
        self.assertEqual(loaded.artifact_hash, artifact.artifact_hash)

    def test_cores2_and_covariance_are_differentiable(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        adjusted = cores2_adjusted_losses(logits, torch.tensor([0, 1]), torch.tensor([0.5, 0.5]), 1.0)
        self.assertTrue(torch.allclose(adjusted, -torch.ones(2), atol=5e-05))
        losses = -torch.log_softmax(logits, dim=1)
        correction, means = cal_covariance_correction(losses, torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([True, True]), torch.tensor([1.0, 0.0]), torch.zeros(2, 2), torch.zeros(2, 2))
        correction.backward()
        self.assertTrue(torch.isfinite(correction))
        self.assertEqual(tuple(means.shape), (2, 2))

    def test_covariance_centers_with_reference_transition_not_batch_mean(self) -> None:
        losses = torch.tensor([[1.0, 0.0], [3.0, 0.0]], requires_grad=True)
        correction, _ = cal_covariance_correction(
            losses,
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            torch.tensor([True, True]),
            torch.tensor([1.0, 0.0]),
            torch.zeros(2, 2),
            torch.tensor([[0.2, 0.8], [0.0, 0.0]]),
        )
        self.assertAlmostEqual(float(correction), 0.1, places=6)

    def test_reference_transition_uses_full_retained_proxy_population(self) -> None:
        proxy = CALProxyArtifact(
            np.array([3, 8, 9]),
            np.array([0, 0, 1]),
            np.array([KEEP, KEEP, DROP]),
            "snapshot",
            -1.0,
            1.0,
        )
        means = _reference_transition_means(
            proxy,
            np.array([9, 3, 8]),
            np.array([1, 0, 1]),
            2,
        )
        torch.testing.assert_close(
            means, torch.tensor([[0.5, 0.5], [0.0, 0.0]])
        )

    def test_transition_indicator(self) -> None:
        value = cal_transition_indicators(torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([True, False]), 2)
        self.assertEqual(value[0, 0, 1].item(), 1.0)
        self.assertEqual(value[1].sum().item(), 0.0)

    def test_warmup_uses_configured_scheduler(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = _build_warmup_scheduler(optimizer, {'scheduler': {'name': 'multistep', 'milestones': [2], 'gamma': 0.1}}, 3)
        self.assertIsNotNone(scheduler)
        for _ in range(3):
            optimizer.zero_grad()
            parameter.sum().backward()
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.01)

    def test_confidence_schedule_starts_with_ce_then_enables_cores2(self) -> None:
        schedule = {'milestones': [10, 40, 80], 'values': [0.0, 2.0, 2.0]}
        self.assertEqual(resolve_confidence_weight(0, 2.0, schedule), 0.0)
        self.assertEqual(resolve_confidence_weight(9, 2.0, schedule), 0.0)
        self.assertEqual(resolve_confidence_weight(10, 2.0, schedule), 0.0)
        self.assertAlmostEqual(resolve_confidence_weight(25, 2.0, schedule), 1.0)
        self.assertEqual(resolve_confidence_weight(40, 2.0, schedule), 2.0)
        self.assertEqual(resolve_confidence_weight(64, 2.0, schedule), 2.0)

    def test_confidence_schedule_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            resolve_confidence_weight(0, 2.0, {'milestones': [2, 1], 'values': [0.0]})
        with self.assertRaises(ValueError):
            resolve_confidence_weight(0, 2.0, {'milestones': [], 'values': []})

    def test_alpha_scaled_scheduler_matches_official_lr_scaling(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = build_alpha_scaled_scheduler(optimizer, {'name': 'multistep', 'milestones': [60], 'gamma': 0.1})
        scheduler.step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.05)
        scheduler.step(2.0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.1 / 3.0)

    def test_alpha_scaled_scheduler_accepts_independent_stage_rate(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = build_alpha_scaled_scheduler(optimizer, {
            'name': 'multistep', 'milestones': [1], 'gamma': 0.1,
            'lr_values': [0.03],
        })
        scheduler.step(0.0)
        scheduler.step(0.0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.03)

    def test_warmup_gradient_guard_rejects_non_finite_gradient(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.weight.grad = torch.full_like(model.weight, float('nan'))
        with self.assertRaises(ValueError):
            _assert_finite_warmup_gradients(model)

    def test_validation_and_test_are_separate_and_test_is_final_only(self) -> None:
        import yaml

        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / 'configs' / 'experiment' / 'cal_cifar10_smoke.yaml').read_text(encoding='utf-8')
        )
        calls = []

        def evaluate(_model, loader, _criterion, _device):
            split = loader.dataset.split
            calls.append(split.identity.fingerprint)
            return {'loss': 1.0, 'accuracy': 0.25, 'samples': float(len(loader.dataset))}

        with tempfile.TemporaryDirectory() as directory:
            from unittest.mock import patch

            with patch('lnl_toolbox.training.cal_experiment.evaluate_classification', side_effect=evaluate):
                run_dir = run_cal_experiment(config, directory)
            rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])
        self.assertEqual([row['event'] for row in rows], ['epoch', 'final'])
        self.assertIn('validation_loss', rows[0])
        self.assertNotIn('test_loss', rows[0])
        self.assertFalse(rows[-1]['test_selection_leakage'])

    def test_no_validation_has_train_only_epoch_and_one_test_call(self) -> None:
        import yaml
        from unittest.mock import patch

        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / 'configs' / 'experiment' / 'cal_cifar10_smoke.yaml').read_text(encoding='utf-8')
        )
        config['data']['validation_size'] = 0
        calls = []

        def evaluate(_model, loader, _criterion, _device):
            calls.append(loader)
            return {'loss': 1.0, 'accuracy': 0.25, 'samples': float(len(loader.dataset))}

        with tempfile.TemporaryDirectory() as directory:
            with patch('lnl_toolbox.training.cal_experiment.evaluate_classification', side_effect=evaluate):
                run_dir = run_cal_experiment(config, directory)
            rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertNotIn('validation_loss', rows[0])
        self.assertNotIn('test_loss', rows[0])
        self.assertEqual(rows[-1]['event'], 'final')

    def test_epoch_metrics_are_written_before_final_evaluation(self) -> None:
        import yaml
        from unittest.mock import patch

        from lnl_toolbox.training import cal_experiment as cal_module

        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / 'configs' / 'experiment' / 'cal_cifar10_smoke.yaml').read_text(encoding='utf-8')
        )
        writes = []
        original_writer = cal_module._write_epoch_metrics

        def record_write(rows, path):
            writes.append([row.get('event', 'epoch') for row in rows])
            original_writer(rows, path)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(cal_module, '_write_epoch_metrics', side_effect=record_write):
                run_cal_experiment(config, directory)
        self.assertIn(['epoch'], writes)
        self.assertEqual(writes[-1], ['epoch', 'final'])
