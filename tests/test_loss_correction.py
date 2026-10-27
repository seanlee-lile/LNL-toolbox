"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_loss_correction_reproduction.py ---
import unittest

# --- merged from test_loss_correction_reproduction.py ---
from pathlib import Path

# --- merged from test_loss_correction_reproduction.py ---
import yaml

# --- merged from test_loss_correction_reproduction.py ---
import torch

# --- merged from test_loss_correction_reproduction.py ---
from lnl_toolbox.algorithms.transition_risk import BackwardRiskCorrector, ForwardRiskCorrector

# --- merged from test_loss_correction_reproduction.py ---
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

# --- merged from test_loss_correction_reproduction.py ---
from lnl_toolbox.models.cifar_resnet import cifar_resnet14, cifar_resnet32

# --- merged from test_loss_correction_reproduction.py ---
from lnl_toolbox.noise.transition import KnownTransition

# --- merged from test_loss_correction_reproduction.py ---
from lnl_toolbox.plugins.builtin import build_builtin_pipeline

# --- merged from test_loss_correction_reproduction.py ---
class _loss_correction_reproduction_LossCorrectionReproductionTest(unittest.TestCase):



    def test_paper_depth_constructors_return_class_logits(self) -> None:
        inputs = torch.randn(2, 3, 32, 32)
        self.assertEqual(cifar_resnet14()(inputs).shape, (2, 10))
        self.assertEqual(cifar_resnet32()(inputs).shape, (2, 10))

    def test_full_experiment_config_routes_correction_through_pipeline(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / 'configs' / 'experiment' / 'cifar10_foundation_smoke.yaml').read_text(encoding='utf-8'))
        pipeline = build_builtin_pipeline(config['pipeline'])
        self.assertIsInstance(pipeline.risk_corrector, ForwardRiskCorrector)
        self.assertIsNotNone(pipeline.transition_estimator)
        self.assertIsInstance(build_builtin_pipeline({'name': 'standard_noisy_erm', 'transition_estimator': {'name': 'anchor'}, 'risk_corrector': {'name': 'backward'}}).risk_corrector, BackwardRiskCorrector)

# --- merged from test_transition_risk.py ---
import unittest

# --- merged from test_transition_risk.py ---
import torch

# --- merged from test_transition_risk.py ---
from lnl_toolbox.algorithms.transition_risk import BackwardRiskCorrector, ForwardRiskCorrector

# --- merged from test_transition_risk.py ---
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

# --- merged from test_transition_risk.py ---
from lnl_toolbox.noise.transition import KnownTransition

# --- merged from test_transition_risk.py ---
class _transition_risk_TransitionRiskTest(unittest.TestCase):

    def setUp(self) -> None:
        self.logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        self.targets = torch.tensor([0, 1])
        self.identity = KnownTransition(torch.eye(2).numpy())

    def test_forward_identity_matches_cross_entropy(self) -> None:
        corrected = ForwardRiskCorrector().per_sample_risk(logits=self.logits, noisy_targets=self.targets, base_loss=CrossEntropyLoss(), transition=self.identity)
        torch.testing.assert_close(corrected, CrossEntropyLoss()(self.logits, self.targets))

    def test_backward_identity_matches_cross_entropy(self) -> None:
        corrected = BackwardRiskCorrector().per_sample_risk(logits=self.logits, noisy_targets=self.targets, base_loss=CrossEntropyLoss(), transition=self.identity)
        torch.testing.assert_close(corrected, CrossEntropyLoss()(self.logits, self.targets))

    def test_backward_returns_finite_corrected_risk(self) -> None:
        transition = KnownTransition(torch.tensor([[0.8, 0.2], [0.1, 0.9]]).numpy())
        corrected = BackwardRiskCorrector().per_sample_risk(logits=self.logits, noisy_targets=self.targets, base_loss=CrossEntropyLoss(), transition=transition)
        self.assertTrue(torch.isfinite(corrected).all())


class LossCorrectionPaperProtocolTest(unittest.TestCase):
    def test_nonidentity_corrections_match_equations_and_gradients(self):
        logits = torch.tensor([[2., -1.], [-.3, .7]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([0, 1])
        matrix = torch.tensor([[.8, .2], [.1, .9]], dtype=torch.float64)
        expected_forward = -(logits.softmax(1) @ matrix).log()[torch.arange(2), targets]
        expected_backward = ((-logits.log_softmax(1)) @ matrix.inverse().T)[torch.arange(2), targets]
        for corrector, expected in ((ForwardRiskCorrector(), expected_forward), (BackwardRiskCorrector(), expected_backward)):
            actual = corrector.per_sample_risk(logits=logits, noisy_targets=targets,
                base_loss=CrossEntropyLoss(), transition=KnownTransition(matrix.numpy()))
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
                                       torch.autograd.grad(expected.sum(), logits, retain_graph=True)[0])

    def test_preactivation_preserves_raw_shortcut_and_negative_outputs(self):
        from lnl_toolbox.models.cifar_resnet import LossCorrectionPreActBlock, BasicBlock
        block = LossCorrectionPreActBlock(3, 3, first=True).eval()
        self.assertIsInstance(block.bn1, torch.nn.Identity)
        with torch.no_grad():
            block.conv1.weight.zero_(); block.conv1.bias.zero_()
            block.conv2.weight.zero_(); block.conv2.bias.zero_()
        inputs = -torch.ones(2, 3, 8, 8)
        torch.testing.assert_close(block(inputs), inputs)
        standard = cifar_resnet32()
        corrected = cifar_resnet32(variant='loss_correction')
        self.assertTrue(any(isinstance(layer, BasicBlock) for layer in standard.modules()))
        self.assertTrue(any(isinstance(layer, LossCorrectionPreActBlock) for layer in corrected.modules()))
        self.assertEqual(corrected(torch.randn(2, 3, 32, 32)).shape, (2, 10))

    def test_percentile_anchor_does_not_mutate_posterior_rows(self):
        import numpy as np
        from lnl_toolbox.noise.estimators import PosteriorSnapshot, AnchorTransitionEstimator
        scores = np.linspace(.01, .99, 100)
        probabilities = np.column_stack((scores, 1 - scores))
        snapshot = PosteriorSnapshot(probabilities, np.arange(100) % 2, np.arange(100), 'cifar10', 'train+validation')
        artifact = AnchorTransitionEstimator(97).estimate(snapshot)
        self.assertEqual(artifact.metadata['anchor_global_indices'], [96, 3])
        np.testing.assert_allclose(artifact.matrix, probabilities[[96, 3]])
        np.testing.assert_array_equal(snapshot.noisy_probabilities, probabilities)
        self.assertEqual(AnchorTransitionEstimator().estimate(snapshot).metadata['anchor_global_indices'], [99, 0])
        tied = PosteriorSnapshot(np.full((4, 2), .5), np.arange(4) % 2, np.arange(4), 'cifar10', 'train')
        with self.assertRaisesRegex(ValueError, 'predictions are tied'):
            AnchorTransitionEstimator(97).estimate(tied)

    def test_unknown_T_requires_trained_CE_classifier(self):
        with self.assertRaisesRegex(ValueError, 'positive CE pretraining'):
            build_builtin_pipeline({'name': 'standard_noisy_erm', 'pretraining_protocol': 'loss_correction',
                'warmup_epochs': 0, 'transition_estimator': {'name': 'anchor', 'percentile': 97},
                'risk_corrector': {'name': 'forward'}})

    def test_CE_best_loss_checkpoint_and_completed_stage_reuse(self):
        import tempfile
        from unittest.mock import patch
        from lnl_toolbox.training.snapshots import pretrain_noisy_classifier
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1], gamma=.1)
        loader = [{'input': torch.tensor([[1., 0.], [0., 1.]]), 'target': torch.tensor([0, 1]), 'index': torch.tensor([0, 1])}]
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.evaluation.classification.evaluate_classification',
                side_effect=[{'loss': .2}, {'loss': .5}]) as evaluate:
            result = pretrain_noisy_classifier(model, optimizer, loader, 'cpu', epochs=2,
                validation_loader=loader, scheduler=scheduler, checkpoint_dir=directory, checkpoint_identity={'seed': 1})
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(result['steps'], 2)
            best = torch.load(Path(directory)/'pretraining_best.pt', weights_only=False)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, best['model'][key])
            evaluate.reset_mock()
            repeated = pretrain_noisy_classifier(model, optimizer, loader, 'cpu', epochs=2,
                validation_loader=loader, scheduler=scheduler, checkpoint_dir=directory, checkpoint_identity={'seed': 1})
            self.assertEqual(repeated, result)
            evaluate.assert_not_called()
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .01)

    def test_web_selectors_and_CE_budget_round_trip(self):
        import tempfile
        from web import command_console as web
        recipe = 'loss-correction-cifar10-asymmetric04'
        fields = {field['path']: field for field in web._config_schema(recipe)['fields']}
        self.assertFalse(fields['pipeline.transition_estimator.name']['editable'])
        self.assertFalse(fields['pipeline.transition_estimator.name']['visible'])
        self.assertEqual([item['value'] for item in fields['pipeline.risk_corrector.name']['choices']], ['forward', 'backward'])
        from copy import deepcopy
        from unittest.mock import patch
        native = yaml.safe_load((web.ROOT/'configs/experiment/loss_correction_cifar10_asymmetric04.yaml').read_text(encoding='utf-8'))
        native['noise'] = {'name': 'native', 'validation_targets': 'noisy', 'seed': 1, 'rng': 'numpy_legacy', 'manifest_filename': 'noise_manifest.npz'}
        native['pipeline']['transition_estimator'] = {'name': 'anchor', 'percentile': 97.0}
        native['pipeline']['warmup_epochs'] = 120
        native_fields = {field['path']: field for field in web._registry_config_fields(native, 'loss_correction', allow_missing=True)[0]}
        self.assertTrue(native_fields['pipeline.warmup_epochs']['visible'])
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.catalog.load_yaml', side_effect=lambda *args: deepcopy(native)):
            destination = Path(directory)/'config.yaml'
            payload = {'recipe': recipe, 'patches': [
                {'path': 'pipeline.risk_corrector.name', 'value': 'backward'},
                {'path': 'pipeline.warmup_epochs', 'value': 2}], 'acknowledge_paper_impact': True}
            web._save_config(payload, for_review=True, destination=destination)
            config = yaml.safe_load(destination.read_text(encoding='utf-8'))
            self.assertEqual(config['pipeline']['transition_estimator'], {'name': 'anchor', 'percentile': 97.0})
            self.assertEqual(config['pipeline']['warmup_epochs'], 2)
            payload['patches'][-1]['value'] = 0
            with self.assertRaisesRegex(ValueError, '预训练轮数必须大于 0'):
                web._save_config(payload, for_review=True, destination=destination)

    def _training_config(self, estimated=True, epochs=2):
        return {'execution': {'runner': 'supervised'}, 'seed': 7,
            'data': {'name': 'cifar10', 'root': 'unused', 'validation_size': 10, 'augment': False},
            'noise': {'name': 'symmetric', 'rate': .2, 'validation_targets': 'noisy'},
            'model': {'name': 'tiny_cnn', 'width': 4}, 'loss': {'name': 'ce'},
            'optimizer': {'name': 'sgd', 'lr': .01, 'momentum': .9}, 'scheduler': {'name': 'none'},
            'loader': {'batch_size': 10, 'num_workers': 0, 'pin_memory': False},
            'trainer': {'epochs': epochs, 'device': 'cpu'},
            'evaluation': {'selection_split': 'validation', 'selection_metric': 'loss', 'corrected_loss': True},
            'pipeline': {'name': 'standard_noisy_erm', 'pretraining_protocol': 'loss_correction',
                'warmup_epochs': 1 if estimated else 0,
                'transition_estimator': {'name': 'anchor', 'percentile': 97} if estimated else
                    {'name': 'known', 'matrix': (torch.eye(10)*.8 + (1-torch.eye(10))*(.2/9)).tolist()},
                'risk_corrector': {'name': 'forward'}}}

    def _data(self, root, split):
        import numpy as np
        from lnl_toolbox.data.cifar import CifarData
        size = 40 if split == 'train' else 20
        images = np.random.default_rng(9 if split == 'train' else 10).integers(0, 256, (size, 32, 32, 3), dtype=np.uint8)
        return CifarData(images, np.arange(size) % 10, tuple(map(str, range(10))), split, 'cifar10')

    def test_real_two_stage_training_reinitializes_and_resumes(self):
        import json, tempfile
        from unittest.mock import patch
        from lnl_toolbox.training import experiment
        from lnl_toolbox.noise.estimators import PosteriorSnapshot
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._data), \
                patch.object(experiment, 'build_model', wraps=experiment.build_model) as build:
            run = experiment.run_experiment(self._training_config(), Path(directory)/'run')
            self.assertEqual(build.call_count, 2)
            snapshot = PosteriorSnapshot.load(run/'posterior_snapshot.npz')
            self.assertEqual(snapshot.num_samples, 40)
            self.assertEqual(len(set(snapshot.global_indices)), 40)
            first = torch.load(run/'last.pt', weights_only=False)
            self.assertEqual(first['pipeline']['state']['metadata']['pretraining']['epochs'], 1)
            rows = [json.loads(row) for row in (run/'metrics.jsonl').read_text().splitlines()]
            losses = [row['selection_loss'] for row in rows if row['event'] == 'epoch']
            self.assertAlmostEqual(first['pipeline']['state']['metadata']['best_selection_loss'], min(losses))
            stage_mtime = (run/'pretraining_last.pt').stat().st_mtime_ns
            experiment.run_experiment(self._training_config(epochs=3), resume=run/'last.pt')
            self.assertEqual((run/'pretraining_last.pt').stat().st_mtime_ns, stage_mtime)
            resumed = torch.load(run/'last.pt', weights_only=False)
            full_run = experiment.run_experiment(self._training_config(epochs=3), Path(directory)/'full')
            full = torch.load(full_run/'last.pt', weights_only=False)
            for name, value in resumed['model'].items():
                torch.testing.assert_close(value, full['model'][name], rtol=0, atol=0)

    def test_known_T_selects_corrected_validation_loss_not_accuracy(self):
        import tempfile
        from unittest.mock import patch
        from lnl_toolbox.training import experiment
        criteria = []
        def evaluate(model, loader, criterion, device):
            criteria.append(criterion)
            return ({'loss': .8, 'accuracy': .9}, {'loss': .2, 'accuracy': .1},
                    {'loss': .5, 'accuracy': .4})[len(criteria)-1]
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=self._data), \
                patch.object(experiment, 'evaluate_classification', side_effect=evaluate):
            run = experiment.run_experiment(self._training_config(estimated=False), Path(directory)/'known')
            best = torch.load(run/'best.pt', weights_only=False)
            self.assertEqual(best['best_epoch'], 1)
            self.assertFalse((run/'pretraining_last.pt').exists())
            self.assertIs(criteria[0], criteria[1])
            logits, labels = torch.randn(2, 10), torch.tensor([0, 1])
            matrix = self._training_config(estimated=False)['pipeline']['transition_estimator']['matrix']
            expected = ForwardRiskCorrector().per_sample_risk(logits=logits, noisy_targets=labels,
                base_loss=CrossEntropyLoss(), transition=KnownTransition(matrix))
            torch.testing.assert_close(criteria[0](logits, labels), expected)
            self.assertIsInstance(criteria[2], CrossEntropyLoss)
