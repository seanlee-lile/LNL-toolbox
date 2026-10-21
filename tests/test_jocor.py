"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_jocor.py ---
from pathlib import Path

# --- merged from test_jocor.py ---
import tempfile

# --- merged from test_jocor.py ---
import unittest

# --- merged from test_jocor.py ---
import torch

# --- merged from test_jocor.py ---
from torch import nn

# --- merged from test_jocor.py ---
import yaml

# --- merged from test_jocor.py ---
from lnl_toolbox.algorithms.jocor import JoCoRAlgorithm, jocor_joint_scores, symmetric_kl_per_sample

# --- merged from test_jocor.py ---
from lnl_toolbox.algorithms.multi_model import ModelGroup

# --- merged from test_jocor.py ---
from lnl_toolbox.core import Batch, ExperimentContext, RunState

# --- merged from test_jocor.py ---
from lnl_toolbox.core.hyperparameters import resolve_parameter_sampling

# --- merged from test_jocor.py ---
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss

# --- merged from test_jocor.py ---
from lnl_toolbox.models.cifar_six_conv import CifarSixConvNet

# --- merged from test_jocor.py ---
from lnl_toolbox.plugins.builtin import build_builtin_multi_model_algorithm

# --- merged from test_jocor.py ---
from lnl_toolbox.selectors import SelectionInput, SmallLossSelector

# --- merged from test_jocor.py ---
from lnl_toolbox.training.checkpoint import load_checkpoint, save_checkpoint

# --- merged from test_jocor.py ---
class _jocor_JoCoRMathTest(unittest.TestCase):

    def test_identical_logits_have_zero_symmetric_kl(self) -> None:
        logits = torch.tensor([[2.0, -1.0], [0.5, 0.2]])
        torch.testing.assert_close(symmetric_kl_per_sample(logits, logits), torch.zeros(2), atol=1e-07, rtol=0.0)

    def test_joint_score_matches_formula_and_is_swap_invariant(self) -> None:
        first = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        second = torch.tensor([[1.0, 0.5], [0.2, 1.4]], requires_grad=True)
        targets = torch.tensor([0, 1])
        loss_1 = nn.functional.cross_entropy(first, targets, reduction='none')
        loss_2 = nn.functional.cross_entropy(second, targets, reduction='none')
        actual = jocor_joint_scores(loss_1, loss_2, first, second, lambda_=0.7)
        expected = 0.3 * (loss_1 + loss_2) + 0.7 * symmetric_kl_per_sample(first, second)
        torch.testing.assert_close(actual, expected)
        swapped = jocor_joint_scores(loss_2, loss_1, second, first, lambda_=0.7)
        torch.testing.assert_close(actual, swapped)
        actual.mean().backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())

    def test_floor_selection_uses_stable_global_index_ties(self) -> None:
        selector = SmallLossSelector(0.5, rounding='floor')
        result = selector.select(SelectionInput(scores=torch.ones(3), sample_indices=torch.tensor([9, 4, 7])))
        torch.testing.assert_close(result.selected_mask, torch.tensor([False, True, False]))
        self.assertEqual(result.metrics['selected_samples'], 1.0)
        with self.assertRaisesRegex(ValueError, 'rounding'):
            SmallLossSelector(0.5, rounding='nearest')

# --- merged from test_jocor.py ---
class _jocor__CountingSelector:

    def __init__(self) -> None:
        self.calls = 0
        self.last_scores = None

    def select(self, selection_input):
        self.calls += 1
        self.last_scores = selection_input.scores.clone()
        return SmallLossSelector(0.5, rounding='floor').select(selection_input)

# --- merged from test_jocor.py ---
class _jocor_JoCoRAlgorithmTest(unittest.TestCase):

    @staticmethod
    def _algorithm(selector=None):
        torch.manual_seed(3)
        models = ModelGroup({'model_1': nn.Linear(2, 2, bias=False), 'model_2': nn.Linear(2, 2, bias=False)})
        optimizer = torch.optim.Adam(models.parameters(), lr=0.01)
        algorithm = JoCoRAlgorithm(models, optimizer, CrossEntropyLoss(), selector or SmallLossSelector(0.5, rounding='floor'), torch.device('cpu'), lambda_=0.9)
        algorithm.setup(ExperimentContext(Path('.')))
        return algorithm

    def test_one_shared_selection_updates_both_models(self) -> None:
        selector = _jocor__CountingSelector()
        algorithm = self._algorithm(selector)
        before = {name: model.weight.detach().clone() for name, model in algorithm.models.models.items()}
        state = RunState(phase='train', cycle=0)
        algorithm.on_cycle_start(state)
        result = algorithm.step(Batch({'input': torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]), 'target': torch.tensor([0, 1, 0, 1]), 'index': torch.tensor([8, 3, 5, 1])}), state)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(result.metrics['selected_samples'], 2.0)
        for name, model in algorithm.models.models.items():
            self.assertFalse(torch.equal(before[name], model.weight))

    def test_plugin_and_checkpoint_roundtrip(self) -> None:
        source = self._algorithm()
        state = RunState(phase='train', cycle=2, step=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'last.pt')
            save_checkpoint(path, source, state, completed_epoch=2, config={'algorithm': {'name': 'jocor', 'lambda': 0.9}})
            restored_models = ModelGroup({'model_1': nn.Linear(2, 2, bias=False), 'model_2': nn.Linear(2, 2, bias=False)})
            restored_optimizer = torch.optim.Adam(restored_models.parameters(), lr=0.01)
            restored = build_builtin_multi_model_algorithm({'name': 'jocor', 'lambda': 0.9}, models=restored_models, optimizer=restored_optimizer, loss=CrossEntropyLoss(), selector=SmallLossSelector(0.5, rounding='floor'), device=torch.device('cpu'))
            restored.setup(ExperimentContext(Path(directory)))
            loaded_state, completed, payload = load_checkpoint(path, restored, torch.device('cpu'))
            self.assertEqual(payload['format_version'], 2)
            self.assertEqual(completed, 2)
            self.assertEqual(loaded_state.step, 7)
            for name in source.models.models:
                torch.testing.assert_close(source.models.models[name].weight, restored.models.models[name].weight)

    def test_checkpoint_roundtrip_without_selection_metrics(self) -> None:
        source = self._algorithm()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'last.pt')
            save_checkpoint(path, source, RunState(step=7), 0, {},
                            selection_split='none', best_validation_accuracy=None,
                            best_selection_accuracy=None)
            restored = self._algorithm()
            state, completed, payload = load_checkpoint(path, restored, torch.device('cpu'))
            self.assertIsNone(payload['best_validation_accuracy'])
            self.assertIsNone(payload['best_selection_accuracy'])
            self.assertEqual(payload['selection_split'], 'none')
            self.assertEqual(state.step, 7)
            self.assertEqual(completed, 0)

# --- merged from test_jocor.py ---
class _jocor_JoCoRConfigurationTest(unittest.TestCase):
    def test_estimated_rate_changes_selection_not_artificial_noise(self) -> None:
        from lnl_toolbox.core.config_schema import normalize_experiment_config
        path = Path('configs/experiment/jocor_cifar10_symmetric05_reproduction.yaml')
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        legacy = normalize_experiment_config(config)
        self.assertEqual(legacy['selector']['keep_rate']['end'], 0.5)
        self.assertEqual(legacy['algorithm']['estimated_noise_rate'], 0.5)
        config['algorithm']['estimated_noise_rate'] = 0.2
        normalized = normalize_experiment_config(config)
        self.assertEqual(normalized['noise']['rate'], 0.5)
        self.assertEqual(normalized['selector']['keep_rate']['end'], 0.8)
        config['algorithm']['estimated_noise_rate'] = 1.1
        with self.assertRaisesRegex(ValueError, 'estimated_noise_rate'):
            normalize_experiment_config(config)

    def test_web_estimate_is_editable_and_saves_independently(self) -> None:
        from web import command_console
        from lnl_toolbox.catalog import load_yaml
        recipe = 'jocor-cifar10-symmetric05-reproduction'
        fields = {field['path']: field for field in command_console._config_schema(recipe)['fields']}
        estimate = fields['algorithm.estimated_noise_rate']
        self.assertTrue(estimate['editable'])
        self.assertEqual(estimate['display_group'], 'default')
        self.assertFalse(fields['selector.keep_rate.end']['visible'])
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            saved = command_console._save_config({
                'recipe': recipe, 'path': str(Path(directory) / 'jocor.yaml'),
                'patches': [{'path': 'algorithm.estimated_noise_rate', 'value': 0.2}],
                'acknowledge_paper_impact': True,
            })
            config = load_yaml(command_console.ROOT / saved['path'])
            self.assertEqual(config['noise']['rate'], 0.5)
            self.assertEqual(config['selector']['keep_rate']['end'], 0.8)


    def test_official_model_shape(self) -> None:
        model = CifarSixConvNet(10)
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(output.shape, (2, 10))
        self.assertEqual(model.classifier.in_features, 256)

    def test_reproduction_config_records_official_sym50_setting(self) -> None:
        path = Path('configs/experiment/jocor_cifar10_symmetric05_reproduction.yaml')
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        resolved, record = resolve_parameter_sampling(config)
        self.assertEqual(record.parameters['setting'], {'noise': 'symmetric', 'rate': 0.5, 'lambda': 0.9})
        self.assertEqual(resolved['trainer']['epochs'], 200)
        self.assertEqual(resolved['loader']['batch_size'], 128)
        self.assertEqual(resolved['optimizer']['name'], 'adam')
        self.assertEqual(resolved['optimizer']['lr'], 0.001)
        self.assertEqual(resolved['algorithm']['lambda'], 0.9)
        self.assertEqual(resolved['selector']['rounding'], 'floor')
        self.assertEqual(resolved['selector']['keep_rate']['warmup_epochs'], 9)
        self.assertEqual(resolved['scheduler']['start_epoch'], 80)
        self.assertEqual(resolved['evaluation']['report_last_epochs'], 10)
        self.assertEqual(resolved['evaluation']['selection_split'], 'none')
        self.assertFalse(resolved['evaluation']['allow_test_selection'])

    def test_runner_reports_test_without_selecting_best_or_aliasing_validation(self) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from lnl_toolbox.training import multi_model_experiment as runner

        config = yaml.safe_load(Path('configs/experiment/jocor_cifar10_symmetric05_reproduction.yaml').read_text(encoding='utf-8'))
        config['trainer'].update(epochs=2, device='cpu', progress=False)
        config['scheduler'] = {'name': 'none'}
        prepared = Mock(dataset='cifar10', num_classes=10, manifest=None,
                        manifest_path=None, input_spec=None)
        prepared.loader.return_value = [{}]
        algorithm = Mock()
        algorithm.step.return_value = SimpleNamespace(metrics={
            'samples': 2., 'selected_samples': 1., 'loss': 1.,
            'all_sample_loss': 1., 'accuracy': .5,
            'model_1_accuracy': .5, 'model_2_accuracy': .5, 'agreement_loss': .1,
        })
        metrics = [dict(loss=1., accuracy=.9, model_1_accuracy=.8, model_2_accuracy=1.),
                   dict(loss=1., accuracy=.1, model_1_accuracy=0., model_2_accuracy=.2),
                   dict(loss=1., accuracy=.1, model_1_accuracy=0., model_2_accuracy=.2)]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, 'prepare_experiment_data', return_value=prepared), \
             patch.object(runner, 'bind_model_input', side_effect=lambda config, spec: config), \
             patch.object(runner, '_build_member', side_effect=lambda config, classes: nn.Linear(2, 2)), \
             patch.object(runner, 'build_builtin_multi_model_algorithm', return_value=algorithm), \
             patch.object(runner, 'evaluate_model_group', side_effect=metrics), \
             patch.object(runner, 'save_checkpoint') as save:
            runner.run_multi_model_experiment(config, directory, requirements=SimpleNamespace(validation_size=0))
            self.assertEqual([call.args[0].name for call in save.call_args_list], ['last.pt', 'last.pt'])
            for call in save.call_args_list:
                self.assertIsNone(call.kwargs['best_selection_accuracy'])
                self.assertIsNone(call.kwargs['best_validation_accuracy'])
            rows = [json.loads(line) for line in Path(directory, 'metrics.jsonl').read_text().splitlines()]
            for row in rows[:2]:
                self.assertIn('test_accuracy', row)
                self.assertNotIn('validation_accuracy', row)
                self.assertNotIn('validation_loss', row)
                self.assertNotIn('selection_accuracy', row)
            final = rows[-1]
            self.assertFalse(final['test_selection_leakage'])
            self.assertIsNone(final['best_selection_accuracy'])
            self.assertEqual(final['last_window_epochs'], 2)
            self.assertAlmostEqual(final['last_window_member_mean_accuracy'], .5)
