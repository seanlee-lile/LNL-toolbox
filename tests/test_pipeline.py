"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_pipeline.py ---
from pathlib import Path

# --- merged from test_pipeline.py ---
from copy import deepcopy

# --- merged from test_pipeline.py ---
import tempfile

# --- merged from test_pipeline.py ---
import unittest

# --- merged from test_pipeline.py ---
import numpy as np

# --- merged from test_pipeline.py ---
import torch

# --- merged from test_pipeline.py ---
from torch import nn

# --- merged from test_pipeline.py ---
from torch.utils.data import DataLoader

# --- merged from test_pipeline.py ---
from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm

# --- merged from test_pipeline.py ---
from lnl_toolbox.algorithms.transition_risk import ForwardRiskCorrector

# --- merged from test_pipeline.py ---
from lnl_toolbox.core import Batch, ExperimentContext, RunState

# --- merged from test_pipeline.py ---
from lnl_toolbox.noise import KnownTransition

# --- merged from test_pipeline.py ---
from lnl_toolbox.plugins.builtin import build_builtin_pipeline

# --- merged from test_pipeline.py ---
from lnl_toolbox.treatments import SupervisedWeightInput, WeightResult

# --- merged from test_pipeline.py ---
from lnl_toolbox.training.pipeline import PipelinePhase, StandardNoisyERMPipeline

# --- merged from test_pipeline.py ---
class _pipeline__WeightProvider:

    def __init__(self) -> None:
        self.inputs: list[SupervisedWeightInput] = []

    def compute(self, weight_input: SupervisedWeightInput) -> WeightResult:
        self.inputs.append(weight_input)
        return WeightResult(sample_weights=torch.full((weight_input.noisy_targets.numel(),), 0.5, device=weight_input.noisy_targets.device), metrics={'provider_seen': 1.0})

# --- merged from test_pipeline.py ---
class _pipeline__IndexedModel(nn.Module):

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.eye(classes) * 5.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits[inputs.view(-1).long()]

# --- merged from test_pipeline.py ---
class _pipeline__StatefulEstimator:

    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self):
        return {'value': self.value}

    def load_state_dict(self, state):
        self.value = int(state['value'])

# --- merged from test_pipeline.py ---
class _pipeline_PipelineIntegrationTest(unittest.TestCase):

    @staticmethod
    def _transition_fixture():
        classes = 4
        model = _pipeline__IndexedModel(classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.arange(classes).view(-1, 1)
        loader = DataLoader([{'input': inputs[index], 'target': torch.tensor(index), 'index': torch.tensor(index)} for index in range(classes)], batch_size=2, shuffle=False)
        pipeline = build_builtin_pipeline({'name': 'standard_noisy_erm', 'warmup_epochs': 0, 'transition_estimator': {'name': 'dual_t'}, 'risk_corrector': {'name': 'forward'}})
        return (pipeline, model, optimizer, loader)

    def test_standard_weight_provider_without_posterior_still_works(self) -> None:
        model = nn.Linear(2, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        provider = _pipeline__WeightProvider()
        algorithm = SupervisedClassificationAlgorithm(model, optimizer, nn.CrossEntropyLoss(), torch.device('cpu'), risk_corrector=ForwardRiskCorrector(), transition=KnownTransition(np.eye(2)), weight_provider=provider)
        algorithm.setup(ExperimentContext(Path('.')))
        result = algorithm.step(Batch({'input': torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 'target': torch.tensor([0, 1]), 'index': torch.tensor([4, 9])}), RunState(phase='train'))
        self.assertEqual(len(provider.inputs), 1)
        self.assertTrue(torch.equal(provider.inputs[0].sample_indices, torch.tensor([4, 9])))
        self.assertFalse(hasattr(provider.inputs[0], 'posterior_probabilities'))
        self.assertFalse(provider.inputs[0].logits.requires_grad)
        self.assertFalse(provider.inputs[0].per_sample_loss.requires_grad)
        self.assertEqual(result.metrics['selected_samples'], 2.0)
        self.assertEqual(result.metrics['treatment_provider_seen'], 1.0)

    def test_dual_t_pipeline_persists_snapshot_and_transition(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        self.assertIsInstance(pipeline, StandardNoisyERMPipeline)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = pipeline.prepare_transition(model=model, optimizer=optimizer, loader=loader, device=torch.device('cpu'), dataset='synthetic', split='train', run_dir=directory)
            self.assertIsNotNone(artifacts.snapshot)
            self.assertIsNotNone(artifacts.transition)
            self.assertTrue(Path(directory, 'posterior_snapshot.npz').is_file())
            self.assertTrue(Path(directory, 'transition_artifact.npz').is_file())
            restored = build_builtin_pipeline({'name': 'standard_noisy_erm', 'transition_estimator': {'name': 'dual_t'}})
            self.assertTrue(restored.load_artifacts(directory))
            self.assertEqual(restored.artifacts.snapshot.snapshot_hash, artifacts.snapshot.snapshot_hash)

    def test_fresh_training_can_prepare_transition(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            artifacts = pipeline.prepare_transition(model=model, optimizer=optimizer, loader=loader, device=torch.device('cpu'), dataset='synthetic', split='train', run_dir=directory)
            self.assertIsNotNone(artifacts.snapshot)
            self.assertIsNotNone(artifacts.transition)

    def test_resume_missing_transition_artifact_fails(self) -> None:
        pipeline, _, _, _ = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'artifact missing'):
                pipeline.restore_for_resume(directory, checkpoint_state=pipeline.state_dict(), component_states={}, dataset='synthetic', split='train')

    def test_resume_transition_hash_mismatch_fails(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.prepare_transition(model=model, optimizer=optimizer, loader=loader, device=torch.device('cpu'), dataset='synthetic', split='train', run_dir=directory)
            checkpoint_state = pipeline.state_dict()
            transition_path = Path(directory, 'transition_artifact.npz')
            with np.load(transition_path, allow_pickle=False) as data:
                matrix = data['matrix'].copy()
                metadata_json = data['metadata_json'].copy()
            matrix[0] = np.roll(matrix[0], 1)
            np.savez_compressed(transition_path, matrix=matrix, metadata_json=metadata_json)
            restored, _, _, _ = self._transition_fixture()
            with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                restored.restore_for_resume(directory, checkpoint_state=checkpoint_state, component_states={}, dataset='synthetic', split='train')

    def test_resume_artifact_provenance_mismatch_fails(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.prepare_transition(model=model, optimizer=optimizer, loader=loader, device=torch.device('cpu'), dataset='synthetic', split='train', run_dir=directory)
            restored, _, _, _ = self._transition_fixture()
            with self.assertRaisesRegex(ValueError, 'artifact provenance mismatch'):
                restored.restore_for_resume(directory, checkpoint_state=pipeline.state_dict(), component_states={}, dataset='different', split='train')

    def test_checkpoint_artifact_identity_verified(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.prepare_transition(model=model, optimizer=optimizer, loader=loader, device=torch.device('cpu'), dataset='synthetic', split='train', run_dir=directory)
            checkpoint_state = deepcopy(pipeline.state_dict())
            checkpoint_state['artifacts']['transition_artifact_hash'] = '0' * 64
            restored, _, _, _ = self._transition_fixture()
            with self.assertRaisesRegex(ValueError, 'checkpoint identity mismatch'):
                restored.restore_for_resume(directory, checkpoint_state=checkpoint_state, component_states={}, dataset='synthetic', split='train')

    def test_pipeline_state_roundtrip(self) -> None:
        pipeline = StandardNoisyERMPipeline()
        pipeline.state.phase = PipelinePhase.EVALUATE
        pipeline.state.cycle = 7
        pipeline.state.metadata['marker'] = 'restored'
        restored = StandardNoisyERMPipeline()
        restored.load_state_dict(pipeline.state_dict())
        self.assertEqual(restored.state.phase, PipelinePhase.EVALUATE)
        self.assertEqual(restored.state.cycle, 7)
        self.assertEqual(restored.state.metadata['marker'], 'restored')

    def test_pipeline_state_restored_before_training(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.prepare_transition(model=model, optimizer=optimizer, loader=loader, device=torch.device('cpu'), dataset='synthetic', split='train', run_dir=directory)
            pipeline.state.phase = PipelinePhase.EVALUATE
            pipeline.state.cycle = 7
            pipeline.state.metadata['marker'] = 'restored'
            restored, _, _, _ = self._transition_fixture()
            restored.restore_for_resume(directory, checkpoint_state=pipeline.state_dict(), component_states={}, dataset='synthetic', split='train')
            self.assertEqual(restored.state.phase, PipelinePhase.EVALUATE)
            self.assertEqual(restored.state.cycle, 7)
            self.assertEqual(restored.state.metadata['marker'], 'restored')

    def test_component_states_roundtrip(self) -> None:
        source = StandardNoisyERMPipeline(transition_estimator=_pipeline__StatefulEstimator(11))
        states = source.component_state_dict()
        restored = StandardNoisyERMPipeline(transition_estimator=_pipeline__StatefulEstimator(0))
        restored.load_component_states(states)
        self.assertEqual(restored.transition_estimator.value, 11)

    def test_legacy_checkpoint_without_pipeline_state(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.prepare_transition(model=model, optimizer=optimizer, loader=loader, device=torch.device('cpu'), dataset='synthetic', split='train', run_dir=directory)
            restored, _, _, _ = self._transition_fixture()
            warnings = restored.restore_for_resume(directory, checkpoint_state=None, component_states=None, dataset='synthetic', split='train')
            self.assertTrue(warnings)
            self.assertEqual(restored.artifacts.transition.artifact_hash, pipeline.artifacts.transition.artifact_hash)

    def test_binary_rcn_rejects_current_model_softmax_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, 'explicit noisy-label posterior producer'):
            build_builtin_pipeline({'name': 'standard_noisy_erm', 'weight_provider': {'name': 'binary_rcn_importance', 'rho_positive': 0.1, 'rho_negative': 0.2}})

# --- merged from test_runner_planning.py ---
import inspect

# --- merged from test_runner_planning.py ---
import json

# --- merged from test_runner_planning.py ---
from pathlib import Path

# --- merged from test_runner_planning.py ---
import unittest

# --- merged from test_runner_planning.py ---
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# --- merged from test_runner_planning.py ---
from lnl_toolbox.cli.main import _print_plan

# --- merged from test_runner_planning.py ---
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner

# --- merged from test_runner_planning.py ---
class _runner_planning_RunnerPlanningTest(unittest.TestCase):

    def test_supervised_runner_owns_plan_and_budget_override(self) -> None:
        config = {'execution': {'runner': 'supervised'}, 'data': {'name': 'cifar10'}, 'trainer': {'epochs': 10}, 'loss': {'name': 'gce'}}
        runner = resolve_runner(config)
        self.assertEqual(runner.describe(config).training_budget, '10 epochs')
        apply_epoch_override(config, 3)
        self.assertEqual(config['trainer']['epochs'], 3)

    def test_multistage_runner_uses_registered_budget_path(self) -> None:
        config = {'execution': {'runner': 'upm'}, 'method': 'upm', 'data': {'name': 'cifar10'}, 'upm': {'main': {'epochs': 100}}}
        apply_epoch_override(config, 4)
        self.assertEqual(config['upm']['main']['epochs'], 4)

    def test_cli_preview_contains_no_method_specific_dispatch(self) -> None:
        source = inspect.getsource(_print_plan)
        for method in ('upm', 'coteaching', 'dld', 'dividemix', 'lend'):
            self.assertNotIn(f'== "{method}"', source)

    def test_coteaching_plan_uses_nested_remember_schedule(self) -> None:
        config = {'method': 'coteaching', 'execution': {'runner': 'coteaching'}, 'data': {'name': 'cifar10'}, 'model': {'name': 'cifar_cnn8'}, 'loader': {'batch_size': 128}, 'optimizer': {'name': 'adam', 'lr': 0.001}, 'noise': {'rate': 0.2}, 'coteaching': {'model_count': 2, 'noise_rate': 0.2, 'gradual_epochs': 999, 'remember_schedule': {'gradual_epochs': 10}}, 'trainer': {'epochs': 200}}
        fields = {field.label: field.value for field in resolve_runner(config).describe(config).fields}
        self.assertEqual(fields['Co-teaching networks'], '2')
        self.assertEqual(fields['Co-teaching batch size'], '128')
        self.assertEqual(fields['Co-teaching optimizer'], 'adam')
        self.assertEqual(fields['Co-teaching learning rate'], '0.001')
        self.assertEqual(fields['Co-teaching Tk / gradual epochs'], '10')
        self.assertEqual(fields['Co-teaching tau / noise rate'], '0.2')

    def test_dld_plan_exposes_current_fidelity_and_provenance(self) -> None:
        config = {'method': 'dld', 'execution': {'runner': 'dld'}, 'data': {'name': 'cifar10'}, 'loader': {'batch_size': 256}, 'dld': {'fidelity': {'name': 'paper_oriented_v2_cosine_similarity', 'neighbor_metric': 'cosine_similarity', 'self_neighbor': 'include', 'divergence': 'kl_ps_to_pw'}, 'feature_extractor': {'source': 'external_checkpoint', 'model': {'name': 'resnet18'}, 'external': {'adapter': 'upm_main_best', 'checkpoint_sha256': 'a' * 64}}, 'precorrection': {'k_neighbors': 50}, 'diffusion': {'epochs': 15, 'timesteps': 100}, 'inference': {'steps': 5}}}
        plan = resolve_runner(config).describe(config)
        fields = {field.label: field.value for field in plan.fields}
        self.assertEqual(plan.training_budget, '15 (diffusion)')
        self.assertEqual(fields['DLD feature source'], 'external_checkpoint')
        self.assertEqual(fields['DLD feature model'], 'resnet18')
        self.assertEqual(fields['DLD source adapter'], 'upm_main_best')
        self.assertEqual(fields['DLD checkpoint identity'], 'a' * 64)
        self.assertEqual(fields['DLD neighbors'], 'K=50')
        self.assertEqual(fields['DLD neighbor metric'], 'cosine_similarity')
        self.assertEqual(fields['DLD self-neighbor'], 'include')
        self.assertEqual(fields['DLD divergence'], 'kl_ps_to_pw')
        self.assertEqual(fields['DLD timesteps'], '100')
        self.assertEqual(fields['DLD inference steps'], '5')

    def test_dividemix_and_lend_plans_expose_training_contracts(self) -> None:
        dividemix = {'method': 'dividemix', 'execution': {'runner': 'dividemix'}, 'data': {'name': 'cifar10'}, 'loader': {'batch_size': 128}, 'optimizer': {'name': 'sgd', 'lr': 0.02}, 'dividemix': {'warmup': {'epochs': 10}, 'training': {'epochs': 300}, 'gmm': {'threshold': 0.5, 'loss_history': {'name': 'official_auto'}}, 'mixmatch': {'temperature': 0.5, 'mixup_alpha': 4.0}, 'objective': {'lambda_u': 25.0, 'rampup_epochs': 16}, 'inference': {'ensemble': 'official_logits_sum'}}}
        plan = resolve_runner(dividemix).describe(dividemix)
        fields = {field.label: field.value for field in plan.fields}
        self.assertEqual(plan.training_budget, '10/300/310 (warmup/main/total)')
        self.assertEqual(fields['DivideMix loss history'], 'official_auto')
        self.assertEqual(fields['DivideMix lambda_u'], '25.0')
        self.assertEqual(fields['DivideMix ramp-up epochs'], '16')
        self.assertEqual(fields['DivideMix ensemble'], 'official_logits_sum')
        lend = {'method': 'lend', 'execution': {'runner': 'lend'}, 'data': {'name': 'cifar10'}, 'loader': {'batch_size': 256}, 'optimizer': {'name': 'sgd', 'lr': 0.05}, 'lend': {'graph': {'k': 8, 'gamma': 1.0, 'metric': 'inner_product', 'normalize_features': False, 'zero_degree_policy': 'self_loop'}, 'dilution': {'alpha': 0.99, 'policy': 'fixed_steps', 'steps': 10}, 'history': {'beta': 0.9, 'first_observation': 'current'}, 'selection': {'rule': 'noisy_equals_diluted_argmax', 'reduction': 'batch_mean', 'empty_batch': 'skip_update'}, 'training': {'epochs': 200}}}
        plan = resolve_runner(lend).describe(lend)
        fields = {field.label: field.value for field in plan.fields}
        self.assertEqual(plan.training_budget, '200 (LEND)')
        self.assertEqual(fields['LEND batch size'], '256')
        self.assertEqual(fields['LEND optimizer'], 'sgd')
        self.assertEqual(fields['LEND learning rate'], '0.05')
        self.assertIn('zero_degree=self_loop', fields['LEND graph'])
        self.assertIn('first=current', fields['LEND history'])

    def test_public_workflow_configs_are_declared_as_package_data(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
        data_files = pyproject['tool']['setuptools']['data-files']
        declared = {path for paths in data_files.values() for path in paths}
        manifest = json.loads((root / 'src/lnl_toolbox/cli/data/recipe_catalog.json').read_text(encoding='utf-8'))
        catalog_recipes = set(manifest['recipes']) | set(manifest.get('conditional', []))
        self.assertEqual(catalog_recipes - declared, set())
        for path in catalog_recipes:
            self.assertTrue((root / path).is_file(), path)

# --- merged from test_experiment_service.py ---
import json

# --- merged from test_experiment_service.py ---
from pathlib import Path

# --- merged from test_experiment_service.py ---
import tempfile

# --- merged from test_experiment_service.py ---
import unittest

# --- merged from test_experiment_service.py ---
from unittest.mock import patch

# --- merged from test_experiment_service.py ---
from unittest.mock import Mock

# --- merged from test_experiment_service.py ---
import yaml

# --- merged from test_experiment_service.py ---
from lnl_toolbox.training.service import ExperimentService

# --- merged from test_experiment_service.py ---
class _experiment_service__FakeRunner:
    name = 'fake'

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, config, output_dir=None, resume=None):
        self.calls += 1
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / 'final_metrics.json').write_text(json.dumps({'test_accuracy': 0.8}), encoding='utf-8')
        return root

# --- merged from test_experiment_service.py ---
class _experiment_service_ExperimentServiceTest(unittest.TestCase):

    def test_preflight_reuses_config_and_data_validation_without_running(self) -> None:
        data_service = Mock()
        service = ExperimentService(data_service=data_service)
        config = {'schema_version': 1, 'kind': 'experiment', 'execution': {'runner': 'fake'}, 'data': {'name': 'cifar10', 'root': 'missing'}}
        runtime = {**config, 'schema_version': 1, 'kind': 'experiment'}
        runner = object()
        with patch('lnl_toolbox.catalog.validate_config', return_value=runner) as validate:
            self.assertIs(service.preflight(config), runner)
        validate.assert_called_once_with(runtime, check_data=False)
        data_service.validate_config.assert_called_once_with(runtime)
        data_service.reset_mock()
        with patch('lnl_toolbox.catalog.validate_config', return_value=runner):
            self.assertIs(service.preflight(config, check_data=False), runner)
        data_service.validate_config.assert_not_called()

    def test_service_invokes_runner_and_writes_standard_artifacts(self) -> None:
        runner = _experiment_service__FakeRunner()
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.training.service.resolve_runner', return_value=runner):
            root = ExperimentService().run({'schema_version': 1, 'kind': 'experiment', 'seed': 2, 'method': 'fake', 'execution': {'runner': 'fake'}, 'data': {'name': 'synthetic_multiclass'}, 'noise': {'name': 'symmetric', 'rate': 0.2}}, directory, recipe='fake-smoke')
            result = json.loads((root / 'final_metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(runner.calls, 1)
            self.assertEqual(result['recipe'], 'fake-smoke')
            self.assertTrue((root / 'resolved_config.yaml').is_file())
            resolved = yaml.safe_load((root / 'resolved_config.yaml').read_text(encoding='utf-8'))
            self.assertEqual(resolved['noise']['name'], 'symmetric')
            self.assertNotIn('type', resolved['noise'])
            self.assertTrue((root / 'environment.json').is_file())

    def test_resume_of_completed_run_is_strict_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'resolved_config.yaml').write_text(yaml.safe_dump({'method': 'fake'}), encoding='utf-8')
            final = root / 'final_metrics.json'
            final.write_text(json.dumps({'status': 'completed', 'completed': True}), encoding='utf-8')
            before = final.read_bytes()
            with patch('lnl_toolbox.training.service.resolve_runner') as resolver:
                self.assertEqual(ExperimentService().resume(root), root.resolve())
            resolver.assert_not_called()
            self.assertEqual(final.read_bytes(), before)

    def test_config_compatibility_loads_dataset_capabilities_once(self) -> None:
        data_service = Mock()
        capabilities = object()
        data_service.capabilities.return_value = capabilities
        data_service.apply.side_effect = lambda config, alias: {**config, 'local_dataset': {'alias': alias}}
        service = ExperimentService(data_service=data_service)
        runner = Mock(name='runner')
        first = Mock()
        second = Mock()
        with patch('lnl_toolbox.training.service.resolve_runner', return_value=runner) as resolver, patch.object(service, '_resolve_for_capabilities', side_effect=(first, second)) as resolve:
            value = service.list_config_compatibility('lab', {'paper-a': {'execution': {'runner': 'a'}}, 'paper-b': {'execution': {'runner': 'b'}}})
        self.assertEqual(value, (('paper-a', first), ('paper-b', second)))
        data_service.capabilities.assert_called_once_with('lab', persist=False)
        self.assertEqual(data_service.apply.call_count, 2)
        self.assertEqual(resolver.call_count, 2)
        self.assertEqual(resolve.call_count, 2)
        self.assertIs(resolve.call_args_list[0].args[0], capabilities)
