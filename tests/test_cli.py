"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_cli.py ---
from pathlib import Path

# --- merged from test_cli.py ---
import tempfile

# --- merged from test_cli.py ---
import unittest

# --- merged from test_cli.py ---
from unittest.mock import patch

# --- merged from test_cli.py ---
import yaml

# --- merged from test_cli.py ---
from lnl_toolbox.cli import PromptSession, TrainingSelection

# --- merged from test_cli.py ---
import lnl_toolbox.cli as cli_shared

# --- merged from test_cli.py ---
from lnl_toolbox.cli import clean_train as clean_cli

# --- merged from test_cli.py ---
from lnl_toolbox.cli import inspect_data as inspect_cli

# --- merged from test_cli.py ---
from lnl_toolbox.cli import make_noise as noise_cli

# --- merged from test_cli.py ---
from lnl_toolbox.cli import train as train_cli

# --- merged from test_cli.py ---
def _cli_scripted_session(values: list[str]) -> tuple[PromptSession, list[str]]:
    answers = iter(values)
    output: list[str] = []
    return (PromptSession(read=lambda _: next(answers), write=output.append), output)

# --- merged from test_cli.py ---
class _cli_PromptSessionTest(unittest.TestCase):

    def test_clean_template_discovery_excludes_noisy_configs(self) -> None:
        names = {path.name for path, _ in cli_shared._experiment_templates(clean=True)}
        self.assertNotIn('cifar10_symmetric_ce_smoke.yaml', names)

    def test_choice_and_number_retry(self) -> None:
        session, output = _cli_scripted_session(['invalid', '2', 'bad', '0', '3'])
        self.assertEqual(session.choose('choice', [('First', 'first'), ('Second', 'second')]), 'second')
        self.assertEqual(session.integer('count', minimum=1), 3)
        self.assertTrue(any(('选项无效' in item for item in output)))
        self.assertTrue(any(('请输入整数' in item for item in output)))

    def test_loss_wizard_builds_gce_and_nested_apl(self) -> None:
        gce_session, _ = _cli_scripted_session(['gce', '0.6'])
        self.assertEqual(cli_shared._prompt_loss(gce_session, {'name': 'ce'}), {'name': 'gce', 'q': 0.6})
        apl_session, _ = _cli_scripted_session(['apl', 'mae', '', '2', '0.5', ''])
        self.assertEqual(cli_shared._prompt_loss(apl_session, {'name': 'ce'}), {'name': 'apl', 'alpha': 2.0, 'beta': 0.5, 'active': {'name': 'nce', 'eps': 1e-08}, 'passive': {'name': 'mae', 'scale': 2.0}})

    def test_apl_wizard_retries_zero_weights(self) -> None:
        session, _ = _cli_scripted_session(['apl', 'rce', '', '0', '2', '0', '0.5', ''])
        self.assertEqual(cli_shared._prompt_loss(session, {'name': 'ce'}), {'name': 'apl', 'alpha': 2.0, 'beta': 0.5, 'active': {'name': 'nce', 'eps': 1e-08}, 'passive': {'name': 'rce', 'log_zero': -4.0}})

    def test_clean_scheduler_wizard_builds_multistep(self) -> None:
        session, _ = _cli_scripted_session(['multistep', '5,10', '0.2'])
        self.assertEqual(cli_shared._prompt_scheduler(session, {'name': 'none'}, 20), {'name': 'multistep', 'milestones': [5, 10], 'gamma': 0.2})

    def test_noise_wizard_builds_generated_and_external_modes(self) -> None:
        generated, _ = _cli_scripted_session(['2', '', '2', '0.3', '9'])
        self.assertEqual(cli_shared._prompt_noise(generated, None, 1), {'name': 'pairflip', 'rate': 0.3, 'seed': 9, 'manifest_filename': 'noise_manifest.npz'})
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'source.npz'
            manifest.touch()
            external, _ = _cli_scripted_session(['3', '', str(manifest)])
            self.assertEqual(cli_shared._prompt_noise(external, None, 1), {'manifest': str(manifest), 'manifest_filename': 'noise_manifest.npz'})

    def test_training_wizard_keeps_template_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = {'seed': 7, 'data': {'name': 'cifar10', 'root': str(root)}, 'loader': {'batch_size': 16}, 'model': {'name': 'tiny_cnn', 'width': 8}, 'loss': {'name': 'ce'}, 'optimizer': {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.0}, 'trainer': {'epochs': 2, 'device': 'cpu'}}
            session, _ = _cli_scripted_session(['', '', '', '', 'gce', '0.5', '', '', '', '', '', '', '', '', '', 'y'])
            with patch.object(cli_shared, '_choose_template', return_value=(root / 'base.yaml', template)):
                selection = cli_shared.prompt_training_selection(session, clean=False)
            self.assertIsNotNone(selection)
            assert selection is not None
            self.assertEqual(selection.config['loss'], {'name': 'gce', 'q': 0.5})
            self.assertNotIn('noise', selection.config)
            self.assertEqual(template['loss'], {'name': 'ce'})

# --- merged from test_cli.py ---
class _cli_CliEntryPointTest(unittest.TestCase):

    def test_train_help_lists_t_revision(self) -> None:
        help_text = train_cli.build_parser().format_help()
        self.assertIn('t_revision', help_text)

    def test_missing_training_config_has_clear_error(self) -> None:
        missing = Path('definitely-missing-t-revision-config.yaml')
        with self.assertRaisesRegex(FileNotFoundError, 'does not exist'):
            train_cli.main(['--config', str(missing)])

    def test_epochs_override_targets_t_revision_final_stage(self) -> None:
        config = {'method': 't_revision', 'trainer': {'device': 'cpu'}, 't_revision': {'revision': {'epochs': 2}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            with patch.object(train_cli, 'run_experiment') as run:
                self.assertEqual(train_cli.main(['--config', str(path), '--epochs', '5']), 0)
        called = run.call_args.args[0]
        self.assertEqual(called['t_revision']['revision']['epochs'], 5)
        self.assertNotIn('epochs', called['trainer'])

    def test_epochs_override_rejects_ambiguous_staged_method(self) -> None:
        config = {'method': 'dual_t', 'posterior_stage': {'epochs': 1}, 'final_stage': {'epochs': 1}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'ambiguous.*dual_t'):
                train_cli.main(['--config', str(path), '--epochs', '5'])

    def test_all_no_argument_entrypoints_dispatch_interactively(self) -> None:
        session, _ = _cli_scripted_session([])
        selection = TrainingSelection({'trainer': {'epochs': 1}}, Path('template.yaml'))
        with patch.object(train_cli, 'prompt_training_selection', return_value=selection), patch.object(train_cli, 'run_experiment') as run:
            self.assertEqual(train_cli.main([], session), 0)
            run.assert_called_once_with(selection.config, None, None)
        seed_selection = TrainingSelection({'output_root': 'artifacts/runs'}, Path('template.yaml'), Path('suite'), seeds=[1, 2])
        with patch.object(clean_cli, 'prompt_training_selection', return_value=seed_selection), patch.object(clean_cli, 'run_seed_suite') as run_suite:
            self.assertEqual(clean_cli.main([], session), 0)
            run_suite.assert_called_once_with(seed_selection.config, [1, 2], Path('suite'))
        with patch.object(inspect_cli, '_interactive', return_value=('cifar10', Path('data'), 'all')), patch.object(inspect_cli, '_execute') as inspect:
            self.assertEqual(inspect_cli.main([], session), 0)
            inspect.assert_called_once_with('cifar10', Path('data'), 'all')
        noise_values = (Path('labels.npy'), Path('noise.npz'), 'symmetric', 0.2, 10, 1, 'cifar10')
        with patch.object(noise_cli, '_interactive', return_value=noise_values), patch.object(noise_cli, '_execute') as generate:
            self.assertEqual(noise_cli.main([], session), 0)
            generate.assert_called_once_with(*noise_values)

    def test_argument_mode_bypasses_terminal_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / 'config.yaml'
            config_path.write_text(yaml.safe_dump({'trainer': {'epochs': 2}}), encoding='utf-8')
            session = PromptSession(read=lambda _: self.fail('stdin should not be read'))
            with patch.object(train_cli, 'run_experiment') as run:
                result = train_cli.main(['--config', str(config_path), '--epochs', '3', '--output-dir', 'out'], session)
            self.assertEqual(result, 0)
            called_config, output_dir, resume = run.call_args.args
            self.assertEqual(called_config['trainer']['epochs'], 3)
            self.assertEqual(output_dir, Path('out'))
            self.assertIsNone(resume)

    def test_cancelled_prompt_returns_130(self) -> None:

        def cancel(_: str) -> str:
            raise EOFError
        output: list[str] = []
        result = inspect_cli.main([], PromptSession(read=cancel, write=output.append))
        self.assertEqual(result, 130)
        self.assertTrue(any(('已取消' in item for item in output)))

    def test_clean_argument_mode_rejects_resume_with_seed_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / 'config.yaml'
            config_path.write_text(yaml.safe_dump({'trainer': {'epochs': 2}}), encoding='utf-8')
            with self.assertRaises(SystemExit) as raised:
                clean_cli.main(['--config', str(config_path), '--resume', 'last.pt', '--seeds', '1', '2'])
            self.assertEqual(raised.exception.code, 2)

# --- merged from test_unified_cli.py ---
from contextlib import redirect_stderr, redirect_stdout

# --- merged from test_unified_cli.py ---
from io import StringIO

# --- merged from test_unified_cli.py ---
from pathlib import Path

# --- merged from test_unified_cli.py ---
import json

# --- merged from test_unified_cli.py ---
import os

# --- merged from test_unified_cli.py ---
import subprocess

# --- merged from test_unified_cli.py ---
import sys

# --- merged from test_unified_cli.py ---
import tempfile

# --- merged from test_unified_cli.py ---
import unittest

# --- merged from test_unified_cli.py ---
from unittest.mock import Mock, patch

# --- merged from test_unified_cli.py ---
import yaml

# --- merged from test_unified_cli.py ---
from lnl_toolbox import catalog as catalog_module

# --- merged from test_unified_cli.py ---
from lnl_toolbox.catalog import discover_recipes, default_paper_config, find_project_root, load_papers, load_yaml, load_recipe_config, mentornet_preparation_status, recipe_by_id, resolve_config_paths, select_paper_config, validate_config

from lnl_toolbox.models.mentornet import MentorNet

from lnl_toolbox.training.mentor_artifacts import MentorArtifact

# --- merged from test_unified_cli.py ---
from lnl_toolbox.cli.main import main

# --- merged from test_unified_cli.py ---
from lnl_toolbox.data.profile import DatasetProfile, KnowledgeState, Modality, NoiseKnowledge, resolve_dataset_capabilities

# --- merged from test_unified_cli.py ---
from lnl_toolbox.training.compatibility import CompatibilityReason, CompatibilityResult, CompatibilityStatus

# --- merged from test_unified_cli.py ---
from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE, DatasetStatusReport

# --- merged from test_unified_cli.py ---
from lnl_toolbox.training.runners import resolve_runner, runner_names

from lnl_toolbox.training.service import ExperimentService

# --- merged from test_unified_cli.py ---
_unified_cli_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_unified_cli.py ---
class _unified_cli_RunnerResolutionTest(unittest.TestCase):

    def test_all_public_runners_are_registered(self) -> None:
        self.assertEqual(set(runner_names()), {'binary', 'cal', 'ca2c', 'clean', 'coteaching', 'cwd', 'dld', 'dual_t', 'fine', 'importance_reweighting', 'instance_transition', 'multi_model', 'l2rw', 'lend', 'mc_ldce', 'pcse', 'supervised', 'cnlcu', 'dividemix', 't_revision', 'upm', 'volmin', 'volminnet'})

    def test_unknown_method_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, 'unknown method'):
            resolve_runner({'method': 'cotaching', 'data': {'name': 'cifar10'}})

    def test_dedicated_config_cannot_use_supervised_runner(self) -> None:
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            resolve_runner({'execution': {'runner': 'supervised'}, 'data': {'name': 'cifar100'}, 'fine': {'warmup_epochs': 1}})

    def test_legacy_special_sections_are_inferred(self) -> None:
        self.assertEqual(resolve_runner({'data': {}, 'cwd': {}}).name, 'cwd')
        self.assertEqual(resolve_runner({'data': {}, 'algorithm': {'name': 'jocor'}}).name, 'multi_model')

# --- merged from test_unified_cli.py ---
class _unified_cli_CatalogTest(unittest.TestCase):

    def test_installed_recipe_uses_distribution_file_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / 'venv'
            site_packages = prefix / 'lib' / 'site-packages'
            site_packages.mkdir(parents=True)
            relative = 'configs/experiment/cifar10_volminnet_smoke.yaml'
            entry = Path('../..') / 'share' / 'lnl-toolbox' / relative

            class Distribution:
                files = (entry,)

                @staticmethod
                def locate_file(value: Path) -> Path:
                    return site_packages / value
            with patch.object(catalog_module.metadata, 'distribution', return_value=Distribution()):
                installed = catalog_module._installed_recipe_path(relative)
            self.assertEqual(installed, (prefix / 'share' / 'lnl-toolbox' / relative).resolve())

    def test_every_builtin_recipe_has_explicit_valid_runner(self) -> None:
        recipes = discover_recipes(_unified_cli_ROOT)
        self.assertGreaterEqual(len(recipes), 39)
        for recipe in recipes:
            config = load_recipe_config(recipe)
            self.assertIn('execution', config, recipe.id)
            self.assertEqual(validate_config(config).name, recipe.runner, recipe.id)

    def test_paper_catalog_only_references_runnable_recipes(self) -> None:
        recipes = {item.id for item in discover_recipes(_unified_cli_ROOT, include_conditional=True)}
        papers = load_papers(_unified_cli_ROOT)
        self.assertGreaterEqual(len(papers), 10)
        for paper in papers:
            for item in paper.configs:
                self.assertIn(item.recipe_id, recipes)

    def test_every_paper_has_one_formal_default(self) -> None:
        papers = load_papers(_unified_cli_ROOT)
        self.assertEqual(len(papers), 26)
        defaults = [default_paper_config(paper, root=_unified_cli_ROOT) for paper in papers]
        self.assertEqual(len(defaults), 26)
        self.assertTrue(all((config.profile == 'reproduction' for config, _ in defaults)))
        self.assertEqual(len({paper.id for paper in papers}), 26)

    def test_manifest_excludes_local_untracked_yaml_and_conditional_recipes(self) -> None:
        recipes = {item.id for item in discover_recipes(_unified_cli_ROOT)}
        self.assertNotIn('cifar10-symmetric40-all-e5', recipes)
        self.assertNotIn('cifar10-symmetric40-small-loss-e5', recipes)
        self.assertFalse(any(('mentornet' in recipe for recipe in recipes)))
        self.assertIn('cifar10-pcse-reproduction', recipes)
        all_recipes = {item.id for item in discover_recipes(_unified_cli_ROOT, include_conditional=True)}
        self.assertIn('mentornet-dd-cifar100-symmetric04-smoke', all_recipes)
        self.assertIn('cifar10-pcse-reproduction', all_recipes)

    def test_public_recipe_catalog_is_small_without_hiding_internal_lookup(self) -> None:
        public = discover_recipes(_unified_cli_ROOT, public_only=True)
        self.assertEqual(len(public), 4)
        self.assertTrue(all((item.visibility == 'public' for item in public)))
        public_ids = {item.id for item in public}
        self.assertIn('cifar10-clean-smoke', public_ids)
        self.assertNotIn('fine-cifar100n-reproduction', public_ids)
        self.assertEqual(recipe_by_id('fine-cifar100n-reproduction', _unified_cli_ROOT).id, 'fine-cifar100n-reproduction')

    def test_method_specific_preflight_and_conditional_artifact(self) -> None:
        cnlcu = load_recipe_config(next((item for item in discover_recipes(_unified_cli_ROOT) if item.id == 'cifar10-cnlcu-soft-smoke')))
        self.assertEqual(validate_config(cnlcu).name, 'cnlcu')
        mentor = load_recipe_config(next((item for item in discover_recipes(_unified_cli_ROOT, include_conditional=True) if item.id == 'mentornet-dd-cifar100-symmetric04-smoke')))
        preparation = mentornet_preparation_status(
            resolve_config_paths(mentor, _unified_cli_ROOT),
            _unified_cli_ROOT,
            student_recipe='mentornet-dd-cifar100-symmetric04-smoke',
        )
        self.assertIn('lnl mentor prepare', preparation['commands']['prepare'])
        self.assertIn('lnl mentor train', preparation['commands']['train'])
        self.assertIn('mentornet_dd_teacher_cifar10_symmetric04.yaml', preparation['commands']['prepare'])
        mentor['pipeline']['weight_provider']['artifact_path'] = str(_unified_cli_ROOT / 'data/mentornet/missing-artifact-for-test.pt')
        with self.assertRaisesRegex(ValueError, 'conditional.*MentorArtifact') as raised:
            validate_config(resolve_config_paths(mentor, _unified_cli_ROOT))
        self.assertIn('MentorArtifact: NOT READY', str(raised.exception))
        pcse = load_recipe_config(next((item for item in discover_recipes(_unified_cli_ROOT, include_conditional=True) if item.id == 'cifar10-pcse-reproduction')))
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('LNL_PCSE_SOURCE_RUN', None)
            self.assertEqual(
                validate_config(resolve_config_paths(pcse, _unified_cli_ROOT)).name,
                'pcse',
            )

    def test_mentornet_ready_requires_a_valid_artifact_and_keeps_cross_dataset_contract(self) -> None:
        recipe = recipe_by_id('mentornet-dd-cifar100-symmetric04-smoke', _unified_cli_ROOT)
        config = load_recipe_config(recipe)
        self.assertEqual(config['data']['name'], 'cifar100')
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / 'mentor_artifact.pt'
            provider = config['pipeline']['weight_provider']
            provider['artifact_path'] = str(artifact_path)
            status = mentornet_preparation_status(config, _unified_cli_ROOT, student_recipe=recipe.id)
            self.assertFalse(status['artifact_ready'])
            artifact_path.write_bytes(b'not a MentorArtifact')
            status = mentornet_preparation_status(config, _unified_cli_ROOT, student_recipe=recipe.id)
            self.assertFalse(status['artifact_ready'])
            self.assertTrue(status['artifact_error'])
            model = MentorNet(num_labels=1)
            MentorArtifact.create(
                architecture=model.architecture(),
                feature_schema={'label': 'fixed_zero'},
                source={'dataset': 'cifar10', 'role': 'trusted_mentor'},
                model_state=model.state_dict(),
            ).save(artifact_path)
            status = mentornet_preparation_status(config, _unified_cli_ROOT, student_recipe=recipe.id)
            self.assertTrue(status['artifact_ready'])
            self.assertIn('cifar10', Path(status['teacher_config']).name)
            self.assertEqual(validate_config(config).name, 'supervised')

    def test_mentor_teacher_config_is_packaged(self) -> None:
        relative = 'configs/experiment/mentornet_dd_teacher_cifar10_symmetric04.yaml'
        self.assertTrue((_unified_cli_ROOT / relative).is_file())
        self.assertIn(
            f'"{relative}"',
            (_unified_cli_ROOT / 'pyproject.toml').read_text(encoding='utf-8'),
        )

    def test_multiple_paper_variants_require_selection(self) -> None:
        paper = next((item for item in load_papers(_unified_cli_ROOT) if item.id == 'apl'))
        with self.assertRaisesRegex(ValueError, 'choose --variant'):
            select_paper_config(paper, profile='reproduction', root=_unified_cli_ROOT)

    def test_paths_are_resolved_against_project_not_cwd(self) -> None:
        config = {'data': {'name': 'cifar10', 'root': 'data/cifar10'}, 'output_root': 'artifacts/runs'}
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                import os
                os.chdir(directory)
                resolved = resolve_config_paths(config, _unified_cli_ROOT)
            finally:
                os.chdir(previous)
        self.assertEqual(Path(resolved['data']['root']), (_unified_cli_ROOT / 'data/cifar10').resolve())
        self.assertEqual(Path(resolved['output_root']), (_unified_cli_ROOT / 'artifacts/runs').resolve())

    def test_packaged_recipe_without_project_uses_caller_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(find_project_root(), Path(directory).resolve())
            finally:
                os.chdir(previous)

# --- merged from test_unified_cli.py ---
class _unified_cli_UnifiedCliTest(unittest.TestCase):

    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout, stderr = (StringIO(), StringIO())
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(list(arguments))
        return (code, stdout.getvalue(), stderr.getvalue())

    def test_unified_mentor_commands_reuse_prepare_and_train_producers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_dir = root / 'mentor'
            feature_path = feature_dir / 'mentor_features.npz'
            artifact_path = feature_dir / 'mentor_artifact.pt'
            config_path = root / 'teacher.yaml'
            config_path.write_text(yaml.safe_dump({
                'schema_version': 1,
                'kind': 'mentor_artifact',
                'seed': 3,
                'feature_data': str(feature_path),
                'data': {'name': 'synthetic_binary_2d', 'train_size': 18, 'validation_size': 0, 'test_size': 6},
                'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 3},
                'loader': {'batch_size': 6, 'num_workers': 0},
                'student_model': {'name': 'mlp', 'width': 8},
                'student_optimizer': {'name': 'sgd', 'lr': 0.1},
                'student_trainer': {'trusted_size': 18, 'epochs': 1, 'device': 'cpu'},
                'model': {'num_labels': 1, 'hidden_size': 2, 'sequence_length': 2, 'label_embedding_dim': 2, 'epoch_embedding_dim': 2, 'dense_size': 4},
                'optimizer': {'name': 'adam', 'lr': 0.001},
                'trainer': {'epochs': 1, 'device': 'cpu'},
                'execution': {'runner': 'mentor_artifact'},
            }, sort_keys=False), encoding='utf-8')
            code, output, error = self.invoke(
                'mentor', 'prepare', '--config', str(config_path),
                '--output-dir', str(feature_dir), '--project-root', str(root),
            )
            self.assertEqual(code, 0, error)
            self.assertIn('Mentor features: READY', output)
            self.assertTrue(feature_path.is_file())
            code, output, error = self.invoke(
                'mentor', 'train', '--config', str(config_path),
                '--output', str(artifact_path), '--project-root', str(root),
            )
            self.assertEqual(code, 0, error)
            self.assertIn('MentorArtifact: READY', output)
            self.assertTrue(artifact_path.is_file())
            MentorArtifact.load(artifact_path)
            student = load_recipe_config(
                recipe_by_id(
                    'mentornet-dd-cifar100-symmetric04-smoke',
                    _unified_cli_ROOT,
                )
            )
            student['pipeline']['weight_provider']['artifact_path'] = str(artifact_path)
            student_path = root / 'student.yaml'
            student_path.write_text(
                yaml.safe_dump(student, sort_keys=False), encoding='utf-8'
            )
            code, output, error = self.invoke(
                'validate', '--config', str(student_path),
                '--project-root', str(_unified_cli_ROOT),
            )
            self.assertEqual(code, 0, error)
            self.assertIn('supervised', output)
            code, output, error = self.invoke(
                'run', '--config', str(student_path), '--dry-run',
                '--no-check-data', '--project-root', str(_unified_cli_ROOT),
            )
            self.assertEqual(code, 0, error)
            self.assertIn('supervised', output)

    def test_mentor_status_reports_readiness_without_starting_student(self) -> None:
        recipe = Mock(id='mentor-smoke')
        status = {
            'status': 'not_ready', 'artifact_ready': False,
            'artifact_path': 'mentor_artifact.pt', 'artifact_error': None,
            'feature_ready': False,
            'commands': {'prepare': 'lnl mentor prepare ...', 'train': 'lnl mentor train ...', 'student': 'lnl run ...'},
        }
        with patch('lnl_toolbox.cli.main.recipe_by_id', return_value=recipe), patch(
            'lnl_toolbox.cli.main.load_recipe_config', return_value={}
        ), patch(
            'lnl_toolbox.cli.main.resolve_config_paths', return_value={}
        ), patch(
            'lnl_toolbox.cli.main.mentornet_preparation_status', return_value=status
        ):
            code, output, error = self.invoke('mentor', 'status', '--recipe', 'mentor-smoke')
        self.assertEqual(code, 1, error)
        self.assertIn('MentorArtifact: NOT_READY', output)
        self.assertIn('prepare: lnl mentor prepare', output)

    @staticmethod
    def compatibility_profile() -> DatasetProfile:
        return DatasetProfile(dataset='fixture', adapter='fixture', source='fixture-root', task='classification', modality=Modality.IMAGE, num_classes=3, input_shape=(8, 8, 3), channels=3, sample_counts_by_split=(('train', 6), ('test', 3)), available_splits=('train', 'test'), class_names=('a', 'b', 'c'), class_distribution_by_split=(('train', (2, 2, 2)), ('test', (1, 1, 1))), observed_train_labels=KnowledgeState.AVAILABLE, clean_train_labels=KnowledgeState.UNKNOWN, clean_validation_labels=KnowledgeState.UNAVAILABLE, stable_indices=KnowledgeState.AVAILABLE, dataset_fingerprint='d' * 64, split_fingerprints=(('train', 'a' * 64), ('test', 'b' * 64)), noise=NoiseKnowledge())

    def test_data_inspect_exposes_profile_in_human_and_json_formats(self) -> None:
        profile = self.compatibility_profile()
        report = DatasetStatusReport('fixture', 'fixture', 'ready', location='fixture-root', train_samples=6, test_samples=3, classes=3, fingerprint=profile.dataset_fingerprint, profile=profile)
        with patch.object(DEFAULT_DATA_SERVICE, 'inspect', return_value=report):
            code, output, error = self.invoke('data', 'inspect', 'fixture')
            self.assertEqual(code, 0, error)
            self.assertIn('Modality         image', output)
            self.assertIn('Clean labels     unknown', output)
            self.assertIn('Profile fingerprint', output)
            code, output, error = self.invoke('data', 'inspect', 'fixture', '--format', 'json')
        self.assertEqual(code, 0, error)
        value = json.loads(output)
        self.assertEqual(value['profile']['modality'], 'image')
        self.assertEqual(value['profile']['noise']['rate']['status'], 'unknown')

    def test_data_declare_uses_persisted_service_contract(self) -> None:
        capabilities = Mock()
        capabilities.clean_train_labels = KnowledgeState.AVAILABLE
        capabilities.noise_status.value = 'noisy'
        capabilities.noise_origin.value = 'native'
        capabilities.noise_rate.status.value = 'estimated'
        with patch.object(DEFAULT_DATA_SERVICE, 'update_declarations', return_value=capabilities) as update:
            code, output, error = self.invoke('data', 'declare', 'fixture', '--clean-train-labels', 'available', '--noise-status', 'noisy', '--noise-origin', 'native', '--noise-rate', '0.2', '--noise-rate-status', 'estimated', '--noise-rate-provenance', 'user-audit')
        self.assertEqual(code, 0, error)
        updates = update.call_args.args[1]
        self.assertEqual(updates['clean_train_labels'], 'available')
        self.assertEqual(updates['noise_rate']['status'], 'estimated')
        self.assertEqual(updates['noise_rate']['provenance'], 'user-audit')
        self.assertIn('Declarations updated', output)

    def test_methods_compatible_has_grouped_and_machine_readable_results(self) -> None:
        results = (CompatibilityResult(CompatibilityStatus.COMPATIBLE, 'upm', 'fixture'), CompatibilityResult(CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS, 'coteaching', 'fixture', reasons=(CompatibilityReason('requires_noise_rate_prior', 'prior required'),), required_user_inputs=('noise_rate_prior',)), CompatibilityResult(CompatibilityStatus.INCOMPATIBLE, 'importance_reweighting', 'fixture', reasons=(CompatibilityReason('unsupported_modality', 'image unsupported'),)))
        with patch('lnl_toolbox.cli.main.ExperimentService.list_compatible_methods', return_value=results):
            code, output, error = self.invoke('methods', 'compatible', '--dataset', 'fixture')
            self.assertEqual(code, 0, error)
            self.assertIn('Compatible:', output)
            self.assertIn('Requires additional input:', output)
            self.assertIn('Unavailable:', output)
            self.assertIn('requires_noise_rate_prior', output)
            code, output, error = self.invoke('methods', 'compatible', '--dataset', 'fixture', '--format', 'json')
        self.assertEqual(code, 0, error)
        value = json.loads(output)
        self.assertEqual(value[1]['required_user_inputs'], ['noise_rate_prior'])
        self.assertEqual(value[2]['reason_codes'], ['unsupported_modality'])

    def test_compatibility_failure_prevents_validate_dry_run_and_run(self) -> None:
        error = ValueError("method 'upm' is not ready for dataset 'heart': incompatible; unsupported_modality: image required")
        with patch('lnl_toolbox.cli.main.ExperimentService.preflight', side_effect=error), patch('lnl_toolbox.cli.main.ExperimentService.run') as runner:
            code, _, stderr = self.invoke('validate', '--recipe', 'cifar10-upm-smoke', '--check-data')
            self.assertEqual(code, 2)
            self.assertIn('unsupported_modality', stderr)
            code, _, stderr = self.invoke('run', '--recipe', 'cifar10-upm-smoke', '--dry-run')
            self.assertEqual(code, 2)
            self.assertIn('unsupported_modality', stderr)
            code, _, stderr = self.invoke('run', '--recipe', 'cifar10-upm-smoke')
            self.assertEqual(code, 2)
            self.assertIn('unsupported_modality', stderr)
        runner.assert_not_called()

    def test_plain_ce_validate_check_data_reports_compatible(self) -> None:
        data_service = Mock()
        data_service.capabilities.return_value = resolve_dataset_capabilities(
            self.compatibility_profile()
        )
        service = ExperimentService(data_service=data_service)
        with patch('lnl_toolbox.cli.main.ExperimentService', return_value=service):
            code, output, error = self.invoke(
                'validate', '--recipe', 'cifar10-symmetric-ce-smoke', '--check-data'
            )
        self.assertEqual(code, 0, error)
        self.assertIn('Compatibility:\n  COMPATIBLE', output)
        self.assertNotIn('NOT_CHECKED', output)
        data_service.validate_config.assert_called_once()

    def test_web_command_starts_main_page_and_supports_no_open(self) -> None:
        with patch('lnl_toolbox.cli.main.subprocess.call', return_value=0) as call:
            code, _output, error = self.invoke('web', '--host', '127.0.0.1', '--port', '9000')
        self.assertEqual(code, 0, error)
        command = call.call_args.args[0]
        self.assertIn('command_console.py', command[1])
        self.assertIn('--open', command)
        self.assertIn('9000', command)
        with patch('lnl_toolbox.cli.main.subprocess.call', return_value=0) as call:
            code, _output, error = self.invoke('web', '--no-open')
        self.assertEqual(code, 0, error)
        self.assertNotIn('--open', call.call_args.args[0])

    def test_local_dataset_registration_and_recipe_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'cifar100'
            source.mkdir()
            environment = {'LNL_DATA_CATALOG': str(root / 'datasets.json')}
            with patch.dict(os.environ, environment, clear=False):
                code, output, error = self.invoke('data', 'register', 'lab-cifar100', '--adapter', 'cifar100', '--root', str(source))
                self.assertEqual(code, 0, error)
                self.assertIn('state: registered', output)
                self.assertIn('does not prove trainability', output)
                code, output, error = self.invoke('data', 'list')
                self.assertEqual(code, 0, error)
                self.assertIn('lab-cifar100', output)
                self.assertIn('incomplete', output)
                code, output, error = self.invoke('data', 'status', 'lab-cifar100')
                self.assertEqual(code, 0, error)
                self.assertIn('Status           INCOMPLETE', output)
                code, output, error = self.invoke('data', 'path', 'lab-cifar100')
                self.assertEqual(code, 0, error)
                self.assertIn(str(source.resolve()), output)
                code, output, error = self.invoke('run', 'cifar10-clean-smoke', '--data', 'lab-cifar100', '--dry-run', '--no-check-data')
                self.assertEqual(code, 0, error)
                self.assertIn('Dataset: cifar100', output)
                code, output, error = self.invoke('data', 'remove', 'lab-cifar100')
                self.assertEqual(code, 0, error)
                self.assertIn('Removed', output)

    def test_local_dataset_registration_requires_adapter_specific_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = {'LNL_DATA_CATALOG': str(Path(directory) / 'datasets.json')}
            with patch.dict(os.environ, environment, clear=False):
                code, _, error = self.invoke('data', 'register', 'human-noise', '--adapter', 'cifar10n', '--root', directory)
                self.assertEqual(code, 2)
                self.assertIn('requires --labels', error)
                source = Path(directory) / 'heart.dat'
                rows = []
                for row in range(40):
                    features = [f'{(row + column) % 11 + column / 10:.1f}' for column in range(13)]
                    rows.append(' '.join(features + [str(1 + row % 2)]))
                source.write_text('\n'.join(rows) + '\n', encoding='utf-8')
                code, output, error = self.invoke('data', 'register', 'heart', '--adapter', 'uci_binary', '--path', str(source))
                self.assertEqual(code, 0, error)
                self.assertIn('heart: uci_binary', output)
                code, output, error = self.invoke('data', 'inspect', 'heart', '--format', 'json')
                self.assertEqual(code, 0, error)
                inspected = json.loads(output)
                self.assertEqual(inspected['profile']['modality'], 'tabular')
                self.assertEqual(inspected['profile']['num_classes'], 2)
                code, output, error = self.invoke('methods', 'compatible', '--dataset', 'heart', '--format', 'json')
                self.assertEqual(code, 0, error)
                compatibility = {item['method']: item for item in json.loads(output)}
                importance = compatibility['importance_reweighting']
                self.assertEqual(importance['status'], 'compatible_with_requirements')
                self.assertIn(
                    'requires_binary_noise_prior', importance['reason_codes']
                )
                self.assertIn(
                    'config:requires_binary_noise_prior',
                    importance['required_user_inputs'],
                )
                upm = compatibility['upm']
                self.assertEqual(upm['status'], 'compatible_with_requirements')
                self.assertIn('requires_noisy_training_labels', upm['reason_codes'])
                self.assertNotIn('unsupported_modality', upm['reason_codes'])
                code, output, error = self.invoke('data', 'verify', 'heart', '--output-dir', str(Path(directory) / 'heart-run'), '--project-root', str(_unified_cli_ROOT))
                self.assertEqual(code, 0, error)
                self.assertIn('Training check   VERIFIED', output)
                self.assertIn('Train samples', output)
                self.assertIn('completed one-epoch run', output)

    def test_data_verify_uses_automatic_profile_without_a_recipe(self) -> None:
        record = Mock(alias='fashion', adapter='fashion_mnist', signature='a' * 64)
        report = DatasetStatusReport('fashion', 'fashion_mnist', 'ready', training_evidence={'run_dir': 'verify-run'})
        with tempfile.TemporaryDirectory() as directory, patch.object(DEFAULT_DATA_SERVICE, 'record', return_value=record), patch.object(DEFAULT_DATA_SERVICE, 'verify', return_value=(report, Path(directory))) as verify:
            destination = Path(directory) / 'run'
            code, output, error = self.invoke('data', 'verify', 'fashion', '--output-dir', str(destination))
        self.assertEqual(code, 0, error)
        verify.assert_called_once_with('fashion', None, destination, recipe=None)
        self.assertIn('automatic dataset profile', output)

    def test_list_and_paper_show_are_user_facing(self) -> None:
        code, output, _ = self.invoke('list', 'experiments', '--profile', 'smoke')
        self.assertEqual(code, 0)
        self.assertIn('个可运行实验（规模=smoke）', output)
        self.assertIn('数据集：', output)
        self.assertIn('执行器：', output)
        self.assertIn('先预览：', output)
        self.assertNotIn('RECIPE | PROFILE', output)
        code, output, _ = self.invoke('papers', 'list')
        self.assertEqual(code, 0)
        self.assertIn('dual-t', output)
        self.assertIn('篇具有可运行配置的论文', output)
        self.assertIn('实现保真度：', output)
        self.assertIn('建议先看：', output)
        self.assertNotIn('ID | PAPER', output)
        code, output, _ = self.invoke('papers', 'show', 'jocor')
        self.assertEqual(code, 0)
        self.assertIn('Toolbox 生命周期', output)
        self.assertIn('论文概念 -> config -> 实现', output)
        self.assertIn('已知差异与限制', output)
        self.assertIn('Recipe：', output)
        self.assertIn('标签来源：', output)
        self.assertIn('Scheduler：', output)
        self.assertIn('生成 symmetric 噪声', output)
        self.assertIn('由执行器决定', output)
        code, output, _ = self.invoke('papers', 'show', 'dual-t')
        self.assertEqual(code, 0)
        self.assertIn('posterior_stage=', output)

    def test_experiment_list_supports_tsv_for_scripts(self) -> None:
        code, output, _ = self.invoke('list', 'experiments', '--profile', 'smoke', '--format', 'tsv')
        self.assertEqual(code, 0)
        lines = output.splitlines()
        self.assertEqual(lines[0], 'recipe\tprofile\tdataset\tnoise\tmethod\trunner\tepochs')
        self.assertTrue(all(('\t' in line for line in lines[1:])))

    def test_other_catalogs_are_human_readable_and_support_tsv(self) -> None:
        code, output, _ = self.invoke('list', 'components', '--kind', 'loss')
        self.assertEqual(code, 0)
        self.assertIn('个可组合组件', output)
        self.assertIn('能力：', output)
        self.assertNotIn('KIND | NAME', output)
        code, output, _ = self.invoke('list', 'components', '--kind', 'loss', '--format', 'tsv')
        self.assertEqual(code, 0)
        self.assertEqual(output.splitlines()[0], 'kind\tname\tcapabilities\tpaper')
        code, output, _ = self.invoke('papers', 'list', '--format', 'tsv')
        self.assertEqual(code, 0)
        self.assertEqual(output.splitlines()[0], 'id\tacronym\ttitle\tvenue\tyear\tprofiles\tfidelity\trunners\trecommended')

    def test_paper_config_resolved_does_not_change_source(self) -> None:
        path = _unified_cli_ROOT / 'configs/experiment/cifar10_coteaching_smoke.yaml'
        before = path.read_bytes()
        code, output, _ = self.invoke('papers', 'config', 'coteaching', '--profile', 'smoke', '--resolved')
        self.assertEqual(code, 0)
        self.assertEqual(path.read_bytes(), before)
        parsed = yaml.safe_load(output)
        self.assertEqual(validate_config(parsed).name, 'coteaching')

    def test_dry_run_does_not_create_output(self) -> None:
        with patch('lnl_toolbox.cli.main.ExperimentService.run') as runner:
            code, output, _ = self.invoke('run', '--recipe', 'cifar10-symmetric-ce-smoke', '--dry-run', '--no-check-data')
        self.assertEqual(code, 0)
        self.assertIn('runner: supervised', output)
        self.assertIn('Dataset: cifar10', output)
        runner.assert_not_called()

    def test_dry_run_checks_data_unless_explicitly_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / 'missing-data.yaml'
            config_path.write_text(yaml.safe_dump({'data': {'name': 'cifar10', 'root': str(root / 'missing')}, 'execution': {'runner': 'supervised'}}), encoding='utf-8')
            code, _, error = self.invoke('run', str(config_path), '--dry-run')
            self.assertEqual(code, 2)
            self.assertIn('data path does not exist', error)
            with patch('lnl_toolbox.cli.main.ExperimentService.run') as runner:
                code, output, error = self.invoke('run', str(config_path), '--dry-run', '--no-check-data')
            self.assertEqual(code, 0, error)
            self.assertIn('runner: supervised', output)
            runner.assert_not_called()
            code, _, error = self.invoke('run', str(config_path), '--no-check-data')
            self.assertEqual(code, 2)
            self.assertIn('only valid together with --dry-run', error)

    def test_run_checks_data_before_invoking_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / 'missing-data.yaml'
            config_path.write_text(yaml.safe_dump({'data': {'name': 'cifar10', 'root': str(root / 'missing')}, 'execution': {'runner': 'supervised'}}), encoding='utf-8')
            with patch('lnl_toolbox.training.experiment.run_experiment') as runner:
                code, _, error = self.invoke('run', '--config', str(config_path), '--project-root', str(_unified_cli_ROOT))
            self.assertEqual(code, 2)
            self.assertIn('data path does not exist', error)
            runner.assert_not_called()

    def test_run_normalizes_final_result_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'data'
            data_root.mkdir()
            run_dir = root / 'run'
            config_path = root / 'custom.yaml'
            config_path.write_text(yaml.safe_dump({'data': {'name': 'cifar10', 'root': str(data_root)}, 'loss': {'name': 'gce'}, 'execution': {'runner': 'supervised'}}), encoding='utf-8')

            def fake_run(_config, output_dir, _resume):
                output = Path(output_dir)
                output.mkdir()
                (output / 'metrics.jsonl').write_text(json.dumps({'epoch': 1, 'test_accuracy': 0.75}) + '\n', encoding='utf-8')
                return output
            with patch('lnl_toolbox.training.runners.RunnerSpec.invoke', side_effect=fake_run), patch('lnl_toolbox.cli.main.ExperimentService.preflight'):
                code, _, error = self.invoke('run', '--config', str(config_path), '--project-root', str(_unified_cli_ROOT), '--output-dir', str(run_dir))
            self.assertEqual(code, 0, error)
            final = json.loads((run_dir / 'final_metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(final['test_accuracy'], 0.75)
            self.assertEqual(final['event'], 'final')
            self.assertEqual(final['method'], 'gce')
            self.assertEqual(final['runner'], 'supervised')
            self.assertEqual(final['status'], 'completed')
            self.assertIs(final['completed'], True)

    def test_completed_resume_is_strict_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / 'resolved_config.yaml').write_text(yaml.safe_dump({'data': {'name': 'cifar10', 'root': 'missing-data-is-irrelevant'}, 'execution': {'runner': 'supervised'}}), encoding='utf-8')
            (run_dir / 'last.pt').write_bytes(b'checkpoint')
            (run_dir / 'metrics.jsonl').write_text('original\n', encoding='utf-8')
            (run_dir / 'final_metrics.json').write_text(json.dumps({'test_accuracy': 0.5}), encoding='utf-8')
            before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in run_dir.iterdir()}
            with patch('lnl_toolbox.training.experiment.run_experiment') as runner:
                code, output, error = self.invoke('resume', str(run_dir))
            after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in run_dir.iterdir()}
            self.assertEqual(code, 0, error)
            self.assertIn('resume complete', output)
            self.assertEqual(after, before)
            runner.assert_not_called()

    def test_compose_lists_compatible_slots_and_dedicated_runners(self) -> None:
        code, output, _ = self.invoke('compose', 'list', '--runner', 'supervised')
        self.assertEqual(code, 0)
        self.assertIn('可组合结构', output)
        self.assertIn('transition_estimator 必须与 risk_corrector 配对', output)
        self.assertIn('regularizer 虽已注册，但尚未接入', output)
        code, output, _ = self.invoke('compose', 'list', '--runner', 'dual_t')
        self.assertEqual(code, 0)
        self.assertIn('专用生命周期', output)
        self.assertIn('cifar10-dual-t-smoke', output)

    def test_compose_creates_valid_yaml_without_overwriting(self) -> None:
        source = _unified_cli_ROOT / 'configs/experiment/cifar10_symmetric_ce_smoke.yaml'
        before = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'gce_small_loss.yaml'
            code, text, error = self.invoke('compose', 'create', '--base', 'cifar10-symmetric-ce-smoke', '--loss', 'gce', '--selector', 'small_loss', '--keep-rate', '0.6', '--output', str(output))
            self.assertEqual((code, error), (0, ''))
            self.assertIn('已生成新配置', text)
            config = load_yaml(output)
            self.assertEqual(config['loss'], {'name': 'gce'})
            self.assertEqual(config['selector'], {'name': 'small_loss', 'keep_rate': 0.6})
            self.assertEqual(config['execution'], {'runner': 'supervised'})
            code, _, error = self.invoke('compose', 'create', '--base', 'cifar10-symmetric-ce-smoke', '--output', str(output))
            self.assertEqual(code, 2)
            self.assertIn('refusing to overwrite', error)
        self.assertEqual(source.read_bytes(), before)

    def test_compose_rejects_incompatible_objective_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'invalid.yaml'
            code, _, error = self.invoke('compose', 'create', '--base', 'dss-cifar10-symmetric05-smoke', '--selector', 'small_loss', '--keep-rate', '0.5', '--output', str(output))
            self.assertEqual(code, 2)
            self.assertIn("DSS requires selector.name='all'", error)
            self.assertFalse(output.exists())

    def test_compose_copies_dedicated_paper_recipe_without_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'volminnet.yaml'
            code, text, error = self.invoke('compose', 'create', '--base', 'volminnet-cifar10-reproduction', '--output', str(output))
            self.assertEqual((code, error), (0, ''))
            self.assertIn('论文生命周期：volminnet', text)
            self.assertEqual(load_yaml(output), load_yaml(_unified_cli_ROOT / 'configs/experiment/volminnet_cifar10_reproduction.yaml'))
            rejected = Path(directory) / 'invalid.yaml'
            code, _, error = self.invoke('compose', 'create', '--base', 'volminnet-cifar10-reproduction', '--loss', 'ce', '--output', str(rejected))
            self.assertEqual(code, 2)
            self.assertIn('only be copied without component overrides', error)
            self.assertFalse(rejected.exists())

    def test_list_experiments_marks_status_and_hides_conditional(self) -> None:
        code, output, _ = self.invoke('list', 'experiments', '--profile', 'smoke')
        self.assertEqual(code, 0)
        self.assertIn('IMPLEMENTATION', output)
        self.assertIn('cifar10-clean-smoke', output)
        self.assertNotIn('cifar10-cnlcu-soft-smoke', output)
        self.assertNotIn('mentornet', output)
        code, output, _ = self.invoke('list', 'experiments', '--profile', 'smoke', '--all')
        self.assertEqual(code, 0)
        self.assertIn('cifar10-cnlcu-soft-smoke', output)
        self.assertNotIn('mentornet', output)

    def test_browsing_does_not_import_optional_sklearn(self) -> None:
        script = '\nimport builtins, contextlib, io\noriginal = builtins.__import__\ndef blocked(name, *args, **kwargs):\n    if name == "sklearn" or name.startswith("sklearn."):\n        raise ModuleNotFoundError("blocked sklearn")\n    return original(name, *args, **kwargs)\nbuiltins.__import__ = blocked\nfrom lnl_toolbox.cli.main import main\nwith contextlib.redirect_stdout(io.StringIO()):\n    assert main(["list", "experiments"]) == 0\n    assert main(["papers", "list"]) == 0\n'
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(_unified_cli_ROOT / 'src')
        result = subprocess.run([sys.executable, '-c', script], cwd=_unified_cli_ROOT, env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_volminnet_is_discoverable_and_paper_mapped(self) -> None:
        code, output, _ = self.invoke('list', 'experiments', '--profile', 'smoke', '--all')
        self.assertEqual(code, 0)
        self.assertIn('cifar10-volminnet-smoke', output)
        code, output, _ = self.invoke('papers', 'show', 'volminnet')
        self.assertEqual(code, 0)
        self.assertIn('VolMinNet', output)
        self.assertIn('paper_positive_logdet', output)
        code, output, _ = self.invoke('papers', 'config', 'volminnet', '--profile', 'smoke', '--path-only')
        self.assertEqual(code, 0)
        self.assertTrue(output.strip().endswith('cifar10_volminnet_smoke.yaml'))

    def test_volminnet_validate_dry_run_and_epoch_override(self) -> None:
        code, _, error = self.invoke('validate', '--recipe', 'cifar10-volminnet-smoke')
        self.assertEqual(code, 0, error)
        code, output, error = self.invoke('run', '--recipe', 'cifar10-volminnet-smoke', '--dry-run', '--no-check-data', '--epochs', '3')
        self.assertEqual(code, 0, error)
        self.assertIn('volminnet', output)
        self.assertIn('3', output)

    def test_dividemix_is_discoverable_and_has_staged_epoch_override(self) -> None:
        code, output, error = self.invoke('papers', 'show', 'dividemix')
        self.assertEqual(code, 0, error)
        self.assertIn('cross-network GMM', output)
        code, output, error = self.invoke('run', '--recipe', 'cifar10-dividemix-smoke', '--dry-run', '--no-check-data', '--epochs', '3')
        self.assertEqual(code, 0, error)
        self.assertIn('1/3/4 (warmup/main/total)', output)
        self.assertIn('Models: 2', output)

    def test_positional_source_and_dotted_override(self) -> None:
        code, output, error = self.invoke('run', 'cifar10-symmetric-ce-smoke', '--set', 'trainer.epochs=2', '--dry-run', '--no-check-data')
        self.assertEqual(code, 0, error)
        self.assertIn('Training budget: 2', output)

    def test_positional_yaml_is_accepted(self) -> None:
        path = _unified_cli_ROOT / 'configs/experiment/cifar10_symmetric_ce_smoke.yaml'
        code, output, error = self.invoke('validate', str(path))
        self.assertEqual(code, 0, error)
        self.assertIn('supervised', output)

    def test_compare_and_report_dispatch_to_evaluation_service(self) -> None:
        summary = {'group_by': ['method', 'noise.rate', 'primary_metric.name'], 'summaries': [{'group': {'method': 'ce', 'noise.rate': 0.2, 'primary_metric.name': 'test_accuracy'}, 'metric': 'test_accuracy', 'n': 1, 'mean': 0.8, 'std': 0.0, 'median': 0.8, 'min': 0.8, 'max': 0.8}], 'compatibility': {'model': 'consistent'}, 'warnings': [], 'excluded_runs': [], 'failed_runs': []}
        with patch('lnl_toolbox.cli.main.compare_runs', return_value=summary):
            code, output, error = self.invoke('compare', str(_unified_cli_ROOT), '--group-by', 'method,noise.rate')
        self.assertEqual(code, 0, error)
        self.assertIn('METHOD\tNOISE\tMETRIC', output)
        self.assertIn('test_accuracy', output)
        self.assertIn('Compatibility', output)
        leaked = dict(summary, excluded_runs=[{'run_dir': 'x', 'reason': 'leakage'}])
        with patch('lnl_toolbox.cli.main.compare_runs', return_value=leaked):
            code, _, _ = self.invoke('compare', str(_unified_cli_ROOT), '--strict')
        self.assertEqual(code, 1)
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.cli.main.compare_runs', return_value=summary), patch('lnl_toolbox.cli.main.write_report', return_value={'report': Path(directory) / 'report.md'}):
            code, output, error = self.invoke('report', str(_unified_cli_ROOT), '--output-dir', directory)
        self.assertEqual(code, 0, error)
        self.assertIn('report.md', output)

    def test_matrix_sweep_dry_run_and_status_do_not_invoke_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / 'sweep-output'
            spec = root / 'sweep.yaml'
            spec.write_text(yaml.safe_dump({'version': 1, 'base': {'recipe': 'cifar10-symmetric-ce-smoke'}, 'matrix': {'trainer.epochs': [1, 2]}, 'seeds': [1, 2]}, sort_keys=False), encoding='utf-8')
            with patch('lnl_toolbox.cli.main.ExperimentService.run') as runner:
                code, output, error = self.invoke('sweep', str(spec), '--dry-run', '--no-check-data', '--output-dir', str(output_dir))
            self.assertEqual(code, 0, error)
            self.assertIn('Total runs:\n  4', output)
            self.assertIn('trainer.epochs=1', output)
            self.assertFalse(output_dir.exists())
            runner.assert_not_called()
            output_dir.mkdir()
            (output_dir / 'sweep_manifest.json').write_text(json.dumps({'schema_version': 2, 'sweep_id': 'example', 'status': 'running', 'runs': [{'seed': 1, 'status': 'completed'}, {'seed': 2, 'status': 'failed', 'overrides': {'trainer.epochs': 2}, 'error': 'failure'}, {'seed': 3, 'status': 'pending'}]}), encoding='utf-8')
            code, output, error = self.invoke('sweep', 'status', str(output_dir))
            self.assertEqual(code, 0, error)
            self.assertIn('1 / 3 completed', output)
            self.assertIn('trainer.epochs=2', output)

    def test_cli_matrix_is_typed_and_seed_defaults_to_recipe_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.cli.main.ExperimentService.run') as runner:
            code, output, error = self.invoke('sweep', '--recipe', 'cifar10-clean-smoke', '--matrix', 'loader.batch_size=[256,512]', '--matrix', 'optimizer.lr=[0.01,0.001]', '--output-dir', directory, '--dry-run', '--no-check-data')
        self.assertEqual(code, 0, error)
        self.assertIn('Total runs:\n  4', output)
        self.assertIn('loader.batch_size=256', output)
        runner.assert_not_called()

    def test_cli_matrix_multiplies_optional_seeds_and_rejects_bad_json(self) -> None:
        code, output, error = self.invoke('sweep', '--recipe', 'cifar10-clean-smoke', '--matrix', 'loader.batch_size=[256,512]', '--matrix', 'optimizer.lr=[0.01,0.001]', '--seeds', '1', '2', '3', '--dry-run', '--no-check-data')
        self.assertEqual(code, 0, error)
        self.assertIn('Total runs:\n  12', output)
        code, _, error = self.invoke('sweep', '--recipe', 'cifar10-clean-smoke', '--matrix', 'optimizer.lr=0.1')
        self.assertEqual(code, 2)
        self.assertIn('JSON array', error)

    def test_table_commands_offer_json_contracts(self) -> None:
        for arguments in (('list', 'experiments', '--format', 'json'), ('list', 'components', '--format', 'json'), ('papers', 'list', '--format', 'json'), ('data', 'list', '--format', 'json')):
            code, output, error = self.invoke(*arguments)
            self.assertEqual(code, 0, error)
            self.assertIsInstance(json.loads(output), list)
        summary = {'summaries': [], 'warnings': [], 'excluded_runs': [], 'failed_runs': [], 'group_by': [], 'compatibility': {}}
        with patch('lnl_toolbox.cli.main.compare_runs', return_value=summary):
            code, output, error = self.invoke('compare', str(_unified_cli_ROOT), '--format', 'json')
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(output)['summaries'], [])

# --- merged from test_dividemix_cli.py ---
import unittest

# --- merged from test_dividemix_cli.py ---
from lnl_toolbox.catalog import discover_recipes, load_recipe_config, recipe_by_id, validate_config

# --- merged from test_dividemix_cli.py ---
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner

# --- merged from test_dividemix_cli.py ---
class _dividemix_cli_DivideMixCliTest(unittest.TestCase):

    def test_recipe_validates_and_epoch_override_only_changes_main(self):
        config = load_recipe_config(recipe_by_id('cifar10-dividemix-smoke'))
        self.assertEqual(resolve_runner(config).name, 'dividemix')
        self.assertEqual(validate_config(config).name, 'dividemix')
        warmup = config['dividemix']['warmup']['epochs']
        apply_epoch_override(config, 7)
        self.assertEqual(config['dividemix']['training']['epochs'], 7)
        self.assertEqual(config['dividemix']['warmup']['epochs'], warmup)

    def test_formal_recipe_is_discoverable_and_epoch_override_is_main_only(self):
        recipe_ids = {item.id for item in discover_recipes()}
        self.assertIn('cifar10-dividemix-smoke', recipe_ids)
        self.assertIn('cifar10-dividemix-sym20', recipe_ids)
        config = load_recipe_config(recipe_by_id('cifar10-dividemix-sym20'))
        self.assertEqual(resolve_runner(config).name, 'dividemix')
        self.assertEqual(validate_config(config).name, 'dividemix')
        self.assertEqual(config['dividemix']['warmup']['epochs'], 10)
        self.assertEqual(config['dividemix']['training']['epochs'], 300)
        apply_epoch_override(config, 1)
        self.assertEqual(config['dividemix']['warmup']['epochs'], 10)
        self.assertEqual(config['dividemix']['training']['epochs'], 1)

# --- merged from test_dld_cli.py ---
from contextlib import redirect_stdout

# --- merged from test_dld_cli.py ---
from copy import deepcopy

# --- merged from test_dld_cli.py ---
from io import StringIO

# --- merged from test_dld_cli.py ---
from pathlib import Path

# --- merged from test_dld_cli.py ---
import unittest

# --- merged from test_dld_cli.py ---
import yaml

# --- merged from test_dld_cli.py ---
from lnl_toolbox.catalog import paper_by_id, recipe_by_id, validate_config

# --- merged from test_dld_cli.py ---
from lnl_toolbox.cli.main import main

# --- merged from test_dld_cli.py ---
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner

# --- merged from test_dld_cli.py ---
_dld_cli_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_dld_cli.py ---
class _dld_cli_DLDCliTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = _dld_cli_ROOT / 'configs' / 'experiment' / 'cifar10_dld_smoke.yaml'
        cls.config = yaml.safe_load(cls.path.read_text(encoding='utf-8'))

    def test_registry_recipe_paper_and_preflight(self) -> None:
        self.assertEqual(resolve_runner(self.config).name, 'dld')
        self.assertEqual(validate_config(self.config).name, 'dld')
        self.assertEqual(recipe_by_id('cifar10-dld-smoke').runner, 'dld')
        self.assertEqual(paper_by_id('dld').implementation_status, 'user_ready')

    def test_epoch_override_only_changes_diffusion(self) -> None:
        config = deepcopy(self.config)
        before = deepcopy(config['trainer'])
        apply_epoch_override(config, 7)
        self.assertEqual(config['dld']['diffusion']['epochs'], 7)
        self.assertEqual(config['trainer'], before)

    def test_dry_run_discloses_fidelity_and_plan(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['run', '--config', str(self.path), '--dry-run', '--no-check-data']), 0)
        text = output.getvalue()
        for value in ('dld', '2 (diffusion)', 'K=10', 'cosine_similarity', 'dld_precorrection.npz', 'paper_oriented_v2_cosine_similarity', 'DLD inference steps: 5'):
            self.assertIn(value, text)

    def test_real_short_config_validates_and_dry_runs(self) -> None:
        path = _dld_cli_ROOT / 'configs' / 'reproduction' / 'cifar10_dld_sym20_short.yaml'
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        self.assertEqual(validate_config(config).name, 'dld')
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['run', '--config', str(path), '--dry-run', '--no-check-data']), 0)
        text = output.getvalue()
        for value in ('dld', '15 (diffusion)', 'K=50', 'paper_oriented_v2_cosine_similarity'):
            self.assertIn(value, text)

    def test_protected_local_yaml_is_not_catalogued(self) -> None:
        with self.assertRaises(ValueError):
            recipe_by_id('cifar10-symmetric40-all-e5')

# --- merged from test_lend_cli.py ---
from contextlib import redirect_stdout

# --- merged from test_lend_cli.py ---
from copy import deepcopy

# --- merged from test_lend_cli.py ---
from io import StringIO

# --- merged from test_lend_cli.py ---
from pathlib import Path

# --- merged from test_lend_cli.py ---
import unittest

# --- merged from test_lend_cli.py ---
import yaml

# --- merged from test_lend_cli.py ---
from lnl_toolbox.catalog import paper_by_id, recipe_by_id, validate_config

# --- merged from test_lend_cli.py ---
from lnl_toolbox.cli.main import main

# --- merged from test_lend_cli.py ---
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner

# --- merged from test_lend_cli.py ---
_lend_cli_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_lend_cli.py ---
class _lend_cli_LENDCliTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = _lend_cli_ROOT / 'configs/experiment/cifar10_lend_smoke.yaml'
        cls.config = yaml.safe_load(cls.path.read_text(encoding='utf-8'))
        cls.reproduction_path = _lend_cli_ROOT / 'configs/experiment/lend_cifar10_reproduction.yaml'
        cls.reproduction_config = yaml.safe_load(cls.reproduction_path.read_text(encoding='utf-8'))

    def test_registry_recipe_paper_and_preflight(self):
        self.assertEqual(resolve_runner(self.config).name, 'lend')
        self.assertEqual(validate_config(self.config).name, 'lend')
        self.assertEqual(recipe_by_id('cifar10-lend-smoke').runner, 'lend')
        paper = paper_by_id('lend')
        self.assertEqual(paper.implementation_status, 'user_ready')
        self.assertIn('paper-oriented', ' '.join(paper.limitations))

    def test_epoch_override_only_changes_lend_training(self):
        config = deepcopy(self.config)
        trainer = deepcopy(config['trainer'])
        apply_epoch_override(config, 7)
        self.assertEqual(config['lend']['training']['epochs'], 7)
        self.assertEqual(config['trainer'], trainer)

    def test_dry_run_discloses_all_fidelity_choices(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['run', '--config', str(self.path), '--dry-run', '--no-check-data']), 0)
        text = output.getvalue()
        for value in ('lend', '2 (LEND)', 'k=15', 'gamma=1.0', 'inner_product', 'normalize_features=False', 'alpha=0.99', 'fixed_steps', 'steps=3', 'beta=0.9', 'batch_mean', 'skip_update'):
            self.assertIn(value, text)

    def test_preflight_rejects_unresolved_and_small_final_batch(self):
        bad = deepcopy(self.config)
        bad['lend']['graph']['gamma'] = None
        with self.assertRaises((TypeError, ValueError)):
            validate_config(bad)
        bad = deepcopy(self.config)
        bad['data']['max_train_samples'] = 68
        with self.assertRaisesRegex(ValueError, 'final partial batch'):
            validate_config(bad)
        bad = deepcopy(self.config)
        bad['model']['name'] = 'opaque_model'
        with self.assertRaisesRegex(ValueError, 'feature-aware'):
            validate_config(bad)

    def test_full_budget_recipe_is_public_and_valid(self):
        recipe = recipe_by_id('lend-cifar10-reproduction')
        self.assertEqual(recipe.runner, 'lend')
        self.assertEqual(recipe.profile, 'reproduction')
        self.assertEqual(recipe.configuration_fidelity, 'paper_oriented')
        self.assertEqual(recipe.reproduction_status, 'not_run')
        self.assertEqual(validate_config(self.reproduction_config).name, 'lend')
        paper = paper_by_id('lend')
        formal = [item for item in paper.configs if item.recipe_id == recipe.id]
        self.assertEqual(len(formal), 1)
        self.assertEqual(formal[0].configuration_fidelity, 'paper_oriented')
        self.assertEqual(formal[0].reproduction_status, 'not_run')

    def test_full_budget_config_and_dry_run_disclose_formal_setting(self):
        config = self.reproduction_config
        self.assertEqual(config['method'], 'lend')
        self.assertEqual(config['execution']['runner'], 'lend')
        self.assertEqual(config['data']['name'], 'cifar10')
        for key in ('max_train_samples', 'max_validation_samples', 'max_test_samples'):
            self.assertNotIn(key, config['data'])
        self.assertEqual(config['noise']['validation_targets'], 'noisy')
        self.assertEqual(config['model'], {'name': 'resnet18', 'base_width': 64})
        self.assertEqual(config['lend']['training']['epochs'], 200)
        self.assertEqual(config['loader']['batch_size'], 256)
        self.assertEqual(config['optimizer'], {'name': 'sgd', 'lr': 0.05, 'momentum': 0.9, 'weight_decay': 0.0005})
        self.assertEqual(config['scheduler'], {'name': 'multistep', 'milestones': [100], 'gamma': 0.1})
        self.assertEqual(config['lend']['graph']['k'], 8)
        self.assertEqual(config['lend']['graph']['gamma'], 1.0)
        self.assertEqual(config['lend']['dilution'], {'alpha': 0.99, 'policy': 'fixed_steps', 'steps': 10})
        self.assertEqual(config['lend']['history']['beta'], 0.9)
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['run', '--recipe', 'lend-cifar10-reproduction', '--dry-run', '--no-check-data']), 0)
        text = output.getvalue()
        for value in ('lend', '200 (LEND)', 'resnet18', 'LEND batch size: 256', 'k=8', 'gamma=1.0', 'alpha=0.99', 'steps=10', 'beta=0.9'):
            self.assertIn(value, text)

# --- merged from test_pcse_cli.py ---
from pathlib import Path

# --- merged from test_pcse_cli.py ---
import os

# --- merged from test_pcse_cli.py ---
import unittest

# --- merged from test_pcse_cli.py ---
from unittest import mock

# --- merged from test_pcse_cli.py ---
from lnl_toolbox.algorithms.pcse import PCSEConfig

# --- merged from test_pcse_cli.py ---
from lnl_toolbox.catalog import discover_recipes, load_recipe_config, paper_by_id, recipe_by_id, validate_config

# --- merged from test_pcse_cli.py ---
class _pcse_cli_PCSECliTest(unittest.TestCase):

    def test_real_cifar_recipe_is_registered_and_valid(self) -> None:
        recipe = recipe_by_id('cifar10-pcse-reproduction')
        self.assertEqual(recipe.profile, 'reproduction')
        self.assertEqual(recipe.runner, 'pcse')
        self.assertEqual(recipe.configuration_fidelity, 'engineering')
        self.assertEqual(recipe.availability, 'runnable')
        config = load_recipe_config(recipe)
        parsed = PCSEConfig.from_mapping(config)
        self.assertEqual(parsed.pretraining.mode, 'train')
        self.assertEqual(parsed.pretraining.method, 'cross_entropy')
        self.assertIsNone(parsed.pretraining.source)
        self.assertEqual(config['noise']['mode'], 'generated')
        self.assertEqual(config['pretraining_stage']['epochs'], 100)
        self.assertEqual(config['data']['name'], 'cifar10')
        self.assertNotIn('max_train_samples', config['data'])
        self.assertEqual([(item.name, item.pooling) for item in parsed.feature_layers], [('layer3', 'global_average'), ('layer4', 'global_average')])
        self.assertEqual(parsed.transition_backend, 'paper_volmin')

    def test_real_cifar_recipe_is_discoverable_without_conditional_flag(self) -> None:
        default_ids = {item.id for item in discover_recipes()}
        all_ids = {item.id for item in discover_recipes(include_conditional=True)}
        self.assertIn('cifar10-pcse-reproduction', default_ids)
        self.assertIn('cifar10-pcse-reproduction', all_ids)

    def test_real_cifar_preflight_does_not_require_source_environment(self) -> None:
        config = load_recipe_config(recipe_by_id('cifar10-pcse-reproduction'))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('LNL_PCSE_SOURCE_RUN', None)
            self.assertEqual(validate_config(config).name, 'pcse')

    def test_paper_catalog_does_not_claim_numerical_reproduction(self) -> None:
        paper = paper_by_id('pcse')
        recipe_ids = {item.recipe_id for item in paper.configs}
        self.assertIn('cifar10-pcse-reproduction', recipe_ids)
        self.assertEqual(paper.reproduction_status, 'not_run')
        real_config = next((item for item in paper.configs if item.recipe_id == 'cifar10-pcse-reproduction'))
        self.assertEqual(real_config.availability, 'runnable')

# --- merged from test_upm_cli.py ---
from contextlib import redirect_stdout

# --- merged from test_upm_cli.py ---
from copy import deepcopy

# --- merged from test_upm_cli.py ---
import io

# --- merged from test_upm_cli.py ---
from pathlib import Path

# --- merged from test_upm_cli.py ---
import unittest

# --- merged from test_upm_cli.py ---
from lnl_toolbox.catalog import load_recipe_config, paper_by_id, recipe_by_id, validate_config

# --- merged from test_upm_cli.py ---
from lnl_toolbox.cli import main as cli_main

# --- merged from test_upm_cli.py ---
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner

# --- merged from test_upm_cli.py ---
def _upm_cli__config():
    stage = {'epochs': 2, 'model': {'name': 'tiny_cnn'}, 'optimizer': {'name': 'sgd', 'lr': 0.01}, 'scheduler': {'name': 'none'}}
    return {'method': 'upm', 'execution': {'runner': 'upm'}, 'data': {'name': 'cifar10'}, 'loader': {'batch_size': 2}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'validation_targets': 'noisy'}, 'evaluation': {'selection_split': 'validation'}, 'trainer': {'device': 'cpu'}, 'upm': {'stage1': {**stage, 'best_metric': 'noisy_validation_accuracy'}, 'psi': {'source': 'stage1_best', 'split': 'train', 'augmentation': False}, 'main': {**stage, 'initialization': 'fresh'}, 'confusing_probability': {'initial_value': 0.01, 'learning_rate': 0.1, 'epsilon': 0.0001, 'update_start_epoch': 0, 'update_interval_epochs': 1}}}

# --- merged from test_upm_cli.py ---
class _upm_cli_UPMCliTest(unittest.TestCase):

    def test_registry_recipe_paper_and_validation(self) -> None:
        self.assertEqual(resolve_runner(_upm_cli__config()).name, 'upm')
        self.assertEqual(validate_config(_upm_cli__config()).name, 'upm')
        recipe = recipe_by_id('cifar10-upm-smoke')
        self.assertEqual(recipe.runner, 'upm')
        paper = paper_by_id('upm')
        self.assertEqual(paper.implementation_status, 'user_ready')

    def test_full_run_recipe_uses_current_upm_schema(self) -> None:
        recipe = recipe_by_id('upm-cifar10-reproduction')
        self.assertEqual(recipe.profile, 'reproduction')
        self.assertEqual(recipe.runner, 'upm')
        self.assertEqual(recipe.configuration_fidelity, 'engineering')
        self.assertEqual(recipe_by_id('cifar10-upm-smoke').configuration_fidelity, 'smoke')
        config = load_recipe_config(recipe)
        self.assertEqual(validate_config(config).name, 'upm')
        self.assertEqual(config['method'], 'upm')
        self.assertEqual(config['execution']['runner'], 'upm')
        self.assertEqual(config['data']['name'], 'cifar10')
        self.assertNotIn('max_train_samples', config['data'])
        self.assertEqual(config['noise']['name'], 'symmetric')
        self.assertEqual(config['noise']['rate'], 0.4)
        self.assertEqual(config['noise']['validation_targets'], 'noisy')
        self.assertEqual(config['upm']['stage1']['epochs'], 160)
        self.assertEqual(config['upm']['main']['epochs'], 160)
        self.assertEqual(config['upm']['main']['model'], {'name': 'resnet18', 'base_width': 16})

    def test_epoch_override_only_changes_main(self) -> None:
        config = _upm_cli__config()
        stage1 = deepcopy(config['upm']['stage1'])
        apply_epoch_override(config, 7)
        self.assertEqual(config['upm']['main']['epochs'], 7)
        self.assertEqual(config['upm']['stage1'], stage1)
        self.assertNotIn('epochs', config['trainer'])

    def test_dry_run_reports_stages_psi_and_eta(self) -> None:
        config_path = Path('configs/experiment/cifar10_upm_smoke.yaml').resolve()
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli_main.main(['run', '--config', str(config_path), '--dry-run', '--no-check-data'])
        self.assertEqual(result, 0)
        text = output.getvalue()
        for expected in ('upm', '2/2 (stage1/main)', 'UPM psi source: stage1_best', 'UPM eta initial value: 0.01', 'UPM eta update start epoch: 0', 'UPM eta update interval: 1'):
            self.assertIn(expected, text)

    def test_protected_yaml_is_not_a_recipe(self) -> None:
        with self.assertRaises(ValueError):
            recipe_by_id('cifar10-symmetric40-all-e5')
