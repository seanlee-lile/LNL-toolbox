"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_config_schema.py ---
from pathlib import Path

# --- merged from test_config_schema.py ---
import unittest

# --- merged from test_config_schema.py ---
from lnl_toolbox.catalog import discover_recipes, load_yaml, load_papers, default_paper_config

# --- merged from test_config_schema.py ---
from lnl_toolbox.core.config_schema import normalize_experiment_config, runtime_experiment_config

# --- merged from test_config_schema.py ---
class _config_schema_ConfigSchemaTest(unittest.TestCase):

    def test_one_runtime_seed_controls_nested_stochastic_components(self) -> None:
        config = normalize_experiment_config({
            'seed': 42,
            'execution': {'runner': 'supervised'},
            'data': {'name': 'synthetic_multiclass', 'seed': 0, 'input_seed': 2},
            'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 3},
            'dividemix': {'gmm': {'random_seed': 4}, 'initialization': {'peer_seed_offset': 1}},
            'parameter_record': {'sampling_seed': 99},
        })
        self.assertEqual(config['data']['seed'], 42)
        self.assertEqual(config['data']['input_seed'], 42)
        self.assertEqual(config['noise']['seed'], 42)
        self.assertEqual(config['dividemix']['gmm']['random_seed'], 42)
        self.assertEqual(config['dividemix']['initialization']['peer_seed_offset'], 1)
        self.assertEqual(config['parameter_record']['sampling_seed'], 99)


    def test_binary_legacy_aliases_normalize_to_shared_sections(self) -> None:
        value = normalize_experiment_config({'execution': {'runner': 'binary'}, 'data': {'name': 'synthetic_binary_2d'}, 'batch_size': 8, 'learning_rate': 0.02, 'epochs': 3})
        self.assertEqual(value['loader']['batch_size'], 8)
        self.assertEqual(value['optimizer']['lr'], 0.02)
        self.assertEqual(value['trainer']['epochs'], 3)
        for key in ('batch_size', 'learning_rate', 'epochs'):
            self.assertNotIn(key, value)

    def test_noise_type_is_publicly_removed_but_runtime_compatible(self) -> None:
        value = normalize_experiment_config({'execution': {'runner': 'supervised'}, 'data': {'name': 'synthetic_multiclass'}, 'noise': {'type': 'symmetric', 'rate': 0.2}})
        self.assertEqual(value['noise']['name'], 'symmetric')
        self.assertNotIn('type', value['noise'])
        self.assertEqual(runtime_experiment_config(value)['noise']['type'], 'symmetric')

    def test_unknown_top_level_field_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, 'unknown top-level'):
            normalize_experiment_config({'execution': {'runner': 'clean'}, 'data': {'name': 'synthetic_multiclass'}, 'trianer': {'epochs': 1}})

    def test_every_active_yaml_is_versioned_and_canonical(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / 'configs').rglob('*.yaml'))
        self.assertTrue(paths, 'the active configuration catalog must not be empty')
        for path in paths:
            value = load_yaml(path)
            self.assertEqual(value['schema_version'], 1, path)
            self.assertIn(value['kind'], {'experiment', 'fragment', 'mentor_artifact'})
            if value['kind'] == 'experiment':
                self.assertIn('runner', value['execution'], path)
                self.assertNotIn('root', value['data'], path)
                self.assertNotIn('type', value.get('noise', {}), path)
        recipes = discover_recipes(root, include_conditional=True)
        recipe_ids = [recipe.id for recipe in recipes]
        self.assertTrue(recipes)
        self.assertEqual(len(recipe_ids), len(set(recipe_ids)), 'recipe IDs must be unique')
        for recipe in recipes:
            self.assertIn(recipe.config_path.resolve(), {path.resolve() for path in paths})
        self.assertIn('gce-cifar10-noise02-reproduction', recipe_ids)

    def test_default_paper_filenames_follow_paper_and_default(self) -> None:
        root = Path(__file__).resolve().parents[1]
        papers = load_papers(root)
        self.assertEqual(len(papers), 26)
        for paper in papers:
            with self.subTest(paper=paper.id):
                selected, recipe = default_paper_config(paper, root=root)
                expected = f"{paper.id.replace('-', '_')}_default.yaml"
                self.assertEqual(recipe.config_path.name, expected)
                self.assertEqual(recipe.config_path.parent, root / 'configs/reproduction')
                self.assertEqual(selected.profile, 'reproduction')

    def test_relocated_variants_keep_recipe_ids_and_profiles(self) -> None:
        from lnl_toolbox.catalog import recipe_by_id
        for recipe_id, profile in (
            ('apl-cifar10-noise04-reproduction', 'reproduction'),
            ('cifar10-clean', 'experiment'),
            ('cifar10-clean-smoke', 'smoke'),
        ):
            with self.subTest(recipe=recipe_id):
                recipe = recipe_by_id(recipe_id)
                self.assertEqual(recipe.id, recipe_id)
                self.assertEqual(recipe.profile, profile)
                self.assertEqual(recipe.config_path.parent.name, 'smoke')

    def test_historical_best_cifar_configs_are_registered_without_replacing_defaults(self) -> None:
        root = Path(__file__).resolve().parents[1]
        papers = load_papers(root)
        recipes = {r.id: r for r in discover_recipes(root, include_conditional=True)}
        missing_c100 = {'binary-risk', 'importance-reweighting', 'mc-ldce', 'pdl'}
        outputs = set()
        count = 0
        for paper in papers:
            _, default = default_paper_config(paper, root=root)
            self.assertEqual(default.config_path.name, paper.id.replace('-', '_') + '_default.yaml')
            for dataset in ('cifar10', 'cifar100'):
                path = root / 'configs/reproduction' / f"{paper.id.replace('-', '_')}_{dataset}.yaml"
                if dataset == 'cifar100' and paper.id in missing_c100:
                    self.assertFalse(path.exists())
                    self.assertNotIn(f'{paper.id}-{dataset}-best', recipes)
                    continue
                with self.subTest(paper=paper.id, dataset=dataset):
                    recipe = recipes[f'{paper.id}-{dataset}-best']
                    self.assertEqual(recipe.config_path, path.resolve())
                    self.assertEqual(recipe.visibility, 'internal')
                    config = load_yaml(path)
                    expected = 'cifar10_airplane_automobile' if dataset == 'cifar10' and paper.id in {
                        'binary-risk', 'importance-reweighting', 'cwd',
                    } else dataset
                    self.assertEqual(config['data']['name'], expected)
                    self.assertNotIn('root', config['data'])
                    self.assertNotIn('local_dataset', config)
                    self.assertNotIn('diagnostic_control', config)
                    self.assertNotIn('web_parameter_record', config.get('meta', {}))
                    self.assertNotIn(config['output_root'], outputs)
                    outputs.add(config['output_root'])
                    header = path.read_text(encoding='utf-8')
                    self.assertIn('# Source SHA256:', header)
                    self.assertIn('# Recovery:', header)
                    count += 1
        self.assertEqual(count, 48)
        self.assertEqual(len(list((root / 'configs/reproduction').glob('*.yaml'))), 74)

    def test_historical_best_keeps_distinct_training_settings(self) -> None:
        root = Path(__file__).resolve().parents[1] / 'configs/reproduction'
        def config(name):
            return load_yaml(root / (name + '.yaml'))
        self.assertEqual(config('cwd_cifar10')['model']['feature_dropout'], 0.5)
        self.assertEqual(config('apl_cifar100')['loss']['alpha'], 10.0)
        self.assertEqual(config('dividemix_cifar100')['dividemix']['objective']['lambda_u'], 150)
        self.assertEqual(config('l2rw_cifar10')['data']['num_clean'], 300)
        self.assertEqual(config('l2rw_cifar100')['trainer']['max_steps'], 60000)
        self.assertEqual(config('lend_default')['lend']['dilution']['policy'], 'until_converged')
        self.assertEqual(config('lend_cifar10')['lend']['dilution']['policy'], 'fixed_steps')
        self.assertEqual(config('upm_default')['upm']['psi']['source'], 'stage1_best')
        self.assertEqual(config('upm_cifar10')['upm']['psi']['source'], 'stage1_last')
        self.assertEqual(config('loss_correction_default')['optimizer']['lr'], 0.01)
        self.assertEqual(config('loss_correction_cifar10')['optimizer']['lr'], 0.1)
        for dataset in ('cifar10', 'cifar100'):
            pcse = config('pcse_' + dataset)
            self.assertEqual(pcse['pretraining_stage']['method'], 'cross_entropy')
            self.assertNotIn('mode', pcse['pretraining_stage'])
            self.assertNotIn('manifest_sha256', pcse['noise'])

# --- merged from test_config_overrides.py ---
from copy import deepcopy

# --- merged from test_config_overrides.py ---
import unittest

# --- merged from test_config_overrides.py ---
from lnl_toolbox.core.config_overrides import apply_override, apply_override_assignments, parse_override_value

# --- merged from test_config_overrides.py ---
class _config_overrides_ConfigOverridesTest(unittest.TestCase):

    def test_values_are_typed(self) -> None:
        self.assertIs(parse_override_value('true'), True)
        self.assertEqual(parse_override_value('12'), 12)
        self.assertEqual(parse_override_value('0.25'), 0.25)
        self.assertEqual(parse_override_value('[1, 2]'), [1, 2])
        self.assertEqual(parse_override_value('cifar10'), 'cifar10')

    def test_nested_existing_values_are_overridden(self) -> None:
        config = {'trainer': {'epochs': 10}, 'seed': 1}
        result = apply_override_assignments(config, ['trainer.epochs=20', 'seed=7'])
        self.assertEqual(result, {'trainer': {'epochs': 20}, 'seed': 7})
        self.assertEqual(config, {'trainer': {'epochs': 10}, 'seed': 1})

    def test_typo_fails_without_partial_mutation(self) -> None:
        config = {'trainer': {'epochs': 10}}
        before = deepcopy(config)
        with self.assertRaisesRegex(ValueError, 'epochs'):
            apply_override(config, 'trainer.epoch', 20)
        self.assertEqual(config, before)

    def test_assignment_requires_equals(self) -> None:
        with self.assertRaisesRegex(ValueError, 'path=value'):
            apply_override_assignments({'seed': 1}, ['seed'])

# --- merged from test_hyperparameters.py ---
import unittest

# --- merged from test_hyperparameters.py ---
from lnl_toolbox.core.hyperparameters import ParameterRecord, resolve_parameter_sampling, sample_parameters

# --- merged from test_hyperparameters.py ---
class _hyperparameters_HyperparameterTest(unittest.TestCase):

    def test_sampling_is_deterministic_and_does_not_touch_global_rng(self):
        import random
        candidates = {'lr': (0.01, 0.05), 'depth': (14, 32)}
        random.seed(91)
        before = random.getstate()
        first = sample_parameters('loss_correction', 7, candidates)
        after = random.getstate()
        second = sample_parameters('loss_correction', 7, candidates)
        self.assertEqual(before, after)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertIn(first.parameters['lr'], candidates['lr'])
        self.assertIn(first.parameters['depth'], candidates['depth'])

    def test_parameter_record_round_trip(self) -> None:
        record = sample_parameters('fixture', 3, {'beta': (0.001,)}, sources={'beta': 'fixture source'})
        self.assertEqual(ParameterRecord.from_dict(record.to_dict()), record)

    def test_resolution_does_not_mutate_input(self) -> None:
        config = {'seed': 4, 'nested': {'value': [1, 2]}, 'parameter_sampling': {'paper': 'fixture', 'seed': 9, 'candidates': {'alpha': [0.1, 0.2]}}}
        original_nested = config['nested']
        resolved, record = resolve_parameter_sampling(config)
        self.assertIsNotNone(record)
        self.assertNotIn('parameter_record', config)
        self.assertIs(config['nested'], original_nested)
        self.assertIn(resolved['resolved_parameters']['alpha'], (0.1, 0.2))

    def test_invalid_candidates_and_non_json_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-empty'):
            sample_parameters('fixture', 1, {'alpha': ()})
        with self.assertRaisesRegex(TypeError, 'JSON-compatible'):
            sample_parameters('fixture', 1, {'alpha': (object(),)})
