"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_config_schema.py ---
from pathlib import Path

# --- merged from test_config_schema.py ---
import unittest

# --- merged from test_config_schema.py ---
from lnl_toolbox.catalog import discover_recipes, load_yaml

# --- merged from test_config_schema.py ---
from lnl_toolbox.core.config_schema import normalize_experiment_config, runtime_experiment_config

# --- merged from test_config_schema.py ---
class _config_schema_ConfigSchemaTest(unittest.TestCase):

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
        self.assertEqual(len(paths), 95)
        for path in paths:
            value = load_yaml(path)
            self.assertEqual(value['schema_version'], 1, path)
            self.assertIn(value['kind'], {'experiment', 'fragment', 'mentor_artifact'})
            if value['kind'] == 'experiment':
                self.assertIn('runner', value['execution'], path)
                self.assertNotIn('root', value['data'], path)
                self.assertNotIn('type', value.get('noise', {}), path)
        self.assertEqual(len(discover_recipes(root, include_conditional=True)), 67)

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
