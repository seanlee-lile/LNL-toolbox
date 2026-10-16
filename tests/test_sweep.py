"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_sweep.py ---
import json

# --- merged from test_sweep.py ---
from pathlib import Path

# --- merged from test_sweep.py ---
import tempfile

# --- merged from test_sweep.py ---
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import yaml

# --- merged from test_sweep.py ---
from unittest.mock import patch

# --- merged from test_sweep.py ---
from lnl_toolbox.training.results import discover_run_results, load_metric_history

# --- merged from test_sweep.py ---
from lnl_toolbox.training.sweep import plan_sweep, run_sweep, sweep_status

# --- merged from test_sweep.py ---
class _sweep__Runner:
    supports_resume = True

# --- merged from test_sweep.py ---
class _sweep__Service:

    def __init__(self) -> None:
        self.calls: list[int] = []

    def run(self, config, output_dir, resume=None, **kwargs):
        seed = int(config['seed'])
        self.calls.append(seed)
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        if seed == 13:
            raise RuntimeError('deliberate failure')
        (root / 'final_metrics.json').write_text(json.dumps({'status': 'completed', 'completed': True, 'primary_metric': {'name': 'test_accuracy', 'value': 0.5}}), encoding='utf-8')
        return root

# --- merged from test_sweep.py ---
class _sweep_SweepTest(unittest.TestCase):

    @staticmethod
    def _matrix_config() -> dict:
        return {'seed': 0, 'method': 'fake', 'noise': {'rate': 0.1}, 'optimizer': {'lr': 0.1}}

    def test_matrix_planning_is_cartesian_and_deterministic(self) -> None:
        base = self._matrix_config()
        single = plan_sweep(base, [1, 2], matrix={'noise.rate': [0.2, 0.4]})
        self.assertEqual(len(single.runs), 4)
        matrix = {'optimizer.lr': [0.1, 0.01, 0.001], 'noise.rate': [0.2, 0.4]}
        first = plan_sweep(base, [1, 2], matrix=matrix)
        second = plan_sweep(base, [1, 2], matrix=dict(reversed(tuple(matrix.items()))))
        self.assertEqual(first.runs, second.runs)
        self.assertEqual(len(first.runs), 12)
        self.assertEqual(first.runs[0].seed, 1)
        self.assertEqual(first.runs[1].seed, 2)
        self.assertEqual(first.runs[0].overrides, (('noise.rate', 0.2), ('optimizer.lr', 0.1)))

    def test_batch_lr_matrix_and_optional_seeds_have_expected_size(self) -> None:
        base = {**self._matrix_config(), 'loader': {'batch_size': 128}}
        matrix = {'loader.batch_size': [256, 512], 'optimizer.lr': [0.01, 0.001]}
        self.assertEqual(len(plan_sweep(base, [base['seed']], matrix=matrix).runs), 4)
        self.assertEqual(len(plan_sweep(base, [1, 2, 3], matrix=matrix).runs), 12)

    def test_invalid_matrix_override_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            with self.assertRaisesRegex(ValueError, 'Unknown config path'):
                run_sweep(self._matrix_config(), [1], matrix={'optimizer.lrr': [0.1]}, output_dir=output, service=_sweep__Service())
            self.assertFalse(output.exists())

    def test_sweep_is_sequential_resumable_and_failure_isolated(self) -> None:
        service = _sweep__Service()
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.training.sweep.resolve_runner', return_value=_sweep__Runner()):
            first = run_sweep({'method': 'fake', 'seed': 1}, [2, 13, 3], output_dir=directory, service=service)
            self.assertEqual(service.calls, [2, 13, 3])
            self.assertEqual(first.completed, 2)
            self.assertEqual(first.failed, 1)
            second = run_sweep({'method': 'fake', 'seed': 1}, [2, 13, 3], output_dir=directory, service=service)
            self.assertEqual(service.calls, [2, 13, 3, 13])
            self.assertEqual(second.skipped, 2)

    def test_duplicate_seeds_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, 'unique'):
            run_sweep({}, [1, 1])

    def test_actual_sweep_isolates_admission_failure_and_runs_valid_member(self):
        from lnl_toolbox.training.service import ExperimentService

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'execution': {'runner': 'supervised'}, 'data': {
                'name': 'synthetic_multiclass', 'num_classes': 3, 'dimension': 4,
                'train_size': 12, 'test_size': 6, 'validation_size': 6, 'root': str(root),
            }}
            calls = []

            def finish(candidate, output, resume):
                calls.append((candidate, resume))
                output = Path(output)
                output.mkdir(parents=True)
                (output / 'final_metrics.json').write_text(
                    json.dumps({'test_accuracy': 0.5}), encoding='utf-8'
                )
                return output

            service = ExperimentService()
            with patch('lnl_toolbox.training.runners.RunnerSpec.invoke', side_effect=finish):
                result = run_sweep(config, [1], matrix={
                    'data.root': [str(root / 'missing'), str(root)],
                }, output_dir=root / 'sweep', service=service)
            self.assertEqual((result.failed, result.completed), (1, 1))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0]['data']['root'], str(root))
            self.assertIsNone(calls[0][1])
            self.assertEqual([row['status'] for row in result.manifest['runs']], ['failed', 'completed'])
            self.assertIn('data path does not exist', result.manifest['runs'][0]['error'])

    def test_actual_cli_sweep_keeps_failure_exit_code_and_continues(self):
        from lnl_toolbox.cli.main import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'execution': {'runner': 'supervised'}, 'data': {
                'name': 'synthetic_multiclass', 'num_classes': 3, 'dimension': 4,
                'train_size': 12, 'test_size': 6, 'validation_size': 6, 'root': str(root),
            }}
            source = root / 'config.yaml'
            source.write_text(yaml.safe_dump(config), encoding='utf-8')

            def finish(_config, output, _resume):
                output = Path(output)
                output.mkdir(parents=True)
                (output / 'final_metrics.json').write_text(json.dumps({'test_accuracy': 0.5}), encoding='utf-8')
                return output

            stdout, stderr = StringIO(), StringIO()
            with patch('lnl_toolbox.training.runners.RunnerSpec.invoke', side_effect=finish) as runner, redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(['sweep', '--config', str(source), '--seeds', '1', '--matrix',
                             'data.root=' + json.dumps([str(root / 'missing'), str(root)]),
                             '--output-dir', str(root / 'sweep')])
            self.assertEqual(code, 1, stderr.getvalue())
            runner.assert_called_once()
            self.assertIn('completed=1 skipped=0 failed=1', stdout.getvalue())

    def test_matrix_manifest_identity_resume_and_status(self) -> None:
        service = _sweep__Service()
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.training.sweep.resolve_runner', return_value=_sweep__Runner()):
            first = run_sweep(self._matrix_config(), [2, 13], matrix={'noise.rate': [0.2, 0.4]}, output_dir=directory, service=service)
            self.assertEqual(first.completed, 2)
            self.assertEqual(first.failed, 2)
            self.assertTrue(all(('config_hash' in item and 'overrides' in item for item in first.manifest['runs'])))
            status = sweep_status(directory)
            self.assertEqual(status['counts']['completed'], 2)
            self.assertEqual(status['counts']['failed'], 2)
            self.assertEqual(len(status['failed_runs']), 2)
            self.assertEqual(len(status['runs']), 4)
            self.assertIn('run_dir', status['runs'][0])
            second = run_sweep(self._matrix_config(), [2, 13], matrix={'noise.rate': [0.2, 0.4]}, output_dir=directory, service=service)
            self.assertEqual(second.skipped, 2)

    def test_result_discovery_reads_partial_history_and_reports_old_damage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'run-a'
            first.mkdir()
            (first / 'metrics.jsonl').write_text('\n'.join([json.dumps({'epoch': 1, 'validation_accuracy': 0.4}), 'not-json', json.dumps({'epoch': 2, 'validation_accuracy': 0.5}), '{"epoch": 3']), encoding='utf-8')
            rows, errors = load_metric_history(first)
            self.assertEqual([row['epoch'] for row in rows], [1, 2])
            self.assertEqual(len(errors), 1)
            discovered = discover_run_results(root)
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0]['status'], 'incomplete')
            self.assertEqual(discovered[0]['current_epoch'], 2)
            self.assertIn('validation_accuracy', discovered[0]['metric_names'])
