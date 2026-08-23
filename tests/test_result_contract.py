"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_result_contract.py ---
import json

# --- merged from test_result_contract.py ---
from pathlib import Path

# --- merged from test_result_contract.py ---
import tempfile

# --- merged from test_result_contract.py ---
import unittest

# --- merged from test_result_contract.py ---
from lnl_toolbox.training.results import finalize_result, is_completed_result

# --- merged from test_result_contract.py ---
class _result_contract_ResultContractTest(unittest.TestCase):

    def test_finalizer_preserves_runner_metrics_and_adds_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'seed-3'
            root.mkdir()
            (root / 'final_metrics.json').write_text(json.dumps({'test_accuracy': 0.75, 'best_epoch': 4}), encoding='utf-8')
            result = finalize_result(root, {'seed': 3, 'method': 'gce', 'evaluation': {'primary': 'accuracy', 'selection_split': 'validation'}}, runner='supervised', recipe='example')
            self.assertEqual(result['primary_metric'], {'name': 'test_accuracy', 'value': 0.75})
            self.assertEqual(result['metrics']['test_accuracy'], 0.75)
            self.assertEqual(result['selection']['best_epoch'], 4)
            self.assertFalse(result['test_selection_leakage'])
            self.assertTrue(is_completed_result(root))

    def test_test_selection_is_explicitly_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metrics.jsonl').write_text(json.dumps({'test_accuracy': 0.5}) + '\n', encoding='utf-8')
            result = finalize_result(root, {'evaluation': {'selection_split': 'test'}}, runner='supervised')
            self.assertTrue(result['test_selection_leakage'])

# --- merged from test_curve_comparison.py ---
import json

# --- merged from test_curve_comparison.py ---
import math

# --- merged from test_curve_comparison.py ---
import re

# --- merged from test_curve_comparison.py ---
import tempfile

# --- merged from test_curve_comparison.py ---
import unittest

# --- merged from test_curve_comparison.py ---
from pathlib import Path

# --- merged from test_curve_comparison.py ---
from lnl_toolbox.evaluation.curve_comparison import compare_curves, load_metrics_jsonl, write_curve_comparison

# --- merged from test_curve_comparison.py ---
class _curve_comparison_CurveComparisonTest(unittest.TestCase):

    def test_compare_and_write_outputs(self) -> None:
        rows = [{'event': 'epoch', 'epoch': 1, 'validation_accuracy': 0.5}, {'event': 'epoch', 'epoch': 2, 'validation_accuracy': 0.7}]
        summary = compare_curves(rows, {'epoch': [1, 2], 'validation_accuracy': [0.4, 0.8]})
        self.assertEqual(summary['overlap_epochs'], 2)
        self.assertAlmostEqual(summary['mean_absolute_error'], 0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.jsonl'
            path.write_text('\n'.join((json.dumps(row) for row in rows)), encoding='utf-8')
            self.assertEqual(len(load_metrics_jsonl(path)), 2)
            outputs = write_curve_comparison(rows, {'epoch': [1, 2], 'validation_accuracy': [0.4, 0.8]}, directory)
            self.assertTrue(all((path.is_file() for path in outputs.values())))
            self.assertEqual(json.loads(outputs['summary'].read_text('utf-8'))['overlap_epochs'], 2)

    def test_invalid_curve_values_and_json_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'finite'):
            compare_curves([{'epoch': 1, 'validation_accuracy': math.nan}], {'epoch': [1], 'validation_accuracy': [0.5]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.jsonl'
            path.write_text('{invalid', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'line 1'):
                load_metrics_jsonl(path)

    def test_epoch_contract_rejects_missing_duplicate_and_no_overlap(self):
        with self.assertRaisesRegex(KeyError, 'missing epoch'):
            compare_curves([{'validation_accuracy': 0.5}], {'epoch': [1], 'validation_accuracy': [0.5]})
        with self.assertRaisesRegex(ValueError, 'duplicate epoch 1'):
            compare_curves([{'epoch': 1, 'validation_accuracy': 0.5}, {'epoch': 1, 'validation_accuracy': 0.6}], {'epoch': [1], 'validation_accuracy': [0.5]})
        with self.assertRaisesRegex(ValueError, 'finite integers'):
            compare_curves([{'epoch': 1.0, 'validation_accuracy': 0.5}], {'epoch': [1], 'validation_accuracy': [0.5]})
        with self.assertRaisesRegex(ValueError, 'no overlapping epochs'):
            compare_curves([{'epoch': 1, 'validation_accuracy': 0.5}], {'epoch': [2], 'validation_accuracy': [0.5]})

    def test_out_of_order_partial_overlap_uses_max_common_epoch(self):
        summary = compare_curves([{'epoch': 5, 'validation_accuracy': 0.9}, {'epoch': 1, 'validation_accuracy': 0.1}, {'epoch': 3, 'validation_accuracy': 0.7}], [{'epoch': 4, 'validation_accuracy': 0.8}, {'epoch': 3, 'validation_accuracy': 0.6}, {'epoch': 1, 'validation_accuracy': 0.2}])
        self.assertEqual(summary['epochs'], [1, 3])
        self.assertEqual(summary['final_epoch'], 3)
        self.assertEqual(summary['final_reproduced'], 0.7)
        self.assertEqual(summary['final_paper'], 0.6)
        self.assertEqual(len(summary['differences']), 2)
        self.assertAlmostEqual(summary['differences'][0], -0.1)
        self.assertAlmostEqual(summary['differences'][1], 0.1)

    def test_outputs_share_coordinates_and_are_byte_deterministic(self):
        ours = [{'epoch': 1, 'validation_accuracy': 0.0}, {'epoch': 3, 'validation_accuracy': 10.0}]
        paper = [{'epoch': 2, 'validation_accuracy': 5.0}, {'epoch': 3, 'validation_accuracy': 10.0}]
        with tempfile.TemporaryDirectory() as directory:
            first = write_curve_comparison(ours, paper, directory)
            first_bytes = {name: path.read_bytes() for name, path in first.items()}
            second = write_curve_comparison(ours, paper, directory)
            self.assertEqual(first_bytes, {name: path.read_bytes() for name, path in second.items()})
            svg = first['overlay'].read_text(encoding='utf-8')
            lines = dict(re.findall('data-series="([^"]+)".*?points="([^"]+)"', svg))
            self.assertEqual(lines['reproduction'], '80.00,300.00 840.00,80.00')
            self.assertEqual(lines['reference'], '460.00,190.00 840.00,80.00')
            csv_text = first['difference'].read_text(encoding='utf-8')
            self.assertIn('3,10.0,10.0,0.0', csv_text)

# --- merged from test_run_comparison.py ---
import csv

# --- merged from test_run_comparison.py ---
import json

# --- merged from test_run_comparison.py ---
from pathlib import Path

# --- merged from test_run_comparison.py ---
import tempfile

# --- merged from test_run_comparison.py ---
import unittest

# --- merged from test_run_comparison.py ---
import yaml

# --- merged from test_run_comparison.py ---
from lnl_toolbox.evaluation.run_comparison import compare_runs, write_report

# --- merged from test_run_comparison.py ---
class _run_comparison_RunComparisonTest(unittest.TestCase):

    def _run(self, root: Path, seed: int, accuracy: float, model: str, *, method: str='gce', rate: float=0.4, manifest: str | None=None, metric: str='test_accuracy', leakage: bool=False) -> Path:
        run = root / f'{method}-seed-{seed}-rate-{rate}-{metric}'
        run.mkdir()
        (run / 'resolved_config.yaml').write_text(yaml.safe_dump({'seed': seed, 'execution': {'runner': 'supervised'}, 'data': {'name': 'cifar10', 'augment': True}, 'model': {'name': model}, 'trainer': {'epochs': 10}, 'loss': {'name': method}, 'noise': {'name': 'symmetric', 'rate': rate}, 'evaluation': {'primary': metric}}), encoding='utf-8')
        (run / 'final_metrics.json').write_text(json.dumps({'status': 'completed', 'method': method, 'primary_metric': {'name': metric, 'value': accuracy}, 'selection': {'split': 'validation'}, 'test_selection_leakage': leakage, 'metrics': {'noise': {'mapping_hash': manifest}}}), encoding='utf-8')
        return run

    def test_statistics_warnings_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1, 0.7, 'resnet18')
            self._run(root, 2, 0.9, 'resnet34')
            summary = compare_runs(root)
            row = summary['summaries'][0]
            self.assertEqual(row['n'], 2)
            self.assertAlmostEqual(row['mean'], 0.8)
            self.assertTrue(any(('model' in warning for warning in summary['warnings'])))
            paths = write_report(summary, root / 'report')
            self.assertTrue(all((path.is_file() for path in paths.values())))
            with paths['csv'].open(encoding='utf-8') as handle:
                self.assertEqual(next(csv.DictReader(handle))['method'], 'gce')

    def test_methods_compare_without_warning_and_noise_rate_is_research_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1, 0.7, 'resnet18', method='ce', rate=0.2, manifest='same')
            self._run(root, 1, 0.8, 'resnet18', method='gce', rate=0.2, manifest='same')
            self._run(root, 1, 0.6, 'resnet18', method='ce', rate=0.4, manifest='other')
            summary = compare_runs(root, group_by=('method', 'noise.rate'), require_equal=('dataset', 'model', 'noise.rate'))
            self.assertEqual(len(summary['summaries']), 3)
            self.assertFalse(any(('noise.rate differs' in item for item in summary['warnings'])))
            self.assertFalse(summary['compatibility_findings'])

    def test_manifest_is_compared_across_methods_only_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1, 0.7, 'resnet18', method='ce', manifest='a')
            self._run(root, 1, 0.8, 'resnet18', method='gce', manifest='b')
            self._run(root, 2, 0.75, 'resnet18', method='ce', manifest='c')
            self._run(root, 2, 0.85, 'resnet18', method='gce', manifest='c')
            summary = compare_runs(root)
            manifest_findings = [item for item in summary['compatibility_findings'] if item['field'] == 'noise_manifest']
            self.assertEqual(len(manifest_findings), 1)
            self.assertEqual(manifest_findings[0]['group']['seed'], 1)

    def test_different_metrics_never_share_an_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1, 0.7, 'resnet18', metric='test_accuracy')
            self._run(root, 2, 0.6, 'resnet18', metric='test_f1')
            summary = compare_runs(root, group_by=('method', 'noise.rate'))
            self.assertEqual(len(summary['summaries']), 2)
            self.assertEqual({item['metric'] for item in summary['summaries']}, {'test_accuracy', 'test_f1'})

    def test_strict_leakage_excludes_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaked = self._run(root, 1, 0.9, 'resnet18', leakage=True)
            normal = compare_runs(root)
            self.assertEqual(normal['summaries'][0]['n'], 1)
            strict = compare_runs(root, strict=True)
            self.assertFalse(strict['summaries'])
            self.assertEqual(strict['excluded_runs'][0]['run_dir'], str(leaked))
