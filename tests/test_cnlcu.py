"""Merged unit tests; source modules were consolidated without changing assertions."""

# --- merged from test_cnlcu_algorithm.py ---
import copy

# --- merged from test_cnlcu_algorithm.py ---
import unittest

# --- merged from test_cnlcu_algorithm.py ---
from pathlib import Path

# --- merged from test_cnlcu_algorithm.py ---
import torch

# --- merged from test_cnlcu_algorithm.py ---
from lnl_toolbox.algorithms.cnlcu import CNLCUAlgorithm, CNLCUConfig

# --- merged from test_cnlcu_algorithm.py ---
from lnl_toolbox.algorithms.coteaching.selection import determine_keep_count, stable_small_loss_mask

# --- merged from test_cnlcu_algorithm.py ---
from lnl_toolbox.core import Batch, ExperimentContext, RunState

# --- merged from test_cnlcu_algorithm.py ---
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss, validate_per_sample_loss

# --- merged from test_cnlcu_algorithm.py ---
def _cnlcu_algorithm__config(window_size=3, sigma=0.1, variant='soft'):
    result = {'method': 'cnlcu', 'noise': {'name': 'symmetric', 'rate': 0.5}, 'cnlcu': {'variant': variant, 'model_count': 2, 'noise_rate': 0.5, 'initialization': {'peer_seed_offset': 1}, 'remember_schedule': {'name': 'linear', 'start': 1.0, 'end': 0.5, 'gradual_epochs': 1}, 'history': {'window_size': window_size, 'storage_dtype': 'float32'}, 'selection': {'count_rule': 'floor', 'tie_break': 'stable_sample_index'}}, 'evaluation': {'selection_split': 'validation', 'primary': 'mean_peer_accuracy', 'ensemble': 'mean_probabilities'}}
    if variant == 'soft':
        result['cnlcu']['uncertainty'] = {'sigma_squared': sigma}
    elif variant == 'hard':
        result['cnlcu'].update({'hard_fidelity': 'paper_formula_corrected_lof', 'uncertainty': {'tau_min': 0.0001, 'loss_upper_bound': {'mode': 'fixed', 'value': 20.0}}, 'truncation': {'method': 'lof', 'n_neighbors': 2, 'contamination': 0.1, 'minimum_observations': 3}})
    return result

# --- merged from test_cnlcu_algorithm.py ---
class _cnlcu_algorithm__IndexedLogits(torch.nn.Module):

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs):
        return self.logits[inputs.reshape(-1).long()]

# --- merged from test_cnlcu_algorithm.py ---
def _cnlcu_algorithm__algorithm(variant='soft'):
    a = _cnlcu_algorithm__IndexedLogits([[5, -5], [5, -5], [-5, 5], [-5, 5]])
    b = _cnlcu_algorithm__IndexedLogits([[-5, 5], [-5, 5], [5, -5], [5, -5]])
    result = CNLCUAlgorithm(model_a=a, model_b=b, optimizer_a=torch.optim.SGD(a.parameters(), lr=0.1), optimizer_b=torch.optim.SGD(b.parameters(), lr=0.1), scheduler_a=None, scheduler_b=None, loss=CrossEntropyLoss(), device=torch.device('cpu'), method_config=CNLCUConfig.from_mapping(_cnlcu_algorithm__config(variant=variant)), canonical_global_indices=torch.tensor([40, 10, 30, 20]))
    result.setup(ExperimentContext(Path.cwd()))
    result.on_cycle_start(RunState(cycle=1))
    return result

# --- merged from test_cnlcu_algorithm.py ---
def _cnlcu_algorithm__reference_step(algorithm, batch, state):
    """Execute the pre-telemetry CNLCU step contract for regression comparison."""
    payload = batch.payload
    inputs = payload['input'].to(algorithm.device)
    targets = payload['target'].to(algorithm.device)
    indices = torch.as_tensor(payload['index'], dtype=torch.long, device=algorithm.device)
    logits_a, logits_b = (algorithm.model_a(inputs), algorithm.model_b(inputs))
    losses_a = validate_per_sample_loss(algorithm.loss(logits_a, targets), targets.numel())
    losses_b = validate_per_sample_loss(algorithm.loss(logits_b, targets), targets.numel())
    rows_a = algorithm.private_state.history_a.append(indices, losses_a.detach())
    rows_b = algorithm.private_state.history_b.append(indices, losses_b.detach())
    score_a, _ = algorithm._score(algorithm.private_state.history_a, rows_a)
    score_b, _ = algorithm._score(algorithm.private_state.history_b, rows_b)
    keep_count = determine_keep_count(int(targets.numel()), algorithm.method_config.rate_at(state.cycle))
    selected_a = stable_small_loss_mask(score_a.to(algorithm.device), indices, keep_count)
    selected_b = stable_small_loss_mask(score_b.to(algorithm.device), indices, keep_count)
    objective_a = losses_a[selected_b].mean()
    objective_b = losses_b[selected_a].mean()
    algorithm.optimizer_a.zero_grad(set_to_none=True)
    objective_a.backward()
    algorithm.optimizer_a.step()
    algorithm.private_state.optimizer_steps_a += 1
    algorithm.optimizer_b.zero_grad(set_to_none=True)
    objective_b.backward()
    algorithm.optimizer_b.step()
    algorithm.private_state.optimizer_steps_b += 1
    algorithm.private_state.history_a.increment_selected(rows_a, selected_a)
    algorithm.private_state.history_b.increment_selected(rows_b, selected_b)
    state.step += 1
    return {'selected_a': indices[selected_a].detach().cpu(), 'selected_b': indices[selected_b].detach().cpu(), 'objective_a': objective_a.detach(), 'objective_b': objective_b.detach()}

# --- merged from test_cnlcu_algorithm.py ---
class _cnlcu_algorithm_CNLCUAlgorithmTest(unittest.TestCase):

    def test_read_only_telemetry_preserves_selection_loss_and_parameter_updates(self):
        observed, reference = (_cnlcu_algorithm__algorithm(), _cnlcu_algorithm__algorithm())
        observed_state, reference_state = (RunState(cycle=1), RunState(cycle=1))
        batch = Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.tensor([10, 20, 30, 40])})
        for cycle in (1, 2):
            if cycle > 1:
                observed.on_cycle_start(RunState(cycle=cycle))
                reference.on_cycle_start(RunState(cycle=cycle))
                observed_state.cycle = reference_state.cycle = cycle
            result = observed.step(batch, observed_state)
            expected = _cnlcu_algorithm__reference_step(reference, batch, reference_state)
            self.assertEqual(result.metadata['selected_by_a_indices'].tolist(), expected['selected_a'].tolist())
            self.assertEqual(result.metadata['selected_by_b_indices'].tolist(), expected['selected_b'].tolist())
            torch.testing.assert_close(torch.tensor(result.metrics['loss_a_on_selected_by_b']), expected['objective_a'], rtol=0.0, atol=0.0)
            torch.testing.assert_close(torch.tensor(result.metrics['loss_b_on_selected_by_a']), expected['objective_b'], rtol=0.0, atol=0.0)
            for observed_model, reference_model in ((observed.model_a, reference.model_a), (observed.model_b, reference.model_b)):
                for left, right in zip(observed_model.parameters(), reference_model.parameters(), strict=True):
                    torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
            for key in ('gradient_norm_a', 'gradient_norm_b', 'parameter_norm_a', 'parameter_norm_b'):
                self.assertTrue(torch.isfinite(torch.tensor(result.metrics[key])))

    def test_optimizer_parameters_exactly_match_and_do_not_overlap(self):
        algorithm = _cnlcu_algorithm__algorithm()
        model_a = {id(parameter) for parameter in algorithm.model_a.parameters()}
        model_b = {id(parameter) for parameter in algorithm.model_b.parameters()}
        optimizer_a = {id(parameter) for group in algorithm.optimizer_a.param_groups for parameter in group['params']}
        optimizer_b = {id(parameter) for group in algorithm.optimizer_b.param_groups for parameter in group['params']}
        self.assertEqual(optimizer_a, model_a)
        self.assertEqual(optimizer_b, model_b)
        self.assertFalse(optimizer_a & optimizer_b)

    def test_misbound_or_overlapping_optimizer_parameters_fail(self):
        model_a = _cnlcu_algorithm__IndexedLogits([[1, 0]])
        model_b = _cnlcu_algorithm__IndexedLogits([[0, 1]])
        common = dict(scheduler_a=None, scheduler_b=None, loss=CrossEntropyLoss(), device=torch.device('cpu'), method_config=CNLCUConfig.from_mapping(_cnlcu_algorithm__config()), canonical_global_indices=torch.tensor([10]))
        with self.assertRaisesRegex(ValueError, 'optimizer a parameters'):
            CNLCUAlgorithm(model_a=model_a, model_b=model_b, optimizer_a=torch.optim.SGD(model_b.parameters(), lr=0.1), optimizer_b=torch.optim.SGD(model_b.parameters(), lr=0.1), **common)
        shared = torch.nn.Parameter(torch.tensor([[1.0, 0.0]]))
        model_a.logits = shared
        model_b.logits = shared
        with self.assertRaisesRegex(ValueError, 'model parameter sets must not overlap'):
            CNLCUAlgorithm(model_a=model_a, model_b=model_b, optimizer_a=torch.optim.SGD(model_a.parameters(), lr=0.1), optimizer_b=torch.optim.SGD(model_b.parameters(), lr=0.1), **common)

    def test_peer_cross_update_and_peer_specific_counts(self):
        algorithm = _cnlcu_algorithm__algorithm()
        before_a, before_b = (algorithm.model_a.logits.detach().clone(), algorithm.model_b.logits.detach().clone())
        state = RunState(cycle=1)
        result = algorithm.step(Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.tensor([10, 20, 30, 40])}), state)
        self.assertEqual(result.metadata['selected_by_a_indices'].tolist(), [10, 20])
        self.assertEqual(result.metadata['selected_by_b_indices'].tolist(), [30, 40])
        torch.testing.assert_close(algorithm.model_a.logits[:2], before_a[:2])
        self.assertFalse(torch.equal(algorithm.model_a.logits[2:], before_a[2:]))
        self.assertFalse(torch.equal(algorithm.model_b.logits[:2], before_b[:2]))
        torch.testing.assert_close(algorithm.model_b.logits[2:], before_b[2:])
        rows_a = algorithm.private_state.history_a.resolve(torch.tensor([10, 20, 30, 40]))
        rows_b = algorithm.private_state.history_b.resolve(torch.tensor([10, 20, 30, 40]))
        self.assertEqual(algorithm.private_state.history_a.selected_count[rows_a].tolist(), [1, 1, 0, 0])
        self.assertEqual(algorithm.private_state.history_b.selected_count[rows_b].tolist(), [0, 0, 1, 1])

    def test_hard_peer_cross_update_uses_peer_selections(self):
        algorithm = _cnlcu_algorithm__algorithm('hard')
        before_a = algorithm.model_a.logits.detach().clone()
        before_b = algorithm.model_b.logits.detach().clone()
        result = algorithm.step(Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.tensor([10, 20, 30, 40])}), RunState(cycle=1))
        self.assertEqual(result.metadata['selected_by_a_indices'].tolist(), [10, 20])
        self.assertEqual(result.metadata['selected_by_b_indices'].tolist(), [30, 40])
        torch.testing.assert_close(algorithm.model_a.logits[:2], before_a[:2])
        self.assertFalse(torch.equal(algorithm.model_a.logits[2:], before_a[2:]))
        self.assertFalse(torch.equal(algorithm.model_b.logits[:2], before_b[:2]))
        torch.testing.assert_close(algorithm.model_b.logits[2:], before_b[2:])
        self.assertIn('hard_confidence_bonus_a', result.metrics)
        self.assertIn('outlier_ratio_b', result.metrics)

    def test_current_loss_is_appended_before_score_and_clean_oracle_is_ignored(self):
        first, second = (_cnlcu_algorithm__algorithm(), _cnlcu_algorithm__algorithm())
        payload = {'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.tensor([10, 20, 30, 40])}
        left = first.step(Batch(payload), RunState(cycle=1))
        right = second.step(Batch({**payload, 'clean_target': torch.ones(4, dtype=torch.long)}), RunState(cycle=1))
        self.assertEqual(left.metadata['selected_by_a_indices'].tolist(), right.metadata['selected_by_a_indices'].tolist())
        self.assertEqual(left.metrics['history_length_a'], 1.0)
        for a, b in zip(first.model_a.parameters(), second.model_a.parameters()):
            torch.testing.assert_close(a, b)

    def test_uncertainty_count_can_change_current_loss_ranking(self):
        algorithm = _cnlcu_algorithm__algorithm()
        algorithm.model_a.logits.data.copy_(torch.tensor([[0.4, 0.0], [0.4, 0.0], [0.2, 0.0], [0.2, 0.0]]))
        rows = algorithm.private_state.history_a.resolve(torch.tensor([10, 20, 30, 40]))
        algorithm.private_state.history_a.selected_count[rows] = torch.tensor([100, 100, 0, 0])
        result = algorithm.step(Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.tensor([10, 20, 30, 40])}), RunState(cycle=1))
        self.assertEqual(result.metadata['selected_by_a_indices'].tolist(), [30, 40])

    def test_checkpoint_roundtrip_and_wrong_identity_fail(self):
        algorithm = _cnlcu_algorithm__algorithm()
        algorithm.step(Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.tensor([10, 20, 30, 40])}), RunState(cycle=1))
        saved = copy.deepcopy(algorithm.state_dict())
        restored = _cnlcu_algorithm__algorithm()
        restored.load_state_dict(saved)
        self.assertEqual(restored.private_state.history_a.selected_count.tolist(), algorithm.private_state.history_a.selected_count.tolist())
        wrong = copy.deepcopy(saved)
        wrong['method_identity'] = 'coteaching'
        with self.assertRaisesRegex(ValueError, 'identity'):
            restored.load_state_dict(wrong)
        swapped = copy.deepcopy(saved)
        private = swapped['cnlcu_state']
        private['history_a'], private['history_b'] = (private['history_b'], private['history_a'])
        with self.assertRaisesRegex(ValueError, 'identity'):
            _cnlcu_algorithm__algorithm().load_state_dict(swapped)
        schedule_drift = copy.deepcopy(saved)
        schedule_drift['remember_schedule']['gradual_epochs'] += 1
        with self.assertRaisesRegex(ValueError, 'configuration'):
            _cnlcu_algorithm__algorithm().load_state_dict(schedule_drift)
        count_scope_drift = copy.deepcopy(saved)
        count_scope_drift['selected_count_scope'] = 'global'
        with self.assertRaisesRegex(ValueError, 'configuration'):
            _cnlcu_algorithm__algorithm().load_state_dict(count_scope_drift)

    def test_configuration_rejects_unknown_variant_and_single_model_composition(self):
        values = _cnlcu_algorithm__config()
        values['cnlcu']['variant'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'soft or hard'):
            CNLCUConfig.from_mapping(values)
        for key in ('selector', 'parameter_update', 'weight_provider', 'objective_consumer', 'dss'):
            values = _cnlcu_algorithm__config()
            values[key] = {'name': 'anything'}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                CNLCUConfig.from_mapping(values)

    def test_hard_configuration_and_checkpoint_identity_are_strict(self):
        config = CNLCUConfig.from_mapping(_cnlcu_algorithm__config(variant='hard'))
        self.assertEqual(config.hard_fidelity, 'paper_formula_corrected_lof')
        for field, value in (('hard_fidelity', 'released_code'), ('truncation.method', 'knn'), ('truncation.minimum_observations', 2), ('uncertainty.loss_upper_bound.mode', 'percentile')):
            values = _cnlcu_algorithm__config(variant='hard')
            owner, key = field.split('.', 1) if '.' in field else ('cnlcu', field)
            if owner == 'cnlcu':
                values['cnlcu'][key] = value
            elif '.' in key:
                first, second = key.split('.')
                values['cnlcu'][owner][first][second] = value
            else:
                values['cnlcu'][owner][key] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                CNLCUConfig.from_mapping(values)
        algorithm = _cnlcu_algorithm__algorithm('hard')
        saved = copy.deepcopy(algorithm.state_dict())
        self.assertEqual(saved['hard_identity']['truncation']['method'], 'lof')
        changed = copy.deepcopy(saved)
        changed['hard_identity']['tau_min'] = 0.5
        with self.assertRaisesRegex(ValueError, 'CNLCU-H checkpoint'):
            _cnlcu_algorithm__algorithm('hard').load_state_dict(changed)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            _cnlcu_algorithm__algorithm('soft').load_state_dict(saved)

    def test_hard_fixed_loss_bound_is_enforced_without_clipping(self):
        algorithm = _cnlcu_algorithm__algorithm('hard')
        algorithm.method_config = CNLCUConfig.from_mapping(_cnlcu_algorithm__config(variant='hard'))
        algorithm.model_a.logits.data[0] = torch.tensor([-100.0, 100.0])
        with self.assertRaisesRegex(ValueError, 'exceeded fixed loss_upper_bound'):
            algorithm.step(Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.tensor([10, 20, 30, 40])}), RunState(cycle=1))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_step_keeps_history_on_cpu(self):
        model_a = _cnlcu_algorithm__IndexedLogits([[5, -5], [5, -5], [-5, 5], [-5, 5]])
        model_b = _cnlcu_algorithm__IndexedLogits([[-5, 5], [-5, 5], [5, -5], [5, -5]])
        algorithm = CNLCUAlgorithm(model_a=model_a, model_b=model_b, optimizer_a=torch.optim.SGD(model_a.parameters(), lr=0.1), optimizer_b=torch.optim.SGD(model_b.parameters(), lr=0.1), scheduler_a=None, scheduler_b=None, loss=CrossEntropyLoss(), device=torch.device('cuda'), method_config=CNLCUConfig.from_mapping(_cnlcu_algorithm__config()), canonical_global_indices=torch.tensor([40, 10, 30, 20]))
        algorithm.setup(ExperimentContext(Path.cwd()))
        algorithm.on_cycle_start(RunState(cycle=1))
        algorithm.step(Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.tensor([10, 20, 30, 40])}), RunState(cycle=1))
        self.assertEqual(algorithm.private_state.history_a.values.device.type, 'cpu')
        self.assertEqual(algorithm.private_state.history_b.values.device.type, 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_hard_cuda_step_keeps_lof_history_on_cpu(self):
        algorithm = _cnlcu_algorithm__algorithm('hard')
        algorithm.device = torch.device('cuda')
        algorithm.setup(ExperimentContext(Path.cwd()))
        algorithm.step(Batch({'input': torch.arange(4), 'target': torch.zeros(4, dtype=torch.long), 'index': torch.tensor([10, 20, 30, 40])}), RunState(cycle=1))
        self.assertEqual(algorithm.private_state.history_a.values.device.type, 'cpu')
        self.assertEqual(algorithm.private_state.history_b.values.device.type, 'cpu')

# --- merged from test_cnlcu_estimators.py ---
import copy

# --- merged from test_cnlcu_estimators.py ---
import math

# --- merged from test_cnlcu_estimators.py ---
import unittest

# --- merged from test_cnlcu_estimators.py ---
from unittest.mock import patch

# --- merged from test_cnlcu_estimators.py ---
import torch

# --- merged from test_cnlcu_estimators.py ---
from lnl_toolbox.algorithms.cnlcu import HardRobustLossEstimator, PeerLossHistory, cnlcu_hard_score, cnlcu_soft_score, lof_retained_mask, soft_influence, soft_robust_mean

# --- merged from test_cnlcu_estimators.py ---
class _cnlcu_estimators_CNLCUSoftEstimatorTest(unittest.TestCase):

    def test_influence_and_eq3_match_hand_calculation(self):
        losses = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        observed = torch.tensor([[True, True], [True, False]])
        transformed = soft_influence(losses)
        torch.testing.assert_close(transformed[0], torch.log(torch.tensor([1.0, 2.5])))
        means, lengths = soft_robust_mean(losses, observed)
        self.assertEqual(lengths.tolist(), [2, 1])
        self.assertAlmostEqual(means[0].item(), math.log(2.5) / 2, places=6)
        self.assertAlmostEqual(means[1].item(), math.log(5.0), places=6)

    def test_eq7_matches_hand_calculation_without_relu(self):
        mean = torch.tensor([0.1])
        t = torch.tensor([2])
        count = torch.tensor([1])
        score, bonus = cnlcu_soft_score(mean, t, count, 0.2)
        expected = 0.2 * (2 + 0.2 * math.log(4) / 4) / 0.8
        self.assertAlmostEqual(bonus.item(), expected, places=6)
        self.assertAlmostEqual(score.item(), 0.1 - expected, places=6)
        self.assertLess(score.item(), 0.0)

    def test_less_selected_sample_gets_larger_bonus(self):
        mean = torch.tensor([1.0, 1.0])
        score, bonus = cnlcu_soft_score(mean, torch.tensor([3, 3]), torch.tensor([1, 5]), 0.1)
        self.assertGreater(bonus[0], bonus[1])
        self.assertLess(score[0], score[1])

    def test_invalid_values_fail(self):
        for sigma in (0.0, 1.0, math.nan):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                cnlcu_soft_score(torch.tensor([1.0]), torch.tensor([1]), torch.tensor([1]), sigma)
        with self.assertRaisesRegex(ValueError, 'denominator'):
            cnlcu_soft_score(torch.tensor([1.0]), torch.tensor([1]), torch.tensor([0.05]), 0.1)
        with self.assertRaisesRegex(ValueError, 'finite'):
            soft_influence(torch.tensor([math.inf]))
        with self.assertRaisesRegex(ValueError, 'at least one'):
            soft_robust_mean(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))

# --- merged from test_cnlcu_estimators.py ---
class _cnlcu_estimators_CNLCUHistoryTest(unittest.TestCase):

    def test_sparse_mapping_append_permutation_and_window_reset(self):
        history = PeerLossHistory(torch.tensor([40, 10, 90]), 2, 'a')
        history.prepare_epoch(0)
        rows = history.append(torch.tensor([90, 10]), torch.tensor([0.9, 0.1]))
        values, observed, counts = history.lookup_rows(rows)
        torch.testing.assert_close(values[:, 0], torch.tensor([0.9, 0.1]))
        self.assertTrue(bool(observed[:, 0].all()))
        history.increment_selected(rows, torch.tensor([False, True]))
        self.assertEqual(history.lookup_rows(rows)[2].tolist(), [0, 1])
        history.prepare_epoch(1)
        history.append(torch.tensor([10, 90]), torch.tensor([0.2, 0.8]))
        self.assertEqual(history.lookup_rows(history.resolve(torch.tensor([90])))[1].sum().item(), 2)
        history.prepare_epoch(2)
        self.assertFalse(bool(history.observed.any()))
        self.assertEqual(history.selected_count.sum().item(), 0)

    def test_selection_count_resets_with_each_epoch_window(self):
        history = PeerLossHistory(torch.tensor([10, 20]), 2, 'a')
        history.prepare_epoch(0)
        rows = history.append(torch.tensor([10, 20]), torch.tensor([0.1, 0.2]))
        history.increment_selected(rows, torch.tensor([True, False]))
        history.prepare_epoch(1)
        self.assertEqual(history.selected_count.tolist(), [1, 0])
        history.prepare_epoch(2)
        self.assertEqual(history.window_start_epoch, 2)
        self.assertEqual(history.selected_count.tolist(), [0, 0])

    def test_duplicate_missing_and_double_observation_fail(self):
        history = PeerLossHistory(torch.tensor([10, 20]), 2, 'a')
        history.prepare_epoch(0)
        with self.assertRaisesRegex(ValueError, 'unique'):
            history.resolve(torch.tensor([10, 10]))
        with self.assertRaises(KeyError):
            history.resolve(torch.tensor([30]))
        history.append(torch.tensor([10]), torch.tensor([0.1]))
        with self.assertRaisesRegex(ValueError, 'twice'):
            history.append(torch.tensor([10]), torch.tensor([0.2]))

    def test_state_roundtrip_and_mapping_drift_rejected(self):
        first = PeerLossHistory(torch.tensor([7, 20]), 3, 'a')
        first.prepare_epoch(0)
        rows = first.append(torch.tensor([20]), torch.tensor([0.5]))
        first.increment_selected(rows, torch.tensor([True]))
        state = copy.deepcopy(first.state_dict())
        restored = PeerLossHistory(torch.tensor([20, 7]), 3, 'a')
        restored.load_state_dict(state)
        self.assertEqual(restored.selected_count.tolist(), first.selected_count.tolist())
        with self.assertRaisesRegex(ValueError, 'mapping'):
            PeerLossHistory(torch.tensor([7, 21]), 3, 'a').load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'identity'):
            PeerLossHistory(torch.tensor([7, 20]), 3, 'b').load_state_dict(state)
        negative = copy.deepcopy(state)
        negative['selected_count'][0] = -1
        with self.assertRaisesRegex(ValueError, 'counts'):
            PeerLossHistory(torch.tensor([7, 20]), 3, 'a').load_state_dict(negative)
        bad_cursor = copy.deepcopy(state)
        bad_cursor['active_epoch'] = 4
        with self.assertRaisesRegex(ValueError, 'cursor'):
            PeerLossHistory(torch.tensor([7, 20]), 3, 'a').load_state_dict(bad_cursor)

# --- merged from test_cnlcu_estimators.py ---
class _cnlcu_estimators_CNLCUHardEstimatorTest(unittest.TestCase):

    def test_short_history_keeps_every_observation(self):
        history = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
        observed = torch.tensor([[True, True, False], [True, False, False]])
        retained = lof_retained_mask(history, observed, n_neighbors=2, contamination=0.1, minimum_observations=3)
        torch.testing.assert_close(retained, observed)
        estimate = HardRobustLossEstimator(n_neighbors=2, contamination=0.1, minimum_observations=3).estimate(history, observed)
        torch.testing.assert_close(estimate.robust_mean, torch.tensor([1.5, 3.0], dtype=torch.float64))
        self.assertEqual(estimate.outlier_count.tolist(), [0, 0])

    def test_corrected_lof_removes_minus_one_and_uses_retained_denominator(self):

        class FakeLOF:

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit_predict(self, samples):
                self.samples = samples
                return [-1, 1, 1]
        history = torch.tensor([[100.0, 1.0, 2.0, 999.0]])
        observed = torch.tensor([[True, True, True, False]])
        with patch('lnl_toolbox.algorithms.cnlcu.outliers._load_local_outlier_factor', return_value=FakeLOF):
            estimate = HardRobustLossEstimator(n_neighbors=2, contamination=0.1, minimum_observations=3).estimate(history, observed)
        self.assertEqual(estimate.retained_mask.tolist(), [[False, True, True, False]])
        self.assertEqual(estimate.observation_count.tolist(), [3])
        self.assertEqual(estimate.outlier_count.tolist(), [1])
        self.assertEqual(estimate.retained_count.tolist(), [2])
        self.assertEqual(estimate.robust_mean.tolist(), [1.5])
        self.assertNotEqual(estimate.robust_mean.item(), (1.0 + 2.0) / 3.0)

    def test_actual_lof_is_deterministic_and_removes_extreme_value(self):
        history = torch.tensor([[1.0] * 9 + [100.0], [2.0] * 10])
        observed = torch.ones_like(history, dtype=torch.bool)
        kwargs = dict(n_neighbors=2, contamination=0.1, minimum_observations=5)
        first = lof_retained_mask(history, observed, **kwargs)
        second = lof_retained_mask(history, observed, **kwargs)
        torch.testing.assert_close(first, second)
        self.assertFalse(first[0, -1].item())
        self.assertTrue(bool(first[1].any()))

    def test_padding_permutations_and_invalid_detector_results(self):
        history = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 0.0]])
        observed = torch.tensor([[True, True, True], [True, True, False]])
        estimator = HardRobustLossEstimator(n_neighbors=2, contamination=0.1, minimum_observations=4)
        first = estimator.estimate(history, observed)
        permutation = torch.tensor([1, 0])
        second = estimator.estimate(history[permutation], observed[permutation])
        torch.testing.assert_close(first.robust_mean[permutation], second.robust_mean)
        self.assertFalse(bool((first.retained_mask & ~observed).any()))

        class RejectAll:

            def __init__(self, **kwargs):
                pass

            def fit_predict(self, samples):
                return [-1] * len(samples)
        with patch('lnl_toolbox.algorithms.cnlcu.outliers._load_local_outlier_factor', return_value=RejectAll), self.assertRaisesRegex(RuntimeError, 'every observation'):
            lof_retained_mask(history[:1], observed[:1], n_neighbors=2, contamination=0.1, minimum_observations=3)

    def test_missing_sklearn_and_invalid_parameters_fail(self):
        history = torch.ones(1, 3)
        observed = torch.ones(1, 3, dtype=torch.bool)
        with patch('lnl_toolbox.algorithms.cnlcu.outliers._load_local_outlier_factor', side_effect=ImportError('optional training dependency [train]')), self.assertRaisesRegex(ImportError, 'optional training dependency'):
            lof_retained_mask(history, observed, n_neighbors=2, contamination=0.1, minimum_observations=3)
        for kwargs in (dict(n_neighbors=0, contamination=0.1, minimum_observations=3), dict(n_neighbors=2, contamination=0.5, minimum_observations=3), dict(n_neighbors=2, contamination=0.1, minimum_observations=2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                lof_retained_mask(history, observed, **kwargs)

    def test_eq8_matches_hand_calculation_and_allows_negative_score(self):
        mean = torch.tensor([0.1], dtype=torch.float64)
        t = torch.tensor([4])
        outliers = torch.tensor([1])
        retained = torch.tensor([3])
        count = torch.tensor([2])
        score, bonus = cnlcu_hard_score(mean, t, outliers, retained, count, 0.01, 2.0)
        factor = 2 * math.sqrt(0.02) * 2.0 * (4 + math.sqrt(2)) / (3 * math.sqrt(4))
        expected = factor * math.sqrt(math.log(16) / 2)
        self.assertAlmostEqual(bonus.item(), expected, places=12)
        self.assertAlmostEqual(score.item(), 0.1 - expected, places=12)
        self.assertLess(score.item(), 0.0)

    def test_eq8_zero_outliers_count_effect_and_invalid_inputs(self):
        mean = torch.tensor([1.0, 1.0], dtype=torch.float64)
        score, bonus = cnlcu_hard_score(mean, torch.tensor([2, 2]), torch.tensor([0, 0]), torch.tensor([2, 2]), torch.tensor([1, 4]), 0.01, 2.0)
        self.assertGreater(bonus[0], bonus[1])
        self.assertLess(score[0], score[1])
        for tau, bound in ((0.0, 2.0), (0.01, 0.0), (math.nan, 2.0)):
            with self.subTest(tau=tau, bound=bound), self.assertRaises(ValueError):
                cnlcu_hard_score(mean[:1], torch.tensor([1]), torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), tau, bound)
        with self.assertRaisesRegex(ValueError, 'outlier_count'):
            cnlcu_hard_score(mean[:1], torch.tensor([1]), torch.tensor([1]), torch.tensor([0]), torch.tensor([1]), 0.01, 2.0)
        with self.assertRaisesRegex(ValueError, 'finite'):
            cnlcu_hard_score(torch.tensor([math.inf]), torch.tensor([1]), torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), 0.01, 2.0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_eq8_cpu_cuda_agree(self):
        arguments = (torch.tensor([0.5, 1.0], dtype=torch.float64), torch.tensor([3, 4]), torch.tensor([0, 1]), torch.tensor([3, 3]), torch.tensor([1, 2]), 0.01, 2.0)
        cpu = cnlcu_hard_score(*arguments)
        cuda = cnlcu_hard_score(*(value.cuda() if torch.is_tensor(value) else value for value in arguments))
        torch.testing.assert_close(cpu[0], cuda[0].cpu())
        torch.testing.assert_close(cpu[1], cuda[1].cpu())

# --- merged from test_cnlcu_workflow.py ---
import hashlib

# --- merged from test_cnlcu_workflow.py ---
import json

# --- merged from test_cnlcu_workflow.py ---
import tempfile

# --- merged from test_cnlcu_workflow.py ---
import unittest

# --- merged from test_cnlcu_workflow.py ---
from pathlib import Path

# --- merged from test_cnlcu_workflow.py ---
from unittest.mock import patch

# --- merged from test_cnlcu_workflow.py ---
import numpy as np

# --- merged from test_cnlcu_workflow.py ---
import torch

# --- merged from test_cnlcu_workflow.py ---
from lnl_toolbox.data.cifar import CifarData

# --- merged from test_cnlcu_workflow.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_cnlcu_workflow.py ---
def _cnlcu_workflow__cifar(size, split, classes=10):
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(np.zeros((size, 32, 32, 3), dtype=np.uint8), labels, tuple(map(str, range(classes))), split, f'cifar{classes}')

# --- merged from test_cnlcu_workflow.py ---
def _cnlcu_workflow__config(epochs=2, dataset='cifar10'):
    classes = 10 if dataset == 'cifar10' else 100
    return {'method': 'cnlcu', 'seed': 7, 'data': {'name': dataset, 'root': 'unused', 'validation_size': classes, 'max_train_samples': classes * 2, 'max_validation_samples': classes, 'max_test_samples': classes, 'augment': False}, 'noise': {'name': 'symmetric', 'rate': 0.4, 'seed': 17, 'validation_targets': 'noisy'}, 'loss': {'name': 'ce'}, 'model': {'name': 'tiny_cnn', 'width': 2}, 'optimizer': {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0}, 'scheduler': {'name': 'none'}, 'cnlcu': {'variant': 'soft', 'model_count': 2, 'noise_rate': 0.4, 'initialization': {'peer_seed_offset': 1}, 'remember_schedule': {'name': 'linear', 'start': 1.0, 'end': 0.6, 'gradual_epochs': 2}, 'history': {'window_size': 2, 'storage_dtype': 'float32'}, 'uncertainty': {'sigma_squared': 0.01}, 'selection': {'count_rule': 'floor', 'tie_break': 'stable_sample_index'}}, 'loader': {'batch_size': classes, 'num_workers': 0, 'pin_memory': False}, 'evaluation': {'selection_split': 'validation', 'primary': 'mean_peer_accuracy', 'ensemble': 'mean_probabilities'}, 'trainer': {'epochs': epochs, 'device': 'cpu'}}

# --- merged from test_cnlcu_workflow.py ---
def _cnlcu_workflow__hard_config(epochs=2, dataset='cifar10'):
    result = _cnlcu_workflow__config(epochs, dataset)
    result['cnlcu'] = {'variant': 'hard', 'hard_fidelity': 'paper_formula_corrected_lof', 'model_count': 2, 'noise_rate': 0.4, 'initialization': {'peer_seed_offset': 1}, 'remember_schedule': {'name': 'linear', 'start': 1.0, 'end': 0.6, 'gradual_epochs': 2}, 'history': {'window_size': 3, 'storage_dtype': 'float32'}, 'uncertainty': {'tau_min': 0.0001, 'loss_upper_bound': {'mode': 'fixed', 'value': 10.0}}, 'truncation': {'method': 'lof', 'n_neighbors': 2, 'contamination': 0.1, 'minimum_observations': 3}, 'selection': {'count_rule': 'floor', 'tie_break': 'stable_sample_index'}}
    return result

# --- merged from test_cnlcu_workflow.py ---
def _cnlcu_workflow__sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# --- merged from test_cnlcu_workflow.py ---
def _cnlcu_workflow__assert_nested_equal(test_case, left, right):
    test_case.assertEqual(type(left), type(right))
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        test_case.assertEqual(set(left), set(right))
        for key in left:
            _cnlcu_workflow__assert_nested_equal(test_case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test_case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _cnlcu_workflow__assert_nested_equal(test_case, left_item, right_item)
    else:
        test_case.assertEqual(left, right)

# --- merged from test_cnlcu_workflow.py ---
class _cnlcu_workflow_CNLCUWorkflowTest(unittest.TestCase):

    def _run_case(self, dataset='cifar10', variant='soft'):
        classes = 10 if dataset == 'cifar10' else 100
        config_factory = _cnlcu_workflow__config if variant == 'soft' else _cnlcu_workflow__hard_config
        train, test = (_cnlcu_workflow__cifar(classes * 4, 'train', classes), _cnlcu_workflow__cifar(classes * 2, 'test', classes))
        loader_name = 'load_cifar10' if dataset == 'cifar10' else 'load_cifar100'
        with tempfile.TemporaryDirectory() as directory, patch(f'lnl_toolbox.data.sources.{loader_name}', side_effect=lambda _root, split: train if split == 'train' else test):
            run_dir = run_experiment(config_factory(2, dataset), Path(directory) / 'run')
            first = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            state = first['algorithm_private_state']['cnlcu_state']
            self.assertEqual(first['algorithm_private_state']['method_identity'], 'cnlcu')
            self.assertEqual(set(state), {'history_a', 'history_b', 'optimizer_steps_a', 'optimizer_steps_b'})
            self.assertFalse(torch.equal(state['history_a']['values'], state['history_b']['values']))
            manifest_hash, manifest_mtime = (_cnlcu_workflow__sha(run_dir / 'noise_manifest.npz'), (run_dir / 'noise_manifest.npz').stat().st_mtime_ns)
            run_experiment(config_factory(3, dataset), resume=run_dir / 'last.pt')
            resumed = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(resumed['completed_epoch'], 2)
            self.assertEqual(resumed['run_state']['step'], 6)
            self.assertEqual(_cnlcu_workflow__sha(run_dir / 'noise_manifest.npz'), manifest_hash)
            self.assertEqual((run_dir / 'noise_manifest.npz').stat().st_mtime_ns, manifest_mtime)
            uninterrupted_dir = run_experiment(config_factory(3, dataset), Path(directory) / 'uninterrupted')
            uninterrupted = torch.load(uninterrupted_dir / 'last.pt', map_location='cpu', weights_only=False)
            for peer in ('a', 'b'):
                for name, value in resumed['model'][peer].items():
                    torch.testing.assert_close(value, uninterrupted['model'][peer][name])
            self.assertEqual(resumed['run_state'], uninterrupted['run_state'])
            _cnlcu_workflow__assert_nested_equal(self, resumed['optimizer'], uninterrupted['optimizer'])
            _cnlcu_workflow__assert_nested_equal(self, resumed['algorithm_private_state']['schedulers'], uninterrupted['algorithm_private_state']['schedulers'])
            _cnlcu_workflow__assert_nested_equal(self, resumed['algorithm_private_state']['cnlcu_state'], uninterrupted['algorithm_private_state']['cnlcu_state'])
            self.assertEqual(resumed['component_states'], uninterrupted['component_states'])
            resumed_epochs = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(line)['event'] == 'epoch']
            uninterrupted_epochs = [json.loads(line) for line in (uninterrupted_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(line)['event'] == 'epoch']
            self.assertEqual(resumed_epochs, uninterrupted_epochs)
            for row in resumed_epochs:
                for key in ('selected_by_a_ratio', 'selected_by_b_ratio', 'train_gradient_norm_a', 'train_gradient_norm_b', 'train_gradient_norm_a_max', 'train_gradient_norm_b_max', 'train_parameter_norm_a', 'train_parameter_norm_b', 'train_uncertainty_score_min_a', 'train_uncertainty_score_max_a', 'train_history_length_min_a', 'train_history_length_max_a', 'history_window_start_epoch', 'history_window_epoch_count'):
                    self.assertIn(key, row)
                    self.assertTrue(np.isfinite(row[key]), key)
            checkpoint_hash, metrics_hash = (_cnlcu_workflow__sha(run_dir / 'last.pt'), _cnlcu_workflow__sha(run_dir / 'metrics.jsonl'))
            run_experiment(config_factory(3, dataset), resume=run_dir / 'last.pt')
            self.assertEqual(_cnlcu_workflow__sha(run_dir / 'last.pt'), checkpoint_hash)
            self.assertEqual(_cnlcu_workflow__sha(run_dir / 'metrics.jsonl'), metrics_hash)
            final = json.loads((run_dir / 'final_metrics.json').read_text())
            self.assertIn('test_mean_peer_accuracy', final)

    def test_cifar10_fresh_resume_and_completed_noop(self):
        self._run_case('cifar10')

    def test_cifar100_lightweight_workflow(self):
        self._run_case('cifar100')

    def test_hard_cifar10_fresh_resume_and_completed_noop(self):
        self._run_case('cifar10', 'hard')

    def test_hard_cifar100_lightweight_workflow(self):
        self._run_case('cifar100', 'hard')

    def test_resume_rejects_method_and_history_configuration_drift(self):
        train, test = (_cnlcu_workflow__cifar(40, 'train'), _cnlcu_workflow__cifar(20, 'test'))
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=lambda _root, split: train if split == 'train' else test):
            run_dir = run_experiment(_cnlcu_workflow__config(1), Path(directory) / 'run')
            changed = _cnlcu_workflow__config(2)
            changed['cnlcu']['history']['window_size'] = 3
            with self.assertRaisesRegex(ValueError, 'CNLCU settings'):
                run_experiment(changed, resume=run_dir / 'last.pt')

    def test_lazy_cli_dispatch(self):
        with patch('lnl_toolbox.training.cnlcu_experiment.run_cnlcu_experiment', return_value=Path('cnlcu-run')) as run:
            self.assertEqual(run_experiment(_cnlcu_workflow__config()), Path('cnlcu-run'))
            run.assert_called_once()

    def test_soft_hard_resume_and_hard_detector_drift_are_rejected(self):
        train, test = (_cnlcu_workflow__cifar(40, 'train'), _cnlcu_workflow__cifar(20, 'test'))
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=lambda _root, split: train if split == 'train' else test):
            soft_dir = run_experiment(_cnlcu_workflow__config(1), Path(directory) / 'soft')
            with self.assertRaisesRegex(ValueError, 'CNLCU settings'):
                run_experiment(_cnlcu_workflow__hard_config(2), resume=soft_dir / 'last.pt')
            hard_dir = run_experiment(_cnlcu_workflow__hard_config(1), Path(directory) / 'hard')
            with self.assertRaisesRegex(ValueError, 'CNLCU settings'):
                run_experiment(_cnlcu_workflow__config(2), resume=hard_dir / 'last.pt')
            drift = _cnlcu_workflow__hard_config(2)
            drift['cnlcu']['truncation']['contamination'] = 0.2
            with self.assertRaisesRegex(ValueError, 'CNLCU settings'):
                run_experiment(drift, resume=hard_dir / 'last.pt')
