import copy
import unittest
from pathlib import Path

import torch

from lnl_toolbox.algorithms.cnlcu import CNLCUAlgorithm, CNLCUConfig
from lnl_toolbox.algorithms.coteaching.selection import (
    determine_keep_count,
    stable_small_loss_mask,
)
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.losses.torch_losses import (
    CrossEntropyLoss,
    validate_per_sample_loss,
)


def _config(window_size=3, sigma=0.1, variant="soft"):
    result = {
        "method": "cnlcu", "noise": {"name": "symmetric", "rate": 0.5},
        "cnlcu": {"variant": variant, "model_count": 2, "noise_rate": 0.5,
            "initialization": {"peer_seed_offset": 1},
            "remember_schedule": {"name": "linear", "start": 1.0, "end": 0.5, "gradual_epochs": 1},
            "history": {"window_size": window_size, "storage_dtype": "float32"},
            "selection": {"count_rule": "floor", "tie_break": "stable_sample_index"}},
        "evaluation": {"selection_split": "validation", "primary": "mean_peer_accuracy", "ensemble": "mean_probabilities"},
    }
    if variant == "soft":
        result["cnlcu"]["uncertainty"] = {"sigma_squared": sigma}
    elif variant == "hard":
        result["cnlcu"].update({
            "hard_fidelity": "paper_formula_corrected_lof",
            "uncertainty": {
                "tau_min": 0.0001,
                "loss_upper_bound": {"mode": "fixed", "value": 20.0},
            },
            "truncation": {
                "method": "lof", "n_neighbors": 2,
                "contamination": 0.1, "minimum_observations": 3,
            },
        })
    return result


class _IndexedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))
    def forward(self, inputs):
        return self.logits[inputs.reshape(-1).long()]


def _algorithm(variant="soft"):
    a = _IndexedLogits([[5, -5], [5, -5], [-5, 5], [-5, 5]])
    b = _IndexedLogits([[-5, 5], [-5, 5], [5, -5], [5, -5]])
    result = CNLCUAlgorithm(
        model_a=a, model_b=b, optimizer_a=torch.optim.SGD(a.parameters(), lr=0.1),
        optimizer_b=torch.optim.SGD(b.parameters(), lr=0.1), scheduler_a=None, scheduler_b=None,
        loss=CrossEntropyLoss(), device=torch.device("cpu"),
        method_config=CNLCUConfig.from_mapping(_config(variant=variant)),
        canonical_global_indices=torch.tensor([40, 10, 30, 20]),
    )
    result.setup(ExperimentContext(Path.cwd()))
    result.on_cycle_start(RunState(cycle=1))
    return result


def _reference_step(algorithm, batch, state):
    """Execute the pre-telemetry CNLCU step contract for regression comparison."""

    payload = batch.payload
    inputs = payload["input"].to(algorithm.device)
    targets = payload["target"].to(algorithm.device)
    indices = torch.as_tensor(payload["index"], dtype=torch.long, device=algorithm.device)
    logits_a, logits_b = algorithm.model_a(inputs), algorithm.model_b(inputs)
    losses_a = validate_per_sample_loss(algorithm.loss(logits_a, targets), targets.numel())
    losses_b = validate_per_sample_loss(algorithm.loss(logits_b, targets), targets.numel())
    rows_a = algorithm.private_state.history_a.append(indices, losses_a.detach())
    rows_b = algorithm.private_state.history_b.append(indices, losses_b.detach())
    score_a, _ = algorithm._score(algorithm.private_state.history_a, rows_a)
    score_b, _ = algorithm._score(algorithm.private_state.history_b, rows_b)
    keep_count = determine_keep_count(
        int(targets.numel()), algorithm.method_config.rate_at(state.cycle)
    )
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
    return {
        "selected_a": indices[selected_a].detach().cpu(),
        "selected_b": indices[selected_b].detach().cpu(),
        "objective_a": objective_a.detach(),
        "objective_b": objective_b.detach(),
    }


class CNLCUAlgorithmTest(unittest.TestCase):
    def test_read_only_telemetry_preserves_selection_loss_and_parameter_updates(self):
        observed, reference = _algorithm(), _algorithm()
        observed_state, reference_state = RunState(cycle=1), RunState(cycle=1)
        batch = Batch({
            "input": torch.arange(4),
            "target": torch.zeros(4, dtype=torch.long),
            "index": torch.tensor([10, 20, 30, 40]),
        })
        for cycle in (1, 2):
            if cycle > 1:
                observed.on_cycle_start(RunState(cycle=cycle))
                reference.on_cycle_start(RunState(cycle=cycle))
                observed_state.cycle = reference_state.cycle = cycle
            result = observed.step(batch, observed_state)
            expected = _reference_step(reference, batch, reference_state)
            self.assertEqual(
                result.metadata["selected_by_a_indices"].tolist(),
                expected["selected_a"].tolist(),
            )
            self.assertEqual(
                result.metadata["selected_by_b_indices"].tolist(),
                expected["selected_b"].tolist(),
            )
            torch.testing.assert_close(
                torch.tensor(result.metrics["loss_a_on_selected_by_b"]),
                expected["objective_a"], rtol=0.0, atol=0.0,
            )
            torch.testing.assert_close(
                torch.tensor(result.metrics["loss_b_on_selected_by_a"]),
                expected["objective_b"], rtol=0.0, atol=0.0,
            )
            for observed_model, reference_model in (
                (observed.model_a, reference.model_a),
                (observed.model_b, reference.model_b),
            ):
                for left, right in zip(
                    observed_model.parameters(), reference_model.parameters(), strict=True
                ):
                    torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
            for key in (
                "gradient_norm_a", "gradient_norm_b",
                "parameter_norm_a", "parameter_norm_b",
            ):
                self.assertTrue(torch.isfinite(torch.tensor(result.metrics[key])))

    def test_optimizer_parameters_exactly_match_and_do_not_overlap(self):
        algorithm = _algorithm()
        model_a = {id(parameter) for parameter in algorithm.model_a.parameters()}
        model_b = {id(parameter) for parameter in algorithm.model_b.parameters()}
        optimizer_a = {
            id(parameter)
            for group in algorithm.optimizer_a.param_groups
            for parameter in group["params"]
        }
        optimizer_b = {
            id(parameter)
            for group in algorithm.optimizer_b.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(optimizer_a, model_a)
        self.assertEqual(optimizer_b, model_b)
        self.assertFalse(optimizer_a & optimizer_b)

    def test_misbound_or_overlapping_optimizer_parameters_fail(self):
        model_a = _IndexedLogits([[1, 0]])
        model_b = _IndexedLogits([[0, 1]])
        common = dict(
            scheduler_a=None,
            scheduler_b=None,
            loss=CrossEntropyLoss(),
            device=torch.device("cpu"),
            method_config=CNLCUConfig.from_mapping(_config()),
            canonical_global_indices=torch.tensor([10]),
        )
        with self.assertRaisesRegex(ValueError, "optimizer a parameters"):
            CNLCUAlgorithm(
                model_a=model_a,
                model_b=model_b,
                optimizer_a=torch.optim.SGD(model_b.parameters(), lr=0.1),
                optimizer_b=torch.optim.SGD(model_b.parameters(), lr=0.1),
                **common,
            )

        shared = torch.nn.Parameter(torch.tensor([[1.0, 0.0]]))
        model_a.logits = shared
        model_b.logits = shared
        with self.assertRaisesRegex(ValueError, "model parameter sets must not overlap"):
            CNLCUAlgorithm(
                model_a=model_a,
                model_b=model_b,
                optimizer_a=torch.optim.SGD(model_a.parameters(), lr=0.1),
                optimizer_b=torch.optim.SGD(model_b.parameters(), lr=0.1),
                **common,
            )

    def test_peer_cross_update_and_peer_specific_counts(self):
        algorithm = _algorithm()
        before_a, before_b = algorithm.model_a.logits.detach().clone(), algorithm.model_b.logits.detach().clone()
        state = RunState(cycle=1)
        result = algorithm.step(Batch({"input": torch.arange(4), "target": torch.zeros(4, dtype=torch.long),
                                       "index": torch.tensor([10, 20, 30, 40])}), state)
        self.assertEqual(result.metadata["selected_by_a_indices"].tolist(), [10, 20])
        self.assertEqual(result.metadata["selected_by_b_indices"].tolist(), [30, 40])
        torch.testing.assert_close(algorithm.model_a.logits[:2], before_a[:2])
        self.assertFalse(torch.equal(algorithm.model_a.logits[2:], before_a[2:]))
        self.assertFalse(torch.equal(algorithm.model_b.logits[:2], before_b[:2]))
        torch.testing.assert_close(algorithm.model_b.logits[2:], before_b[2:])
        rows_a = algorithm.private_state.history_a.resolve(torch.tensor([10, 20, 30, 40]))
        rows_b = algorithm.private_state.history_b.resolve(torch.tensor([10, 20, 30, 40]))
        self.assertEqual(algorithm.private_state.history_a.selected_count[rows_a].tolist(), [1, 1, 0, 0])
        self.assertEqual(algorithm.private_state.history_b.selected_count[rows_b].tolist(), [0, 0, 1, 1])

    def test_hard_peer_cross_update_uses_peer_selections(self):
        algorithm = _algorithm("hard")
        before_a = algorithm.model_a.logits.detach().clone()
        before_b = algorithm.model_b.logits.detach().clone()
        result = algorithm.step(Batch({
            "input": torch.arange(4),
            "target": torch.zeros(4, dtype=torch.long),
            "index": torch.tensor([10, 20, 30, 40]),
        }), RunState(cycle=1))
        self.assertEqual(result.metadata["selected_by_a_indices"].tolist(), [10, 20])
        self.assertEqual(result.metadata["selected_by_b_indices"].tolist(), [30, 40])
        torch.testing.assert_close(algorithm.model_a.logits[:2], before_a[:2])
        self.assertFalse(torch.equal(algorithm.model_a.logits[2:], before_a[2:]))
        self.assertFalse(torch.equal(algorithm.model_b.logits[:2], before_b[:2]))
        torch.testing.assert_close(algorithm.model_b.logits[2:], before_b[2:])
        self.assertIn("hard_confidence_bonus_a", result.metrics)
        self.assertIn("outlier_ratio_b", result.metrics)

    def test_current_loss_is_appended_before_score_and_clean_oracle_is_ignored(self):
        first, second = _algorithm(), _algorithm()
        payload = {"input": torch.arange(4), "target": torch.zeros(4, dtype=torch.long),
                   "index": torch.tensor([10, 20, 30, 40])}
        left = first.step(Batch(payload), RunState(cycle=1))
        right = second.step(Batch({**payload, "clean_target": torch.ones(4, dtype=torch.long)}), RunState(cycle=1))
        self.assertEqual(left.metadata["selected_by_a_indices"].tolist(), right.metadata["selected_by_a_indices"].tolist())
        self.assertEqual(left.metrics["history_length_a"], 1.0)
        for a, b in zip(first.model_a.parameters(), second.model_a.parameters()):
            torch.testing.assert_close(a, b)

    def test_uncertainty_count_can_change_current_loss_ranking(self):
        algorithm = _algorithm()
        algorithm.model_a.logits.data.copy_(torch.tensor(
            [[0.4, 0.0], [0.4, 0.0], [0.2, 0.0], [0.2, 0.0]]
        ))
        rows = algorithm.private_state.history_a.resolve(torch.tensor([10, 20, 30, 40]))
        algorithm.private_state.history_a.selected_count[rows] = torch.tensor([100, 100, 0, 0])
        result = algorithm.step(Batch({"input": torch.arange(4), "target": torch.zeros(4, dtype=torch.long),
            "index": torch.tensor([10, 20, 30, 40])}), RunState(cycle=1))
        # A's current losses favor 10/20, but the uncertainty bonus gives 30/40 a trial.
        self.assertEqual(result.metadata["selected_by_a_indices"].tolist(), [30, 40])

    def test_checkpoint_roundtrip_and_wrong_identity_fail(self):
        algorithm = _algorithm()
        algorithm.step(Batch({"input": torch.arange(4), "target": torch.zeros(4, dtype=torch.long),
            "index": torch.tensor([10, 20, 30, 40])}), RunState(cycle=1))
        saved = copy.deepcopy(algorithm.state_dict())
        restored = _algorithm(); restored.load_state_dict(saved)
        self.assertEqual(restored.private_state.history_a.selected_count.tolist(),
                         algorithm.private_state.history_a.selected_count.tolist())
        wrong = copy.deepcopy(saved); wrong["method_identity"] = "coteaching"
        with self.assertRaisesRegex(ValueError, "identity"):
            restored.load_state_dict(wrong)
        swapped = copy.deepcopy(saved)
        private = swapped["cnlcu_state"]
        private["history_a"], private["history_b"] = private["history_b"], private["history_a"]
        with self.assertRaisesRegex(ValueError, "identity"):
            _algorithm().load_state_dict(swapped)
        schedule_drift = copy.deepcopy(saved)
        schedule_drift["remember_schedule"]["gradual_epochs"] += 1
        with self.assertRaisesRegex(ValueError, "configuration"):
            _algorithm().load_state_dict(schedule_drift)
        count_scope_drift = copy.deepcopy(saved)
        count_scope_drift["selected_count_scope"] = "global"
        with self.assertRaisesRegex(ValueError, "configuration"):
            _algorithm().load_state_dict(count_scope_drift)

    def test_configuration_rejects_unknown_variant_and_single_model_composition(self):
        values = _config(); values["cnlcu"]["variant"] = "unknown"
        with self.assertRaisesRegex(ValueError, "soft or hard"):
            CNLCUConfig.from_mapping(values)
        for key in ("selector", "parameter_update", "weight_provider", "objective_consumer", "dss"):
            values = _config(); values[key] = {"name": "anything"}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                CNLCUConfig.from_mapping(values)

    def test_hard_configuration_and_checkpoint_identity_are_strict(self):
        config = CNLCUConfig.from_mapping(_config(variant="hard"))
        self.assertEqual(config.hard_fidelity, "paper_formula_corrected_lof")
        for field, value in (
            ("hard_fidelity", "released_code"),
            ("truncation.method", "knn"),
            ("truncation.minimum_observations", 2),
            ("uncertainty.loss_upper_bound.mode", "percentile"),
        ):
            values = _config(variant="hard")
            owner, key = field.split(".", 1) if "." in field else ("cnlcu", field)
            if owner == "cnlcu":
                values["cnlcu"][key] = value
            elif "." in key:
                first, second = key.split(".")
                values["cnlcu"][owner][first][second] = value
            else:
                values["cnlcu"][owner][key] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                CNLCUConfig.from_mapping(values)

        algorithm = _algorithm("hard")
        saved = copy.deepcopy(algorithm.state_dict())
        self.assertEqual(saved["hard_identity"]["truncation"]["method"], "lof")
        changed = copy.deepcopy(saved)
        changed["hard_identity"]["tau_min"] = 0.5
        with self.assertRaisesRegex(ValueError, "CNLCU-H checkpoint"):
            _algorithm("hard").load_state_dict(changed)
        with self.assertRaisesRegex(ValueError, "configuration"):
            _algorithm("soft").load_state_dict(saved)

    def test_hard_fixed_loss_bound_is_enforced_without_clipping(self):
        algorithm = _algorithm("hard")
        algorithm.method_config = CNLCUConfig.from_mapping(_config(variant="hard"))
        # The helper bound is 20; force a clearly larger observed CE.
        algorithm.model_a.logits.data[0] = torch.tensor([-100.0, 100.0])
        with self.assertRaisesRegex(ValueError, "exceeded fixed loss_upper_bound"):
            algorithm.step(Batch({
                "input": torch.arange(4),
                "target": torch.zeros(4, dtype=torch.long),
                "index": torch.tensor([10, 20, 30, 40]),
            }), RunState(cycle=1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_step_keeps_history_on_cpu(self):
        model_a = _IndexedLogits([[5, -5], [5, -5], [-5, 5], [-5, 5]])
        model_b = _IndexedLogits([[-5, 5], [-5, 5], [5, -5], [5, -5]])
        algorithm = CNLCUAlgorithm(
            model_a=model_a,
            model_b=model_b,
            optimizer_a=torch.optim.SGD(model_a.parameters(), lr=0.1),
            optimizer_b=torch.optim.SGD(model_b.parameters(), lr=0.1),
            scheduler_a=None,
            scheduler_b=None,
            loss=CrossEntropyLoss(),
            device=torch.device("cuda"),
            method_config=CNLCUConfig.from_mapping(_config()),
            canonical_global_indices=torch.tensor([40, 10, 30, 20]),
        )
        algorithm.setup(ExperimentContext(Path.cwd()))
        algorithm.on_cycle_start(RunState(cycle=1))
        algorithm.step(
            Batch(
                {
                    "input": torch.arange(4),
                    "target": torch.zeros(4, dtype=torch.long),
                    "index": torch.tensor([10, 20, 30, 40]),
                }
            ),
            RunState(cycle=1),
        )
        self.assertEqual(algorithm.private_state.history_a.values.device.type, "cpu")
        self.assertEqual(algorithm.private_state.history_b.values.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_hard_cuda_step_keeps_lof_history_on_cpu(self):
        algorithm = _algorithm("hard")
        algorithm.device = torch.device("cuda")
        algorithm.setup(ExperimentContext(Path.cwd()))
        algorithm.step(Batch({
            "input": torch.arange(4),
            "target": torch.zeros(4, dtype=torch.long),
            "index": torch.tensor([10, 20, 30, 40]),
        }), RunState(cycle=1))
        self.assertEqual(algorithm.private_state.history_a.values.device.type, "cpu")
        self.assertEqual(algorithm.private_state.history_b.values.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
