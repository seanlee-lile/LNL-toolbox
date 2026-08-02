import copy
import math
import unittest
from pathlib import Path

import torch

from lnl_toolbox.algorithms.coteaching import (
    CoTeachingAlgorithm,
    CoTeachingConfig,
    determine_keep_count,
    stable_small_loss_mask,
)
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.training.coteaching_experiment import _build_peer_models


def _method_config(*, noise_rate=0.5, gradual_epochs=1):
    return {
        "method": "coteaching",
        "noise": {"name": "symmetric", "rate": noise_rate},
        "coteaching": {
            "model_count": 2,
            "noise_rate": noise_rate,
            "initialization": {"peer_seed_offset": 1},
            "remember_schedule": {
                "name": "linear",
                "start": 1.0,
                "end": 1.0 - noise_rate,
                "gradual_epochs": gradual_epochs,
            },
            "selection": {
                "count_rule": "floor",
                "tie_break": "stable_sample_index",
            },
        },
        "evaluation": {
            "selection_split": "validation",
            "primary": "mean_peer_accuracy",
            "ensemble": "mean_probabilities",
        },
    }


class _IndexedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs):
        return self.logits[inputs.reshape(-1).long()]


class _TracingIndexedLogits(_IndexedLogits):
    def __init__(self, logits, name, events):
        super().__init__(logits)
        self.name = name
        self.events = events

    def forward(self, inputs):
        self.events.append(f"forward_{self.name}")
        return super().forward(inputs)


class _TracingSGD(torch.optim.SGD):
    def __init__(self, params, name, events):
        super().__init__(params, lr=0.1)
        self.name = name
        self.events = events

    def step(self, closure=None):
        self.events.append(f"step_{self.name}")
        return super().step(closure)


def _cross_update_algorithm():
    model_a = _IndexedLogits([[5, -5], [5, -5], [-5, 5], [-5, 5]])
    model_b = _IndexedLogits([[-5, 5], [-5, 5], [5, -5], [5, -5]])
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    algorithm = CoTeachingAlgorithm(
        model_a=model_a,
        model_b=model_b,
        optimizer_a=optimizer_a,
        optimizer_b=optimizer_b,
        scheduler_a=None,
        scheduler_b=None,
        loss=CrossEntropyLoss(),
        device=torch.device("cpu"),
        method_config=CoTeachingConfig.from_mapping(_method_config()),
    )
    algorithm.setup(ExperimentContext(Path.cwd()))
    algorithm.on_cycle_start(RunState(cycle=1))
    return algorithm


class CoTeachingSelectionTest(unittest.TestCase):
    def test_zero_based_schedule(self):
        config = CoTeachingConfig.from_mapping(
            _method_config(noise_rate=0.4, gradual_epochs=10)
        )
        expected = {0: 1.0, 1: 0.96, 5: 0.8, 10: 0.6, 14: 0.6}
        for epoch, rate in expected.items():
            with self.subTest(epoch=epoch):
                self.assertAlmostEqual(config.rate_at(epoch), rate)

    def test_floor_count_and_minimum_one(self):
        self.assertEqual(determine_keep_count(7, 0.5), 3)
        self.assertEqual(determine_keep_count(2, 0.01), 1)

    def test_ties_use_stable_global_index_not_batch_position(self):
        losses = torch.tensor([0.2, 0.2, 0.1, 0.2])
        indices = torch.tensor([30, 10, 40, 20])
        mask = stable_small_loss_mask(losses, indices, 3)
        self.assertEqual(set(indices[mask].tolist()), {10, 20, 40})
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = stable_small_loss_mask(
            losses[permutation], indices[permutation], 3
        )
        self.assertEqual(set(indices[permutation][permuted].tolist()), {10, 20, 40})

    def test_invalid_selection_inputs_fail(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                stable_small_loss_mask(
                    torch.tensor([0.1, value]), torch.tensor([0, 1]), 1
                )
        with self.assertRaisesRegex(ValueError, "unique"):
            stable_small_loss_mask(
                torch.tensor([0.1, 0.2]), torch.tensor([1, 1]), 1
            )


class CoTeachingAlgorithmTest(unittest.TestCase):
    def test_peer_cross_update_uses_other_models_selection(self):
        algorithm = _cross_update_algorithm()
        before_a = algorithm.model_a.logits.detach().clone()
        before_b = algorithm.model_b.logits.detach().clone()
        state = RunState(cycle=1)
        result = algorithm.step(
            Batch({
                "input": torch.arange(4),
                "target": torch.zeros(4, dtype=torch.long),
                "index": torch.arange(4),
            }),
            state,
        )

        self.assertEqual(result.metadata["selected_by_a_indices"].tolist(), [0, 1])
        self.assertEqual(result.metadata["selected_by_b_indices"].tolist(), [2, 3])
        torch.testing.assert_close(algorithm.model_a.logits[:2], before_a[:2])
        self.assertFalse(torch.equal(algorithm.model_a.logits[2:], before_a[2:]))
        self.assertFalse(torch.equal(algorithm.model_b.logits[:2], before_b[:2]))
        torch.testing.assert_close(algorithm.model_b.logits[2:], before_b[2:])
        self.assertEqual(state.step, 1)
        self.assertEqual(algorithm.private_state.optimizer_steps_a, 1)
        self.assertEqual(algorithm.private_state.optimizer_steps_b, 1)
        self.assertIsNot(algorithm.optimizer_a, algorithm.optimizer_b)

    def test_both_forwards_happen_before_either_optimizer_step(self):
        events = []
        logits_a = [[5, -5], [5, -5], [-5, 5], [-5, 5]]
        logits_b = [[-5, 5], [-5, 5], [5, -5], [5, -5]]
        model_a = _TracingIndexedLogits(logits_a, "a", events)
        model_b = _TracingIndexedLogits(logits_b, "b", events)
        algorithm = CoTeachingAlgorithm(
            model_a=model_a,
            model_b=model_b,
            optimizer_a=_TracingSGD(model_a.parameters(), "a", events),
            optimizer_b=_TracingSGD(model_b.parameters(), "b", events),
            scheduler_a=None,
            scheduler_b=None,
            loss=CrossEntropyLoss(),
            device=torch.device("cpu"),
            method_config=CoTeachingConfig.from_mapping(_method_config()),
        )
        algorithm.setup(ExperimentContext(Path.cwd()))
        algorithm.step(
            Batch({
                "input": torch.arange(4),
                "target": torch.zeros(4, dtype=torch.long),
                "index": torch.arange(4),
            }),
            RunState(cycle=1),
        )
        self.assertEqual(events, ["forward_a", "forward_b", "step_a", "step_b"])

    def test_clean_label_oracle_field_cannot_change_selection_or_update(self):
        baseline = _cross_update_algorithm()
        with_oracle = _cross_update_algorithm()
        payload = {
            "input": torch.arange(4),
            "target": torch.zeros(4, dtype=torch.long),
            "index": torch.arange(4),
        }
        first = baseline.step(Batch(payload), RunState(cycle=1))
        second = with_oracle.step(
            Batch({**payload, "clean_target": torch.ones(4, dtype=torch.long)}),
            RunState(cycle=1),
        )
        self.assertEqual(first.metadata["selected_by_a_indices"].tolist(), second.metadata["selected_by_a_indices"].tolist())
        self.assertEqual(first.metadata["selected_by_b_indices"].tolist(), second.metadata["selected_by_b_indices"].tolist())
        for left, right in zip(baseline.model_a.parameters(), with_oracle.model_a.parameters()):
            torch.testing.assert_close(left, right)
        for left, right in zip(baseline.model_b.parameters(), with_oracle.model_b.parameters()):
            torch.testing.assert_close(left, right)

    def test_schedulers_are_distinct_and_roundtrip(self):
        algorithm = _cross_update_algorithm()
        algorithm.scheduler_a = torch.optim.lr_scheduler.StepLR(
            algorithm.optimizer_a, step_size=1, gamma=0.5
        )
        algorithm.scheduler_b = torch.optim.lr_scheduler.StepLR(
            algorithm.optimizer_b, step_size=1, gamma=0.5
        )
        self.assertIsNot(algorithm.scheduler_a, algorithm.scheduler_b)
        algorithm.optimizer_a.step()
        algorithm.optimizer_b.step()
        algorithm.step_schedulers()
        saved = copy.deepcopy(algorithm.state_dict())

        restored = _cross_update_algorithm()
        restored.scheduler_a = torch.optim.lr_scheduler.StepLR(
            restored.optimizer_a, step_size=1, gamma=0.5
        )
        restored.scheduler_b = torch.optim.lr_scheduler.StepLR(
            restored.optimizer_b, step_size=1, gamma=0.5
        )
        restored.load_state_dict(saved)
        self.assertEqual(restored.scheduler_a.state_dict(), algorithm.scheduler_a.state_dict())
        self.assertEqual(restored.scheduler_b.state_dict(), algorithm.scheduler_b.state_dict())

    def test_peer_initialization_is_distinct_and_reproducible(self):
        model_config = {"name": "tiny_cnn", "width": 4}
        first = _build_peer_models(model_config, 10, 13, 1)
        second = _build_peer_models(model_config, 10, 13, 1)
        self.assertEqual(tuple(first[0].state_dict()), tuple(first[1].state_dict()))
        self.assertTrue(any(
            not torch.equal(left, right)
            for left, right in zip(first[0].parameters(), first[1].parameters())
        ))
        for left_peer, right_peer in zip(first, second):
            for left, right in zip(left_peer.state_dict().values(), right_peer.state_dict().values()):
                torch.testing.assert_close(left, right)

    def test_checkpoint_roundtrip_keeps_fixed_peer_identity_and_optimizer_state(self):
        algorithm = _cross_update_algorithm()
        algorithm.step(
            Batch({
                "input": torch.arange(4),
                "target": torch.zeros(4, dtype=torch.long),
                "index": torch.arange(4),
            }),
            RunState(cycle=1),
        )
        saved = copy.deepcopy(algorithm.state_dict())
        restored = _cross_update_algorithm()
        restored.load_state_dict(saved)
        self.assertEqual(restored.private_state.optimizer_steps_a, 1)
        self.assertEqual(restored.private_state.optimizer_steps_b, 1)
        for original, loaded in zip(algorithm.model_a.parameters(), restored.model_a.parameters()):
            torch.testing.assert_close(original, loaded)
        for original, loaded in zip(algorithm.model_b.parameters(), restored.model_b.parameters()):
            torch.testing.assert_close(original, loaded)
        swapped = copy.deepcopy(saved)
        swapped["peer_identity"] = ("b", "a")
        with self.assertRaisesRegex(ValueError, "peer identity"):
            restored.load_state_dict(swapped)

    def test_configuration_rejects_single_model_composition_and_invalid_values(self):
        for key in ("selector", "parameter_update", "weight_provider", "target_provider", "objective_consumer"):
            values = _method_config()
            values[key] = {"name": "anything"}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                CoTeachingConfig.from_mapping(values)
        values = _method_config()
        values["coteaching"]["model_count"] = 1
        with self.assertRaisesRegex(ValueError, "exactly two"):
            CoTeachingConfig.from_mapping(values)
        values = _method_config()
        values["coteaching"]["noise_rate"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            CoTeachingConfig.from_mapping(values)


if __name__ == "__main__":
    unittest.main()
