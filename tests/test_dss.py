from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from lnl_toolbox.algorithms.dss import DSSObjective
from lnl_toolbox.algorithms.masked_risk import (
    candidate_masked_cross_entropy,
)
from lnl_toolbox.core import RunState
from lnl_toolbox.core.hyperparameters import resolve_parameter_sampling
from lnl_toolbox.core.objectives import ObjectiveResult
from lnl_toolbox.plugins.builtin.catalog import (
    build_builtin_objective_consumer,
)
from lnl_toolbox.selectors.dss import DSSSelectorState


class DSSRiskTest(unittest.TestCase):
    def test_candidate_masked_ce_matches_manual_denominator(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        targets = torch.tensor([0])
        excluded = torch.tensor([[False, True, False]])
        loss = candidate_masked_cross_entropy(logits, targets, excluded)
        expected = torch.logsumexp(logits[:, [0, 2]], dim=1) - logits[:, 0]
        torch.testing.assert_close(loss, expected)
        loss.sum().backward()
        self.assertEqual(float(logits.grad[0, 1]), 0.0)

    def test_target_class_cannot_be_excluded(self) -> None:
        with self.assertRaisesRegex(ValueError, "target class"):
            candidate_masked_cross_entropy(
                torch.zeros(1, 3),
                torch.tensor([1]),
                torch.tensor([[False, True, False]]),
            )


class DSSSelectorStateTest(unittest.TestCase):
    def test_selection_is_finalized_at_epoch_boundary(self) -> None:
        selector = DSSSelectorState(
            2, 2, 3, warmup_epochs=1, mda=False, ccs=False
        )
        indices = torch.tensor([0, 1])
        targets = torch.tensor([0, 1])
        selector.on_cycle_start(0)
        before, _ = selector.masks(indices, targets)
        selector.observe(
            indices,
            targets,
            torch.tensor([[0.9, 0.1], [0.8, 0.2]]),
            0,
        )
        during, _ = selector.masks(indices, targets)
        selector.on_cycle_end(0)
        after, _ = selector.masks(indices, targets)
        self.assertTrue(bool(before.all()))
        self.assertTrue(bool(during.all()))
        torch.testing.assert_close(
            after, torch.tensor([True, False])
        )

    def test_mda_uses_batch_ema_and_renormalization(self) -> None:
        selector = DSSSelectorState(
            2, 2, 2,
            warmup_epochs=2,
            prior_decay=0.5,
            mda=True,
            ccs=False,
        )
        selector.on_cycle_start(0)
        probabilities = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
        selector.observe(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            probabilities,
            0,
        )
        expected_marginal = torch.tensor([0.6, 0.4])
        adjusted = probabilities / (2 * expected_marginal)
        adjusted = adjusted / adjusted.sum(dim=1, keepdim=True)
        torch.testing.assert_close(selector.marginal, expected_marginal)
        torch.testing.assert_close(
            selector.current_prediction[:2], adjusted
        )

    def test_ccs_excludes_significantly_rising_non_target_class(self) -> None:
        selector = DSSSelectorState(
            1, 2, 4, warmup_epochs=1, mda=False, ccs=True
        )
        for epoch, probability in enumerate((0.1, 0.2, 0.3, 0.4)):
            selector.on_cycle_start(epoch)
            selector.observe(
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([[1.0 - probability, probability]]),
                epoch,
            )
            selector.on_cycle_end(epoch)
        _, excluded = selector.masks(
            torch.tensor([0]), torch.tensor([0])
        )
        torch.testing.assert_close(
            excluded, torch.tensor([[False, True]])
        )

    def test_state_roundtrip_restores_history_and_masks(self) -> None:
        selector = DSSSelectorState(
            2, 2, 3, warmup_epochs=1, mda=False
        )
        selector.on_cycle_start(0)
        selector.observe(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.tensor([[0.9, 0.1], [0.7, 0.3]]),
            0,
        )
        selector.on_cycle_end(0)
        state = selector.state_dict()
        restored = DSSSelectorState(
            2, 2, 3, warmup_epochs=1, mda=False
        )
        restored.load_state_dict(state)
        self.assertEqual(restored.current_epoch, 0)
        torch.testing.assert_close(restored.selected, selector.selected)
        torch.testing.assert_close(
            restored.history.values, selector.history.values
        )


class DSSObjectiveTest(unittest.TestCase):
    def test_objective_uses_batch_mean_but_reports_selected_mean(self) -> None:
        objective = DSSObjective(
            2, 2, 2, warmup_epochs=0, mda=False, ccs=False
        )
        objective.selector.selected[:] = torch.tensor([True, False])
        state = RunState(cycle=0)
        objective.on_cycle_start(state)
        logits = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0]], requires_grad=True
        )
        result = objective.compute(
            model=None,
            logits=logits,
            features=None,
            noisy_targets=torch.tensor([0, 1]),
            sample_indices=torch.tensor([0, 1]),
            base_loss=None,
            metadata={"epoch": 0},
        )
        self.assertIsInstance(result, ObjectiveResult)
        per_sample = torch.nn.functional.cross_entropy(
            logits, torch.tensor([0, 1]), reduction="none"
        )
        torch.testing.assert_close(result.objective, per_sample[0] / 2)
        torch.testing.assert_close(result.reporting_loss, per_sample[0])
        result.objective.backward()
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.0)

    def test_plugin_builds_dss_without_runner_branch(self) -> None:
        objective = build_builtin_objective_consumer({
            "name": "dss",
            "num_samples": 8,
            "num_classes": 2,
            "total_epochs": 3,
            "warmup_epochs": 1,
        })
        self.assertIsInstance(objective, DSSObjective)

    def test_reproduction_config_matches_official_single_run(self) -> None:
        path = Path(
            "configs/experiment/dss_cifar10_symmetric05_reproduction.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        resolved, record = resolve_parameter_sampling(config)
        self.assertEqual(resolved["seed"], 4)
        self.assertEqual(record.parameters["train_seed"], 4)
        self.assertEqual(record.parameters["setting"], {
            "noise": "symmetric",
            "rate": 0.5,
            "warmup_epochs": 30,
        })
        self.assertEqual(resolved["trainer"]["epochs"], 150)
        self.assertEqual(resolved["loader"]["batch_size"], 128)
        self.assertEqual(resolved["model"]["name"], "preact_resnet18")
        self.assertEqual(resolved["optimizer"], {
            "name": "sgd",
            "lr": 0.02,
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 0.001,
        })
        self.assertEqual(resolved["scheduler"]["milestones"], [80])
        objective = resolved["pipeline"]["objective_consumer"]
        self.assertEqual(objective["warmup_epochs"], 30)
        self.assertEqual(objective["prior_decay"], 0.99)
        self.assertEqual(objective["alpha"], 0.10)
        self.assertEqual(
            resolved["data"]["validation_split"]["strategy"],
            "classwise_legacy",
        )
        self.assertEqual(resolved["noise"]["validation_targets"], "noisy")


if __name__ == "__main__":
    unittest.main()
