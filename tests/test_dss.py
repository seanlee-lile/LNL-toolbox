from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from lnl_toolbox.algorithms.dss import DSSObjective
from lnl_toolbox.algorithms.masked_risk import candidate_masked_cross_entropy
from lnl_toolbox.core import RunState
from lnl_toolbox.core.objectives import ObjectiveResult
from lnl_toolbox.plugins.builtin.catalog import (
    build_builtin_objective_consumer,
)
from lnl_toolbox.selectors.dss import DSSSelectorState
from lnl_toolbox.training.experiment import _resolve_dss_epoch_contract
from lnl_toolbox.training.pipeline import StandardNoisyERMPipeline


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

    def test_invalid_mask_contract_is_rejected(self) -> None:
        logits = torch.zeros(1, 3)
        targets = torch.tensor([1])
        with self.assertRaisesRegex(ValueError, "target class"):
            candidate_masked_cross_entropy(
                logits, targets, torch.tensor([[False, True, False]])
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            candidate_masked_cross_entropy(
                logits, targets, torch.zeros(1, 2, dtype=torch.bool)
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
        torch.testing.assert_close(after, torch.tensor([True, False]))

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
        torch.testing.assert_close(selector.current_prediction[:2], adjusted)

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
        _, excluded = selector.masks(torch.tensor([0]), torch.tensor([0]))
        torch.testing.assert_close(excluded, torch.tensor([[False, True]]))

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
        restored = DSSSelectorState(
            2, 2, 3, warmup_epochs=1, mda=False
        )
        restored.load_state_dict(selector.state_dict())
        self.assertEqual(restored.current_epoch, 0)
        torch.testing.assert_close(restored.selected, selector.selected)
        torch.testing.assert_close(
            restored.history.values, selector.history.values
        )

    def test_stable_index_and_epoch_contracts_fail_explicitly(self) -> None:
        selector = DSSSelectorState(3, 2, 2, warmup_epochs=1)
        selector.on_cycle_start(0)
        with self.assertRaisesRegex(ValueError, "unique"):
            selector.masks(torch.tensor([0, 0]), torch.tensor([0, 0]))
        with self.assertRaisesRegex(ValueError, "integer"):
            selector.masks(torch.tensor([0.0]), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "prior epoch"):
            selector.current_epoch = 1
            selector.observe(
                torch.tensor([1]),
                torch.tensor([0]),
                torch.tensor([[0.6, 0.4]]),
                1,
            )


class DSSObjectiveTest(unittest.TestCase):
    def test_objective_uses_batch_mean_but_reports_selected_mean(self) -> None:
        objective = DSSObjective(
            2, 2, 2, warmup_epochs=0, mda=False, ccs=False
        )
        objective.selector.selected[:] = torch.tensor([True, False])
        objective.on_cycle_start(RunState(cycle=0))
        logits = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0]], requires_grad=True
        )
        result = objective.compute(
            logits=logits,
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

    def test_plugin_builds_dss_and_pipeline_rejects_composition(self) -> None:
        objective_config = {
            "name": "dss",
            "num_samples": 8,
            "num_classes": 2,
            "total_epochs": 3,
            "warmup_epochs": 1,
        }
        objective = build_builtin_objective_consumer(objective_config)
        self.assertIsInstance(objective, DSSObjective)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            StandardNoisyERMPipeline.from_config({
                "objective_consumer": objective_config,
                "weight_provider": {"name": "missing"},
            })

    def test_trainer_epochs_are_the_only_resolved_horizon(self) -> None:
        config = {
            "pipeline": {
                "objective_consumer": {
                    "name": "dss",
                    "num_samples": 8,
                    "num_classes": 2,
                }
            }
        }
        _resolve_dss_epoch_contract(config, 4)
        self.assertEqual(
            config["pipeline"]["objective_consumer"]["total_epochs"], 4
        )
        conflicting = {
            "pipeline": {
                "objective_consumer": {
                    "name": "dss",
                    "total_epochs": 3,
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "trainer.epochs"):
            _resolve_dss_epoch_contract(conflicting, 4)
        for section, value in (
            ("loss", {"name": "gce"}),
            ("selector", {"name": "small_loss", "keep_rate": 0.5}),
            ("parameter_update", {"name": "cdr"}),
        ):
            invalid = {
                section: value,
                "pipeline": {
                    "objective_consumer": {"name": "dss"}
                },
            }
            with self.subTest(section=section), self.assertRaisesRegex(
                ValueError, "DSS requires"
            ):
                _resolve_dss_epoch_contract(invalid, 4)

    def test_pipeline_component_state_roundtrip_is_strict(self) -> None:
        config = {
            "objective_consumer": {
                "name": "dss",
                "num_samples": 2,
                "num_classes": 2,
                "total_epochs": 2,
                "warmup_epochs": 1,
                "mda": False,
                "ccs": False,
            }
        }
        pipeline = StandardNoisyERMPipeline.from_config(config)
        objective = pipeline.objective_consumer
        objective.on_cycle_start(RunState(cycle=0))
        objective.compute(
            logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            noisy_targets=torch.tensor([0, 1]),
            sample_indices=torch.tensor([0, 1]),
            base_loss=None,
            metadata={"epoch": 0},
        )
        objective.on_cycle_end(RunState(cycle=0))
        states = pipeline.component_state_dict()
        restored = StandardNoisyERMPipeline.from_config(config)
        restored.load_component_states(states)
        torch.testing.assert_close(
            restored.objective_consumer.selector.history.values,
            objective.selector.history.values,
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            restored.load_component_states({})

    def test_configs_do_not_duplicate_total_epochs(self) -> None:
        for filename, expected_epochs in (
            ("dss_cifar10_symmetric05_smoke.yaml", 2),
            ("dss_cifar10_symmetric05_reproduction.yaml", 150),
        ):
            config = yaml.safe_load(
                (
                    Path("configs/experiment") / filename
                ).read_text(encoding="utf-8")
            )
            objective = config["pipeline"]["objective_consumer"]
            self.assertNotIn("total_epochs", objective)
            _resolve_dss_epoch_contract(
                config, int(config["trainer"]["epochs"])
            )
            self.assertEqual(objective.get("total_epochs"), None)
            self.assertEqual(
                config["pipeline"]["objective_consumer"]["total_epochs"],
                expected_epochs,
            )
            self.assertEqual(config["evaluation"]["selection_split"], "validation")


if __name__ == "__main__":
    unittest.main()
