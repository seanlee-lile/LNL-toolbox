import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from lnl_toolbox.algorithms.t_revision import t_revision_reweight_objective
from lnl_toolbox.catalog import (
    load_recipe_config,
    paper_by_id,
    recipe_by_id,
    validate_config,
)
from tests.test_t_revision_workflow import _algorithm


class TRevisionReadinessTest(unittest.TestCase):
    def test_short_recipe_is_full_data_sym20_resnet18(self) -> None:
        recipe = recipe_by_id("cifar10-t-revision-sym20-short")
        self.assertEqual(recipe.profile, "reproduction")
        self.assertEqual(recipe.runner, "t_revision")
        self.assertEqual(recipe.configuration_fidelity, "engineering")
        config = load_recipe_config(recipe)
        self.assertEqual(validate_config(config).name, "t_revision")
        self.assertEqual(config["data"]["name"], "cifar10")
        for limit in (
            "max_train_samples",
            "max_validation_samples",
            "max_test_samples",
        ):
            self.assertNotIn(limit, config["data"])
        self.assertEqual(config["data"]["validation_size"], 5000)
        self.assertIs(config["data"]["augment"], True)
        self.assertEqual(config["noise"]["rate"], 0.2)
        self.assertEqual(config["noise"]["sampling"], "transition")
        self.assertEqual(config["noise"]["validation_targets"], "noisy")
        self.assertEqual(config["loader"]["batch_size"], 128)
        self.assertEqual(
            config["t_revision"]["stage1"]["model"]["name"], "resnet18"
        )
        self.assertEqual(config["t_revision"]["stage1"]["epochs"], 15)
        self.assertEqual(
            config["t_revision"]["classifier_initialization"]["epochs"], 15
        )
        self.assertEqual(config["t_revision"]["revision"]["epochs"], 20)
        paper = paper_by_id("t-revision")
        self.assertIn(recipe.id, {item.recipe_id for item in paper.configs})

    def test_objective_exposes_detached_diagnostics_without_changing_loss(self) -> None:
        logits = torch.tensor([[1.2, -0.2], [0.1, 0.7]], requires_grad=True)
        targets = torch.tensor([0, 1])
        transition = torch.tensor([[0.8, 0.2], [0.1, 0.9]], requires_grad=True)
        result = t_revision_reweight_objective(
            logits, targets, transition, denominator_floor=1.0e-12
        )
        clean = torch.softmax(logits, dim=1)
        denominator = (clean @ transition).gather(1, targets[:, None]).squeeze(1)
        expected_weights = clean.gather(1, targets[:, None]).squeeze(1) / denominator
        torch.testing.assert_close(result.sample_weights, expected_weights.detach())
        torch.testing.assert_close(result.sample_denominators, denominator.detach())
        self.assertFalse(result.sample_weights.requires_grad)
        self.assertFalse(result.sample_denominators.requires_grad)
        result.objective.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(transition.grad)

    def test_stage2a_detaches_ratio_without_changing_equation_three(self) -> None:
        logits = torch.tensor(
            [[1.2, -0.2], [0.1, 0.7]], dtype=torch.float64, requires_grad=True
        )
        targets = torch.tensor([0, 1])
        transition = torch.tensor(
            [[0.8, 0.2], [0.1, 0.9]], dtype=torch.float64, requires_grad=True
        )
        result = t_revision_reweight_objective(
            logits,
            targets,
            transition,
            denominator_floor=1.0e-12,
            detach_ratio=True,
        )
        clean = torch.softmax(logits, dim=1)
        numerator = clean.gather(1, targets[:, None]).squeeze(1)
        denominator = (clean @ transition).gather(
            1, targets[:, None]
        ).squeeze(1)
        expected_ratio = numerator / denominator
        expected = (
            expected_ratio.detach()
            * F.cross_entropy(logits, targets, reduction="none")
        ).mean()
        torch.testing.assert_close(result.sample_weights, expected_ratio.detach())
        torch.testing.assert_close(result.objective, expected)
        result.objective.backward()
        self.assertIsNone(transition.grad)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_stage2a_and_revision_share_ratio_values_but_not_gradient_ownership(self) -> None:
        logits_stage2a = torch.tensor(
            [[0.3, 1.1, -0.4], [1.2, -0.1, 0.2]],
            dtype=torch.float64,
            requires_grad=True,
        )
        logits_revision = logits_stage2a.detach().clone().requires_grad_(True)
        targets = torch.tensor([1, 0])
        transition_stage2a = torch.tensor(
            [[0.75, 0.20, 0.05], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]],
            dtype=torch.float64,
            requires_grad=True,
        )
        transition_revision = (
            transition_stage2a.detach().clone().requires_grad_(True)
        )
        stage2a = t_revision_reweight_objective(
            logits_stage2a,
            targets,
            transition_stage2a,
            denominator_floor=1.0e-12,
            detach_ratio=True,
        )
        revision = t_revision_reweight_objective(
            logits_revision,
            targets,
            transition_revision,
            denominator_floor=1.0e-12,
            detach_ratio=False,
        )
        torch.testing.assert_close(stage2a.sample_weights, revision.sample_weights)
        torch.testing.assert_close(stage2a.objective, revision.objective)
        stage2a.objective.backward()
        revision.objective.backward()
        self.assertIsNone(transition_stage2a.grad)
        self.assertIsNotNone(transition_revision.grad)
        self.assertTrue(torch.isfinite(transition_revision.grad).all())
        self.assertGreater(float(transition_revision.grad.abs().sum()), 0.0)
        for gradient in (logits_stage2a.grad, logits_revision.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_denominator_fail_fast_is_unchanged_when_ratio_is_detached(self) -> None:
        logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "strictly greater"):
            t_revision_reweight_objective(
                logits,
                torch.tensor([0]),
                torch.zeros(2, 2),
                denominator_floor=0.0,
                detach_ratio=True,
            )

    def test_epoch_telemetry_covers_weights_updates_and_transition_health(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _algorithm(directory)
            algorithm.diagnostic_transition = np.eye(3, dtype=np.float64)
            final = algorithm.run()
            rows = [
                json.loads(line)
                for line in (Path(directory) / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            transition_row = next(
                row for row in rows if row["event"] == "transition_initialization"
            )
            self.assertEqual(
                transition_row["posterior_finite_count"],
                transition_row["posterior_value_count"],
            )
            self.assertLessEqual(transition_row["posterior_row_sum_max_error"], 1e-6)
            self.assertEqual(len(transition_row["pseudo_anchor_indices"]), 3)
            for stage in ("classifier_initialization", "revision"):
                row = next(
                    value
                    for value in rows
                    if value.get("event") == "epoch" and value.get("stage") == stage
                )
                for field in (
                    "weight_p50",
                    "weight_p90",
                    "weight_p95",
                    "weight_p99",
                    "weight_ess",
                    "weight_ess_fraction",
                    "gradient_norm",
                    "parameter_norm",
                    "update_norm",
                    "relative_update_norm",
                    "optimizer_step_count",
                    "denominator_min",
                ):
                    self.assertIn(field, row)
                    self.assertTrue(np.isfinite(row[field]), field)
                self.assertEqual(row["weight_negative_count"], 0)
                self.assertEqual(row["weight_nonfinite_count"], 0)
                self.assertGreater(row["optimizer_step_count"], 0)
            revision = next(
                row
                for row in rows
                if row.get("event") == "epoch" and row.get("stage") == "revision"
            )
            self.assertIn("initial_transition", revision)
            self.assertIn("delta_transition", revision)
            self.assertIn("revised_transition", revision)
            self.assertIn("revised_transition_negative_entry_count", revision)
            self.assertGreater(revision["delta_l1"], 0.0)
            self.assertIn("initial_true_T_relative_L1_error", final)
            self.assertIn("revised_true_T_relative_L1_error", final)


if __name__ == "__main__":
    unittest.main()
