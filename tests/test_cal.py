from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from lnl_toolbox.algorithms.cal import (
    cal_covariance_correction,
    cal_transition_indicators,
    cores2_adjusted_losses,
    resolve_confidence_weight,
)
from lnl_toolbox.noise.cal import DROP, KEEP, RELABEL, CALProxyArtifact, build_cal_proxy_artifact
from lnl_toolbox.noise.estimators import PosteriorSnapshot
from lnl_toolbox.training.cal_experiment import (
    _assert_finite_warmup_gradients,
    _build_warmup_scheduler,
)
from lnl_toolbox.training.experiment import build_alpha_scaled_scheduler


class CALTest(unittest.TestCase):
    def _snapshot(self) -> PosteriorSnapshot:
        return PosteriorSnapshot(
            np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]),
            np.array([0, 0, 1]), np.array([8, 3, 5]), "fixture", "train",
        )

    def test_proxy_status_and_global_index_lookup(self) -> None:
        artifact = build_cal_proxy_artifact(
            self._snapshot(), np.array([-2.0, 2.0, 0.0]),
            lower_threshold=-1.0, upper_threshold=1.0,
        )
        targets, retained, status = artifact.lookup(torch.tensor([8, 5, 3]))
        self.assertEqual(status.tolist(), [KEEP, DROP, RELABEL])
        self.assertEqual(retained.tolist(), [True, False, True])
        self.assertEqual(targets.tolist(), [0, 1, 1])

    def test_artifact_round_trip(self) -> None:
        artifact = build_cal_proxy_artifact(
            self._snapshot(), np.array([-2.0, 2.0, 0.0]),
            lower_threshold=-1.0, upper_threshold=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.npz"
            artifact.save(path)
            loaded = CALProxyArtifact.load(path)
        self.assertEqual(loaded.artifact_hash, artifact.artifact_hash)

    def test_cores2_and_covariance_are_differentiable(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        adjusted = cores2_adjusted_losses(
            logits, torch.tensor([0, 1]), torch.tensor([0.5, 0.5]), 1.0
        )
        self.assertTrue(torch.allclose(adjusted, -torch.ones(2), atol=5e-5))
        losses = -torch.log_softmax(logits, dim=1)
        correction, means = cal_covariance_correction(
            losses, torch.tensor([0, 0]), torch.tensor([0, 1]),
            torch.tensor([True, True]), torch.tensor([1.0, 0.0]), torch.zeros(2, 2),
        )
        correction.backward()
        self.assertTrue(torch.isfinite(correction))
        self.assertEqual(tuple(means.shape), (2, 2))

    def test_transition_indicator(self) -> None:
        value = cal_transition_indicators(
            torch.tensor([0, 1]), torch.tensor([1, 0]),
            torch.tensor([True, False]), 2,
        )
        self.assertEqual(value[0, 0, 1].item(), 1.0)
        self.assertEqual(value[1].sum().item(), 0.0)

    def test_warmup_uses_configured_scheduler(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = _build_warmup_scheduler(
            optimizer,
            {"scheduler": {"name": "multistep", "milestones": [2], "gamma": 0.1}},
            3,
        )
        self.assertIsNotNone(scheduler)
        for _ in range(3):
            optimizer.zero_grad()
            parameter.sum().backward()
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_confidence_schedule_starts_with_ce_then_enables_cores2(self) -> None:
        schedule = {"milestones": [10, 40, 80], "values": [0.0, 2.0, 2.0]}
        self.assertEqual(resolve_confidence_weight(0, 2.0, schedule), 0.0)
        self.assertEqual(resolve_confidence_weight(9, 2.0, schedule), 0.0)
        self.assertEqual(resolve_confidence_weight(10, 2.0, schedule), 0.0)
        self.assertAlmostEqual(resolve_confidence_weight(25, 2.0, schedule), 1.0)
        self.assertEqual(resolve_confidence_weight(40, 2.0, schedule), 2.0)
        self.assertEqual(resolve_confidence_weight(64, 2.0, schedule), 2.0)

    def test_confidence_schedule_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            resolve_confidence_weight(0, 2.0, {"milestones": [2, 1], "values": [0.0]})
        with self.assertRaises(ValueError):
            resolve_confidence_weight(0, 2.0, {"milestones": [], "values": []})

    def test_alpha_scaled_scheduler_matches_official_lr_scaling(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = build_alpha_scaled_scheduler(
            optimizer, {"name": "multistep", "milestones": [60], "gamma": 0.1}
        )
        scheduler.step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        scheduler.step(2.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1 / 3.0)

    def test_warmup_gradient_guard_rejects_non_finite_gradient(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.weight.grad = torch.full_like(model.weight, float("nan"))
        with self.assertRaises(ValueError):
            _assert_finite_warmup_gradients(model)


if __name__ == "__main__":
    unittest.main()
