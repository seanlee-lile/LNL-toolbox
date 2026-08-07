from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from lnl_toolbox.algorithms.volminnet import VolMinNetAlgorithm, VolMinTransition


class VolMinNetAlgorithmTest(unittest.TestCase):
    def _algorithm(self) -> VolMinNetAlgorithm:
        model = nn.Linear(4, 3)
        transition = VolMinTransition(3)
        optimizer_model = torch.optim.SGD(model.parameters(), lr=0.05)
        optimizer_transition = torch.optim.SGD(transition.parameters(), lr=0.05)
        return VolMinNetAlgorithm(
            model=model,
            transition=transition,
            classifier_optimizer=optimizer_model,
            transition_optimizer=optimizer_transition,
            classifier_scheduler=None,
            transition_scheduler=None,
            lambda_volume=1e-4,
            device=torch.device("cpu"),
        )

    def test_joint_step_updates_both_once_with_disjoint_optimizers(self) -> None:
        algorithm = self._algorithm()
        model_before = copy.deepcopy(algorithm.model.state_dict())
        transition_before = algorithm.transition.off_diagonal_logits.detach().clone()
        metrics = algorithm.train_batch({
            "input": torch.randn(6, 4),
            "target": torch.tensor([0, 1, 2, 0, 1, 2]),
        })
        self.assertTrue(any(
            not torch.equal(value, model_before[name])
            for name, value in algorithm.model.state_dict().items()
        ))
        self.assertFalse(torch.equal(algorithm.transition.off_diagonal_logits, transition_before))
        self.assertEqual(algorithm.state.global_step, 1)
        self.assertEqual(algorithm.state.classifier_optimizer_steps, 1)
        self.assertEqual(algorithm.state.transition_optimizer_steps, 1)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_optimizer_ownership_mismatch_fails(self) -> None:
        model = nn.Linear(4, 3)
        transition = VolMinTransition(3)
        wrong = torch.optim.SGD(transition.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "exactly"):
            VolMinNetAlgorithm(
                model=model,
                transition=transition,
                classifier_optimizer=wrong,
                transition_optimizer=torch.optim.SGD(transition.parameters(), lr=0.1),
                classifier_scheduler=None,
                transition_scheduler=None,
                lambda_volume=1e-4,
                device=torch.device("cpu"),
            )

    def test_state_roundtrip_keeps_roles_and_progress(self) -> None:
        first = self._algorithm()
        first.train_batch({"input": torch.randn(3, 4), "target": torch.tensor([0, 1, 2])})
        state = first.state_dict()
        second = self._algorithm()
        second.load_state_dict(state)
        self.assertEqual(second.state.state_dict(), first.state.state_dict())
        self.assertEqual(second.transition.state_dict().keys(), first.transition.state_dict().keys())
