from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np
import torch
from torch import nn
from unittest.mock import patch

from lnl_toolbox.algorithms.dividemix import DivideMixAlgorithm, DivideMixConfig, DivideMixPhase, DivideMixState, append_loss_history, build_co_divide, history_input
from lnl_toolbox.estimators import ReliabilityResult


def config():
    return {
        "method": "dividemix", "execution": {"runner": "dividemix"},
        "noise": {"name": "symmetric", "rate": 0.2, "validation_targets": "noisy"},
        "evaluation": {"selection_split": "validation"},
        "dividemix": {
            "fidelity": "official_cifar_v1", "warmup": {"epochs": 1},
            "gmm": {"threshold": 0.5, "loss_history": {"name": "official_auto", "window_epochs": 5}},
            "mixmatch": {"augmentations": 2, "temperature": 0.5, "mixup_alpha": 4.0, "mixup_lambda_scope": "minibatch"},
            "objective": {"lambda_u": 25.0, "lambda_r": 1.0, "rampup_epochs": 16},
            "training": {"epochs": 2}, "inference": {"ensemble": "official_logits_sum"},
        },
    }


class DivideMixAlgorithmTest(unittest.TestCase):
    def make_algorithm(self):
        torch.manual_seed(1)
        a = nn.Linear(2, 2)
        torch.manual_seed(2)
        b = nn.Linear(2, 2)
        oa, ob = torch.optim.SGD(a.parameters(), 0.1), torch.optim.SGD(b.parameters(), 0.1)
        return DivideMixAlgorithm(model_a=a, model_b=b, optimizer_a=oa, optimizer_b=ob, scheduler_a=None, scheduler_b=None, config=DivideMixConfig.from_mapping(config()), device=torch.device("cpu"))

    def test_optimizer_ownership_and_peer_isolation(self):
        algorithm = self.make_algorithm()
        before_b = deepcopy(algorithm.model_b.state_dict())
        views_x = (torch.randn(2, 2), torch.randn(2, 2))
        views_u = (torch.randn(2, 2), torch.randn(2, 2))
        algorithm.train_peer_step("a", views_x, views_u, torch.tensor([0, 1]), torch.tensor([0.8, 0.7]), epoch=0, batch_index=0, num_batches=1, rng=np.random.default_rng(1))
        for name, value in before_b.items():
            self.assertTrue(torch.equal(value, algorithm.model_b.state_dict()[name]))
        self.assertEqual(algorithm.state.optimizer_steps_a, 1)
        self.assertEqual(algorithm.state.optimizer_steps_b, 0)

    def test_phase_machine_rejects_illegal_transition(self):
        state = DivideMixState()
        with self.assertRaisesRegex(ValueError, "illegal"):
            state.transition(DivideMixPhase.TRAIN_NETWORK_A)

    def test_config_rejects_self_pipeline_composition(self):
        value = config(); value["selector"] = {"name": "small_loss"}
        with self.assertRaisesRegex(ValueError, "owns its complete pipeline"):
            DivideMixConfig.from_mapping(value)

    def test_history_average_aligns_by_stable_index(self):
        history = []
        append_loss_history(history, torch.tensor([20, 10]), torch.tensor([4.0, 2.0]), 5)
        append_loss_history(history, torch.tensor([10, 20]), torch.tensor([2.0, 6.0]), 5)
        averaged = history_input(history, torch.tensor([20, 10]), use_average=True)
        self.assertTrue(torch.allclose(averaged, torch.tensor([1.0, 0.0], dtype=torch.float64)))

    def test_co_divide_uses_cross_not_self_probabilities(self):
        class Estimator:
            calls = 0
            def __init__(self, **_kwargs): pass
            def estimate(self, value):
                scores = torch.tensor([0.9, 0.1, 0.9, 0.1], dtype=torch.float64)
                if self.calls == 1: scores = 1.0 - scores
                self.calls += 1
                return ReliabilityResult(value.sample_indices, scores, {"clean_component_mean": 0.2, "noisy_component_mean": 0.8, "gmm_iterations": 1.0})
        indices = torch.tensor([3, 8, 20, 44]); history_a, history_b = [], []
        append_loss_history(history_a, indices, torch.tensor([1.0, 2.0, 3.0, 4.0]), 5)
        append_loss_history(history_b, indices, torch.tensor([4.0, 3.0, 2.0, 1.0]), 5)
        with patch("lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator", Estimator):
            result = build_co_divide(indices, history_a, history_b, DivideMixConfig.from_mapping(config()), 0.2)
        self.assertTrue(torch.equal(result.labeled_for_a, torch.tensor([False, True, False, True])))
        self.assertTrue(torch.equal(result.labeled_for_b, torch.tensor([True, False, True, False])))


if __name__ == "__main__":
    unittest.main()
