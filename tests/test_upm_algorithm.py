import unittest

import torch

from lnl_toolbox.algorithms.upm import (
    ConfusingProbabilityState,
    UPMConfig,
    UPMTargetProvider,
    predict_true_posterior,
)
from lnl_toolbox.core import TargetInput
from lnl_toolbox.noise import PosteriorSnapshot


def _config():
    stage = {
        "epochs": 1, "model": {"name": "tiny_cnn"},
        "optimizer": {"name": "sgd", "lr": 0.01},
        "scheduler": {"name": "none"},
    }
    return {
        "method": "upm", "execution": {"runner": "upm"},
        "noise": {"validation_targets": "noisy"},
        "evaluation": {"selection_split": "validation"},
        "upm": {
            "stage1": {**stage, "best_metric": "noisy_validation_accuracy"},
            "psi": {"source": "stage1_best", "split": "train", "augmentation": False},
            "main": {**stage, "initialization": "fresh"},
            "confusing_probability": {
                "initial_value": 0.01, "learning_rate": 0.1,
                "epsilon": 1e-4, "update_start_epoch": 0,
                "update_interval_epochs": 1,
            },
        },
    }


class UPMTargetProviderTest(unittest.TestCase):
    def test_same_fixed_q_is_returned_after_eta_update(self) -> None:
        snapshot = PosteriorSnapshot(
            [[0.8, 0.2], [0.3, 0.7]], [0, 1], [4, 9], "tiny", "train"
        )
        eta = ConfusingProbabilityState(torch.tensor([4, 9]), 0.01)
        provider = UPMTargetProvider(
            snapshot=snapshot, eta_state=eta,
            config=UPMConfig.from_mapping(_config()), device=torch.device("cpu"),
        )
        logits = torch.tensor([[0.4, -0.1], [-0.2, 0.6]], requires_grad=True)
        result = provider.resolve(TargetInput(
            logits.detach(), torch.tensor([0, 1]), torch.tensor([4, 9]),
            {"epoch": 0},
        ))
        expected = predict_true_posterior(
            torch.softmax(logits.detach(), 1), torch.tensor([0, 1]),
            torch.tensor([0.8, 0.7]), torch.full((2,), 0.01),
        )
        torch.testing.assert_close(result.targets, expected)
        torch.testing.assert_close(provider.last_q, expected)
        self.assertFalse(result.targets.requires_grad)
        self.assertFalse(torch.equal(eta.eta, torch.full((2,), 0.01, dtype=torch.float64)))

    def test_schedule_skips_eta_but_still_produces_q(self) -> None:
        config = _config()
        config["upm"]["confusing_probability"]["update_start_epoch"] = 2
        snapshot = PosteriorSnapshot([[0.6, 0.4]], [0], [5], "tiny", "train")
        eta = ConfusingProbabilityState(torch.tensor([5]), 0.1)
        provider = UPMTargetProvider(
            snapshot=snapshot, eta_state=eta,
            config=UPMConfig.from_mapping(config), device=torch.device("cpu"),
        )
        provider.resolve(TargetInput(
            torch.tensor([[0.2, -0.2]]), torch.tensor([0]), torch.tensor([5]),
            {"epoch": 1},
        ))
        self.assertEqual(int(eta.update_count.item()), 0)


if __name__ == "__main__":
    unittest.main()
