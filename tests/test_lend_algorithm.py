import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from lnl_toolbox.algorithms.lend import LENDAlgorithm, LENDConfig
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.models.feature_output import FeatureOutput


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 3)
        self.classifier = nn.Linear(3, 2)
        self.feature_calls = 0

    def forward_with_features(self, inputs):
        self.feature_calls += 1
        features = torch.relu(self.encoder(inputs)) + 0.1
        return FeatureOutput(self.classifier(features), features)

    def forward(self, inputs):
        features = torch.relu(self.encoder(inputs)) + 0.1
        return self.classifier(features)


def _config(beta=1.0):
    return {"method": "lend", "data": {"name": "cifar10", "max_train_samples": 3},
        "model": {"name": "tiny_cnn"},
        "loader": {"batch_size": 3, "drop_last": False}, "loss": {"name": "ce"},
        "lend": {"graph": {"k": 2, "gamma": 1., "metric": "inner_product",
            "normalize_features": False, "zero_degree_policy": "error"},
            "dilution": {"alpha": .99, "policy": "fixed_steps", "steps": 1},
            "history": {"beta": beta, "first_observation": "current"},
            "selection": {"rule": "noisy_equals_diluted_argmax",
                "reduction": "paper_sum", "empty_batch": "skip_update"},
            "training": {"epochs": 2}}}


class LENDAlgorithmTest(unittest.TestCase):
    def _algorithm(self, beta=1.0, lr=0.0):
        torch.manual_seed(3)
        model = FeatureModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9,
                                    weight_decay=.01)
        algorithm = LENDAlgorithm(model=model, optimizer=optimizer,
            loss=nn.CrossEntropyLoss(reduction="none"), device=torch.device("cpu"),
            method_config=LENDConfig.from_mapping(_config(beta)),
            canonical_global_indices=torch.tensor([10, 20, 30]), num_classes=2)
        algorithm.setup(ExperimentContext(__import__('pathlib').Path('.')))
        return algorithm

    def test_one_forward_paper_sum_and_noisy_targets(self):
        algorithm = self._algorithm(lr=0.0)
        history = algorithm.private_state.history
        history.values[:] = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        history.initialized[:] = True
        inputs = torch.tensor([[1., 1.], [2., 1.], [1., 2.]])
        targets = torch.zeros(3, dtype=torch.long)
        before = copy.deepcopy(algorithm.model.state_dict())
        expected_model = FeatureModel(); expected_model.load_state_dict(before)
        expected = nn.CrossEntropyLoss(reduction="none")(
            expected_model(inputs), targets)[torch.tensor([True, False, True])].sum()
        expected.backward()
        result = algorithm.step(Batch({"input": inputs, "target": targets,
            "index": torch.tensor([10, 20, 30])}), RunState(cycle=0))
        self.assertEqual(algorithm.model.feature_calls, 1)
        self.assertEqual(result.metadata["selected_mask"].tolist(), [True, False, True])
        self.assertAlmostEqual(result.metrics["selected_train_loss_sum"], expected.item(), places=6)
        for parameter, expected_parameter in zip(algorithm.model.parameters(), expected_model.parameters()):
            torch.testing.assert_close(parameter.grad, expected_parameter.grad)

    def test_empty_selection_skips_optimizer_but_commits_history(self):
        algorithm = self._algorithm(lr=.1)
        history = algorithm.private_state.history
        history.values[:] = torch.tensor([[0., 1.], [0., 1.], [0., 1.]])
        history.initialized[:] = True
        before = copy.deepcopy(algorithm.model.state_dict())
        state = RunState(cycle=0)
        result = algorithm.step(Batch({"input": torch.tensor([[1., 1.], [2., 1.], [1., 2.]]),
            "target": torch.zeros(3, dtype=torch.long), "index": torch.tensor([10, 20, 30])}), state)
        self.assertEqual(result.metrics["selected_samples"], 0.0)
        self.assertEqual(algorithm.private_state.optimizer_steps, 0)
        self.assertTrue(bool((history.last_updated_epoch == 0).all()))
        for name, value in algorithm.model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_state_roundtrip_and_config_drift(self):
        source = self._algorithm()
        state = source.state_dict()
        restored = self._algorithm()
        restored.load_state_dict(state)
        changed = self._algorithm(beta=.5)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            changed.load_state_dict(state)

    def test_optimizer_failure_does_not_commit_history(self):
        algorithm = self._algorithm(beta=0.9, lr=.1)
        history = algorithm.private_state.history
        before = history.state_dict()
        batch = Batch({"input": torch.tensor([[1., 1.], [2., 1.], [1., 2.]]),
            "target": torch.zeros(3, dtype=torch.long), "index": torch.tensor([10, 20, 30])})
        with patch.object(algorithm.optimizer, "step", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                algorithm.step(batch, RunState(cycle=0))
        self.assertTrue(torch.equal(history.values, before["values"]))
        self.assertTrue(torch.equal(history.initialized, before["initialized"]))
        self.assertTrue(torch.equal(history.last_updated_epoch, before["last_updated_epoch"]))

    def test_clean_target_payload_is_ignored(self):
        left, right = self._algorithm(beta=.9, lr=0.), self._algorithm(beta=.9, lr=0.)
        inputs = torch.tensor([[1., 1.], [2., 1.], [1., 2.]])
        common = {"input": inputs, "target": torch.tensor([0, 1, 0]),
                  "index": torch.tensor([10, 20, 30])}
        left.step(Batch({**common, "clean_target": torch.tensor([0, 0, 0])}), RunState(cycle=0))
        right.step(Batch({**common, "clean_target": torch.tensor([1, 1, 1])}), RunState(cycle=0))
        for name, value in left.model.state_dict().items():
            torch.testing.assert_close(value, right.model.state_dict()[name])


if __name__ == "__main__": unittest.main()
