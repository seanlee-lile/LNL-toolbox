import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from lnl_toolbox.algorithms.lend import LENDAlgorithm, LENDConfig
from lnl_toolbox.algorithms.lend.graph import build_lend_similarity
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.models.cifar_resnet import cifar_resnet18
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
            "normalize_features": False},
            "dilution": {"alpha": .99, "policy": "fixed_steps", "steps": 1},
            "history": {"beta": beta, "first_observation": "current"},
            "selection": {"rule": "noisy_equals_diluted_argmax",
                "reduction": "batch_mean", "empty_batch": "skip_update"},
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

    def test_exact_batch_mean_formula_uses_four_sample_denominator(self):
        config = _config()
        config["loader"]["batch_size"] = 4
        model = FeatureModel()
        algorithm = LENDAlgorithm(
            model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            loss=nn.CrossEntropyLoss(reduction="none"), device=torch.device("cpu"),
            method_config=LENDConfig.from_mapping(config),
            canonical_global_indices=torch.tensor([10, 20, 30, 40]),
            num_classes=2,
        )
        algorithm.setup(ExperimentContext(__import__('pathlib').Path('.')))
        algorithm.private_state.history.values[:] = torch.tensor(
            [[1., 0.], [1., 0.], [0., 1.], [1., 0.]]
        )
        algorithm.private_state.history.initialized[:] = True
        logits = torch.tensor(
            [[2., 0.], [1., 0.], [0., 1.], [.5, 0.]], requires_grad=True
        )
        losses = nn.CrossEntropyLoss(reduction="none")(
            logits, torch.zeros(4, dtype=torch.long)
        )
        featured = FeatureOutput(logits, torch.ones(4, 3))
        with patch("lnl_toolbox.algorithms.lend.algorithm.forward_with_features",
                   return_value=featured):
            result = algorithm.step(Batch({"input": torch.zeros(4, 2),
                "target": torch.zeros(4, dtype=torch.long),
                "index": torch.tensor([10, 20, 30, 40])}), RunState(cycle=0))
        self.assertEqual(
            result.metadata["selected_mask"].tolist(), [True, True, False, True]
        )
        self.assertAlmostEqual(
            result.metrics["selected_train_loss_sum"],
            ((losses[0] + losses[1] + losses[3]) / 4).item(), places=6,
        )

    def test_one_forward_batch_mean_and_noisy_targets(self):
        algorithm = self._algorithm(lr=0.0)
        history = algorithm.private_state.history
        history.values[:] = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        history.initialized[:] = True
        inputs = torch.tensor([[1., 1.], [2., 1.], [1., 2.]])
        targets = torch.zeros(3, dtype=torch.long)
        before = copy.deepcopy(algorithm.model.state_dict())
        expected_model = FeatureModel(); expected_model.load_state_dict(before)
        expected = nn.CrossEntropyLoss(reduction="none")(
            expected_model(inputs), targets)[torch.tensor([True, False, True])].sum() / 3
        expected.backward()
        result = algorithm.step(Batch({"input": inputs, "target": targets,
            "index": torch.tensor([10, 20, 30])}), RunState(cycle=0))
        self.assertEqual(algorithm.model.feature_calls, 1)
        self.assertEqual(result.metadata["selected_mask"].tolist(), [True, False, True])
        self.assertAlmostEqual(result.metrics["selected_train_loss_sum"], expected.item(), places=6)
        for parameter, expected_parameter in zip(algorithm.model.parameters(), expected_model.parameters()):
            torch.testing.assert_close(parameter.grad, expected_parameter.grad)

    def test_all_selected_batch_mean_equals_ce_mean(self):
        algorithm = self._algorithm(lr=0.0)
        history = algorithm.private_state.history
        history.values[:] = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
        history.initialized[:] = True
        inputs = torch.tensor([[1., 1.], [2., 1.], [1., 2.]])
        targets = torch.zeros(3, dtype=torch.long)
        expected = nn.CrossEntropyLoss()(algorithm.model(inputs), targets)
        result = algorithm.step(Batch({"input": inputs, "target": targets,
            "index": torch.tensor([10, 20, 30])}), RunState(cycle=0))
        self.assertEqual(result.metadata["selected_mask"].tolist(), [True] * 3)
        self.assertAlmostEqual(
            result.metrics["selected_train_loss_sum"], expected.item(), places=6
        )

    def test_partial_selection_uses_batch_size_not_selected_count(self):
        algorithm = self._algorithm(lr=0.0)
        history = algorithm.private_state.history
        history.values[:] = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        history.initialized[:] = True
        inputs = torch.tensor([[1., 1.], [2., 1.], [1., 2.]])
        targets = torch.zeros(3, dtype=torch.long)
        losses = nn.CrossEntropyLoss(reduction="none")(algorithm.model(inputs), targets)
        result = algorithm.step(Batch({"input": inputs, "target": targets,
            "index": torch.tensor([10, 20, 30])}), RunState(cycle=0))
        self.assertAlmostEqual(
            result.metrics["selected_train_loss_sum"],
            ((losses[0] + losses[2]) / 3).item(), places=6,
        )
        self.assertNotAlmostEqual(
            result.metrics["selected_train_loss_sum"],
            ((losses[0] + losses[2]) / 2).item(), places=6,
        )

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

        legacy = copy.deepcopy(state)
        legacy["method_config"]["zero_degree_policy"] = "error"
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            restored.load_state_dict(legacy)

        legacy = copy.deepcopy(state)
        legacy["method_config"]["reduction"] = "paper_sum"
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            restored.load_state_dict(legacy)

    def test_resnet18_batch_256_step_keeps_finite_nonzero_features(self):
        torch.manual_seed(19)
        model = cifar_resnet18(num_classes=10, base_width=64)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=.05, momentum=.9, weight_decay=5e-4
        )
        config = _config(beta=.9)
        config["data"]["max_train_samples"] = 256
        config["loader"]["batch_size"] = 256
        config["lend"]["graph"]["k"] = 8
        algorithm = LENDAlgorithm(
            model=model, optimizer=optimizer,
            loss=nn.CrossEntropyLoss(reduction="none"), device=torch.device("cpu"),
            method_config=LENDConfig.from_mapping(config),
            canonical_global_indices=torch.arange(256), num_classes=10,
        )
        algorithm.setup(ExperimentContext(__import__('pathlib').Path('.')))
        inputs = torch.randn(256, 3, 32, 32)
        targets = torch.arange(256) % 10
        before = torch.cat([value.detach().flatten() for value in model.parameters()])
        classifier_weight_before = model.classifier.weight.detach().clone()
        classifier_before = classifier_weight_before.norm()
        result = algorithm.step(Batch({"input": inputs, "target": targets,
            "index": torch.arange(256)}), RunState(cycle=0))
        after = torch.cat([value.detach().flatten() for value in model.parameters()])
        self.assertTrue(bool(torch.isfinite(after).all()))
        self.assertLess(float((after - before).norm() / before.norm()), .1)
        self.assertLess(
            float((model.classifier.weight.detach() - classifier_weight_before).norm()
                  / classifier_before),
            1.0,
        )
        model.train()
        with torch.no_grad():
            features = model.forward_with_features(inputs).features
        self.assertGreater(float(features.norm(dim=1).median()), 0.0)
        adjacency = build_lend_similarity(
            features.detach(), torch.arange(256), k=8, gamma=1.0,
            metric="inner_product", normalize_features=False,
        )
        self.assertGreater(int(torch.count_nonzero(adjacency)), 0)
        self.assertTrue(torch.isfinite(torch.tensor(
            result.metrics["selected_train_loss_sum"]
        )))

    def test_zero_degree_graph_completes_algorithm_step(self):
        algorithm = self._algorithm(lr=0.0)
        featured = FeatureOutput(
            torch.tensor([[2., 0.], [0., 2.], [2., 0.]], requires_grad=True),
            torch.tensor([[1., 0.], [1., 0.], [-1., 0.]]),
        )
        batch = Batch({"input": torch.zeros(3, 2),
            "target": torch.tensor([0, 1, 0]), "index": torch.tensor([10, 20, 30])})
        with patch("lnl_toolbox.algorithms.lend.algorithm.forward_with_features",
                   return_value=featured):
            result = algorithm.step(batch, RunState(cycle=0))
        self.assertEqual(result.metrics["graph_degree_min"], 0.0)
        self.assertEqual(algorithm.private_state.history.initialized.tolist(), [True] * 3)
        self.assertEqual(result.metadata["diluted_labels"][2].argmax().item(), 0)

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
