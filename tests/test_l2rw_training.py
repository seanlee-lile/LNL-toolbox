from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml
import torch
from torch import nn

from lnl_toolbox.algorithms.l2rw import meta_gradient
from lnl_toolbox.training.experiment import run_experiment
from lnl_toolbox.training.l2rw_experiment import (
    _official_flip_labels, _official_l2rw_transform, _official_partition,
    _official_train_subsets,
)


ROOT = Path(__file__).resolve().parents[1]


class L2RWTrainingTest(unittest.TestCase):
    def test_official_meta_gradient_matches_source_hessian_vector_product(self) -> None:
        torch.manual_seed(5)
        model = nn.Linear(3, 2)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 0, 1])
        trusted_inputs = torch.randn(2, 3)
        trusted_targets = torch.tensor([0, 1])
        actual = meta_gradient(
            model, inputs, targets, trusted_inputs, trusted_targets,
            virtual_learning_rate=1.0, weight_decay=0.2, implementation="official",
        )
        parameters = tuple(model.parameters())
        epsilon = torch.zeros(4, requires_grad=True)
        train_loss = torch.sum(epsilon * torch.nn.functional.cross_entropy(
            model(inputs), targets, reduction="none"
        ))
        train_gradients = torch.autograd.grad(train_loss, parameters, create_graph=True)
        trusted_loss = torch.nn.functional.cross_entropy(model(trusted_inputs), trusted_targets)
        trusted_loss = trusted_loss + 0.1 * sum(parameter.square().sum() for parameter in parameters)
        trusted_gradients = torch.autograd.grad(trusted_loss, parameters, retain_graph=True)
        expected = torch.autograd.grad(
            train_gradients, epsilon, grad_outputs=trusted_gradients,
        )[0]
        torch.testing.assert_close(actual, expected)

    def test_official_preprocessing_maps_zero_and_max_pixels_to_minus_one_and_one(self) -> None:
        from PIL import Image

        transform = _official_l2rw_transform(False, False)
        np.testing.assert_allclose(
            transform(Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))).numpy(),
            -1.0,
        )
        np.testing.assert_allclose(
            transform(Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))).numpy(),
            1.0,
        )

    def test_formal_config_selects_official_l2rw_resnet(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/experiment/l2rw_cifar10_reproduction.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["model"]["name"], "l2rw_resnet32")
        self.assertEqual(config["seed"], 1234)
        self.assertEqual(config["data"]["seed"], 0)
        self.assertEqual(config["noise"]["seed"], 0)
        from lnl_toolbox.models.cifar_resnet import L2RWResNet32
        from lnl_toolbox.training.reproduction_data import build_reproduction_model

        model = build_reproduction_model(config["model"], config["data"], 10)
        self.assertIsInstance(model, L2RWResNet32)
        self.assertEqual(len(model.stages), 15)
        self.assertEqual(tuple(model(torch.randn(2, 3, 32, 32)).shape), (2, 10))

    def test_official_meta_resnet_uses_batch_statistics_without_running_state(self) -> None:
        from lnl_toolbox.models.cifar_resnet import (
            L2RWResNet32,
            share_l2rw_meta_parameters,
        )

        model = L2RWResNet32(10, 16, meta_batch_statistics=True)
        self.assertTrue(model.meta_batch_statistics)
        self.assertFalse(any("running_mean" in key for key in model.state_dict()))
        self.assertFalse(any("running_var" in key for key in model.state_dict()))
        self.assertTrue(all("weight" not in name for name, _ in model.named_parameters()
                            if ".bn" in name or "_bn" in name))

        weighted = L2RWResNet32(10, 16, meta_batch_statistics=False)
        share_l2rw_meta_parameters(weighted, model)
        weighted_parameters = dict(weighted.named_parameters())
        for name, parameter in model.named_parameters():
            self.assertIs(parameter, weighted_parameters[name])

    def test_official_train_split_excludes_clean_meta_examples(self) -> None:
        positions = np.arange(12, dtype=np.int64)
        noisy, clean = _official_train_subsets(positions, num_clean=3, seed=7)
        self.assertEqual(noisy.size, 9)
        self.assertEqual(clean.size, 3)
        self.assertEqual(np.intersect1d(noisy, clean).size, 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([noisy, clean])), positions)

    def test_official_partition_matches_random_state_shuffle(self) -> None:
        actual = _official_partition(12, [9, 3], seed=7)
        expected = np.arange(12, dtype=np.int64)
        np.random.RandomState(7).shuffle(expected)
        np.testing.assert_array_equal(actual[0], expected[:9])
        np.testing.assert_array_equal(actual[1], expected[9:])

    def test_official_flip_labels_matches_source_operation(self) -> None:
        labels = np.arange(10, dtype=np.int64)
        actual_noisy, actual_clean_mask, actual_order = _official_flip_labels(
            labels, num_classes=10, rate=0.4, seed=7
        )
        random = np.random.RandomState(8)
        replacements = np.floor(random.uniform(0.0, 9.0, [4])).astype(np.int64)
        expected_noisy = np.concatenate([replacements, labels[4:]])
        expected_clean_mask = np.concatenate([
            np.zeros(4, dtype=np.int64), np.ones(6, dtype=np.int64)
        ])
        order = np.arange(10, dtype=np.int64)
        random.shuffle(order)
        np.testing.assert_array_equal(actual_noisy, expected_noisy[order])
        np.testing.assert_array_equal(actual_clean_mask, expected_clean_mask[order])
        np.testing.assert_array_equal(actual_order, order)
        # The serialized image order must use the same permutation as labels.
        np.testing.assert_array_equal(labels[actual_order], labels[order])

    def test_step_budget_stops_exactly(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/experiment/l2rw_cifar10_smoke.yaml").read_text(encoding="utf-8")
        )
        config["trainer"] = {"epochs": 10, "max_steps": 2, "device": "cpu"}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(config, Path(directory) / "run")
            payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["global_step"], 2)

    def test_official_meta_path_runs_end_to_end_for_short_budget(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/experiment/l2rw_cifar10_smoke.yaml").read_text(encoding="utf-8")
        )
        config["meta"] = {
            "virtual_learning_rate": 1.0,
            "implementation": "official",
        }
        config["trainer"] = {"epochs": 10, "max_steps": 2, "device": "cpu"}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(config, Path(directory) / "run")
            payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["global_step"], 2)

    def test_official_schedule_switches_before_boundary_step(self) -> None:
        milestones = [40000, 60000]
        self.assertEqual(sum(39999 + 1 >= value for value in milestones), 1)
        self.assertEqual(sum(40000 + 1 >= value for value in milestones), 1)
        self.assertEqual(sum(59999 + 1 >= value for value in milestones), 2)

    def test_smoke_and_completed_resume(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/experiment/l2rw_cifar10_smoke.yaml").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(config, Path(directory) / "run")
            checkpoint = run_dir / "last.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((run_dir / "trusted_validation_manifest.npz").is_file())
            self.assertEqual(run_experiment(config, resume=checkpoint), run_dir)


if __name__ == "__main__": unittest.main()
