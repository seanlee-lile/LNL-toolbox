import tempfile
import unittest
import pickle
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.data.multi_view import (
    IndexedMultiViewCifarDataset,
    build_strong_cifar_transform,
)
from lnl_toolbox.models.fine_cnn import FineSevenCNN
from lnl_toolbox.training.fine_experiment import run_fine_experiment
from lnl_toolbox.training.model_ema import ModelEMA


def _assert_nested_equal(test: unittest.TestCase, left, right) -> None:
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            _assert_nested_equal(test, first, second)
    else:
        test.assertEqual(left, right)


class _RecordingLoader:
    def __init__(self, loader, sink, stream) -> None:
        self.loader = loader
        self.dataset = loader.dataset
        self.sink = sink
        self.stream = stream

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            values = [batch["index"].clone(), batch["input"].clone()]
            if "strong_input" in batch:
                values.append(batch["strong_input"].clone())
            self.sink.setdefault(self.stream, []).append(tuple(values))
            yield batch


def _cifar(split: str, samples_per_class: int) -> CifarData:
    labels = np.repeat(np.arange(100, dtype=np.int64), samples_per_class)
    images = np.zeros((labels.size, 32, 32, 3), dtype=np.uint8)
    images[:, 0, 0, 0] = labels.astype(np.uint8)
    return CifarData(images, labels, tuple(map(str, range(100))), split, "cifar100")


class FINETrainingTest(unittest.TestCase):
    @staticmethod
    def _resume_config():
        return {
            "seed": 3,
            "data": {
                "name": "cifar100", "root": "unused", "validation_size": 0,
                "augment": True, "strong_magnitude": 1,
            },
            "noise": {"name": "symmetric", "rate": 0.2, "seed": 3},
            "loader": {"batch_size": 100, "num_workers": 0, "pin_memory": False},
            "model": {"name": "tiny_cnn", "width": 2},
            "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.0},
            "scheduler": {"eta_min": 0.0005},
            "trainer": {"epochs": 3, "device": "cpu"},
            "fine": {
                "warmup_epochs": 1, "warmup_lr": 0.01, "ema_momentum": 0.9,
                "momentum_scs": 0.9, "momentum_scr": 0.9,
                "beta": 0.1, "gamma": 0.002, "alpha": 1.0,
            },
        }

    def test_seven_cnn_exposes_logits_and_features(self) -> None:
        model = FineSevenCNN(100, base_width=2)
        output = model.forward_with_features(torch.randn(2, 3, 32, 32))
        self.assertEqual(output.logits.shape, (2, 100))
        self.assertEqual(output.features.shape, (2, 16))
        outputs = model.forward_outputs(torch.randn(2, 3, 32, 32))
        self.assertEqual(outputs["logits"].shape, (2, 100))
        self.assertEqual(outputs["prob"].shape, (2, 100))

    def test_seven_cnn_initialization_matches_official_scope(self) -> None:
        model = FineSevenCNN(100, base_width=64)
        linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
        convolution_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        self.assertEqual(len(linear_layers), 4)
        self.assertEqual(len(convolution_layers), 6)
        for layer in linear_layers:
            self.assertTrue(torch.isfinite(layer.weight).all())
            self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))

    def test_model_ema_round_trip(self) -> None:
        model = torch.nn.Linear(2, 2)
        ema = ModelEMA(model, 0.9)
        with torch.no_grad():
            model.weight.add_(1.0)
        ema.update(model)
        restored = ModelEMA(model, 0.9)
        restored.load_state_dict(ema.state_dict())
        self.assertTrue(torch.equal(restored.model.weight, ema.model.weight))

    def test_fine_ema_matches_official_parameter_only_updates(self) -> None:
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
        ema = ModelEMA(model, 0.5, update_buffers=False)
        initial_mean = ema.model[0].running_mean.clone()
        with torch.no_grad():
            model[0].running_mean.add_(3.0)
            model[1].weight.add_(2.0)
        before = ema.model[1].weight.clone()
        ema.update(model)
        self.assertTrue(torch.equal(ema.model[0].running_mean, initial_mean))
        torch.testing.assert_close(ema.model[1].weight, 0.5 * before + 0.5 * model[1].weight)

    def test_multi_view_preserves_global_index_and_target(self) -> None:
        data = _cifar("train", 1)
        transform = lambda image: torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
        dataset = IndexedMultiViewCifarDataset(
            data,
            np.asarray([7]),
            weak_transform=transform,
            strong_transform=transform,
            targets_by_index={7: 3},
        )
        sample = dataset[0]
        self.assertEqual(sample["index"], 7)
        self.assertEqual(sample["target"], 3)
        self.assertEqual(sample["input"].shape, sample["strong_input"].shape)

    def test_official_cifar_strong_policy_returns_tensor(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        transform = build_strong_cifar_transform(policy="official_cifar10")
        result = transform(Image.fromarray(image, mode="RGB"))
        self.assertEqual(tuple(result.shape), (3, 32, 32))
        pickle.dumps(transform)

    def test_two_epoch_cli_lifecycle_writes_resumable_outputs(self) -> None:
        train = _cifar("train", 2)
        test = _cifar("test", 1)
        config = {
            "seed": 3,
            "data": {
                "name": "cifar100",
                "root": "unused",
                "validation_size": 100,
                "max_test_samples": 100,
                "augment": False,
                "strong_magnitude": 1,
            },
            "noise": {"name": "symmetric", "rate": 0.2, "seed": 3},
            "loader": {"batch_size": 100, "num_workers": 0, "pin_memory": False},
            "model": {"name": "tiny_cnn", "width": 2},
            "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.0},
            "scheduler": {"eta_min": 0.0005},
            "trainer": {"epochs": 2, "device": "cpu"},
            "fine": {
                "warmup_epochs": 1,
                "warmup_lr": 0.01,
                "ema_momentum": 0.9,
                "momentum_scs": 0.9,
                "momentum_scr": 0.9,
                "beta": 0.1,
                "gamma": 0.002,
                "alpha": 1.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "lnl_toolbox.training.fine_experiment.load_cifar100",
                side_effect=lambda _root, split: train if split == "train" else test,
            ):
                result = run_fine_experiment(config, directory)
            self.assertTrue((result / "last.pt").is_file())
            self.assertTrue((result / "metrics.jsonl").is_file())
            self.assertTrue((result / "training_curves.svg").is_file())
            payload = torch.load(result / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["completed_epoch"], 1)
            self.assertIn("ema", payload)
            self.assertIn("scs", payload)
            self.assertIn("scr", payload)

    def test_zero_validation_size_does_not_reuse_clean_test(self) -> None:
        train = _cifar("train", 1)
        test = _cifar("test", 1)
        config = {
            "seed": 5,
            "data": {
                "name": "cifar100", "root": "unused", "validation_size": 0,
                "max_test_samples": 100, "augment": False, "strong_magnitude": 1,
            },
            "noise": {"name": "symmetric", "rate": 0.2, "seed": 5},
            "loader": {"batch_size": 100, "num_workers": 0, "pin_memory": False},
            "model": {"name": "tiny_cnn", "width": 2},
            "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.0},
            "scheduler": {"eta_min": 0.0005},
            "trainer": {"epochs": 1, "device": "cpu"},
            "fine": {
                "warmup_epochs": 1, "warmup_lr": 0.01, "ema_momentum": 0.9,
                "momentum_scs": 0.9, "momentum_scr": 0.9,
                "beta": 0.1, "gamma": 0.002, "alpha": 1.0,
            },
        }
        evaluated_splits = []
        from lnl_toolbox.training import fine_experiment
        original = fine_experiment.evaluate_classification

        def record(model, loader, criterion, device):
            evaluated_splits.append(loader.dataset.data.split)
            return original(model, loader, criterion, device)

        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.fine_experiment.load_cifar100",
            side_effect=lambda _root, split: train if split == "train" else test,
        ), patch(
            "lnl_toolbox.training.fine_experiment.evaluate_classification",
            side_effect=record,
        ):
            result = run_fine_experiment(config, directory)
            metrics_text = (result / "metrics.jsonl").read_text()
        self.assertEqual(evaluated_splits, ["test"])
        rows = [json.loads(line) for line in metrics_text.splitlines()]
        epoch = next(row for row in rows if row["event"] == "epoch")
        final = next(row for row in rows if row["event"] == "final")
        self.assertEqual(epoch["validation_metric"], "unavailable")
        self.assertNotIn("validation_accuracy", epoch)
        self.assertNotIn("test_accuracy", epoch)
        self.assertEqual(final["validation_metric"], "unavailable")
        self.assertIn("test_accuracy", final)

    def test_epoch_boundary_resume_reproduces_fine_stream_and_state(self) -> None:
        from lnl_toolbox.training import fine_experiment
        train = _cifar("train", 2)
        test = _cifar("test", 1)
        config = self._resume_config()
        config["method"] = "fine"
        original_builder = fine_experiment.build_epoch_loader
        original_save = fine_experiment.atomic_save

        def run(directory, sink, *, interrupt=False, resume=None):
            def build(*args, **kwargs):
                loader = original_builder(*args, **kwargs)
                return _RecordingLoader(
                    loader, sink, (kwargs["namespace"], kwargs["epoch"])
                )

            stopped = False

            def save(payload, path):
                nonlocal stopped
                original_save(payload, path)
                if (
                    interrupt and not stopped and Path(path).name == "last.pt"
                    and int(payload["completed_epoch"]) == 0
                ):
                    stopped = True
                    raise RuntimeError("simulated interruption")

            with patch(
                "lnl_toolbox.training.fine_experiment.load_cifar100",
                side_effect=lambda _root, split: train if split == "train" else test,
            ), patch(
                "lnl_toolbox.training.fine_experiment.build_epoch_loader", side_effect=build
            ), patch(
                "lnl_toolbox.training.fine_experiment.atomic_save", side_effect=save
            ):
                if interrupt:
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        run_fine_experiment(config, directory, resume=resume)
                    return None
                return run_fine_experiment(config, directory, resume=resume)

        with tempfile.TemporaryDirectory() as uninterrupted_dir, tempfile.TemporaryDirectory() as resumed_dir:
            uninterrupted_stream = {}
            uninterrupted = run(uninterrupted_dir, uninterrupted_stream)
            interrupted_stream = {}
            run(resumed_dir, interrupted_stream, interrupt=True)
            legacy = torch.load(
                Path(resumed_dir) / "last.pt", map_location="cpu", weights_only=False
            )
            legacy["method"] = "fine_sed"
            legacy["config"].pop("method", None)
            torch.save(legacy, Path(resumed_dir) / "last.pt")
            resumed_stream = {}
            resumed = run(
                resumed_dir, resumed_stream,
                resume=Path(resumed_dir) / "last.pt",
            )
            expected = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=False)
            actual = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
        for epoch in (1, 2):
            for namespace in ("fine.snapshot", "fine.train"):
                key = (namespace, epoch)
                self.assertEqual(len(uninterrupted_stream[key]), len(resumed_stream[key]))
                for left, right in zip(uninterrupted_stream[key], resumed_stream[key]):
                    self.assertEqual(len(left), len(right))
                    for first, second in zip(left, right):
                        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        for key in ("model", "optimizer", "ema", "scs", "scr", "regularizer"):
            _assert_nested_equal(self, expected[key], actual[key])
        self.assertEqual(expected["metrics"], actual["metrics"])


if __name__ == "__main__":
    unittest.main()
