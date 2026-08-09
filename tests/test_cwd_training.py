import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.training.cwd_experiment import run_cwd_experiment


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
            self.sink.setdefault(self.stream, []).append((
                batch["index"].clone(), batch["input"].clone()
            ))
            yield batch


def _cifar(split: str, samples_per_class: int) -> CifarData:
    labels = np.repeat(np.asarray([0, 1], dtype=np.int64), samples_per_class)
    images = np.zeros((labels.size, 32, 32, 3), dtype=np.uint8)
    images[:, 0, 0, 0] = labels.astype(np.uint8)
    return CifarData(
        images,
        labels,
        tuple(map(str, range(10))),
        split,
        "cifar10",
    )


class CWDTrainingTest(unittest.TestCase):
    @staticmethod
    def _config(epochs: int = 3):
        return {
            "seed": 7,
            "data": {
                "name": "cifar10_airplane_automobile", "root": "unused",
                "folds": 2, "fold_index": 0, "validation_size": 0,
                "augment": True,
            },
            "noise": {"rho_positive": 0.0, "rho_negative": 0.0, "seed": 7},
            "loader": {"batch_size": 4, "num_workers": 0, "pin_memory": False},
            "model": {"name": "tiny_cnn", "width": 1},
            "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"milestones": [], "gamma": 0.1},
            "trainer": {"epochs": epochs, "device": "cpu"},
            "cwd": {"ridge": 1e-8, "dynamic_centroid": True},
        }

    def test_one_fold_writes_artifacts_and_resumable_checkpoint(self) -> None:
        train = _cifar("train", 6)
        test = _cifar("test", 2)
        config = {
            "seed": 7,
            "data": {
                "name": "cifar10_airplane_automobile",
                "root": "unused",
                "folds": 2,
                "fold_index": 0,
                "augment": False,
            },
            "noise": {
                "rho_positive": 0.0,
                "rho_negative": 0.0,
                "seed": 7,
            },
            "loader": {
                "batch_size": 8,
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {"name": "tiny_cnn", "width": 1},
            "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"milestones": [], "gamma": 0.1},
            "trainer": {"epochs": 1, "device": "cpu"},
            "cwd": {"ridge": 1e-8},
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "lnl_toolbox.training.cwd_experiment.load_cifar10",
                side_effect=lambda _root, split: train if split == "train" else test,
            ):
                result = run_cwd_experiment(config, directory)
            for name in (
                "last.pt",
                "noise_manifest.npz",
                "feature_snapshot.npz",
                "statistic_artifact.npz",
                "metrics.jsonl",
                "resolved_config.yaml",
            ):
                self.assertTrue((result / name).is_file(), name)
            payload = torch.load(
                result / "last.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["completed_epoch"], 0)
            self.assertTrue(payload["statistic_hash"])
            self.assertTrue(payload["feature_snapshot_hash"])
            rows = [
                json.loads(line)
                for line in (result / "metrics.jsonl").read_text().splitlines()
            ]
            epoch = next(row for row in rows if row["event"] == "epoch")
            final = next(row for row in rows if row["event"] == "final")
            self.assertEqual(epoch["validation_metric"], "unavailable")
            self.assertNotIn("validation_accuracy", epoch)
            self.assertNotIn("test_accuracy", epoch)
            self.assertEqual(final["validation_metric"], "unavailable")
            self.assertIn("test_accuracy", final)

    def test_validation_is_optional_and_never_aliases_held_out_test(self) -> None:
        train = _cifar("train", 8)
        test = _cifar("test", 2)
        config = {
            "seed": 9,
            "data": {
                "name": "cifar10_airplane_automobile", "root": "unused",
                "folds": 2, "fold_index": 0, "validation_size": 2,
                "augment": False,
            },
            "noise": {"rho_positive": 0.0, "rho_negative": 0.0, "seed": 9},
            "loader": {"batch_size": 8, "num_workers": 0, "pin_memory": False},
            "model": {"name": "tiny_cnn", "width": 1},
            "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"milestones": [], "gamma": 0.1},
            "trainer": {"epochs": 1, "device": "cpu"},
            "cwd": {"ridge": 1e-8},
        }
        evaluated_splits = []
        from lnl_toolbox.training import cwd_experiment
        original = cwd_experiment._evaluate_cwd

        def record(model, loader, device, **kwargs):
            evaluated_splits.append(loader.dataset.data.split)
            return original(model, loader, device, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.cwd_experiment.load_cifar10",
            side_effect=lambda _root, split: train if split == "train" else test,
        ), patch(
            "lnl_toolbox.training.cwd_experiment._evaluate_cwd", side_effect=record
        ):
            result = run_cwd_experiment(config, directory)
            metrics_text = (result / "metrics.jsonl").read_text()
        self.assertEqual(evaluated_splits, ["validation", "test"])
        rows = [json.loads(line) for line in metrics_text.splitlines()]
        epoch = next(row for row in rows if row["event"] == "epoch")
        self.assertEqual(epoch["validation_metric"], "clean_validation_accuracy")
        self.assertNotIn("test_accuracy", epoch)

    def test_epoch_boundary_resume_reproduces_cwd_stream_and_state(self) -> None:
        from lnl_toolbox.training import cwd_experiment
        train = _cifar("train", 8)
        test = _cifar("test", 2)
        config = self._config()
        original_builder = cwd_experiment.build_epoch_loader
        original_save = cwd_experiment.atomic_save

        def run(directory, sink, *, interrupt=False, resume=None):
            def build(*args, **kwargs):
                loader = original_builder(*args, **kwargs)
                stream = (kwargs["namespace"], kwargs["epoch"])
                return _RecordingLoader(loader, sink, stream)

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

            patches = (
                patch("lnl_toolbox.training.cwd_experiment.load_cifar10",
                      side_effect=lambda _root, split: train if split == "train" else test),
                patch("lnl_toolbox.training.cwd_experiment.build_epoch_loader", side_effect=build),
                patch("lnl_toolbox.training.cwd_experiment.atomic_save", side_effect=save),
            )
            with patches[0], patches[1], patches[2]:
                if interrupt:
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        run_cwd_experiment(config, directory, resume=resume)
                    return None
                return run_cwd_experiment(config, directory, resume=resume)

        with tempfile.TemporaryDirectory() as uninterrupted_dir, tempfile.TemporaryDirectory() as resumed_dir:
            uninterrupted_stream = {}
            uninterrupted = run(uninterrupted_dir, uninterrupted_stream)
            interrupted_stream = {}
            run(resumed_dir, interrupted_stream, interrupt=True)
            resumed_stream = {}
            resumed = run(
                resumed_dir, resumed_stream,
                resume=Path(resumed_dir) / "last.pt",
            )
            expected = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=False)
            actual = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
        for epoch in (1, 2):
            key = ("cwd.train", epoch)
            self.assertEqual(len(uninterrupted_stream[key]), len(resumed_stream[key]))
            for left, right in zip(uninterrupted_stream[key], resumed_stream[key]):
                torch.testing.assert_close(left[0], right[0], rtol=0.0, atol=0.0)
                torch.testing.assert_close(left[1], right[1], rtol=0.0, atol=0.0)
        _assert_nested_equal(self, expected["model"], actual["model"])
        _assert_nested_equal(self, expected["optimizer"], actual["optimizer"])
        self.assertEqual(expected["metrics"], actual["metrics"])
        self.assertEqual(expected["statistic_hash"], actual["statistic_hash"])


if __name__ == "__main__":
    unittest.main()
