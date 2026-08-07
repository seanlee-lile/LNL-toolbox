import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.training.experiment import run_experiment


def _cifar(size, split, classes=10):
    rng = np.random.default_rng(401 + size + classes)
    images = rng.integers(1, 255, size=(size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(images, labels, tuple(map(str, range(classes))), split, f"cifar{classes}")


def _config(epochs=2, dataset="cifar10"):
    classes = 10 if dataset == "cifar10" else 100
    return {"method": "lend", "execution": {"runner": "lend"}, "seed": 7,
        "data": {"name": dataset, "root": "unused", "validation_size": classes,
            "max_train_samples": classes * 2, "max_validation_samples": classes,
            "max_test_samples": classes, "augment": False},
        "noise": {"name": "symmetric", "rate": .2, "seed": 17,
            "validation_targets": "noisy"},
        "loss": {"name": "ce"}, "model": {"name": "tiny_cnn", "width": 2},
        "optimizer": {"name": "adam", "lr": .001, "weight_decay": 0.},
        "scheduler": {"name": "none"},
        "loader": {"batch_size": classes, "num_workers": 0,
            "pin_memory": False, "drop_last": False},
        "lend": {"graph": {"k": classes - 1, "gamma": 1.,
                "metric": "inner_product", "normalize_features": False,
                "zero_degree_policy": "error"},
            "dilution": {"alpha": .99, "policy": "fixed_steps", "steps": 2},
            "history": {"beta": .9, "first_observation": "current"},
            "selection": {"rule": "noisy_equals_diluted_argmax",
                "reduction": "paper_sum", "empty_batch": "skip_update"},
            "training": {"epochs": epochs}},
        "evaluation": {"selection_split": "validation", "primary": "accuracy"},
        "trainer": {"device": "cpu"}}


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _nested_equal(test, left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0., atol=0.)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left: _nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right): _nested_equal(test, a, b)
    else: test.assertEqual(left, right)


class LENDWorkflowTest(unittest.TestCase):
    def _run_case(self, dataset="cifar10"):
        classes = 10 if dataset == "cifar10" else 100
        train, test = _cifar(classes * 4, "train", classes), _cifar(classes * 2, "test", classes)
        loader_name = "load_cifar10" if dataset == "cifar10" else "load_cifar100"
        with tempfile.TemporaryDirectory() as directory, patch(
            f"lnl_toolbox.training.lend_experiment.{loader_name}",
            side_effect=lambda _root, split: train if split == "train" else test,
        ):
            run_dir = run_experiment(_config(1, dataset), Path(directory) / "resumed")
            first = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(first["completed_epoch"], 0)
            manifest_hash = _sha(run_dir / "noise_manifest.npz")
            manifest_mtime = (run_dir / "noise_manifest.npz").stat().st_mtime_ns
            run_experiment(_config(2, dataset), resume=run_dir / "last.pt")
            resumed = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            uninterrupted_dir = run_experiment(_config(2, dataset), Path(directory) / "fresh")
            uninterrupted = torch.load(uninterrupted_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["completed_epoch"], 1)
            self.assertEqual(resumed["run_state"]["phase"], "completed")
            self.assertEqual(resumed["run_state"]["step"], 4)
            _nested_equal(self, resumed["model"], uninterrupted["model"])
            _nested_equal(self, resumed["optimizer"], uninterrupted["optimizer"])
            _nested_equal(self, resumed["algorithm_private_state"], uninterrupted["algorithm_private_state"])
            epoch_rows = lambda path: [json.loads(line) for line in path.read_text().splitlines()
                if json.loads(line)["event"] == "epoch"]
            self.assertEqual(epoch_rows(run_dir / "metrics.jsonl"),
                             epoch_rows(uninterrupted_dir / "metrics.jsonl"))
            self.assertEqual(_sha(run_dir / "noise_manifest.npz"), manifest_hash)
            self.assertEqual((run_dir / "noise_manifest.npz").stat().st_mtime_ns, manifest_mtime)
            tracked = {name: (_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns)
                       for name in ("last.pt", "metrics.jsonl", "noise_manifest.npz")}
            run_experiment(_config(2, dataset), resume=run_dir / "last.pt")
            self.assertEqual(tracked, {name: (_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns)
                       for name in tracked})
            final = json.loads((run_dir / "final_metrics.json").read_text())
            self.assertIn("clean_test_accuracy", final)
            self.assertEqual(final["fidelity"], "paper_oriented")

    def test_cifar10_fresh_resume_and_completed_noop(self): self._run_case()
    def test_cifar100_lightweight_workflow(self): self._run_case("cifar100")

    def test_resume_rejects_graph_drift(self):
        train, test = _cifar(40, "train"), _cifar(20, "test")
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.lend_experiment.load_cifar10",
            side_effect=lambda _root, split: train if split == "train" else test,
        ):
            run_dir = run_experiment(_config(1), Path(directory) / "run")
            changed = copy.deepcopy(_config(2)); changed["lend"]["graph"]["gamma"] = 2.
            with self.assertRaisesRegex(ValueError, "LEND settings"):
                run_experiment(changed, resume=run_dir / "last.pt")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_lightweight_workflow(self):
        train, test = _cifar(40, "train"), _cifar(20, "test")
        config = _config(1); config["trainer"]["device"] = "cuda"
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.lend_experiment.load_cifar10",
            side_effect=lambda _root, split: train if split == "train" else test,
        ):
            run_dir = run_experiment(config, Path(directory) / "cuda")
            final = json.loads((run_dir / "final_metrics.json").read_text())
            self.assertGreater(final["max_cuda_memory_mb"], 0.0)


if __name__ == "__main__": unittest.main()
