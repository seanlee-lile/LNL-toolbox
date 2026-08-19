from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.training.experiment import run_experiment


def _cifar(size: int, split: str, classes: int = 10) -> CifarData:
    rng = np.random.default_rng(12 if split == "train" else 13)
    images = rng.integers(0, 256, (size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(images, labels, tuple(map(str, range(classes))), split, f"cifar{classes}")


def _config(epochs: int = 2, dataset: str = "cifar10") -> dict:
    classes = 10 if dataset == "cifar10" else 100
    return {
        "method": "volminnet",
        "execution": {"runner": "volminnet"},
        "seed": 5,
        "data": {
            "name": dataset, "root": "unused", "num_classes": classes,
            "validation_size": classes,
            "max_train_samples": classes * 2,
            "max_validation_samples": classes,
            "max_test_samples": classes,
            "augment": False,
        },
        "noise": {
            "name": "symmetric", "rate": 0.2, "seed": 7,
            "sampling": "transition", "rng": "default_rng",
            "validation_targets": "noisy", "manifest_filename": "noise_manifest.npz",
        },
        "volminnet": {
            "fidelity": "paper_positive_logdet",
            "model": {"name": "tiny_cnn", "width": 2},
            "transition": {
                "parameterization": "fixed_diagonal_sigmoid_offdiag",
                "convention": "clean_to_noisy_row", "normalization_axis": "row",
                "initialization": {"mode": "paper"},
            },
            "objective": {
                "classification": "noisy_nll",
                "volume": {"mode": "positive_logdet", "coefficient": 1e-4},
            },
            "optimizer": {
                "classifier": {"name": "sgd", "lr": 0.01, "momentum": 0.0},
                "transition": {"name": "sgd", "lr": 0.01, "momentum": 0.0},
            },
            "scheduler": {"classifier": {"name": "none"}, "transition": {"name": "none"}},
            "checkpoint_selection": {"split": "noisy_validation", "metric": "loss", "mode": "min"},
        },
        "loader": {"batch_size": classes, "num_workers": 0, "pin_memory": False},
        "trainer": {"epochs": epochs, "device": "cpu"},
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VolMinNetWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = _cifar(40, "train")
        self.test = _cifar(20, "test")

    def _load(self, _root, split):
        return self.train if split == "train" else self.test

    def test_fresh_resume_extension_and_completed_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10", side_effect=self._load
        ):
            run_dir = run_experiment(_config(1), Path(directory) / "run")
            first = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertTrue(first["algorithm"]["volminnet_state"]["completed"])
            self.assertEqual(first["algorithm"]["volminnet_state"]["global_step"], 2)
            self.assertIn("model", first["best_pair"])
            self.assertIn("transition", first["best_pair"])
            for name in (
                "resolved_config.yaml", "environment.json", "noise_manifest.npz",
                "metrics.jsonl", "final_metrics.json", "last.pt", "best.pt",
                "transition_initial.npz", "transition_best.npz", "transition_last.npz",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            manifest_hash = _sha(run_dir / "noise_manifest.npz")
            initial_hash = _sha(run_dir / "transition_initial.npz")
            run_experiment(_config(2), resume=run_dir / "last.pt")
            resumed = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["algorithm"]["volminnet_state"]["global_step"], 4)
            self.assertEqual(_sha(run_dir / "noise_manifest.npz"), manifest_hash)
            self.assertEqual(_sha(run_dir / "transition_initial.npz"), initial_hash)
            watched = {
                name: (_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns)
                for name in (
                    "last.pt", "best.pt", "metrics.jsonl", "final_metrics.json",
                    "noise_manifest.npz", "transition_initial.npz", "transition_best.npz",
                    "transition_last.npz",
                )
            }
            run_experiment(_config(2), resume=run_dir / "last.pt")
            for name, expected in watched.items():
                self.assertEqual((_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns), expected)

    def test_uninterrupted_matches_epoch_boundary_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10", side_effect=self._load
        ):
            direct = run_experiment(_config(2), Path(directory) / "direct")
            resumed = run_experiment(_config(1), Path(directory) / "resumed")
            run_experiment(_config(2), resume=resumed / "last.pt")
            direct_state = torch.load(direct / "last.pt", map_location="cpu", weights_only=False)["algorithm"]
            resumed_state = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)["algorithm"]
            for owner in ("model", "transition"):
                for key in direct_state[owner]:
                    self.assertTrue(torch.equal(direct_state[owner][key], resumed_state[owner][key]), f"{owner}.{key}")

    def test_resume_rejects_method_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10", side_effect=self._load
        ):
            run_dir = run_experiment(_config(1), Path(directory) / "run")
            changed = _config(2)
            changed["volminnet"]["objective"]["volume"]["coefficient"] = 0.01
            with self.assertRaisesRegex(ValueError, "method settings"):
                run_experiment(changed, resume=run_dir / "last.pt")

    def test_cifar100_lightweight_dispatch(self) -> None:
        train = _cifar(200, "train", 100)
        test = _cifar(100, "test", 100)
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar100",
            side_effect=lambda _root, split: train if split == "train" else test,
        ):
            run_dir = run_experiment(_config(1, "cifar100"), Path(directory) / "run")
            final = json.loads((run_dir / "final_metrics.json").read_text())
            self.assertEqual(len(final["learned_transition"]), 100)

    def test_test_metrics_do_not_select_best(self) -> None:
        config = _config(1)
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10", side_effect=self._load
        ), patch(
            "lnl_toolbox.training.volminnet_experiment._evaluate_clean",
            return_value={"loss": 999.0, "accuracy": 0.0, "samples": 10.0},
        ):
            run_dir = run_experiment(config, Path(directory) / "run")
            payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["best_pair"]["validation_loss"], payload["algorithm"]["volminnet_state"]["best_validation_loss"])


if __name__ == "__main__":
    unittest.main()
