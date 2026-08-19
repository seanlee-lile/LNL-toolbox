from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.training.experiment import run_experiment
from lnl_toolbox.training.upm_experiment import UPMWorkflow


def _cifar(size: int, split: str, classes: int = 10) -> CifarData:
    rng = np.random.default_rng(31 if split == "train" else 32)
    images = rng.integers(0, 256, (size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(images, labels, tuple(map(str, range(classes))), split, f"cifar{classes}")


def _config(main_epochs: int = 2, *, stage1_epochs: int = 1, dataset: str = "cifar10") -> dict:
    classes = 10 if dataset == "cifar10" else 100
    stage = {
        "epochs": stage1_epochs,
        "model": {"name": "tiny_cnn", "width": 2},
        "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.0},
        "scheduler": {"name": "none"},
    }
    return {
        "method": "upm", "execution": {"runner": "upm"}, "seed": 9,
        "data": {
            "name": dataset, "root": "unused", "validation_size": classes,
            "max_train_samples": classes * 2,
            "max_validation_samples": classes, "max_test_samples": classes,
            "augment": False,
        },
        "loader": {"batch_size": classes, "num_workers": 0, "pin_memory": False},
        "noise": {
            "name": "symmetric", "rate": 0.2, "seed": 11,
            "validation_targets": "noisy", "manifest_filename": "noise_manifest.npz",
        },
        "upm": {
            "stage1": {**stage, "best_metric": "noisy_validation_accuracy"},
            "psi": {"source": "stage1_best", "split": "train", "augmentation": False},
            "main": {**stage, "epochs": main_epochs, "initialization": "fresh"},
            "confusing_probability": {
                "initial_value": 0.01, "learning_rate": 0.1, "epsilon": 1e-4,
                "update_start_epoch": 0, "update_interval_epochs": 1,
            },
        },
        "evaluation": {"selection_split": "validation", "primary": "accuracy"},
        "trainer": {"device": "cpu"},
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UPMWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = _cifar(40, "train")
        self.test = _cifar(20, "test")

    def _load(self, _root, split):
        return self.train if split == "train" else self.test

    def test_fresh_resume_extension_and_completed_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10", side_effect=self._load
        ):
            run_dir = run_experiment(_config(2), Path(directory) / "run")
            payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["upm_state"]["phase"], "completed")
            self.assertEqual(payload["upm_state"]["main_completed_epochs"], 2)
            self.assertEqual(payload["upm_state"]["main_global_step"], 4)
            self.assertIsNotNone(payload["best_main_model_state"])
            self.assertIsNotNone(payload["best_eta_state"])
            names = (
                "resolved_config.yaml", "environment.json", "noise_manifest.npz",
                "stage1_best.pt", "psi_snapshot.npz", "eta_initial.npz",
                "eta_best.npz", "eta_last.npz", "best.pt", "last.pt",
                "metrics.jsonl", "final_metrics.json",
            )
            for name in names:
                self.assertTrue((run_dir / name).is_file(), name)
            artifact_hashes = {
                name: _sha(run_dir / name)
                for name in ("psi_snapshot.npz", "eta_initial.npz", "noise_manifest.npz")
            }
            run_experiment(_config(3), resume=run_dir / "last.pt")
            extended = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(extended["upm_state"]["main_completed_epochs"], 3)
            self.assertEqual(extended["upm_state"]["main_global_step"], 6)
            resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
            self.assertEqual(resolved["upm"]["main"]["epochs"], 3)
            for name, digest in artifact_hashes.items():
                self.assertEqual(_sha(run_dir / name), digest)
            watched = {
                name: (_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns)
                for name in names
            }
            run_experiment(resolved, resume=run_dir / "last.pt")
            for name, expected in watched.items():
                self.assertEqual(
                    (_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns), expected
                )

    def test_snapshot_corruption_and_identity_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10", side_effect=self._load
        ):
            run_dir = run_experiment(_config(1), Path(directory) / "run")
            changed = _config(2)
            changed["upm"]["confusing_probability"]["learning_rate"] = 0.2
            with self.assertRaisesRegex(ValueError, "identity"):
                run_experiment(changed, resume=run_dir / "last.pt")
            (run_dir / "psi_snapshot.npz").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "hash"):
                run_experiment(_config(2), resume=run_dir / "last.pt")

    def test_stage1_and_main_interrupted_resume(self) -> None:
        original_stage1 = UPMWorkflow.train_stage1
        original_main = UPMWorkflow.train_main

        def one_stage1(owner):
            return original_stage1(owner, max_epochs=1)

        def one_main(owner):
            return original_main(owner, max_epochs=1)

        config = _config(2, stage1_epochs=2)
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10", side_effect=self._load
        ):
            stage1_dir = Path(directory) / "stage1"
            with patch.object(UPMWorkflow, "train_stage1", one_stage1):
                run_experiment(config, stage1_dir)
            stage1_payload = torch.load(stage1_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(stage1_payload["upm_state"]["phase"], "stage1_training")
            self.assertEqual(stage1_payload["upm_state"]["stage1_completed_epochs"], 1)
            run_experiment(config, resume=stage1_dir / "last.pt")
            self.assertEqual(
                torch.load(stage1_dir / "last.pt", map_location="cpu", weights_only=False)["upm_state"]["phase"],
                "completed",
            )

            main_dir = Path(directory) / "main"
            with patch.object(UPMWorkflow, "train_main", one_main):
                run_experiment(_config(2), main_dir)
            main_payload = torch.load(main_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(main_payload["upm_state"]["phase"], "main_training")
            self.assertEqual(main_payload["upm_state"]["main_completed_epochs"], 1)
            run_experiment(_config(2), resume=main_dir / "last.pt")
            self.assertEqual(
                torch.load(main_dir / "last.pt", map_location="cpu", weights_only=False)["upm_state"]["phase"],
                "completed",
            )

    def test_uninterrupted_matches_completed_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10", side_effect=self._load
        ):
            direct = run_experiment(_config(2), Path(directory) / "direct")
            resumed = run_experiment(_config(1), Path(directory) / "resumed")
            run_experiment(_config(2), resume=resumed / "last.pt")
            left = torch.load(direct / "last.pt", map_location="cpu", weights_only=False)
            right = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
            for owner in ("main",):
                for key, value in left[owner]["model"].items():
                    self.assertTrue(torch.equal(value, right[owner]["model"][key]), key)
            torch.testing.assert_close(left["eta_state"]["eta"], right["eta_state"]["eta"], rtol=0, atol=0)
            torch.testing.assert_close(left["eta_state"]["update_count"], right["eta_state"]["update_count"], rtol=0, atol=0)

    def test_clean_test_does_not_select_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10", side_effect=self._load
        ), patch(
            "lnl_toolbox.training.upm_experiment.evaluate_classification",
            side_effect=[
                {"loss": 1.0, "accuracy": 0.4, "samples": 10.0},
                {"loss": 1.2, "accuracy": 0.3, "samples": 10.0},
                {"loss": 999.0, "accuracy": 0.0, "samples": 10.0},
            ],
        ):
            run_dir = run_experiment(_config(1), Path(directory) / "run")
            final = json.loads((run_dir / "final_metrics.json").read_text())
            self.assertEqual(final["best_noisy_validation_accuracy"], 0.3)
            self.assertEqual(final["clean_test_accuracy"], 0.0)
            self.assertFalse(final["test_selection_leakage"])

    def test_cifar100_lightweight(self) -> None:
        train = _cifar(200, "train", 100)
        test = _cifar(100, "test", 100)
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar100",
            side_effect=lambda _root, split: train if split == "train" else test,
        ):
            run_dir = run_experiment(
                _config(1, dataset="cifar100"), Path(directory) / "run"
            )
            self.assertTrue((run_dir / "final_metrics.json").is_file())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cifar10_cuda_workflow(self) -> None:
        config = _config(1)
        config["trainer"]["device"] = "cuda"
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10", side_effect=self._load
        ):
            run_dir = run_experiment(config, Path(directory) / "run")
            payload = torch.load(
                run_dir / "last.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["upm_state"]["phase"], "completed")
            self.assertEqual(payload["upm_state"]["main_completed_epochs"], 1)
            self.assertTrue((run_dir / "final_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
