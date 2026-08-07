from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lnl_toolbox.algorithms.dld.precorrection import DLDPartitionResult
from lnl_toolbox.algorithms.dld.state import DLDPhase, DLDState
from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.dld_experiment import run_dld_experiment


def _data(split: str, samples_per_class: int, classes: int = 10) -> CifarData:
    labels = np.repeat(np.arange(classes, dtype=np.int64), samples_per_class)
    rng = np.random.default_rng(91 if split == "train" else 92)
    images = rng.integers(0, 256, (labels.size, 32, 32, 3), dtype=np.uint8)
    dataset = "cifar10" if classes == 10 else "cifar100"
    return CifarData(images, labels, tuple(str(i) for i in range(classes)), split, dataset)


def _config(epochs: int) -> dict:
    return {
        "method": "dld", "execution": {"runner": "dld"}, "seed": 5,
        "data": {
            "name": "cifar10", "root": "unused", "validation_size": 10,
            "max_train_samples": 20, "max_validation_samples": 10,
            "max_test_samples": 10,
        },
        "loader": {"batch_size": 10, "num_workers": 0, "pin_memory": False},
        "noise": {"name": "symmetric", "rate": 0.2, "seed": 8, "validation_targets": "noisy"},
        "dld": {
            "fidelity": {
                "name": "paper_oriented_v1", "hard_y0": "averaged_views",
                "direction_endpoint": "estimated_yn", "direction": "yn_minus_y0",
                "neighbor_metric": "cosine_distance", "self_neighbor": "include",
                "divergence": "kl_ps_to_pw", "divergence_softmax": False,
                "hard_yn_zero_denominator": "fail", "schedule": "average",
                "inference_initialization": "zero", "inference_steps": 5,
            },
            "feature_extractor": {"source": "repository_frozen_model", "model": {"name": "tiny_cnn", "width": 4}},
            "precorrection": {"k_neighbors": 3, "delta": 1e-6, "gmm_components": 2, "gmm_seed": 0, "minimum_mean_separation": 0.0},
            "diffusion": {
                "timesteps": 5, "epochs": epochs,
                "model": {"independent_predictors": True, "hidden_dim": 8, "time_dim": 4},
                "optimizer": {"direction": {"name": "adam", "lr": 1e-3}, "noise": {"name": "adam", "lr": 1e-3}},
                "scheduler": {"direction": {"name": "none"}, "noise": {"name": "none"}},
                "ema": {"enabled": True, "decay": 0.9},
            },
            "inference": {"steps": 5, "deterministic": True, "initialization": "zero"},
        },
        "evaluation": {"selection_split": "validation", "primary": "accuracy"},
        "trainer": {"device": "cpu"}, "output_root": "unused",
    }


def _partition(p_w, p_s, targets, **kwargs):
    p_ws = ((p_w + p_s) / 2).detach()
    partition = torch.arange(p_w.shape[0], device=p_w.device) % 2
    divergence = torch.linspace(0.01, 1.0, p_w.shape[0], device=p_w.device, dtype=p_w.dtype)
    return DLDPartitionResult(divergence, partition, p_ws, 0.1, 0.9)


class DLDWorkflowTest(unittest.TestCase):
    def test_state_rejects_illegal_transition(self) -> None:
        with self.assertRaisesRegex(ValueError, "illegal"):
            DLDState().advance(DLDPhase.DIFFUSION_TRAINING)

    def test_fresh_resume_extension_and_completed_noop(self) -> None:
        train, test = _data("train", 4), _data("test", 2)
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.dld_experiment.load_cifar10",
            side_effect=lambda root, split: train if split == "train" else test,
        ), patch("lnl_toolbox.training.dld_experiment.partition_samples", side_effect=_partition):
            run = Path(directory) / "run"
            run_dld_experiment(_config(1), run)
            first = read_checkpoint(run / "last.pt", "cpu")
            self.assertEqual(first["dld_state"]["phase"], "completed")
            artifact = run / "dld_precorrection.npz"
            hash_before = first["dld_state"]["precorrection_artifact_hash"]
            mtime_before = artifact.stat().st_mtime_ns
            run_dld_experiment(_config(2), resume=run / "last.pt")
            second = read_checkpoint(run / "last.pt", "cpu")
            self.assertEqual(second["dld_state"]["completed_epochs"], 2)
            self.assertGreater(second["dld_state"]["global_step"], first["dld_state"]["global_step"])
            self.assertEqual(second["dld_state"]["precorrection_artifact_hash"], hash_before)
            self.assertEqual(artifact.stat().st_mtime_ns, mtime_before)
            self.assertIn("direction_model", second["algorithm"])
            self.assertIn("noise_model", second["algorithm"])
            drifted = _config(2)
            drifted["dld"]["precorrection"]["k_neighbors"] = 4
            with self.assertRaisesRegex(ValueError, "identity"):
                run_dld_experiment(drifted, resume=run / "last.pt")
            with self.assertRaisesRegex(ValueError, "cannot be reduced"):
                run_dld_experiment(_config(1), resume=run / "last.pt")
            files = {name: (run / name).stat().st_mtime_ns for name in ("last.pt", "best.pt", "metrics.jsonl", "dld_precorrection.npz")}
            run_dld_experiment(_config(2), resume=run / "last.pt")
            self.assertEqual(files, {name: (run / name).stat().st_mtime_ns for name in files})
            final = json.loads((run / "final_metrics.json").read_text())
            self.assertFalse(final["test_selection_leakage"])
            for name in ("last.pt", "best.pt", "metrics.jsonl", "final_metrics.json", "noise_manifest.npz"):
                self.assertTrue((run / name).is_file(), name)

    def test_cifar100_dispatch_is_supported(self) -> None:
        config = _config(1)
        config["data"]["name"] = "cifar100"
        config["data"].update({
            "validation_size": 100, "max_train_samples": 100,
            "max_validation_samples": 100, "max_test_samples": 100,
        })
        config["loader"]["batch_size"] = 100
        train, test = _data("train", 2, 100), _data("test", 1, 100)
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.dld_experiment.load_cifar100",
            side_effect=lambda root, split: train if split == "train" else test,
        ), patch("lnl_toolbox.training.dld_experiment.partition_samples", side_effect=_partition):
            run = run_dld_experiment(config, Path(directory) / "cifar100")
            final = json.loads((run / "final_metrics.json").read_text())
            self.assertEqual(final["completed_epochs"], 1)


if __name__ == "__main__":
    unittest.main()
