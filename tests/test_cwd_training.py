import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.training.cwd_experiment import run_cwd_experiment


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
                "lnl_toolbox.data.sources.load_cifar10",
                side_effect=lambda _root, split: train if split == "train" else test,
            ):
                result = run_cwd_experiment(config, directory)
            for name in (
                "last.pt",
                "noise_manifest.npz",
                "feature_snapshot.npz",
                "statistic_artifact.npz",
                "metrics.jsonl",
                "training_curves.svg",
                "resolved_config.yaml",
            ):
                self.assertTrue((result / name).is_file(), name)
            payload = torch.load(
                result / "last.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["completed_epoch"], 0)
            self.assertTrue(payload["statistic_hash"])
            self.assertTrue(payload["feature_snapshot_hash"])


if __name__ == "__main__":
    unittest.main()
