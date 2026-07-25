import json
import os
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.training.experiment import _validate_resume_config, run_experiment
from lnl_toolbox.training.noisy_labels import prepare_noise_manifest


def _cifar(size: int, split: str) -> CifarData:
    labels = np.arange(size, dtype=np.int64) % 10
    images = np.zeros((size, 32, 32, 3), dtype=np.uint8)
    return CifarData(images, labels, tuple(map(str, range(10))), split, "cifar10")


def _config(epochs: int = 1) -> dict:
    return {
        "seed": 7,
        "data": {
            "name": "cifar10", "root": "unused", "validation_size": 10,
            "max_train_samples": 20, "max_validation_samples": 10,
            "max_test_samples": 10, "augment": False,
        },
        "noise": {"name": "symmetric", "rate": 0.4, "seed": 17},
        "loss": {"name": "ce"},
        "loader": {"batch_size": 10, "num_workers": 0, "pin_memory": False},
        "model": {"name": "tiny_cnn", "width": 4},
        "optimizer": {"name": "adamw", "lr": 0.001},
        "scheduler": {"name": "cosine", "t_max": 2},
        "trainer": {"epochs": epochs, "device": "cpu"},
    }


class NoisyCeBaselineTest(unittest.TestCase):
    def test_relative_output_root_is_resolved_before_manifest_metadata(self) -> None:
        config = _config()
        config["output_root"] = "runs"
        train_data = _cifar(40, "train")
        test_data = _cifar(20, "test")

        def load_data(_root, split):
            return train_data if split == "train" else test_data

        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.experiment.load_cifar10", side_effect=load_data
        ), patch(
            "lnl_toolbox.training.experiment.evaluate_classification",
            return_value={"loss": 1.0, "accuracy": 0.25, "samples": 10.0},
        ):
            previous = Path.cwd()
            try:
                os.chdir(directory)
                run_dir = run_experiment(config)
            finally:
                os.chdir(previous)

            self.assertTrue(run_dir.is_absolute())
            self.assertEqual(run_dir.parent, (Path(directory) / "runs").resolve())
            self.assertTrue((run_dir / "noise_manifest.npz").is_file())
            self.assertTrue((run_dir / "noise_summary.json").is_file())

    def test_unconnected_transition_estimator_configuration_is_rejected(self) -> None:
        config = _config()
        config["transition_estimator"] = {"name": "anchor"}
        with self.assertRaisesRegex(
            ValueError, r"field 'transition_estimator'.*not connected"
        ):
            run_experiment(config)

    def test_noisy_training_writes_manifest_metadata_and_clean_evaluation_sets(self) -> None:
        train_data = _cifar(40, "train")
        test_data = _cifar(20, "test")
        observed_evaluation_targets = []

        def load_data(_root, split):
            return train_data if split == "train" else test_data

        def evaluate(model, loader, criterion, device):
            sample = loader.dataset[0]
            observed_evaluation_targets.append((sample["index"], sample["target"]))
            return {"loss": 1.0, "accuracy": 0.25, "samples": float(len(loader.dataset))}

        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.experiment.load_cifar10", side_effect=load_data
        ), patch("lnl_toolbox.training.experiment.evaluate_classification", side_effect=evaluate):
            run_dir = run_experiment(_config(), directory)
            manifest = NoiseManifest.load(run_dir / "noise_manifest.npz")
            checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
            epoch_rows = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("event") == "epoch"
            ]

        self.assertEqual(set(checkpoint["noise"]), {
            "mode", "manifest_path", "manifest_version", "manifest_sha256", "mapping_hash", "dataset", "split",
            "dataset_fingerprint", "noise_type", "requested_rate", "seed", "num_classes",
            "manifest_actual_rate", "effective_train_subset_actual_rate",
            "validation_targets", "effective_validation_subset_actual_rate",
            "has_transition_matrix", "has_per_sample_transition",
        })
        self.assertEqual(checkpoint["noise"]["mapping_hash"], manifest.mapping_hash)
        self.assertEqual(final["noise"], checkpoint["noise"])
        self.assertEqual(checkpoint["config"]["selector"], {"name": "all"})
        self.assertEqual(epoch_rows[0]["selected_ratio"], 1.0)
        self.assertEqual(manifest.global_indices.size, 30)
        self.assertEqual(len(observed_evaluation_targets), 2)
        for index, target in observed_evaluation_targets:
            self.assertEqual(target, index % 10)

    def test_paper_mode_can_apply_one_manifest_to_train_and_validation(self) -> None:
        config = _config()
        config["noise"]["validation_targets"] = "noisy"
        train_data = _cifar(40, "train")
        test_data = _cifar(20, "test")
        observed = []

        def load_data(_root, split):
            return train_data if split == "train" else test_data

        def evaluate(model, loader, criterion, device):
            sample = loader.dataset[0]
            observed.append((sample["index"], sample["target"]))
            return {
                "loss": 1.0,
                "accuracy": 0.25,
                "samples": float(len(loader.dataset)),
            }

        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.experiment.load_cifar10", side_effect=load_data
        ), patch(
            "lnl_toolbox.training.experiment.evaluate_classification",
            side_effect=evaluate,
        ):
            run_dir = run_experiment(config, directory)
            manifest = NoiseManifest.load(run_dir / "noise_manifest.npz")
            summary = json.loads(
                (run_dir / "noise_summary.json").read_text(encoding="utf-8")
            )

        mapping = dict(zip(manifest.global_indices, manifest.noisy_targets))
        validation_index, validation_target = observed[0]
        test_index, test_target = observed[1]
        self.assertEqual(manifest.global_indices.size, 40)
        self.assertEqual(validation_target, mapping[validation_index])
        self.assertEqual(test_target, test_index % 10)
        self.assertEqual(summary["validation_targets"], "noisy")
        self.assertIsNotNone(
            summary["effective_validation_subset_actual_rate"]
        )

    def test_resume_reuses_manifest_and_rejects_hash_mismatch(self) -> None:
        config = _config()
        indices = np.arange(20, dtype=np.int64)
        targets = indices % 10
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            manifest, path = prepare_noise_manifest(
                config, dataset="cifar10", clean_targets=targets, global_indices=indices,
                num_classes=10, run_dir=run_dir,
            )
            metadata = {
                "manifest_path": path.name, "mapping_hash": manifest.mapping_hash,
                "dataset": manifest.dataset, "split": manifest.split,
                "dataset_fingerprint": manifest.dataset_fingerprint,
                "noise_type": manifest.noise_type, "requested_rate": manifest.requested_rate,
                "seed": manifest.seed, "num_classes": manifest.num_classes,
                "manifest_actual_rate": manifest.actual_rate,
            }
            loaded, _ = prepare_noise_manifest(
                config, dataset="cifar10", clean_targets=targets, global_indices=indices,
                num_classes=10, run_dir=run_dir, checkpoint_payload={"noise": metadata},
            )
            self.assertEqual(loaded.mapping_hash, manifest.mapping_hash)
            metadata["mapping_hash"] = "tampered"
            with self.assertRaisesRegex(ValueError, "mapping hash"):
                prepare_noise_manifest(
                    config, dataset="cifar10", clean_targets=targets, global_indices=indices,
                    num_classes=10, run_dir=run_dir, checkpoint_payload={"noise": metadata},
                )

    def test_resume_requires_existing_manifest(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            metadata = {"manifest_path": "noise_manifest.npz", "mapping_hash": "missing"}
            with self.assertRaises(FileNotFoundError):
                prepare_noise_manifest(
                    config, dataset="cifar10", clean_targets=np.arange(10),
                    global_indices=np.arange(10), num_classes=10, run_dir=Path(directory),
                    checkpoint_payload={"noise": metadata},
                )

    def test_noisy_runner_accepts_configured_gce_loss(self) -> None:
        config = _config()
        config["loss"] = {"name": "gce", "q": 0.7}
        train_data = _cifar(40, "train")
        test_data = _cifar(20, "test")

        def load_data(_root, split):
            return train_data if split == "train" else test_data

        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.experiment.load_cifar10", side_effect=load_data
        ), patch(
            "lnl_toolbox.training.experiment.evaluate_classification",
            return_value={"loss": 1.0, "accuracy": 0.25, "samples": 10.0},
        ):
            run_dir = run_experiment(config, directory)
            checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["loss"], {"name": "gce", "q": 0.7})

    def test_small_loss_selector_is_applied_and_resume_config_is_checked(self) -> None:
        config = _config()
        config["selector"] = {"name": "small_loss", "keep_rate": 0.5}
        train_data = _cifar(40, "train")
        test_data = _cifar(20, "test")

        def load_data(_root, split):
            return train_data if split == "train" else test_data

        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.experiment.load_cifar10", side_effect=load_data
        ), patch(
            "lnl_toolbox.training.experiment.evaluate_classification",
            return_value={"loss": 1.0, "accuracy": 0.25, "samples": 10.0},
        ):
            run_dir = run_experiment(config, directory)
            checkpoint = torch.load(
                run_dir / "last.pt", map_location="cpu", weights_only=False
            )
            rows = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
        epoch_row = next(row for row in rows if row.get("event") == "epoch")
        self.assertEqual(epoch_row["selected_samples"], 10.0)
        self.assertEqual(epoch_row["selected_ratio"], 0.5)
        self.assertIn("train_all_sample_loss", epoch_row)
        self.assertEqual(checkpoint["config"]["selector"], config["selector"])

        changed = _config()
        changed["selector"] = {"name": "small_loss", "keep_rate": 0.75}
        with self.assertRaisesRegex(ValueError, "selector"):
            _validate_resume_config(changed, config)

    def test_resume_rejects_every_linear_schedule_configuration_change(self) -> None:
        saved = _config()
        saved["selector"] = {
            "name": "small_loss",
            "keep_rate": {
                "name": "linear",
                "start": 1.0,
                "end": 0.6,
                "warmup_epochs": 10,
            },
        }
        _validate_resume_config(deepcopy(saved), saved)
        changes = {
            "name": "constant",
            "start": 0.9,
            "end": 0.5,
            "warmup_epochs": 11,
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                current = deepcopy(saved)
                current["selector"]["keep_rate"][key] = value
                with self.assertRaisesRegex(ValueError, "selector"):
                    _validate_resume_config(current, saved)

    def test_resume_rejects_preprocessing_or_validation_target_change(self) -> None:
        saved = _config()
        saved["data"]["preprocessing"] = "gce2018"
        saved["noise"]["validation_targets"] = "noisy"
        changed = deepcopy(saved)
        changed["data"]["preprocessing"] = "standard"
        with self.assertRaisesRegex(ValueError, "data.preprocessing"):
            _validate_resume_config(changed, saved)
        changed = deepcopy(saved)
        changed["noise"]["validation_targets"] = "clean"
        with self.assertRaisesRegex(ValueError, "noise.validation_targets"):
            _validate_resume_config(changed, saved)


if __name__ == "__main__":
    unittest.main()
