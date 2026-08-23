from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from lnl_toolbox.algorithms.pcse.config import PCSEFeatureLayerConfig
from lnl_toolbox.algorithms.pcse.features import collect_pcse_features
from lnl_toolbox.data.noisy_dataset import NoisyTargetDataset
from lnl_toolbox.data.torch_cifar import TorchCifarDataset
from lnl_toolbox.noise.generators import generate_symmetric
from lnl_toolbox.noise.manifest import fingerprint_labels
from lnl_toolbox.training.checkpoint import atomic_save
from lnl_toolbox.training.experiment import build_model
from lnl_toolbox.training.noisy_labels import file_sha256
from lnl_toolbox.training.pcse_pretrained import load_upm_main_best_source


class _CifarFixture:
    def __init__(self, size: int) -> None:
        self.images = np.zeros((size, 32, 32, 3), dtype=np.uint8)
        self.labels = np.arange(size, dtype=np.int64) % 10

    def __len__(self) -> int:
        return int(self.labels.size)


def _source(directory: Path) -> tuple[dict, torch.nn.Module]:
    model_config = {"name": "resnet18", "base_width": 16}
    model = build_model(model_config, 10)
    labels = np.arange(20, dtype=np.int64) % 10
    manifest = generate_symmetric(
        labels, 10, 0.4, 1, "cifar10", sampling="per_class"
    )
    manifest.global_indices = np.arange(labels.size, dtype=np.int64)
    manifest.dataset_fingerprint = fingerprint_labels(labels)
    manifest_path = directory / "noise_manifest.npz"
    manifest.save(manifest_path)
    noise = {
        "dataset": "cifar10",
        "num_classes": 10,
        "mapping_hash": manifest.mapping_hash,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "manifest_sha256": file_sha256(manifest_path),
    }
    payload = {
        "method": "upm",
        "checkpoint_role": "main_best",
        "config": {"upm": {"main": {"model": model_config}}},
        "noise": noise,
        "upm_state": {
            "main_completed_epochs": 3,
            "main_global_step": 9,
            "main_best_epoch": 1,
            "main_best_validation_accuracy": 0.5,
        },
        "best_main_model_state": model.state_dict(),
    }
    checkpoint_path = directory / "best.pt"
    atomic_save(payload, checkpoint_path)
    config = {
        "adapter": "upm_main_best",
        "run_directory_env": "PCSE_TEST_SOURCE",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "manifest_sha256": file_sha256(manifest_path),
        "mapping_hash": manifest.mapping_hash,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "model": model_config,
    }
    return config, model


class PCSEPretrainedAdapterTest(unittest.TestCase):
    def test_valid_source_and_identity_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _source(root)
            with mock.patch.dict(os.environ, {"PCSE_TEST_SOURCE": directory}):
                source = load_upm_main_best_source(
                    config, build_model(config["model"], 10), num_classes=10
                )
                self.assertEqual(source.checkpoint.sha256, config["checkpoint_sha256"])
                source.assert_unchanged()
                original = (root / "noise_manifest.npz").read_bytes()
                (root / "noise_manifest.npz").write_bytes(original + b"changed")
                with self.assertRaisesRegex(RuntimeError, "immutable source changed"):
                    source.assert_unchanged()
                (root / "noise_manifest.npz").write_bytes(original)
                for key, message in (
                    ("checkpoint_sha256", "checkpoint SHA-256"),
                    ("manifest_sha256", "manifest SHA-256"),
                    ("mapping_hash", "mapping hash"),
                    ("dataset_fingerprint", "dataset fingerprint"),
                ):
                    invalid = {**config, key: "f" * 64}
                    with self.assertRaisesRegex(ValueError, message):
                        load_upm_main_best_source(
                            invalid,
                            build_model(config["model"], 10),
                            num_classes=10,
                        )

    def test_wrong_architecture_classes_role_and_state_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _source(root)
            with mock.patch.dict(os.environ, {"PCSE_TEST_SOURCE": directory}):
                with self.assertRaisesRegex(ValueError, "num_classes"):
                    load_upm_main_best_source(
                        config, build_model(config["model"], 10), num_classes=3
                    )
                wrong = {**config, "model": {"name": "resnet34", "base_width": 16}}
                with self.assertRaisesRegex(ValueError, "architecture"):
                    load_upm_main_best_source(
                        wrong, build_model(wrong["model"], 10), num_classes=10
                    )
                payload = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
                payload["checkpoint_role"] = "last"
                atomic_save(payload, root / "best.pt")
                role_config = {**config, "checkpoint_sha256": file_sha256(root / "best.pt")}
                with self.assertRaisesRegex(ValueError, "main_best"):
                    load_upm_main_best_source(
                        role_config,
                        build_model(config["model"], 10),
                        num_classes=10,
                    )
                payload["checkpoint_role"] = "main_best"
                payload["best_main_model_state"] = {
                    "not_a_real_parameter": torch.ones(1)
                }
                atomic_save(payload, root / "best.pt")
                state_config = {
                    **config,
                    "checkpoint_sha256": file_sha256(root / "best.pt"),
                }
                with self.assertRaisesRegex(ValueError, "state_dict"):
                    load_upm_main_best_source(
                        state_config,
                        build_model(config["model"], 10),
                        num_classes=10,
                    )

    def test_real_resnet_layers_are_gap_pooled_and_logits_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _source(root)
            with mock.patch.dict(os.environ, {"PCSE_TEST_SOURCE": directory}):
                model = build_model(config["model"], 10)
                load_upm_main_best_source(config, model, num_classes=10)
            data = _CifarFixture(20)
            dataset = NoisyTargetDataset(
                TorchCifarDataset(data),
                np.arange(20, dtype=np.int64),
                np.arange(20, dtype=np.int64) % 10,
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=5)
            result = collect_pcse_features(
                model,
                loader,
                "cpu",
                dataset="cifar10",
                split="train",
                layers=(
                    PCSEFeatureLayerConfig("layer3", "global_average"),
                    PCSEFeatureLayerConfig("layer4", "global_average"),
                ),
            )
            self.assertEqual(result.snapshots[0].features.shape, (20, 64))
            self.assertEqual(result.snapshots[1].features.shape, (20, 128))
            with self.assertRaisesRegex(ValueError, "model logits output"):
                collect_pcse_features(
                    model,
                    loader,
                    "cpu",
                    dataset="cifar10",
                    split="train",
                    layers=(
                        PCSEFeatureLayerConfig("classifier"),
                        PCSEFeatureLayerConfig("layer4"),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
