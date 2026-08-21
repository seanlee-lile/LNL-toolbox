from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.dld import DLDConfig
from lnl_toolbox.noise.generators import generate_symmetric
from lnl_toolbox.noise.manifest import fingerprint_labels
from lnl_toolbox.training.checkpoint import atomic_save
from lnl_toolbox.training.dld_pretrained import load_upm_main_best_feature_source
from lnl_toolbox.training.experiment import build_model
from lnl_toolbox.training.noisy_labels import file_sha256


ROOT = Path(__file__).resolve().parents[1]


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
    return {
        "adapter": "upm_main_best",
        "run_directory_env": "DLD_TEST_SOURCE",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "manifest_sha256": file_sha256(manifest_path),
        "mapping_hash": manifest.mapping_hash,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "model": model_config,
    }, model


class DLDReadinessTest(unittest.TestCase):
    def test_formal_config_uses_current_contract_and_full_budget(self) -> None:
        path = ROOT / "configs/experiment/dld_cifar10_reproduction.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        parsed = DLDConfig.from_mapping(config)
        self.assertEqual(config["configuration_fidelity"], "paper_oriented")
        self.assertEqual(config["loader"]["batch_size"], 200)
        self.assertEqual(parsed.precorrection["k_neighbors"], 50)
        self.assertEqual(parsed.diffusion["timesteps"], 1000)
        self.assertEqual(parsed.epochs, 200)

    def test_real_short_config_is_full_data_external_sym20(self) -> None:
        path = ROOT / "configs" / "reproduction" / "cifar10_dld_sym20_short.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        parsed = DLDConfig.from_mapping(config)
        self.assertEqual(config["method"], "dld")
        self.assertEqual(config["data"]["name"], "cifar10")
        for name in ("max_train_samples", "max_validation_samples", "max_test_samples"):
            self.assertNotIn(name, config["data"])
        self.assertEqual(config["noise"]["rate"], 0.2)
        self.assertEqual(
            parsed.fidelity["name"], "paper_oriented_v2_cosine_similarity"
        )
        self.assertEqual(parsed.fidelity["neighbor_metric"], "cosine_similarity")
        self.assertEqual(
            parsed.fidelity["neighbor_weighting"], "inverse_neighbor_value"
        )
        self.assertEqual(parsed.feature_extractor["source"], "external_checkpoint")
        self.assertEqual(parsed.precorrection["query_chunk_size"], 64)
        self.assertEqual(parsed.epochs, 15)
        legacy = yaml.safe_load(path.read_text(encoding="utf-8"))
        legacy["dld"]["fidelity"]["name"] = "paper_oriented_v1"
        legacy["dld"]["fidelity"]["neighbor_metric"] = "cosine_distance"
        legacy["dld"]["fidelity"].pop("neighbor_weighting")
        with self.assertRaisesRegex(ValueError, "fidelity"):
            DLDConfig.from_mapping(legacy)

    def test_external_source_is_strict_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _source(root)
            with mock.patch.dict(os.environ, {"DLD_TEST_SOURCE": directory}):
                source = load_upm_main_best_feature_source(
                    config, build_model(config["model"], 10), num_classes=10
                )
                self.assertEqual(source.provenance["adapter"], "upm_main_best")
                source.assert_unchanged()
                original = (root / "noise_manifest.npz").read_bytes()
                (root / "noise_manifest.npz").write_bytes(original + b"changed")
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    source.assert_unchanged()

    def test_external_source_rejects_identity_and_role_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _source(root)
            with mock.patch.dict(os.environ, {"DLD_TEST_SOURCE": directory}):
                invalid = {**config, "checkpoint_sha256": "f" * 64}
                with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                    load_upm_main_best_feature_source(
                        invalid, build_model(config["model"], 10), num_classes=10
                    )
                payload = torch.load(
                    root / "best.pt", map_location="cpu", weights_only=False
                )
                payload["checkpoint_role"] = "last"
                atomic_save(payload, root / "best.pt")
                role_config = {
                    **config,
                    "checkpoint_sha256": file_sha256(root / "best.pt"),
                }
                with self.assertRaisesRegex(ValueError, "main_best"):
                    load_upm_main_best_feature_source(
                        role_config,
                        build_model(config["model"], 10),
                        num_classes=10,
                    )


if __name__ == "__main__":
    unittest.main()
