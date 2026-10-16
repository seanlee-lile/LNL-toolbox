from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from lnl_toolbox.catalog import load_recipe_config, recipe_by_id
from lnl_toolbox.data.contracts import (
    DataSpec,
    RawDatasetSplit,
    UnsupportedDatasetSplitError,
)
from lnl_toolbox.data.local_catalog import LocalDatasetCatalog
from lnl_toolbox.data.registry import DatasetRegistry
from lnl_toolbox.training.data_service import DataService
from lnl_toolbox.training.service import ExperimentService


class _Adapter:
    name = "cifar10"
    aliases = ()

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError(spec.root)

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del spec, seed
        if split == "validation":
            raise UnsupportedDatasetSplitError("split must be train or test")
        count = 20 if split == "train" else 10
        labels = np.arange(count, dtype=np.int64) % 10
        return RawDatasetSplit(
            np.zeros((count, 32, 32, 3), dtype=np.uint8), labels,
            np.arange(count, dtype=np.int64), self.name, split, 10,
            clean_targets=labels, source="fixture",
        )


class ExperimentServiceDataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        data_root = root / "data"
        data_root.mkdir()
        data_service = DataService(
            registry=DatasetRegistry((_Adapter(),)),
            catalog=LocalDatasetCatalog(root / "catalog.json"),
        )
        data_service.register("local-cifar10", "cifar10", {"root": str(data_root)})
        self.service = ExperimentService(data_service=data_service)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def execution_config(self) -> dict:
        return {
            "data": {
                "name": "cifar10", "root": str(Path(self.temp.name) / "data"),
                "num_classes": 10, "validation_size": 0,
            },
            "execution": {"runner": "supervised"},
        }

    def cal_execution_config(self) -> tuple[dict, Path]:
        config = load_recipe_config(recipe_by_id("cal-cifar10-reproduction"))
        config["data"] = self.execution_config()["data"]
        artifact = Path(self.temp.name) / "execution-labels.pt"
        clean = torch.arange(20) % 10
        torch.save({"clean": clean, "noisy": (clean + 1) % 10}, artifact)
        config["noise"].update(
            path=str(artifact), clean_key="clean", noisy_key="noisy"
        )
        return config, artifact

    def unfinished_run(self, config: dict) -> Path:
        root = Path(self.temp.name) / "run"
        root.mkdir(exist_ok=True)
        (root / "resolved_config.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
        (root / "environment.json").write_text("{}", encoding="utf-8")
        # The runner boundary is mocked: these bytes must never be loaded or
        # rewritten by Application admission, which owns no checkpoint schema.
        for name in ("last.pt", "best.pt"):
            (root / name).write_bytes(b"runner-owned checkpoint")
        return root

    def test_direct_run_admits_valid_context_without_writing_before_runner(self):
        config = self.execution_config()
        output = Path(self.temp.name) / "not-created"
        with patch("lnl_toolbox.training.runners.RunnerSpec.invoke", return_value=output) as runner:
            self.assertEqual(self.service.run(config, output), output)
        runner.assert_called_once()
        self.assertIsNone(runner.call_args.args[2])
        self.assertFalse(output.exists())

    def test_direct_run_rejects_invalid_config_and_current_dataset(self):
        for change in ("config", "dataset"):
            with self.subTest(change=change):
                config = self.execution_config()
                if change == "config":
                    config["trainer"] = {"epochs": 0}
                else:
                    config["data"]["root"] += "-missing"
                with patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        self.service.run(config)
                runner.assert_not_called()

    def test_direct_run_rejects_needs_input_and_invalid_prerequisites(self):
        config, artifact = self.cal_execution_config()
        cases = ("needs_input", "schema", "identity")
        for case in cases:
            with self.subTest(case=case):
                clean = torch.arange(20) % 10
                torch.save({"clean": clean, "noisy": clean}, artifact)
                if case == "needs_input":
                    candidate = {**config, "noise": {"name": "external_torch"}}
                else:
                    candidate = config
                    payload = {"clean": clean} if case == "schema" else {
                        "clean": (clean + 1) % 10, "noisy": clean,
                    }
                    torch.save(payload, artifact)
                with patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
                    with self.assertRaises(ValueError):
                        self.service.run(candidate)
                runner.assert_not_called()

    def test_run_refreshes_ready_after_artifact_deletion_or_content_change(self):
        config, artifact = self.cal_execution_config()
        for change in ("deleted", "schema", "identity"):
            with self.subTest(change=change):
                service = ExperimentService(data_service=self.service.data_service)
                clean = torch.arange(20) % 10
                torch.save({"clean": clean, "noisy": clean}, artifact)
                service.preflight(config, check_data=True)
                self.assertEqual(service.last_compatibility.status.value, "compatible")
                if change == "deleted":
                    artifact.unlink()
                elif change == "schema":
                    torch.save({"clean": clean}, artifact)
                else:
                    torch.save({"clean": (clean + 1) % 10, "noisy": clean}, artifact)
                with patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
                    with self.assertRaises(ValueError):
                        service.run(config)
                runner.assert_not_called()

    def test_resume_rechecks_current_artifact_and_preserves_run_files(self):
        config, artifact = self.cal_execution_config()
        root = self.unfinished_run(config)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
        for change in ("deleted", "schema", "identity"):
            with self.subTest(change=change):
                service = ExperimentService(data_service=self.service.data_service)
                clean = torch.arange(20) % 10
                torch.save({"clean": clean, "noisy": clean}, artifact)
                service.preflight(config, check_data=True)
                if change == "deleted":
                    artifact.unlink()
                elif change == "schema":
                    torch.save({"clean": clean}, artifact)
                else:
                    torch.save({"clean": (clean + 1) % 10, "noisy": clean}, artifact)
                with patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
                    with self.assertRaises(ValueError):
                        service.resume(root)
                    with self.assertRaises(ValueError):
                        service.run(config, root, root / "last.pt")
                runner.assert_not_called()
        self.assertEqual(before, {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()
        })

    def test_resume_rejects_unavailable_dataset_before_runner(self):
        config = self.execution_config()
        root = self.unfinished_run(config)
        self.service.preflight(config, check_data=True)
        Path(config["data"]["root"]).rmdir()
        with patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
            with self.assertRaises(FileNotFoundError):
                self.service.resume(root)
        runner.assert_not_called()

    def test_valid_resume_keeps_directory_checkpoint_and_resolved_identity(self):
        config, _artifact = self.cal_execution_config()
        root = self.unfinished_run(config)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
        for selection in ("last", "best"):
            with self.subTest(selection=selection):
                with patch("lnl_toolbox.training.runners.RunnerSpec.invoke", return_value=root) as runner:
                    self.assertEqual(self.service.resume(root, selection), root.resolve())
                runner.assert_called_once()
                self.assertEqual(runner.call_args.args[1:], (root.resolve(), root / f"{selection}.pt"))
        self.assertEqual(before, {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()
        })

    def test_completed_resume_is_noop_even_when_current_data_is_missing(self):
        config = self.execution_config()
        config["data"]["root"] += "-missing"
        root = self.unfinished_run(config)
        (root / "final_metrics.json").write_text(json.dumps({"test_accuracy": 0.5}), encoding="utf-8")
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
        with patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
            self.assertEqual(self.service.resume(root), root.resolve())
        runner.assert_not_called()
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()})

    def test_execution_refreshes_environment_provided_feature_source(self):
        from lnl_toolbox.noise.generators import generate_symmetric
        from lnl_toolbox.noise.manifest import fingerprint_labels
        from lnl_toolbox.training.experiment import build_model
        from lnl_toolbox.training.noisy_labels import file_sha256

        root = Path(self.temp.name)
        model_config = {"name": "resnet18", "base_width": 16}
        model = build_model(model_config, 10)
        labels = np.arange(20, dtype=np.int64) % 10
        manifest = generate_symmetric(labels, 10, 0.4, 1, "cifar10", sampling="per_class")
        manifest.global_indices = np.arange(20, dtype=np.int64)
        manifest.dataset_fingerprint = fingerprint_labels(labels)
        manifest_path = root / "noise_manifest.npz"
        manifest.save(manifest_path)
        noise = {"dataset": "cifar10", "num_classes": 10,
                 "mapping_hash": manifest.mapping_hash,
                 "dataset_fingerprint": manifest.dataset_fingerprint,
                 "manifest_sha256": file_sha256(manifest_path)}
        checkpoint = root / "best.pt"
        torch.save({
            "method": "upm", "checkpoint_role": "main_best",
            "config": {"upm": {"main": {"model": model_config}}},
            "noise": noise,
            "upm_state": {"main_completed_epochs": 3, "main_global_step": 9,
                          "main_best_epoch": 1, "main_best_validation_accuracy": 0.5},
            "best_main_model_state": model.state_dict(),
        }, checkpoint)
        config = load_recipe_config(recipe_by_id("dld-cifar10-reproduction"))
        config["data"] = {**self.execution_config()["data"], "validation_size": 4}
        config["dld"]["feature_extractor"] = {
            "source": "external_checkpoint", "model": model_config,
            "external": {
                "adapter": "upm_main_best", "run_directory_env": "LNL_PHASE1_SOURCE",
                "checkpoint_sha256": file_sha256(checkpoint),
                "manifest_sha256": file_sha256(manifest_path),
                "mapping_hash": manifest.mapping_hash,
                "dataset_fingerprint": manifest.dataset_fingerprint,
                "model": model_config,
            },
        }
        for change in ("unset", "different_directory"):
            with self.subTest(change=change), patch.dict(os.environ, {"LNL_PHASE1_SOURCE": str(root)}):
                service = ExperimentService(data_service=self.service.data_service)
                service.preflight(config, check_data=True)
                self.assertEqual(service.last_compatibility.status.value, "compatible")
                if change == "unset":
                    del os.environ["LNL_PHASE1_SOURCE"]
                else:
                    os.environ["LNL_PHASE1_SOURCE"] = str(root / "missing-producer")
                with patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
                    with self.assertRaises(ValueError):
                        service.run(config)
                runner.assert_not_called()

    def test_run_and_resume_reject_dataset_method_incompatibility(self):
        config = load_recipe_config(recipe_by_id("importance-reweighting-binary-smoke"))
        config["data"] = {
            **self.execution_config()["data"],
            "train_size": 20, "validation_size": 4, "test_size": 10,
        }
        root = self.unfinished_run(config)
        with patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
            with self.assertRaisesRegex(ValueError, "wrong_class_count"):
                self.service.run(config)
            with self.assertRaisesRegex(ValueError, "wrong_class_count"):
                self.service.resume(root)
        runner.assert_not_called()

    def test_data_verify_uses_single_execution_admission(self):
        service = self.service.data_service
        config = self.execution_config()
        output = Path(self.temp.name) / "verified-run"

        def finish(_config, destination, _resume):
            destination = Path(destination)
            destination.mkdir()
            (destination / "metrics.jsonl").write_text(
                json.dumps({"event": "epoch", "epoch": 1, "test_accuracy": 0.5}) + "\n",
                encoding="utf-8",
            )
            (destination / "data_manifest.json").write_text(
                json.dumps({"fingerprint": "verified-fixture"}), encoding="utf-8"
            )
            return destination

        with patch.object(service, "validate_config", wraps=service.validate_config) as validate, patch(
            "lnl_toolbox.training.runners.RunnerSpec.invoke", side_effect=finish
        ) as runner:
            report, returned = service.verify("local-cifar10", config, output)
        self.assertEqual(returned, output.resolve())
        self.assertEqual(report.training_evidence["completed_epochs"], 1)
        runner.assert_called_once()
        validate.assert_called_once()

    def test_inspection_and_compatibility_use_read_only_data_contract(self) -> None:
        report = self.service.inspect_dataset("local-cifar10")
        self.assertEqual(report.status, "ready")
        config = {"data": {"name": "cifar10"}, "execution": {"runner": "supervised"}}
        result = self.service.resolve_method_compatibility("local-cifar10", config)
        self.assertIsNotNone(result.status)
        listed = self.service.list_config_compatibility("local-cifar10", {"fixture": config})
        self.assertEqual(listed[0][0], "fixture")

    def test_cal_external_artifact_requires_schema_and_clean_identity(self) -> None:
        root = Path(self.temp.name)
        artifact = root / "cal-labels.pt"
        clean = np.arange(20, dtype=np.int64) % 10
        noisy = clean.copy()
        noisy[0] = 1
        torch.save({"clean": torch.from_numpy(clean), "noisy": torch.from_numpy(noisy)}, artifact)
        config = {
            "seed": 0,
            "data": {"name": "cifar10"},
            "noise": {
                "path": str(artifact),
                "clean_key": "clean",
                "noisy_key": "noisy",
            },
        }
        value = self.service.data_service.validate_cal_external_labels(config)
        self.assertEqual(value["samples"], 20)
        self.assertEqual(
            value["clean_usage"], "identity_alignment_and_evaluation_only"
        )

        torch.save({"clean": torch.from_numpy(clean[::-1].copy()), "noisy": torch.from_numpy(noisy)}, artifact)
        with self.assertRaisesRegex(ValueError, "identity alignment"):
            self.service.data_service.validate_cal_external_labels(config)

        torch.save({"clean": torch.from_numpy(clean)}, artifact)
        with self.assertRaisesRegex(ValueError, "configured clean/noisy keys"):
            self.service.data_service.validate_cal_external_labels(config)


if __name__ == "__main__":
    unittest.main()
