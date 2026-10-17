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
from lnl_toolbox.training.service import ExperimentReadinessError, ExperimentService


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

    def test_failure_projection_retains_order_and_backend_status_without_second_validation(self):
        from lnl_toolbox.training.compatibility import CompatibilityResult, CompatibilityStatus
        from lnl_toolbox.training.prerequisites import SourceDescriptor, ValidationMetadata, ReadinessLevel

        first = SourceDescriptor("a", "pretrained_classifier", "Classifier", "trained model", "train source", "set source", "fixture")
        second = SourceDescriptor("b", "external_label_artifact", "Labels", "aligned labels", "obtain labels", "set labels", "fixture")
        result = CompatibilityResult(
            CompatibilityStatus.INCOMPATIBLE, "fixture", "fixture",
            prerequisites=(ValidationMetadata.needs_input(first, "source missing"),
                           ValidationMetadata.invalid(second, ReadinessLevel.PATH_EXISTS, "schema rejected")),
        )
        with patch.object(self.service, "_resolve_for_capabilities", return_value=result) as evaluate, \
             patch.object(self.service, "prerequisite_readiness") as recheck, \
             patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
            with self.assertRaises(ExperimentReadinessError) as raised:
                self.service.run(self.execution_config())
        self.assertIs(raised.exception.compatibility, result)
        self.assertIsInstance(raised.exception, ValueError)
        self.assertEqual([item.status for item in result.prerequisites], ["needs_input", "invalid"])
        self.assertEqual(str(raised.exception), "method 'fixture' is not ready for dataset 'fixture': incompatible; ")
        evaluate.assert_called_once()
        recheck.assert_not_called()
        runner.assert_not_called()

    def test_early_failure_preserves_exception_when_diagnostic_snapshot_fails(self):
        config, _ = self.cal_execution_config()
        for original in (FileNotFoundError("original missing path"), ValueError("original schema error")):
            with self.subTest(error=type(original).__name__), \
                 patch("lnl_toolbox.catalog.validate_config", side_effect=original), \
                 patch.object(self.service, "_config_value", side_effect=RuntimeError("diagnostic lookup failed")), \
                 patch.object(self.service.prerequisites, "evaluate") as evaluate, \
                 patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
                with self.assertRaises(type(original)) as raised:
                    self.service.run(config)
                self.assertIs(raised.exception, original)
                self.assertNotIn("diagnostic lookup", str(raised.exception))
                evaluate.assert_not_called()
                runner.assert_not_called()

    def test_plain_validate_does_not_add_dld_or_cal_readiness_checks(self):
        for recipe in ("dld-cifar10-reproduction", "cal-cifar10-reproduction"):
            with self.subTest(recipe=recipe), patch.object(self.service.prerequisites, "evaluate") as evaluate:
                self.service.preflight(load_recipe_config(recipe_by_id(recipe)), check_data=False)
                evaluate.assert_not_called()

    def test_cal_failure_retains_source_and_runs_consumer_validation_once(self):
        config, artifact = self.cal_execution_config()
        torch.save({"clean": torch.arange(20) % 10}, artifact)
        validate = self.service.data_service.validate_cal_external_labels
        with patch.object(self.service.data_service, "validate_cal_external_labels", wraps=validate) as consumer, \
             patch("lnl_toolbox.training.runners.RunnerSpec.invoke") as runner:
            with self.assertRaises(ExperimentReadinessError) as raised:
                self.service.run(config)
        consumer.assert_called_once()
        runner.assert_not_called()
        self.assertIn("configured clean/noisy keys", str(raised.exception))
        self.assertEqual(raised.exception.compatibility.prerequisites[0].status, "invalid")
        self.assertEqual(raised.exception.prerequisite_context[0]["configured"][0]["value"], str(artifact))

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
