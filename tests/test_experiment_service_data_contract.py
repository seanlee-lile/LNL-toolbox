from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

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
    def test_web_split_repartitions_the_original_pool_and_keeps_test_separate(self) -> None:
        from lnl_toolbox.data.contracts import DataRequirements, DataRole
        from lnl_toolbox.training.data_service import prepare_experiment_data

        class PoolAdapter:
            name, aliases = "pool_fixture", ()

            def validate(self, spec):
                pass

            def load(self, spec, split, *, seed):
                indices = {"train": np.arange(90), "validation": np.arange(90, 100), "test": np.arange(20)}[split]
                labels = indices % 2
                return RawDatasetSplit(
                    np.column_stack((indices, indices)).astype(np.float32),
                    labels, indices, self.name, split, 2, clean_targets=labels,
                )

        registry = DatasetRegistry((PoolAdapter(),))
        requirements = DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.CLEAN_VALIDATION, DataRole.TEST}),
            validation_targets="clean", needs_noise_manifest=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            initial = None
            for size in (20, 30, 20):
                config = {"data": {"name": "pool_fixture", "root": directory, "validation_size": size,
                                   "validation_split": {"source": "training_pool"}},
                          "loader": {"batch_size": 8, "num_workers": 0}}
                prepared = prepare_experiment_data(config, requirements=requirements,
                    run_dir=Path(directory) / str(size), seed=1, registry=registry)
                self.assertEqual(prepared.train_indices.size, 100 - size)
                self.assertEqual(prepared.validation_indices.size, size)
                self.assertFalse(np.intersect1d(prepared.train_indices, prepared.validation_indices).size)
                np.testing.assert_array_equal(np.sort(np.concatenate((prepared.train_indices, prepared.validation_indices))), np.arange(100))
                np.testing.assert_array_equal(prepared.dataset_for(DataRole.TEST).indices, np.arange(20))
                if size == 20:
                    if initial is not None:
                        np.testing.assert_array_equal(prepared.validation_indices, initial)
                    initial = prepared.validation_indices.copy()
            config["data"]["validation_size"] = 100
            with self.assertRaisesRegex(ValueError, "原始训练池"):
                prepare_experiment_data(config, requirements=requirements,
                    run_dir=Path(directory) / "invalid", seed=1, registry=registry)
            config["data"].pop("validation_size")
            config["data"].update(num_val=20, num_clean=5)
            config["trusted_validation"] = {"source": "official_generated"}
            trusted_requirements = DataRequirements(
                roles=requirements.roles | {DataRole.TRUSTED_VALIDATION},
                validation_targets="clean", needs_noise_manifest=False,
            )
            prepared = prepare_experiment_data(config, requirements=trusted_requirements,
                run_dir=Path(directory) / "trusted", seed=1, registry=registry)
            trusted = prepared.dataset_for(DataRole.TRUSTED_VALIDATION).indices
            self.assertEqual((prepared.train_indices.size, prepared.validation_indices.size, trusted.size), (75, 20, 5))
            self.assertFalse(np.intersect1d(trusted, prepared.train_indices).size)
            self.assertFalse(np.intersect1d(trusted, prepared.validation_indices).size)

    def test_registration_does_not_overwrite_the_web_validation_count(self) -> None:
        service = self.service.data_service
        service.register("registered-holdout", "cifar10", {
            "root": str(Path(self.temp.name) / "data"), "validation_size": 5,
            "validation_split": {"source": "native"},
        })
        config = {"data": {"name": "cifar10", "validation_size": 3,
                           "validation_split": {"source": "training_pool"}}}
        resolved = service.apply(config, "registered-holdout")
        self.assertEqual(resolved["data"]["validation_size"], 3)
        self.assertEqual(resolved["data"]["validation_split"]["source"], "training_pool")

    def test_web_synthetic_validation_is_a_holdout_not_additional_generated_data(self) -> None:
        from lnl_toolbox.data.contracts import DataRequirements, DataRole
        from lnl_toolbox.data.sources import SyntheticAdapter
        from lnl_toolbox.training.data_service import prepare_experiment_data

        registry = DatasetRegistry((SyntheticAdapter("synthetic_binary_2d"),))
        config = {"data": {"name": "synthetic_binary_2d", "train_size": 100,
                           "validation_size": 20, "test_size": 10,
                           "validation_split": {"source": "training_pool"}},
                  "loader": {"batch_size": 8, "num_workers": 0}}
        requirements = DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.CLEAN_VALIDATION, DataRole.TEST}),
            validation_targets="clean", needs_noise_manifest=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_experiment_data(config, requirements=requirements,
                run_dir=directory, seed=1, registry=registry)
        self.assertEqual(prepared.train_indices.size, 80)
        self.assertEqual(prepared.validation_indices.size, 20)
        self.assertEqual(len(prepared.dataset_for(DataRole.TEST)), 10)
        spec = DataSpec.from_mapping(config["data"])
        original_test = registry.load(spec, "test", seed=1)
        config["data"]["validation_size"] = 30
        modified_test = registry.load(DataSpec.from_mapping(config["data"]), "test", seed=1)
        np.testing.assert_array_equal(original_test.global_indices, modified_test.global_indices)
        np.testing.assert_array_equal(original_test.inputs, modified_test.inputs)

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
