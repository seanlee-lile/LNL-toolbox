from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lnl_toolbox.data.contracts import DataSpec, RawDatasetSplit
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
            raise ValueError("split must be train or test")
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
