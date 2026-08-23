from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from lnl_toolbox.data.contracts import DataSpec, RawDatasetSplit
from lnl_toolbox.data.local_catalog import LocalDatasetCatalog
from lnl_toolbox.data.registry import DatasetRegistry
from lnl_toolbox.training.data_service import DataService
from lnl_toolbox.quickstart.models import QuickStartNoiseSelection
from lnl_toolbox.quickstart.service import QuickStartService


class _FakeImageAdapter:
    name = "cifar10"
    aliases = ()

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError(spec.root)

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        count = 12 if split == "train" else 6
        images = np.zeros((count, 32, 32, 3), dtype=np.uint8)
        labels = np.arange(count, dtype=np.int64) % 2
        return RawDatasetSplit(
            images, labels, np.arange(count, dtype=np.int64), self.name, split, 2,
            clean_targets=labels, class_names=("zero", "one"), source="test-fixture",
        )


class QuickStartServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.data_root = root / "data"
        self.data_root.mkdir()
        registry = DatasetRegistry((_FakeImageAdapter(),))
        catalog = LocalDatasetCatalog(root / "catalog.json")
        self.data_service = DataService(registry=registry, catalog=catalog)
        self.service = QuickStartService(self.data_service, artifact_root=root / "artifacts")
        self.registered = self.data_service.register(
            "local-cifar10", "cifar10", {"root": str(self.data_root)}
        )
        self.data_service.inspect("local-cifar10")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_registered_dataset_is_reused(self) -> None:
        result = self.service.register_and_inspect(str(self.data_root))
        self.assertEqual(result.alias, "local-cifar10")
        self.assertEqual(len(self.data_service.catalog.records()), 1)

    def test_noise_options_are_capability_based(self) -> None:
        result = self.service.noise_options("local-cifar10")
        keys = {item["key"] for item in result["options"]}
        self.assertIn("symmetric", keys)
        self.assertNotIn("external_torch", keys)

    def test_method_options_are_paper_catalog_driven(self) -> None:
        result = self.service.method_options(
            "local-cifar10", QuickStartNoiseSelection("clean", "clean")
        )
        self.assertGreater(len(result), 5)
        self.assertTrue(all(item.acronym for item in result))
        fine = next(item for item in result if item.paper_id == "fine")
        self.assertEqual(fine.status, "unsupported")

    def test_unknown_path_is_not_marked_ready(self) -> None:
        result = self.service.register_and_inspect(str(Path(self.temp.name) / "missing"))
        self.assertEqual(result.status, "unsupported")


if __name__ == "__main__":
    unittest.main()
