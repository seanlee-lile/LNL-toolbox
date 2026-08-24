from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from lnl_toolbox.data.contracts import DataSpec, RawDatasetSplit
from lnl_toolbox.data.local_catalog import LocalDatasetCatalog
from lnl_toolbox.data.profile import (
    DatasetSemanticHints,
    KnowledgeState,
    NoiseKnowledge,
    NoiseOrigin,
    NoiseStatus,
)
from lnl_toolbox.data.registry import DatasetRegistry
from lnl_toolbox.training.data_service import DataService


class _Adapter:
    name = "toy_images"
    aliases = ()

    def __init__(self, *, hints: DatasetSemanticHints | None = None) -> None:
        self.fail = False
        self.semantic_hints = hints

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError(spec.root)

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del spec, seed
        if self.fail:
            raise RuntimeError("fixture load failed")
        if split == "validation":
            raise ValueError("split must be train or test")
        count = 20 if split == "train" else 10
        labels = np.arange(count, dtype=np.int64) % 2
        return RawDatasetSplit(
            np.zeros((count, 8, 8, 3), dtype=np.uint8), labels,
            np.arange(count, dtype=np.int64), self.name, split, 2,
            clean_targets=None, class_names=("zero", "one"), source="fixture",
        )


class DataCapabilityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.data_root = root / "data"
        self.data_root.mkdir()
        self.adapter = _Adapter()
        self.service = DataService(
            registry=DatasetRegistry((self.adapter,)),
            catalog=LocalDatasetCatalog(root / "catalog.json"),
        )
        self.service.register("toy-local", "toy_images", {"root": str(self.data_root)})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_inspect_constructs_profile_and_capabilities(self) -> None:
        report = self.service.inspect("toy-local")
        self.assertEqual(report.status, "ready")
        self.assertIsNotNone(report.profile)
        self.assertEqual(report.profile.input_shape, (8, 8, 3))
        self.assertEqual(report.profile.clean_validation_labels, KnowledgeState.UNKNOWN)
        capabilities = self.service.capabilities("toy-local")
        self.assertEqual(capabilities.dataset, "toy-local")
        self.assertEqual(capabilities.clean_train_labels, KnowledgeState.UNKNOWN)

    def test_read_only_inspection_never_marks_catalog_failed(self) -> None:
        self.assertEqual(self.service.inspect("toy-local").status, "ready")
        self.adapter.fail = True
        report = self.service.inspect("toy-local", persist=False)
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(self.service.record("toy-local").effective_state, "layout_validated")

    def test_semantic_hints_fill_unknown_without_replacing_observation(self) -> None:
        root = Path(self.temp.name)
        adapter = _Adapter(hints=DatasetSemanticHints(
            clean_train_labels=KnowledgeState.UNAVAILABLE,
            noise=NoiseKnowledge(NoiseStatus.NOISY, NoiseOrigin.NATIVE),
        ))
        service = DataService(
            registry=DatasetRegistry((adapter,)),
            catalog=LocalDatasetCatalog(root / "hint-catalog.json"),
        )
        service.register("hinted", "toy_images", {"root": str(self.data_root)})
        capabilities = service.capabilities("hinted")
        self.assertEqual(capabilities.clean_train_labels, KnowledgeState.UNAVAILABLE)
        self.assertEqual(capabilities.noise_origin, NoiseOrigin.NATIVE)


if __name__ == "__main__":
    unittest.main()
