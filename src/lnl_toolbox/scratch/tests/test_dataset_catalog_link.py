from __future__ import annotations

import json
import gzip
import os
import pickle
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lnl_toolbox.scratch import ScratchContext
from lnl_toolbox.scratch.data_runtime import (
    ScratchSample,
    ScratchSplit,
    registered_dataset_catalog,
    registered_dataset_config,
)
from lnl_toolbox.scratch.blocks.data import load_dataset
from lnl_toolbox.scratch.web.data_bridge import dataset_catalog_payload, dataset_fact_payload


class ScratchDatasetCatalogLinkTest(unittest.TestCase):
    def test_scratch_catalog_exposes_shared_registered_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cifar"
            root.mkdir()
            catalog = Path(directory) / "datasets.json"
            catalog.write_text(json.dumps({
                "version": 1,
                "datasets": {
                    "my-cifar": {
                        "adapter": "cifar10",
                        "data": {"name": "cifar10", "root": str(root)},
                        "state": "registered",
                        "evidence": None,
                        "profile": None,
                    },
                },
            }), encoding="utf-8")
            old = os.environ.get("LNL_DATA_CATALOG")
            os.environ["LNL_DATA_CATALOG"] = str(catalog)
            try:
                record = registered_dataset_config("my-cifar")
                self.assertIsNotNone(record)
                self.assertEqual(record["adapter"], "cifar10")
                self.assertEqual(record["data"]["root"], str(root))
                item = next(item for item in dataset_catalog_payload()["datasets"] if item["alias"] == "my-cifar")
                self.assertEqual(item["adapter"], "cifar10")
                self.assertEqual(item["status"], "available")
                self.assertEqual(dataset_fact_payload("my-cifar")["path"], str(root))
            finally:
                if old is None:
                    os.environ.pop("LNL_DATA_CATALOG", None)
                else:
                    os.environ["LNL_DATA_CATALOG"] = old

    def test_load_dataset_resolves_registered_adapter_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cifar"
            root.mkdir()
            catalog = Path(directory) / "datasets.json"
            catalog.write_text(json.dumps({
                "version": 1,
                "datasets": {
                    "my-cifar": {
                        "adapter": "cifar10",
                        "data": {"name": "cifar10", "root": str(root)},
                        "state": "registered",
                    },
                },
            }), encoding="utf-8")
            old = os.environ.get("LNL_DATA_CATALOG")
            os.environ["LNL_DATA_CATALOG"] = str(catalog)
            train = ScratchSplit("my-cifar", "train", (ScratchSample(torch.zeros(2), 0, 0, 0),), 10)
            test = ScratchSplit("my-cifar", "test", (ScratchSample(torch.ones(2), 0, 1, 1),), 10)
            try:
                ctx = ScratchContext()
                with patch("lnl_toolbox.scratch.blocks.data.load_sources", return_value=(
                    {"data": {"name": "my-cifar", "adapter": "cifar10", "root": str(root)}},
                    train, None, test,
                )) as mocked:
                    load_dataset(ctx, "my-cifar")
                effective_plan = mocked.call_args.args[0]
                self.assertEqual(effective_plan["data"]["adapter"], "cifar10")
                self.assertEqual(effective_plan["data"]["root"], str(root))
                self.assertEqual(ctx["train_source"], train)
                self.assertEqual(ctx["test_source"], test)
            finally:
                if old is None:
                    os.environ.pop("LNL_DATA_CATALOG", None)
                else:
                    os.environ["LNL_DATA_CATALOG"] = old

    def test_registered_uci_alias_materializes_train_validation_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "heart.data"
            source.write_text("0.0 1.0 0\n1.0 0.0 1\n0.2 0.8 0\n0.8 0.2 1\n0.3 0.7 0\n0.7 0.3 1\n", encoding="utf-8")
            catalog = root / "datasets.json"
            catalog.write_text(json.dumps({
                "version": 1,
                "datasets": {
                    "demo-heart": {
                        "adapter": "uci_binary",
                        "data": {
                            "name": "uci_binary", "path": str(source),
                            "preprocessing": {"format": "whitespace", "target_column": -1},
                            "split": {"validation_fraction": 0.2, "test_fraction": 0.2, "seed": 3},
                        },
                        "state": "registered",
                    },
                },
            }), encoding="utf-8")
            old = os.environ.get("LNL_DATA_CATALOG")
            os.environ["LNL_DATA_CATALOG"] = str(catalog)
            try:
                ctx = ScratchContext(seed=5)
                load_dataset(ctx, "demo-heart")
                self.assertEqual(ctx["train_source"].num_classes, 2)
                self.assertTrue(ctx["train_source"].samples)
                self.assertTrue(ctx["test_source"].samples)
                self.assertIsNotNone(ctx["validation_source"])
            finally:
                if old is None:
                    os.environ.pop("LNL_DATA_CATALOG", None)
                else:
                    os.environ["LNL_DATA_CATALOG"] = old

    def test_registered_cifar_alias_reads_direct_official_pickle_layout(self) -> None:
        """The catalog stores the extracted CIFAR root, not torchvision's parent directory."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cifar10"
            root.mkdir()
            image = np.zeros((1, 3072), dtype=np.uint8)
            for index in range(1, 6):
                (root / f"data_batch_{index}").write_bytes(
                    pickle.dumps({b"data": image, b"labels": [index - 1]})
                )
            (root / "test_batch").write_bytes(
                pickle.dumps({b"data": image, b"labels": [0]})
            )
            (root / "batches.meta").write_bytes(
                pickle.dumps({b"label_names": [str(index).encode() for index in range(10)]})
            )
            catalog = Path(directory) / "datasets.json"
            catalog.write_text(json.dumps({
                "version": 1,
                "datasets": {
                    "direct-cifar": {
                        "adapter": "cifar10",
                        "data": {"name": "cifar10", "root": str(root)},
                        "state": "registered",
                    },
                },
            }), encoding="utf-8")
            old = os.environ.get("LNL_DATA_CATALOG")
            os.environ["LNL_DATA_CATALOG"] = str(catalog)
            try:
                ctx = ScratchContext()
                load_dataset(ctx, "direct-cifar")
                self.assertEqual(len(ctx["train_source"].samples), 5)
                self.assertEqual(len(ctx["test_source"].samples), 1)
                self.assertEqual(ctx["train_source"].samples[0].input.shape, (32, 32, 3))
            finally:
                if old is None:
                    os.environ.pop("LNL_DATA_CATALOG", None)
                else:
                    os.environ["LNL_DATA_CATALOG"] = old

    def test_registered_mnist_alias_reads_direct_idx_layout(self) -> None:
        """The registration root may contain IDX.GZ files directly under it."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fashion-mnist"
            root.mkdir()

            def write_split(prefix: str, count: int, labels: list[int]) -> None:
                images = bytes([value % 255 for value in range(count * 28 * 28)])
                image_payload = struct.pack(">IIII", 2051, count, 28, 28) + images
                label_payload = struct.pack(">II", 2049, count) + bytes(labels)
                with gzip.open(root / f"{prefix}-images-idx3-ubyte.gz", "wb") as handle:
                    handle.write(image_payload)
                with gzip.open(root / f"{prefix}-labels-idx1-ubyte.gz", "wb") as handle:
                    handle.write(label_payload)

            write_split("train", 2, [0, 1])
            write_split("t10k", 1, [1])
            catalog = Path(directory) / "datasets.json"
            catalog.write_text(json.dumps({
                "version": 1,
                "datasets": {
                    "direct-fashion": {
                        "adapter": "fashion_mnist",
                        "data": {"name": "fashion_mnist", "root": str(root)},
                        "state": "registered",
                    },
                },
            }), encoding="utf-8")
            old = os.environ.get("LNL_DATA_CATALOG")
            os.environ["LNL_DATA_CATALOG"] = str(catalog)
            try:
                ctx = ScratchContext()
                load_dataset(ctx, "direct-fashion")
                self.assertEqual(len(ctx["train_source"].samples), 2)
                self.assertEqual(len(ctx["test_source"].samples), 1)
                self.assertEqual(ctx["train_source"].samples[0].input.shape, (28, 28))
            finally:
                if old is None:
                    os.environ.pop("LNL_DATA_CATALOG", None)
                else:
                    os.environ["LNL_DATA_CATALOG"] = old


if __name__ == "__main__":
    unittest.main()
