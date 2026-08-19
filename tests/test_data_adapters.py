from __future__ import annotations

import gzip
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.data.cifar_n import CifarNAdapter
from lnl_toolbox.data.contracts import DataSpec
from lnl_toolbox.data.mnist import MnistAdapter
from lnl_toolbox.data.real_noise import Animal10NAdapter, Clothing1MAdapter
from lnl_toolbox.data.sources import CifarAdapter, SyntheticAdapter, UciBinaryAdapter


def _cifar(size: int, split: str, classes: int = 10) -> CifarData:
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(
        np.zeros((size, 32, 32, 3), dtype=np.uint8),
        labels,
        tuple(map(str, range(classes))),
        split,
        f"cifar{classes}",
    )


def _write_idx(root: Path, split: str, count: int) -> None:
    prefix = "train" if split == "train" else "t10k"
    images = np.arange(count * 28 * 28, dtype=np.uint8).reshape(count, 28, 28)
    labels = np.arange(count, dtype=np.uint8) % 10
    with gzip.open(root / f"{prefix}-images-idx3-ubyte.gz", "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, count, 28, 28) + images.tobytes())
    with gzip.open(root / f"{prefix}-labels-idx1-ubyte.gz", "wb") as handle:
        handle.write(struct.pack(">II", 2049, count) + labels.tobytes())


def _animal_record(index: int, label: int) -> bytes:
    pixels = np.full(3 * 64 * 64, index % 256, dtype=np.uint8)
    return index.to_bytes(4, "little") + label.to_bytes(4, "little") + pixels.tobytes()


class DataAdapterFixtureTest(unittest.TestCase):
    def test_cifar_and_cifar_n_observed_clean_separation(self) -> None:
        corpus = _cifar(6, "train")
        with patch("lnl_toolbox.data.sources.load_cifar10", return_value=corpus):
            split = CifarAdapter("cifar10", 10).load(DataSpec("cifar10"), "train", seed=1)
        np.testing.assert_array_equal(split.observed_targets, split.clean_targets)
        self.assertEqual(split.global_indices.tolist(), list(range(6)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            noisy = np.asarray([1, 1, 2, 3, 4, 5], dtype=np.int64)
            torch.save(
                {"aggre_label": noisy, "clean_label": corpus.labels.copy()},
                root / "CIFAR-10_human.pt",
            )
            with patch("lnl_toolbox.data.cifar_n.load_cifar10", return_value=corpus):
                split = CifarNAdapter("cifar10n", 10).load(
                    DataSpec("cifar10n", root=root), "train", seed=1
                )
            self.assertEqual(split.observed_targets.tolist(), noisy.tolist())
            self.assertEqual(split.clean_targets.tolist(), corpus.labels.tolist())
            self.assertIn("human_annotation", split.source)

    def test_mnist_official_idx_gzip_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_idx(root, "train", 8)
            _write_idx(root, "test", 4)
            adapter = MnistAdapter("mnist")
            adapter.validate(DataSpec("mnist", root=root))
            split = adapter.load(DataSpec("mnist", root=root), "train", seed=1)
        self.assertEqual(len(split), 8)
        self.assertEqual(split.inputs.shape, (8, 28, 28))
        self.assertEqual(split.source, "official_idx_gzip")

    def test_clothing1m_and_animal10n_lazy_file_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            Image.new("RGB", (4, 4), "white").save(image)
            for name in (
                "noisy_train_key_list.txt",
                "clean_val_key_list.txt",
                "clean_test_key_list.txt",
            ):
                (root / name).write_text("sample.jpg\n", encoding="utf-8")
            (root / "noisy_label_kv.txt").write_text("sample.jpg 3\n", encoding="utf-8")
            (root / "clean_label_kv.txt").write_text("sample.jpg 3\n", encoding="utf-8")
            clothing = Clothing1MAdapter()
            clothing.validate(DataSpec("clothing1m", root=root))
            train = clothing.load(DataSpec("clothing1m", root=root), "train", seed=1)
            test = clothing.load(DataSpec("clothing1m", root=root), "test", seed=1)
            self.assertIsNone(train.clean_targets)
            self.assertEqual(test.clean_targets.tolist(), [3])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split_name in ("training", "testing"):
                split_root = root / split_name
                split_root.mkdir(parents=True)
                for label in range(10):
                    Image.new("RGB", (4, 4), "white").save(split_root / f"{label}_one.png")
            animal = Animal10NAdapter()
            train = animal.load(DataSpec("animal10n", root=root), "train", seed=1)
            test = animal.load(DataSpec("animal10n", root=root), "test", seed=1)
            self.assertIsNone(train.clean_targets)
            self.assertEqual(len(test), 10)
            self.assertIsNotNone(test.clean_targets)

    def test_animal10n_official_binary_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data_batch_1.bin").write_bytes(
                _animal_record(0, 2) + _animal_record(1, 7)
            )
            (root / "test_batch.bin").write_bytes(_animal_record(2, 4))
            adapter = Animal10NAdapter()
            adapter.validate(DataSpec("animal10n", root=root))
            train = adapter.load(DataSpec("animal10n", root=root), "train", seed=1)
            test = adapter.load(DataSpec("animal10n", root=root), "test", seed=1)
            self.assertEqual(train.inputs.shape, (2, 64, 64, 3))
            self.assertEqual(train.observed_targets.tolist(), [2, 7])
            self.assertEqual(test.clean_targets.tolist(), [4])

    def test_uci_fits_preprocessing_on_training_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "heart.csv"
            source.write_text(
                "value,color,label\n"
                + "\n".join(
                    f"{index},{'red' if index % 2 else 'blue'},{index % 2}"
                    for index in range(20)
                )
                + "\n",
                encoding="utf-8",
            )
            spec = DataSpec(
                "uci_binary",
                path=source,
                options={
                    "preprocessing": {
                        "format": "csv",
                        "target_column": "label",
                        "has_header": True,
                        "standardize": True,
                    },
                    "split": {"validation_fraction": 0.2, "test_fraction": 0.2, "seed": 4},
                },
            )
            adapter = UciBinaryAdapter()
            train = adapter.load(spec, "train", seed=4)
            validation = adapter.load(spec, "validation", seed=4)
            test = adapter.load(spec, "test", seed=4)
            self.assertEqual(len(train) + len(validation) + len(test), 20)
            self.assertFalse(set(train.global_indices) & set(validation.global_indices))
            self.assertTrue(np.isfinite(validation.inputs).all())

    def test_synthetic_splits_have_disjoint_stable_indices(self) -> None:
        spec = DataSpec(
            "synthetic_multiclass",
            options={"num_classes": 3, "dimension": 3, "train_size": 9, "validation_size": 6, "test_size": 6},
        )
        adapter = SyntheticAdapter("synthetic_multiclass")
        train = adapter.load(spec, "train", seed=7)
        validation = adapter.load(spec, "validation", seed=7)
        test = adapter.load(spec, "test", seed=7)
        self.assertFalse(set(train.global_indices) & set(validation.global_indices))
        self.assertFalse(set(validation.global_indices) & set(test.global_indices))
        repeated = adapter.load(spec, "train", seed=7)
        np.testing.assert_array_equal(train.inputs, repeated.inputs)


if __name__ == "__main__":
    unittest.main()
