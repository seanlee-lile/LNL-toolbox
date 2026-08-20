from __future__ import annotations

import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image

from lnl_toolbox.training.service import ExperimentService


def _write_idx(root: Path, split: str, count: int) -> None:
    """Write the official MNIST/Fashion-MNIST IDX.GZ pair."""

    prefix = "train" if split == "train" else "t10k"
    images = np.arange(count * 28 * 28, dtype=np.uint8).reshape(count, 28, 28)
    labels = np.arange(count, dtype=np.uint8) % 10
    with gzip.open(root / f"{prefix}-images-idx3-ubyte.gz", "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, count, 28, 28))
        handle.write(images.tobytes())
    with gzip.open(root / f"{prefix}-labels-idx1-ubyte.gz", "wb") as handle:
        handle.write(struct.pack(">II", 2049, count))
        handle.write(labels.tobytes())


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((40, 40, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _write_clothing1m(root: Path) -> None:
    train = [f"images/train/{index}.jpg" for index in range(8)]
    validation = [f"images/val/{index}.jpg" for index in range(4)]
    test = [f"images/test/{index}.jpg" for index in range(4)]
    for offset, key in enumerate(train + validation + test):
        _write_rgb(root / key, 16 + offset * 8)
    (root / "noisy_train_key_list.txt").write_text(
        "\n".join(train) + "\n", encoding="utf-8"
    )
    (root / "clean_val_key_list.txt").write_text(
        "\n".join(validation) + "\n", encoding="utf-8"
    )
    (root / "clean_test_key_list.txt").write_text(
        "\n".join(test) + "\n", encoding="utf-8"
    )
    (root / "noisy_label_kv.txt").write_text(
        "\n".join(f"{key} {index % 2}" for index, key in enumerate(train)) + "\n",
        encoding="utf-8",
    )
    clean = validation + test
    (root / "clean_label_kv.txt").write_text(
        "\n".join(f"{key} {index % 2}" for index, key in enumerate(clean)) + "\n",
        encoding="utf-8",
    )


def _animal_record(identifier: int, label: int) -> bytes:
    pixels = np.full((3, 64, 64), 16 + identifier * 7, dtype=np.uint8)
    return struct.pack("<II", identifier, label) + pixels.tobytes()


def _write_animal10n(root: Path) -> None:
    (root / "data_batch_1.bin").write_bytes(
        b"".join(_animal_record(index, index % 2) for index in range(8))
    )
    (root / "test_batch.bin").write_bytes(
        b"".join(_animal_record(100 + index, index % 2) for index in range(4))
    )


def _write_uci_heart(path: Path) -> None:
    rows = []
    for index in range(40):
        features = [f"{(index + column) % 11 + column / 10:.1f}" for column in range(13)]
        rows.append(" ".join(features + [str(1 + index % 2)]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _image_config(name: str, root: Path) -> dict:
    return {
        "seed": 13,
        "data": {
            "name": name,
            "root": str(root),
            "validation_size": 4,
            "augment": False,
            "image_size": 32,
        },
        "loader": {"batch_size": 64, "num_workers": 0, "pin_memory": False},
        "model": {"name": "tiny_cnn", "width": 2},
        "loss": {"name": "ce"},
        "optimizer": {"name": "adam", "lr": 0.001},
        "scheduler": {"name": "none"},
        "trainer": {"epochs": 1, "device": "cpu", "progress": False},
        "execution": {"runner": "clean"},
    }


class DatasetTrainingFixturesTest(unittest.TestCase):
    def _assert_trained(self, config: dict, run_dir: Path) -> None:
        result = ExperimentService().run(config, run_dir)
        final = json.loads((result / "final_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(final["completed_epochs"], 1)
        self.assertTrue(np.isfinite(float(final["test_accuracy"])))
        self.assertTrue((result / "data_manifest.json").is_file())
        epoch_rows = []
        for line in (result / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "epoch":
                epoch_rows.append(row)
        self.assertEqual(len(epoch_rows), 1)

    def test_official_mnist_idx_gzip_trains_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "MNIST" / "raw"
            data_root.mkdir(parents=True)
            _write_idx(data_root, "train", 40)
            _write_idx(data_root, "test", 20)
            self._assert_trained(_image_config("mnist", root), root / "run")

    def test_official_fashion_mnist_idx_gzip_trains_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "FashionMNIST" / "raw"
            data_root.mkdir(parents=True)
            _write_idx(data_root, "train", 40)
            _write_idx(data_root, "test", 20)
            self._assert_trained(_image_config("fashion_mnist", root), root / "run")

    def test_official_clothing1m_lists_and_label_maps_train_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_clothing1m(root)
            self._assert_trained(_image_config("clothing1m", root), root / "run")

    def test_official_animal10n_binary_records_train_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_animal10n(root)
            self._assert_trained(_image_config("animal10n", root), root / "run")

    def test_official_uci_heart_whitespace_rows_train_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "heart.dat"
            run_dir = root / "run"
            _write_uci_heart(source)
            config = {
                "seed": 13,
                "data": {
                    "name": "uci_binary",
                    "path": str(source),
                    "preprocessing": {
                        "format": "whitespace",
                        "target_column": -1,
                        "standardize": False,
                        "label_values": ["1", "2"],
                    },
                    "split": {
                        "validation_fraction": 0.2,
                        "test_fraction": 0.2,
                        "seed": 13,
                    },
                },
                "loader": {"batch_size": 64, "num_workers": 0},
                "model": {"name": "linear"},
                "learning_rate": 0.01,
                "epochs": 1,
                "execution": {"runner": "binary"},
            }
            result = ExperimentService().run(config, run_dir)
            rows = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 1)
            self.assertTrue(np.isfinite(float(rows[0]["test_accuracy"])))
            self.assertTrue((result / "data_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
