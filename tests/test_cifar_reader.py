import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lnl_toolbox.data.cifar import load_cifar10, load_cifar100, summarize_cifar


def dump(path: Path, payload: dict) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


class CifarReaderTest(unittest.TestCase):
    def test_cifar10_pickle_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flat = np.arange(2 * 3072, dtype=np.uint8).reshape(2, 3072)
            for index in range(1, 6):
                dump(root / f"data_batch_{index}", {b"data": flat, b"labels": [0, 1]})
            dump(root / "test_batch", {b"data": flat, b"labels": [1, 0]})
            dump(root / "batches.meta", {b"label_names": [b"zero", b"one"]})
            data = load_cifar10(root, "train")
        self.assertEqual(data.images.shape, (10, 32, 32, 3))
        self.assertEqual(data.class_names, ("zero", "one"))

    def test_cifar100_pickle_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flat = np.zeros((3, 3072), dtype=np.uint8)
            dump(root / "train", {b"data": flat, b"fine_labels": [0, 1, 1]})
            dump(root / "meta", {b"fine_label_names": [b"zero", b"one"]})
            data = load_cifar100(root, "train")
            summary = summarize_cifar(data)
        self.assertEqual(data.images.shape, (3, 32, 32, 3))
        self.assertEqual(summary["class_count_max"], 2)


if __name__ == "__main__":
    unittest.main()

