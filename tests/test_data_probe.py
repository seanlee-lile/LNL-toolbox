from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lnl_toolbox.data.probe import probe_dataset_path, suggest_dataset_alias
from lnl_toolbox.training.data_service import DataService
from lnl_toolbox.data.local_catalog import LocalDatasetCatalog


class DataProbeTests(unittest.TestCase):
    def _root(self) -> Path:
        self.temp = tempfile.TemporaryDirectory()
        return Path(self.temp.name)

    def tearDown(self) -> None:
        if hasattr(self, "temp"):
            self.temp.cleanup()

    @staticmethod
    def _cifar10(root: Path) -> Path:
        inner = root / "cifar-10-batches-py"
        inner.mkdir(parents=True)
        for name in ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch", "batches.meta"):
            (inner / name).touch()
        return root

    def test_cifar10(self) -> None:
        result = probe_dataset_path(self._cifar10(self._root()))
        self.assertEqual(result.status, "detected")
        self.assertEqual(result.candidates[0].adapter, "cifar10")
        self.assertEqual(result.candidates[0].confidence, "high")

    def test_cifar100(self) -> None:
        root = self._root()
        inner = root / "cifar-100-python"
        inner.mkdir()
        for name in ("train", "test", "meta"):
            (inner / name).touch()
        result = probe_dataset_path(root)
        self.assertEqual(result.candidates[0].adapter, "cifar100")

    def test_cifar10n_candidate(self) -> None:
        root = self._cifar10(self._root())
        (root / "cifar-10-batches-py" / "CIFAR-10_human.pt").touch()
        result = probe_dataset_path(root)
        self.assertIn("cifar10n", {item.adapter for item in result.candidates})
        candidate = next(item for item in result.candidates if item.adapter == "cifar10n")
        self.assertIn("noise_path", candidate.data)

    def test_clothing1m(self) -> None:
        root = self._root()
        for name in ("noisy_train_key_list.txt", "clean_val_key_list.txt", "clean_test_key_list.txt", "noisy_label_kv.txt", "clean_label_kv.txt"):
            (root / name).touch()
        result = probe_dataset_path(root)
        self.assertEqual(result.candidates[0].adapter, "clothing1m")

    def test_animal10n(self) -> None:
        root = self._root()
        (root / "train").mkdir()
        (root / "test").mkdir()
        (root / "train" / "0_sample.jpg").touch()
        (root / "test" / "0_sample.jpg").touch()
        result = probe_dataset_path(root)
        self.assertEqual(result.candidates[0].adapter, "animal10n")

    def test_registered_path(self) -> None:
        root = self._cifar10(self._root())
        catalog = LocalDatasetCatalog(Path(self.temp.name) / "catalog.json")
        service = DataService(catalog=catalog)
        service.register("existing", "cifar10", {"root": str(root / "cifar-10-batches-py")})
        result = probe_dataset_path(root / "cifar-10-batches-py", data_service=service)
        self.assertEqual(result.status, "already_registered")
        self.assertEqual(result.existing_alias, "existing")

    def test_unsupported(self) -> None:
        result = probe_dataset_path(self._root())
        self.assertEqual(result.status, "unsupported")

    def test_ambiguous_mnist_family(self) -> None:
        root = self._root()
        raw = root / "raw"
        raw.mkdir()
        for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
            (raw / name).touch()
        result = probe_dataset_path(root)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual({item.adapter for item in result.candidates}, {"mnist", "fashion_mnist"})

    def test_alias_collision(self) -> None:
        self.assertEqual(suggest_dataset_alias("cifar10", "/tmp/a", []), "cifar10-local")
        self.assertEqual(suggest_dataset_alias("cifar10", "/tmp/a", ["cifar10-local"]), "cifar10-local-2")


if __name__ == "__main__":
    unittest.main()
