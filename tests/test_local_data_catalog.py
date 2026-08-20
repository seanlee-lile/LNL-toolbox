from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lnl_toolbox.data.local_catalog import LocalDatasetCatalog


class LocalDatasetCatalogTest(unittest.TestCase):
    def test_registration_merge_states_and_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset"
            source.mkdir()
            (source / "train").write_text("fixture", encoding="utf-8")
            catalog = LocalDatasetCatalog(root / "catalog.json")

            record = catalog.register("My-CIFAR", "cifar10", {"root": source})
            self.assertEqual(record.effective_state, "registered")
            config = catalog.apply(
                {
                    "data": {
                        "name": "cifar100",
                        "root": "stale",
                        "validation_size": 5,
                        "augment": False,
                    }
                },
                "my-cifar",
            )
            self.assertEqual(config["data"]["name"], "cifar10")
            self.assertEqual(config["data"]["root"], str(source.resolve()))
            self.assertEqual(config["data"]["validation_size"], 5)
            self.assertEqual(config["local_dataset"]["alias"], "my-cifar")

            self.assertEqual(
                catalog.mark_layout_validated("my-cifar").effective_state,
                "layout_validated",
            )
            verified = catalog.mark_training_verified(
                "my-cifar", {"run_dir": "run", "data_fingerprint": "abc"}
            )
            self.assertEqual(verified.effective_state, "training_verified")
            (source / "changed").write_text("changed", encoding="utf-8")
            self.assertEqual(catalog.get("my-cifar").effective_state, "verification_stale")

            catalog.remove("my-cifar")
            self.assertEqual(catalog.records(), ())

    def test_reregister_and_failure_clear_old_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            catalog = LocalDatasetCatalog(root / "catalog.json")
            catalog.register("data", "cifar10", {"root": first})
            catalog.mark_training_verified("data", {"run_dir": "run"})
            record = catalog.register("data", "cifar100", {"root": second})
            self.assertEqual(record.adapter, "cifar100")
            self.assertEqual(record.effective_state, "registered")
            failed = catalog.mark_failed("data", "bad labels")
            self.assertEqual(failed.effective_state, "failed")
            self.assertEqual(failed.error, "bad labels")


if __name__ == "__main__":
    unittest.main()
