from pathlib import Path
import pickle
import tarfile
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image
import yaml

from lnl_toolbox.data.archive import archive_layout
from lnl_toolbox.data.cifar import _unpickle
from lnl_toolbox.data.contracts import DataSpec
from lnl_toolbox.data.contracts import DataRequirements
from lnl_toolbox.data.local_catalog import LocalDatasetCatalog
from lnl_toolbox.data.probe import probe_dataset_path
from lnl_toolbox.data.standard import StandardDatasetAdapter
from lnl_toolbox.quickstart.service import QuickStartService
from lnl_toolbox.training.data_service import DataService, prepare_experiment_data


class StandardDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.adapter = StandardDatasetAdapter()

    def tearDown(self):
        self.temporary.cleanup()

    def write_dataset(self, modality="tabular", classes=2, split=False, clean=False):
        metadata = {"name": "user_data", "task": "classification", "modality": modality,
                    "classes": [{"id": i, "name": f"class{i}"} for i in range(classes)],
                    "label_status": "clean" if clean else "unknown"}
        (self.dataset / "dataset.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
        rows = ["sample_id,label," + ("path" if modality == "image" else "feature_x,feature_y") + (",split" if split else "")]
        for i in range(12):
            if modality == "image":
                Image.new("RGB", (32, 32)).save(self.dataset / f"{i}.png")
                values = f"{i}.png"
            else:
                values = f"{i},{i+1}"
            rows.append(f"s{i},{i % classes},{values}" + (",train" if i < 9 else ",test") if split else f"s{i},{i % classes},{values}")
        (self.dataset / "samples.csv").write_text("\n".join(rows), encoding="utf-8")
        return DataSpec("standard", self.dataset)

    def quickstart(self):
        service = DataService(catalog=LocalDatasetCatalog(self.root / "catalog.json"))
        return QuickStartService(service, artifact_root=self.root / "artifacts")

    def test_unsplit_binary_is_not_partitioned_on_import(self):
        spec = self.write_dataset()
        self.assertEqual(len(self.adapter.load(spec, "train", seed=1)), 12)
        self.assertEqual(len(self.adapter.load(spec, "test", seed=1)), 0)
        self.assertIsNone(self.adapter.load(spec, "train", seed=1).clean_targets)

    def test_unsplit_training_reports_missing_test_partition(self):
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "no test split"):
            prepare_experiment_data({"data": {"name": "standard", "root": str(self.dataset)}},
                                    requirements=DataRequirements(), run_dir=self.root / "run", seed=1)

    def test_multiclass_images_via_quickstart(self):
        self.write_dataset("image", classes=3, split=True, clean=True)
        qs = self.quickstart()
        self.assertEqual(qs.probe(str(self.dataset)).candidates[0].adapter, "standard")
        summary = qs.register_and_inspect(str(self.dataset))
        self.assertEqual((summary.num_classes, summary.train_size, summary.test_size), (3, 9, 3))
        self.assertEqual(qs.noise_options(summary.alias)["dataset_state"], "clean")

    def test_configured_partition_is_deterministic_and_disjoint(self):
        self.write_dataset()
        spec = DataSpec("standard", self.dataset, options={"split": {"test_fraction": .25}})
        train = self.adapter.load(spec, "train", seed=3)
        test = self.adapter.load(spec, "test", seed=3)
        self.assertFalse(set(train.global_indices) & set(test.global_indices))
        self.assertEqual(len(train) + len(test), 12)
        np.testing.assert_array_equal(test.global_indices, self.adapter.load(spec, "test", seed=3).global_indices)

    def test_duplicate_ids_rejected(self):
        spec = self.write_dataset()
        p = self.dataset / "samples.csv"
        p.write_text(p.read_text().replace("s1,", "s0,"))
        with self.assertRaisesRegex(ValueError, "unique"):
            self.adapter.load(spec, "train", seed=1)

    def test_declared_noisy_status_reaches_quickstart(self):
        self.write_dataset(split=True)
        path = self.dataset / "dataset.yaml"
        metadata = yaml.safe_load(path.read_text())
        metadata.update(label_status="noisy", noise_rate=.3)
        path.write_text(yaml.safe_dump(metadata))
        qs = self.quickstart()
        summary = qs.register_and_inspect(str(self.dataset))
        facts = qs.noise_options(summary.alias)
        self.assertEqual(facts["dataset_state"], "native")
        self.assertEqual(facts["noise_rate"]["value"], .3)

    def test_group_cannot_cross_explicit_splits(self):
        spec = self.write_dataset(split=True)
        path = self.dataset / "samples.csv"
        lines = path.read_text().splitlines()
        path.write_text(lines[0] + ",group_id\n" + "\n".join(line + ",same_patient" for line in lines[1:]))
        with self.assertRaisesRegex(ValueError, "cross"):
            self.adapter.load(spec, "train", seed=1)

    def test_text_is_rejected_instead_of_reported_trainable(self):
        spec = self.write_dataset()
        path = self.dataset / "dataset.yaml"
        metadata = yaml.safe_load(path.read_text())
        metadata["modality"] = "text"
        path.write_text(yaml.safe_dump(metadata))
        with self.assertRaisesRegex(ValueError, "text training"):
            self.adapter.load(spec, "train", seed=1)

    def test_invalid_class_label_rejected(self):
        spec = self.write_dataset()
        p = self.dataset / "samples.csv"
        p.write_text(p.read_text().replace("s0,0,", "s0,3,"))
        with self.assertRaisesRegex(ValueError, "outside"):
            self.adapter.load(spec, "train", seed=1)

    def test_image_path_escape_rejected(self):
        spec = self.write_dataset("image")
        p = self.dataset / "samples.csv"
        p.write_text(p.read_text().replace("0.png", "../outside.png"))
        with self.assertRaisesRegex(ValueError, "inside"):
            self.adapter.load(spec, "train", seed=1)

    def test_non_numeric_features_rejected(self):
        spec = self.write_dataset()
        p = self.dataset / "samples.csv"
        p.write_text(p.read_text().replace("s0,0,0,1", "s0,0,nan,1"))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.adapter.load(spec, "train", seed=1)

    def test_standard_zip_uses_same_quickstart_registry(self):
        self.write_dataset("image", split=True)
        archive = self.root / "custom.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for f in self.dataset.iterdir():
                z.write(f, "user_dataset/" + f.name)
        qs = self.quickstart()
        self.assertEqual(qs.probe(str(archive)).status, "detected")
        self.assertFalse(qs.artifact_root.exists())
        result = qs.register_and_inspect(str(archive))
        self.assertEqual((result.adapter, result.train_size, result.test_size), ("standard", 9, 3))
        self.assertEqual(qs.probe(str(archive)).existing_alias, result.alias)
        renamed = self.root / "renamed.zip"
        renamed.write_bytes(archive.read_bytes())
        self.assertEqual(qs.register_and_inspect(str(renamed)).alias, result.alias)

    def test_cifar_tar_gz_uses_existing_cifar_adapter(self):
        source = self.root / "cifar-10-batches-py"
        source.mkdir()
        for name in [*(f"data_batch_{i}" for i in range(1, 6)), "test_batch"]:
            (source / name).write_bytes(pickle.dumps({"data": np.zeros((2, 3072), dtype=np.uint8), "labels": [0, 1]}, protocol=2))
        (source / "batches.meta").write_bytes(pickle.dumps({"label_names": [str(i) for i in range(10)]}, protocol=2))
        archive = self.root / "cifar.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source, arcname=source.name)
        qs = self.quickstart()
        self.assertEqual(qs.probe(str(archive)).candidates[0].adapter, "cifar10")
        result = qs.register_and_inspect(str(archive))
        self.assertEqual((result.adapter, result.train_size, result.test_size), ("cifar10", 10, 2))

    def test_archive_traversal_rejected_without_extraction(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../outside.txt", "bad")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            probe_dataset_path(archive)
        self.assertFalse((self.root / "outside.txt").exists())

    def test_archive_symlink_rejected(self):
        archive = self.root / "bad.zip"
        member = zipfile.ZipInfo("link")
        member.external_attr = 0o120777 << 16
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(member, "outside")
        with self.assertRaisesRegex(ValueError, "links"):
            archive_layout(archive)

    def test_pickle_executable_global_rejected(self):
        path = self.root / "pickle"
        path.write_bytes(pickle.dumps(eval))
        with self.assertRaisesRegex(pickle.UnpicklingError, "unsupported"):
            _unpickle(path)


class CIFAR10NWorstBinaryViewTest(unittest.TestCase):
    def test_full_mapping_keeps_indices_and_omits_train_clean_targets(self):
        import numpy as np
        from unittest.mock import patch
        from lnl_toolbox.data.cifar_n import CifarNAdapter, CifarNWorstBinaryAdapter
        from lnl_toolbox.data.contracts import RawDatasetSplit, DataSpec
        labels = np.arange(10, dtype=np.int64)
        indices = np.arange(10, dtype=np.int64) + 100
        raw = RawDatasetSplit(np.zeros((10,32,32,3), dtype=np.uint8), labels, indices,
                              'cifar10n', 'train', 10, clean_targets=labels[::-1])
        spec = DataSpec.from_mapping({'name':'cifar10n_vehicle_animal','noise_variant':'worse_label'})
        with patch.object(CifarNAdapter, 'load', return_value=raw):
            result = CifarNWorstBinaryAdapter().load(spec, 'train', seed=17)
        np.testing.assert_array_equal(result.observed_targets, [0,0,1,1,1,1,1,1,0,0])
        np.testing.assert_array_equal(result.global_indices, indices)
        self.assertEqual(len(result), len(raw))
        self.assertIsNone(result.clean_targets)
        self.assertEqual(result.num_classes, 2)
        self.assertIs(result.inputs, raw.inputs)

    def test_clean_train_labels_do_not_change_observations(self):
        import numpy as np
        from unittest.mock import patch
        from lnl_toolbox.data.cifar_n import CifarNAdapter, CifarNWorstBinaryAdapter
        from lnl_toolbox.data.contracts import RawDatasetSplit, DataSpec
        observed=np.arange(10, dtype=np.int64)
        spec=DataSpec.from_mapping({'name':'cifar10n_vehicle_animal','noise_variant':'worse_label'})
        results=[]
        for clean in (observed, observed[::-1]):
            raw=RawDatasetSplit(np.zeros((10,32,32,3),dtype=np.uint8),observed,np.arange(10),'cifar10n','train',10,clean_targets=clean)
            with patch.object(CifarNAdapter,'load',return_value=raw):
                results.append(CifarNWorstBinaryAdapter().load(spec,'train',seed=17))
        np.testing.assert_array_equal(results[0].observed_targets,results[1].observed_targets)
        self.assertIsNone(results[0].clean_targets)
        self.assertIsNone(results[1].clean_targets)

    def test_test_labels_and_worst_variant_guard(self):
        import numpy as np
        from unittest.mock import patch
        from lnl_toolbox.data.cifar_n import CifarNAdapter, CifarNWorstBinaryAdapter
        from lnl_toolbox.data.contracts import RawDatasetSplit, DataSpec
        labels=np.arange(10,dtype=np.int64)
        raw=RawDatasetSplit(np.zeros((10,32,32,3),dtype=np.uint8),labels,np.arange(10),'cifar10n','test',10,clean_targets=labels)
        spec=DataSpec.from_mapping({'name':'cifar10n_vehicle_animal','noise_variant':'worse_label'})
        with patch.object(CifarNAdapter,'load',return_value=raw):
            result=CifarNWorstBinaryAdapter().load(spec,'test',seed=17)
        np.testing.assert_array_equal(result.clean_targets,result.observed_targets)
        with self.assertRaisesRegex(ValueError,'worse_label'):
            CifarNWorstBinaryAdapter().load(DataSpec.from_mapping({'name':'cifar10n_vehicle_animal'}),'train',seed=17)


if __name__ == "__main__":
    unittest.main()
