import tempfile
from pathlib import Path
import unittest

import numpy as np

import lnl_toolbox.data as data_package
from lnl_toolbox.data import binary_benchmarks
from lnl_toolbox.data.binary_benchmarks import BinaryBenchmark
from lnl_toolbox.data.preprocessing import (
    BinaryPreprocessingConfig,
    BinaryPreprocessor,
)


class BinaryPreprocessingTest(unittest.TestCase):
    def test_delimited_categorical_missing_and_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.csv"
            source.write_text(
                "age,color,label\n"
                "10,red,negative\n"
                "20,blue,positive\n"
                "?,red,negative\n",
                encoding="utf-8",
            )
            config = BinaryPreprocessingConfig(
                file_format="csv",
                target_column="label",
                has_header=True,
                standardize=True,
            )
            processor = BinaryPreprocessor(config).fit(source)
            benchmark = processor.transform(source, dataset="fixture")
            self.assertEqual(benchmark.features.shape, (3, 3))
            self.assertEqual(benchmark.targets.tolist(), [0, 1, 0])
            self.assertTrue(np.isfinite(benchmark.features).all())
            state = root / "preprocessing.json"
            processor.save(state)
            restored = BinaryPreprocessor.load(state)
            restored_benchmark = restored.transform(
                source,
                dataset="fixture",
            )
            np.testing.assert_allclose(
                benchmark.features,
                restored_benchmark.features,
            )
            np.testing.assert_array_equal(
                benchmark.targets,
                restored_benchmark.targets,
            )
            self.assertEqual(
                processor.label_mapping,
                restored.label_mapping,
            )

    def test_validation_uses_saved_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            validation = root / "validation.csv"
            train.write_text(
                "value,label\n1,negative\n3,positive\n",
                encoding="utf-8",
            )
            validation.write_text(
                "value,label\n5,positive\n",
                encoding="utf-8",
            )
            processor = BinaryPreprocessor(BinaryPreprocessingConfig(
                file_format="csv",
                target_column="label",
                has_header=True,
                standardize=True,
            )).fit(train)
            benchmark = processor.transform(
                validation,
                split="validation",
            )
            self.assertEqual(benchmark.split, "validation")
            self.assertEqual(benchmark.features.item(), 3.0)

    def test_whitespace_and_libsvm_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whitespace = root / "data.txt"
            whitespace.write_text(
                "0.0 1.0 0\n1.0 0.0 1\n",
                encoding="utf-8",
            )
            benchmark = BinaryPreprocessor(
                BinaryPreprocessingConfig(
                    file_format="whitespace",
                    target_column=-1,
                )
            ).fit_transform(whitespace)
            self.assertEqual(
                benchmark.features.tolist(),
                [[0.0, 1.0], [1.0, 0.0]],
            )

            libsvm = root / "data.libsvm"
            libsvm.write_text(
                "0 1:1.0 3:2.0\n1 2:4.0\n",
                encoding="utf-8",
            )
            benchmark = BinaryPreprocessor(
                BinaryPreprocessingConfig(
                    file_format="libsvm",
                    target_column=0,
                )
            ).fit_transform(libsvm)
            self.assertEqual(benchmark.features.shape, (2, 3))
            np.testing.assert_allclose(
                benchmark.features,
                [[1.0, 0.0, 2.0], [0.0, 4.0, 0.0]],
            )

    def test_empty_invalid_labels_and_missing_error_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.csv"
            empty.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no data"):
                BinaryPreprocessor().fit(empty)

            labels = root / "labels.csv"
            labels.write_text("1,a\n2,b\n3,c\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly two"):
                BinaryPreprocessor().fit(labels)

            missing = root / "missing.csv"
            missing.write_text("?,a\n1,b\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing value"):
                BinaryPreprocessor(BinaryPreprocessingConfig(
                    missing_policy="error",
                )).fit(missing)

    def test_binary_benchmark_rejects_invalid_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            BinaryBenchmark(
                np.asarray([[np.nan]], dtype=np.float32),
                np.asarray([0]),
                "fixture",
            )
        with self.assertRaisesRegex(ValueError, "labels 0 and 1"):
            BinaryBenchmark(
                np.asarray([[1.0]], dtype=np.float32),
                np.asarray([2]),
                "fixture",
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            BinaryBenchmark(
                np.asarray([[1.0], [2.0]], dtype=np.float32),
                np.asarray([0, 1]),
                "fixture",
                global_indices=np.asarray([4, 4]),
            )
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            BinaryBenchmark(
                np.asarray([[1.0], [2.0]], dtype=np.float32),
                np.asarray([0, 1]),
                "fixture",
                global_indices=np.asarray([4.0, 5.0]),
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            BinaryBenchmark(
                np.asarray([[1.0], [2.0]], dtype=np.float32),
                np.asarray([0, 1]),
                "fixture",
                global_indices=np.asarray([-1, 5]),
            )

    def test_delimited_transform_rejects_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            train.write_text(
                "first,second,label\n1,2,no\n3,4,yes\n",
                encoding="utf-8",
            )
            processor = BinaryPreprocessor(BinaryPreprocessingConfig(
                file_format="csv",
                target_column="label",
                has_header=True,
            )).fit(train)
            fixtures = {
                "missing.csv": "first,label\n1,no\n",
                "extra.csv": "first,second,extra,label\n1,2,3,no\n",
                "reordered.csv": "second,first,label\n2,1,no\n",
            }
            for name, contents in fixtures.items():
                with self.subTest(name=name):
                    path = root / name
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError,
                        "schema|columns",
                    ):
                        processor.transform(path)

    def test_libsvm_transform_requires_exact_fitted_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.libsvm"
            train.write_text(
                "0 1:1 3:2\n1 2:1\n",
                encoding="utf-8",
            )
            processor = BinaryPreprocessor(BinaryPreprocessingConfig(
                file_format="libsvm",
                target_column=0,
            )).fit(train)
            for name, contents in {
                "smaller.libsvm": "0 1:1 2:2\n1 2:1\n",
                "larger.libsvm": "0 1:1 4:2\n1 2:1\n",
            }.items():
                with self.subTest(name=name):
                    path = root / name
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError,
                        "observed feature width",
                    ):
                        processor.transform(path)

    def test_state_dict_is_deep_copy_and_load_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.csv"
            source.write_text(
                "value,color,label\n"
                "1,red,no\n2,blue,yes\n",
                encoding="utf-8",
            )
            processor = BinaryPreprocessor(BinaryPreprocessingConfig(
                file_format="csv",
                target_column="label",
                has_header=True,
                standardize=True,
            )).fit(source)
            state = processor.state_dict()
            state["column_specs"][0]["mean"] = 999.0
            state["input_header"][0] = "changed"
            self.assertNotEqual(
                processor.column_specs[0]["mean"],
                999.0,
            )
            self.assertEqual(processor.input_header[0], "value")

            valid = processor.state_dict()
            restored = BinaryPreprocessor().load_state_dict(valid)
            np.testing.assert_allclose(
                restored.transform(source).features,
                processor.transform(source).features,
            )
            corruptions = []
            bad = processor.state_dict()
            bad["label_mapping"] = {"no": 0, "yes": 2}
            corruptions.append((bad, "label_mapping"))
            bad = processor.state_dict()
            bad["feature_names"] = ["wrong"]
            corruptions.append((bad, "feature_names"))
            bad = processor.state_dict()
            bad["column_specs"][0]["scale"] = 0.0
            corruptions.append((bad, "scale"))
            bad = processor.state_dict()
            bad["column_specs"][0]["mean"] = float("nan")
            corruptions.append((bad, "finite"))
            bad = processor.state_dict()
            bad["feature_columns"] = list(reversed(bad["feature_columns"]))
            corruptions.append((bad, "feature_columns"))
            for bad, message in corruptions:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        BinaryPreprocessor().load_state_dict(bad)

    def test_binary_package_exports_only_consumed_public_api(self) -> None:
        self.assertFalse(hasattr(data_package, "load_uci_binary"))
        self.assertFalse(
            hasattr(binary_benchmarks, "corrupt_binary_labels")
        )
        self.assertFalse(
            hasattr(binary_benchmarks, "stratified_binary_splits")
        )
        self.assertFalse(
            hasattr(binary_benchmarks, "cifar_airplane_automobile_view")
        )


if __name__ == "__main__":
    unittest.main()
