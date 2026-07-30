import tempfile
from pathlib import Path
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
