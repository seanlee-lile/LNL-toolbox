import tempfile
from pathlib import Path
import unittest

import numpy as np

from lnl_toolbox.data.preprocessing import BinaryPreprocessingConfig, BinaryPreprocessor


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
            restored_benchmark = restored.transform(source, dataset="fixture")
            np.testing.assert_allclose(benchmark.features, restored_benchmark.features)
            np.testing.assert_array_equal(benchmark.targets, restored_benchmark.targets)

    def test_whitespace_and_libsvm_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whitespace = root / "data.txt"
            whitespace.write_text("0.0 1.0 0\n1.0 0.0 1\n", encoding="utf-8")
            benchmark = BinaryPreprocessor(BinaryPreprocessingConfig(
                file_format="whitespace", target_column=-1,
            )).fit_transform(whitespace)
            self.assertEqual(benchmark.features.tolist(), [[0.0, 1.0], [1.0, 0.0]])

            libsvm = root / "data.libsvm"
            libsvm.write_text("0 1:1.0 3:2.0\n1 2:4.0\n", encoding="utf-8")
            benchmark = BinaryPreprocessor(BinaryPreprocessingConfig(
                file_format="libsvm", target_column=0,
            )).fit_transform(libsvm)
            self.assertEqual(benchmark.features.shape, (2, 3))
            np.testing.assert_allclose(benchmark.features, [[1.0, 0.0, 2.0], [0.0, 4.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
