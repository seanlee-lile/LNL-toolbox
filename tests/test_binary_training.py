import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from lnl_toolbox.data.binary_benchmarks import BinaryBenchmark, stratified_binary_splits
from lnl_toolbox.training.binary_experiment import (
    BinaryTensorDataset,
    build_binary_linear,
    build_binary_mlp,
    train_binary_epoch,
)


class BinaryTrainingTest(unittest.TestCase):
    def test_dataset_and_epoch_keep_stable_indices(self) -> None:
        benchmark = BinaryBenchmark(np.asarray([[0.0], [1.0], [0.2], [0.8]]), np.asarray([0, 1, 0, 1]), "fixture")
        dataset = BinaryTensorDataset(benchmark)
        self.assertEqual(int(dataset[2]["index"]), 2)
        model = build_binary_mlp(1, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        result = train_binary_epoch(model, DataLoader(dataset, batch_size=2), optimizer)
        self.assertEqual(result["samples"], 4.0)

    def test_stratified_splits_are_deterministic(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1])
        first = stratified_binary_splits(labels, 3, 9)
        second = stratified_binary_splits(labels, 3, 9)
        self.assertEqual([(a.tolist(), b.tolist()) for a, b in first], [(a.tolist(), b.tolist()) for a, b in second])

    def test_linear_classifier_is_available_for_paper_logistic_path(self) -> None:
        model = build_binary_linear(2)
        self.assertEqual(tuple(model(torch.zeros(3, 2)).shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
