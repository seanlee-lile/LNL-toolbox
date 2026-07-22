import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.data.cifar import CifarData, default_data_root
from lnl_toolbox.data.noisy_dataset import NoisyTargetDataset
from lnl_toolbox.data.torch_cifar import TorchCifarDataset, stratified_split
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.models import TinyCNN
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device
from lnl_toolbox.training.checkpoint import load_checkpoint, save_checkpoint


class TorchTrainingTest(unittest.TestCase):
    def test_repository_data_path(self):
        self.assertEqual(default_data_root().name, "data")
        self.assertTrue((default_data_root() / "cifar10" / "data_batch_1").is_file())

    def test_dataset_shape_dtype_target_and_stable_index(self):
        data = CifarData(np.zeros((3, 32, 32, 3), dtype=np.uint8), np.array([2, 1, 0]),
                         ("a", "b", "c"), "train", "fixture")
        sample = TorchCifarDataset(data, [2])[0]
        self.assertEqual(sample["input"].shape, (3, 32, 32))
        self.assertEqual(sample["input"].dtype, torch.float32)
        self.assertEqual(sample["target"], 0)
        self.assertEqual(sample["index"], 2)

    def test_dataset_injects_noisy_target_by_global_index_without_clean_label_leak(self):
        data = CifarData(
            np.zeros((3, 32, 32, 3), dtype=np.uint8),
            np.array([2, 1, 0]),
            ("a", "b", "c"),
            "train",
            "fixture",
        )
        clean_dataset = TorchCifarDataset(data, [2])
        clean_sample = clean_dataset[0]
        sample = NoisyTargetDataset(
            clean_dataset,
            global_indices=np.array([0, 1, 2]),
            noisy_targets=np.array([1, 2, 2]),
        )[0]
        self.assertEqual(set(sample), {"input", "target", "index"})
        self.assertTrue(torch.equal(sample["input"], clean_sample["input"]))
        self.assertEqual(clean_sample["target"], 0)
        self.assertEqual(sample["target"], 2)
        self.assertEqual(sample["index"], clean_sample["index"])
        self.assertEqual(sample["index"], 2)

    def test_dataset_rejects_misaligned_noisy_targets(self):
        data = CifarData(
            np.zeros((3, 32, 32, 3), dtype=np.uint8),
            np.array([2, 1, 0]),
            ("a", "b", "c"),
            "train",
            "fixture",
        )
        dataset = TorchCifarDataset(data, [0, 2])
        with self.assertRaisesRegex(ValueError, "matching one-dimensional"):
            NoisyTargetDataset(dataset, [0, 2], [1])
        with self.assertRaisesRegex(ValueError, "no noisy target"):
            NoisyTargetDataset(dataset, [0], [1])

    def test_stratified_split_is_reproducible(self):
        labels = np.repeat(np.arange(10), 100)
        first = stratified_split(labels, 100, 9)
        second = stratified_split(labels, 100, 9)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(np.all(np.bincount(labels[first[1]]) == 10))

    def test_tinycnn_shape(self):
        self.assertEqual(TinyCNN(10, 8)(torch.randn(4, 3, 32, 32)).shape, (4, 10))

    def test_training_step_changes_parameters(self):
        model = TinyCNN(10, 8)
        algorithm = SupervisedClassificationAlgorithm(
            model, torch.optim.SGD(model.parameters(), lr=0.01), CrossEntropyLoss(), torch.device("cpu"))
        algorithm.setup(ExperimentContext(Path.cwd()))
        before = model.classifier.weight.detach().clone()
        result = algorithm.step(Batch({"input": torch.randn(4, 3, 32, 32),
                                       "target": torch.tensor([0, 1, 2, 3])}), RunState())
        self.assertTrue(np.isfinite(result.metrics["loss"]))
        self.assertFalse(torch.equal(before, model.classifier.weight))

    def test_each_p0_loss_completes_one_training_step(self):
        configs = [
            {"name": "ce"}, {"name": "gce"}, {"name": "nce"},
            {"name": "mae"}, {"name": "rce"}, {"name": "apl"},
        ]
        for config in configs:
            with self.subTest(name=config["name"]):
                model = TinyCNN(10, 8)
                algorithm = SupervisedClassificationAlgorithm(
                    model,
                    torch.optim.SGD(model.parameters(), lr=0.01),
                    build_builtin_loss(config),
                    torch.device("cpu"),
                )
                algorithm.setup(ExperimentContext(Path.cwd()))
                before = model.classifier.weight.detach().clone()
                result = algorithm.step(
                    Batch({
                        "input": torch.randn(4, 3, 32, 32),
                        "target": torch.tensor([0, 1, 2, 3]),
                    }),
                    RunState(),
                )
                self.assertTrue(np.isfinite(result.metrics["loss"]))
                self.assertFalse(torch.equal(before, model.classifier.weight))

    def test_training_rejects_scalar_loss(self):
        model = TinyCNN(10, 8)
        algorithm = SupervisedClassificationAlgorithm(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            torch.nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )
        with self.assertRaises(ValueError):
            algorithm.step(
                Batch({
                    "input": torch.randn(2, 3, 32, 32),
                    "target": torch.tensor([0, 1]),
                }),
                RunState(),
            )

    def test_device_selection(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(resolve_device("auto").type, expected)
        if torch.cuda.is_available():
            self.assertEqual(resolve_device("cuda").type, "cuda")

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyCNN(10, 8)
            optimizer = torch.optim.AdamW(model.parameters())
            algorithm = SupervisedClassificationAlgorithm(model, optimizer, CrossEntropyLoss(), torch.device("cpu"))
            algorithm.setup(ExperimentContext(Path(directory)))
            state = RunState(cycle=2, step=17)
            path = Path(directory) / "last.pt"
            saved = model.classifier.weight.detach().clone()
            save_checkpoint(path, algorithm, state, 2, {"seed": 1})
            with torch.no_grad():
                model.classifier.weight.zero_()
            restored, epoch, _ = load_checkpoint(path, algorithm, torch.device("cpu"))
            self.assertTrue(torch.equal(saved, model.classifier.weight))
            self.assertEqual(restored.step, 17)
            self.assertEqual(epoch, 2)

    def test_legacy_checkpoint_layouts_load_without_fabricated_best_metric(self):
        for layout in ("top-level", "nested"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                model = TinyCNN(10, 8)
                optimizer = torch.optim.AdamW(model.parameters())
                algorithm = SupervisedClassificationAlgorithm(
                    model, optimizer, CrossEntropyLoss(), torch.device("cpu")
                )
                state = RunState(cycle=1, step=7)
                payload = {
                    "format_version": 1,
                    "run_state": {
                        "cycle": state.cycle, "step": state.step, "phase": state.phase,
                        "metrics": state.metrics, "metadata": state.metadata,
                    },
                    "completed_epoch": 1,
                    "config": {"loss": {"name": "ce"}},
                }
                if layout == "top-level":
                    payload.update(algorithm.state_dict())
                else:
                    payload["algorithm"] = algorithm.state_dict()
                path = Path(directory) / "legacy.pt"
                torch.save(payload, path)
                restored, epoch, loaded = load_checkpoint(
                    path, algorithm, torch.device("cpu")
                )
                self.assertEqual(restored.step, 7)
                self.assertEqual(epoch, 1)
                self.assertEqual(loaded["best_validation_accuracy"], float("-inf"))
                self.assertTrue(loaded["_compatibility_warnings"])

    def test_checkpoint_requires_scheduler_state_when_scheduler_is_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyCNN(10, 8)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            algorithm = SupervisedClassificationAlgorithm(
                model, optimizer, CrossEntropyLoss(), torch.device("cpu")
            )
            path = Path(directory) / "legacy.pt"
            torch.save({
                "algorithm": algorithm.state_dict(),
                "run_state": {
                    "cycle": 0, "step": 1, "phase": "train", "metrics": {}, "metadata": {},
                },
                "completed_epoch": 0,
                "config": {},
            }, path)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            with self.assertRaisesRegex(ValueError, "scheduler"):
                load_checkpoint(
                    path, algorithm, torch.device("cpu"), scheduler=scheduler
                )


if __name__ == "__main__":
    unittest.main()
