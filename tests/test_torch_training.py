import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.core import SoftTargetResult
from lnl_toolbox.data.cifar import CifarData, default_data_root
from lnl_toolbox.data.noisy_dataset import NoisyTargetDataset
from lnl_toolbox.data.torch_cifar import (
    TorchCifarDataset,
    build_cifar_transform,
    cifar_pixel_mean,
    stratified_split,
    train_validation_split,
)
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.models import TinyCNN
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device
from lnl_toolbox.selectors import AllSelector, SelectionResult, SmallLossSelector
from lnl_toolbox.training.checkpoint import (
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from lnl_toolbox.training.experiment import build_model


class TorchTrainingTest(unittest.TestCase):
    @staticmethod
    def _target_algorithm(provider):
        model = torch.nn.Linear(2, 2, bias=False)
        algorithm = SupervisedClassificationAlgorithm(
            model,
            torch.optim.SGD(model.parameters(), lr=0.1),
            CrossEntropyLoss(),
            torch.device("cpu"),
            target_provider=provider,
        )
        algorithm.setup(ExperimentContext(Path.cwd()))
        return algorithm

    @staticmethod
    def _target_batch():
        return Batch({
            "input": torch.eye(2),
            "target": torch.tensor([0, 1]),
            "index": torch.tensor([9, 3]),
        })

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

    def test_legacy_random_split_matches_choice_then_complement(self):
        labels = np.arange(20) % 3
        train, validation = train_validation_split(
            labels,
            4,
            7,
            strategy="random",
            rng="numpy_legacy",
        )
        random = np.random.RandomState(7)
        expected_train = random.choice(20, 16, replace=False)
        expected_validation = np.delete(np.arange(20), expected_train)
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(validation, expected_validation)

    def test_tinycnn_shape(self):
        self.assertEqual(TinyCNN(10, 8)(torch.randn(4, 3, 32, 32)).shape, (4, 10))

    def test_cifar_resnet34_shape_and_stage_depths(self):
        model = build_model(
            {"name": "resnet34", "base_width": 8}, num_classes=10
        )
        self.assertEqual(
            tuple(len(layer) for layer in (
                model.layer1, model.layer2, model.layer3, model.layer4
            )),
            (3, 4, 6, 3),
        )
        self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 10))

    def test_cifar_cnn8_shape_and_reference_channels(self):
        model = build_model({"name": "cifar_cnn8"}, num_classes=10)
        convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        self.assertEqual([module.out_channels for module in convolutions], [64, 64, 128, 128, 196, 196])
        self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 10))

    def test_gce2018_preprocessing_subtracts_training_pixel_mean(self):
        images = np.full((2, 32, 32, 3), 128, dtype=np.uint8)
        mean = cifar_pixel_mean(images)
        transform = build_cifar_transform(
            False, preprocessing="gce2018", pixel_mean=mean
        )
        data = CifarData(
            images,
            np.array([0, 1]),
            ("a", "b"),
            "train",
            "fixture",
        )
        sample = TorchCifarDataset(data, [0], transform=transform)[0]
        torch.testing.assert_close(sample["input"], torch.zeros_like(sample["input"]))

    def test_standard_preprocessing_accepts_explicit_normalization(self):
        transform = build_cifar_transform(
            False,
            normalization_mean=(0.5, 0.5, 0.5),
            normalization_std=(0.25, 0.25, 0.25),
        )
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        output = transform(CifarData(
            image[None],
            np.array([0]),
            ("a",),
            "train",
            "fixture",
        ).images[0])
        expected = (128.0 / 255.0 - 0.5) / 0.25
        torch.testing.assert_close(
            output,
            torch.full_like(output, expected),
        )

    def test_training_step_changes_parameters(self):
        model = TinyCNN(10, 8)
        algorithm = SupervisedClassificationAlgorithm(
            model, torch.optim.SGD(model.parameters(), lr=0.01), CrossEntropyLoss(), torch.device("cpu"))
        algorithm.setup(ExperimentContext(Path.cwd()))
        before = model.classifier.weight.detach().clone()
        result = algorithm.step(Batch({"input": torch.randn(4, 3, 32, 32),
                                       "target": torch.tensor([0, 1, 2, 3]),
                                       "index": torch.arange(4)}), RunState())
        self.assertTrue(np.isfinite(result.metrics["loss"]))
        self.assertEqual(result.metrics["selected_ratio"], 1.0)
        self.assertFalse(torch.equal(before, model.classifier.weight))

    def test_target_sample_indices_exact_match_passes(self):
        class Provider:
            def resolve(self, target_input):
                return SoftTargetResult(
                    targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    sample_indices=target_input.sample_indices,
                )

        result = self._target_algorithm(Provider()).step(
            self._target_batch(), RunState()
        )
        self.assertTrue(np.isfinite(result.metrics["loss"]))

    def test_target_sample_indices_permutation_rejected(self):
        class Provider:
            def resolve(self, target_input):
                return SoftTargetResult(
                    targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    sample_indices=target_input.sample_indices.flip(0),
                )

        with self.assertRaisesRegex(ValueError, "batch-order aligned"):
            self._target_algorithm(Provider()).step(
                self._target_batch(), RunState()
            )

    def test_target_sample_indices_duplicate_rejected(self):
        class Provider:
            def resolve(self, target_input):
                return SoftTargetResult(
                    targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    sample_indices=torch.tensor([9, 9]),
                )

        with self.assertRaisesRegex(ValueError, "unique"):
            self._target_algorithm(Provider()).step(
                self._target_batch(), RunState()
            )

    def test_target_sample_indices_length_mismatch_rejected(self):
        class Provider:
            def resolve(self, target_input):
                return SoftTargetResult(
                    targets=torch.tensor([[1.0, 0.0]]),
                    sample_indices=torch.tensor([9]),
                )

        with self.assertRaisesRegex(ValueError, "length mismatch"):
            self._target_algorithm(Provider()).step(
                self._target_batch(), RunState()
            )

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
                        "index": torch.arange(4),
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
                    "index": torch.arange(2),
                }),
                RunState(),
            )

    def test_selector_receives_detached_scores_and_only_selected_loss_backpropagates(self):
        class FirstSampleSelector:
            def __init__(self) -> None:
                self.scores_require_grad = None
                self.sample_indices = None
                self.epoch = None

            def select(self, selection_input):
                self.scores_require_grad = selection_input.scores.requires_grad
                self.sample_indices = selection_input.sample_indices.detach().clone()
                self.epoch = selection_input.metadata.get("epoch")
                return SelectionResult(
                    selected_mask=torch.tensor([True, False]),
                    metrics={"selected_samples": 1.0, "selected_ratio": 0.5},
                )

        model = torch.nn.Linear(2, 2, bias=False)
        selector = FirstSampleSelector()
        algorithm = SupervisedClassificationAlgorithm(
            model,
            torch.optim.SGD(model.parameters(), lr=0.1),
            CrossEntropyLoss(),
            torch.device("cpu"),
            selector=selector,
        )
        algorithm.setup(ExperimentContext(Path.cwd()))
        before = model.weight.detach().clone()
        result = algorithm.step(
            Batch({
                "input": torch.eye(2),
                "target": torch.tensor([0, 1]),
                "index": torch.tensor([9, 3]),
            }),
            RunState(cycle=4),
        )

        self.assertIs(selector.scores_require_grad, False)
        self.assertTrue(torch.equal(selector.sample_indices, torch.tensor([9, 3])))
        self.assertEqual(selector.epoch, 4)
        self.assertFalse(torch.equal(before[:, 0], model.weight[:, 0]))
        self.assertTrue(torch.equal(before[:, 1], model.weight[:, 1]))
        self.assertEqual(result.metrics["selected_samples"], 1.0)
        self.assertEqual(result.metrics["selected_ratio"], 0.5)

    def test_small_loss_selector_completes_training_step(self):
        model = TinyCNN(10, 8)
        algorithm = SupervisedClassificationAlgorithm(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            CrossEntropyLoss(),
            torch.device("cpu"),
            selector=SmallLossSelector(keep_rate=0.5),
        )
        algorithm.setup(ExperimentContext(Path.cwd()))
        result = algorithm.step(
            Batch({
                "input": torch.randn(4, 3, 32, 32),
                "target": torch.tensor([0, 1, 2, 3]),
                "index": torch.tensor([100, 20, 80, 40]),
            }),
            RunState(),
        )
        self.assertEqual(result.metrics["selected_samples"], 2.0)
        self.assertEqual(result.metrics["selected_ratio"], 0.5)

    def test_small_loss_reducer_matches_selected_mean_and_parameter_update(self):
        selected_model = torch.nn.Linear(2, 2, bias=False)
        reference_model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            selected_model.weight.copy_(torch.tensor([[0.4, -0.2], [-0.1, 0.3]]))
        reference_model.load_state_dict(selected_model.state_dict())
        inputs = torch.eye(2)
        targets = torch.tensor([0, 0])

        algorithm = SupervisedClassificationAlgorithm(
            selected_model,
            torch.optim.SGD(selected_model.parameters(), lr=0.1),
            CrossEntropyLoss(),
            torch.device("cpu"),
            selector=SmallLossSelector(keep_rate=0.5),
        )
        algorithm.setup(ExperimentContext(Path.cwd()))
        result = algorithm.step(
            Batch({
                "input": inputs,
                "target": targets,
                "index": torch.tensor([9, 3]),
            }),
            RunState(),
        )

        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_losses = CrossEntropyLoss()(reference_model(inputs), targets)
        reference_loss = reference_losses[reference_losses.argmin()].mean()
        reference_loss.backward()
        reference_optimizer.step()

        self.assertAlmostEqual(result.metrics["loss"], reference_loss.item(), places=7)
        self.assertEqual(result.metrics["selected_samples"], 1.0)
        self.assertEqual(result.metrics["selected_ratio"], 0.5)
        for selected, reference in zip(
            selected_model.parameters(), reference_model.parameters()
        ):
            self.assertTrue(torch.equal(selected, reference))

    def test_all_selector_matches_full_batch_mean_and_parameter_update(self):
        torch.manual_seed(23)
        selected_model = torch.nn.Linear(3, 2)
        reference_model = torch.nn.Linear(3, 2)
        reference_model.load_state_dict(selected_model.state_dict())
        inputs = torch.tensor([
            [1.0, 0.0, -1.0],
            [0.0, 2.0, 1.0],
            [-1.0, 1.0, 0.5],
        ])
        targets = torch.tensor([0, 1, 0])

        selected_optimizer = torch.optim.SGD(selected_model.parameters(), lr=0.05)
        algorithm = SupervisedClassificationAlgorithm(
            selected_model,
            selected_optimizer,
            CrossEntropyLoss(),
            torch.device("cpu"),
            selector=AllSelector(),
        )
        algorithm.setup(ExperimentContext(Path.cwd()))
        result = algorithm.step(
            Batch({
                "input": inputs,
                "target": targets,
                "index": torch.tensor([50, 10, 30]),
            }),
            RunState(),
        )

        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.05)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = CrossEntropyLoss()(reference_model(inputs), targets).mean()
        reference_loss.backward()
        reference_optimizer.step()

        self.assertAlmostEqual(result.metrics["loss"], reference_loss.item(), places=7)
        self.assertEqual(result.metrics["selected_ratio"], 1.0)
        for selected, reference in zip(
            selected_model.parameters(), reference_model.parameters()
        ):
            self.assertTrue(torch.allclose(selected, reference, atol=0.0, rtol=0.0))

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

    def test_restore_cpu_rng_state_on_cpu(self):
        original = torch.get_rng_state()
        expected = torch.Generator().manual_seed(1234).get_state()
        try:
            restore_rng_state({"torch": expected.clone()})
            self.assertTrue(torch.equal(torch.get_rng_state(), expected))
        finally:
            torch.set_rng_state(original)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_restore_cpu_rng_state_loaded_on_cuda_device(self):
        original = torch.get_rng_state()
        expected = torch.Generator().manual_seed(5678).get_state()
        try:
            restore_rng_state({"torch": expected.to("cuda")})
            self.assertTrue(torch.equal(torch.get_rng_state(), expected))
        finally:
            torch.set_rng_state(original)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_restore_cuda_rng_state_list_normalizes_devices(self):
        original = torch.cuda.get_rng_state_all()
        try:
            torch.cuda.manual_seed_all(9012)
            expected = torch.cuda.get_rng_state_all()
            torch.cuda.set_rng_state_all(original)
            restore_rng_state({
                "cuda": [value.to("cuda") for value in expected],
            })
            actual = torch.cuda.get_rng_state_all()
            self.assertEqual(len(actual), len(expected))
            for actual_state, expected_state in zip(actual, expected):
                self.assertEqual(actual_state.device.type, "cpu")
                self.assertTrue(torch.equal(actual_state, expected_state))
        finally:
            torch.cuda.set_rng_state_all(original)

    def test_restore_rng_state_rejects_invalid_tensor_contract(self):
        with self.assertRaisesRegex(TypeError, "torch.uint8"):
            restore_rng_state({
                "torch": torch.zeros(8, dtype=torch.int64),
            })
        with self.assertRaisesRegex(ValueError, "non-empty 1D"):
            restore_rng_state({
                "torch": torch.zeros((2, 4), dtype=torch.uint8),
            })
        with self.assertRaisesRegex(TypeError, "list or tuple"):
            restore_rng_state({
                "cuda": torch.zeros(8, dtype=torch.uint8),
            })
        with self.assertRaisesRegex(TypeError, "must be a tensor"):
            restore_rng_state({"cuda": [b"not a tensor"]})

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
