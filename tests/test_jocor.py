from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

import torch
from torch import nn
import yaml

from lnl_toolbox.algorithms.jocor import (
    JoCoRAlgorithm,
    jocor_joint_scores,
    symmetric_kl_per_sample,
)
from lnl_toolbox.algorithms.multi_model import ModelGroup
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.core.hyperparameters import resolve_parameter_sampling
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.models.cifar_six_conv import CifarSixConvNet
from lnl_toolbox.plugins.builtin import (
    build_builtin_multi_model_algorithm,
)
from lnl_toolbox.selectors import SelectionInput, SmallLossSelector
from lnl_toolbox.training.checkpoint import load_checkpoint, save_checkpoint
from lnl_toolbox.training.adapters import NativeMultiModelRunner
from lnl_toolbox.training.runners import resolve_runner


class JoCoRMathTest(unittest.TestCase):
    def test_identical_logits_have_zero_symmetric_kl(self) -> None:
        logits = torch.tensor([[2.0, -1.0], [0.5, 0.2]])
        torch.testing.assert_close(
            symmetric_kl_per_sample(logits, logits),
            torch.zeros(2),
            atol=1e-7,
            rtol=0.0,
        )

    def test_joint_score_matches_formula_and_is_swap_invariant(self) -> None:
        first = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        second = torch.tensor([[1.0, 0.5], [0.2, 1.4]], requires_grad=True)
        targets = torch.tensor([0, 1])
        loss_1 = nn.functional.cross_entropy(first, targets, reduction="none")
        loss_2 = nn.functional.cross_entropy(second, targets, reduction="none")
        actual = jocor_joint_scores(
            loss_1, loss_2, first, second, lambda_=0.7
        )
        expected = 0.3 * (loss_1 + loss_2) + 0.7 * symmetric_kl_per_sample(
            first, second
        )
        torch.testing.assert_close(actual, expected)
        swapped = jocor_joint_scores(
            loss_2, loss_1, second, first, lambda_=0.7
        )
        torch.testing.assert_close(actual, swapped)
        actual.mean().backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())

    def test_floor_selection_uses_stable_global_index_ties(self) -> None:
        selector = SmallLossSelector(0.5, rounding="floor")
        result = selector.select(SelectionInput(
            scores=torch.ones(3),
            sample_indices=torch.tensor([9, 4, 7]),
        ))
        torch.testing.assert_close(
            result.selected_mask, torch.tensor([False, True, False])
        )
        self.assertEqual(result.metrics["selected_samples"], 1.0)
        with self.assertRaisesRegex(ValueError, "rounding"):
            SmallLossSelector(0.5, rounding="nearest")


class _CountingSelector:
    def __init__(self) -> None:
        self.calls = 0
        self.last_scores = None

    def select(self, selection_input):
        self.calls += 1
        self.last_scores = selection_input.scores.clone()
        return SmallLossSelector(0.5, rounding="floor").select(
            selection_input
        )


class JoCoRAlgorithmTest(unittest.TestCase):
    @staticmethod
    def _algorithm(selector=None):
        torch.manual_seed(3)
        models = ModelGroup({
            "model_1": nn.Linear(2, 2, bias=False),
            "model_2": nn.Linear(2, 2, bias=False),
        })
        optimizer = torch.optim.Adam(models.parameters(), lr=0.01)
        algorithm = JoCoRAlgorithm(
            models,
            optimizer,
            CrossEntropyLoss(),
            selector or SmallLossSelector(0.5, rounding="floor"),
            torch.device("cpu"),
            lambda_=0.9,
        )
        algorithm.setup(ExperimentContext(Path(".")))
        return algorithm

    def test_one_shared_selection_updates_both_models(self) -> None:
        selector = _CountingSelector()
        algorithm = self._algorithm(selector)
        before = {
            name: model.weight.detach().clone()
            for name, model in algorithm.models.models.items()
        }
        state = RunState(phase="train", cycle=0)
        algorithm.on_cycle_start(state)
        result = algorithm.step(Batch({
            "input": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
            ),
            "target": torch.tensor([0, 1, 0, 1]),
            "index": torch.tensor([8, 3, 5, 1]),
        }), state)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(result.metrics["selected_samples"], 2.0)
        for name, model in algorithm.models.models.items():
            self.assertFalse(torch.equal(before[name], model.weight))

    def test_plugin_and_checkpoint_roundtrip(self) -> None:
        source = self._algorithm()
        state = RunState(phase="train", cycle=2, step=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "last.pt")
            save_checkpoint(
                path,
                source,
                state,
                completed_epoch=2,
                config={"algorithm": {"name": "jocor", "lambda": 0.9}},
            )
            restored_models = ModelGroup({
                "model_1": nn.Linear(2, 2, bias=False),
                "model_2": nn.Linear(2, 2, bias=False),
            })
            restored_optimizer = torch.optim.Adam(
                restored_models.parameters(), lr=0.01
            )
            restored = build_builtin_multi_model_algorithm(
                {"name": "jocor", "lambda": 0.9},
                models=restored_models,
                optimizer=restored_optimizer,
                loss=CrossEntropyLoss(),
                selector=SmallLossSelector(0.5, rounding="floor"),
                device=torch.device("cpu"),
            )
            restored.setup(ExperimentContext(Path(directory)))
            loaded_state, completed, payload = load_checkpoint(
                path, restored, torch.device("cpu")
            )
            self.assertEqual(payload["format_version"], 2)
            self.assertEqual(completed, 2)
            self.assertEqual(loaded_state.step, 7)
            for name in source.models.models:
                torch.testing.assert_close(
                    source.models.models[name].weight,
                    restored.models.models[name].weight,
                )


class JoCoRConfigurationTest(unittest.TestCase):
    def test_official_model_shape(self) -> None:
        model = CifarSixConvNet(10)
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(output.shape, (2, 10))
        self.assertEqual(model.classifier.in_features, 256)

    def test_reproduction_config_records_official_sym50_setting(self) -> None:
        path = Path(
            "configs/experiment/jocor_cifar10_symmetric05_reproduction.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["method"], "jocor")
        self.assertEqual(resolve_runner(config).name, "multi_model")
        resolved, record = resolve_parameter_sampling(config)
        self.assertEqual(record.parameters["setting"], {
            "noise": "symmetric",
            "rate": 0.5,
            "lambda": 0.9,
        })
        self.assertEqual(resolved["trainer"]["epochs"], 200)
        self.assertEqual(resolved["loader"]["batch_size"], 128)
        self.assertEqual(resolved["optimizer"]["name"], "adam")
        self.assertEqual(resolved["optimizer"]["lr"], 0.001)
        self.assertEqual(resolved["algorithm"]["lambda"], 0.9)
        self.assertEqual(resolved["selector"]["rounding"], "floor")
        self.assertEqual(
            resolved["selector"]["keep_rate"]["warmup_epochs"], 9
        )
        self.assertEqual(resolved["scheduler"]["start_epoch"], 80)
        self.assertEqual(resolved["evaluation"]["report_last_epochs"], 10)
        self.assertTrue(resolved["evaluation"]["allow_test_selection"])

    def test_completed_resume_is_strict_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last.pt"
            torch.save({"completed_epoch": 1}, checkpoint)
            (root / "best.pt").write_bytes(b"best-checkpoint")
            (root / "metrics.jsonl").write_text(
                '{"event":"final","completed_epochs":2}\n', encoding="utf-8"
            )
            (root / "final_metrics.json").write_text(
                '{"method":"jocor","runner":"multi_model"}', encoding="utf-8"
            )
            (root / "noise_manifest.npz").write_bytes(b"noise-manifest")
            protected = tuple(
                root / name
                for name in (
                    "last.pt",
                    "best.pt",
                    "metrics.jsonl",
                    "final_metrics.json",
                    "noise_manifest.npz",
                )
            )
            before = {
                path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in protected
            }
            time.sleep(0.01)

            def unexpected_load():
                self.fail("completed resume must not load the native runner")

            spec = SimpleNamespace(
                name="multi_model",
                supports_resume=True,
                load=unexpected_load,
            )
            result = NativeMultiModelRunner(spec, method="jocor").fit(
                config={"method": "jocor", "trainer": {"epochs": 2}},
                output_dir=root,
                resume=checkpoint,
            )

            self.assertEqual(result.final_metrics["method"], "jocor")
            self.assertEqual(result.final_metrics["runner"], "multi_model")
            after = {
                path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in protected
            }
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
