import tempfile
import unittest
from pathlib import Path

import torch

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.core import ExperimentContext, RunState
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.training.clean_baseline import (
    build_clean_model,
    build_clean_optimizer,
    build_clean_scheduler,
    run_clean_experiment,
)
from lnl_toolbox.training.checkpoint import load_checkpoint, save_checkpoint


class CleanBaselineTest(unittest.TestCase):
    def test_clean_runner_rejects_noise_manifest_configuration(self):
        with self.assertRaisesRegex(ValueError, "does not accept noise configuration"):
            run_clean_experiment({"noise": {"manifest": "noise.npz"}})

    def test_supported_models_produce_class_logits(self):
        inputs = torch.randn(2, 3, 32, 32)
        for name in ("tiny_cnn", "resnet18", "preact_resnet18"):
            config = {"name": name, "width": 8, "base_width": 8}
            with self.subTest(name=name):
                self.assertEqual(build_clean_model(config, 10)(inputs).shape, (2, 10))

    def test_component_builders_reject_unknown_names(self):
        with self.assertRaises(ValueError):
            build_clean_model({"name": "unknown"}, 10)
        model = build_clean_model({"name": "tiny_cnn", "width": 8}, 10)
        with self.assertRaises(ValueError):
            build_clean_optimizer(model, {"name": "unknown", "lr": 0.1})

    def test_scheduler_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            model = build_clean_model({"name": "tiny_cnn", "width": 8}, 10)
            optimizer = build_clean_optimizer(model, {"name": "sgd", "lr": 0.1, "momentum": 0.9})
            scheduler = build_clean_scheduler(optimizer, {"name": "cosine", "t_max": 5}, 5)
            optimizer.step()
            scheduler.step()
            saved_lr = optimizer.param_groups[0]["lr"]
            path = Path(directory) / "last.pt"
            algorithm = SupervisedClassificationAlgorithm(
                model, optimizer, CrossEntropyLoss(), torch.device("cpu")
            )
            save_checkpoint(
                path, algorithm, RunState(cycle=1, step=9), 1, {},
                scheduler=scheduler, best_epoch=0, best_validation_accuracy=0.5,
            )

            new_model = build_clean_model({"name": "tiny_cnn", "width": 8}, 10)
            new_optimizer = build_clean_optimizer(new_model, {"name": "sgd", "lr": 0.1, "momentum": 0.9})
            new_scheduler = build_clean_scheduler(new_optimizer, {"name": "cosine", "t_max": 5}, 5)
            new_algorithm = SupervisedClassificationAlgorithm(
                new_model, new_optimizer, CrossEntropyLoss(), torch.device("cpu")
            )
            new_algorithm.setup(ExperimentContext(Path(directory)))
            state, _, restored = load_checkpoint(
                path, new_algorithm, torch.device("cpu"), scheduler=new_scheduler
            )
            self.assertAlmostEqual(new_optimizer.param_groups[0]["lr"], saved_lr)
            self.assertEqual(new_scheduler.last_epoch, scheduler.last_epoch)
            self.assertEqual(state.step, 9)
            self.assertEqual(restored["best_validation_accuracy"], 0.5)

    def test_multistep_scheduler(self):
        model = build_clean_model({"name": "tiny_cnn", "width": 8}, 10)
        optimizer = build_clean_optimizer(model, {"name": "sgd", "lr": 0.1})
        scheduler = build_clean_scheduler(
            optimizer, {"name": "multistep", "milestones": [2, 4], "gamma": 0.1}, 5
        )
        self.assertIsNotNone(scheduler)


if __name__ == "__main__":
    unittest.main()
