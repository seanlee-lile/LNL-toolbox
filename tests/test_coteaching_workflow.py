import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.catalog import load_recipe_config, paper_by_id, recipe_by_id, validate_config
from lnl_toolbox.cli import main as cli_main
from lnl_toolbox.training.experiment import run_experiment


def _cifar(size, split):
    labels = np.arange(size, dtype=np.int64) % 10
    images = np.zeros((size, 32, 32, 3), dtype=np.uint8)
    return CifarData(images, labels, tuple(map(str, range(10))), split, "cifar10")


def _config(epochs=2):
    return {
        "method": "coteaching",
        "seed": 7,
        "data": {
            "name": "cifar10",
            "root": "unused",
            "validation_size": 10,
            "max_train_samples": 20,
            "max_validation_samples": 10,
            "max_test_samples": 10,
            "augment": False,
        },
        "noise": {
            "name": "symmetric",
            "rate": 0.4,
            "seed": 17,
            "validation_targets": "noisy",
        },
        "loss": {"name": "ce"},
        "model": {"name": "tiny_cnn", "width": 4},
        "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "coteaching": {
            "model_count": 2,
            "noise_rate": 0.4,
            "initialization": {"peer_seed_offset": 1},
            "remember_schedule": {
                "name": "linear",
                "start": 1.0,
                "end": 0.6,
                "gradual_epochs": 2,
            },
            "selection": {
                "count_rule": "floor",
                "tie_break": "stable_sample_index",
            },
        },
        "loader": {"batch_size": 10, "num_workers": 0, "pin_memory": False},
        "evaluation": {
            "selection_split": "validation",
            "primary": "mean_peer_accuracy",
            "ensemble": "mean_probabilities",
        },
        "trainer": {"epochs": epochs, "device": "cpu"},
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CoTeachingWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.train_data = _cifar(40, "train")
        self.test_data = _cifar(20, "test")

    def _load_data(self, _root, split):
        return self.train_data if split == "train" else self.test_data

    def test_fresh_resume_and_completed_resume_preserve_dual_state(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10",
            side_effect=self._load_data,
        ):
            run_dir = run_experiment(_config(2), Path(directory) / "run")
            first = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            manifest = run_dir / "noise_manifest.npz"
            manifest_hash = _sha256(manifest)
            manifest_mtime = manifest.stat().st_mtime_ns
            rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            epochs = [row for row in rows if row["event"] == "epoch"]
            self.assertEqual(len(epochs), 2)
            self.assertEqual(set(first["model"]), {"a", "b"})
            self.assertEqual(set(first["optimizer"]), {"a", "b"})
            self.assertEqual(set(first["algorithm_private_state"]["schedulers"]), {"a", "b"})
            self.assertEqual(first["algorithm_private_state"]["method_identity"], "coteaching")
            self.assertEqual(first["run_state"]["step"], 4)
            self.assertEqual(first["algorithm_private_state"]["coteaching_state"]["optimizer_steps_a"], 4)
            self.assertEqual(first["algorithm_private_state"]["coteaching_state"]["optimizer_steps_b"], 4)
            self.assertTrue((run_dir / "best.pt").is_file())
            final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
            for key in ("test_accuracy_a", "test_accuracy_b", "test_mean_peer_accuracy", "test_accuracy_ensemble"):
                self.assertIn(key, final)

            run_experiment(_config(3), resume=run_dir / "last.pt")
            resumed = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["completed_epoch"], 2)
            self.assertEqual(resumed["run_state"]["step"], 6)
            self.assertEqual(_sha256(manifest), manifest_hash)
            self.assertEqual(manifest.stat().st_mtime_ns, manifest_mtime)
            self.assertAlmostEqual(
                json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-2])["remember_rate"],
                0.6,
            )

            checkpoint_hash = _sha256(run_dir / "last.pt")
            checkpoint_mtime = (run_dir / "last.pt").stat().st_mtime_ns
            metrics_hash = _sha256(run_dir / "metrics.jsonl")
            run_experiment(_config(3), resume=run_dir / "last.pt")
            self.assertEqual(_sha256(run_dir / "last.pt"), checkpoint_hash)
            self.assertEqual((run_dir / "last.pt").stat().st_mtime_ns, checkpoint_mtime)
            self.assertEqual(_sha256(run_dir / "metrics.jsonl"), metrics_hash)

    def test_resume_rejects_method_configuration_drift(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.data.sources.load_cifar10",
            side_effect=self._load_data,
        ):
            run_dir = run_experiment(_config(1), Path(directory) / "run")
            changed = _config(2)
            changed["coteaching"]["initialization"]["peer_seed_offset"] = 2
            with self.assertRaisesRegex(ValueError, "Co-teaching settings"):
                run_experiment(changed, resume=run_dir / "last.pt")

    def test_cli_dispatch_is_lazy_and_does_not_change_supervised_default(self):
        with patch(
            "lnl_toolbox.training.coteaching_experiment.run_coteaching_experiment",
            return_value=Path("coteaching-run"),
        ) as run:
            self.assertEqual(run_experiment(_config()), Path("coteaching-run"))
            run.assert_called_once()

    def test_full_run_recipe_uses_engineering_cifar10_protocol(self):
        recipe = recipe_by_id("cifar10-coteaching-reproduction")
        self.assertEqual(recipe.profile, "reproduction")
        self.assertEqual(recipe.method, "coteaching")
        self.assertEqual(recipe.runner, "coteaching")
        self.assertEqual(recipe.configuration_fidelity, "engineering")
        config = load_recipe_config(recipe)
        self.assertEqual(validate_config(config).name, "coteaching")
        self.assertEqual(config["method"], "coteaching")
        self.assertEqual(config["execution"]["runner"], "coteaching")
        self.assertEqual(config["data"]["name"], "cifar10")
        for key in (
            "max_train_samples",
            "max_validation_samples",
            "max_test_samples",
        ):
            self.assertNotIn(key, config["data"])
        self.assertEqual(config["noise"]["name"], "symmetric")
        self.assertEqual(config["noise"]["rate"], 0.2)
        self.assertEqual(config["noise"]["sampling"], "transition")
        self.assertEqual(config["noise"]["validation_targets"], "noisy")
        self.assertEqual(config["model"], {"name": "cifar_cnn8"})
        self.assertEqual(config["trainer"]["epochs"], 200)
        self.assertEqual(config["loader"]["batch_size"], 128)
        self.assertEqual(config["optimizer"]["name"], "adam")
        self.assertEqual(config["optimizer"]["lr"], 0.001)
        self.assertEqual(config["coteaching"]["model_count"], 2)
        self.assertEqual(config["coteaching"]["noise_rate"], 0.2)
        self.assertEqual(
            config["coteaching"]["remember_schedule"]["gradual_epochs"],
            10,
        )
        paper = paper_by_id("coteaching")
        exposed = {item.recipe_id: item for item in paper.configs}
        self.assertEqual(
            exposed[recipe.id].configuration_fidelity,
            "engineering",
        )

    def test_coteaching_dry_run_reports_method_specific_settings(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli_main.main([
                "run",
                "--recipe",
                "cifar10-coteaching-reproduction",
                "--dry-run",
            ])
        self.assertEqual(result, 0)
        text = output.getvalue()
        for expected in (
            "Co-teaching networks: 2",
            "Co-teaching batch size: 128",
            "Co-teaching optimizer: adam",
            "Co-teaching learning rate: 0.001",
            "Co-teaching Tk / gradual epochs: 10",
            "Co-teaching tau / noise rate: 0.2",
        ):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
