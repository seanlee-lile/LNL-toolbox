from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.estimators import ReliabilityResult
from lnl_toolbox.catalog import load_recipe_config, recipe_by_id
from lnl_toolbox.training.experiment import run_experiment


def _data(size, split):
    rng = np.random.default_rng(12 if split == "train" else 13)
    images = rng.integers(0, 256, (size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % 10
    return CifarData(images, labels, tuple(map(str, range(10))), split, "cifar10")


def _data100(size, split):
    rng = np.random.default_rng(22 if split == "train" else 23)
    images = rng.integers(0, 256, (size, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(size, dtype=np.int64) % 100
    return CifarData(images, labels, tuple(map(str, range(100))), split, "cifar100")


def _config(epochs=1):
    return {
        "method": "dividemix", "seed": 4,
        "data": {"name": "cifar10", "root": "unused", "validation_size": 10, "max_train_samples": 20, "max_validation_samples": 10, "max_test_samples": 10, "augment": False},
        "noise": {"name": "symmetric", "rate": 0.2, "seed": 8, "validation_targets": "noisy"},
        "model": {"name": "tiny_cnn", "width": 4},
        "optimizer": {"name": "adam", "lr": 0.001}, "scheduler": {"name": "none"},
        "loader": {"batch_size": 5, "num_workers": 0, "pin_memory": False},
        "evaluation": {"selection_split": "validation", "primary": "ensemble_accuracy"},
        "dividemix": {
            "fidelity": "official_cifar_v1", "initialization": {"peer_seed_offset": 1},
            "warmup": {"epochs": 1, "confidence_penalty_weight": 1.0},
            "gmm": {"threshold": 0.5, "loss_history": {"name": "official_auto", "window_epochs": 5}},
            "mixmatch": {"augmentations": 2, "temperature": 0.5, "mixup_alpha": 4.0, "mixup_lambda_scope": "minibatch"},
            "objective": {"lambda_u": 25.0, "lambda_r": 1.0, "rampup_epochs": 16},
            "training": {"epochs": epochs}, "inference": {"ensemble": "official_logits_sum"},
        },
        "trainer": {"device": "cpu"}, "execution": {"runner": "dividemix"},
    }


class _FakeEstimator:
    def __init__(self, **_kwargs): pass
    def estimate(self, value):
        count = value.sample_indices.numel()
        scores = torch.where(torch.arange(count) % 2 == 0, 0.9, 0.1).to(torch.float64)
        return ReliabilityResult(value.sample_indices.detach(), scores, {
            "gmm_iterations": 1.0,
            "clean_component_mean": 0.25,
            "noisy_component_mean": 0.75,
        })


def _hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DivideMixWorkflowTest(unittest.TestCase):
    def setUp(self): self.train, self.test = _data(40, "train"), _data(20, "test")
    def load(self, _root, split): return self.train if split == "train" else self.test

    def test_formal_recipe_has_full_data_and_paper_oriented_contract(self):
        config = load_recipe_config(recipe_by_id("cifar10-dividemix-sym20"))
        self.assertFalse(
            {"max_train_samples", "max_validation_samples", "max_test_samples"}
            & set(config["data"])
        )
        self.assertEqual(config["data"]["name"], "cifar10")
        self.assertTrue(config["data"]["augment"])
        self.assertEqual(config["noise"]["name"], "symmetric")
        self.assertEqual(config["noise"]["rate"], 0.2)
        self.assertEqual(config["noise"]["sampling"], "global")
        self.assertEqual(config["model"], {"name": "preact_resnet18", "base_width": 64})
        self.assertEqual(
            config["optimizer"],
            {"name": "sgd", "lr": 0.02, "momentum": 0.9, "weight_decay": 0.0005},
        )
        self.assertEqual(config["scheduler"], {"name": "multistep", "milestones": [150], "gamma": 0.1})
        self.assertEqual(config["loader"]["batch_size"], 128)
        method = config["dividemix"]
        self.assertEqual(method["warmup"]["epochs"], 10)
        self.assertEqual(method["training"]["epochs"], 300)
        self.assertEqual(method["gmm"]["threshold"], 0.5)
        self.assertEqual(method["mixmatch"]["augmentations"], 2)
        self.assertEqual(method["mixmatch"]["temperature"], 0.5)
        self.assertEqual(method["mixmatch"]["mixup_alpha"], 4.0)
        self.assertEqual(method["objective"]["lambda_u"], 25.0)
        self.assertEqual(method["objective"]["lambda_r"], 1.0)

    def test_fresh_extension_and_completed_noop(self):
        with tempfile.TemporaryDirectory() as directory, patch("lnl_toolbox.data.sources.load_cifar10", side_effect=self.load), patch("lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator", _FakeEstimator):
            run_dir = run_experiment(_config(1), Path(directory) / "run")
            payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["algorithm"]["dividemix_state"]["phase"], "completed")
            self.assertEqual(payload["algorithm"]["dividemix_state"]["main_completed_epochs"], 1)
            self.assertEqual(set(payload["algorithm"]["model"]), {"a", "b"})
            first_artifact = run_dir / "dividemix_epoch_0001.npz"; first_hash, first_mtime = _hash(first_artifact), first_artifact.stat().st_mtime_ns
            run_experiment(_config(2), resume=run_dir / "last.pt")
            resumed = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["algorithm"]["dividemix_state"]["main_completed_epochs"], 2)
            self.assertEqual(_hash(first_artifact), first_hash); self.assertEqual(first_artifact.stat().st_mtime_ns, first_mtime)
            checkpoint_hash, metrics_hash = _hash(run_dir / "last.pt"), _hash(run_dir / "metrics.jsonl")
            run_experiment(_config(2), resume=run_dir / "last.pt")
            self.assertEqual(_hash(run_dir / "last.pt"), checkpoint_hash); self.assertEqual(_hash(run_dir / "metrics.jsonl"), metrics_hash)
            final = json.loads((run_dir / "final_metrics.json").read_text())
            self.assertEqual(final["ensemble"], "official_logits_sum")

    def test_corrupt_ready_artifact_fails_instead_of_refitting(self):
        with tempfile.TemporaryDirectory() as directory, patch("lnl_toolbox.data.sources.load_cifar10", side_effect=self.load), patch("lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator", _FakeEstimator):
            run_dir = run_experiment(_config(1), Path(directory) / "run")
            payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            payload["algorithm"]["dividemix_state"]["phase"] = "co_divide_ready"
            atomic = run_dir / "last.pt"
            torch.save(payload, atomic)
            (run_dir / payload["algorithm"]["dividemix_state"]["current_artifact"]).write_bytes(b"broken")
            with self.assertRaises(Exception):
                run_experiment(_config(2), resume=atomic)

    def test_network_a_ready_resume_does_not_repeat_a(self):
        import lnl_toolbox.training.dividemix_experiment as workflow
        original = workflow._train_peer_epoch
        failed = {"value": False}
        def interrupt_b(*args, **kwargs):
            peer = args[1]
            if peer == "b" and not failed["value"]:
                failed["value"] = True
                raise RuntimeError("controlled B interruption")
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with patch("lnl_toolbox.data.sources.load_cifar10", side_effect=self.load), patch("lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator", _FakeEstimator), patch("lnl_toolbox.training.dividemix_experiment._train_peer_epoch", side_effect=interrupt_b):
                with self.assertRaisesRegex(RuntimeError, "controlled B"):
                    run_experiment(_config(1), run_dir)
            interrupted = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(interrupted["algorithm"]["dividemix_state"]["phase"], "network_a_ready")
            steps_a = interrupted["algorithm"]["dividemix_state"]["optimizer_steps_a"]
            with patch("lnl_toolbox.data.sources.load_cifar10", side_effect=self.load), patch("lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator", _FakeEstimator):
                run_experiment(_config(1), resume=run_dir / "last.pt")
            resumed = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["algorithm"]["dividemix_state"]["optimizer_steps_a"], steps_a)

    def test_cifar100_lightweight_workflow(self):
        train, test = _data100(220, "train"), _data100(100, "test")
        value = _config(1); value["data"].update({"name": "cifar100", "validation_size": 100, "max_train_samples": 100, "max_validation_samples": 100, "max_test_samples": 100})
        with tempfile.TemporaryDirectory() as directory, patch("lnl_toolbox.data.sources.load_cifar100", side_effect=lambda _root, split: train if split == "train" else test), patch("lnl_toolbox.algorithms.dividemix.gmm.DivideMixGMMCleanProbabilityEstimator", _FakeEstimator):
            run_dir = run_experiment(value, Path(directory) / "run")
            final = json.loads((run_dir / "final_metrics.json").read_text())
            self.assertEqual(final["completed_epochs"], 1)


if __name__ == "__main__": unittest.main()
