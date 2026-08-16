import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lnl_toolbox.data.cifar import CifarData
from lnl_toolbox.training.experiment import run_experiment


def _cifar(size, split, classes=10):
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(np.zeros((size, 32, 32, 3), dtype=np.uint8), labels,
                     tuple(map(str, range(classes))), split, f"cifar{classes}")


def _config(epochs=2, dataset="cifar10"):
    classes = 10 if dataset == "cifar10" else 100
    return {"method": "cnlcu", "seed": 7,
        "data": {"name": dataset, "root": "unused", "validation_size": classes,
                 "max_train_samples": classes * 2, "max_validation_samples": classes,
                 "max_test_samples": classes, "augment": False},
        "noise": {"name": "symmetric", "rate": 0.4, "seed": 17, "validation_targets": "noisy"},
        "loss": {"name": "ce"}, "model": {"name": "tiny_cnn", "width": 2},
        "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "cnlcu": {"variant": "soft", "model_count": 2, "noise_rate": 0.4,
            "initialization": {"peer_seed_offset": 1},
            "remember_schedule": {"name": "linear", "start": 1.0, "end": 0.6, "gradual_epochs": 2},
            "history": {"window_size": 2, "storage_dtype": "float32"},
            "uncertainty": {"sigma_squared": 0.01},
            "selection": {"count_rule": "floor", "tie_break": "stable_sample_index"}},
        "loader": {"batch_size": classes, "num_workers": 0, "pin_memory": False},
        "evaluation": {"selection_split": "validation", "primary": "mean_peer_accuracy", "ensemble": "mean_probabilities"},
        "trainer": {"epochs": epochs, "device": "cpu"}}


def _hard_config(epochs=2, dataset="cifar10"):
    result = _config(epochs, dataset)
    result["cnlcu"] = {
        "variant": "hard", "hard_fidelity": "paper_formula_corrected_lof",
        "model_count": 2, "noise_rate": 0.4,
        "initialization": {"peer_seed_offset": 1},
        "remember_schedule": {
            "name": "linear", "start": 1.0, "end": 0.6,
            "gradual_epochs": 2,
        },
        "history": {"window_size": 3, "storage_dtype": "float32"},
        "uncertainty": {
            "tau_min": 0.0001,
            "loss_upper_bound": {"mode": "fixed", "value": 10.0},
        },
        "truncation": {
            "method": "lof", "n_neighbors": 2, "contamination": 0.1,
            "minimum_observations": 3,
        },
        "selection": {"count_rule": "floor", "tie_break": "stable_sample_index"},
    }
    return result


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_nested_equal(test_case, left, right):
    test_case.assertEqual(type(left), type(right))
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        test_case.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test_case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test_case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(test_case, left_item, right_item)
    else:
        test_case.assertEqual(left, right)


class CNLCUWorkflowTest(unittest.TestCase):
    def _run_case(self, dataset="cifar10", variant="soft"):
        classes = 10 if dataset == "cifar10" else 100
        config_factory = _config if variant == "soft" else _hard_config
        train, test = _cifar(classes * 4, "train", classes), _cifar(classes * 2, "test", classes)
        loader_name = "load_cifar10" if dataset == "cifar10" else "load_cifar100"
        with tempfile.TemporaryDirectory() as directory, patch(
            f"lnl_toolbox.training.cnlcu_experiment.{loader_name}",
            side_effect=lambda _root, split: train if split == "train" else test,
        ):
            run_dir = run_experiment(config_factory(2, dataset), Path(directory) / "run")
            first = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            state = first["algorithm_private_state"]["cnlcu_state"]
            self.assertEqual(first["algorithm_private_state"]["method_identity"], "cnlcu")
            self.assertEqual(set(state), {"history_a", "history_b", "optimizer_steps_a", "optimizer_steps_b"})
            self.assertFalse(torch.equal(state["history_a"]["values"], state["history_b"]["values"]))
            manifest_hash, manifest_mtime = _sha(run_dir / "noise_manifest.npz"), (run_dir / "noise_manifest.npz").stat().st_mtime_ns
            run_experiment(config_factory(3, dataset), resume=run_dir / "last.pt")
            resumed = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["completed_epoch"], 2)
            self.assertEqual(resumed["run_state"]["step"], 6)
            self.assertEqual(_sha(run_dir / "noise_manifest.npz"), manifest_hash)
            self.assertEqual((run_dir / "noise_manifest.npz").stat().st_mtime_ns, manifest_mtime)
            uninterrupted_dir = run_experiment(config_factory(3, dataset), Path(directory) / "uninterrupted")
            uninterrupted = torch.load(uninterrupted_dir / "last.pt", map_location="cpu", weights_only=False)
            for peer in ("a", "b"):
                for name, value in resumed["model"][peer].items():
                    torch.testing.assert_close(value, uninterrupted["model"][peer][name])
            self.assertEqual(resumed["run_state"], uninterrupted["run_state"])
            _assert_nested_equal(self, resumed["optimizer"], uninterrupted["optimizer"])
            _assert_nested_equal(
                self,
                resumed["algorithm_private_state"]["schedulers"],
                uninterrupted["algorithm_private_state"]["schedulers"],
            )
            _assert_nested_equal(
                self,
                resumed["algorithm_private_state"]["cnlcu_state"],
                uninterrupted["algorithm_private_state"]["cnlcu_state"],
            )
            self.assertEqual(resumed["component_states"], uninterrupted["component_states"])
            resumed_epochs = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "epoch"
            ]
            uninterrupted_epochs = [
                json.loads(line)
                for line in (uninterrupted_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "epoch"
            ]
            self.assertEqual(resumed_epochs, uninterrupted_epochs)
            for row in resumed_epochs:
                for key in (
                    "selected_by_a_ratio", "selected_by_b_ratio",
                    "train_gradient_norm_a", "train_gradient_norm_b",
                    "train_gradient_norm_a_max", "train_gradient_norm_b_max",
                    "train_parameter_norm_a", "train_parameter_norm_b",
                    "train_uncertainty_score_min_a",
                    "train_uncertainty_score_max_a",
                    "train_history_length_min_a", "train_history_length_max_a",
                    "history_window_start_epoch", "history_window_epoch_count",
                ):
                    self.assertIn(key, row)
                    self.assertTrue(np.isfinite(row[key]), key)
            checkpoint_hash, metrics_hash = _sha(run_dir / "last.pt"), _sha(run_dir / "metrics.jsonl")
            run_experiment(config_factory(3, dataset), resume=run_dir / "last.pt")
            self.assertEqual(_sha(run_dir / "last.pt"), checkpoint_hash)
            self.assertEqual(_sha(run_dir / "metrics.jsonl"), metrics_hash)
            final = json.loads((run_dir / "final_metrics.json").read_text())
            self.assertIn("test_mean_peer_accuracy", final)

    def test_cifar10_fresh_resume_and_completed_noop(self): self._run_case("cifar10")
    def test_cifar100_lightweight_workflow(self): self._run_case("cifar100")
    def test_hard_cifar10_fresh_resume_and_completed_noop(self):
        self._run_case("cifar10", "hard")
    def test_hard_cifar100_lightweight_workflow(self):
        self._run_case("cifar100", "hard")

    def test_resume_rejects_method_and_history_configuration_drift(self):
        train, test = _cifar(40, "train"), _cifar(20, "test")
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.cnlcu_experiment.load_cifar10",
            side_effect=lambda _root, split: train if split == "train" else test,
        ):
            run_dir = run_experiment(_config(1), Path(directory) / "run")
            changed = _config(2); changed["cnlcu"]["history"]["window_size"] = 3
            with self.assertRaisesRegex(ValueError, "CNLCU settings"):
                run_experiment(changed, resume=run_dir / "last.pt")

    def test_lazy_cli_dispatch(self):
        with patch("lnl_toolbox.training.cnlcu_experiment.run_cnlcu_experiment",
                   return_value=Path("cnlcu-run")) as run:
            self.assertEqual(run_experiment(_config()), Path("cnlcu-run")); run.assert_called_once()

    def test_soft_hard_resume_and_hard_detector_drift_are_rejected(self):
        train, test = _cifar(40, "train"), _cifar(20, "test")
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.cnlcu_experiment.load_cifar10",
            side_effect=lambda _root, split: train if split == "train" else test,
        ):
            soft_dir = run_experiment(_config(1), Path(directory) / "soft")
            with self.assertRaisesRegex(ValueError, "CNLCU settings"):
                run_experiment(_hard_config(2), resume=soft_dir / "last.pt")
            hard_dir = run_experiment(_hard_config(1), Path(directory) / "hard")
            with self.assertRaisesRegex(ValueError, "CNLCU settings"):
                run_experiment(_config(2), resume=hard_dir / "last.pt")
            drift = _hard_config(2)
            drift["cnlcu"]["truncation"]["contamination"] = 0.2
            with self.assertRaisesRegex(ValueError, "CNLCU settings"):
                run_experiment(drift, resume=hard_dir / "last.pt")


if __name__ == "__main__": unittest.main()
