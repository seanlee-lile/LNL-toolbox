from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from lnl_toolbox.algorithms.t_revision import (
    TRevisionAlgorithm,
    TRevisionConfig,
    TRevisionPhase,
)
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.experiment import run_experiment
from lnl_toolbox.training.t_revision_experiment import _preflight_model_output


class _Classifier(nn.Module):
    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(classes) * 2.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight.t()


def _records(*, clean_offset: int = 0):
    records = []
    for position, target in enumerate((0, 0, 1, 1, 2, 2)):
        records.append({
            "input": torch.nn.functional.one_hot(
                torch.tensor(target), num_classes=3
            ).float(),
            "target": torch.tensor(target),
            "index": torch.tensor(100 + position * 11),
            "clean_target": torch.tensor((target + clean_offset) % 3),
        })
    return records


def _config(
    *, stage1_epochs: int = 1, stage2a_epochs: int = 1, revision_epochs: int = 1
):
    return {
        "method": "t_revision",
        "seed": 5,
        "data": {"name": "fixture", "width": 3, "validation_size": 3},
        "noise": {
            "name": "symmetric", "rate": 0.2, "seed": 9,
            "validation_targets": "noisy", "mapping_hash": "a" * 64,
        },
        "loader": {"batch_size": 3},
        "trainer": {"device": "cpu"},
        "evaluation": {"selection_split": "validation"},
        "t_revision": {
            "objective": "reweight",
            "fidelity": "paper_experiment_raw_additive",
            "stage1": {
                "epochs": stage1_epochs,
                "model": {"name": "fixture"},
                "optimizer": {"name": "sgd", "lr": 0.01},
                "scheduler": {"name": "none"},
                "best_metric": "noisy_validation_accuracy",
            },
            "transition_initialization": {
                "method": "pseudo_anchor_max_posterior",
                "posterior_split": "train",
                "extraction_augmentation": False,
                "tie_break": "stable_sample_index",
            },
            "classifier_initialization": {
                "epochs": stage2a_epochs,
                "optimizer": {"name": "sgd", "lr": 0.01},
                "scheduler": {"name": "none"},
                "start_from": "stage1_best",
                "best_metric": "revised_noisy_validation_accuracy",
            },
            "revision": {
                "epochs": revision_epochs,
                "transition_mode": "paper_experiment_raw_additive",
                "delta_initialization": "zeros",
                "start_from": "classifier_initialization_best",
                "optimizer": {"name": "adam", "lr": 0.01},
                "scheduler": {"name": "none"},
                "ratio": {
                    "detach": False,
                    "clamp": "none",
                    "denominator_floor": 1e-12,
                },
                "best_metric": "revised_noisy_validation_accuracy",
            },
        },
    }


def _algorithm(
    directory: str | Path,
    *,
    stage1_epochs: int = 1,
    stage2a_epochs: int = 1,
    revision_epochs: int = 1,
    clean_offset: int = 0,
    device: str = "cpu",
):
    torch_device = torch.device(device)
    model = _Classifier().to(torch_device)
    stage1_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def classifier_optimizer(value):
        return torch.optim.SGD(value.parameters(), lr=0.01)

    def revision_optimizer(value, revision):
        return torch.optim.Adam(list(value.parameters()) + [revision.delta], lr=0.01)

    generator = torch.Generator().manual_seed(5)
    train = DataLoader(_records(clean_offset=clean_offset), batch_size=3, shuffle=True, generator=generator)
    posterior = DataLoader(_records(clean_offset=clean_offset), batch_size=3, shuffle=False)
    validation = DataLoader(_records(clean_offset=2), batch_size=3, shuffle=False)
    test = DataLoader(_records(clean_offset=1), batch_size=3, shuffle=False)
    return TRevisionAlgorithm(
        model=model,
        stage1_optimizer=stage1_optimizer,
        stage1_scheduler=None,
        classifier_optimizer_factory=classifier_optimizer,
        classifier_scheduler_factory=lambda optimizer: None,
        revision_optimizer_factory=revision_optimizer,
        revision_scheduler_factory=lambda optimizer: None,
        loss=nn.CrossEntropyLoss(reduction="none"),
        train_loader=train,
        posterior_loader=posterior,
        noisy_validation_loader=validation,
        clean_test_loader=test,
        device=torch_device,
        run_dir=directory,
        config=_config(
            stage1_epochs=stage1_epochs,
            stage2a_epochs=stage2a_epochs,
            revision_epochs=revision_epochs,
        ),
        dataset="fixture",
        num_classes=3,
        noise_metadata={"manifest_sha256": "b" * 64, "mapping_hash": "a" * 64},
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TRevisionWorkflowTest(unittest.TestCase):
    def test_config_rejects_unsupported_variants_and_test_selection(self) -> None:
        values = _config()
        values["t_revision"]["objective"] = "forward"
        with self.assertRaisesRegex(NotImplementedError, "objective: reweight"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        values["t_revision"]["revision"]["transition_mode"] = "softmax"
        with self.assertRaisesRegex(ValueError, "transition_mode"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        values["evaluation"]["selection_split"] = "test"
        with self.assertRaisesRegex(ValueError, "must use validation"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        values["data"]["validation_size"] = 0
        with self.assertRaisesRegex(ValueError, "validation_size"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        values["noise"]["name"] = "instance_dependent"
        with self.assertRaisesRegex(ValueError, "class-dependent"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        del values["loader"]
        with self.assertRaisesRegex(TypeError, "loader"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        del values["trainer"]["device"]
        with self.assertRaisesRegex(ValueError, "trainer.device"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        values["t_revision"]["classifier_initialization"]["start_from"] = "last"
        with self.assertRaisesRegex(ValueError, "start_from"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        values["t_revision"]["revision"]["start_from"] = "stage1_best"
        with self.assertRaisesRegex(ValueError, "start_from"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        values["t_revision"]["revision"]["delta_initialization"] = "random"
        with self.assertRaisesRegex(ValueError, "delta_initialization"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        values["t_revision"]["revision"]["ratio"]["detach"] = True
        with self.assertRaisesRegex(ValueError, "detach"):
            TRevisionConfig.from_mapping(values)
        values = _config()
        values["t_revision"]["revision"]["ratio"]["clamp"] = "zero"
        with self.assertRaisesRegex(ValueError, "clamp"):
            TRevisionConfig.from_mapping(values)

    def test_model_preflight_rejects_class_mismatch_and_preserves_backward(self) -> None:
        loader = [{"input": torch.ones(2, 3)}]
        wrong = nn.Linear(3, 2)
        with self.assertRaisesRegex(ValueError, "class dimension"):
            _preflight_model_output(wrong, loader, torch.device("cpu"), 3)
        model = nn.Linear(3, 3)
        _preflight_model_output(model, loader, torch.device("cpu"), 3)
        model(torch.ones(2, 3)).sum().backward()
        self.assertIsNotNone(model.weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_model_preflight_does_not_create_inference_parameters(self) -> None:
        device = torch.device("cuda")
        model = nn.Linear(3, 3).to(device)
        _preflight_model_output(
            model, [{"input": torch.ones(2, 3)}], device, 3
        )
        model(torch.ones(2, 3, device=device)).sum().backward()
        self.assertIsNotNone(model.weight.grad)

    def test_full_three_stage_workflow_and_completed_resume_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            algorithm = _algorithm(run_dir)
            result = algorithm.run()
            self.assertIs(algorithm.state.phase, TRevisionPhase.COMPLETED)
            self.assertEqual(result["method"], "t_revision")
            self.assertEqual(result["phase"], "completed")
            self.assertIs(result["revised_transition_is_probability_matrix"], False)
            self.assertIn("artifact_paths", result)
            self.assertEqual(
                result["artifact_paths"]["transition_revised"],
                "transition_revised.npz",
            )
            self.assertIn("best_stage1_noisy_validation_accuracy", result)
            self.assertIn(
                "best_classifier_initialization_noisy_validation_accuracy",
                result,
            )
            for name in (
                "stage1_best.pt", "posterior_snapshot.npz", "transition_initial.npz",
                "stage2a_best.pt", "best.pt", "transition_revised.npz",
                "last.pt", "final_metrics.json",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            delta = read_checkpoint(run_dir / "best.pt", "cpu")["delta"]
            self.assertGreater(float(delta.abs().sum()), 0.0)
            protected = {
                name: (_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns)
                for name in (
                    "posterior_snapshot.npz", "transition_initial.npz",
                    "transition_revised.npz", "last.pt", "metrics.jsonl",
                )
            }
            resumed = _algorithm(run_dir)
            resumed.resume(run_dir / "last.pt")
            second = resumed.run()
            self.assertEqual(second, result)
            self.assertEqual(protected, {
                name: (_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns)
                for name in protected
            })

    def test_completed_revision_can_extend_epochs_without_rebuilding_initial_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = _algorithm(run_dir, revision_epochs=1)
            first.run()
            initial_artifacts = {
                name: (_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns)
                for name in ("posterior_snapshot.npz", "transition_initial.npz")
            }
            old_step = first.state.revision_global_step

            extended = _algorithm(run_dir, revision_epochs=2)
            extended.resume(run_dir / "last.pt")
            self.assertIs(extended.state.phase, TRevisionPhase.REVISION_TRAINING)
            extended.run()
            self.assertEqual(extended.state.revision_completed_epochs, 2)
            self.assertGreater(extended.state.revision_global_step, old_step)
            self.assertEqual(initial_artifacts, {
                name: (_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns)
                for name in initial_artifacts
            })
            self.assertEqual(len(extended.state.revised_transition_hash), 64)

    def test_resume_rejects_reduced_epochs_and_other_identity_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = _algorithm(run_dir, revision_epochs=2)
            first.train_stage1()
            first.initialize_transition()
            first.start_classifier_initialization()
            first.train_classifier_initialization()
            first.start_revision()
            first.train_revision(max_epochs=1)
            reduced = _algorithm(run_dir, revision_epochs=1)
            with self.assertRaisesRegex(ValueError, "cannot reduce"):
                reduced.resume(run_dir / "last.pt")

    def test_t_revision_import_does_not_require_sklearn(self) -> None:
        code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'sklearn' or name.startswith('sklearn.'):
        raise ModuleNotFoundError('blocked sklearn')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from lnl_toolbox.algorithms.t_revision import TRevisionAlgorithm
from lnl_toolbox.training.t_revision_experiment import run_t_revision_experiment
print(TRevisionAlgorithm.__name__, run_t_revision_experiment.__name__)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = "src"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TRevisionAlgorithm", result.stdout)

    def test_stage1_interruption_resume_reaches_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _algorithm(directory, stage1_epochs=2)
            first.train_stage1(max_epochs=1)
            self.assertIs(first.state.phase, TRevisionPhase.STAGE1_TRAINING)
            step = first.state.stage1_global_step
            second = _algorithm(directory, stage1_epochs=2)
            second.resume(Path(directory) / "last.pt")
            second.run()
            self.assertIs(second.state.phase, TRevisionPhase.COMPLETED)
            self.assertGreater(second.state.stage1_global_step, step)

    def test_transition_and_later_stages_resume_without_reestimating_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = _algorithm(run_dir)
            first.train_stage1()
            first.initialize_transition()
            initial_hashes = {
                path.name: (_sha(path), path.stat().st_mtime_ns)
                for path in (first.snapshot_path, first.initial_transition_path)
            }
            second = _algorithm(run_dir)
            second.resume(run_dir / "last.pt")
            second.run()
            self.assertEqual(initial_hashes, {
                name: (_sha(run_dir / name), (run_dir / name).stat().st_mtime_ns)
                for name in initial_hashes
            })

    def test_snapshot_and_anchor_transition_come_from_stage1_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _algorithm(directory, stage1_epochs=2)
            algorithm.train_stage1()
            self.assertEqual(algorithm.state.stage1_best_epoch, 0)
            best = read_checkpoint(algorithm.stage1_best_path, "cpu")
            best_weight = best["model"]["weight"].clone()
            with torch.no_grad():
                algorithm.model.weight.add_(100.0 * torch.randn_like(algorithm.model.weight))
            algorithm.initialize_transition()
            torch.testing.assert_close(algorithm.model.weight.cpu(), best_weight)
            self.assertEqual(
                algorithm.initial_transition.metadata["stage1_best_checkpoint_sha256"],
                algorithm.state.stage1_best_hash,
            )
            self.assertEqual(
                algorithm.initial_transition.source_snapshot_hash,
                algorithm.snapshot.snapshot_hash,
            )

    def test_transition_initialization_rejects_snapshot_missing_target_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _algorithm(directory)
            algorithm.train_stage1()
            records = [record for record in _records() if int(record["target"]) < 2]
            algorithm.posterior_loader = DataLoader(
                records, batch_size=2, shuffle=False
            )
            with self.assertRaisesRegex(ValueError, "missing noisy target classes"):
                algorithm.initialize_transition()

    def test_classifier_initialization_and_revision_resume_match_uninterrupted(self) -> None:
        with tempfile.TemporaryDirectory() as uninterrupted_dir, tempfile.TemporaryDirectory() as resumed_dir:
            torch.manual_seed(31)
            uninterrupted = _algorithm(
                uninterrupted_dir, stage2a_epochs=2, revision_epochs=2
            )
            uninterrupted.run()
            uninterrupted_best = read_checkpoint(
                Path(uninterrupted_dir) / "best.pt", "cpu"
            )

            torch.manual_seed(31)
            first = _algorithm(resumed_dir, stage2a_epochs=2, revision_epochs=2)
            first.train_stage1()
            first.initialize_transition()
            first.start_classifier_initialization()
            first.train_classifier_initialization(max_epochs=1)
            self.assertIs(first.state.phase, TRevisionPhase.CLASSIFIER_INITIALIZATION)
            second = _algorithm(resumed_dir, stage2a_epochs=2, revision_epochs=2)
            second.resume(Path(resumed_dir) / "last.pt")
            second.train_classifier_initialization()
            second.start_revision()
            second.train_revision(max_epochs=1)
            self.assertIs(second.state.phase, TRevisionPhase.REVISION_TRAINING)
            third = _algorithm(resumed_dir, stage2a_epochs=2, revision_epochs=2)
            third.resume(Path(resumed_dir) / "last.pt")
            third.run()
            resumed_best = read_checkpoint(Path(resumed_dir) / "best.pt", "cpu")
            for name, value in uninterrupted_best["model"].items():
                torch.testing.assert_close(value, resumed_best["model"][name])
            torch.testing.assert_close(
                uninterrupted_best["delta"], resumed_best["delta"]
            )

    def test_stage2a_fixes_transition_and_revision_rebuilds_adam(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _algorithm(directory)
            algorithm.train_stage1()
            algorithm.initialize_transition()
            transition_before = algorithm.initial_transition.matrix.copy()
            algorithm.start_classifier_initialization()
            self.assertIsInstance(algorithm.optimizer, torch.optim.SGD)
            algorithm.train_classifier_initialization()
            torch.testing.assert_close(
                torch.as_tensor(algorithm.initial_transition.matrix.copy()),
                torch.as_tensor(transition_before),
            )
            algorithm.start_revision()
            self.assertIsInstance(algorithm.optimizer, torch.optim.Adam)
            self.assertEqual(len(algorithm.optimizer.state), 0)
            self.assertTrue(torch.equal(algorithm.revision.delta, torch.zeros_like(algorithm.revision.delta)))

    def test_clean_target_is_not_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            torch.manual_seed(17)
            first = _algorithm(first_dir, clean_offset=0)
            first.train_stage1()
            first.initialize_transition()
            torch.manual_seed(17)
            second = _algorithm(second_dir, clean_offset=2)
            second.train_stage1()
            second.initialize_transition()
            self.assertEqual(first.state.snapshot_hash, second.state.snapshot_hash)
            self.assertEqual(first.state.initial_transition_hash, second.state.initial_transition_hash)

    def test_artifact_and_manifest_drift_fail_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            algorithm = _algorithm(run_dir)
            algorithm.train_stage1()
            algorithm.initialize_transition()
            snapshot = run_dir / "posterior_snapshot.npz"
            snapshot.write_bytes(b"broken")
            resumed = _algorithm(run_dir)
            with self.assertRaises(Exception):
                resumed.resume(run_dir / "last.pt")
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _algorithm(directory)
            algorithm.train_stage1(max_epochs=1)
            changed = _algorithm(directory)
            changed.config["noise"] = dict(changed.config["noise"], mapping_hash="c" * 64)
            with self.assertRaisesRegex(ValueError, "changed noise"):
                changed.resume(Path(directory) / "last.pt")

    def test_revision_best_corruption_fails_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            algorithm = _algorithm(run_dir)
            algorithm.run()
            (run_dir / "best.pt").write_bytes(b"broken")
            resumed = _algorithm(run_dir)
            with self.assertRaisesRegex(ValueError, "revision best.*hash mismatch"):
                resumed.resume(run_dir / "last.pt")

    def test_dispatch_is_lazy_and_available(self) -> None:
        with unittest.mock.patch(
            "lnl_toolbox.training.t_revision_experiment.run_t_revision_experiment",
            return_value=Path("fixture"),
        ) as runner:
            result = run_experiment(_config())
        self.assertEqual(result, Path("fixture"))
        runner.assert_called_once()

    def test_unknown_method_fails_instead_of_supervised_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported training method"):
            run_experiment({"method": "not_a_method"})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_three_stage_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _algorithm(directory, device="cuda")
            algorithm.run()
            self.assertIs(algorithm.state.phase, TRevisionPhase.COMPLETED)


if __name__ == "__main__":
    unittest.main()
