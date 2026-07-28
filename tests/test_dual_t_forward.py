from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lnl_toolbox.algorithms.dual_t import (
    DualTConfig,
    DualTAlgorithm,
    DualTPhase,
    DualTState,
)
from lnl_toolbox.algorithms.dual_t.algorithm import (
    _train_supervised_epoch,
)
from lnl_toolbox.core import RunState
from lnl_toolbox.noise import DualTransitionEstimator, PosteriorSnapshot
from lnl_toolbox.noise.transition import TransitionArtifact
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.experiment import run_experiment


class _IndexedClassifier(nn.Module):
    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(classes) * 5.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight.t()


def _records(*, clean_offset: int = 0):
    noisy_targets = (0, 1, 1, 2, 2, 0)
    records = []
    for index, noisy_target in enumerate(noisy_targets):
        latent_class = index // 2
        records.append({
            "input": torch.nn.functional.one_hot(
                torch.tensor(latent_class), num_classes=3
            ).to(dtype=torch.float32),
            "target": torch.tensor(noisy_target),
            "index": torch.tensor(100 + index * 7),
            "clean_target": torch.tensor(
                (latent_class + clean_offset) % 3
            ),
        })
    return records


def _validation_records():
    return [
        {
            "input": torch.nn.functional.one_hot(
                torch.tensor(index), num_classes=3
            ).to(dtype=torch.float32),
            "target": torch.tensor(index),
            "index": torch.tensor(500 + index),
        }
        for index in range(3)
    ]


def _test_records():
    return [
        {
            "input": torch.nn.functional.one_hot(
                torch.tensor(index), num_classes=3
            ).to(dtype=torch.float32),
            "target": torch.tensor(index),
            "index": torch.tensor(800 + index),
        }
        for index in range(3)
    ]


def _config(*, posterior_epochs: int = 2, final_epochs: int = 2):
    stage = {
        "model": {"name": "fixture"},
        "optimizer": {"name": "sgd", "lr": 0.0},
        "scheduler": {"name": "none"},
        "loss": {"name": "ce"},
    }
    return {
        "method": "dual_t",
        "seed": 3,
        "data": {"name": "fixture"},
        "noise": {
            "name": "symmetric",
            "rate": 0.2,
            "seed": 5,
            "validation_targets": "noisy",
            "mapping_hash": "a" * 64,
        },
        "loader": {"batch_size": 3},
        "evaluation": {"selection_split": "validation"},
        "posterior_stage": {
            **stage,
            "epochs": posterior_epochs,
            "checkpoint_selection": "noisy_validation_accuracy",
        },
        "transition_stage": {},
        "final_stage": {
            **stage,
            "epochs": final_epochs,
            "fresh_model": True,
        },
    }


def _algorithm(
    directory: str | Path,
    *,
    posterior_epochs: int = 2,
    final_epochs: int = 2,
    clean_offset: int = 0,
) -> DualTAlgorithm:
    posterior_model = _IndexedClassifier()
    final_model = _IndexedClassifier()
    posterior_optimizer = torch.optim.SGD(
        posterior_model.parameters(), lr=0.0
    )
    final_optimizer = torch.optim.SGD(final_model.parameters(), lr=0.0)
    loss = nn.CrossEntropyLoss(reduction="none")
    return DualTAlgorithm(
        posterior_model=posterior_model,
        posterior_optimizer=posterior_optimizer,
        posterior_scheduler=None,
        final_model=final_model,
        final_optimizer=final_optimizer,
        final_scheduler=None,
        posterior_loss=loss,
        final_loss=nn.CrossEntropyLoss(reduction="none"),
        train_loader=DataLoader(
            _records(clean_offset=clean_offset), batch_size=3, shuffle=True
        ),
        noisy_validation_loader=DataLoader(
            _validation_records(), batch_size=3
        ),
        clean_test_loader=DataLoader(_test_records(), batch_size=3),
        device=torch.device("cpu"),
        run_dir=directory,
        config=_config(
            posterior_epochs=posterior_epochs, final_epochs=final_epochs
        ),
        dataset="fixture",
        noise_metadata={
            "manifest_sha256": "b" * 64,
            "mapping_hash": "a" * 64,
        },
    )


class DualTMathTest(unittest.TestCase):
    @staticmethod
    def _snapshot() -> PosteriorSnapshot:
        return PosteriorSnapshot(
            noisy_probabilities=np.asarray([
                [0.90, 0.05, 0.05],
                [0.60, 0.20, 0.20],
                [0.10, 0.80, 0.10],
                [0.20, 0.60, 0.20],
                [0.05, 0.05, 0.90],
                [0.20, 0.20, 0.60],
            ]),
            noisy_targets=np.asarray([0, 1, 1, 2, 2, 0]),
            global_indices=np.asarray([30, 20, 10, 40, 50, 60]),
            dataset="fixture",
            split="train",
        )

    def test_t_club_times_t_spade_and_not_reverse(self) -> None:
        artifact = DualTransitionEstimator().estimate(self._snapshot())
        t_club = np.asarray(artifact.metadata["t_club"])
        t_spade = np.asarray(artifact.metadata["t_spade"])
        np.testing.assert_allclose(artifact.matrix, t_club @ t_spade)
        self.assertFalse(np.allclose(t_club @ t_spade, t_spade @ t_club))

    def test_stable_index_permutation_does_not_change_estimate(self) -> None:
        original = self._snapshot()
        order = np.asarray([4, 0, 5, 2, 1, 3])
        permuted = PosteriorSnapshot(
            original.noisy_probabilities[order],
            original.noisy_targets[order],
            original.global_indices[order],
            original.dataset,
            original.split,
        )
        first = DualTransitionEstimator().estimate(original)
        second = DualTransitionEstimator().estimate(permuted)
        np.testing.assert_allclose(first.matrix, second.matrix)
        self.assertEqual(
            first.metadata["anchor_global_indices"],
            second.metadata["anchor_global_indices"],
        )

    def test_empty_intermediate_class_fails(self) -> None:
        snapshot = PosteriorSnapshot(
            np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
            np.asarray([0, 1]),
            np.asarray([3, 9]),
            "fixture",
            "train",
        )
        with self.assertRaisesRegex(ValueError, "empty intermediate classes"):
            DualTransitionEstimator().estimate(snapshot)


class DualTStateTest(unittest.TestCase):
    def test_only_sequential_phase_transitions_are_allowed(self) -> None:
        state = DualTState()
        with self.assertRaisesRegex(ValueError, "illegal Dual-T phase"):
            state.advance(DualTPhase.TRANSITION_READY)
        state.best_posterior_epoch = 0
        state.posterior_completed_epochs = 1
        state.best_posterior_checkpoint_sha256 = "a" * 64
        state.advance(DualTPhase.POSTERIOR_READY)
        state.posterior_snapshot_hash = "b" * 64
        state.transition_artifact_hash = "c" * 64
        state.advance(DualTPhase.TRANSITION_READY)
        state.advance(DualTPhase.FINAL_TRAINING)
        state.best_final_epoch = 0
        state.final_completed_epochs = 1
        state.advance(DualTPhase.COMPLETED)
        restored = DualTState.from_state_dict(state.state_dict())
        self.assertIs(restored.phase, DualTPhase.COMPLETED)

    def test_config_rejects_test_selection_and_clean_validation(self) -> None:
        test_selection = _config()
        test_selection["evaluation"] = {"selection_split": "test"}
        with self.assertRaisesRegex(ValueError, "validation, not test"):
            DualTConfig.from_mapping(test_selection)
        clean_validation = _config()
        clean_validation["noise"]["validation_targets"] = "clean"
        with self.assertRaisesRegex(ValueError, "validation_targets: noisy"):
            DualTConfig.from_mapping(clean_validation)

    def test_config_defaults_to_forward_and_accepts_explicit_forward(self) -> None:
        implicit = DualTConfig.from_mapping(_config())
        self.assertEqual(implicit.classifier_backend, "forward")
        explicit_values = _config()
        explicit_values["final_stage"]["classifier"] = "forward"
        explicit = DualTConfig.from_mapping(explicit_values)
        self.assertEqual(explicit.classifier_backend, "forward")

    def test_config_rejects_other_classifier_backends(self) -> None:
        values = _config()
        values["final_stage"]["classifier"] = "revision"
        with self.assertRaisesRegex(
            NotImplementedError,
            "only supports classifier backend 'forward'",
        ):
            DualTConfig.from_mapping(values)

    def test_old_method_name_has_an_explicit_rename_error(self) -> None:
        values = _config()
        values["method"] = "dual_t_forward"
        with self.assertRaisesRegex(
            ValueError, "dual_t_forward.*renamed to 'dual_t'"
        ):
            DualTConfig.from_mapping(values)
        with self.assertRaisesRegex(
            ValueError, "dual_t_forward.*renamed to 'dual_t'"
        ):
            run_experiment(values)

    def test_transition_stage_does_not_expose_internal_composition(self) -> None:
        values = _config()
        values["transition_stage"] = {
            "estimator": "dual_t",
            "correction": "forward",
        }
        with self.assertRaisesRegex(ValueError, "internal method fields"):
            DualTConfig.from_mapping(values)

    def test_experiment_module_has_one_run_experiment_entry(self) -> None:
        source_path = Path(inspect.getsourcefile(run_experiment))
        definitions = [
            line
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("def run_experiment")
        ]
        self.assertEqual(definitions, ["def run_experiment("])


class DualTAlgorithmLifecycleTest(unittest.TestCase):
    def test_extracted_epoch_helper_preserves_production_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            torch.manual_seed(31)
            first = _algorithm(first_dir)
            torch.manual_seed(31)
            second = _algorithm(second_dir)
            first_state = RunState(phase="posterior_train")
            second_state = RunState(phase="posterior_train")
            production = first._train_epoch(
                first.posterior_algorithm,
                first_state,
                0,
            )
            extracted = _train_supervised_epoch(
                second.posterior_algorithm,
                second.train_loader,
                second_state,
                0,
            )
            self.assertEqual(production, extracted)
            self.assertEqual(first_state.step, second_state.step)

    def test_tiny_cpu_end_to_end_uses_fresh_models_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _algorithm(directory, posterior_epochs=1, final_epochs=1)
            self.assertIsNot(
                algorithm.posterior_algorithm.model,
                algorithm.final_algorithm.model,
            )
            self.assertIsNot(
                algorithm.posterior_algorithm.optimizer,
                algorithm.final_algorithm.optimizer,
            )
            final = algorithm.run()
            self.assertIs(algorithm.state.phase, DualTPhase.COMPLETED)
            self.assertEqual(final["method"], "dual_t")
            self.assertEqual(final["completed_posterior_epochs"], 1)
            self.assertEqual(final["completed_final_epochs"], 1)
            for filename in (
                "posterior_best.pt",
                "posterior_snapshot.npz",
                "transition_artifact.npz",
                "last.pt",
                "best.pt",
                "metrics.jsonl",
                "final_metrics.json",
            ):
                self.assertTrue(Path(directory, filename).is_file(), filename)
            self.assertEqual(
                algorithm.transition.metadata["composition"],
                "t_club @ t_spade",
            )
            self.assertEqual(
                algorithm.transition.metadata[
                    "posterior_best_checkpoint_sha256"
                ],
                algorithm.state.best_posterior_checkpoint_sha256,
            )
            self.assertEqual(algorithm.transition.metadata["method"], "dual_t")

            metric_rows = [
                json.loads(line)
                for line in Path(directory, "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            final_epoch = next(
                row
                for row in metric_rows
                if row.get("stage") == "final" and row.get("event") == "epoch"
            )
            self.assertIn("validation_observed_ce_loss", final_epoch)
            self.assertNotIn("validation_loss", final_epoch)

            actual_best_sha = hashlib.sha256(
                Path(directory, "posterior_best.pt").read_bytes()
            ).hexdigest()
            best_payload = read_checkpoint(
                Path(directory, "posterior_best.pt"), "cpu"
            )
            last_payload = read_checkpoint(Path(directory, "last.pt"), "cpu")
            self.assertNotEqual(
                best_payload["dual_t_state"][
                    "best_posterior_checkpoint_sha256"
                ],
                actual_best_sha,
            )
            self.assertEqual(
                last_payload["dual_t_state"][
                    "best_posterior_checkpoint_sha256"
                ],
                actual_best_sha,
            )

            snapshot_mtime = Path(
                directory, "posterior_snapshot.npz"
            ).stat().st_mtime_ns
            transition_mtime = Path(
                directory, "transition_artifact.npz"
            ).stat().st_mtime_ns
            restored = _algorithm(
                directory, posterior_epochs=1, final_epochs=1
            )
            restored.resume(Path(directory, "last.pt"))
            restored.run()
            self.assertEqual(
                Path(directory, "posterior_snapshot.npz").stat().st_mtime_ns,
                snapshot_mtime,
            )
            self.assertEqual(
                Path(directory, "transition_artifact.npz").stat().st_mtime_ns,
                transition_mtime,
            )

    def test_transition_snapshot_is_loaded_from_best_not_current_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _algorithm(
                directory, posterior_epochs=1, final_epochs=1
            )
            algorithm.train_posterior()
            with torch.no_grad():
                algorithm.posterior_algorithm.model.weight.zero_()
                algorithm.posterior_algorithm.model.weight[2, 0] = 10.0
            algorithm.estimate_transition()
            expected = torch.softmax(torch.eye(3) * 5.0, dim=1).numpy()
            first_by_class = algorithm.snapshot.noisy_probabilities[[0, 2, 4]]
            np.testing.assert_allclose(first_by_class, expected, atol=1e-7)

    def test_clean_target_is_not_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            torch.manual_seed(11)
            first = _algorithm(
                first_dir, posterior_epochs=1, final_epochs=1, clean_offset=0
            )
            first.train_posterior()
            first.estimate_transition()
            torch.manual_seed(11)
            second = _algorithm(
                second_dir, posterior_epochs=1, final_epochs=1, clean_offset=2
            )
            second.train_posterior()
            second.estimate_transition()
            self.assertEqual(
                first.snapshot.snapshot_hash, second.snapshot.snapshot_hash
            )
            np.testing.assert_allclose(
                first.transition.matrix, second.transition.matrix
            )

    def test_posterior_training_interruption_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _algorithm(directory)
            first.train_posterior(max_epochs=1)
            self.assertIs(first.state.phase, DualTPhase.POSTERIOR_TRAINING)
            self.assertEqual(first.state.posterior_completed_epochs, 1)
            restored = _algorithm(directory)
            restored.resume(Path(directory) / "last.pt")
            final = restored.run()
            self.assertEqual(final["completed_posterior_epochs"], 2)
            self.assertEqual(final["posterior_global_step"], 4)

    def test_transition_ready_resume_requires_and_loads_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _algorithm(directory)
            first.train_posterior()
            first.estimate_transition()
            self.assertIs(first.state.phase, DualTPhase.TRANSITION_READY)
            restored = _algorithm(directory)
            restored.resume(Path(directory) / "last.pt")
            self.assertIsNotNone(restored.transition)
            self.assertEqual(
                restored.transition.artifact_hash,
                first.transition.artifact_hash,
            )

    def test_transition_ready_resume_rejects_missing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _algorithm(directory)
            first.train_posterior()
            first.estimate_transition()
            Path(directory, "posterior_snapshot.npz").unlink()
            restored = _algorithm(directory)
            with self.assertRaisesRegex(FileNotFoundError, "artifact missing"):
                restored.resume(Path(directory) / "last.pt")

    def test_transition_save_failure_does_not_advance_phase_or_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _algorithm(directory)
            algorithm.train_posterior()
            self.assertIs(algorithm.state.phase, DualTPhase.POSTERIOR_READY)
            last_before = Path(directory, "last.pt").read_bytes()
            with patch.object(
                TransitionArtifact,
                "save",
                side_effect=OSError("simulated artifact save failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    algorithm.estimate_transition()
            self.assertIs(algorithm.state.phase, DualTPhase.POSTERIOR_READY)
            self.assertEqual(algorithm.state.posterior_snapshot_hash, "")
            self.assertEqual(algorithm.state.transition_artifact_hash, "")
            self.assertEqual(Path(directory, "last.pt").read_bytes(), last_before)

    def test_transition_ready_resume_rejects_artifact_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _algorithm(directory)
            first.train_posterior()
            first.estimate_transition()
            transition_path = Path(directory, "transition_artifact.npz")
            with np.load(transition_path, allow_pickle=False) as data:
                matrix = data["matrix"].copy()
                metadata_json = data["metadata_json"].copy()
            matrix[0] = np.roll(matrix[0], 1)
            np.savez_compressed(
                transition_path,
                matrix=matrix,
                metadata_json=metadata_json,
            )
            restored = _algorithm(directory)
            with self.assertRaisesRegex(ValueError, "hash"):
                restored.resume(Path(directory) / "last.pt")

    def test_resume_rejects_posterior_best_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _algorithm(directory)
            first.train_posterior(max_epochs=1)
            with Path(directory, "posterior_best.pt").open("ab") as stream:
                stream.write(b"tamper")
            restored = _algorithm(directory)
            with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                restored.resume(Path(directory) / "last.pt")

    def test_final_training_interruption_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _algorithm(directory)
            first.train_posterior()
            first.estimate_transition()
            first.start_final_training()
            first.train_final(max_epochs=1)
            self.assertIs(first.state.phase, DualTPhase.FINAL_TRAINING)
            self.assertEqual(first.state.final_completed_epochs, 1)
            restored = _algorithm(directory)
            restored.resume(Path(directory) / "last.pt")
            final = restored.run()
            self.assertEqual(final["completed_final_epochs"], 2)
            self.assertEqual(final["final_global_step"], 4)

    def test_posterior_best_cannot_be_used_as_run_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _algorithm(directory)
            first.train_posterior(max_epochs=1)
            restored = _algorithm(directory)
            with self.assertRaisesRegex(ValueError, "run-state checkpoint"):
                restored.resume(Path(directory) / "posterior_best.pt")


if __name__ == "__main__":
    unittest.main()
