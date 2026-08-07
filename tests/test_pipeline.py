from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.algorithms.transition_risk import ForwardRiskCorrector
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.noise import KnownTransition
from lnl_toolbox.plugins.builtin import build_builtin_pipeline
from lnl_toolbox.treatments import SupervisedWeightInput, WeightResult
from lnl_toolbox.training.pipeline import (
    PipelinePhase,
    StandardNoisyERMPipeline,
)


class _WeightProvider:
    def __init__(self) -> None:
        self.inputs: list[SupervisedWeightInput] = []

    def compute(self, weight_input: SupervisedWeightInput) -> WeightResult:
        self.inputs.append(weight_input)
        return WeightResult(
            sample_weights=torch.full(
                (weight_input.noisy_targets.numel(),),
                0.5,
                device=weight_input.noisy_targets.device,
            ),
            metrics={"provider_seen": 1.0},
        )


class _IndexedModel(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.eye(classes) * 5.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits[inputs.view(-1).long()]


class _StatefulEstimator:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = int(state["value"])


class PipelineIntegrationTest(unittest.TestCase):
    @staticmethod
    def _transition_fixture():
        classes = 4
        model = _IndexedModel(classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.arange(classes).view(-1, 1)
        loader = DataLoader(
            [
                {
                    "input": inputs[index],
                    "target": torch.tensor(index),
                    "index": torch.tensor(index),
                }
                for index in range(classes)
            ],
            batch_size=2,
            shuffle=False,
        )
        pipeline = build_builtin_pipeline({
            "name": "standard_noisy_erm",
            "warmup_epochs": 0,
            "transition_estimator": {"name": "dual_t"},
            "risk_corrector": {"name": "forward"},
        })
        return pipeline, model, optimizer, loader

    def test_standard_weight_provider_without_posterior_still_works(self) -> None:
        model = nn.Linear(2, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        provider = _WeightProvider()
        algorithm = SupervisedClassificationAlgorithm(
            model,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            risk_corrector=ForwardRiskCorrector(),
            transition=KnownTransition(np.eye(2)),
            weight_provider=provider,
        )
        algorithm.setup(ExperimentContext(Path(".")))
        result = algorithm.step(
            Batch({
                "input": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "target": torch.tensor([0, 1]),
                "index": torch.tensor([4, 9]),
            }),
            RunState(phase="train"),
        )
        self.assertEqual(len(provider.inputs), 1)
        self.assertTrue(torch.equal(provider.inputs[0].sample_indices, torch.tensor([4, 9])))
        self.assertFalse(hasattr(provider.inputs[0], "posterior_probabilities"))
        self.assertFalse(provider.inputs[0].logits.requires_grad)
        self.assertFalse(provider.inputs[0].per_sample_loss.requires_grad)
        self.assertEqual(result.metrics["selected_samples"], 2.0)
        self.assertEqual(result.metrics["treatment_provider_seen"], 1.0)

    def test_dual_t_pipeline_persists_snapshot_and_transition(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        self.assertIsInstance(pipeline, StandardNoisyERMPipeline)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = pipeline.prepare_transition(
                model=model,
                optimizer=optimizer,
                loader=loader,
                device=torch.device("cpu"),
                dataset="synthetic",
                split="train",
                run_dir=directory,
            )
            self.assertIsNotNone(artifacts.snapshot)
            self.assertIsNotNone(artifacts.transition)
            self.assertTrue(Path(directory, "posterior_snapshot.npz").is_file())
            self.assertTrue(Path(directory, "transition_artifact.npz").is_file())
            restored = build_builtin_pipeline({
                "name": "standard_noisy_erm",
                "transition_estimator": {"name": "dual_t"},
            })
            self.assertTrue(restored.load_artifacts(directory))
            self.assertEqual(
                restored.artifacts.snapshot.snapshot_hash,
                artifacts.snapshot.snapshot_hash,
            )

    def test_fresh_training_can_prepare_transition(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            artifacts = pipeline.prepare_transition(
                model=model,
                optimizer=optimizer,
                loader=loader,
                device=torch.device("cpu"),
                dataset="synthetic",
                split="train",
                run_dir=directory,
            )
            self.assertIsNotNone(artifacts.snapshot)
            self.assertIsNotNone(artifacts.transition)

    def test_resume_missing_transition_artifact_fails(self) -> None:
        pipeline, _, _, _ = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "artifact missing"):
                pipeline.restore_for_resume(
                    directory,
                    checkpoint_state=pipeline.state_dict(),
                    component_states={},
                    dataset="synthetic",
                    split="train",
                )

    def test_resume_transition_hash_mismatch_fails(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.prepare_transition(
                model=model,
                optimizer=optimizer,
                loader=loader,
                device=torch.device("cpu"),
                dataset="synthetic",
                split="train",
                run_dir=directory,
            )
            checkpoint_state = pipeline.state_dict()
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
            restored, _, _, _ = self._transition_fixture()
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                restored.restore_for_resume(
                    directory,
                    checkpoint_state=checkpoint_state,
                    component_states={},
                    dataset="synthetic",
                    split="train",
                )

    def test_resume_artifact_provenance_mismatch_fails(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.prepare_transition(
                model=model,
                optimizer=optimizer,
                loader=loader,
                device=torch.device("cpu"),
                dataset="synthetic",
                split="train",
                run_dir=directory,
            )
            restored, _, _, _ = self._transition_fixture()
            with self.assertRaisesRegex(
                ValueError, "artifact provenance mismatch"
            ):
                restored.restore_for_resume(
                    directory,
                    checkpoint_state=pipeline.state_dict(),
                    component_states={},
                    dataset="different",
                    split="train",
                )

    def test_checkpoint_artifact_identity_verified(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.prepare_transition(
                model=model,
                optimizer=optimizer,
                loader=loader,
                device=torch.device("cpu"),
                dataset="synthetic",
                split="train",
                run_dir=directory,
            )
            checkpoint_state = deepcopy(pipeline.state_dict())
            checkpoint_state["artifacts"]["transition_artifact_hash"] = "0" * 64
            restored, _, _, _ = self._transition_fixture()
            with self.assertRaisesRegex(
                ValueError, "checkpoint identity mismatch"
            ):
                restored.restore_for_resume(
                    directory,
                    checkpoint_state=checkpoint_state,
                    component_states={},
                    dataset="synthetic",
                    split="train",
                )

    def test_pipeline_state_roundtrip(self) -> None:
        pipeline = StandardNoisyERMPipeline()
        pipeline.state.phase = PipelinePhase.EVALUATE
        pipeline.state.cycle = 7
        pipeline.state.metadata["marker"] = "restored"
        restored = StandardNoisyERMPipeline()
        restored.load_state_dict(pipeline.state_dict())
        self.assertEqual(restored.state.phase, PipelinePhase.EVALUATE)
        self.assertEqual(restored.state.cycle, 7)
        self.assertEqual(restored.state.metadata["marker"], "restored")

    def test_pipeline_state_restored_before_training(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.prepare_transition(
                model=model,
                optimizer=optimizer,
                loader=loader,
                device=torch.device("cpu"),
                dataset="synthetic",
                split="train",
                run_dir=directory,
            )
            pipeline.state.phase = PipelinePhase.EVALUATE
            pipeline.state.cycle = 7
            pipeline.state.metadata["marker"] = "restored"
            restored, _, _, _ = self._transition_fixture()
            restored.restore_for_resume(
                directory,
                checkpoint_state=pipeline.state_dict(),
                component_states={},
                dataset="synthetic",
                split="train",
            )
            self.assertEqual(restored.state.phase, PipelinePhase.EVALUATE)
            self.assertEqual(restored.state.cycle, 7)
            self.assertEqual(restored.state.metadata["marker"], "restored")

    def test_component_states_roundtrip(self) -> None:
        source = StandardNoisyERMPipeline(
            transition_estimator=_StatefulEstimator(11)
        )
        states = source.component_state_dict()
        restored = StandardNoisyERMPipeline(
            transition_estimator=_StatefulEstimator(0)
        )
        restored.load_component_states(states)
        self.assertEqual(restored.transition_estimator.value, 11)

    def test_legacy_checkpoint_without_pipeline_state(self) -> None:
        pipeline, model, optimizer, loader = self._transition_fixture()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.prepare_transition(
                model=model,
                optimizer=optimizer,
                loader=loader,
                device=torch.device("cpu"),
                dataset="synthetic",
                split="train",
                run_dir=directory,
            )
            restored, _, _, _ = self._transition_fixture()
            warnings = restored.restore_for_resume(
                directory,
                checkpoint_state=None,
                component_states=None,
                dataset="synthetic",
                split="train",
            )
            self.assertTrue(warnings)
            self.assertEqual(
                restored.artifacts.transition.artifact_hash,
                pipeline.artifacts.transition.artifact_hash,
            )

    def test_binary_rcn_rejects_current_model_softmax_fallback(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "explicit noisy-label posterior producer"
        ):
            build_builtin_pipeline({
                "name": "standard_noisy_erm",
                "weight_provider": {
                    "name": "binary_rcn_importance",
                    "rho_positive": 0.1,
                    "rho_negative": 0.2,
                },
            })


if __name__ == "__main__":
    unittest.main()
