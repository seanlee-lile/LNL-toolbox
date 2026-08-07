from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from lnl_toolbox.algorithms.t_revision import (
    AdditiveTransitionRevision,
    RevisedTransitionArtifact,
    TRevisionPhase,
    TRevisionState,
    validate_revision_optimizer,
)
from lnl_toolbox.noise.transition import TransitionArtifact


class TRevisionTransitionTest(unittest.TestCase):
    def test_delta_starts_zero_and_raw_addition_is_unprojected(self) -> None:
        initial = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        revision = AdditiveTransitionRevision(initial)
        torch.testing.assert_close(revision(), initial)
        with torch.no_grad():
            revision.delta.copy_(torch.tensor([[0.4, -0.5], [0.0, 0.2]]))
        expected = torch.tensor([[1.2, -0.3], [0.1, 1.1]])
        torch.testing.assert_close(revision(), expected)
        diagnostics = revision.diagnostics()
        self.assertIs(diagnostics["non_negative"], False)
        self.assertIs(diagnostics["row_stochastic"], False)

    def test_optimizer_must_contain_model_and_delta_exactly_once(self) -> None:
        model = nn.Linear(2, 2)
        revision = AdditiveTransitionRevision(torch.eye(2))
        optimizer = torch.optim.Adam(
            list(model.parameters()) + [revision.delta], lr=1e-3
        )
        validate_revision_optimizer(optimizer, model.parameters(), revision)
        missing = torch.optim.Adam(model.parameters(), lr=1e-3)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_revision_optimizer(missing, model.parameters(), revision)
        duplicate = torch.optim.Adam(
            list(model.parameters()) + [revision.delta], lr=1e-3
        )
        duplicate.param_groups[0]["params"].append(revision.delta)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_revision_optimizer(duplicate, model.parameters(), revision)

    def test_one_step_updates_model_and_delta(self) -> None:
        model = nn.Linear(2, 2)
        revision = AdditiveTransitionRevision(torch.tensor([[0.8, 0.2], [0.2, 0.8]]))
        optimizer = torch.optim.Adam(list(model.parameters()) + [revision.delta], lr=1e-2)
        model_before = model.weight.detach().clone()
        delta_before = revision.delta.detach().clone()
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        logits = model(inputs)
        from lnl_toolbox.algorithms.t_revision import t_revision_reweight_objective
        objective = t_revision_reweight_objective(
            logits, torch.tensor([0, 1]), revision(), denominator_floor=1e-12
        ).objective
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        self.assertFalse(torch.equal(model_before, model.weight.detach()))
        self.assertFalse(torch.equal(delta_before, revision.delta.detach()))

    def test_revised_artifact_is_not_a_transition_artifact_and_roundtrips(self) -> None:
        artifact = RevisedTransitionArtifact(
            initial_transition=np.asarray([[0.8, 0.2], [0.1, 0.9]]),
            delta=np.asarray([[0.3, -0.4], [0.0, 0.1]]),
            source_initial_artifact_hash="a" * 64,
            stage2a_best_checkpoint_sha256="b" * 64,
            best_noisy_validation_accuracy=0.75,
        )
        self.assertNotIsInstance(artifact, TransitionArtifact)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "revised.npz"
            artifact.save(path)
            restored = RevisedTransitionArtifact.load(path)
        self.assertEqual(restored.artifact_hash, artifact.artifact_hash)
        np.testing.assert_allclose(restored.revised_transition, artifact.revised_transition)


class TRevisionStateTest(unittest.TestCase):
    def test_only_forward_phase_transitions_are_allowed(self) -> None:
        state = TRevisionState()
        with self.assertRaisesRegex(ValueError, "illegal T-Revision"):
            state.advance(TRevisionPhase.TRANSITION_INITIALIZED)
        state.stage1_completed_epochs = 1
        state.stage1_best_epoch = 0
        state.stage1_best_hash = "a" * 64
        state.advance(TRevisionPhase.STAGE1_READY)
        state.snapshot_hash = "b" * 64
        state.initial_transition_hash = "c" * 64
        state.advance(TRevisionPhase.TRANSITION_INITIALIZED)
        state.advance(TRevisionPhase.CLASSIFIER_INITIALIZATION)
        state.stage2a_completed_epochs = 1
        state.stage2a_best_epoch = 0
        state.stage2a_best_hash = "d" * 64
        state.advance(TRevisionPhase.CLASSIFIER_READY)
        state.advance(TRevisionPhase.REVISION_TRAINING)
        state.revision_completed_epochs = 1
        state.revision_best_epoch = 0
        state.revision_best_checkpoint_hash = "f" * 64
        state.revised_transition_hash = "e" * 64
        state.advance(TRevisionPhase.COMPLETED)
        restored = TRevisionState.from_state_dict(state.state_dict())
        self.assertIs(restored.phase, TRevisionPhase.COMPLETED)


if __name__ == "__main__":
    unittest.main()
