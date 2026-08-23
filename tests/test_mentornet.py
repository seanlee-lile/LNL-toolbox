from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from lnl_toolbox.algorithms.mentornet import (
    MentorNetWeightProvider,
    MovingPercentileState,
)
from lnl_toolbox.algorithms.update_policy import (
    ParameterUpdateInput,
    StepMilestoneUpdatePolicy,
)
from lnl_toolbox.core import RunState
from lnl_toolbox.models.cifar_resnet import cifar_resnet101
from lnl_toolbox.models.mentor_wide_resnet import MentorWideResNet101
from lnl_toolbox.models.mentornet import OfficialMentorNet, MentorNet
from lnl_toolbox.plugins.builtin.catalog import (
    build_builtin_parameter_update_policy,
    build_builtin_weight_provider,
)
from lnl_toolbox.training.mentor_artifacts import MentorArtifact
from lnl_toolbox.training.mentor_learning import prepare_trusted_mentor_features
from lnl_toolbox.treatments.weights import SupervisedWeightInput


class MentorNetTest(unittest.TestCase):
    def _artifact(self, path: Path) -> MentorArtifact:
        model = MentorNet(num_labels=1)
        artifact = MentorArtifact.create(
            architecture=model.architecture(),
            feature_schema={"label": "fixed_zero"},
            source={"dataset": "trusted"},
            model_state=model.state_dict(),
        )
        artifact.save(path)
        return artifact

    def test_model_returns_bounded_vector_for_non_multiple_sequence(self) -> None:
        model = MentorNet(num_labels=1)
        weights = model(
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([-0.1, 0.0, 0.1]),
            torch.zeros(3, dtype=torch.long),
            torch.full((3,), 25, dtype=torch.long),
        )
        self.assertEqual(tuple(weights.shape), (3,))
        self.assertTrue(bool(((weights >= 0) & (weights <= 1)).all()))

    def test_official_model_uses_one_timestep_per_sample(self) -> None:
        model = OfficialMentorNet()
        values = model(
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([-0.1, 0.0, 0.1]),
            torch.zeros(3, dtype=torch.long),
            torch.full((3,), 18, dtype=torch.long),
        )
        self.assertEqual(tuple(values.shape), (3,))
        self.assertTrue(bool(((values >= 0) & (values <= 1)).all()))
        self.assertEqual(model.architecture()["implementation"], "official")

    def test_moving_percentile_round_trip(self) -> None:
        state = MovingPercentileState(0.75, 0.95)
        state.update(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        restored = MovingPercentileState(0.75, 0.95)
        restored.load_state_dict(state.state_dict())
        self.assertEqual(restored.value, state.value)

    def test_provider_burn_in_and_resume_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mentor.pt"
            artifact = self._artifact(path)
            provider = MentorNetWeightProvider(
                str(path),
                10,
                burn_in_fraction=0.2,
                dropout_schedule=((0.5, 10),),
                seed=7,
            )
            request = SupervisedWeightInput(
                logits=torch.randn(4, 3),
                noisy_targets=torch.zeros(4, dtype=torch.long),
                sample_indices=torch.arange(4),
                per_sample_loss=torch.tensor([0.1, 0.2, 0.3, 0.4]),
                metadata={"epoch": 0},
            )
            first = provider.compute(request)
            saved = provider.state_dict()
            expected = provider.compute(request)
            restored = MentorNetWeightProvider(
                str(path),
                10,
                burn_in_fraction=0.2,
                dropout_schedule=((0.5, 10),),
                seed=999,
            )
            restored.load_state_dict(saved)
            actual = restored.compute(request)
            self.assertTrue(torch.equal(actual.sample_weights, expected.sample_weights))
            self.assertEqual(saved["artifact_hash"], artifact.artifact_hash)
            self.assertFalse(first.sample_weights.requires_grad)

    def test_artifact_detects_tampered_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mentor.pt"
            self._artifact(path)
            payload = torch.load(path, weights_only=False)
            payload["artifact_hash"] = "bad"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                MentorArtifact.load(path)

    def test_step_milestone_policy_and_plugin(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        policy = StepMilestoneUpdatePolicy([1], gamma=0.1)
        state = RunState(step=1)
        objective = model(torch.ones(1, 2)).sum()
        policy.update(ParameterUpdateInput(objective, model, optimizer, state))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)
        built = build_builtin_parameter_update_policy(
            {"name": "step_milestone", "milestones": [1]}
        )
        self.assertIsInstance(built, StepMilestoneUpdatePolicy)

    def test_weight_provider_plugin_and_resnet101(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mentor.pt"
            self._artifact(path)
            provider = build_builtin_weight_provider(
                {
                    "name": "mentornet",
                    "artifact_path": str(path),
                    "total_epochs": 10,
                }
            )
            self.assertIsInstance(provider, MentorNetWeightProvider)
        model = cifar_resnet101(num_classes=100, base_width=4)
        self.assertEqual(model(torch.randn(1, 3, 32, 32)).shape, (1, 100))

    def test_official_student_decay_is_weighted_and_conv_only(self) -> None:
        model = MentorWideResNet101(num_classes=10, num_residual_units=1, width_multiplier=0.1)
        penalty = model.weighted_parameter_decay(torch.ones(4))
        self.assertGreater(float(penalty.item()), 0.0)
        self.assertTrue(penalty.requires_grad)

    def test_feature_preparation_accepts_shared_non_cifar_data_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = prepare_trusted_mentor_features(
                {
                    "seed": 3,
                    "data": {
                        "name": "synthetic_binary_2d",
                        "train_size": 18,
                        "validation_size": 0,
                        "test_size": 6,
                    },
                    "noise": {"name": "symmetric", "rate": 0.2, "seed": 3},
                    "loader": {"batch_size": 6, "num_workers": 0},
                    "student_trainer": {"trusted_size": 18, "epochs": 1},
                    "student_model": {"name": "mlp", "width": 8},
                    "student_optimizer": {"name": "sgd", "lr": 0.1},
                },
                directory,
            )
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
