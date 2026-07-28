import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lnl_toolbox.algorithms.cwd import CWDUnbiasedRisk
from lnl_toolbox.algorithms.cwd import CWDGlobalObjective
from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.estimators.cwd import CWDEstimator
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.training.snapshots import FeatureSnapshot
from lnl_toolbox.models.feature_output import FeatureOutput
from lnl_toolbox.training.pipeline import StandardNoisyERMPipeline


class _FeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward_with_features(self, inputs: torch.Tensor) -> FeatureOutput:
        return FeatureOutput(self.classifier(inputs), inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs).logits


class CWDTest(unittest.TestCase):
    def test_estimator_builds_hashed_classwise_artifact(self) -> None:
        snapshot = FeatureSnapshot(np.asarray([[1., 0.], [0., 1.], [1., 1.], [0., 0.]]), np.asarray([0, 1, 0, 1]), np.arange(4), "fixture", "train")
        artifact = CWDEstimator().estimate(snapshot)
        self.assertEqual(artifact.values.shape, (2, 2))
        self.assertTrue(artifact.artifact_hash)

    def test_cwd_risk_is_differentiable(self) -> None:
        logits = torch.randn(3, 2, requires_grad=True)
        values = CWDUnbiasedRisk().per_sample_risk(logits=logits, noisy_targets=torch.tensor([0, 1, 0]), base_loss=CrossEntropyLoss())
        values.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_global_objective_matches_identity_artifact_and_is_differentiable(self) -> None:
        snapshot = FeatureSnapshot(
            np.asarray([[1., 0.], [0., 1.], [1., 1.], [0., 0.]]),
            np.asarray([0, 1, 0, 1]),
            np.arange(4),
            "fixture",
            "train",
        )
        artifact = CWDEstimator().estimate(snapshot)
        model = _FeatureModel()
        features = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32)
        output = model.forward_with_features(features)
        value = CWDGlobalObjective(artifact).compute(
            model=model,
            logits=output.logits,
            features=output.features,
            noisy_targets=torch.tensor([0, 1]),
            sample_indices=torch.tensor([0, 1]),
            base_loss=CrossEntropyLoss(),
            metadata={},
        )
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())

    def test_pipeline_prepares_feature_statistic_and_binds_consumer(self) -> None:
        model = _FeatureModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = DataLoader([
            {"input": torch.tensor([1., 0.]), "target": torch.tensor(0), "index": torch.tensor(0)},
            {"input": torch.tensor([0., 1.]), "target": torch.tensor(1), "index": torch.tensor(1)},
        ], batch_size=2)
        pipeline = StandardNoisyERMPipeline.from_config({
            "statistic_estimator": {"name": "cwd"},
            "objective_consumer": {"name": "cwd"},
        })
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            artifacts = pipeline.prepare(
                model=model,
                optimizer=optimizer,
                loader=loader,
                device=torch.device("cpu"),
                dataset="fixture",
                split="train",
                run_dir=directory,
            )
            self.assertIsNotNone(artifacts.feature_snapshot)
            self.assertIsNotNone(artifacts.statistic)
            self.assertIs(pipeline.objective_consumer.statistic, artifacts.statistic)
            algorithm = SupervisedClassificationAlgorithm(
                model,
                optimizer,
                CrossEntropyLoss(),
                torch.device("cpu"),
                objective_consumer=pipeline.objective_consumer,
            )
            algorithm.setup(ExperimentContext(Path(directory)))
            result = algorithm.step(
                Batch({
                    "input": torch.tensor([[1., 0.], [0., 1.]]),
                    "target": torch.tensor([0, 1]),
                    "index": torch.tensor([0, 1]),
                }),
                RunState(phase="train"),
            )
            self.assertTrue(np.isfinite(result.metrics["loss"]))
            checkpoint_state = pipeline.state_dict()
            restored = StandardNoisyERMPipeline.from_config({
                "statistic_estimator": {"name": "cwd"},
                "objective_consumer": {"name": "cwd"},
            })
            restored.restore_for_resume(
                directory,
                checkpoint_state=checkpoint_state,
                component_states=pipeline.component_state_dict(),
                dataset="fixture",
                split="train",
            )
            self.assertEqual(
                restored.artifacts.statistic.artifact_hash,
                artifacts.statistic.artifact_hash,
            )

    def test_singular_flip_matrix_fails_explicitly(self) -> None:
        snapshot = FeatureSnapshot(
            np.asarray([[1., 0.], [0., 1.]]),
            np.asarray([0, 1]),
            np.arange(2),
            "fixture",
            "train",
        )
        with self.assertRaisesRegex(ValueError, "identifiable"):
            CWDEstimator(label_flip_matrix=np.ones((2, 2)) / 2).estimate(snapshot)


if __name__ == "__main__":
    unittest.main()
