from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from lnl_toolbox.algorithms.mc_ldce import MCLDCEObjective
from lnl_toolbox.estimators.mc_ldce import (
    MCLDCEEstimator,
    build_label_imputation_matrix,
    estimate_clean_prior,
)
from lnl_toolbox.noise.transition import TransitionArtifact
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.models.mc_ldce_cnn import MCLDCECifarCNN
from lnl_toolbox.training.mc_ldce_experiment import _prepare_fixed_feature_classifier
from lnl_toolbox.training.snapshots import FeatureSnapshot


class _LinearFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2, bias=False)

    def forward(self, value):
        return self.classifier(value)


class MCLDCETest(unittest.TestCase):
    def test_paper_cnn_exposes_128_dimensional_features(self) -> None:
        output = forward_with_features(MCLDCECifarCNN(10), torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output.logits.shape), (2, 10))
        self.assertEqual(tuple(output.features.shape), (2, 128))

    def test_fixed_feature_stage_disables_bias_and_feature_drift(self) -> None:
        torch.manual_seed(3)
        model = MCLDCECifarCNN(10)
        inputs = torch.randn(2, 3, 32, 32)
        model.eval()
        before = forward_with_features(model, inputs).features.detach().clone()

        _prepare_fixed_feature_classifier(model)
        model.train()
        after = forward_with_features(model, inputs).features.detach()

        self.assertIsNone(model.classifier.bias)
        self.assertTrue(torch.equal(before, after))
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertEqual(trainable, ["classifier.weight"])

    def test_fixed_feature_objective_updates_only_classifier(self) -> None:
        model = _LinearFeatures()
        _prepare_fixed_feature_classifier(model)
        self.assertIsNone(model.classifier.bias)
        self.assertEqual(
            [name for name, value in model.named_parameters() if value.requires_grad],
            ["classifier.weight"],
        )

    def test_identity_transition_recovers_joint_centroid(self) -> None:
        snapshot = FeatureSnapshot(
            np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]),
            np.array([0, 0, 1, 1]),
            np.array([3, 1, 4, 2]),
            "fixture",
            "train",
        )
        transition = TransitionArtifact(np.eye(2), "known")
        artifact = MCLDCEEstimator().estimate(snapshot, transition)
        np.testing.assert_allclose(artifact.values, [[1.0, 0.0], [0.0, 1.5]])
        self.assertEqual(
            artifact.metadata["transition_artifact_hash"], transition.artifact_hash
        )

    def test_prior_and_coefficient_are_explicit(self) -> None:
        transition = np.array([[0.8, 0.2], [0.1, 0.9]])
        clean = np.array([0.6, 0.4])
        observed = clean @ transition
        np.testing.assert_allclose(estimate_clean_prior(observed, transition), clean)
        coefficient = build_label_imputation_matrix(clean, transition)
        self.assertEqual(coefficient.shape, (2, 2))
        self.assertEqual(np.linalg.matrix_rank(coefficient), 2)

    def test_global_objective_matches_expansion_and_has_gradient(self) -> None:
        snapshot = FeatureSnapshot(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0, 1]),
            np.array([0, 1]),
            "fixture",
            "train",
        )
        artifact = MCLDCEEstimator().estimate(
            snapshot, TransitionArtifact(np.eye(2), "known")
        )
        model = _LinearFeatures()
        features = torch.tensor(snapshot.features, dtype=torch.float32)
        objective = MCLDCEObjective(artifact).compute(
            model=model,
            logits=model(features),
            features=features,
            noisy_targets=torch.tensor([0, 1]),
            sample_indices=torch.tensor([0, 1]),
            base_loss=nn.CrossEntropyLoss(),
            metadata={},
        )
        objective.backward()
        self.assertTrue(torch.isfinite(objective))
        self.assertIsNotNone(model.classifier.weight.grad)


if __name__ == "__main__":
    unittest.main()
