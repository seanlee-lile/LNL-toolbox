from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from lnl_toolbox.algorithms.dld import (
    DLDPrecorrectionArtifact,
    build_knn_label_distribution,
    precorrect_two_views,
)
from lnl_toolbox.models.directional_diffusion import DirectionalDiffusion


class DLDAlgorithmTest(unittest.TestCase):
    def _features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weak = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0], [-0.9, 0.1]],
            dtype=np.float32,
        )
        strong = weak + np.asarray(
            [[0.0, 0.01], [0.01, 0.0], [0.0, -0.01], [-0.01, 0.0], [0.0, 0.01], [0.01, 0.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        return weak, strong, labels

    def test_knn_distribution_is_aligned_and_normalized(self) -> None:
        weak, _, labels = self._features()
        distribution, neighbors = build_knn_label_distribution(weak, labels, k=2)
        self.assertEqual(distribution.shape, (6, 3))
        self.assertEqual(neighbors.shape, (6, 2))
        np.testing.assert_allclose(distribution.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.array_equal(neighbors, neighbors.astype(np.int64)))

    def test_pre_correction_artifact_round_trip_and_index_identity(self) -> None:
        weak, strong, labels = self._features()
        indices = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.int64)
        artifact = precorrect_two_views(weak, strong, labels, indices, k=2, seed=7)
        self.assertEqual(artifact.global_indices.tolist(), indices.tolist())
        self.assertEqual(artifact.partition.shape, labels.shape)
        self.assertTrue(np.all(artifact.loss_weights > 0.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dld.npz"
            artifact.save(path)
            restored = DLDPrecorrectionArtifact.load(path)
        self.assertEqual(restored.artifact_hash, artifact.artifact_hash)
        np.testing.assert_allclose(restored.weak_targets, artifact.weak_targets)


class DLDModelTest(unittest.TestCase):
    def test_cosine_schedule_matches_official_state_shape(self) -> None:
        model = DirectionalDiffusion(
            3,
            5,
            num_timesteps=32,
            hidden_width=16,
            time_dim=8,
            schedule="cosine",
            image_base_width=2,
        )
        self.assertEqual(model.schedule, "cosine")
        self.assertEqual(tuple(model.alpha_cumsum.shape), (32,))
        self.assertEqual(tuple(model.beta2_cumsum.shape), (32,))
        self.assertTrue(torch.isfinite(model.alpha_cumsum).all())
        self.assertTrue(torch.isfinite(model.beta_cumsum).all())
        self.assertGreater(float(model.alpha_cumsum[-1]), float(model.alpha_cumsum[0]))

    def test_residual_and_noise_paths_are_independent(self) -> None:
        torch.manual_seed(4)
        model = DirectionalDiffusion(3, 5, num_timesteps=12, hidden_width=16, time_dim=8)
        y_input = torch.zeros(4, 3)
        y0 = torch.softmax(torch.randn(4, 3), dim=1)
        features = torch.randn(4, 5)
        timesteps = torch.tensor([0, 1, 10, 11])
        predicted_residual, predicted_noise, residual, noise, y_t = model.forward_t(
            y_input, y0, features, timesteps, noise=torch.zeros_like(y0)
        )
        self.assertEqual(predicted_residual.shape, (4, 3))
        self.assertEqual(predicted_noise.shape, (4, 3))
        self.assertTrue(torch.isfinite(y_t).all())
        residual_loss = (predicted_residual - residual).square().mean()
        residual_loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.residual_model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.noise_model.parameters()))

    def test_sampling_is_deterministic_and_normalized(self) -> None:
        torch.manual_seed(9)
        model = DirectionalDiffusion(4, 6, num_timesteps=16, hidden_width=16, time_dim=8)
        features = torch.randn(3, 6)
        first = model.sample(features, sampling_timesteps=5)
        second = model.sample(features, sampling_timesteps=5)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first.sum(dim=1), torch.ones(3), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(first).all())

    def test_image_condition_is_trainable_and_used_by_sampling(self) -> None:
        torch.manual_seed(12)
        model = DirectionalDiffusion(
            3,
            5,
            num_timesteps=12,
            hidden_width=16,
            time_dim=8,
            image_base_width=2,
        )
        y_input = torch.zeros(4, 3)
        y0 = torch.softmax(torch.randn(4, 3), dim=1)
        features = torch.randn(4, 5)
        images = torch.randn(4, 3, 32, 32)
        timesteps = torch.tensor([0, 1, 10, 11])
        predicted_residual, _, residual, _, _ = model.forward_t(
            y_input, y0, features, timesteps, noise=torch.zeros_like(y0), images=images
        )
        (predicted_residual - residual).square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.residual_image_encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.noise_image_encoder.parameters()))
        model.eval()
        probabilities = model.sample(features, images=images, sampling_timesteps=4)
        self.assertEqual(tuple(probabilities.shape), (4, 3))
        self.assertTrue(torch.isfinite(probabilities).all())


if __name__ == "__main__":
    unittest.main()
