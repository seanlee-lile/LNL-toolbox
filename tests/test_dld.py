from __future__ import annotations

import unittest

import torch

from lnl_toolbox.models.directional_diffusion import DirectionalDiffusion


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
