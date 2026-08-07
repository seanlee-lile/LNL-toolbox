import unittest

import torch
from torch import nn

from lnl_toolbox.algorithms.dld import (
    DirectionalDiffusionSchedule,
    accelerated_timesteps,
    sample_labels,
)


class _Constant(nn.Module):
    def __init__(self, classes: int, value: float) -> None:
        super().__init__()
        self.num_classes = classes
        self.value = value

    def forward(self, y_t, y_n, features, timestep):
        return torch.full_like(y_t, self.value)


class DLDSamplingTest(unittest.TestCase):
    def test_five_step_sequence_and_manual_reverse(self) -> None:
        sequence = accelerated_timesteps(20, 5)
        self.assertEqual(len(sequence), 5)
        self.assertEqual(sequence[-1][1], -1)
        features = torch.ones(2, 3)
        result = sample_labels(
            _Constant(2, 1.0), _Constant(2, 0.0), features,
            DirectionalDiffusionSchedule.average(20), inference_steps=5,
        )
        self.assertTrue(torch.allclose(result, -torch.ones(2, 2), atol=1e-6))

    def test_sampling_is_deterministic_and_restores_train_mode(self) -> None:
        torch.manual_seed(4)
        direction = _Constant(3, 0.25)
        noise = _Constant(3, -0.25)
        features = torch.randn(4, 2)
        schedule = DirectionalDiffusionSchedule.average(10)
        first = sample_labels(direction, noise, features, schedule, inference_steps=5)
        second = sample_labels(direction, noise, features, schedule, inference_steps=5)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(direction.training)
        self.assertEqual(tuple(first.shape), (4, 3))


if __name__ == "__main__":
    unittest.main()
