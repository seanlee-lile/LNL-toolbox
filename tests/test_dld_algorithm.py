import unittest

import numpy as np
import torch

from lnl_toolbox.algorithms.dld import (
    DLDAlgorithm,
    DLDLabelPredictor,
    DLDPreCorrectionArtifact,
    DirectionalDiffusionSchedule,
    construct_direction,
    dld_objective,
    sample_forward_state,
)


def _artifact() -> DLDPreCorrectionArtifact:
    pw = np.array([[0.8, 0.2], [0.2, 0.8]])
    y0 = np.eye(2)
    yn = np.array([[0.0, 0.0], [1.0, 0.0]])
    return DLDPreCorrectionArtifact(
        np.array([7, 3]), np.array([0, 1]), pw, pw, pw,
        np.array([0.0, 0.0]), np.array([0, 1]), y0, yn, yn - y0,
        np.array([[1.0, 2.0], [3.0, 4.0]]), {"test": True},
    )


class DLDDiffusionTest(unittest.TestCase):
    def test_schedule_endpoint_and_forward_equation(self) -> None:
        schedule = DirectionalDiffusionSchedule.average(4)
        self.assertAlmostEqual(float(schedule.alpha_bar[-1]), 1.0)
        self.assertAlmostEqual(float(schedule.beta_bar[-1]), 1.0)
        y0 = torch.tensor([[1.0, 0.0]])
        yn = torch.tensor([[0.0, 1.0]])
        epsilon = torch.tensor([[2.0, -2.0]])
        result = sample_forward_state(y0, construct_direction(y0, yn), torch.tensor([3]), epsilon, schedule)
        self.assertTrue(torch.allclose(result, yn + epsilon))

    def test_predictors_and_optimizers_are_independent_and_update(self) -> None:
        direction = DLDLabelPredictor(2, 2, hidden_dim=8, time_dim=4)
        noise = DLDLabelPredictor(2, 2, hidden_dim=8, time_dim=4)
        direction_optimizer = torch.optim.Adam(direction.parameters(), lr=1e-2)
        noise_optimizer = torch.optim.Adam(noise.parameters(), lr=1e-2)
        algorithm = DLDAlgorithm(
            direction_model=direction, noise_model=noise,
            direction_optimizer=direction_optimizer, noise_optimizer=noise_optimizer,
            direction_scheduler=None, noise_scheduler=None,
            schedule=DirectionalDiffusionSchedule.average(5), artifact=_artifact(),
            device="cpu", ema_decay=0.9,
        )
        self.assertFalse({id(p) for p in direction.parameters()} & {id(p) for p in noise.parameters()})
        before_d = [p.detach().clone() for p in direction.parameters()]
        before_n = [p.detach().clone() for p in noise.parameters()]
        metrics = algorithm.train_step(torch.tensor([3, 7]))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_d, direction.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_n, noise.parameters())))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_objective_rejects_nonfinite(self) -> None:
        good = torch.zeros(1, 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            dld_objective(good + float("nan"), good, good, good)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_predictor_cuda(self) -> None:
        model = DLDLabelPredictor(3, 4).cuda()
        result = model(torch.zeros(2, 3, device="cuda"), torch.zeros(2, 3, device="cuda"), torch.zeros(2, 4, device="cuda"), torch.tensor([0, 1], device="cuda"))
        self.assertEqual(tuple(result.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
