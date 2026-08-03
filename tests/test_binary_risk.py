import unittest

import torch

from lnl_toolbox.algorithms.binary_risk import NatarajanUnbiasedRisk
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss


class BinaryRiskTest(unittest.TestCase):
    def test_unbiased_risk_is_finite_and_differentiable(self) -> None:
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
        targets = torch.tensor([0, 1])
        risk = NatarajanUnbiasedRisk(0.2, 0.1)
        values = risk.per_sample_risk(logits=logits, noisy_targets=targets, base_loss=CrossEntropyLoss())
        self.assertEqual(values.shape, (2,))
        self.assertTrue(torch.isfinite(values).all())
        values.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rates_must_be_identifiable(self) -> None:
        with self.assertRaises(ValueError):
            NatarajanUnbiasedRisk(0.7, 0.3)


if __name__ == "__main__":
    unittest.main()
