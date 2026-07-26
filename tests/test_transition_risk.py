from __future__ import annotations

import unittest

import torch

from lnl_toolbox.algorithms.transition_risk import (
    BackwardRiskCorrector,
    ForwardRiskCorrector,
)
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.noise.transition import KnownTransition


class TransitionRiskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        self.targets = torch.tensor([0, 1])
        self.identity = KnownTransition(torch.eye(2).numpy())

    def test_forward_identity_matches_cross_entropy(self) -> None:
        corrected = ForwardRiskCorrector().per_sample_risk(
            logits=self.logits,
            noisy_targets=self.targets,
            base_loss=CrossEntropyLoss(),
            transition=self.identity,
        )
        torch.testing.assert_close(corrected, CrossEntropyLoss()(self.logits, self.targets))

    def test_backward_identity_matches_cross_entropy(self) -> None:
        corrected = BackwardRiskCorrector().per_sample_risk(
            logits=self.logits,
            noisy_targets=self.targets,
            base_loss=CrossEntropyLoss(),
            transition=self.identity,
        )
        torch.testing.assert_close(corrected, CrossEntropyLoss()(self.logits, self.targets))

    def test_backward_returns_finite_corrected_risk(self) -> None:
        transition = KnownTransition(torch.tensor([[0.8, 0.2], [0.1, 0.9]]).numpy())
        corrected = BackwardRiskCorrector().per_sample_risk(
            logits=self.logits,
            noisy_targets=self.targets,
            base_loss=CrossEntropyLoss(),
            transition=transition,
        )
        self.assertTrue(torch.isfinite(corrected).all())
