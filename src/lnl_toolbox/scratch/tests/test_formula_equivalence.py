import unittest

import torch

from lnl_toolbox.scratch import ScratchContext
from lnl_toolbox.scratch.formula import execute_formula, get_formula


class FormulaEquivalenceTest(unittest.TestCase):
    def test_gce_matches_formal_legacy_loss(self):
        from lnl_toolbox.losses.torch_losses import GeneralizedCrossEntropyLoss

        logits = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, 0.0]], requires_grad=True)
        labels = torch.tensor([2, 1])
        context = ScratchContext({"logits": logits, "targets": labels})
        execute_formula(get_formula("builtin/gce"), context, parameter_values={"q": 0.7})
        expected = GeneralizedCrossEntropyLoss(q=0.7)(logits, labels).mean()
        self.assertTrue(torch.allclose(context["loss"], expected, atol=1e-6, rtol=1e-6))

    def test_apl_matches_public_nce_rce_composition(self):
        from lnl_toolbox.losses.torch_losses import ActivePassiveLoss, NormalizedCrossEntropyLoss, ReverseCrossEntropyLoss

        logits = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, 0.0]])
        labels = torch.tensor([2, 1])
        context = ScratchContext({"logits": logits, "targets": labels})
        execute_formula(get_formula("builtin/apl"), context, parameter_values={"log_zero": -9.210340371976184})
        expected = ActivePassiveLoss(NormalizedCrossEntropyLoss(), ReverseCrossEntropyLoss(log_zero=-9.210340371976184))(logits, labels)
        self.assertTrue(torch.allclose(context["loss"], expected, atol=1e-5, rtol=1e-5))

    def test_loss_correction_matches_forward_risk_corrector(self):
        from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
        from lnl_toolbox.algorithms.transition_risk import ForwardRiskCorrector
        from lnl_toolbox.noise.transition import KnownTransition

        logits = torch.tensor([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.0]])
        labels = torch.tensor([2, 1])
        transition = torch.tensor([[0.8, 0.2, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]])
        context = ScratchContext({"logits": logits, "targets": labels, "transition": transition})
        execute_formula(get_formula("builtin/transition_corrected_risk"), context)
        expected = ForwardRiskCorrector().per_sample_risk(logits=logits, noisy_targets=labels, base_loss=CrossEntropyLoss(), transition=KnownTransition(transition.numpy())).mean()
        self.assertTrue(torch.allclose(context["loss"], expected, atol=1e-6, rtol=1e-6))
