import unittest

import torch

from lnl_toolbox.scratch import ScratchContext
from lnl_toolbox.scratch.formula import execute_formula, get_formula


class FormulaEquivalenceTest(unittest.TestCase):
    def test_gce_matches_formal_legacy_loss(self):
        from lnl_toolbox.losses.torch_losses import GeneralizedCrossEntropyLoss

        values = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, 0.0]])
        labels = torch.tensor([2, 1])
        logits = values.clone().requires_grad_()
        context = ScratchContext({"logits": logits, "targets": labels})
        execute_formula(get_formula("builtin/gce"), context, parameter_values={"q": 0.7})
        reference_logits = values.clone().requires_grad_()
        expected = GeneralizedCrossEntropyLoss(q=0.7)(reference_logits, labels).mean()
        self.assertTrue(torch.allclose(context["loss"], expected, atol=1e-6, rtol=1e-6))
        context["loss"].backward()
        expected.backward()
        self.assertTrue(torch.allclose(logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-6))

    def test_apl_matches_public_nce_rce_composition(self):
        from lnl_toolbox.losses.torch_losses import ActivePassiveLoss, NormalizedCrossEntropyLoss, ReverseCrossEntropyLoss

        values = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, 0.0]])
        labels = torch.tensor([2, 1])
        logits = values.clone().requires_grad_()
        context = ScratchContext({"logits": logits, "targets": labels})
        execute_formula(get_formula("builtin/apl"), context, parameter_values={"log_zero": -9.210340371976184})
        reference_logits = values.clone().requires_grad_()
        expected = ActivePassiveLoss(NormalizedCrossEntropyLoss(), ReverseCrossEntropyLoss(log_zero=-9.210340371976184))(reference_logits, labels)
        self.assertTrue(torch.allclose(context["loss"], expected, atol=1e-5, rtol=1e-5))
        context["loss"].sum().backward()
        expected.sum().backward()
        self.assertTrue(torch.allclose(logits.grad, reference_logits.grad, atol=1e-5, rtol=1e-5))

    def test_loss_correction_matches_forward_risk_corrector(self):
        from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
        from lnl_toolbox.algorithms.transition_risk import ForwardRiskCorrector
        from lnl_toolbox.noise.transition import KnownTransition

        values = torch.tensor([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.0]])
        labels = torch.tensor([2, 1])
        transition = torch.tensor([[0.8, 0.2, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]])
        logits = values.clone().requires_grad_()
        context = ScratchContext({"logits": logits, "targets": labels, "transition": transition})
        execute_formula(get_formula("builtin/transition_corrected_risk"), context)
        reference_logits = values.clone().requires_grad_()
        expected = ForwardRiskCorrector().per_sample_risk(logits=reference_logits, noisy_targets=labels, base_loss=CrossEntropyLoss(), transition=KnownTransition(transition.numpy())).mean()
        self.assertTrue(torch.allclose(context["loss"], expected, atol=1e-6, rtol=1e-6))
        context["loss"].backward()
        expected.backward()
        self.assertTrue(torch.allclose(logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-6))
