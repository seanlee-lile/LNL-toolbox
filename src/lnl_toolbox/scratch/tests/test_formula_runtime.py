import unittest

import torch

from lnl_toolbox.scratch import ScratchContext
from lnl_toolbox.scratch.formula import execute_formula, get_formula, register_formula, unregister_formula


class FormulaRuntimeTest(unittest.TestCase):
    def test_builtin_ce_preserves_autograd_and_reduction(self):
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        context = ScratchContext({"logits": logits, "targets": torch.tensor([0, 1])})
        execute_formula(get_formula("builtin/standard_ce"), context)
        self.assertEqual(context["loss"].ndim, 0)
        context["loss"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_per_sample_formula_has_no_implicit_mean(self):
        context = ScratchContext({"logits": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "targets": torch.tensor([0, 1])})
        execute_formula(get_formula("builtin/confidence_score"), context)
        self.assertEqual(tuple(context["score"].shape), (2,))

    def test_explicit_detach_stops_gradient(self):
        spec = {
            "id": "user/detached_score",
            "name": "Detached Score",
            "inputs": {"logits": {}, "targets": {"type": "labels"}},
            "steps": [
                {"id": "detached_logits", "block": "detach", "bindings": {"input": "logits"}},
                {"id": "probabilities", "block": "softmax", "bindings": {"logits": "detached_logits"}},
            ],
            "outputs": {"score": {"source": "probabilities"}},
        }
        logits = torch.randn(2, 3, requires_grad=True)
        context = ScratchContext({"logits": logits, "targets": torch.tensor([0, 1])})
        execute_formula(spec, context)
        self.assertFalse(context["score"].requires_grad)

    def test_parameter_override_changes_gce(self):
        logits = torch.tensor([[2.0, 0.0]])
        targets = torch.tensor([0])
        a, b = ScratchContext({"logits": logits, "targets": targets}), ScratchContext({"logits": logits, "targets": targets})
        execute_formula(get_formula("builtin/gce"), a, parameter_values={"q": 0.5})
        execute_formula(get_formula("builtin/gce"), b, parameter_values={"q": 0.7})
        self.assertNotEqual(float(a["loss"]), float(b["loss"]))

    def test_nested_formula_provenance_is_promoted_to_parent(self):
        register_formula({
            "id": "user/provenance_inner",
            "name": "Provenance Inner",
            "inputs": {"x": {}},
            "steps": [{"id": "out", "block": "detach", "bindings": {"input": "x"}}],
            "outputs": {"value": {"source": "out"}},
        })
        register_formula({
            "id": "user/provenance_outer",
            "name": "Provenance Outer",
            "inputs": {"x": {}},
            "steps": [{
                "id": "inner",
                "block": "formula/user/provenance_inner",
                "bindings": {"x": "x"},
            }],
            "outputs": {"value": {"source": "inner"}},
        })
        try:
            context = ScratchContext({"x": torch.tensor([1.0])})
            execute_formula(get_formula("user/provenance_outer"), context)
            self.assertEqual(
                {item["formula_id"] for item in context["_formula_provenance"]},
                {"user/provenance_inner", "user/provenance_outer"},
            )
        finally:
            unregister_formula("user/provenance_outer")
            unregister_formula("user/provenance_inner")
