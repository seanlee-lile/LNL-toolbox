from __future__ import annotations

import unittest

import torch

from lnl_toolbox.scratch import ScratchContext
from lnl_toolbox.scratch.formula import execute_formula, get_formula
from lnl_toolbox.scratch.registry import get_block


def _execute_block(block_id: str, context: dict, **params):
    target = ScratchContext(context)
    get_block(block_id).execute(target, **params)
    return target


class FormulaCompositeEquivalenceTest(unittest.TestCase):
    def test_negative_log_value_and_gradient(self) -> None:
        values = torch.tensor([0.2, 0.7], dtype=torch.float64)
        direct_input = values.clone().requires_grad_()
        composed_input = values.clone().requires_grad_()
        direct = _execute_block("negative_log", {"x": direct_input}, input="x", minimum=1e-6, save_as="result")["result"]
        composed_context = ScratchContext({"x": composed_input})
        execute_formula(get_formula("builtin/negative_log"), composed_context, parameter_values={"epsilon": 1e-6}, output_bindings={"z": "result"})
        torch.testing.assert_close(direct, composed_context["result"])
        direct.sum().backward(); composed_context["result"].sum().backward()
        torch.testing.assert_close(direct_input.grad, composed_input.grad)

    def test_row_normalize_and_weighted_blend_values(self) -> None:
        rows = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
        direct = _execute_block("row_normalize", {"x": rows}, input="x", minimum=1e-12, save_as="result")["result"]
        composed_context = ScratchContext({"x": rows})
        execute_formula(get_formula("builtin/row_normalize"), composed_context, output_bindings={"normalized": "result"})
        torch.testing.assert_close(direct, composed_context["result"])

    def test_first_batch_composites_match_values_and_gradients(self) -> None:
        # Keep these comparisons against the existing Python blocks (or the
        # direct mathematical expression where no wrapper exists).  The YAML
        # definitions are therefore checked as executable compositions rather
        # than only as metadata.
        logits = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, 0.0]], dtype=torch.float64)
        labels = torch.tensor([2, 1])

        for block_id, formula_id in (("nce_loss", "builtin/nce_loss"), ("mae_loss", "builtin/mae_loss"), ("per_sample_ce", "builtin/per_sample_ce")):
            direct_logits = logits.clone().requires_grad_()
            formula_logits = logits.clone().requires_grad_()
            direct = _execute_block(block_id, {"logits": direct_logits, "labels": labels}, logits="logits", labels="labels", save_as="result")["result"]
            formula_context = ScratchContext({"logits": formula_logits, "labels": labels})
            execute_formula(get_formula(formula_id), formula_context, input_bindings={"logits": "logits", "labels": "labels"}, output_bindings={"loss": "result"})
            torch.testing.assert_close(direct, formula_context["result"], atol=1e-6, rtol=1e-6)
            direct.sum().backward()
            formula_context["result"].sum().backward()
            torch.testing.assert_close(direct_logits.grad, formula_logits.grad, atol=1e-6, rtol=1e-6)

        values = torch.tensor([[0.2, 0.8], [0.5, 0.5]], dtype=torch.float64)
        direct_values = values.clone().requires_grad_()
        formula_values = values.clone().requires_grad_()
        direct = _execute_block("sharpen_distribution", {"values": direct_values}, input="values", temperature=0.5, save_as="result")["result"]
        formula_context = ScratchContext({"values": formula_values})
        execute_formula(get_formula("builtin/sharpen_distribution"), formula_context, parameter_values={"temperature": 0.5}, input_bindings={"probabilities": "values"}, output_bindings={"sharpened": "result"})
        torch.testing.assert_close(direct, formula_context["result"], atol=1e-6, rtol=1e-6)
        direct.sum().backward(); formula_context["result"].sum().backward()
        torch.testing.assert_close(direct_values.grad, formula_values.grad, atol=1e-6, rtol=1e-6)

        probabilities = torch.tensor([[0.2, 0.8], [0.5, 0.5]], dtype=torch.float64)
        prior = torch.tensor([0.4, 0.6], dtype=torch.float64)
        direct_probabilities = probabilities.clone().requires_grad_()
        formula_probabilities = probabilities.clone().requires_grad_()
        direct = _execute_block("prior_kl", {"probabilities": direct_probabilities, "prior": prior}, probabilities="probabilities", prior="prior", save_as="result")["result"]
        formula_context = ScratchContext({"probabilities": formula_probabilities, "prior": prior})
        execute_formula(get_formula("builtin/prior_kl"), formula_context, input_bindings={"probabilities": "probabilities", "prior": "prior"}, output_bindings={"loss": "result"})
        torch.testing.assert_close(direct, formula_context["result"], atol=1e-6, rtol=1e-6)
        direct.backward(); formula_context["result"].backward()
        torch.testing.assert_close(direct_probabilities.grad, formula_probabilities.grad, atol=1e-6, rtol=1e-6)

        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        mask = torch.tensor([True, False, True])
        formula_context = ScratchContext({"values": values, "mask": mask})
        execute_formula(get_formula("builtin/masked_mean"), formula_context, input_bindings={"values": "values", "mask": "mask"}, output_bindings={"loss": "result"})
        torch.testing.assert_close(formula_context["result"], values[mask].mean())

        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right = left + 10
        weight = torch.tensor([0.25, 0.75])
        direct = _execute_block("weighted_blend", {"left": left, "right": right, "weight": weight}, save_as="result")["result"]
        composed_context = ScratchContext({"left": left, "right": right, "weight": weight})
        execute_formula(get_formula("builtin/weighted_blend"), composed_context, output_bindings={"blended": "result"})
        torch.testing.assert_close(direct, composed_context["result"])

    def test_loss_composites_match_public_blocks(self) -> None:
        logits = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, 0.0]], dtype=torch.float64)
        labels = torch.tensor([2, 1])
        for block_id, formula_id, params, output in (
            ("per_sample_ce", "builtin/per_sample_ce", {}, "loss"),
            ("nce_loss", "builtin/nce_loss", {}, "loss"),
            ("rce_loss", "builtin/rce_loss", {"log_zero": -4.0}, "loss"),
            ("mae_loss", "builtin/mae_loss", {}, "loss"),
        ):
            direct_context = _execute_block(block_id, {"logits": logits, "labels": labels}, logits="logits", labels="labels", save_as="result", **params)
            composed_context = ScratchContext({"logits": logits, "labels": labels})
            execute_formula(get_formula(formula_id), composed_context, parameter_values=params, output_bindings={output: "result"})
            torch.testing.assert_close(direct_context["result"], composed_context["result"], atol=1e-6, rtol=1e-6)

    def test_constant_operand_is_executable(self) -> None:
        context = ScratchContext({"x": torch.tensor([2.0, 3.0])})
        formula = {
            "id": "user/constant_operand_test",
            "name": "Constant Operand Test",
            "inputs": {"x": {}},
            "steps": [
                {"id": "one", "block": "constant", "parameters": {"value": 1.0}},
                {"id": "result", "block": "add", "bindings": {"left": "x", "right": "one"}},
            ],
            "outputs": {"y": {"source": "result"}},
        }
        execute_formula(formula, context)
        torch.testing.assert_close(context["y"], torch.tensor([3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
