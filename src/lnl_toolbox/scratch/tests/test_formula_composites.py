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
    def _assert_formula_matches_block(self, block_id, formula_id, context, block_params, formula_params, input_bindings=None, output_name="result"):
        direct_context = _execute_block(block_id, context, save_as=output_name, **block_params)
        formula_context = ScratchContext(dict(context))
        execute_formula(
            get_formula(formula_id),
            formula_context,
            parameter_values=formula_params,
            input_bindings=input_bindings,
            output_bindings={next(iter(get_formula(formula_id).outputs)): output_name},
        )
        torch.testing.assert_close(direct_context[output_name], formula_context[output_name], atol=1e-6, rtol=1e-6)

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

    def test_numeric_parameter_branches_match_the_composite_definitions(self) -> None:
        """Exercise non-default numeric controls instead of checking metadata only."""
        values = torch.tensor([0.0, 0.25, 0.75], dtype=torch.float64)
        self._assert_formula_matches_block("negative_log", "builtin/negative_log", {"x": values}, {"input": "x", "minimum": 1e-4}, {"epsilon": 1e-4}, {"x": "x"})
        numerator = torch.tensor([1.0, 2.0], dtype=torch.float64)
        denominator = torch.tensor([0.0, 4.0], dtype=torch.float64)
        self._assert_formula_matches_block("safe_divide", "builtin/safe_divide", {"n": numerator, "d": denominator}, {"numerator": "n", "denominator": "d", "minimum": 0.25}, {"epsilon": 0.25}, {"numerator": "n", "denominator": "d"})
        with self.assertRaises(ValueError):
            _execute_block("safe_divide", {"n": numerator, "d": denominator}, numerator="n", denominator="d", minimum=0.25, on_invalid="error", save_as="result")
        with self.assertRaises(ValueError):
            execute_formula(
                get_formula("builtin/safe_divide"), ScratchContext({"n": numerator, "d": denominator}),
                parameter_values={"epsilon": 0.25, "on_invalid": "error"},
                input_bindings={"numerator": "n", "denominator": "d"}, output_bindings={"quotient": "result"},
            )
        matrix = torch.tensor([[1.0, 3.0], [2.0, 2.0]], dtype=torch.float64)
        self._assert_formula_matches_block("row_normalize", "builtin/row_normalize", {"matrix": matrix}, {"input": "matrix", "minimum": 0.5}, {"epsilon": 0.5}, {"x": "matrix"})
        probabilities = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float64)
        self._assert_formula_matches_block("sharpen_distribution", "builtin/sharpen_distribution", {"p": probabilities}, {"input": "p", "temperature": 0.8, "epsilon": 1e-6}, {"temperature": 0.8, "epsilon": 1e-6}, {"probabilities": "p"})
        weights = torch.tensor([0.0, 2.0, 3.0], dtype=torch.float64)
        self._assert_formula_matches_block("normalize_nonnegative_weights", "builtin/normalize_nonnegative_weights", {"w": weights}, {"weights": "w", "epsilon": 0.25}, {"epsilon": 0.25}, {"weights": "w"})
        first = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float64)
        second = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float64)
        self._assert_formula_matches_block("compose_transition", "builtin/compose_transition", {"a": first, "b": second}, {"first": "a", "second": "b", "epsilon": 1e-6}, {"epsilon": 1e-6}, {"first": "a", "second": "b"})
        logits = torch.tensor([[1.0, -1.0], [-0.2, 0.8]], dtype=torch.float64)
        labels = torch.tensor([0, 1])
        self._assert_formula_matches_block("per_sample_ce", "builtin/per_sample_ce", {"z": logits, "y": labels}, {"logits": "z", "labels": "y", "epsilon": 1e-6}, {"epsilon": 1e-6}, {"logits": "z", "labels": "y"})
        self._assert_formula_matches_block("nce_loss", "builtin/nce_loss", {"z": logits, "y": labels}, {"logits": "z", "labels": "y", "epsilon": 1e-6}, {"epsilon": 1e-6}, {"logits": "z", "labels": "y"})
        probabilities = torch.tensor([[0.3, 0.7], [0.8, 0.2]], dtype=torch.float64)
        prior = torch.tensor([0.55, 0.45], dtype=torch.float64)
        self._assert_formula_matches_block("prior_kl", "builtin/prior_kl", {"p": probabilities, "q": prior}, {"probabilities": "p", "prior": "q", "epsilon": 1e-6}, {"epsilon": 1e-6}, {"probabilities": "p", "prior": "q"})
        left = torch.ones((2, 2), dtype=torch.float64)
        right = torch.full((2, 2), 3.0, dtype=torch.float64)
        blend_weight = torch.tensor([0.2, 0.8], dtype=torch.float64)
        self._assert_formula_matches_block("weighted_blend", "builtin/weighted_blend", {"l": left, "r": right, "w": blend_weight}, {"left": "l", "right": "r", "weight": "w", "clamp_weight": True}, {}, {"left": "l", "right": "r", "weight": "w"}, output_name="blended")
        a, b = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
        self._assert_formula_matches_block("weighted_sum", "builtin/weighted_sum", {"a": a, "b": b}, {"terms": ["a", "b"], "weights": [0.25, 0.75]}, {"weight_a": 0.25, "weight_b": 0.75}, {"a": "a", "b": "b"}, output_name="result")

    def test_high_dimensional_mse_and_weighted_blend_match_direct_blocks(self) -> None:
        predicted = torch.arange(2 * 3 * 2 * 2, dtype=torch.float64).reshape(2, 3, 2, 2)
        target = predicted + torch.tensor(0.5, dtype=torch.float64)
        direct = _execute_block(
            "mean_squared_error", {"predicted": predicted, "target": target},
            predicted="predicted", target="target", reduction="per_sample", save_as="result",
        )["result"]
        composed = ScratchContext({"predicted": predicted, "target": target})
        execute_formula(
            get_formula("builtin/mean_squared_error"), composed,
            parameter_values={"reduction": "per_sample"},
            input_bindings={"predicted": "predicted", "target": "target"},
            output_bindings={"loss": "result"},
        )
        torch.testing.assert_close(composed["result"], direct)
        torch.testing.assert_close(direct, (predicted - target).reshape(2, -1).square().mean(dim=1))

        left = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        right = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        weight = torch.tensor([0.25, 0.75], dtype=torch.float64)
        direct = _execute_block(
            "weighted_blend", {"left": left, "right": right, "weight": weight},
            left="left", right="right", weight="weight", clamp_weight=True, save_as="result",
        )["result"]
        composed = ScratchContext({"left": left, "right": right, "weight": weight})
        execute_formula(
            get_formula("builtin/weighted_blend"), composed,
            input_bindings={"left": "left", "right": "right", "weight": "weight"},
            output_bindings={"blended": "result"},
        )
        torch.testing.assert_close(composed["result"], direct)

    def test_t_revision_ratio_composite_preserves_denominator_failure(self) -> None:
        probabilities = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float64)
        noisy = torch.tensor([[0.0, 1.0], [0.4, 0.6]], dtype=torch.float64)
        labels = torch.tensor([0, 1])
        with self.assertRaises(ValueError):
            _execute_block(
                "t_revision_importance_ratio",
                {"p": probabilities, "q": noisy, "y": labels},
                probabilities="p", noisy_probabilities="q", labels="y", denominator_floor=1e-6,
                save_as="weights", denominators_as="denominators",
            )
        composed = ScratchContext({"p": probabilities, "q": noisy, "y": labels})
        with self.assertRaises(ValueError):
            execute_formula(
                get_formula("builtin/t_revision_importance_ratio"), composed,
                parameter_values={"denominator_floor": 1e-6},
                input_bindings={"probabilities": "p", "noisy_probabilities": "q", "labels": "y"},
                output_bindings={"weights": "weights", "denominators": "denominators"},
            )

    def test_dispatch_branches_have_explicit_compositions(self) -> None:
        """Exercise mask/reduction/variadic branches with visible step chains."""
        predicted = torch.tensor([[1.0, 2.0], [5.0, 7.0], [2.0, 1.0]])
        target = torch.tensor([[0.0, 1.0], [4.0, 8.0], [2.0, 3.0]])
        mask = torch.tensor([True, False, True])
        direct = _execute_block(
            "mean_squared_error",
            {"predicted": predicted, "target": target, "mask": mask},
            predicted="predicted", target="target", mask="mask", reduction="per_sample", save_as="result",
        )["result"]
        masked_mse = {
            "id": "user/test_masked_mse_branch", "name": "Masked MSE branch",
            "inputs": {"predicted": {}, "target": {}, "mask": {}},
            "steps": [
                {"id": "selected_indices", "block": "mask_to_indices", "bindings": {"mask": "mask"}},
                {"id": "selected_predicted", "block": "index_select", "bindings": {"input": "predicted", "indices": "selected_indices"}, "parameters": {"dim": 0}},
                {"id": "selected_target", "block": "index_select", "bindings": {"input": "target", "indices": "selected_indices"}, "parameters": {"dim": 0}},
                {"id": "difference", "block": "subtract", "bindings": {"minuend": "selected_predicted", "subtrahend": "selected_target"}},
                {"id": "squared", "block": "elementwise_power", "bindings": {"input": "difference"}, "parameters": {"q": 2.0}},
                {"id": "result", "block": "reduce_mean", "bindings": {"input": "squared"}, "parameters": {"dim": -1}},
            ],
            "outputs": {"loss": {"source": "result"}},
        }
        composed = ScratchContext({"predicted": predicted, "target": target, "mask": mask})
        execute_formula(masked_mse, composed, output_bindings={"loss": "result"})
        torch.testing.assert_close(direct, composed["result"])

        logits = torch.tensor([[1.0, -1.0], [2.0, 0.0], [-1.0, 3.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]])
        direct = _execute_block(
            "soft_target_cross_entropy",
            {"logits": logits, "targets": targets, "mask": mask},
            logits="logits", targets="targets", mask="mask", reduction="per_sample", save_as="result",
        )["result"]
        masked_ce = {
            "id": "user/test_masked_soft_ce_branch", "name": "Masked soft CE branch",
            "inputs": {"logits": {}, "targets": {}, "mask": {}},
            "steps": [
                {"id": "selected_indices", "block": "mask_to_indices", "bindings": {"mask": "mask"}},
                {"id": "selected_logits", "block": "index_select", "bindings": {"input": "logits", "indices": "selected_indices"}, "parameters": {"dim": 0}},
                {"id": "selected_targets", "block": "index_select", "bindings": {"input": "targets", "indices": "selected_indices"}, "parameters": {"dim": 0}},
                {"id": "log_probabilities", "block": "log_softmax", "bindings": {"logits": "selected_logits"}},
                {"id": "weighted", "block": "elementwise_multiply", "bindings": {"left": "selected_targets", "right": "log_probabilities"}},
                {"id": "class_sum", "block": "reduce_sum", "bindings": {"input": "weighted"}, "parameters": {"dim": -1}},
                {"id": "result", "block": "negate", "bindings": {"input": "class_sum"}},
            ],
            "outputs": {"loss": {"source": "result"}},
        }
        composed = ScratchContext({"logits": logits, "targets": targets, "mask": mask})
        execute_formula(masked_ce, composed, output_bindings={"loss": "result"})
        torch.testing.assert_close(direct, composed["result"])

        values = torch.tensor([1.0, 2.0, 4.0])
        empty = torch.zeros(3, dtype=torch.bool)
        direct_empty = _execute_block(
            "masked_mean", {"values": values, "mask": empty}, values="values", mask="mask", denominator="selected", empty="zero", save_as="result",
        )["result"]
        self.assertEqual(float(direct_empty), 0.0)
        direct_batch = _execute_block(
            "masked_mean", {"values": values, "mask": mask}, values="values", mask="mask", denominator="batch", empty="zero", save_as="result",
        )["result"]
        self.assertAlmostEqual(float(direct_batch), float(values[mask].sum() / values.numel()))

        left = torch.tensor([[1.0], [2.0]])
        right = torch.tensor([[5.0], [7.0]])
        weights = torch.tensor([1.5, -0.5])
        direct = _execute_block(
            "weighted_blend", {"left": left, "right": right, "weight": weights},
            left="left", right="right", weight="weight", clamp_weight=False, save_as="result",
        )["result"]
        raw_blend = {
            "id": "user/test_raw_blend_branch", "name": "Unclamped blend branch",
            "inputs": {"left": {}, "right": {}, "weight": {}},
            "steps": [
                {"id": "one", "block": "ones_like", "bindings": {"input": "weight"}},
                {"id": "one_minus_weight", "block": "subtract", "bindings": {"minuend": "one", "subtrahend": "weight"}},
                {"id": "weight_row", "block": "unsqueeze", "bindings": {"input": "weight"}, "parameters": {"dim": -1}},
                {"id": "one_minus_weight_row", "block": "unsqueeze", "bindings": {"input": "one_minus_weight"}, "parameters": {"dim": -1}},
                {"id": "left_part", "block": "elementwise_multiply", "bindings": {"left": "weight_row", "right": "left"}},
                {"id": "right_part", "block": "elementwise_multiply", "bindings": {"left": "one_minus_weight_row", "right": "right"}},
                {"id": "result", "block": "add", "bindings": {"left": "left_part", "right": "right_part"}},
            ],
            "outputs": {"blended": {"source": "result"}},
        }
        composed = ScratchContext({"left": left, "right": right, "weight": weights})
        execute_formula(raw_blend, composed, output_bindings={"blended": "result"})
        torch.testing.assert_close(direct, composed["result"])

        terms = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        direct = _execute_block(
            "weighted_sum", {"a": terms[0], "b": terms[1], "c": terms[2]},
            terms=["a", "b", "c"], weights=[0.5, 1.0, -2.0], save_as="result",
        )["result"]
        variadic = {
            "id": "user/test_variadic_weighted_sum", "name": "Variadic weighted sum",
            "inputs": {"a": {}, "b": {}, "c": {}},
            "steps": [
                {"id": "wa", "block": "constant", "parameters": {"value": 0.5}},
                {"id": "wb", "block": "constant", "parameters": {"value": 1.0}},
                {"id": "wc", "block": "constant", "parameters": {"value": -2.0}},
                {"id": "ta", "block": "elementwise_multiply", "bindings": {"left": "wa", "right": "a"}},
                {"id": "tb", "block": "elementwise_multiply", "bindings": {"left": "wb", "right": "b"}},
                {"id": "tc", "block": "elementwise_multiply", "bindings": {"left": "wc", "right": "c"}},
                {"id": "ab", "block": "add", "bindings": {"left": "ta", "right": "tb"}},
                {"id": "result", "block": "add", "bindings": {"left": "ab", "right": "tc"}},
            ],
            "outputs": {"result": {"source": "result"}},
        }
        composed = ScratchContext({"a": terms[0], "b": terms[1], "c": terms[2]})
        execute_formula(variadic, composed)
        torch.testing.assert_close(direct, composed["result"])

    def test_builtin_variants_execute_each_result_changing_branch(self) -> None:
        """Branch controls are executable FormulaSpec variants, not notes."""
        predicted = torch.tensor([[1.0, 2.0], [5.0, 7.0], [2.0, 1.0]])
        target = torch.tensor([[0.0, 1.0], [4.0, 8.0], [2.0, 3.0]])
        mask = torch.tensor([True, False, True])
        for reduction in ("scalar", "per_sample"):
            for masked in (False, True):
                direct_context = _execute_block(
                    "mean_squared_error",
                    {"predicted": predicted, "target": target, **({"mask": mask} if masked else {})},
                    predicted="predicted", target="target", mask="mask" if masked else None,
                    reduction=reduction, save_as="result",
                )
                formula_context = ScratchContext({"predicted": predicted, "target": target, **({"mask": mask} if masked else {})})
                execute_formula(
                    get_formula("builtin/mean_squared_error"), formula_context,
                    parameter_values={"reduction": reduction},
                    input_bindings={"predicted": "predicted", "target": "target", **({"mask": "mask"} if masked else {})},
                    output_bindings={"loss": "result"},
                )
                torch.testing.assert_close(direct_context["result"], formula_context["result"])

        logits = torch.tensor([[1.0, -1.0], [2.0, 0.0], [-1.0, 3.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]])
        for reduction in ("mean", "per_sample"):
            for masked in (False, True):
                direct_context = _execute_block(
                    "soft_target_cross_entropy",
                    {"logits": logits, "targets": targets, **({"mask": mask} if masked else {})},
                    logits="logits", targets="targets", mask="mask" if masked else None,
                    reduction=reduction, save_as="result",
                )
                formula_context = ScratchContext({"logits": logits, "targets": targets, **({"mask": mask} if masked else {})})
                execute_formula(
                    get_formula("builtin/soft_target_cross_entropy"), formula_context,
                    parameter_values={"reduction": reduction},
                    input_bindings={"logits": "logits", "targets": "targets", **({"mask": "mask"} if masked else {})},
                    output_bindings={"loss": "result"},
                )
                torch.testing.assert_close(direct_context["result"], formula_context["result"])

        values = torch.tensor([1.0, 2.0, 4.0])
        for denominator in ("selected", "batch"):
            for empty in ("zero", "error"):
                current_mask = torch.zeros(3, dtype=torch.bool) if empty == "zero" else mask
                direct_context = _execute_block(
                    "masked_mean", {"values": values, "mask": current_mask},
                    values="values", mask="mask", denominator=denominator, empty=empty, save_as="result",
                )
                formula_context = ScratchContext({"values": values, "mask": current_mask})
                execute_formula(
                    get_formula("builtin/masked_mean"), formula_context,
                    parameter_values={"denominator": denominator, "empty": empty},
                    input_bindings={"values": "values", "mask": "mask"},
                    output_bindings={"loss": "result"},
                )
                torch.testing.assert_close(direct_context["result"], formula_context["result"])


if __name__ == "__main__":
    unittest.main()
