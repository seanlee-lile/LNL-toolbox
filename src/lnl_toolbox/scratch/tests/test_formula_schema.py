import unittest

from lnl_toolbox.scratch.formula import FormulaSpec, FormulaValidationError, validate_formula


def _spec(**overrides):
    value = {
        "id": "user/test_formula",
        "name": "Test Formula",
        "description": "test",
        "inputs": {"logits": {}, "targets": {"type": "labels"}},
        "parameters": {},
        "steps": [
            {"id": "probabilities", "block": "softmax", "bindings": {"logits": "logits"}},
            {"id": "score", "block": "gather_by_label", "bindings": {"values": "probabilities", "labels": "targets"}},
        ],
        "outputs": {"score": {"source": "score"}},
        "metadata": {},
    }
    value.update(overrides)
    return value


class FormulaSchemaTest(unittest.TestCase):
    def test_roundtrip_and_safe_steps(self):
        spec = validate_formula(_spec())
        self.assertIsInstance(spec, FormulaSpec)
        self.assertEqual(spec.to_dict()["steps"][0]["block"], "softmax")

    def test_add_operation_is_a_formula_safe_binary_primitive(self):
        spec = validate_formula(_spec(
            inputs={"loss_a": {}, "loss_b": {}},
            steps=[{"id": "loss", "block": "add", "bindings": {"left": "loss_a", "right": "loss_b"}}],
            outputs={"loss": {"source": "loss"}},
        ))
        self.assertEqual(spec.steps[0].block, "add")

    def test_unknown_or_unsafe_block_rejected(self):
        with self.assertRaises(FormulaValidationError):
            validate_formula(_spec(steps=[{"id": "model", "block": "create_model", "bindings": {}}]))

    def test_forward_reference_and_missing_input_rejected(self):
        with self.assertRaises(FormulaValidationError):
            validate_formula(_spec(steps=[{"id": "score", "block": "gather_by_label", "bindings": {"values": "later", "labels": "targets"}}]))

    def test_parameter_range_is_checked(self):
        with self.assertRaises(FormulaValidationError):
            validate_formula(_spec(parameters={"q": {"type": "float", "default": 2.0, "maximum": 1.0}}))
