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

    def test_add_operation_is_a_formula_kind_binary_primitive(self):
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

    def test_input_and_parameter_required_metadata_roundtrips(self):
        spec = validate_formula(_spec(
            inputs={"logits": {"type": "tensor", "description": "model scores", "required": True}, "targets": {"type": "labels"}},
            parameters={"q": {"type": "float", "default": 0.7, "minimum": 0.0, "maximum": 1.0, "options": [0.7], "required": True}},
            steps=[{"id": "powered", "block": "elementwise_power", "bindings": {"input": "logits"}, "parameters": {"q": "q"}}],
            outputs={"powered": {"source": "powered"}},
        ))
        payload = spec.to_dict()
        self.assertTrue(payload["inputs"]["logits"]["required"])
        self.assertEqual(payload["parameters"]["q"]["options"], [0.7])
        self.assertTrue(payload["parameters"]["q"]["required"])
