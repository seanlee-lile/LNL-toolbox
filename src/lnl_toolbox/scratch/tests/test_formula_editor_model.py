from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


class FormulaEditorExpressionModelTest(unittest.TestCase):
    """Exercise the browser expression model without requiring a browser UI."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.node = shutil.which("node")
        if not cls.node:
            raise unittest.SkipTest("Node.js is required for the formula expression model test")

    def run_node(self, body: str) -> None:
        bootstrap = f"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync({str((WEB_ROOT / 'scratch.js').resolve())!r}, 'utf8');
const context = {{
  document: {{ getElementById: () => null, addEventListener: () => {{}}, querySelectorAll: () => [] }},
  location: {{ pathname: '/' }},
  fetch: () => new Promise(() => {{}}),
  console, crypto: {{ randomUUID: () => 'test-id' }}, setTimeout, clearTimeout,
}};
context.globalThis = context;
vm.runInNewContext(source, context);
vm.runInNewContext({body!r}, context);
"""
        result = subprocess.run([self.node, "-e", bootstrap], cwd=WEB_ROOT, capture_output=True, text=True)
        if result.returncode:
            self.fail(result.stderr or result.stdout)

    def test_formula_steps_round_trip_to_editable_expression(self) -> None:
        self.run_node(
            """
const steps = [
  {id: 'probabilities', block: 'softmax', bindings: {logits: 'logits'}, parameters: {}},
  {id: 'target_probability', block: 'gather_by_label', bindings: {values: 'probabilities', labels: 'targets'}, parameters: {}},
  {id: 'powered_probability', block: 'elementwise_power', bindings: {input: 'target_probability'}, parameters: {q: 'q'}},
  {id: 'loss', block: 'mean_loss', bindings: {input: 'powered_probability'}, parameters: {}},
];
const expression = formulaStepsToExpression(steps, 'loss', new Set(['logits', 'targets']), new Set(['q']));
if (!expression || expression.kind !== 'operation' || expression.block !== 'mean_loss') throw new Error('root expression was not reconstructed');
const powered = expression.bindings.input;
if (powered.block !== 'elementwise_power' || powered.parameters.q.kind !== 'parameter') throw new Error('nested parameter expression was not reconstructed');
powered.parameters.q = expressionParameter('q_prime');
const roundTrip = expressionToFormulaSteps(expression, 'loss');
if (roundTrip.at(-1).id !== 'loss' || roundTrip.at(-1).block !== 'mean_loss') throw new Error('root output was not generated from the expression');
if (!roundTrip.some((step) => step.block === 'elementwise_power' && step.parameters.q === 'q_prime')) throw new Error('edited parameter was not serialized');
"""
        )

    def test_nested_combination_can_be_built_without_step_ids(self) -> None:
        self.run_node(
            """
state.blocks = [
  {id: 'elementwise_multiply', name: 'Multiply', kind: 'action', formula_safe: true, requires: ['left', 'right'], params: {left: {type: 'slot'}, right: {type: 'slot'}, save_as: {type: 'slot'}}},
  {id: 'add', name: 'Add', kind: 'action', formula_safe: true, requires: ['left', 'right'], params: {left: {type: 'slot'}, right: {type: 'slot'}, save_as: {type: 'slot'}}},
];
const expression = {kind: 'operation', block: 'add', bindings: {
  left: {kind: 'operation', block: 'elementwise_multiply', bindings: {left: expressionParameter('alpha'), right: expressionInput('loss_a')}, parameters: {}},
  right: {kind: 'operation', block: 'elementwise_multiply', bindings: {left: expressionParameter('beta'), right: expressionInput('loss_b')}, parameters: {}},
}, parameters: {}};
const steps = expressionToFormulaSteps(expression, 'loss');
if (steps.at(-1).id !== 'loss' || steps.at(-1).block !== 'add') throw new Error('combination root was not serialized as loss');
if (steps.filter((step) => step.block === 'elementwise_multiply').length !== 2) throw new Error('nested terms were not serialized');
"""
        )

    def test_array_editing_preserves_array_and_supports_reorder(self) -> None:
        self.run_node(
            """
const array = {kind: 'array', items: [expressionInput('a'), expressionInput('b')]};
const root = {kind: 'operation', block: 'weighted_sum', bindings: {}, parameters: {terms: array}};
const second = array.items[1];
if (!insertExpressionSibling(root, second, 'after', expressionInput('c'))) throw new Error('array insertion failed');
if (array.items.length !== 3 || array.items[2].name !== 'c') throw new Error('array insertion changed the wrong container');
if (!moveExpressionArrayItem(root, array.items[2], -1) || array.items[1].name !== 'c') throw new Error('array reorder failed');
if (!duplicateExpressionNode(root, array.items[1])) throw new Error('array duplication failed');
if (array.items.length !== 4) throw new Error('array duplicate did not add an item');
const removed = pruneExpressionNode(root, array.items[0]);
if (!removed || removed.parameters.terms.kind !== 'array' || removed.parameters.terms.items.length !== 3) throw new Error('deleting an array item must keep an array node');
"""
        )

    def test_shared_expression_nodes_are_serialized_once(self) -> None:
        self.run_node(
            """
state.blocks = [{id: 'softmax', name: 'Softmax', kind: 'action', formula_safe: true, requires: ['logits'], params: {logits: {type: 'slot'}, save_as: {type: 'slot'}}}, {id: 'add', name: 'Add', kind: 'action', formula_safe: true, requires: ['left', 'right'], params: {left: {type: 'slot'}, right: {type: 'slot'}, save_as: {type: 'slot'}}}];
const shared = {kind: 'operation', block: 'softmax', bindings: {logits: expressionInput('x')}, parameters: {}};
const expression = {kind: 'operation', block: 'add', bindings: {left: shared, right: shared}, parameters: {}};
const steps = expressionToFormulaSteps(expression, 'y');
if (steps.filter((step) => step.block === 'softmax').length !== 1) throw new Error('shared operation was serialized twice');
if (steps.at(-1).bindings.left !== steps.at(-1).bindings.right) throw new Error('shared references did not point to one step');
"""
        )

    def test_parameter_schema_line_keeps_structural_metadata_helpers(self) -> None:
        self.run_node(
            """
const schema = {type: 'float', default: 0.7, minimum: 0, maximum: 1, options: [0.7], description: 'GCE exponent', required: true};
if (parameterSchemaLine('q', schema) !== 'q:float=0.7') throw new Error('parameter line should only encode editable fields');
"""
        )

    def test_advanced_execution_details_are_closed_by_default(self) -> None:
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="formula-editor-advanced" class="formula-editor-advanced">', html)
        self.assertNotIn('id="formula-editor-advanced" class="formula-editor-advanced" open', html)
        self.assertIn('class="formula-editor-side-details"', html)


if __name__ == "__main__":
    unittest.main()
