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
  {id: 'elementwise_multiply', name: 'Multiply', kind: 'action', formula_kind: 'primitive', requires: ['left', 'right'], params: {left: {type: 'slot'}, right: {type: 'slot'}, save_as: {type: 'slot'}}},
  {id: 'add', name: 'Add', kind: 'action', formula_kind: 'primitive', requires: ['left', 'right'], params: {left: {type: 'slot'}, right: {type: 'slot'}, save_as: {type: 'slot'}}},
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

    def test_constant_slot_operand_is_materialized_as_a_constant_step(self) -> None:
        self.run_node(
            """
state.blocks = [
  {id: 'constant', name: 'Constant', kind: 'action', formula_kind: 'primitive', requires: [], params: {value: {type: 'float'}, save_as: {type: 'slot'}}},
  {id: 'add', name: 'Add', kind: 'action', formula_kind: 'primitive', requires: ['left', 'right'], params: {left: {type: 'slot'}, right: {type: 'slot'}, save_as: {type: 'slot'}}},
];
const expression = {kind: 'operation', block: 'add', bindings: {
  left: expressionInput('x'),
  right: expressionConstant(1),
}, parameters: {}};
const steps = expressionToFormulaSteps(expression, 'y');
const constantIndex = steps.findIndex((step) => step.block === 'constant');
const addIndex = steps.findIndex((step) => step.block === 'add');
if (constantIndex < 0 || addIndex < 0 || constantIndex >= addIndex) throw new Error('constant operand was not emitted before its consumer');
const constant = steps[constantIndex];
const add = steps[addIndex];
if (constant.parameters.value !== 1 || add.bindings.right !== constant.id) throw new Error('constant slot binding was not serialized as a constant step reference');
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
state.blocks = [{id: 'softmax', name: 'Softmax', kind: 'action', formula_kind: 'primitive', requires: ['logits'], params: {logits: {type: 'slot'}, save_as: {type: 'slot'}}}, {id: 'add', name: 'Add', kind: 'action', formula_kind: 'primitive', requires: ['left', 'right'], params: {left: {type: 'slot'}, right: {type: 'slot'}, save_as: {type: 'slot'}}}];
const shared = {kind: 'operation', block: 'softmax', bindings: {logits: expressionInput('x')}, parameters: {}};
const expression = {kind: 'operation', block: 'add', bindings: {left: shared, right: shared}, parameters: {}};
const steps = expressionToFormulaSteps(expression, 'y');
if (steps.filter((step) => step.block === 'softmax').length !== 1) throw new Error('shared operation was serialized twice');
if (steps.at(-1).bindings.left !== steps.at(-1).bindings.right) throw new Error('shared references did not point to one step');
"""
        )

    def test_operation_templates_create_ui_holes_and_never_serialize_them(self) -> None:
        self.run_node(
            """
const add = expressionOperationNode({id: 'add', name: 'Add', requires: ['left', 'right'], params: {left: {type: 'slot'}, right: {type: 'slot'}}});
if (add.bindings.left.kind !== 'hole' || add.bindings.right.kind !== 'hole') throw new Error('binary operation did not create two holes');
if (!expressionContainsHole(add)) throw new Error('hole detection failed');
const steps = expressionToFormulaSteps(add, 'loss');
if (steps.length !== 1 || steps[0].bindings.left !== null || steps[0].bindings.right !== null) throw new Error('holes must serialize as incomplete bindings, not UI objects');
const unary = expressionOperationNode({id: 'negative_log', name: 'Log', requires: ['input'], params: {input: {type: 'slot'}}});
if (unary.bindings.input.kind !== 'hole') throw new Error('unary operation did not create one hole');
"""
        )

    def test_optional_formula_inputs_are_bindings_and_stay_absent_by_default(self) -> None:
        self.run_node(
            """
const mse = {id: 'mean_squared_error', name: 'Mean Squared Error', formula_kind: 'composite', requires: ['predicted', 'target'], params: {
  predicted: {type: 'slot'}, target: {type: 'slot'}, mask: {type: 'slot', required: false, default: 'mask'},
  reduction: {type: 'enum', default: 'scalar'}, save_as: {type: 'slot'},
}};
state.blocks = [mse];
const fresh = expressionOperationNode(mse);
if (Object.prototype.hasOwnProperty.call(fresh.parameters, 'mask')) throw new Error('optional mask was exposed as a parameter');
if (Object.prototype.hasOwnProperty.call(fresh.bindings, 'mask')) throw new Error('optional mask was silently connected by its schema default');
const loaded = formulaStepsToExpression([
  {id: 'loss', block: 'mean_squared_error', bindings: {predicted: 'predicted', target: 'target'}, parameters: {mask: 'mask', reduction: 'scalar'}},
], 'loss', new Set(['predicted', 'target', 'mask']), new Set());
if (loaded.bindings.mask?.kind !== 'input' || loaded.bindings.mask.name !== 'mask') throw new Error('optional mask was not promoted to an input binding');
if (Object.prototype.hasOwnProperty.call(loaded.parameters, 'mask')) throw new Error('promoted mask remained in parameters');
const replacement = {kind: 'operation', block: 'add', bindings: {left: expressionInput('a'), right: expressionInput('b')}, parameters: {}};
state.blocks.push({id: 'add', name: 'Add', requires: ['left', 'right'], params: {left: {type: 'slot'}, right: {type: 'slot'}, save_as: {type: 'slot'}}});
expressionReplaceOperation(replacement, 'mean_squared_error');
if (Object.prototype.hasOwnProperty.call(replacement.bindings, 'mask')) throw new Error('operation replacement manufactured an optional mask binding');
"""
        )

    def test_new_operation_focuses_the_next_hole_for_symbol_insertion(self) -> None:
        self.run_node(
            """
state.blocks = [{id: 'add', name: 'Add', kind: 'action', formula_kind: 'primitive', requires: ['left', 'right'], params: {left: {type: 'slot'}, right: {type: 'slot'}}}];
state.formulaEditor.expression = null;
state.formulaEditor.expressionSelection = null;
addExpressionOperation('add');
if (!state.formulaEditor.expressionSelection || state.formulaEditor.expressionSelection.kind !== 'hole') throw new Error('new operation should focus its first hole');
insertExpressionSymbol('input', 'a');
if (state.formulaEditor.expression.bindings.left.kind !== 'input' || state.formulaEditor.expression.bindings.left.name !== 'a') throw new Error('input did not fill the selected hole');
if (!state.formulaEditor.expressionSelection || state.formulaEditor.expressionSelection !== state.formulaEditor.expression.bindings.right) throw new Error('next hole was not selected after filling input');
"""
        )

    def test_formula_palette_uses_declared_operation_group_metadata(self) -> None:
        self.run_node(
            """
if (formulaEditorGroup({id: 'future_binary_operation', formula_group: '基础'}) !== '基础') throw new Error('declared basic operation group was ignored');
if (formulaEditorGroup({id: 'future_function_operation', formula_group: '函数'}) !== '函数') throw new Error('declared function operation group was ignored');
"""
        )

    def test_parameterized_operation_exposes_editable_variables_and_hides_output_slot(self) -> None:
        self.run_node(
            """
const parameterField = {value: ''};
document.getElementById = (id) => id === 'formula-parameters' ? parameterField : null;
renderFormulaParameterFields = () => {};
state.formulaEditor.parameterSchemas = {};
const affine = {id: 'affine_transform', name: 'Affine Transform', requires: ['input'], params: {
  input: {type: 'slot'}, scale: {type: 'float', default: 1}, bias: {type: 'float', default: 0}, save_as: {type: 'slot'},
}};
state.blocks = [affine];
ensureFormulaEditorParameters(affine);
if (!parameterField.value.includes('scale:float=1') || !parameterField.value.includes('bias:float=0')) throw new Error('affine parameters were not exposed as formula variables');
const expression = expressionOperationNode(affine);
if (expression.parameters.scale.kind !== 'parameter' || expression.parameters.bias.kind !== 'parameter') throw new Error('affine variables were not connected to parameter nodes');
if (Object.prototype.hasOwnProperty.call(expression.parameters, 'save_as')) throw new Error('output slot leaked into expression operands');
if (expressionNodeText(expression) !== '(scale × □) ＋ bias') throw new Error('affine expression is not shown as scale × input + bias');
"""
        )

    def test_loaded_operation_promotes_scalar_arguments_to_variables(self) -> None:
        self.run_node(
            """
const parameterField = {value: ''};
document.getElementById = (id) => id === 'formula-parameters' ? parameterField : null;
renderFormulaParameterFields = () => {};
state.formulaEditor.parameterSchemas = {};
const affine = {id: 'affine_transform', name: 'Affine Transform', requires: ['input'], params: {
  input: {type: 'slot'}, scale: {type: 'float', default: 1}, bias: {type: 'float', default: 0}, save_as: {type: 'slot'},
}};
state.blocks = [affine];
const loaded = {kind: 'operation', block: 'affine_transform',
  bindings: {input: expressionInput('x')},
  parameters: {scale: expressionConstant(2), bias: expressionConstant(-1), save_as: expressionConstant('loss')}};
exposeExpressionParameters(loaded);
if (!parameterField.value.includes('scale:float=2') || !parameterField.value.includes('bias:float=-1')) throw new Error('loaded scalar arguments were not promoted with their current values');
if (loaded.parameters.scale.kind !== 'parameter' || loaded.parameters.bias.kind !== 'parameter') throw new Error('loaded scalar arguments were not connected to parameter nodes');
if (expressionNodeEntries(loaded).some(([name]) => name === 'save_as')) throw new Error('output slot leaked into the loaded expression');
if (expressionNodeText(loaded) !== '(x × 2) ＋ -1' && expressionNodeText(loaded) !== '(scale × x) ＋ bias') throw new Error('loaded affine expression is not readable');
"""
        )

    def test_deleted_auto_parameter_stays_removed_across_operation_switches(self) -> None:
        self.run_node(
            """
const parameterField = {value: ''};
document.getElementById = (id) => id === 'formula-parameters' ? parameterField : null;
renderFormulaParameterFields = () => {};
state.formulaEditor.parameterSchemas = {};
state.formulaEditor.parameterStates = {};
const maximum = {id: 'maximum', name: 'Maximum', requires: ['input'], params: {
  input: {type: 'slot'}, maximum: {type: 'float', default: 1}, save_as: {type: 'slot'},
}};
const divide = {id: 'safe_divide', name: 'Safe Divide', requires: ['numerator', 'denominator'], params: {
  numerator: {type: 'slot'}, denominator: {type: 'slot'}, minimum: {type: 'float', default: 1e-12}, save_as: {type: 'slot'},
}};
state.blocks = [maximum, divide];
ensureFormulaEditorParameters(maximum, {}, {force: true});
if (!parameterField.value.includes('maximum:float=1')) throw new Error('operation parameter was not added');
transitionFormulaParameter('maximum', FORMULA_PARAMETER_STATUS.REMOVED, 'user');
parameterField.value = '';
state.formulaEditor.expression = {kind: 'operation', block: 'maximum', bindings: {input: expressionInput('x')}, parameters: {}};
exposeExpressionParameters(state.formulaEditor.expression);
if (parameterField.value.includes('maximum')) throw new Error('deleted parameter was resurrected during render');
ensureFormulaEditorParameters(divide, {}, {force: true});
if (!parameterField.value.includes('minimum:float=')) throw new Error('new operation parameter was not added');
if (parameterField.value.includes('maximum')) throw new Error('switching operations resurrected the deleted parameter');
"""
        )

    def test_deleted_shared_parameter_is_not_readded_by_another_operation(self) -> None:
        self.run_node(
            """
const parameterField = {value: ''};
document.getElementById = (id) => id === 'formula-parameters' ? parameterField : null;
renderFormulaParameterFields = () => {};
state.formulaEditor.parameterSchemas = {};
state.formulaEditor.parameterStates = {};
const clamp = {id: 'clamp_min', name: 'Clamp Minimum', requires: ['input'], params: {
  input: {type: 'slot'}, minimum: {type: 'float', default: 1e-12}, save_as: {type: 'slot'},
}};
const divide = {id: 'safe_divide', name: 'Safe Divide', requires: ['numerator', 'denominator'], params: {
  numerator: {type: 'slot'}, denominator: {type: 'slot'}, minimum: {type: 'float', default: 1e-12}, save_as: {type: 'slot'},
}};
state.blocks = [clamp, divide];
ensureFormulaEditorParameters(clamp, {}, {force: true});
transitionFormulaParameter('minimum', FORMULA_PARAMETER_STATUS.REMOVED, 'user');
parameterField.value = '';
ensureFormulaEditorParameters(divide, {}, {force: true});
if (parameterField.value.includes('minimum')) throw new Error('a shared deleted parameter was re-added automatically');
if (formulaParameterStates().minimum.status !== FORMULA_PARAMETER_STATUS.REMOVED) throw new Error('REMOVED state was not preserved across operation insertion');
"""
        )

    def test_deleting_last_parameter_clears_source_text_and_schema(self) -> None:
        self.run_node(
            """
const parameterField = {value: 'maximum:float=1'};
const parameterRows = {querySelectorAll: () => []};
document.getElementById = (id) => id === 'formula-parameters' ? parameterField : id === 'formula-parameter-fields' ? parameterRows : null;
state.formulaEditor.parameterSchemas = {maximum: {type: 'float', default: 1}};
state.formulaEditor.parameterStates = {};
updateFormulaParameterText();
if (parameterField.value !== '') throw new Error('deleting the last parameter did not clear the source textarea');
if (Object.prototype.hasOwnProperty.call(state.formulaEditor.parameterSchemas, 'maximum')) throw new Error('deleted parameter schema remained in editor state');
"""
        )

    def test_formula_selection_deletion_uses_the_editor_expression_model(self) -> None:
        self.run_node(
            """
draw = () => {};
const root = {kind: 'operation', block: 'add', bindings: {left: expressionInput('a'), right: expressionInput('b')}, parameters: {}};
state.formulaEditor.expression = root;
state.formulaEditor.expressionSelection = root.bindings.right;
if (!deleteSelectedFormulaExpression()) throw new Error('selected formula expression was not deleted');
if (state.formulaEditor.expression.kind !== 'input' || state.formulaEditor.expression.name !== 'a') throw new Error('Backspace deleted the wrong formula node');
if (state.formulaEditor.expressionSelection !== null) throw new Error('deleted formula node remained selected');
"""
        )

    def test_multiple_outputs_reuse_one_shared_expression_step(self) -> None:
        self.run_node(
            """
const shared = {kind: 'operation', block: 'softmax', bindings: {logits: expressionInput('x')}, parameters: {}};
const loss = {kind: 'operation', block: 'mean_loss', bindings: {input: shared}, parameters: {}};
const result = expressionToFormulaStepsForOutputs({loss, score: shared}, ['loss', 'score']);
if (result.steps.filter((step) => step.block === 'softmax').length !== 1) throw new Error('shared output expression was duplicated');
if (!result.sources.loss || result.sources.loss !== 'loss') throw new Error('loss source was not named after its output');
if (!result.sources.score || result.sources.score !== result.steps.find((step) => step.block === 'softmax').id) throw new Error('second output source is not the shared step');
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
