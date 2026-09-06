"""Execute the browser's actual insertion rules without starting a server."""

from pathlib import Path
import shutil
import subprocess
import unittest


class WebAvailabilityTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI behavior checks")
    def test_slot_scope_and_move_rules(self):
        script = Path(__file__).resolve().parents[1] / "web" / "scratch.js"
        result = subprocess.run(
            [shutil.which("node"), "-e", r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const sandbox = {location: {pathname: '/'}, assert};
vm.createContext(sandbox);
vm.runInContext(source.slice(0, source.indexOf('const UI_CATEGORIES')), sandbox);
vm.runInContext(`
const define = (id, requires = [], provides = [], extra = {}) => ({id, name: id, kind: 'action', requires, provides, placement: ['any'], params: {}, ...extra});
state.blocks = [
  define('source', [], ['save_as'], {params: {save_as: {default: 'model'}}}),
  define('fixed', ['model']),
  define('forward', ['model'], ['save_as'], {params: {model: {default: 'model'}, save_as: {default: 'logits'}}, placement: ['batch']}),
  define('inplace', ['model'], ['model']),
  define('epoch_loop', [], ['epoch'], {kind: 'loop', placement: ['top']}),
  define('batch_loop', [], ['batch'], {kind: 'loop', placement: ['epoch']}),
  define('condition', [], [], {kind: 'condition'}),
];
const step = (block, id, params = {}, steps) => ({block, _uiId: id, params, ...(steps ? {steps} : {})});
const at = (parentId, index) => ({parentId, index, context: getPlacementContext(parentId)});
const root = (index) => at('__root__', index);
state.recipe = {settings: {}, steps: [step('source', 's')]};
assert.equal(availabilityReason(blockInfo('fixed'), root(1)), null, 'fixed slot without a parameter default is usable');
assert.match(availabilityReason(blockInfo('fixed'), root(0)), /model/, 'future outputs cannot be consumed');
assert.equal(canInsert(blockInfo('fixed'), '__root__', null, 0, true), true, 'unfinished inputs allow a new draft to be placed');
assert.equal(canInsert(blockInfo('forward'), '__root__', null, 0, true), false, 'draft placement still enforces layer restrictions');
const draft = paletteDraftFor(blockInfo('fixed'));
draft.params.model = 'custom_model';
assert.match(availabilityReason(blockInfo('fixed'), root(1)), /custom_model/, 'palette draft uses the explicitly chosen input');
assert.deepEqual(state.recipe.steps[0].params, {}, 'draft edits do not mutate existing recipe steps');
state.paletteDraft = null;
assert.match(availabilityReason(blockInfo('fixed'), root(2)), /失效/, 'out-of-range target is rejected');
assert.match(availabilityReason(blockInfo('fixed'), at('deleted', 0)), /失效/, 'deleted target is rejected');
assert.match(availabilityReason(blockInfo('fixed'), at('s', 0)), /失效/, 'actions cannot hold children');
state.recipe.steps = [step('epoch_loop', 'e', {}, [step('batch_loop', 'b', {}, [step('condition', 'c', {}, [])])])];
state.recipe.settings = {model_a: {}};
assert.equal(getPlacementContext('c'), 'batch', 'condition inherits its surrounding loop');
assert.equal(availabilityReason(blockInfo('forward'), at('c', 0), null, {model: 'model_a'}), null);
assert.match(availabilityReason(blockInfo('forward'), at('c', 0), null, {model: ''}), /缺少/, 'invalid explicit input is not replaced with its default');
state.recipe.steps[0].steps[0].steps.push(step('forward', 'f', {model: 'model_a'}));
assert.equal(availabilityReason(blockInfo('forward'), at('b', 0), 'f'), null, 'moving respects custom peer-model bindings');
state.recipe.steps[0].steps[0].steps[0].steps.push(step('forward', 'nested', {model: 'model_a'}));
assert.match(availabilityReason(blockInfo('condition'), root(1), 'c'), /层级/, 'moving a container must check the placement of its children');
assert.equal(availabilityReason(blockInfo('epoch_loop'), root(1)), null, 'sequential training stages are legal');
assert.equal(availabilityReason(blockInfo('batch_loop'), at('e', 1)), null, 'multiple batch loops are legal');
assert.match(availabilityReason(blockInfo('condition'), at('c', 0), 'c'), /自己的子层/, 'self-nesting rejected');
state.recipe = {settings: {}, steps: [step('inplace', 'i')]};
assert.match(availabilityReason(blockInfo('inplace'), root(1), 'i'), /缺少/, 'moving cannot supply its own prerequisite');
state.recipe = {settings: {}, steps: [step('source', 's'), step('fixed', 'consumer')]};
const original = JSON.stringify(state.recipe);
assert.match(availabilityReason(blockInfo('source'), root(2), 's'), /断开输入/, 'moving a producer past its consumer is rejected');
assert.equal(JSON.stringify(state.recipe), original, 'checking a move must not change the recipe');
assert.equal(availabilityReason(blockInfo('source'), root(1), 's'), null, 'same-position move is harmless');
state.recipe.steps.push(step('forward', 'unfinished'));
assert.equal(availabilityReason(blockInfo('source'), root(1), 's'), null, 'unrelated existing missing inputs do not block edits');
state.recipe = {settings: {}, steps: [step('epoch_loop', 'e', {}, [
  step('source', 'checkpoint', {save_as: 'checkpoint'}),
  step('batch_loop', 'b', {}, [step('source', 'tmp', {save_as: 'temporary'})]),
  step('condition', 'c', {}, [step('source', 'maybe', {save_as: 'maybe'})]),
])]};
assert.ok(availableKeysBefore(root(1)).has('checkpoint'), 'epoch-level outputs survive the epoch loop');
assert.ok(!availableKeysBefore(root(1)).has('temporary'), 'batch temporaries remain scoped');
assert.ok(!availableKeysBefore(root(1)).has('maybe'), 'conditional child outputs are not assumed available');
`, sandbox);
console.log('Scratch availability behavior checks passed');
""", str(script)],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
