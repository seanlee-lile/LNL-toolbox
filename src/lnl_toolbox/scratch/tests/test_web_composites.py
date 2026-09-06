"""Editing a visual module must not depend on its original exact sequence."""
from pathlib import Path
import shutil
import subprocess
import unittest


class WebCompositeTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js required for browser logic tests")
    def test_module_membership_survives_edits(self):
        script = Path(__file__).resolve().parents[1] / "web" / "scratch.js"
        result = subprocess.run([shutil.which("node"), "-e", r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const context = vm.createContext({location: {pathname: '/'}, assert});
vm.runInContext(source.slice(0, source.indexOf('const UI_CATEGORIES')) +
  source.slice(source.indexOf('function insertionComposite('), source.indexOf('function addStepAtTarget(')) +
  source.slice(source.indexOf('const COMPOSITE_DEFINITIONS'), source.indexOf('function renderStepNode(')) +
  source.slice(source.indexOf('function adjacentPairAt('), source.indexOf('function renderAdjacentPair(')) +
  '\nfunction markDirty() {}', context);
vm.runInContext(`
const mk = (block) => ({block, _uiId: makeUiId(), params: {}});
const steps = state.recipe.steps = [mk('set_seed'), mk('select_device'), mk('backward'), mk('optimizer_step')];
const initial = detectCompositeRanges(steps);
assert.equal(adjacentPairAt(steps, 0).name, '设置实验环境');
assert.equal(adjacentPairAt(steps, 2), null, 'do not compact other adjacent operations');
const dataPair = [mk('load_dataset'), mk('inspect_dataset_semantics')];
const serialized = JSON.stringify(stripUiFields(dataPair));
assert.equal(adjacentPairAt(dataPair, 0).members.length, 2);
assert.equal(JSON.stringify(stripUiFields(dataPair)), serialized, 'pair detection must preserve execution');
assert.equal(adjacentPairAt([dataPair[0], mk('set_value'), dataPair[1]], 0), null, 'never cross intervening operations');
dataPair[0]._uiComposite = {id: 'a'}; dataPair[1]._uiComposite = {id: 'b'};
assert.equal(adjacentPairAt(dataPair, 0), null, 'never cross large-module boundaries');
const first = initial[0], second = initial[1];
const firstKey = compositeKey('__root__', first), secondKey = compositeKey('__root__', second);
state.compositeExpanded.add(firstKey);
const extra = mk('set_value');
insertStep('__root__', 1, extra);
let ranges = detectCompositeRanges(steps);
assert.equal(ranges.length, 2, 'inserting an unrelated operation must not dissolve the module');
assert.equal(ranges[0].end - ranges[0].start, 3);
assert.equal(compositeKey('__root__', ranges[0]), firstKey);
assert.equal(compositeKey('__root__', ranges[1]), secondKey, 'later modules keep identity when shifted');
assert.ok(state.compositeExpanded.has(compositeKey('__root__', ranges[0])));
const end = ranges[0].end;
insertStep('__root__', end, mk('set_value'), {index: end, compositeId: first.groupId});
ranges = detectCompositeRanges(steps);
assert.equal(ranges[0].end, 4, 'explicit trailing insertion belongs to the module');
insertStep('__root__', 0, mk('set_value'), {index: 0});
ranges = detectCompositeRanges(steps);
assert.equal(ranges[0].start, 1, 'outside boundary insertion stays outside');
const removed = steps.splice(1, 1)[0];
ranges = detectCompositeRanges(steps);
assert.equal(compositeKey('__root__', ranges[0]), firstKey, 'deleting original first member preserves identity');
steps.splice(1, 0, removed);
assert.equal(detectCompositeRanges(steps)[0].end - detectCompositeRanges(steps)[0].start, 4, 'undo rejoins original module');
state.compositeUngrouped.add(firstKey);
insertStep('__root__', 2, mk('set_value'));
assert.ok(state.compositeUngrouped.has(compositeKey('__root__', detectCompositeRanges(steps)[0])), 'explicit ungroup persists');
assert.ok(!JSON.stringify(stripUiFields(state.recipe)).includes('_uiComposite'), 'visual membership never enters runtime recipe');
`, context);
""", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
