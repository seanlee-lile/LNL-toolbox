"""Focused checks for Scratch Web source/derived state gating."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from lnl_toolbox.scratch.web.data_bridge import dataset_preflight_payload, _registered_fact
from lnl_toolbox.scratch.web.server import _dataset_preflight
from lnl_toolbox.scratch import validate_recipe


class ScratchWebStateGateTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js required for browser logic tests")
    def test_revision_invalidates_all_derived_results(self) -> None:
        script = Path(__file__).resolve().parents[1] / "web" / "scratch.js"
        result = subprocess.run(
            [
                shutil.which("node"),
                "-e",
                r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const context = vm.createContext({location: {pathname: '/'}, assert});
vm.runInContext(source.slice(0, source.indexOf('const UI_CATEGORIES')), context);
// The source-key helper is declared below the browser palette constants; use
// a deterministic test double while exercising the state gate in isolation.
vm.runInContext(`function datasetSourceKey(){ return state._testDatasetKey || null; }`, context);
vm.runInContext(`
state.blocks = [{id: 'load_dataset', provides: ['train_source']}];
state.recipe = {steps: [{block: 'load_dataset', params: {dataset: 'dataset-a'}}]};
state._testDatasetKey = 'dataset-a';
state.sourceSnapshot = {datasetKey: state._testDatasetKey};
state.derived.datasetCapabilities = {alias: 'dataset-a', num_classes: 10};
state.derived.compatibility = {status: 'compatible'};
state.derived.generatedConfig = {old: true};
state.derived.preflight = {old: true};
state.derived.runPlan = {old: true};
state.derived.command = {old: true};
const before = state.revision;
// A parameter-only edit must invalidate execution artifacts without throwing
// away capabilities that still describe the same selected source.
state.recipe.steps[0].params.unrelated = 1;
invalidateDerivedState('test parameter change');
assert.equal(state.revision, before + 1);
assert.equal(state.validated, false);
assert.equal(state.validatedRevision, -1);
assert.equal(state.derived.datasetCapabilities.alias, 'dataset-a');
assert.equal(state.derived.datasetCapabilitiesRevision, state.revision);
assert.equal(state.derived.compatibility, null);
assert.equal(state.derived.generatedConfig, null);
assert.equal(state.derived.preflight, null);
assert.equal(state.derived.runPlan, null);
assert.equal(state.derived.command, null);
assert.equal(state.derived.invalidatedBy, 'test parameter change');
// A source change invalidates capabilities as well as every downstream result.
state.recipe.steps[0].params.dataset = 'dataset-b';
state._testDatasetKey = 'dataset-b';
invalidateDerivedState('test dataset change');
assert.equal(state.derived.datasetCapabilities, null);
assert.equal(state.derived.datasetCapabilitiesRevision, -1);
const stale = staleStateError(before, '测试检查');
assert.equal(stale.guidanceCode, 'stale-state');
assert.equal(stale.payload.code, 'stale-state');
assert.throws(() => assertCurrentRevision(before, '测试检查'), /过期结果/);

// A run cannot consume a plan whose revision is not the current validated
// revision, even if a stale object is still present in memory.
assert.throws(() => assertCurrentRunState(state.revision), /过期结果/);
state.validated = true;
state.validatedRevision = state.revision;
for (const name of ['generatedConfig', 'compatibility', 'preflight', 'runPlan']) {
  state.derived[name] = {status: 'current'};
  state.derived[name + 'Revision'] = state.revision;
}
assert.doesNotThrow(() => assertCurrentRunState(state.revision));
`, context);
""",
                str(script),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js required for browser logic tests")
    def test_old_run_generation_cannot_mutate_current_run(self) -> None:
        script = Path(__file__).resolve().parents[1] / "web" / "scratch.js"
        result = subprocess.run(
            [
                shutil.which("node"),
                "-e",
                r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const context = vm.createContext({location: {pathname: '/'}, assert});
vm.runInContext(source.slice(0, source.indexOf('const UI_CATEGORIES')) + `
state.jobId = 'job-a';
state.activeRunGeneration = 7;
assert.equal(isCurrentRun('job-a', 7), true);
state.activeRunGeneration = 8;
assert.equal(isCurrentRun('job-a', 7), false);
assert.equal(isCurrentRun('job-a', 8), true);
state.jobId = 'job-b';
assert.equal(isCurrentRun('job-a', 8), false);
assert.equal(isCurrentRun('job-b', 8), true);
`, context);
""",
                str(script),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_custom_dataset_path_preflight_rejects_missing_and_empty_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty"
            empty.mkdir()
            missing = Path(directory) / "missing"
            self.assertEqual(dataset_preflight_payload({"dataset": "local", "path": str(missing)})["code"], "dataset-path-not-found")
            self.assertEqual(dataset_preflight_payload({"dataset": "local", "path": str(empty)})["code"], "dataset-path-empty")
            for mode in ("custom", "custom_path", "local", "path", "folder"):
                self.assertEqual(
                    dataset_preflight_payload({"dataset": "local", "source_mode": mode})["code"],
                    "missing-dataset-path",
                )

    def test_custom_path_mode_is_accepted_by_load_dataset_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "dataset.marker").write_text("fixture", encoding="utf-8")
            recipe = {
                "schema_version": 1,
                "name": "custom-path",
                "steps": [{
                    "block": "load_dataset",
                    "params": {"dataset": "local", "source_mode": "custom_path", "path": str(source)},
                }],
            }
            normalized = validate_recipe(recipe)
            self.assertEqual(normalized["steps"][0]["params"]["source_mode"], "custom_path")
            self.assertEqual(_dataset_preflight(normalized)["code"], "dataset-inspection-failed")

    def test_server_preflight_finds_nested_load_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "dataset.marker").write_text("fixture", encoding="utf-8")
            recipe = {"steps": [{"block": "epoch_loop", "steps": [{"block": "load_dataset", "params": {"dataset": "local", "path": str(source)}}]}]}
            result = _dataset_preflight(recipe)
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["code"], "dataset-inspection-failed")

    def test_unknown_registered_capabilities_are_not_inferred_from_adapter_name(self) -> None:
        for adapter in ("custom", "customn"):
            facts = _registered_fact({"alias": "data", "adapter": adapter})
            for key in ("has_clean_target", "has_noisy_target", "has_sample_index"):
                self.assertIsNone(facts[key])

    def test_trusted_subset_does_not_require_regular_validation(self) -> None:
        recipe = {"steps": [
            {"block": "load_dataset", "params": {"dataset": "synthetic", "options": {"num_clean": 5}}},
            {"block": "create_dataset_split", "params": {"validation_size": 0}},
            {"block": "assign_data_roles", "params": {"roles": ["train", "trusted_validation", "test"]}},
        ]}
        self.assertTrue(_dataset_preflight(recipe)["ok"])
        recipe["steps"][0]["params"]["options"]["num_clean"] = 0
        self.assertEqual(_dataset_preflight(recipe)["code"], "trusted-size-missing")

    def test_current_source_options_determine_model_class_compatibility(self) -> None:
        recipe = {"steps": [
            {"block": "load_dataset", "params": {"dataset": "synthetic", "options": {"classes": 3}}},
            {"block": "create_model", "params": {"num_classes": 3}},
        ]}
        result = _dataset_preflight(recipe)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["capabilities"]["num_classes"], 3)
        recipe["steps"][1]["params"]["num_classes"] = 2
        self.assertEqual(_dataset_preflight(recipe)["code"], "dataset-model-class-mismatch")

    def test_custom_source_is_inspected_with_native_loader(self) -> None:
        import torch
        from lnl_toolbox.scratch.data_runtime import ScratchSample, ScratchSplit
        train = ScratchSplit("custom", "train", (ScratchSample(torch.zeros(2), 0, 0, None),), 2)
        test = ScratchSplit("custom", "test", (ScratchSample(torch.zeros(2), 0, 0, 0),), 2)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "marker").write_text("not sufficient evidence")
            recipe = {"steps": [
                {"block": "load_dataset", "params": {"dataset": "custom", "source_mode": "custom_path", "path": directory}},
                {"block": "assign_data_roles", "params": {"roles": ["train", "trusted_validation"]}},
            ]}
            with patch("lnl_toolbox.scratch.blocks.data.load_sources", return_value=({"data": {}}, train, None, test)) as loader:
                result = _dataset_preflight(recipe)
            self.assertEqual(result["code"], "dataset-clean-target-missing")
            self.assertFalse(loader.call_args.args[0]["data"]["download"])
            recipe["steps"][1] = {"block": "select_label_source", "params": {"test": "clean"}}
            recipe["steps"].append({"block": "assign_data_roles", "params": {"roles": ["train", "test"]}})
            with patch("lnl_toolbox.scratch.blocks.data.load_sources", return_value=({"data": {}}, train, None, test)):
                self.assertTrue(_dataset_preflight(recipe)["ok"])

    @unittest.skipUnless(shutil.which("node"), "Node.js required")
    def test_run_snapshot_cancellation_and_metadata_behaviour(self) -> None:
        script = Path(__file__).resolve().parents[1] / "web" / "scratch.js"
        result = subprocess.run([shutil.which("node"), "-e", r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const context = vm.createContext({location: {pathname: '/'}, assert, document: {getElementById: () => null}});
vm.runInContext(source.slice(0, source.indexOf('const UI_CATEGORIES')), context);
vm.runInContext(source.slice(source.indexOf('async function validateCurrentRecipe('), source.indexOf("if ($('check'))")), context);
vm.runInContext(source.slice(source.indexOf('async function startRun('), source.indexOf("if ($('run'))")), context);
vm.runInContext(`
function renderRuntimeLimits() {}
function updateRunState() {}
function setInspectorTab() {}
function setResultState() {}
function renderRunProgress() {}
function renderRunResult() {}
function clearRunPolling() {}
function clearErrorPresentation() {}
function showError(error) { state.testError = error; }
function resetRunTracking() { state.runGeneration++; state.activeRunGeneration = null; state.running = false; }
function datasetReadiness() { throw new Error('old catalog must not override server'); }
function runPayload() { return {recipe: JSON.parse(JSON.stringify(state.recipe)), runtime_limits: {...state.runtimeLimits}}; }
`, context);
vm.runInContext(`(async () => {
  state.jobId = 'old'; state.running = true; state.activeRunGeneration = 8;
  const revision = state.revision;
  updateRecipeName('renamed');
  assert.equal(state.recipe.name, 'renamed');
  assert.equal(state.running, true); assert.equal(state.jobId, 'old');
  assert.equal(state.revision, revision); assert.equal(state.activeRunGeneration, 8);

  api = async () => { throw new Error('offline'); };
  await cancelJobSilently('old');
  await assert.rejects(ensurePreviousRunsStopped(), /old/);
  assert.equal(state.unresolvedCancellations.has('old'), true);
  const blockedCalls = [];
  api = async (path) => { blockedCalls.push(path); throw new Error('offline'); };
  state.running = false;
  await startRun('check');
  assert.equal(blockedCalls.includes('/api/run'), false);
  assert.equal(blockedCalls.includes('/api/validate'), false);
  assert.ok(state.testError);
  delete state.testError;
  api = async () => ({status: 'running'});
  await assert.rejects(ensurePreviousRunsStopped());
  api = async () => ({status: 'cancelled'});
  await ensurePreviousRunsStopped();
  assert.equal(state.unresolvedCancellations.size, 0);

  state.running = false; state.jobId = null;
  state.runtimeLimits = {fixture: true, max_batches: 2};
  state.datasets = [{alias: 'data', status: 'unavailable'}];
  state.recipe = {name: 'current', steps: [], settings: {rate: 0.3}};
  const calls = [];
  api = async (path, options) => {
    const body = JSON.parse(options.body); calls.push({path, body});
    if (path === '/api/validate') {
      state.runtimeLimits.max_batches = 99;
      return {recipe: {...body.recipe, normalized: true}, dataset_preflight: {ok: true, capabilities: {num_classes: 7}}};
    }
    return {ok: true};
  };
  await startRun('check');
  assert.equal(state.testError, undefined);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].path, '/api/validate'); assert.equal(calls[1].path, '/api/run');
  assert.deepEqual(calls[0].body.runtime_limits, calls[1].body.runtime_limits);
  assert.equal(calls[1].body.runtime_limits.max_batches, 2);
  assert.equal(calls[1].body.recipe.normalized, true);
  assert.equal(state.derived.datasetCapabilities.num_classes, 7);

  state.running = false; calls.length = 0;
  api = async (path, options) => { calls.push(path); state.revision++; return {recipe: {}, dataset_preflight: {ok: true}}; };
  await startRun('check');
  assert.deepEqual(calls, ['/api/validate']);
})()`, context).catch(error => { console.error(error); process.exitCode = 1; });
""", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_server_preflight_rejects_unknown_registered_dataset(self) -> None:
        result = _dataset_preflight({"steps": [{"block": "load_dataset", "params": {"dataset": "not-a-scratch-adapter"}}]})
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "dataset-unavailable")

    def test_server_preflight_rejects_validation_role_without_split(self) -> None:
        result = _dataset_preflight({
            "steps": [
                {"block": "load_dataset", "params": {"dataset": "cifar10"}},
                {"block": "create_dataset_split", "params": {"validation_size": 0}},
                {"block": "assign_data_roles", "params": {"roles": ["train", "noisy_validation", "test"]}},
            ]
        })
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "validation-role-without-split")

    def test_server_preflight_rejects_missing_external_noise_source(self) -> None:
        result = _dataset_preflight({
            "steps": [
                {"block": "load_dataset", "params": {"dataset": "cifar10"}},
                {"block": "apply_noise", "params": {"name": "external"}},
            ]
        })
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "missing-external-source")

    @unittest.skipUnless(shutil.which("node"), "Node.js required for browser logic tests")
    def test_stale_async_requests_are_rejected_by_revision_and_source(self) -> None:
        script = Path(__file__).resolve().parents[1] / "web" / "scratch.js"
        result = subprocess.run(
            [
                shutil.which("node"),
                "-e",
                r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const context = vm.createContext({location: {pathname: '/'}, assert});
vm.runInContext(source.slice(0, source.indexOf('const UI_CATEGORIES')), context);
vm.runInContext(`function datasetSourceKey(){ return state._testDatasetKey || null; }`, context);
vm.runInContext(`
state._testDatasetKey = 'dataset-a';
const requestRevision = state.revision;
const requestSourceKey = datasetSourceKey();
assert.equal(isCurrentRequest(requestRevision, requestSourceKey), true);

// A source switch rejects a late response even before the revision changes.
state._testDatasetKey = 'dataset-b';
assert.equal(isCurrentRequest(requestRevision, requestSourceKey), false);
assert.throws(() => assertCurrentRequest(requestRevision, requestSourceKey, '数据检查'), /过期结果/);

// A parameter/recipe edit changes revision and rejects the same old request.
state._testDatasetKey = 'dataset-a';
state.revision += 1;
assert.equal(isCurrentRequest(requestRevision, requestSourceKey), false);
assert.throws(() => assertCurrentRequest(requestRevision, requestSourceKey, '检查'), /过期结果/);
`, context);
""",
                str(script),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js required for browser logic tests")
    def test_dataset_source_identity_includes_effective_options(self) -> None:
        script = Path(__file__).resolve().parents[1] / "web" / "scratch.js"
        result = subprocess.run(
            [
                shutil.which("node"),
                "-e",
                r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const context = vm.createContext({location: {pathname: '/'}, assert});
const prefixEnd = source.indexOf('const UI_CATEGORIES');
vm.runInContext(source.slice(0, prefixEnd), context);
vm.runInContext(`function datasetSourceEntry(){ return {step: {params: state._testParams}}; }`, context);
const start = source.indexOf('function stableSourceValue');
const end = source.indexOf('function datasetIsUsable');
vm.runInContext(source.slice(start, end), context);
vm.runInContext(`
state._testParams = {dataset: 'local', source_mode: 'custom_path', path: 'data', options: {adapter: 'cifar10', classes: 10}};
const base = datasetSourceKey();
state._testParams.options.classes = 2;
assert.notEqual(datasetSourceKey(), base);
state._testParams.options.classes = 10;
state._testParams.adapter = 'mnist';
assert.notEqual(datasetSourceKey(), base);
`, context);
""",
                str(script),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
