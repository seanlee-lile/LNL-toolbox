"""Scratch UI revision, generation and job coordination regressions."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from lnl_toolbox.scratch.web.server import ScratchJob, _scratch_job_payload, scratch_job_list_payload


class ScratchWebStateGateTest(unittest.TestCase):

    def test_terminal_process_poll_releases_running_state_before_output_drain(self) -> None:
        """A child that has exited must not leave the Stop action stuck."""

        class ExitedProcess:
            pid = 4242

            def poll(self):
                return -9

        with tempfile.TemporaryDirectory() as root:
            job = ScratchJob(
                job_id="job-exited",
                recipe_name="test.yaml",
                output_dir=Path(root),
                command=["python", "-c", ""],
                process=ExitedProcess(),
                cancel_requested=True,
            )
            payload = _scratch_job_payload(job)

        self.assertFalse(payload["running"])
        self.assertEqual(payload["returncode"], -9)
        self.assertEqual(payload["status"], "cancelled")

    def test_job_list_payload_reconciles_each_process(self) -> None:
        class ExitedProcess:
            pid = 4343

            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as root:
            job = ScratchJob(
                job_id="job-listed",
                recipe_name="listed.yaml",
                output_dir=Path(root),
                command=["python", "-c", ""],
                process=ExitedProcess(),
            )
            with patch("lnl_toolbox.scratch.web.server.SCRATCH_JOBS", {job.job_id: job}):
                payload = scratch_job_list_payload()

        self.assertEqual(len(payload["jobs"]), 1)
        self.assertFalse(payload["jobs"][0]["running"])
        self.assertEqual(payload["jobs"][0]["status"], "completed")

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

    @unittest.skipUnless(shutil.which("node"), "Node.js required for browser logic tests")
    def test_poll_failure_keeps_job_owned_until_terminal_confirmation(self) -> None:
        script = Path(__file__).resolve().parents[1] / "web" / "scratch.js"
        result = subprocess.run([shutil.which("node"), "-e", r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const timers = [];
const schedule = fn => { timers.push(fn); return timers.length; };
const context = vm.createContext({location: {pathname: '/'}, assert, process,
  document: {getElementById: () => null},
  setTimeout: schedule, clearTimeout: () => {},
  window: {setTimeout: schedule, clearTimeout: () => {}}});
vm.runInContext(source.slice(0, source.indexOf('const UI_CATEGORIES')), context);
for (const [start, end] of [
  ['function clearRunPolling(', 'function renderRunProgress('],
  ['function renderRunJob(', 'async function stopRun('],
  ['async function validateCurrentRecipe(', "if ($('check'))"],
  ['async function startRun(', "if ($('run'))"],
]) vm.runInContext(source.slice(source.indexOf(start), source.indexOf(end)), context);
vm.runInContext(`
function updateRunState() {}
function renderRuntimeLimits() {}
function setInspectorTab() {}
function setResultState() {}
function renderRunProgress() {}
function renderRunResult() {}
function clearErrorPresentation() {}
function showError(error) { state.testError = error; }
function runPayload() { return {recipe: state.recipe, runtime_limits: {fixture: true}}; }
`, context);
vm.runInContext(`(async () => {
  state.jobId = 'A'; state.running = true;
  state.activeRunGeneration = state.runGeneration;
  state.activeRunRevision = state.revision;
  const generation = state.activeRunGeneration, calls = [];
  api = async path => { calls.push(path); throw new Error('offline'); };
  await pollRunJob('A', generation);
  assert.equal(state.jobId, 'A'); assert.equal(state.running, true);
  assert.notEqual(state.runPollTimer, null);
  await startRun();
  assert.deepEqual(calls, ['/jobs/A']); // no second process, nor cancellation by Run

  // Defence in depth: an idle UI must not silently discard a still-owned job.
  state.running = false;
  await startRun();
  assert.equal(state.unresolvedCancellations.has('A'), true);
  assert.equal(calls.includes('/api/run'), false);
  api = async path => { calls.push(path); return {status: 'running'}; };
  await startRun();
  assert.equal(calls.includes('/api/run'), false);

  // Only the server's terminal acknowledgement releases A before B starts.
  api = async (path, options) => {
    calls.push(path);
    if (path.endsWith('/cancel')) return {status: 'cancelled'};
    if (path === '/api/validate') return {recipe: JSON.parse(options.body).recipe, dataset_preflight: {ok: true}};
    return {ok: true};
  };
  await startRun();
  assert.equal(state.unresolvedCancellations.size, 0);
  assert.equal(calls.filter(path => path === '/api/run').length, 1);
  assert.ok(calls.lastIndexOf('/jobs/A/cancel') < calls.indexOf('/api/run'));
  const count = calls.length;
  await pollRunJob('A', generation); // stale retry cannot touch the replacement
  assert.equal(calls.length, count);
})().then(() => process.exit(0)).catch(error => { console.error(error); process.exitCode = 1; });`, context);
""", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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
