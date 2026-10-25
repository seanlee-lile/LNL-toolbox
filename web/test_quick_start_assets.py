from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import unittest

from web.command_console import STATIC_ASSETS


ROOT = Path(__file__).resolve().parents[1]


class QuickStartAssetsTests(unittest.TestCase):
    def test_noise_inputs_and_pending_recipe_preserve_matrix_path(self) -> None:
        script = (ROOT / 'web/assets/quick_start.js').read_text(encoding='utf-8')
        for marker in ('qs-transition-matrix', 'qs-noise-path', 'Number.isFinite(value)',
                       'transition_matrix:matrix, path:labelPath',
                       'schemaValue(pending.schema, "noise.transition_matrix")',
                       'schemaValue(pending.schema, "noise.path")'):
            self.assertIn(marker, script)

    def test_stale_method_responses_cannot_continue_to_plan(self) -> None:
        script = (ROOT / 'web/assets/quick_start.js').read_text(encoding='utf-8')
        self.assertIn('if (!await loadMethods()) return;', script)
        self.assertIn('if (options === false || !state.selectedPaperId || state.error) return;', script)
        self.assertIn('if (current) recheckSelectedPaper({navigate:false});', script)
        self.assertIn('if (!current()) return false; state.methods', script)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for actual async JavaScript runtime coverage')
    def test_method_generation_gate_at_javascript_runtime(self) -> None:
        script = (ROOT / 'web/assets/quick_start.js').read_text(encoding='utf-8')
        function = script.split('  function loadMethods() {', 1)[1].split('  function planRequestSignature()', 1)[0]
        program = """
const assert = require('node:assert/strict');
let methodsToken=0, rendered=0, calls=0, pending=[];
const state={dataset:{alias:'a'},noiseSelection:{key:'clean'},methods:[]};
const render=()=>rendered++;
const post=()=>new Promise((resolve,reject)=>pending.push({resolve,reject}));
function loadMethods() { FUNCTION
(async()=>{
const stale=loadMethods();
state.dataset={alias:'b'};
const fresh=loadMethods();
pending[1].resolve({methods:[{paper_id:'new'}]});
assert.equal(await fresh,true);
pending[0].resolve({methods:[{paper_id:'old'}]});
if(await stale) calls++;
assert.equal(calls,0);assert.equal(state.methods[0].paper_id,'new');assert.equal(rendered,1);
const failed=loadMethods();pending[2].reject(Error('expected'));assert.equal(await failed,false);
state.dataset=null;assert.equal(await loadMethods(),false);
})().catch(error=>{console.error(error);process.exitCode=1;});
""".replace('FUNCTION', function)
        result = subprocess.run([shutil.which('node'), '-'], input=program, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js required for actual JavaScript syntax/runtime coverage")
    def test_integrated_scripts_parse_and_paper_readiness_gates_at_runtime(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        node = shutil.which("node")
        scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, re.S)
        for source in scripts + [(ROOT / "web" / "assets" / "quick_start.js").read_text(encoding="utf-8")]:
            result = subprocess.run([node, "--check"], input=source, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        function = re.search(r"    function paperCanStart\(config\) \{.*?\n    \}", html, re.S)
        self.assertIsNotNone(function)
        checks = """
const assert = require('node:assert/strict');
assert.equal(paperCanStart(null), false);
for (const prerequisite_status of ['needs_input', 'invalid']) {
  assert.equal(paperCanStart({prerequisite_status}), false);
}
assert.equal(paperCanStart({prerequisite_status:'ready', availability:'conditional'}), true);
assert.equal(paperCanStart({preparation:{artifact_ready:false}}), false);
assert.equal(paperCanStart({preparation:{artifact_ready:true}}), true);
"""
        result = subprocess.run([node, "-"], input=function.group(0) + checks, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_assets_are_served_from_explicit_allowlist(self) -> None:
        self.assertEqual(set(STATIC_ASSETS), {"/assets/quick_start.js", "/assets/quick_start.css", "/assets/run_output.js"})
        self.assertTrue(all(path.is_file() for path, _content_type in STATIC_ASSETS.values()))

    def test_parent_traversal_is_not_an_asset(self) -> None:
        self.assertNotIn("/assets/../index.html", STATIC_ASSETS)

    def test_index_loads_external_quick_start_resources(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn('/assets/quick_start.css', html)
        self.assertIn('/assets/quick_start.js', html)
        self.assertIn('/assets/run_output.js', html)
        self.assertIn('window.quickStartController', html)
        self.assertIn('id="training-output"', html)

    def test_existing_modules_remain(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        for module in ('quickstart', 'yaml', 'data', 'sweep', 'results', 'papers'):
            self.assertIn('id: "' + module + '"', html)

    def test_registered_dataset_and_review_guard_are_wired(self) -> None:
        script = (ROOT / "web" / "assets" / "quick_start.js").read_text(encoding="utf-8")
        for marker in ("esc(item.location)", 'payload.status === "already_registered"',
                       "qs-required-input", "reviewSignature()", 'review?.phase !== "ready"',
                       "acknowledge_paper_impact:state.acknowledgePaperImpact === true",
                       '"/api/configs"', "plan.dry_run_command", 'job.output_complete !== false'):
            self.assertIn(marker, script)
        self.assertNotIn("/api/external-resources", script)
        self.assertNotIn("/api/quick-start/review-config", script)
        self.assertNotIn("/api/quick-start/training-config", script)
        self.assertNotIn("training_pool_size", script)


if __name__ == "__main__":
    unittest.main()
