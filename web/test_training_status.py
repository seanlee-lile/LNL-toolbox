from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from web.training_status import TrainingContext, infer_training_context, training_snapshot


class TrainingStatusTest(unittest.TestCase):
    def test_step_budget_and_flushed_step_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yaml"
            config.write_text("trainer:\n  epochs: 180\n  max_steps: 80000\n", encoding="utf-8")
            context = infer_training_context(["lnl", "run", str(config), "--output-dir", str(root)], root)
            snapshot = training_snapshot(context, lines=['{"event":"epoch","epoch":2,"global_step":4000}'],
                                         running=True, returncode=None, cancel_requested=False)
            self.assertEqual((snapshot.total_steps, snapshot.completed_step, snapshot.total_epochs), (80000, 4000, None))

    def test_progress_moves_up_and_details_stay_below(self) -> None:
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node is required for the Web UI regression")
        script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
function element() { const classes = new Set(); return {hidden:false, textContent:'', value:0,
  classList:{toggle:(key,on)=>on ? classes.add(key) : classes.delete(key),
             add:key=>classes.add(key), remove:key=>classes.delete(key), contains:key=>classes.has(key)},
  removeAttribute(key) {if(key === 'value') delete this.value;}}; }
const nodes = Object.fromEntries(['training-output','training-summary','training-metrics','output',
  'training-simple','training-detail','qs-training-output','qs-training-summary',
  'qs-training-progress','qs-training-progress-label'].map(id=>[id,element()]));
const window = {};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {window,document:{getElementById:id=>nodes[id]}});
window.trainingOutput.render({training:{state:'running',completed_epoch:2,total_epochs:10,metrics:[{epoch:2}]}});
assert(!nodes.output.classList.contains('hidden'));
window.trainingOutput.toggle(false);
assert(!nodes.output.classList.contains('hidden'));
assert.equal(nodes['qs-training-progress'].value,20);
assert(nodes['training-summary'].classList.contains('hidden'));
assert(nodes['qs-training-summary'].textContent.includes('2/10'));
window.trainingOutput.toggle(true);
assert(!nodes.output.classList.contains('hidden'));
assert(!nodes['training-metrics'].classList.contains('hidden'));
window.trainingOutput.render({training:{state:'running',completed_step:4000,total_steps:80000,metrics:[]}});
assert.equal(nodes['qs-training-progress'].value,5);
assert(nodes['qs-training-progress-label'].textContent.includes('4000 / 80000 步'));
window.trainingOutput.render({training:{state:'running',metrics:[]}});
assert.equal(nodes['qs-training-progress'].value,undefined);
window.trainingOutput.render({training:{state:'completed',metrics:[]}});
assert.equal(nodes['qs-training-progress'].value,100);
nodes['qs-training-progress'] = element();
window.trainingOutput.refresh();
assert.equal(nodes['qs-training-progress'].value,100);
delete nodes['qs-training-output'];
window.trainingOutput.refresh();
assert(!nodes['training-summary'].classList.contains('hidden'));
'''
        result = subprocess.run([node, "-e", script, str(Path(__file__).parent / "assets" / "run_output.js")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _context(self, directory: Path) -> TrainingContext:
        return TrainingContext(directory, None, "run", 12)

    def test_regular_epoch_uses_flushed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "metrics.jsonl").write_text(json.dumps({"event": "epoch", "epoch": 2}) + "\n", encoding="utf-8")
            snapshot = training_snapshot(self._context(path), lines=[], running=True, returncode=None, cancel_requested=False)
        assert snapshot is not None
        self.assertEqual((snapshot.source, snapshot.stage, snapshot.completed_epoch), ("metrics", "main", 2))

    def test_ca2c_phase_is_used_as_display_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "metrics.jsonl").write_text(
                json.dumps({"event": "epoch", "epoch": 1, "phase": "warmup"}) + "\n",
                encoding="utf-8",
            )
            snapshot = training_snapshot(self._context(path), lines=[], running=True, returncode=None, cancel_requested=False)
        self.assertEqual(snapshot.stage, "warmup")

    def test_dividemix_warmup_then_main_uses_latest_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            rows = [{"event": "warmup", "epoch": 10}, {"event": "epoch", "epoch": 1}]
            (path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            snapshot = training_snapshot(self._context(path), lines=[], running=True, returncode=None, cancel_requested=False)
        assert snapshot is not None
        self.assertEqual((snapshot.stage, snapshot.completed_epoch), ("main", 1))

    def test_partial_json_line_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "metrics.jsonl").write_text('{"event":"warmup","epoch":1}\n{"event":', encoding="utf-8")
            snapshot = training_snapshot(self._context(path), lines=[], running=True, returncode=None, cancel_requested=False)
        assert snapshot is not None
        self.assertEqual(snapshot.completed_epoch, 1)

    def test_stdout_structured_event_is_a_fallback(self) -> None:
        snapshot = training_snapshot(
            TrainingContext(None, None, "run", 3),
            lines=['{"event":"epoch","epoch":2}'],
            running=False,
            returncode=0,
            cancel_requested=False,
        )
        assert snapshot is not None
        self.assertEqual((snapshot.source, snapshot.state, snapshot.completed_epoch), ("stdout", "completed", 2))

    def test_positional_config_and_explicit_output_are_inferred(self) -> None:
        context = infer_training_context(
            ["lnl", "run", "configs/experiment/cifar10_clean_smoke.yaml", "--output-dir", "artifacts/web"],
            Path.cwd(),
        )
        assert context is not None
        self.assertEqual(context.run_dir, Path.cwd() / "artifacts" / "web")
        self.assertEqual(context.total_epochs, 2)


if __name__ == "__main__":
    unittest.main()
