import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock
import zipfile
from urllib import request, error

from web import external_resources as resources


class ExternalResourcesTest(unittest.TestCase):
    def setUp(self):
        resources._JOBS.clear()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def item(self, payload=b"weights", member=None):
        return dict(id="fixture", title="fixture", path=str(self.root / "weights.pt"),
                    url="https://example.test/weights", checksum=hashlib.sha256(payload).hexdigest(),
                    archive_member=member, downloadable=True, ready=False)

    def test_only_actual_configured_dependencies_are_listed(self):
        cal = {"method": "cal", "noise": {"name": "external_torch", "path": "data/cal/IDN_0.2_C10.pt"}}
        items = resources.resources(cal, self.root)
        self.assertEqual([item["id"] for item in items], ["cal-idn20"])
        self.assertFalse(items[0]["ready"])
        self.assertEqual(items[0]["path"], str(self.root / "data/cal/IDN_0.2_C10.pt"))
        cal["noise"] = {"name": "symmetric", "rate": 0.2}
        self.assertEqual(resources.resources(cal, self.root), [])
        self.assertEqual(resources.resources({"pretraining_stage": {"mode": "train"}}, self.root), [])
        self.assertEqual(resources.resources({"trusted_validation": {"source": "official_generated"}}, self.root), [])

    def test_dld_uses_the_same_cache_directory_as_training(self):
        config = {"dld": {"feature_extractor": {"source": "external_checkpoint", "external": {"adapter": "torchvision_resnet34_imagenet1k_v1"}}}}
        with mock.patch("torch.hub.get_dir", return_value=str(self.root)):
            item = resources.resources(config, self.root)[0]
        self.assertEqual(item["path"], str(self.root / "checkpoints/resnet34-b627a593.pth"))
        self.assertFalse(item["ready"])

    def test_disk_state_wins_over_old_download_status(self):
        config = {"method": "cal", "noise": {"name": "external_torch", "path": "data/cal/IDN_0.2_C10.pt"}}
        item = resources.resources(config, self.root)[0]
        resources._JOBS[item["path"]] = {"status": "ready"}
        self.assertEqual(resources.resources(config, self.root)[0]["status"], "missing")
        resources._JOBS[item["path"]] = {"status": "failed", "error": "old failure"}
        with mock.patch.object(resources, "_valid", return_value=True):
            fixed = resources.resources(config, self.root)[0]
        self.assertTrue(fixed["ready"])
        self.assertEqual(fixed["error"], "")

    def test_success_installs_verified_file_and_cleans_pending_file(self):
        item = self.item()
        with mock.patch.object(resources, "urlopen", return_value=io.BytesIO(b"weights")), mock.patch.object(resources, "_context"):
            resources._download(item)
        self.assertEqual(Path(item["path"]).read_bytes(), b"weights")
        self.assertEqual(resources._JOBS[item["path"]]["status"], "ready")
        self.assertEqual(list(self.root.glob("*.pending")), [])

    def test_failed_download_does_not_create_a_ready_file(self):
        item = self.item()
        with mock.patch.object(resources, "urlopen", side_effect=OSError("network unavailable")), mock.patch.object(resources, "_context"):
            resources._download(item)
        self.assertFalse(Path(item["path"]).exists())
        self.assertEqual(resources._JOBS[item["path"]]["status"], "failed")
        self.assertIn("network unavailable", resources._JOBS[item["path"]]["error"])
        self.assertEqual(list(self.root.glob("*.pending")), [])

    def test_bad_checksum_is_rejected(self):
        item = self.item()
        with mock.patch.object(resources, "urlopen", return_value=io.BytesIO(b"bad data")), mock.patch.object(resources, "_context"):
            resources._download(item)
        self.assertFalse(Path(item["path"]).exists())
        self.assertEqual(resources._JOBS[item["path"]]["status"], "failed")

    def test_existing_file_is_never_overwritten(self):
        item = self.item()
        Path(item["path"]).write_bytes(b"user file")
        with mock.patch.object(resources, "urlopen") as download:
            resources._download(item)
        download.assert_not_called()
        self.assertEqual(Path(item["path"]).read_bytes(), b"user file")

    def test_archive_extracts_only_the_declared_member(self):
        item = self.item(member="heart.dat")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("heart.dat", b"weights")
            bundle.writestr("../outside", b"unsafe")
        with mock.patch.object(resources, "urlopen", return_value=io.BytesIO(archive.getvalue())), mock.patch.object(resources, "_context"):
            resources._download(item)
        self.assertEqual(Path(item["path"]).read_bytes(), b"weights")
        self.assertEqual([path.name for path in self.root.iterdir()], ["weights.pt"])

    def test_repeated_clicks_start_only_one_download(self):
        item = self.item()
        with mock.patch.object(resources.threading, "Thread") as thread:
            resources.start_download(item)
            resources.start_download(item)
        thread.assert_called_once()

    def test_api_rejects_config_paths_outside_project(self):
        from web import command_console
        with mock.patch.object(command_console, "_web_recipe_config", return_value={}):
            with self.assertRaisesRegex(ValueError, "项目目录"):
                command_console._external_resource_payload("fixture", str(self.root / "outside.yaml"))

    def test_quick_start_checks_adapted_config_instead_of_template(self):
        from web import command_console, quick_start_api
        formal = {"method": "cal", "noise": {"name": "external_torch", "path": "data/cal/IDN_0.2_C10.pt"}}
        adapted = {"method": "cal", "noise": {"name": "symmetric", "rate": 0.2}}
        with mock.patch.object(command_console, "_web_recipe_config", return_value=formal), mock.patch.object(
            quick_start_api.SERVICE, "method_options", return_value=[SimpleNamespace(paper_id="cal", candidate_config=adapted)]
        ):
            payload = command_console._external_resource_payload("fixture", dataset="mini", paper_id="cal",
                                                                 noise=json.dumps({"kind": "synthetic", "key": "symmetric", "rate": 0.2}))
        self.assertEqual(payload["resources"], [])

    def test_http_download_failure_is_reported_by_status_endpoint(self):
        from web import command_console
        config = {"data": {"name": "uci_statlog_heart", "path": "data/heart.dat"}}
        with mock.patch.object(command_console, "ROOT", self.root), mock.patch.object(command_console, "_web_recipe_config", return_value=config):
            server = command_console.ThreadingHTTPServer(("127.0.0.1", 0), command_console.ConsoleHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = "http://127.0.0.1:" + str(server.server_address[1])
            try:
                with request.urlopen(base + "/api/external-resources?recipe=fixture") as response:
                    item = json.load(response)["resources"][0]
                self.assertFalse(item["ready"])
                body = json.dumps({"recipe": "fixture", "resource": "uci-heart"}).encode()
                with mock.patch.object(resources, "urlopen", side_effect=OSError("offline")), mock.patch.object(resources, "_context"):
                    with request.urlopen(request.Request(base + "/api/external-resources/download", data=body, headers={"Content-Type": "application/json"})) as response:
                        self.assertEqual(response.status, 202)
                    for _ in range(100):
                        with request.urlopen(base + "/api/external-resources?recipe=fixture") as response:
                            item = json.load(response)["resources"][0]
                        if item["status"] == "failed":
                            break
                        time.sleep(0.01)
                self.assertEqual(item["status"], "failed")
                self.assertIn("offline", item["error"])
                self.assertFalse(item["ready"])
                bad = json.dumps({"recipe": "fixture", "resource": "arbitrary-url"}).encode()
                with self.assertRaises(error.HTTPError) as caught:
                    request.urlopen(request.Request(base + "/api/external-resources/download", data=bad))
                self.assertEqual(caught.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_ui_download_polls_then_refreshes_plan_and_guides_failure(self):
        if not shutil.which("node"):
            self.skipTest("Node.js unavailable")
        script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
let status = 'missing', clicks = [], completed = 0, timers = [];
const item = () => ({id:'dld-resnet34',title:'ResNet',path:'/cache/weights.pth',url:'https://download.pytorch.org/models/resnet34-b627a593.pth',note:'weights',ready:status === 'ready',status,error:status === 'failed' ? 'offline' : '',downloadable:true});
const download = {dataset:{externalDownload:'dld-resnet34'}}, paper = {};
const node = {isConnected:true,innerHTML:'',appendChild(){},replaceChildren(){},
  querySelector(){return {addEventListener(){}}},
  querySelectorAll(selector){return selector === '[data-external-download]' ? [download] : [paper]}};
const context = {window:{},URLSearchParams,clearTimeout(){},setTimeout(fn){timers.push(fn);return timers.length},
  document:{createElement(){return {}}},
  fetch:async (url,options) => {clicks.push([url,options]); if (options) status='downloading'; return {ok:true,json:async()=>options?{started:true}:{resources:[item()]}}}};
vm.runInNewContext(fs.readFileSync('web/assets/quick_start.js','utf8'),context);
const tick = () => new Promise(resolve=>setImmediate(resolve));
(async()=>{
  context.window.paperExternalResources.mount(node,{recipe:'dld'},()=>{},()=>completed++);
  await tick(); assert(node.innerHTML.includes('尚未准备')); assert(node.innerHTML.includes('/cache/weights.pth'));
  await download.onclick(); assert(node.innerHTML.includes('正在下载')); assert.strictEqual(completed,0);
  status='ready'; await timers.pop()(); assert(node.innerHTML.includes('已就绪')); assert.strictEqual(completed,1);
  status='failed'; context.window.paperExternalResources.mount(node,{recipe:'dld'},()=>{}); await tick();
  assert(node.innerHTML.includes('offline')); assert(node.innerHTML.includes('前往论文栏')); assert(node.innerHTML.includes('自行下载'));
  assert(clicks.some(([url,options])=>url==='/api/external-resources/download' && JSON.parse(options.body).resource==='dld-resnet34'));
})().catch(error=>{console.error(error);process.exitCode=1});
'''
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
