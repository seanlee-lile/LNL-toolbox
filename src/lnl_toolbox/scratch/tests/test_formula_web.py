import http.client
import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from lnl_toolbox.scratch.web.server import ScratchHandler
from lnl_toolbox.scratch.formula import unregister_formula


class FormulaWebApiTest(unittest.TestCase):
    def tearDown(self):
        try:
            unregister_formula("user/web_formula")
        except KeyError:
            pass

    def test_list_and_save_formula_api(self):
        with tempfile.TemporaryDirectory() as directory:
            old = os.environ.get("LNL_SCRATCH_WORKSPACE")
            os.environ["LNL_SCRATCH_WORKSPACE"] = directory
            server = ThreadingHTTPServer(("127.0.0.1", 0), ScratchHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection(*server.server_address, timeout=5)
                connection.request("GET", "/api/formulas")
                response = connection.getresponse()
                payload = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertGreaterEqual(len(payload), 5)
                formula = {
                    "id": "user/web_formula",
                    "name": "Web Formula",
                    "inputs": {"x": {}},
                    "steps": [{"id": "out", "block": "detach", "bindings": {"input": "x"}}],
                    "outputs": {"value": {"source": "out"}},
                }
                connection.request("POST", "/api/formulas", body=json.dumps({"formula": formula}), headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                saved = json.loads(response.read())
                self.assertEqual(response.status, 201)
                self.assertEqual(saved["formula"]["id"], "user/web_formula")
                connection.request("GET", "/api/blocks")
                response = connection.getresponse()
                blocks = json.loads(response.read())
                self.assertIn("formula__user__web_formula", {item["id"] for item in blocks})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                if old is None:
                    os.environ.pop("LNL_SCRATCH_WORKSPACE", None)
                else:
                    os.environ["LNL_SCRATCH_WORKSPACE"] = old
