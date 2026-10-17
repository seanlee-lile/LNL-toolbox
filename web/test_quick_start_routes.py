from __future__ import annotations

import http.client
import json
from threading import Thread
import unittest
from unittest.mock import patch

from web import command_console
from web.command_console import ConsoleHandler, ThreadingHTTPServer


class QuickStartRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ConsoleHandler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict | str]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        encoded = None if body is None else json.dumps(body)
        headers = {} if encoded is None else {"Content-Type": "application/json"}
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        content = response.read().decode("utf-8")
        connection.close()
        try:
            return response.status, json.loads(content)
        except json.JSONDecodeError:
            return response.status, content

    def test_assets_and_all_quick_start_routes_are_wired(self) -> None:
        for path in ("/assets/quick_start.js", "/assets/quick_start.css"):
            status, _ = self._request("GET", path)
            self.assertEqual(status, 200)
        with patch("web.quick_start_api.noise_options_payload", return_value={"options": []}), \
             patch("web.quick_start_api.probe_payload", return_value={"status": "detected"}), \
             patch("web.quick_start_api.register_payload", return_value={"kind": "dataset"}), \
             patch("web.quick_start_api.method_options_payload", return_value={"methods": []}), \
             patch("web.quick_start_api.plan_payload", return_value={"status": "ready"}):
            status, _ = self._request("GET", "/api/quick-start/noises?dataset=fixture")
            self.assertNotEqual(status, 404)
            for path in (
                "/api/quick-start/probe",
                "/api/quick-start/register",
                "/api/quick-start/methods",
                "/api/quick-start/plan",
            ):
                status, _ = self._request("POST", path, {})
                self.assertNotEqual(status, 404)

    def test_quick_start_carousel_uses_existing_plan_and_execution_routes(self) -> None:
        script = (command_console.WEB_ROOT / "assets" / "quick_start.js").read_text(encoding="utf-8")
        for step in ("dataset", "noise", "methods", "plan", "review"):
            self.assertIn('id:"' + step + '"', script)
        for marker in ("renderQuickStartCarousel()", "async function reviewPlan()",
                       "async function executePlan()", "review.signature !== reviewSignature()"):
            self.assertIn(marker, script)
        self.assertNotIn("/api/quick-start/review-config", script)
        self.assertNotIn("/api/quick-start/training-config", script)


if __name__ == "__main__":
    unittest.main()
