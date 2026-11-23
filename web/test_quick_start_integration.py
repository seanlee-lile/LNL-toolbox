from __future__ import annotations

import http.client
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

import numpy as np

from lnl_toolbox.data.contracts import (
    DataSpec,
    RawDatasetSplit,
    UnsupportedDatasetSplitError,
)
from lnl_toolbox.data.local_catalog import LocalDatasetCatalog
from lnl_toolbox.data.probe import DatasetProbeResult, ProbeCandidate
from lnl_toolbox.data.registry import DatasetRegistry
from lnl_toolbox.quickstart.service import QuickStartService
from lnl_toolbox.training.data_service import DataService
from web import quick_start_api
from web.command_console import ConsoleHandler, ThreadingHTTPServer


class _CifarFixtureAdapter:
    name = "cifar10"
    aliases = ()

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError(spec.root)

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del spec, seed
        if split == "validation":
            raise UnsupportedDatasetSplitError("split must be train or test")
        count = 40 if split == "train" else 20
        labels = np.arange(count, dtype=np.int64) % 10
        return RawDatasetSplit(
            np.zeros((count, 32, 32, 3), dtype=np.uint8), labels,
            np.arange(count, dtype=np.int64), self.name, split, 10,
            clean_targets=labels, source="quick-start-http-fixture",
        )


class QuickStartIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.dataset_root = root / "cifar"
        self.dataset_root.mkdir()
        data_service = DataService(
            registry=DatasetRegistry((_CifarFixtureAdapter(),)),
            catalog=LocalDatasetCatalog(root / "catalog.json"),
        )
        self.service = QuickStartService(data_service, artifact_root=root / "artifacts")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ConsoleHandler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.patch_service = patch.object(quick_start_api, "SERVICE", self.service)
        self.patch_probe = patch(
            "lnl_toolbox.quickstart.service.probe_dataset_path",
            return_value=DatasetProbeResult(
                str(self.dataset_root), "detected", (
                    ProbeCandidate("cifar10", "high", "HTTP fixture", {"root": str(self.dataset_root)}),
                ),
            ),
        )
        self.patch_service.start()
        self.patch_probe.start()

    def tearDown(self) -> None:
        self.patch_probe.stop()
        self.patch_service.stop()
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.temp.cleanup()

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict | str]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
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

    def test_http_flow_reaches_data_capabilities_and_plan(self) -> None:
        status, page = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("quickstart", page)
        status, _ = self._request("GET", "/assets/quick_start.js")
        self.assertEqual(status, 200)

        status, probe = self._request("POST", "/api/quick-start/probe", {"path": str(self.dataset_root)})
        self.assertEqual(status, 200)
        self.assertEqual(probe["status"], "detected")
        status, registered = self._request("POST", "/api/quick-start/register", {"path": str(self.dataset_root)})
        self.assertEqual(status, 200)
        alias = registered["dataset"]["alias"]

        status, noises = self._request("GET", "/api/quick-start/noises?dataset=" + alias)
        self.assertEqual(status, 200)
        self.assertIn("symmetric", {item["key"] for item in noises["options"]})
        noise = {"kind": "clean", "key": "clean", "seed": 43}
        status, methods = self._request("POST", "/api/quick-start/methods", {"dataset": alias, "noise": noise})
        self.assertEqual(status, 200)
        self.assertTrue(methods["methods"])
        self.assertTrue(any(item["status"] != "metadata_error" for item in methods["methods"]))

        ready = next(item for item in methods["methods"] if item["status"] == "ready")
        status, plan = self._request("POST", "/api/quick-start/plan", {
            "dataset": alias, "noise": noise, "paper_id": ready["paper_id"],
        })
        self.assertEqual(status, 200)
        self.assertEqual(plan["status"], "ready")
        self.assertTrue(plan["command"])
        import yaml

        with patch("lnl_toolbox.training.data_service.DEFAULT_DATA_SERVICE", self.service.data_service):
            status, reviewed = self._request("POST", "/api/quick-start/review-config", {
                "dataset_alias": alias, "noise": noise, "recipe": ready["recipe_id"],
                "patches": [],
            })
        self.assertEqual(status, 200, reviewed)
        effective = yaml.safe_load(reviewed["content"])
        self.assertEqual(effective["local_dataset"]["alias"], alias)
        self.assertEqual(Path(effective["data"]["root"]), self.dataset_root)
        self.assertEqual(effective["noise"]["name"], "clean")
        self.assertEqual(effective["seed"], 43)

        with patch.object(self.service, "accepted_training_pool_size", return_value=2700), \
                patch("lnl_toolbox.training.data_service.DEFAULT_DATA_SERVICE", self.service.data_service):
            status, schema = self._request("GET", "/api/config-schema?recipe=gce-cifar10&quickstart_dataset=" + alias)
            self.assertEqual(status, 200, schema)
            field = next(item for item in schema["fields"] if item["path"] == "data.validation_size")
            self.assertEqual(field["value"], 270)
            self.assertEqual(field["split_reference"], {"count": 5000, "total": 50000})
            for patches, expected in (([], 270), ([{"path": "data.validation_size", "value": 200}], 200)):
                status, reviewed = self._request("POST", "/api/quick-start/review-config", {
                    "dataset_alias": alias, "noise": noise, "recipe": "gce-cifar10", "patches": patches,
                })
                self.assertEqual(status, 200, reviewed)
                self.assertEqual(yaml.safe_load(reviewed["content"])["data"]["validation_size"], expected)

    def test_resnet_stem_is_resolved_after_dataset_overlay(self) -> None:
        import yaml

        status, registered = self._request("POST", "/api/quick-start/register", {"path": str(self.dataset_root)})
        self.assertEqual(status, 200, registered)
        alias = registered["dataset"]["alias"]
        service = self.service.data_service
        apply = service.apply
        selected = {}

        def overlay(config, dataset_alias):
            effective = apply(config, dataset_alias)
            effective["data"].update(selected)
            return effective

        with patch.object(service, "apply", side_effect=overlay), \
                patch("lnl_toolbox.training.data_service.DEFAULT_DATA_SERVICE", service), \
                patch("lnl_toolbox.training.data_service._transforms", side_effect=AssertionError("ran preprocessing")):
            for data, expected in (({"name": "clothing1m"}, "imagenet_standard"),
                                   ({"name": "clothing1m", "image_size": 64}, "standard"),
                                   ({"name": "cifar10"}, "standard")):
                selected = data
                status, schema = self._request("GET", "/api/config-schema?recipe=gce-cifar10&quickstart_dataset=" + alias)
                self.assertEqual(status, 200, schema)
                field = next(item for item in schema["fields"] if item["path"] == "model.variant")
                self.assertEqual(field["value"], expected)
                status, reviewed = self._request("POST", "/api/quick-start/review-config", {
                    "dataset_alias": alias, "noise": {"kind": "clean", "key": "clean", "seed": 1},
                    "recipe": "gce-cifar10", "patches": [],
                })
                self.assertEqual(status, 200, reviewed)
                self.assertEqual(yaml.safe_load(reviewed["content"])["model"]["variant"], expected)

            selected = {"name": "clothing1m"}
            status, reviewed = self._request("POST", "/api/quick-start/review-config", {
                "dataset_alias": alias, "noise": {"kind": "clean", "key": "clean", "seed": 1},
                "recipe": "gce-cifar10", "patches": [{"path": "model.weights", "value": "imagenet1k_v1"}],
            })
            self.assertEqual(status, 200, reviewed)
            model = yaml.safe_load(reviewed["content"])["model"]
            self.assertEqual((model["variant"], model["weights"]), ("imagenet_standard", "imagenet1k_v1"))

    def test_http_never_turns_a_needs_input_method_into_a_runnable_plan(self) -> None:
        status, registered = self._request(
            "POST", "/api/quick-start/register", {"path": str(self.dataset_root)}
        )
        self.assertEqual(status, 200)
        alias = registered["dataset"]["alias"]
        self.service.data_service.update_declarations(alias, {
            "noise_status": "noisy", "noise_origin": "native",
            "noise_rate": {"status": "unknown"},
        })
        noise = {"kind": "native", "key": "native"}
        status, methods = self._request(
            "POST", "/api/quick-start/methods", {"dataset": alias, "noise": noise}
        )
        self.assertEqual(status, 200)
        jocor = next(item for item in methods["methods"] if item["paper_id"] == "jocor")
        self.assertEqual(jocor["status"], "needs_input")
        self.assertIn("noise_rate_prior", jocor["required_user_inputs"])
        status, plan = self._request("POST", "/api/quick-start/plan", {
            "dataset": alias, "noise": noise, "paper_id": "jocor",
        })
        self.assertEqual(status, 200)
        self.assertEqual(plan["status"], "needs_input")
        self.assertFalse(plan["command"])


if __name__ == "__main__":
    unittest.main()
