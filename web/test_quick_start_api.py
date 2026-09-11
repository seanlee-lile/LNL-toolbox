from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from web import quick_start_api
from lnl_toolbox.data.probe import DatasetProbeResult, ProbeCandidate
from lnl_toolbox.quickstart.models import QuickStartDatasetSummary, QuickStartMethodOption


class _FakeService:
    def probe(self, path: str):
        return DatasetProbeResult(path, "detected", (
            ProbeCandidate("cifar10", "high", "fixture", {"root": path}),
        ))

    def register_and_inspect(self, path: str, *, selected_adapter=None):
        del selected_adapter
        return QuickStartDatasetSummary("local", "cifar10", path, "CIFAR-10", 10, 2, None, 1, "clean", "unknown", "available")

    def noise_options(self, alias: str):
        return {"dataset_state": "clean", "options": [{"key": "clean"}], "alias": alias}

    def method_options(self, dataset, noise):
        del dataset, noise
        return ()

    def build_plan(self, **kwargs):
        del kwargs
        raise ValueError("fixture plan error")


class QuickStartApiTests(unittest.TestCase):
    def test_probe_shape(self) -> None:
        with patch.object(quick_start_api, "SERVICE", _FakeService()):
            payload = quick_start_api.probe_payload({"path": "C:/data"})
        self.assertEqual(payload["status"], "detected")
        self.assertEqual(payload["candidates"][0]["adapter"], "cifar10")

    def test_register_shape(self) -> None:
        with patch.object(quick_start_api, "SERVICE", _FakeService()):
            payload = quick_start_api.register_payload({"path": "C:/data"})
        self.assertEqual(payload["kind"], "dataset")
        self.assertEqual(payload["dataset"]["alias"], "local")

    def test_noise_shape(self) -> None:
        with patch.object(quick_start_api, "SERVICE", _FakeService()):
            self.assertEqual(quick_start_api.noise_options_payload("local")["alias"], "local")

    def test_malformed_request(self) -> None:
        with self.assertRaises(ValueError):
            quick_start_api.probe_payload({})

    def test_service_error_is_not_hidden(self) -> None:
        with patch.object(quick_start_api, "SERVICE", _FakeService()):
            with self.assertRaises(ValueError):
                quick_start_api.plan_payload({"dataset": "local", "paper_id": "pdl", "noise": {"key": "clean"}})

    def test_method_input_paths_are_json_ready(self) -> None:
        option = QuickStartMethodOption(
            "cdr", "CDR", "title", "summary", "venue", 2024,
            "needs_input", required_user_inputs=("noise_rate_prior",),
            required_input_paths=(("noise_rate_prior", (("parameter_update", "noise_rate"),)),),
        )
        self.assertEqual(
            option.to_dict()["required_input_paths"],
            [["noise_rate_prior", [["parameter_update", "noise_rate"]]]],
        )


if __name__ == "__main__":
    unittest.main()
