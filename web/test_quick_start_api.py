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
    def test_real_plan_api_uses_registered_adapter_yaml_not_alias_or_default(self) -> None:
        from copy import deepcopy
        import json
        from types import SimpleNamespace
        from lnl_toolbox.catalog import load_yaml
        from lnl_toolbox.quickstart.service import QuickStartService
        from lnl_toolbox.training.compatibility import CompatibilityStatus

        # Check the real selection/plan/API path; only data access and preflight
        # are stubbed. No training is implied by this plan-contract test.
        profile_data = {"adapter": "cifar100", "num_classes": 100,
                        "sample_counts_by_split": {"train": 50000, "test": 10000}}
        def bind(config, alias):
            result = deepcopy(config)
            result.setdefault("data", {}).update(name="cifar100", root="/registered/data")
            return result
        service = QuickStartService(SimpleNamespace(apply=bind))
        report = SimpleNamespace(profile=SimpleNamespace(to_dict=lambda: profile_data))
        capabilities = SimpleNamespace(noise_rate=SimpleNamespace(value=None))
        compatible = SimpleNamespace(status=CompatibilityStatus.COMPATIBLE, reasons=(), required_user_inputs=())
        with tempfile.TemporaryDirectory() as directory:
            service.artifact_root = Path(directory)
            with patch.object(quick_start_api, "SERVICE", service), \
                 patch.object(service, "_accepted_report", return_value=report), \
                 patch.object(service, "_accepted_capabilities", return_value=capabilities), \
                 patch.object(service.experiment_service, "list_config_compatibility", return_value=(("gce", compatible),)), \
                 patch.object(service.experiment_service, "preflight"):
                payload = quick_start_api.plan_payload({"dataset": "alias-says-cifar10", "paper_id": "gce",
                    "noise": {"kind": "synthetic", "key": "symmetric", "rate": 0.2, "seed": 73}})
                self.assertEqual(payload["status"], "ready", payload["details"])
                provenance = json.loads(Path(payload["provenance_path"]).read_text(encoding="utf-8"))
                self.assertEqual(provenance["base_recipe"], "gce-cifar100")
                saved = load_yaml(Path(payload["generated_config_path"]))
                base = load_yaml(Path(provenance["base_config_path"]))
                self.assertEqual(saved["trainer"], base["trainer"])
                self.assertEqual(saved["scheduler"], base["scheduler"])
                self.assertIn(payload["generated_config_path"], payload["command"])
                self.assertIn(payload["generated_config_path"], payload["dry_run_command"])
                with self.assertRaisesRegex(ValueError, "没有对应"):
                    quick_start_api.plan_payload({"dataset": "alias-says-cifar10", "paper_id": "pdl",
                                                 "noise": {"kind": "clean", "key": "clean"}})

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
