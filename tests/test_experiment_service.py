from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from lnl_toolbox.training.service import ExperimentService


class _FakeRunner:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, config, output_dir=None, resume=None):
        self.calls += 1
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "final_metrics.json").write_text(
            json.dumps({"test_accuracy": 0.8}), encoding="utf-8"
        )
        return root


class ExperimentServiceTest(unittest.TestCase):
    def test_preflight_reuses_config_and_data_validation_without_running(self) -> None:
        service = ExperimentService()
        config = {"data": {"name": "cifar10", "root": "missing"}}
        runner = object()
        with patch("lnl_toolbox.catalog.validate_config", return_value=runner) as validate, patch(
            "lnl_toolbox.training.data_service.validate_data_config"
        ) as validate_data:
            self.assertIs(service.preflight(config), runner)
        validate.assert_called_once_with(config, check_data=False)
        validate_data.assert_called_once_with(config)

        with patch("lnl_toolbox.catalog.validate_config", return_value=runner), patch(
            "lnl_toolbox.training.data_service.validate_data_config"
        ) as validate_data:
            self.assertIs(service.preflight(config, check_data=False), runner)
        validate_data.assert_not_called()

    def test_service_invokes_runner_and_writes_standard_artifacts(self) -> None:
        runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.training.service.resolve_runner", return_value=runner
        ):
            root = ExperimentService().run(
                {"seed": 2, "method": "fake"}, directory, recipe="fake-smoke"
            )
            result = json.loads((root / "final_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(runner.calls, 1)
            self.assertEqual(result["recipe"], "fake-smoke")
            self.assertTrue((root / "resolved_config.yaml").is_file())
            self.assertTrue((root / "environment.json").is_file())

    def test_resume_of_completed_run_is_strict_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "resolved_config.yaml").write_text(
                yaml.safe_dump({"method": "fake"}), encoding="utf-8"
            )
            final = root / "final_metrics.json"
            final.write_text(
                json.dumps({"status": "completed", "completed": True}), encoding="utf-8"
            )
            before = final.read_bytes()
            with patch("lnl_toolbox.training.service.resolve_runner") as resolver:
                self.assertEqual(ExperimentService().resume(root), root.resolve())
            resolver.assert_not_called()
            self.assertEqual(final.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
