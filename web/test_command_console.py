import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib import error, request


sys.path.insert(0, str(Path(__file__).resolve().parent))
import command_console  # noqa: E402


class CommandConsoleTest(unittest.TestCase):
    def test_command_catalog_is_fixed_and_displayable(self):
        self.assertGreaterEqual(len(command_console.COMMANDS), 8)
        for spec in command_console.COMMANDS.values():
            self.assertTrue(spec.display_command.startswith("lnl "))
            self.assertNotIn("|", spec.args)
            self.assertNotIn(";", spec.args)

    def test_recipe_menu_covers_built_in_recipes(self):
        recipes = command_console._recipe_payload()
        self.assertGreaterEqual(len(recipes), 26)
        recipe_ids = {item["id"] for item in recipes}
        self.assertIn("cifar10-clean-smoke", recipe_ids)
        self.assertIn("fine-cifar100n-reproduction", recipe_ids)

    def test_sweep_shortcut_is_available(self):
        command = command_console.COMMANDS["sweep-smoke"].display_command
        self.assertIn("lnl sweep", command)
        self.assertIn("--seeds 1 2 3", command)

    def test_paper_menu_exposes_recipe_variants(self):
        papers = command_console._paper_payload()
        self.assertGreaterEqual(len(papers), 1)
        self.assertTrue(all(item["configs"] for item in papers))
        self.assertTrue(all("id" in item and "title" in item for item in papers))

    def test_dataset_payload_distinguishes_registration_from_training_evidence(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"LNL_DATA_CATALOG": str(Path(directory) / "datasets.json")},
            clear=False,
        ):
            from lnl_toolbox.data.local_catalog import LocalDatasetCatalog

            source = Path(directory) / "cifar"
            source.mkdir()
            catalog = LocalDatasetCatalog()
            catalog.register("lab", "cifar10", {"root": source})
            payload = command_console._dataset_payload()
            self.assertIn("cifar10", payload["adapters"])
            self.assertNotIn("synthetic_multiclass", payload["adapters"])
            lab = next(item for item in payload["datasets"] if item["name"] == "lab")
            self.assertEqual(lab["status"], "incomplete")
            catalog.mark_layout_validated(
                "lab", {"train_samples": 10, "test_samples": 2, "classes": 10}
            )
            payload = command_console._dataset_payload()
            lab = next(item for item in payload["datasets"] if item["name"] == "lab")
            self.assertEqual(lab["status"], "ready")
            self.assertEqual(lab["train_samples"], 10)

    def test_dataset_actions_use_data_service_directly(self):
        report = mock.Mock()
        report.to_dict.return_value = {"name": "lab", "status": "ready"}
        service = mock.Mock()
        service.status.return_value = report
        with mock.patch(
            "lnl_toolbox.training.data_service.DataService", return_value=service
        ):
            value = command_console._dataset_action(
                {"action": "status", "name": "lab"}
            )
        service.status.assert_called_once_with("lab")
        self.assertEqual(value["dataset"]["status"], "ready")

    def test_dataset_lifecycle_returns_guidance_without_changing_service_contract(self):
        registered = mock.Mock()
        registered.to_dict.return_value = {"name": "lab", "status": "incomplete"}
        inspected = mock.Mock(status="ready", error=None)
        inspected.to_dict.return_value = {
            "name": "lab",
            "status": "ready",
            "train_samples": 8,
            "test_samples": 2,
        }
        service = mock.Mock()
        service.register.return_value = registered
        service.inspect.return_value = inspected
        with mock.patch(
            "lnl_toolbox.training.data_service.DataService", return_value=service
        ):
            registration = command_console._dataset_action(
                {"action": "register", "name": "lab", "adapter": "cifar10", "root": "data"}
            )
            inspection = command_console._dataset_action(
                {"action": "inspect", "name": "lab"}
            )
            removal = command_console._dataset_action(
                {"action": "remove", "name": "lab"}
            )
        self.assertEqual(registration["next_action"], "inspect")
        self.assertEqual(inspection["next_action"], "verify")
        self.assertIn("原始数据文件未被删除", removal["message"])
        service.remove.assert_called_once_with("lab")

    def test_dataset_http_api_uses_shared_status_contract(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"LNL_DATA_CATALOG": str(Path(directory) / "datasets.json")},
            clear=False,
        ):
            server = command_console.ThreadingHTTPServer(
                ("127.0.0.1", 0), command_console.ConsoleHandler
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with request.urlopen(f"{base}/api/datasets") as response:
                    listing = json.loads(response.read())
                self.assertIn("cifar10", listing["adapters"])
                with request.urlopen(f"{base}/") as response:
                    home = response.read().decode("utf-8-sig")
                with request.urlopen(f"{base}/recipe") as response:
                    recipe = response.read().decode("utf-8-sig")
                self.assertIn("LNL Toolbox Command Console", home)
                self.assertIn("const recipeMode", recipe)
                self.assertNotIn("workspace-tabs", home)
                self.assertIn("1. 登记路径", home)
                self.assertIn("再次点击，确认删除登记", home)
                self.assertIn('dataAdapter: "cifar10"', home)
                self.assertIn("state.dataOutput = event.target.value", home)
                body = json.dumps(
                    {"action": "status", "name": "cifar10"}
                ).encode("utf-8")
                http_request = request.Request(
                    f"{base}/api/datasets",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(http_request) as response:
                    status = json.loads(response.read())
                self.assertEqual(status["dataset"]["status"], "missing")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_dataset_http_permission_error_is_returned_as_json(self):
        server = command_console.ThreadingHTTPServer(
            ("127.0.0.1", 0), command_console.ConsoleHandler
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = json.dumps({"action": "list"}).encode("utf-8")
        http_request = request.Request(
            f"{base}/api/datasets",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with mock.patch.object(
                command_console, "_dataset_action", side_effect=PermissionError("denied")
            ):
                with self.assertRaises(error.HTTPError) as raised:
                    request.urlopen(http_request)
                payload = json.loads(raised.exception.read())
            self.assertEqual(raised.exception.code, 400)
            self.assertEqual(payload["error"], "denied")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_main_passes_browser_opening_option(self):
        with mock.patch.object(command_console, "serve") as serve:
            code = command_console.main(
                ["--host", "0.0.0.0", "--port", "9000", "--open"]
            )
        self.assertEqual(code, 0)
        serve.assert_called_once_with("0.0.0.0", 9000, open_browser=True)

    def test_recipe_yaml_can_be_loaded_and_saved_as_project_local_file(self):
        config = command_console._config_payload("cifar10-clean-smoke")
        self.assertIn("data:", config["content"])
        schema = command_console._config_schema("cifar10-clean-smoke")
        self.assertIn("data.name", {field["path"] for field in schema["fields"]})
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            destination = Path(directory) / "edited.yaml"
            saved = command_console._save_config(
                {
                    "path": str(destination),
                    "recipe": "cifar10-clean-smoke",
                    "patches": [{"path": "data.name", "value": "cifar10"}],
                }
            )
            self.assertEqual(Path(saved["path"]).name, "edited.yaml")
            self.assertTrue(destination.is_file())

    def test_schema_hides_component_wiring(self):
        schema = command_console._config_schema("fine-cifar100n-smoke")
        paths = {field["path"] for field in schema["fields"]}
        self.assertIn("fine.warmup_epochs", paths)
        self.assertNotIn("model.name", paths)
        self.assertNotIn("execution.runner", paths)

        cdr_schema = command_console._config_schema("cifar10-symmetric-cdr-smoke")
        cdr_paths = {field["path"] for field in cdr_schema["fields"]}
        self.assertIn("parameter_update.noise_rate", cdr_paths)
        self.assertNotIn("parameter_update.name", cdr_paths)

    def test_safe_save_rejects_component_changes(self):
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            with self.assertRaises(ValueError):
                command_console._save_config(
                    {
                        "path": str(Path(directory) / "edited.yaml"),
                        "recipe": "fine-cifar100n-smoke",
                        "patches": [{"path": "model.name", "value": "resnet50"}],
                    }
                )

    def test_builtin_recipe_cannot_be_overwritten(self):
        config = command_console._config_payload("cifar10-clean-smoke")
        with self.assertRaises(ValueError):
            command_console._save_config(
                {
                    "path": config["path"],
                    "recipe": "cifar10-clean-smoke",
                    "patches": [],
                    "overwrite": True,
                }
            )

    def test_unknown_command_is_rejected(self):
        with self.assertRaises(KeyError):
            command_console.build_command("not-allowed")

    def test_free_command_is_parsed_without_shell(self):
        with mock.patch.object(command_console.shutil, "which", return_value="lnl"):
            command = command_console.parse_free_command(
                'lnl run --recipe "cifar10-clean-smoke" --epochs 1'
            )
        self.assertEqual(command[0], "lnl")
        self.assertIn("--epochs", command)
        self.assertIn("1", command)

    def test_free_command_rejects_shell_syntax(self):
        for raw in ("python evil.py", "lnl doctor | more", "lnl doctor > out.txt"):
            with self.assertRaises(ValueError):
                command_console.parse_free_command(raw)

    def test_fallback_command_uses_current_python(self):
        with mock.patch.object(command_console.shutil, "which", return_value=None):
            self.assertEqual(
                command_console.resolve_lnl_command(),
                [sys.executable, "-m", "lnl_toolbox.cli.main"],
            )

    def test_command_builder_does_not_use_shell(self):
        with mock.patch.object(command_console.shutil, "which", return_value="lnl"):
            command = command_console.build_command("train-one")
        self.assertEqual(command[:1], ["lnl"])
        self.assertIn("--epochs", command)
        self.assertIn("1", command)

    def test_job_payload_is_json_serializable(self):
        job = command_console.Job(
            job_id="abc",
            key="help",
            command=["lnl", "--help"],
            display_command="lnl --help",
            lines=["ok"],
            returncode=0,
        )
        payload = command_console._job_payload(job)
        self.assertEqual(payload["returncode"], 0)
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
