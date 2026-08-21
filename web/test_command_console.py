import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib import error, parse, request


sys.path.insert(0, str(Path(__file__).resolve().parent))
import command_console  # noqa: E402


class CommandConsoleTest(unittest.TestCase):
    def test_command_catalog_is_fixed_and_displayable(self):
        self.assertGreaterEqual(len(command_console.COMMANDS), 8)
        for spec in command_console.COMMANDS.values():
            self.assertTrue(spec.display_command.startswith("lnl "))
            self.assertNotIn("|", spec.args)
            self.assertNotIn(";", spec.args)

    def test_recipe_menu_defaults_to_curated_templates(self):
        recipes = command_console._recipe_payload()
        self.assertEqual(len(recipes), 4)
        recipe_ids = {item["id"] for item in recipes}
        self.assertIn("cifar10-clean-smoke", recipe_ids)
        self.assertIn("cifar10-clean-baseline", recipe_ids)
        self.assertNotIn("fine-cifar100n-reproduction", recipe_ids)
        self.assertTrue(all(item["visibility"] == "public" for item in recipes))
        self.assertTrue(all(item["label"] for item in recipes))

        advanced = command_console._recipe_payload(include_all=True)
        advanced_ids = {item["id"] for item in advanced}
        self.assertGreaterEqual(len(advanced), 60)
        self.assertIn("fine-cifar100n-reproduction", advanced_ids)

    def test_sweep_shortcut_is_available(self):
        command = command_console.COMMANDS["sweep-smoke"].display_command
        self.assertIn("lnl sweep", command)
        self.assertIn("--seeds 1 2 3", command)

    def test_sweep_plan_supports_parameter_matrix_and_optional_seeds(self):
        payload = command_console._sweep_plan_payload(
            {
                "recipe": "cifar10-clean-smoke",
                "matrix": {
                    "loader.batch_size": [256, 512],
                    "optimizer.lr": [0.01, 0.001],
                },
                "seeds": [],
            }
        )
        self.assertEqual(payload["total"], 4)
        payload = command_console._sweep_plan_payload(
            {
                "recipe": "cifar10-clean-smoke",
                "matrix": {
                    "loader.batch_size": [256, 512],
                    "optimizer.lr": [0.01, 0.001],
                },
                "seeds": [1, 2, 3],
            }
        )
        self.assertEqual(payload["total"], 12)

    def test_windows_picker_returns_selection_or_cancellation(self):
        completed = mock.Mock(returncode=0, stdout="F:\\runs", stderr="")
        with mock.patch.object(command_console.os, "name", "nt"), mock.patch.object(
            command_console.subprocess, "run", return_value=completed
        ) as run:
            result = command_console._picker_payload(
                {"mode": "folder", "initial": "", "kind": "all"}
            )
        self.assertEqual(result["path"], "F:\\runs")
        self.assertFalse(result["cancelled"])
        self.assertFalse(run.call_args.kwargs["shell"] if "shell" in run.call_args.kwargs else False)
        self.assertEqual(run.call_args.kwargs["env"]["LNL_PICKER_INITIAL"], str(command_console.ROOT))

        completed.stdout = ""
        with mock.patch.object(command_console.os, "name", "nt"), mock.patch.object(
            command_console.subprocess, "run", return_value=completed
        ):
            self.assertTrue(command_console._picker_payload({"mode": "open_file"})["cancelled"])

    def test_result_payload_includes_partial_metric_history(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run-a"
            run.mkdir()
            (run / "metrics.jsonl").write_text(
                json.dumps({"epoch": 1, "validation_accuracy": 0.5}) + "\n",
                encoding="utf-8",
            )
            payload = command_console._results_payload(directory)
        self.assertEqual(len(payload["runs"]), 1)
        self.assertEqual(payload["runs"][0]["current_epoch"], 1)

    def test_resume_payload_reports_config_phase_files_and_readiness(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "resolved_config.yaml").write_text(
                "seed: 7\nmethod: supervised\nexecution:\n  runner: supervised\n"
                "data:\n  name: cifar10\nnoise:\n  name: symmetric\n"
                "model:\n  name: tiny_cnn\noptimizer:\n  name: sgd\n"
                "trainer:\n  epochs: 20\n  warmup_epochs: 5\n",
                encoding="utf-8",
            )
            (run / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "epoch": 4,
                        "phase": "warmup",
                        "validation_accuracy": 0.4,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            torch.save(
                {
                    "completed_epoch": 3,
                    "run_state": {"phase": "train", "step": 12},
                    "best_epoch": 3,
                    "best_validation_accuracy": 0.4,
                },
                run / "last.pt",
            )
            payload = command_console._resume_payload(directory, "last")
            self.assertTrue(payload["resumable"])
            self.assertEqual(payload["phase"], "warmup")
            self.assertEqual(payload["current_epoch"], 4)
            self.assertEqual(payload["target_epoch"], 20)
            self.assertEqual(payload["config_summary"]["data"], "cifar10")
            self.assertIn("resolved_config.yaml", {item["name"] for item in payload["files"]})

            completed_config = (run / "resolved_config.yaml").read_text(encoding="utf-8").replace(
                "epochs: 20", "epochs: 4"
            )
            (run / "resolved_config.yaml").write_text(completed_config, encoding="utf-8")
            completed = command_console._resume_payload(directory, "last")
            self.assertFalse(completed["resumable"])
            self.assertEqual(completed["status"], "completed")
            self.assertIn("target epoch", completed["errors"][-1])

            (run / "resolved_config.yaml").unlink()
            blocked = command_console._resume_payload(directory, "last")
            self.assertFalse(blocked["resumable"])
            self.assertIn("missing resolved_config.yaml", blocked["errors"])

    def test_paper_menu_exposes_recipe_variants(self):
        papers = command_console._paper_payload()
        self.assertEqual(len(papers), 26)
        self.assertTrue(all(item["configs"] for item in papers))
        self.assertTrue(all(item["summary"] for item in papers))
        self.assertTrue(all(item["mechanism"] for item in papers))
        self.assertTrue(all(item["concept_to_config"] for item in papers))
        self.assertTrue(all(item["configs"][0]["config_path"] for item in papers))
        self.assertTrue(all(item["configs"][0]["label"] for item in papers))
        self.assertTrue(all("id" in item and "title" in item for item in papers))
        self.assertTrue(all(item["default_recipe_id"] for item in papers))
        self.assertTrue(all(item["default_fidelity"] for item in papers))
        self.assertEqual(
            next(item for item in papers if item["id"] == "cnlcu")["default_recipe_id"],
            "cnlcu-cifar10-reproduction",
        )

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

    def test_dataset_verify_defaults_to_automatic_profile(self):
        job = mock.Mock()
        with mock.patch.object(
            command_console, "resolve_lnl_command", return_value=["python", "-m", "lnl"]
        ), mock.patch.object(
            command_console, "_start_process", return_value=job
        ) as start:
            result = command_console._dataset_verify_job(
                {"name": "fashion", "output_dir": "artifacts/fashion-check"}
            )
        self.assertIs(result, job)
        command = start.call_args.args[1]
        self.assertEqual(command[:6], ["python", "-m", "lnl", "data", "verify", "fashion"])
        self.assertNotIn("--recipe", command)

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
                with request.urlopen(f"{base}/api/recipes") as response:
                    public_recipes = json.loads(response.read())
                with request.urlopen(f"{base}/api/recipes?all=true") as response:
                    all_recipes = json.loads(response.read())
                self.assertEqual(len(public_recipes), 4)
                self.assertGreater(len(all_recipes), len(public_recipes))
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
                self.assertIn('field("本地数据集", "training-data"', home)
                self.assertIn('{value:"paper", label:"论文正式配置"}', home)
                self.assertIn('field("论文方法", "yaml-paper"', home)
                self.assertIn('label="论文正式配置（26）"', home)
                self.assertIn("await loadYamlFromPath(createdPath)", home)
                self.assertIn('body.command.startsWith("lnl compose create ")', home)
                self.assertIn("async function loadYamlFromEditor()", home)
                self.assertIn("function renderSweepV2()", home)
                self.assertIn("function drawResultChart()", home)
                self.assertIn('id="result-list-toggle"', home)
                self.assertIn('id="result-filter"', home)
                self.assertIn("const selectedPaths = new Set(state.resultSelected)", home)
                self.assertIn('id="result-compare-tools"', home)
                self.assertIn('id="result-resume-inspect"', home)
                self.assertIn("function renderResumeDashboard()", home)
                self.assertIn("/api/resume-inspect?path=", home)
                self.assertIn("state.resumeInspectionKey !== currentResumeKey()", home)
                self.assertIn("function updateResultVisibility()", home)
                self.assertIn('id="paper-open-yaml"', home)
                self.assertIn("论文方法、配置字段与代码的关系", home)
                self.assertIn("await loadYamlSelection(state.paperRecipe)", home)
                self.assertNotIn('id="paper-profile"', home)
                self.assertNotIn('id="paper-variant"', home)
                self.assertNotIn("配置 profile 与 recipe 变体", home)
                self.assertIn("/api/picker", home)
                self.assertIn("/api/results?path=", home)
                self.assertIn('id="yaml-text" spellcheck="false" placeholder=', home)
                self.assertNotIn('id="yaml-text" spellcheck="false" readonly', home)
                self.assertIn('let command = "lnl data verify " + quoteArg(alias);', home)
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

            loaded = command_console._config_payload(path_value=saved["path"])
            self.assertEqual(loaded["recipe"], "")
            self.assertEqual(loaded["path"], saved["path"])
            self.assertEqual(loaded["runner"], "clean")

            edited_content = loaded["content"].replace("seed: 7", "seed: 19", 1)
            overwritten = command_console._save_config(
                {
                    "path": saved["path"],
                    "source_path": saved["path"],
                    "content": edited_content,
                    "overwrite": True,
                }
            )
            self.assertIn("seed: 19", overwritten["content"])

    def test_complete_yaml_edit_rejects_invalid_configuration(self):
        config = command_console._config_payload("cifar10-clean-smoke")
        invalid = config["content"].replace("runner: clean", "runner: missing", 1)
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            with self.assertRaises(ValueError):
                command_console._save_config(
                    {
                        "path": str(Path(directory) / "invalid.yaml"),
                        "content": invalid,
                    }
                )

    def test_project_yaml_http_round_trip(self):
        server = command_console.ThreadingHTTPServer(
            ("127.0.0.1", 0), command_console.ConsoleHandler
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        source = command_console._config_payload("cifar10-clean-smoke")["content"]
        try:
            with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
                destination = Path(directory) / "web-created.yaml"
                relative = destination.relative_to(command_console.ROOT).as_posix()
                body = json.dumps(
                    {"path": relative, "content": source, "overwrite": False}
                ).encode("utf-8")
                save_request = request.Request(
                    f"{base}/api/configs",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(save_request) as response:
                    saved = json.loads(response.read())
                self.assertEqual(saved["path"], relative)

                query = parse.urlencode({"path": relative})
                with request.urlopen(f"{base}/api/configs?{query}") as response:
                    loaded = json.loads(response.read())
                with request.urlopen(f"{base}/api/config-schema?{query}") as response:
                    schema = json.loads(response.read())
                self.assertEqual(loaded["path"], relative)
                self.assertEqual(schema["source_path"], relative)
                self.assertTrue(schema["fields"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

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
