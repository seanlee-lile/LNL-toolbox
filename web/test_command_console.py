import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib import error, parse, request


sys.path.insert(0, str(Path(__file__).resolve().parent))
import command_console  # noqa: E402


class CommandConsoleTest(unittest.TestCase):
    def test_every_paper_and_registered_variant_has_readable_parameter_controls(self):
        from lnl_toolbox.catalog import discover_recipes

        bindings = command_console._parameter_registry()["formal_recipe_bindings"]
        covered = set()
        for recipe in discover_recipes(command_console.ROOT, include_conditional=True):
            schema = command_console._config_schema(recipe.id)
            method = schema["method"]
            if method not in bindings:
                continue
            covered.add(method)
            for field in schema["fields"]:
                if field.get("visible") is False:
                    continue
                with self.subTest(recipe=recipe.id, path=field["path"]):
                    self.assertIsNotNone(command_console._parameter_ui_spec(field["path"]))
                    self.assertRegex(field["note"], r"[\u4e00-\u9fff]")
                    self.assertEqual(field["label"], command_console._parameter_user_label(field["path"], method))
                    self.assertNotIn("通用配置参数", field["note"])
                    if field["kind"] in {"text", "string"} and not field["path"].endswith((".path", ".artifact_path")):
                        choices = field["choices"]
                        self.assertTrue(choices, "selectable strings must provide dropdown choices")
                        self.assertIn(str(field["value"]), {choice["value"] for choice in choices})
                        for choice in choices:
                            self.assertTrue(choice["label"])
                        if field["editable"]:
                            self.assertEqual(command_console._coerce_patch_value(field, field["value"]), field["value"])
                            with self.assertRaisesRegex(ValueError, "下拉"):
                                command_console._coerce_patch_value(field, "not-a-supported-option")
                    if field["path"] in {"models", "feature_stage.layers"}:
                        self.assertTrue(field["rows"], "nested string choices must not be a JSON guessing exercise")
                        for row in field["rows"]:
                            for child in row:
                                if child["kind"] == "text":
                                    self.assertTrue(child["choices"])
                                    self.assertIn(child["value"], {choice["value"] for choice in child["choices"]})
        self.assertEqual(covered, set(bindings))
        self.assertEqual(len(covered), 26)

    def test_nested_parameter_dropdowns_reject_unknown_values(self):
        from copy import deepcopy
        from lnl_toolbox.catalog import load_yaml, recipe_by_id

        for method, path in (("pcse", "feature_stage.layers"), ("jocor", "models")):
            recipe = command_console._parameter_registry()["formal_recipe_bindings"][method]
            config = load_yaml(recipe_by_id(recipe, command_console.ROOT).config_path)
            field = command_console._field_map(config, method)[path]
            self.assertEqual(command_console._coerce_patch_value(field, field["value"]), field["value"])
            value = deepcopy(field["value"])
            value[0]["name"] = "invented-string"
            with self.assertRaisesRegex(ValueError, "下拉"):
                command_console._coerce_patch_value(field, value)

    def test_review_temp_cleanup_on_exit_cancel_and_spawn_failure(self):
        for returncode in (0, 1, -15):
            with self.subTest(returncode=returncode):
                saved = command_console._quick_start_config({
                    "recipe": "binary-risk-natarajan-reproduction", "patches": [],
                }, review=True)
                path = Path(saved["path"])
                process = mock.Mock(stdout=io.StringIO("finished\n"))
                process.wait.return_value = returncode
                job = command_console.Job("test", "custom", ["lnl", "run", "--config", str(path), "--dry-run", "--review"], "review", process=process, cancel_requested=returncode == -15)
                command_console._read_output(job)
                self.assertTrue(job.output_complete)
                self.assertFalse(path.exists())
                self.assertFalse(path.parent.exists())
        saved = command_console._quick_start_config({
            "recipe": "binary-risk-natarajan-reproduction", "patches": [],
        }, review=True)
        with mock.patch.object(command_console.subprocess, "Popen", side_effect=OSError("cannot start")):
            job = command_console.start_free_job(f'lnl run --config "{saved["path"]}" --dry-run --review')
        self.assertEqual(job.returncode, -1)
        self.assertFalse(Path(saved["path"]).exists())
        with command_console.JOBS_LOCK:
            command_console.JOBS.pop(job.job_id, None)

    def test_review_snapshot_returns_real_cli_errors_and_dry_run_over_http(self):
        server = command_console.ThreadingHTTPServer(("127.0.0.1", 0), command_console.ConsoleHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        original_save = command_console._save_config
        def post(path, payload):
            req = request.Request(base + path, data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
            with request.urlopen(req) as response:
                return json.load(response)
        def wait(job):
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                with request.urlopen(base + "/api/jobs/" + job["id"]) as response:
                    job = json.load(response)
                if not job["running"] and job["output_complete"]:
                    return job
                time.sleep(0.1)
            self.fail("CLI review timed out")
        try:
            with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
                def save_in_temp(payload, **kwargs):
                    if "destination" in kwargs:
                        return original_save(payload, **kwargs)
                    return original_save({**payload, "path": str(Path(directory) / Path(payload["path"]).name)}, **kwargs)
                with mock.patch.object(command_console, "_save_config", side_effect=save_in_temp):
                    saved = post("/api/quick-start/review-config", {
                        "recipe": "gce-cifar10-noise02-reproduction", "patches": [
                            {"path": "optimizer.lr", "value": "非法 invalid"},
                            {"path": "trainer.epochs", "value": "invalid 中文"},
                        ],
                    })
                    job = wait(post("/api/run", {"command": f'lnl run --config "{saved["path"]}" --dry-run --review'}))
                    self.assertNotEqual(job["returncode"], 0)
                    output = "\n".join(job["lines"])
                    self.assertIn("invalid config for optimizer.lr, expected", output)
                    self.assertIn("invalid config for trainer.epochs, expected", output)
                    self.assertNotIn("Quick Start phase: rehearsing", output)
                    self.assertFalse(Path(saved["path"]).exists())
                    self.assertNotIn(str(command_console.ROOT), saved["path"])
                    saved = post("/api/quick-start/review-config", {"recipe": "binary-risk-natarajan-reproduction", "patches": []})
                    self.assertFalse(Path(saved["output_dir"]).exists())
                    self.assertEqual(Path(saved["config_path"]).parent, Path(saved["output_dir"]))
                    for command in (f'lnl validate --config "{saved["path"]}"', f'lnl run --config "{saved["path"]}" --output-dir "{saved["output_dir"]}" --dry-run --review'):
                        job = wait(post("/api/run", {"command": command}))
                        self.assertEqual(job["returncode"], 0, "\n".join(job["lines"]))
                        if "--review" in command:
                            self.assertIn("Quick Start phase: validating", job["lines"])
                            self.assertIn("Quick Start phase: rehearsing", job["lines"])
                            self.assertIn(f'  artifact directory: {saved["output_dir"]}', job["lines"])
                            self.assertIn(f'  training configuration: {saved["config_path"]}', job["lines"])
                    self.assertFalse(Path(saved["path"]).exists())
                    self.assertFalse(Path(saved["output_dir"]).exists())
                    persisted = post("/api/quick-start/training-config", {
                        "recipe": "binary-risk-natarajan-reproduction", "content": saved["content"], "run_id": saved["run_id"],
                    })
                    self.assertEqual(persisted["output_dir"], saved["output_dir"])
                    self.assertEqual((command_console.ROOT / persisted["path"]).read_text(encoding="utf-8"), saved["content"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


    def test_web_server_refuses_second_listener_on_same_port(self):
        with command_console.SingleInstanceHTTPServer(
            ("127.0.0.1", 0), command_console.ConsoleHandler
        ) as first:
            with self.assertRaises(OSError):
                command_console.ThreadingHTTPServer(
                    first.server_address, command_console.ConsoleHandler
                )

    def test_main_reports_occupied_web_port(self):
        with mock.patch.object(
            command_console, "serve", side_effect=OSError(10048, "address in use")
        ), contextlib.redirect_stderr(io.StringIO()) as output:
            result = command_console.main(["--port", "8765"])
        self.assertEqual(result, 1)
        self.assertIn("Web port 8765 is already in use", output.getvalue())

    def test_number_parameters_do_not_spin_on_mouse_wheel(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('document.addEventListener("wheel", function (event) {', page)
        self.assertIn('target instanceof HTMLInputElement && target.type === "number"', page)
        self.assertIn('document.activeElement === target', page)
        self.assertIn('target.blur();', page)
        self.assertIn('}, {capture:true, passive:true});', page)

    def test_command_preview_is_directly_editable_and_execute_uses_edit(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('textarea id="preview"', page)
        self.assertIn('id="preview-reset"', page)
        self.assertIn("state.previewEdited", page)
        self.assertIn("state.previewAutoCommand", page)
        self.assertIn("state.previewAutoRequest", page)
        self.assertIn("previewNode.value", page)
        self.assertIn("previewNode.addEventListener(\"input\"", page)
        self.assertIn("已修改，执行时将使用编辑后的指令", page)
        self.assertIn("const editedCommand = previewNode.value.trim();", page)
        self.assertIn("state.request = {command: editedCommand};", page)
        self.assertIn("navigator.clipboard.writeText(previewNode.value)", page)
        self.assertIn("function resetPreviewCommand()", page)
        self.assertIn('setRequest(request, command, options = {})', page)
        self.assertIn('setRequest: function (request, command) { setRequest(request, command, {resetPreview:true}); }', page)


    def test_quick_start_is_a_centered_previous_next_carousel(self):
        script = (command_console.WEB_ROOT / "assets" / "quick_start.js").read_text(encoding="utf-8")
        css = (command_console.WEB_ROOT / "assets" / "quick_start.css").read_text(encoding="utf-8")
        for marker in ("currentStep", "quickStartSteps", "renderQuickStartCarousel", "id=\"qs-prev\"", "id=\"qs-next\"", "上一步", "下一步"):
            self.assertIn(marker, script)
        for marker in (".qs-carousel", ".qs-carousel-slide", ".qs-carousel-nav", "margin-inline: auto"):
            self.assertIn(marker, css)
        self.assertIn(".qs-next-actions { width: 100%;", css)
        self.assertIn("max-width: 1280px", css)
        self.assertIn("grid-template-columns: auto minmax(0, 1fr) auto", css)
        self.assertIn("&lt; 上一步", script)
        self.assertIn("下一步 &gt;", script)
        self.assertIn(".qs-carousel-card", css)
        self.assertIn("border: 0; border-radius: 0; background: transparent", css)

    def test_yaml_builder_has_one_path_and_contextual_editor_lifecycle(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertEqual(page.count('id="yaml-path"'), 1)
        self.assertNotIn('id="yaml-output"', page)
        for marker in (
            "hideYamlEditor",
            "yamlDraftDirty",
            "继续编辑 YAML（未保存）",
            "目标已登记数据集（可选）",
            "compatible-recipes",
            "yamlRecipeCompatibilityAlias",
            "state.yamlDataset === alias && state.yamlRecipeCompatibilityAlias !== alias",
            "打开 YAML/配置说明",
        ):
            self.assertIn(marker, page)

    def test_concrete_recipe_compatibility_preserves_paper_identity(self):
        result = mock.Mock()
        result.to_dict.return_value = {
            "method": "fine",
            "dataset": "lab",
            "status": "incompatible",
            "reason_codes": ["wrong_class_count"],
            "reasons": [{"code": "wrong_class_count", "message": "needs 100"}],
            "warnings": [],
            "required_user_inputs": [],
        }
        service = mock.Mock()
        service.list_config_compatibility.return_value = (("fine-formal", result),)
        recipe = mock.Mock(config_path=Path("fine.yaml"))
        paper = {
            "id": "fine",
            "acronym": "FINE",
            "title": "Fine paper",
            "default_recipe_id": "fine-formal",
            "default_fidelity": "paper-protocol",
        }
        with mock.patch.object(command_console, "_paper_payload", return_value=[paper]), mock.patch(
            "lnl_toolbox.catalog.recipe_by_id", return_value=recipe
        ), mock.patch(
            "lnl_toolbox.catalog.load_yaml", return_value={"execution": {"runner": "fine"}}
        ), mock.patch(
            "command_console._dataset_profile_context",
            return_value=(
                {"dataset": "lab", "profile": {}, "detected": {}, "capabilities": {}, "unresolved_dataset_facts": [], "declared": {}},
                mock.Mock(),
            ),
        ), mock.patch(
            "lnl_toolbox.training.service.ExperimentService", return_value=service
        ):
            value = command_console._dataset_recipe_compatibility_payload("lab")
        self.assertEqual(value["recipes"][0]["recipe_id"], "fine-formal")
        self.assertEqual(value["recipes"][0]["acronym"], "FINE")
        self.assertEqual(value["recipes"][0]["status"], "incompatible")
        service.list_config_compatibility.assert_called_once()

    def test_web_profile_and_compatibility_helpers_use_experiment_service(self):
        profile = mock.Mock()
        profile.to_dict.return_value = {
            "dataset": "lab",
            "noise": {"rate": {"status": "unknown", "value": None}},
        }
        report = mock.Mock(
            status="ready", profile=profile, adapter="uci_binary", location="data.txt"
        )
        declarations = mock.Mock()
        declarations.to_dict.return_value = {
            "clean_train_labels": "unknown",
            "pretrained_roles": [],
        }
        result = mock.Mock()
        result.to_dict.return_value = {
            "method": "coteaching",
            "dataset": "lab",
            "status": "compatible_with_requirements",
            "reason_codes": ["requires_noise_rate_prior"],
            "reasons": [],
            "warnings": [],
            "required_user_inputs": ["noise_rate_prior"],
        }
        service = mock.Mock()
        service.inspect_dataset.return_value = report
        service.data_service.declarations.return_value = declarations
        capabilities = mock.Mock()
        capabilities.clean_train_labels.value = "unknown"
        capabilities.noise_status.value = "unknown"
        capabilities.noise_origin.value = "unknown"
        capabilities.noise_rate.status.value = "unknown"
        capabilities.to_dict.return_value = {}
        service.data_service.capabilities.return_value = capabilities
        service.list_compatible_methods.return_value = (result,)
        with mock.patch(
            "lnl_toolbox.training.service.ExperimentService", return_value=service
        ):
            value = command_console._dataset_compatibility_payload("lab")
        self.assertEqual(value["profile"]["noise"]["rate"]["status"], "unknown")
        self.assertEqual(value["methods"], [result.to_dict.return_value])
        service.list_compatible_methods.assert_called_once_with(
            "lab", method_noise_rate_prior=None
        )

    def test_dataset_compatibility_http_contract_preserves_phase2_statuses(self):
        profile = {
            "dataset": "lab",
            "noise": {"rate": {"status": "unknown", "value": None}},
        }
        compatibility = {
            "dataset": "lab",
            "profile": profile,
            "detected": profile,
            "declared": {},
            "method_noise_rate_prior": None,
            "methods": [{
                "method": "coteaching",
                "status": "compatible_with_requirements",
                "reason_codes": ["requires_noise_rate_prior"],
                "reasons": [],
                "warnings": [],
                "required_user_inputs": ["noise_rate_prior"],
            }],
        }
        server = command_console.ThreadingHTTPServer(
            ("127.0.0.1", 0), command_console.ConsoleHandler
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with mock.patch.object(
                command_console, "_dataset_profile_payload",
                return_value={"dataset": "lab", "profile": profile},
            ), mock.patch.object(
                command_console, "_dataset_compatibility_payload",
                return_value=compatibility,
            ):
                with request.urlopen(f"{base}/api/datasets/lab/profile") as response:
                    self.assertEqual(json.loads(response.read())["profile"], profile)
                with request.urlopen(
                    f"{base}/api/datasets/lab/compatible-methods"
                ) as response:
                    payload = json.loads(response.read())
            self.assertEqual(payload["methods"][0]["status"], "compatible_with_requirements")
            self.assertEqual(
                payload["methods"][0]["required_user_inputs"],
                ["noise_rate_prior"],
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_web_service_and_cli_json_compatibility_results_match(self):
        import importlib

        cli_main = importlib.import_module("lnl_toolbox.cli.main")
        profile = mock.Mock()
        profile.to_dict.return_value = {"dataset": "lab", "noise": {}}
        report = mock.Mock(
            status="ready", profile=profile, adapter="uci_binary", location="data.txt"
        )
        declarations = mock.Mock()
        declarations.to_dict.return_value = {}
        result = mock.Mock()
        result.to_dict.return_value = {
            "method": "importance_reweighting", "dataset": "lab",
            "status": "compatible", "reason_codes": [], "reasons": [],
            "warnings": [], "required_user_inputs": [],
        }
        service = mock.Mock()
        service.inspect_dataset.return_value = report
        service.data_service.declarations.return_value = declarations
        service.list_compatible_methods.return_value = (result,)
        with mock.patch(
            "lnl_toolbox.training.service.ExperimentService", return_value=service
        ):
            web_value = command_console._dataset_compatibility_payload("lab")
        output = io.StringIO()
        with mock.patch.object(cli_main, "ExperimentService", return_value=service), contextlib.redirect_stdout(output):
            self.assertEqual(
                cli_main.main([
                    "methods", "compatible", "--dataset", "lab",
                    "--format", "json",
                ]),
                0,
            )
        cli_value = json.loads(output.getvalue())
        self.assertEqual(web_value["methods"], cli_value)

    def test_web_declarations_keep_method_prior_out_of_dataset_facts(self):
        with self.assertRaisesRegex(ValueError, "experiment-specific"):
            command_console._dataset_declarations_payload(
                "lab", {"declarations": {"method_noise_rate_prior": {}}}
            )
        with self.assertRaisesRegex(ValueError, "experiment input"):
            command_console._dataset_declarations_payload(
                "lab", {"method_noise_rate_prior": 0.2, "declarations": {}}
            )

    def test_beginner_tutorial_contract_matches_documented_workflow(self):
        payload = command_console._tutorial_payload()
        self.assertEqual(payload["version"], 1)
        self.assertEqual(
            payload["sequence"],
            ["doctor", "list", "validate", "dry-run", "run", "resume"],
        )
        self.assertEqual(len(payload["steps"]), 6)
        self.assertTrue(all(step["why"] for step in payload["steps"]))
        self.assertTrue(all(step["success"] for step in payload["steps"]))
        self.assertEqual(payload["guide"], "docs/LNL-Toolbox-简明操作教程.md")

    def test_beginner_page_is_not_exposed(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertNotIn('fetch("/api/tutorial")', page)
        self.assertNotIn('id: "beginner"', page)
        self.assertNotIn('module:"beginner"', page)
        self.assertNotIn('完整新手引导', page)

    def test_quick_start_is_first_entry_and_reuses_existing_execution_flow(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertNotIn('id: "beginner"', page)
        self.assertIn('/assets/quick_start.js', page)
        self.assertIn('/assets/quick_start.css', page)
        self.assertIn('window.quickStartController.mount', page)
        self.assertNotIn("quickStartNoiseOptions", page)
        self.assertNotIn("quickStartCompatibilityRecipes", page)
        self.assertIn('fetch("/api/run"', page)

    def test_parameter_editor_exposes_registry_groups_and_deviation_warning(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("yaml-parameter-groups", page)
        self.assertIn("yaml-readonly-value", page)
        self.assertIn("已偏离论文配置", page)
        self.assertIn("acknowledge_paper_impact", page)




    def test_dataset_first_start_page_exposes_workspace_handoffs(self):
        quick_start = (command_console.WEB_ROOT / "assets" / "quick_start.js").read_text(encoding="utf-8")
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("compact: true", page)
        self.assertIn("function renderNextActions()", quick_start)
        self.assertIn("renderPlanParameters()", quick_start)
        self.assertIn('class="qs-parameter-advanced"', quick_start)
        self.assertIn("完整方法兼容性引导", quick_start)
        self.assertNotIn("高级模式：其他工作区", quick_start)

    def test_quick_start_parameter_edits_save_a_checked_run_config(self):
        quick_start = (command_console.WEB_ROOT / "assets" / "quick_start.js").read_text(encoding="utf-8")
        self.assertIn('request("/api/config-schema?" + source', quick_start)
        self.assertIn('jsonRequest("/api/quick-start/review-config",', quick_start)
        self.assertIn('dataset_alias:state.dataset.alias', quick_start)
        self.assertIn('commandForRun(state.plan.command, state.plan, saved)', quick_start)
        self.assertIn('fetch("/api/quick-start/training-config",', quick_start)
        self.assertIn('" --review"', quick_start)
        self.assertIn('plan.dry_run_command', quick_start)
        self.assertNotIn('qs-open-experiment', quick_start)

    def test_web_exposes_one_seed_and_saves_derived_noise_seed(self):
        import tempfile
        import yaml

        for method, recipe_id in command_console._parameter_registry()["formal_recipe_bindings"].items():
            paper_schema = command_console._config_schema(recipe_id)
            visible_seeds = [field["path"] for field in paper_schema["fields"]
                             if field["visible"] and (field["path"] == "seed" or field["path"].rsplit(".", 1)[-1].endswith("seed"))]
            self.assertEqual(visible_seeds, ["seed"], method)
            self.assertTrue(all(not field["editable"] for field in paper_schema["fields"]
                                if field["path"] != "seed" and field["path"].rsplit(".", 1)[-1].endswith("seed")), method)
        schema = command_console._config_schema("gce-cifar10-noise02-reproduction")
        fields = {field["path"]: field for field in schema["fields"]}
        self.assertTrue(fields["seed"]["visible"])
        self.assertFalse(fields["noise.seed"]["visible"])
        self.assertFalse(fields["noise.seed"]["editable"])
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            destination = Path(directory) / "one-seed.yaml"
            with self.assertRaisesRegex(ValueError, "内部随机种子由 seed 自动生成"):
                command_console._save_config({
                    "recipe": "gce-cifar10-noise02-reproduction",
                    "path": str(destination),
                    "patches": [{"path": "noise.seed", "value": 17}],
                })
            command_console._save_config({
                "recipe": "gce-cifar10-noise02-reproduction",
                "path": str(destination),
                "patches": [{"path": "seed", "value": 73}],
                "acknowledge_paper_impact": True,
            })
            stored = yaml.safe_load(destination.read_text(encoding="utf-8"))
            self.assertEqual(stored["seed"], 73)
            self.assertEqual(stored["noise"]["seed"], 73)

    def test_sweep_ui_reuses_parameter_metadata_groups_and_excludes_locks(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("sweepGroupsHtml", page)
        self.assertIn("levelInfo.default_expanded", page)
        self.assertIn("用途：", page)
        self.assertIn("论文依据：", page)
        self.assertIn("复现影响：", page)
        self.assertIn("不可加入参数组合实验：", page)
        self.assertIn("fieldInfo.editable && supported", page)
        self.assertIn("lnl sweep --", page)

    def test_command_catalog_is_fixed_and_displayable(self):
        for spec in command_console.COMMANDS.values():
            self.assertTrue(spec.display_command.startswith("lnl "))
            self.assertNotIn("|", spec.args)
            self.assertNotIn(";", spec.args)

    def test_recipe_menu_defaults_to_curated_templates(self):
        recipes = command_console._recipe_payload()
        recipe_ids = {item["id"] for item in recipes}
        self.assertIn("cifar10-clean-smoke", recipe_ids)
        self.assertIn("cifar10-clean-baseline", recipe_ids)
        self.assertNotIn("fine-cifar100n-reproduction", recipe_ids)
        self.assertTrue(all(item["visibility"] == "public" for item in recipes))
        self.assertTrue(all(item["label"] for item in recipes))

        advanced = command_console._recipe_payload(include_all=True)
        advanced_ids = {item["id"] for item in advanced}
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
        self.assertEqual(payload["source"]["kind"], "recipe")
        self.assertTrue(payload["matrix_fields"])

    def test_sweep_plan_accepts_project_yaml_and_reports_paper_deviation(self):
        source = command_console._config_payload(
            "gce-cifar10-noise02-reproduction"
        )["content"]
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            path = Path(directory) / "gce-custom.yaml"
            relative = path.relative_to(command_console.ROOT).as_posix()
            command_console._save_config(
                {
                    "path": relative,
                    "recipe": "gce-cifar10-noise02-reproduction",
                    "content": source,
                }
            )
            payload = command_console._sweep_plan_payload(
                {
                    "path": relative,
                    "matrix": {"loss.q": [0.7, 0.5]},
                    "seeds": [],
                }
            )
        self.assertEqual(payload["source"], {"kind": "path", "value": relative, "label": relative})
        self.assertEqual(payload["total"], 2)
        self.assertTrue(payload["modified_from_paper"])
        self.assertEqual(payload["paper_deviation_paths"], ["loss.q"])
        self.assertEqual(payload["matrix_fields"][0]["level"], "paper")

    def test_sweep_plan_requires_one_source_and_rejects_locked_or_wrong_types(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            command_console._sweep_plan_payload({"matrix": {}})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            command_console._sweep_plan_payload(
                {
                    "recipe": "fine-cifar100n-reproduction",
                    "path": "configs/experiment/fine_cifar100n_reproduction.yaml",
                }
            )
        with self.assertRaisesRegex(ValueError, "locked parameter"):
            command_console._sweep_plan_payload(
                {
                    "recipe": "fine-cifar100n-reproduction",
                    "matrix": {"execution.runner": ["clean"]},
                }
            )
        with self.assertRaisesRegex(ValueError, "必须是数字"):
            command_console._sweep_plan_payload(
                {
                    "recipe": "gce-cifar10-noise02-reproduction",
                    "matrix": {"loss.q": ["invalid"]},
                }
            )

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
            self.assertEqual(
                command_console._picker_payload({"mode": "open_file"}),
                {"cancelled": True, "path": None},
            )

    def test_windows_picker_uses_disposable_foreground_owner_for_every_mode(self):
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(command_console.os, "name", "nt"), mock.patch.object(
            command_console.subprocess, "run", return_value=completed
        ) as run:
            for mode in ("folder", "open_file", "save_file"):
                with self.subTest(mode=mode):
                    self.assertEqual(
                        command_console._picker_payload({"mode": mode}),
                        {"cancelled": True, "path": None},
                    )
                    script = run.call_args.args[0][-1]
                    self.assertIn("$owner=New-Object System.Windows.Forms.Form", script)
                    self.assertIn("$owner.ShowInTaskbar=$false", script)
                    self.assertIn("$owner.TopMost=$true", script)
                    self.assertIn("$d.ShowDialog($owner)", script)
                    self.assertIn("finally", script)
                    self.assertIn("$d.Dispose()", script)
                    self.assertIn("$owner.Close()", script)
                    self.assertIn("$owner.Dispose()", script)
                    self.assertEqual(
                        run.call_args.kwargs["env"]["LNL_PICKER_MODE"], mode
                    )

    def test_windows_picker_falls_back_to_root_for_relative_output_path(self):
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(command_console.os, "name", "nt"), mock.patch.object(
            command_console.subprocess, "run", return_value=completed
        ) as run:
            command_console._picker_payload(
                {"mode": "folder", "initial": "artifacts/runs/new-output"}
            )
        self.assertEqual(
            run.call_args.kwargs["env"]["LNL_PICKER_INITIAL"],
            str(command_console.ROOT),
        )

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
        mentornet = next(item for item in papers if item["id"] == "mentornet")
        preparation = mentornet["configs"][0]["preparation"]
        self.assertIn(preparation["status"], {"ready", "not_ready"})
        self.assertIn("artifact_ready", preparation)

    def test_mentornet_paper_payload_reports_artifact_readiness(self):
        status = {
            "status": "not_ready",
            "artifact_ready": False,
            "artifact_path": "mentor_artifact.pt",
            "artifact_error": None,
            "feature_ready": False,
            "feature_path": "mentor_features.npz",
            "teacher_config": "teacher.yaml",
            "preparation_available": True,
            "student_recipe": "mentornet-dd-cifar100-symmetric04-smoke",
            "commands": {
                "prepare": "lnl mentor prepare --config teacher.yaml --output-dir mentor",
                "train": "lnl mentor train --config teacher.yaml --output mentor_artifact.pt",
                "student": "lnl run --recipe mentornet-dd-cifar100-symmetric04-smoke --check-data",
            },
        }
        with mock.patch(
            "lnl_toolbox.catalog.mentornet_preparation_status",
            return_value=status,
        ):
            papers = command_console._paper_payload()
        mentornet = next(item for item in papers if item["id"] == "mentornet")
        self.assertEqual(
            mentornet["configs"][0]["preparation"]["status"], "not_ready"
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

    def test_dataset_registration_sources_follow_adapter_contract(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        helper = page[
            page.index("function datasetRegistrationSources(adapter)"):
            page.index("function buildModuleCommand()")
        ]
        self.assertIn('if (adapter === "uci_binary")', helper)
        self.assertIn('sources.path = selectValue("data-path", "").trim()', helper)
        self.assertNotIn('sources.root = selectValue("data-root", "").trim()', helper.split("} else {")[0])
        self.assertIn('sources.root = selectValue("data-root", "").trim()', helper)
        self.assertIn('["cifar10n", "cifar100n"].includes(adapter)', helper)

        command_builder = page[
            page.index("function buildModuleCommand()"):
            page.index("function buildDataApiRequest()")
        ]
        api_builder = page[
            page.index("function buildDataApiRequest()"):
            page.index("function updateModulePreview()")
        ]
        self.assertIn("const sources = datasetRegistrationSources(adapter);", command_builder)
        self.assertIn('if (sources.root) command += " --root "', command_builder)
        self.assertIn('if (sources.path) command += " --path "', command_builder)
        self.assertIn("const sources = datasetRegistrationSources(payload.adapter);", api_builder)
        self.assertIn("if (sources.root) payload.root = sources.root;", api_builder)
        self.assertIn("if (sources.path) payload.path = sources.path;", api_builder)
        self.assertNotIn('selectValue("data-root"', api_builder)
        self.assertNotIn('selectValue("data-path"', api_builder)



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
                self.assertTrue(public_recipes)
                self.assertGreater(len(all_recipes), len(public_recipes))
                with request.urlopen(f"{base}/") as response:
                    home = response.read().decode("utf-8-sig")
                with request.urlopen(f"{base}/recipe") as response:
                    recipe = response.read().decode("utf-8-sig")
                self.assertIn("LNL Toolbox Command Console", home)
                self.assertIn("LNL Toolbox Command Console", recipe)
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

    def test_gce_explicit_learning_rates_save_without_changing_formal_recipe(self):
        schema = command_console._config_schema("gce-cifar10-noise02-reproduction")
        fields = {field["path"]: field for field in schema["fields"]}
        self.assertTrue(fields["scheduler.lr_values"]["editable"])
        self.assertIsNone(fields["scheduler.lr_values"]["value"])
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            destination = Path(directory) / "gce-custom-rates.yaml"
            saved = command_console._save_config({
                "path": str(destination),
                "recipe": "gce-cifar10-noise02-reproduction",
                "patches": [{"path": "scheduler.lr_values", "value": [0.003, 0.0007]}],
                "overwrite": False,
            })
            from lnl_toolbox.catalog import load_yaml

            custom = load_yaml(command_console.ROOT / saved["path"])
            self.assertEqual(custom["scheduler"]["lr_values"], [0.003, 0.0007])
            self.assertEqual(custom["scheduler"]["gamma"], 0.1)
            formal = load_yaml(command_console.ROOT / "configs/experiment/gce_cifar10_noise02_reproduction.yaml")
            self.assertNotIn("lr_values", formal["scheduler"])
            with self.assertRaisesRegex(ValueError, "one value per milestone"):
                command_console._save_config({
                    "path": str(Path(directory) / "invalid-rates.yaml"),
                    "recipe": "gce-cifar10-noise02-reproduction",
                    "patches": [{"path": "scheduler.lr_values", "value": [0.003]}],
                    "overwrite": False,
                })

    def test_lend_can_save_an_independent_rate_without_changing_paper_defaults(self):
        from lnl_toolbox.catalog import load_yaml
        from lnl_toolbox.training.experiment import build_scheduler
        import torch

        recipe = "lend-cifar10-reproduction"
        fields = {field["path"]: field for field in command_console._config_schema(recipe)["fields"]}
        self.assertTrue(fields["scheduler.lr_values"]["editable"])
        self.assertIsNone(fields["scheduler.lr_values"]["value"])
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            destination = Path(directory) / "lend-custom-rates.yaml"
            command_console._save_config({
                "path": str(destination), "recipe": recipe,
                "patches": [{"path": "scheduler.lr_values", "value": [0.007]}],
                "acknowledge_paper_impact": True,
            })
            custom = load_yaml(destination)
            formal = load_yaml(command_console.ROOT / "configs/experiment/lend_cifar10_reproduction.yaml")
            self.assertEqual(custom["scheduler"]["lr_values"], [0.007])
            self.assertNotIn("lr_values", formal["scheduler"])
            self.assertEqual(formal["optimizer"]["lr"], 0.05)
            parameter = torch.nn.Parameter(torch.tensor(1.0))
            optimizer = torch.optim.SGD([parameter], lr=custom["optimizer"]["lr"])
            scheduler = build_scheduler(optimizer, custom["scheduler"], 200)
            for _ in range(100):
                optimizer.step()
                scheduler.step()
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.007)
            with self.assertRaisesRegex(ValueError, "one value per milestone"):
                command_console._save_config({
                    "path": str(Path(directory) / "lend-invalid-rates.yaml"),
                    "recipe": recipe,
                    "patches": [{"path": "scheduler.lr_values", "value": [0.007, 0.003]}],
                    "acknowledge_paper_impact": True,
                })

    def test_all_formal_stage_schedules_accept_independent_rate_patches(self):
        from lnl_toolbox.catalog import load_yaml

        bindings = command_console._parameter_registry()["formal_recipe_bindings"]
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            for method, recipe in bindings.items():
                schema = command_console._config_schema(recipe)
                fields = {field["path"]: field for field in schema["fields"]}
                rate_paths = [path for path in fields if path.endswith(".scheduler.lr_values")
                              or path == "scheduler.lr_values" or ".scheduler." in path and path.endswith(".lr_values")]
                if not rate_paths:
                    continue
                patches = []
                for path in rate_paths:
                    prefix = path.removesuffix(".lr_values")
                    nodes = fields.get(prefix + ".step_milestones") or fields.get(prefix + ".milestones")
                    self.assertIsNotNone(nodes, (method, path))
                    self.assertTrue(fields[path]["editable"], (method, path))
                    patches.append({"path": path, "value": [0.007] * len(nodes["value"])})
                destination = Path(directory) / (method + ".yaml")
                command_console._save_config({
                    "path": str(destination), "recipe": recipe, "patches": patches,
                    "acknowledge_paper_impact": True,
                })
                saved = load_yaml(destination)
                for patch in patches:
                    value = saved
                    for part in patch["path"].split("."):
                        value = value[part]
                    self.assertEqual(value, patch["value"], (method, patch["path"]))

    def test_all_formal_stage_schedules_can_be_disabled_for_fixed_learning_rate(self):
        bindings = command_console._parameter_registry()["formal_recipe_bindings"]
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            for method, recipe in bindings.items():
                schema = command_console._config_schema(recipe)
                fields = {field["path"]: field for field in schema["fields"]}
                patches = []
                for path, field in fields.items():
                    if not field["editable"] or not field["visible"]:
                        continue
                    if path.endswith("scheduler.name") and field["value"] != "none":
                        patches.append({"path": path, "value": "none"})
                    if path == "scheduler.step_milestones" and field["value"]:
                        patches.append({"path": path, "value": []})
                if not patches:
                    continue
                command_console._save_config({
                    "path": str(Path(directory) / (method + "-fixed.yaml")),
                    "recipe": recipe, "patches": patches,
                    "acknowledge_paper_impact": True,
                })

    def test_disabling_step_and_epoch_schedules_saves_fixed_learning_rates(self):
        from lnl_toolbox.catalog import load_yaml

        cases = (
            ("l2rw-cifar10-reproduction", [
                {"path": "scheduler.step_milestones", "value": []},
                {"path": "optimizer.lr", "value": 0.05},
            ]),
            ("gce-cifar10-noise02-reproduction", [
                {"path": "scheduler.name", "value": "none"},
                {"path": "optimizer.lr", "value": 0.005},
            ]),
            ("cwd-cifar10-reproduction", [
                {"path": "scheduler.name", "value": "none"},
                {"path": "optimizer.lr", "value": 0.004},
            ]),
            ("cal-cifar10-reproduction", [
                {"path": "scheduler.name", "value": "none"},
                {"path": "optimizer.lr", "value": 0.03},
            ]),
        )
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            for recipe, patches in cases:
                destination = Path(directory) / (recipe + ".yaml")
                command_console._save_config({
                    "path": str(destination), "recipe": recipe, "patches": patches,
                    "acknowledge_paper_impact": True,
                })
                config = load_yaml(destination)
                self.assertEqual(config["optimizer"]["lr"], patches[-1]["value"])
                if recipe.startswith("l2rw"):
                    self.assertEqual(config["scheduler"]["step_milestones"], [])
                else:
                    self.assertEqual(config["scheduler"]["name"], "none")

    def test_yaml_editor_uses_registry_numeric_types_and_preserves_nullable_values(self):
        schema = command_console._config_schema("gce-cifar10-noise02-reproduction")
        fields = {field["path"]: field for field in schema["fields"]}
        self.assertEqual(fields["data.max_train_samples"]["kind"], "number")
        self.assertTrue(fields["data.max_train_samples"]["nullable"])
        self.assertEqual(fields["data.name"]["kind"], "text")

        source = command_console._config_payload("gce-cifar10-noise02-reproduction")
        broken = source["content"].replace("max_train_samples: null", "max_train_samples: ''")
        broken = broken.replace("max_validation_samples: null", "max_validation_samples: ''")
        broken = broken.replace("max_test_samples: null", "max_test_samples: ''")
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            destination = Path(directory) / "gce-numeric-round-trip.yaml"
            saved = command_console._save_config(
                {
                    "path": str(destination),
                    "recipe": "gce-cifar10-noise02-reproduction",
                    "source_path": source["path"],
                    "content": broken,
                    "overwrite": False,
                }
            )
            from lnl_toolbox.catalog import load_yaml

            config = load_yaml(command_console.ROOT / saved["path"])
            self.assertIsNone(config["data"]["max_train_samples"])
            self.assertIsNone(config["data"]["max_validation_samples"])
            self.assertIsNone(config["data"]["max_test_samples"])

            numeric = command_console._save_config(
                {
                    "path": str(destination),
                    "recipe": "gce-cifar10-noise02-reproduction",
                    "source_path": source["path"],
                    "patches": [
                        {"path": "data.max_train_samples", "value": 3000},
                        {"path": "data.max_validation_samples", "value": ""},
                    ],
                    "overwrite": True,
                }
            )
            config = load_yaml(command_console.ROOT / numeric["path"])
            self.assertEqual(config["data"]["max_train_samples"], 3000)
            self.assertIsNone(config["data"]["max_validation_samples"])
            self.assertEqual(config["data"]["name"], "cifar10")
            # Free text stays text; enumerated dataset names must be selected.
            self.assertEqual(command_console._coerce_patch_value({"kind": "text", "path": "noise.path", "label": "标签文件"}, "123"), "123")
            with self.assertRaisesRegex(ValueError, "下拉"):
                command_console._save_config({
                    "path": str(destination),
                    "recipe": "gce-cifar10-noise02-reproduction",
                    "source_path": source["path"],
                    "patches": [{"path": "data.name", "value": "123"}],
                    "overwrite": True,
                })

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

    def test_schema_uses_revised_permissions_and_keeps_structural_wiring_locked(self):
        schema = command_console._config_schema("fine-cifar100n-reproduction")
        paths = {field["path"] for field in schema["fields"]}
        self.assertIn("fine.warmup_epochs", paths)
        self.assertIn("model.name", paths)
        self.assertIn("execution.runner", paths)
        fields = {field["path"]: field for field in schema["fields"]}
        self.assertEqual(fields["model.name"]["level"], "advanced")
        self.assertTrue(fields["model.name"]["editable"])
        self.assertEqual(fields["execution.runner"]["level"], "locked")
        self.assertFalse(fields["execution.runner"]["editable"])
        self.assertEqual(
            [level["id"] for level in schema["levels"]],
            ["basic", "paper", "advanced", "locked"],
        )

        cdr_schema = command_console._config_schema("cifar10-symmetric-cdr-reproduction")
        cdr_paths = {field["path"] for field in cdr_schema["fields"]}
        self.assertIn("parameter_update.noise_rate", cdr_paths)
        self.assertIn("parameter_update.name", cdr_paths)

        binary_schema = command_console._config_schema(
            "binary-risk-natarajan-reproduction"
        )
        binary_fields = {field["path"]: field for field in binary_schema["fields"]}
        for path in ("noise.rho_positive", "noise.rho_negative"):
            self.assertEqual(binary_fields[path]["level"], "paper")
            self.assertTrue(binary_fields[path]["editable"])

    def test_revised_registry_is_the_only_active_web_policy(self):
        self.assertEqual(
            command_console._PARAMETER_REGISTRY_PATH.name,
            "lnl_parameter_metadata_registry_revised.yaml",
        )
        registry = command_console._parameter_registry()
        self.assertIn("permission_policy_revision", registry)
        self.assertEqual(
            registry["formal_recipe_bindings"]["t_revision"],
            "cifar10-t-revision-sym20-reproduction",
        )
        self.assertEqual(
            registry["formal_recipe_bindings"]["ca2c"],
            "ca2c-cifar100-reproduction",
        )

    def test_parameter_display_policy_separates_runtime_fields_and_resources(self):
        schema = command_console._config_schema("gce-cifar10-noise02-reproduction")
        fields = {field["path"]: field for field in schema["fields"]}
        self.assertFalse(fields["loader.num_workers"]["visible"])
        self.assertFalse(fields["loader.pin_memory"]["visible"])
        self.assertFalse(fields["trainer.device"]["visible"])
        self.assertFalse(fields["data.max_train_samples"]["visible"])
        self.assertTrue(fields["data.name"]["visible"])
        dld_schema = command_console._config_schema("dld-cifar10-reproduction")
        resource_fields = [field for field in dld_schema["fields"] if field["presentation"] == "resource"]
        self.assertTrue(resource_fields)
        self.assertTrue(all(field["research_category"] == "resource" for field in resource_fields))
        # A noise realization changes the data condition, not the runtime
        # implementation. It must be discoverable in the data/noise section.
        self.assertEqual(fields["noise.seed"]["research_category"], "")
        self.assertEqual(fields["noise.seed"]["research_group"], "")

        policy = command_console._parameter_registry()["parameter_display_policy"]
        for field in fields.values():
            path = field["path"]
            expected_presentation, _ = command_console._parameter_presentation(path, "gce")
            expected_visible = expected_presentation != "hidden" or (
                field["level"] == "locked" and path.endswith(".name")
            )
            self.assertEqual(field["visible"], expected_visible, path)
            self.assertEqual(field["presentation"], expected_presentation, path)
        self.assertEqual(command_console._parameter_presentation("data.root", "gce")[0], "hidden")

    def test_recipe_compatibility_requests_share_one_computation(self):
        command_console._invalidate_dataset_recipe_compatibility()
        value = {"dataset": "lab", "recipes": [], "methods": []}
        with mock.patch.object(
            command_console,
            "_compute_dataset_recipe_compatibility_payload",
            return_value=value,
        ) as compute:
            first = command_console._dataset_recipe_compatibility_payload("lab")
            second = command_console._dataset_recipe_compatibility_payload("lab")
        self.assertEqual(first, value)
        self.assertEqual(second, value)
        compute.assert_called_once_with("lab", method_noise_rate_prior=None)
        command_console._invalidate_dataset_recipe_compatibility("lab")

    def test_all_formal_paper_recipes_have_complete_registry_schemas(self):
        from lnl_toolbox.catalog import default_paper_config, load_papers

        papers = load_papers(command_console.ROOT)
        self.assertEqual(len(papers), 26)
        registry = command_console._parameter_registry()
        research_view = registry["research_parameter_view"]
        default_paths = registry["default_parameter_paths"]
        display_policy = registry["parameter_display_policy"]
        for policy_key in (
            "common_paths_by_method", "selection_paths_by_method",
            "resource_paths_by_method", "hidden_paths_by_method",
        ):
            self.assertEqual(set(display_policy[policy_key]), set(registry["formal_recipe_bindings"]))
        self.assertEqual(sum(map(len, display_policy["common_paths_by_method"].values())), 280)
        self.assertEqual(sum(map(len, display_policy["selection_paths_by_method"].values())), 28)
        self.assertEqual(sum(map(len, display_policy["resource_paths_by_method"].values())), 4)
        self.assertEqual(set(research_view["method_paths"]), set(command_console._parameter_registry()["methods"]))
        self.assertEqual(set(default_paths["methods"]), set(command_console._parameter_registry()["formal_recipe_bindings"]))
        for paper in papers:
            config, _ = default_paper_config(paper, root=command_console.ROOT)
            schema = command_console._config_schema(config.recipe_id)
            self.assertTrue(schema["formal_recipe"], paper.id)
            self.assertEqual(
                [level["id"] for level in schema["levels"]],
                ["basic", "paper", "advanced", "locked"],
                paper.id,
            )
            self.assertTrue(
                {field["level"] for field in schema["fields"]}
                <= {"basic", "paper", "advanced", "locked"},
                paper.id,
            )
            self.assertTrue(
                all(field["note"] for field in schema["fields"] if field["level"] == "paper"),
                paper.id,
            )
            fields = {field["path"]: field for field in schema["fields"]}
            self.assertEqual(len(fields), len(schema["fields"]), paper.id)
            for path in (
                "data.max_train_samples", "data.max_validation_samples", "data.max_test_samples"
            ):
                if path in fields:
                    self.assertFalse(fields[path]["visible"], (paper.id, path))
                    self.assertEqual(fields[path]["presentation"], "hidden", (paper.id, path))
                self.assertNotIn(path, display_policy["common_paths_by_method"][schema["method"]])
            for presentation_key, expected_presentation in (
                ("common_paths_by_method", "common"),
                ("selection_paths_by_method", "selection"),
                ("resource_paths_by_method", "resource"),
                ("hidden_paths_by_method", "hidden"),
            ):
                for path in display_policy[presentation_key][schema["method"]]:
                    self.assertIn(path, fields, (paper.id, path))
                    presentation = "hidden" if path.startswith("trusted_validation.") or path != "seed" and path.rsplit(".", 1)[-1].endswith("seed") and not path.endswith("peer_seed_offset") else expected_presentation
                    self.assertEqual(fields[path]["presentation"], presentation, (paper.id, path))
            selected = (set(default_paths["common"]) | set(default_paths["methods"][schema["method"]])) - set(default_paths.get("exclude", {}).get(schema["method"], ()))
            self.assertTrue(set(default_paths["methods"][schema["method"]]) <= set(fields), paper.id)
            defaults = {path for path, field in fields.items() if field["visible"] and field["display_group"] == "default"}
            self.assertTrue({
                path for path in selected
                if path in fields and fields[path]["editable"]
                and fields[path]["presentation"] != "hidden"
                and fields[path]["kind"] not in {"list", "object"}
                and not isinstance(fields[path]["value"], (list, dict))
            } <= defaults, paper.id)
            for field in fields.values():
                self.assertEqual(field["display_group"], command_console._parameter_display_group(field, schema["method"]), (paper.id, field["path"]))
            self.assertTrue({field["display_group"] for field in fields.values()} <=
                            {"default", "advanced", "restricted"}, paper.id)
            self.assertEqual(
                [group["id"] for group in schema["research_groups"]],
                ["method", "data", "training"],
                paper.id,
            )
            self.assertEqual(
                [category["id"] for category in schema["research_categories"]],
                ["method", "data", "comparison", "method_protocol", "data_protocol", "training_protocol", "runtime", "resource"],
                paper.id,
            )
            for field in fields.values():
                if not field["editable"]:
                    self.assertEqual(field["display_group"], "restricted", (paper.id, field["path"]))
                elif field["presentation"] == "hidden":
                    self.assertEqual(field["display_group"], "advanced", (paper.id, field["path"]))
                if field.get("presentation") == "resource":
                    self.assertEqual(field["research_category"], "resource", (paper.id, field["path"]))
                    continue
                if field["research_group"]:
                    self.assertTrue(field["research_category"], (paper.id, field["path"]))
                else:
                    self.assertFalse(field["research_category"], (paper.id, field["path"]))
            for path in research_view["method_paths"][schema["method"]]:
                self.assertIn(path, fields, (paper.id, path))
                if fields[path]["presentation"] in {"common", "selection", "resource", "hidden"}:
                    self.assertFalse(fields[path]["research_group"], (paper.id, path))
                    continue
                if fields[path]["presentation"] == "resource":
                    self.assertEqual(fields[path]["research_group"], "", (paper.id, path))
                    self.assertEqual(fields[path]["research_category"], "resource", (paper.id, path))
                elif fields[path]["visible"]:
                    self.assertEqual(fields[path]["research_group"], "method", (paper.id, path))
                else:
                    self.assertEqual(fields[path]["research_group"], "", (paper.id, path))
            self.assertTrue(all(not field["research_group"] for field in fields.values() if field["level"] == "locked"), paper.id)

        gce = {field["path"]: field for field in command_console._config_schema("gce-cifar10-noise02-reproduction")["fields"]}
        self.assertEqual(gce["loss.q"]["research_group"], "method")
        self.assertEqual(gce["loss.q"]["research_category"], "method")
        self.assertEqual(gce["loss.q"]["display_group"], "default")
        self.assertEqual(gce["noise.rate"]["presentation"], "common")
        self.assertEqual(gce["model.name"]["presentation"], "selection")
        self.assertEqual(gce["noise.seed"]["presentation"], "hidden")
        self.assertEqual(gce["data.preprocessing"]["presentation"], "common")
        self.assertEqual(gce["seed"]["presentation"], "common")
        self.assertEqual(gce["seed"]["display_group"], "default")
        self.assertTrue(gce["loss.name"]["visible"])
        self.assertFalse(gce["loss.name"]["editable"])
        self.assertEqual(gce["loss.name"]["display_group"], "restricted")
        correction = {
            field["path"]: field
            for field in command_console._config_schema("loss-correction-cifar10-asymmetric04")["fields"]
        }
        self.assertEqual(correction["noise.transition_matrix"]["display_group"], "advanced")
        self.assertEqual(correction["pipeline.warmup_epochs"]["display_group"], "advanced")
        binary = {
            field["path"]: field
            for field in command_console._config_schema("binary-risk-natarajan-reproduction")["fields"]
        }
        self.assertEqual(binary["risk.rho_positive"]["display_group"], "default")
        self.assertEqual(binary["risk.rho_positive"]["linked_fields"], ["noise.rho_positive"])
        self.assertEqual(binary["noise.rho_positive"]["presentation"], "common")
        cal = {
            field["path"]: field
            for field in command_console._config_schema("cal-cifar10-reproduction")["fields"]
        }
        self.assertEqual(cal["noise.name"]["presentation"], "common")
        self.assertEqual(cal["noise.rate"]["presentation"], "common")
        self.assertEqual(command_console._parameter_display_group({
            "path": "loss.q", "kind": "number", "value": 0.7, "editable": True,
        }, "gce"), "default")
        self.assertEqual(command_console._parameter_display_group({
            "path": "method.internal", "kind": "number", "value": 1, "editable": False,
        }, "gce"), "restricted")

    def test_three_display_groups_drive_both_parameter_editors(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("function parameterSections(schema)", page)
        self.assertIn("function parameterDisplayGroup(fieldInfo)", page)
        self.assertIn('id:"default", label:"默认显示", default_expanded:true', page)
        self.assertIn('id:"advanced", label:"高级参数", default_expanded:false', page)
        self.assertIn('id:"restricted", label:"禁止在 Web 修改的专属参数", default_expanded:false', page)
        self.assertEqual(page.count('const sections = parameterSections(schema);'), 2)
        self.assertEqual(page.count('parameterDisplayGroup(fieldInfo) === levelInfo.id'), 2)
        self.assertIn('for (const path of fieldInfo.linked_fields || [])', page)
        self.assertIn('function yamlScheduleGroups(fields)', page)
        self.assertIn('function yamlScheduleHtml(group, fields)', page)
        self.assertIn('data-yaml-schedule-toggle', page)
        self.assertIn('data-yaml-schedule-point', page)
        self.assertIn('data-yaml-schedule-rate', page)

    def test_research_controls_are_explained_without_exposing_trusted_wiring(self):
        bindings = command_console._parameter_registry()["formal_recipe_bindings"]
        for method, recipe in bindings.items():
            fields = {field["path"]: field for field in command_console._config_schema(recipe)["fields"]}
            for path, field in fields.items():
                if path.startswith("trusted_validation."):
                    self.assertFalse(field["visible"], (method, path))
            for path in ("data.augment", "data.validation_size", "model.name", "scheduler.name"):
                if path in fields and fields[path]["editable"]:
                    self.assertEqual(fields[path]["display_group"], "default", (method, path))
        l2rw = {field["path"]: field for field in command_console._config_schema(bindings["l2rw"])["fields"]}
        self.assertEqual(l2rw["trainer.max_steps"]["display_group"], "default")
        self.assertIn("更新次数", l2rw["trainer.max_steps"]["note"])
        self.assertEqual(l2rw["trainer.max_steps"]["value"], 80000)

    def test_paper_split_counts_are_default_editors_with_original_fraction(self):
        bindings = command_console._parameter_registry()["formal_recipe_bindings"]
        split_paths = {"data.validation_size", "data.num_val", "data.num_clean", "warmup.noisy_validation_size"}
        for method, recipe in bindings.items():
            for field in command_console._config_schema(recipe)["fields"]:
                if field["path"] in split_paths and field["visible"]:
                    self.assertEqual(field["display_group"], "default", (method, field["path"]))
                    if method != "importance_reweighting":
                        self.assertIn("split_reference", field, (method, field["path"]))
        l2rw = {field["path"]: field for field in command_console._config_schema(bindings["l2rw"])["fields"]}
        self.assertEqual(l2rw["data.num_val"]["split_reference"], {"count": 5000, "total": 50000})

    def test_adapted_schema_keeps_original_split_fraction(self):
        from copy import deepcopy
        import yaml
        from lnl_toolbox.catalog import load_yaml, recipe_by_id

        recipe = "l2rw-cifar10-reproduction"
        config = deepcopy(load_yaml(recipe_by_id(recipe, command_console.ROOT).config_path))
        config["data"]["num_val"] = 300
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            generated = Path(directory) / "mini-l2rw.yaml"
            generated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            schema = command_console._config_schema(path_value=str(generated), recipe_hint=recipe)
        field = next(field for field in schema["fields"] if field["path"] == "data.num_val")
        self.assertEqual(field["value"], 300)
        self.assertEqual(field["split_reference"], {"count": 5000, "total": 50000})

    def test_all_formal_parameter_labels_and_help_are_concise(self):
        bindings = command_console._parameter_registry()["formal_recipe_bindings"]
        for method, recipe in bindings.items():
            fields = command_console._config_schema(recipe)["fields"]
            for field in fields:
                if field["visible"] is False:
                    continue
                self.assertNotIn(" · ", field["label"], (method, field["path"]))
                self.assertNotIn("可修改的高级实验/实现选项", field["note"], (method, field["path"]))
                self.assertNotIn("但不定义算法身份", field["note"], (method, field["path"]))
        l2rw = {field["path"]: field for field in command_console._config_schema(bindings["l2rw"])["fields"]}
        self.assertEqual(l2rw["trainer.max_steps"]["note"], "训练最多执行的参数更新次数。")
        self.assertNotIn("trainer.epochs", l2rw["trainer.max_steps"]["note"])
        gce = {field["path"]: field for field in command_console._config_schema(bindings["gce"])["fields"]}
        self.assertEqual(gce["model.name"]["label"], "model.name（主模型）")
        self.assertEqual(gce["data.augment"]["label"], "data.augment（训练数据增强）")

    def test_parameter_stage_context_is_consistent_across_all_papers(self):
        bindings = command_console._parameter_registry()["formal_recipe_bindings"]
        self.assertEqual(len(bindings), 26)
        for method, recipe in bindings.items():
            for field in command_console._config_schema(recipe)["fields"]:
                if not field["visible"]:
                    continue
                with self.subTest(method=method, path=field["path"]):
                    self.assertTrue(field["label"].startswith(field["path"]))
                    self.assertNotIn("第一阶段", field["label"])
                    self.assertNotIn("该训练阶段", field["note"])
                    if field["editable"]:
                        self.assertTrue(field["note"], "editable parameter lacks help")
                    context = command_console._parameter_context(field["path"], method)
                    if context:
                        short_name = command_console._parameter_context(field["path"], method, brief=True)
                        self.assertIn(short_name, field["label"])
                        self.assertLessEqual(len(short_name), 8)
                        self.assertNotIn("Stage ", field["label"])
                        self.assertIn(context, field["note"])
        for suffix in ("epochs", "optimizer.lr", "optimizer.momentum",
                       "optimizer.weight_decay", "optimizer.name", "model.base_width"):
            path = "t_revision.stage1." + suffix
            self.assertEqual(command_console._parameter_user_label(path, "t_revision"), path + "（初始分类器）")
            self.assertIn("供初始转移矩阵估计使用", command_console._parameter_user_note(path, "", "t_revision"))
        self.assertIn("方向预测器", command_console._parameter_user_label("dld.diffusion.optimizer.direction.lr", "dld"))
        self.assertIn("噪声预测器", command_console._parameter_user_label("dld.diffusion.optimizer.noise.lr", "dld"))

    def test_official_l2rw_virtual_rate_is_not_web_editable(self):
        from copy import deepcopy
        from lnl_toolbox.catalog import load_yaml, recipe_by_id

        recipe = "l2rw-cifar10-reproduction"
        schema = command_console._config_schema(recipe)
        field = next(field for field in schema["fields"] if field["path"] == "meta.virtual_learning_rate")
        self.assertFalse(field["editable"])
        self.assertEqual(field["display_group"], "restricted")
        self.assertIn("固定为 1", field["lock_reason"])

        config = load_yaml(recipe_by_id(recipe, command_console.ROOT).config_path)
        with self.assertRaisesRegex(ValueError, "固定为 1"):
            command_console._assert_locked_parameters_unchanged(
                config, {**config, "meta": {**config["meta"], "virtual_learning_rate": 0.5}}, "l2rw"
            )
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            destination = str(Path(directory) / "l2rw-edited.yaml")
            with self.assertRaisesRegex(ValueError, "锁定参数"):
                command_console._save_config({
                    "recipe": recipe, "path": destination,
                    "patches": [{"path": "meta.virtual_learning_rate", "value": 0.5}],
                })
            edited = deepcopy(config)
            edited["meta"]["virtual_learning_rate"] = 0.5
            import yaml

            with self.assertRaisesRegex(ValueError, "固定为 1"):
                command_console._save_config({
                    "recipe": recipe, "path": destination,
                    "content": yaml.safe_dump(edited, allow_unicode=True),
                })
        paper_variant = deepcopy(config)
        paper_variant["meta"]["implementation"] = "paper"
        paper_variant["meta"]["virtual_learning_rate"] = 0.5
        fields, _ = command_console._registry_config_fields(paper_variant, "l2rw")
        self.assertTrue(next(field for field in fields if field["path"] == "meta.virtual_learning_rate")["editable"])

    def test_runtime_fixed_parameters_are_readonly_across_formal_recipes(self):
        from copy import deepcopy
        from lnl_toolbox.catalog import load_yaml, recipe_by_id

        registry = command_console._parameter_registry()
        fixed_by_method = registry["runtime_fixed_parameter_paths"]
        for method, paths in fixed_by_method.items():
            recipe = registry["formal_recipe_bindings"][method]
            config = load_yaml(recipe_by_id(recipe, command_console.ROOT).config_path)
            fields = {field["path"]: field for field in command_console._config_schema(recipe)["fields"]}
            for path in paths:
                self.assertIn(path, fields, (method, path))
                self.assertFalse(fields[path]["editable"], (method, path))
                self.assertEqual(fields[path]["display_group"], "restricted", (method, path))
            changed = deepcopy(config)
            path = paths[0]
            original = fields[path]["value"]
            replacement = not original if isinstance(original, bool) else (
                original + 1 if isinstance(original, (int, float)) else "__invalid_change__"
            )
            command_console._set_config_path(changed, path, replacement)
            with self.assertRaisesRegex(ValueError, "锁定参数"):
                command_console._assert_locked_parameters_unchanged(config, changed, method)

        cdr = {field["path"]: field for field in command_console._config_schema(
            registry["formal_recipe_bindings"]["cdr"]
        )["fields"]}
        for path in ("optimizer.momentum", "optimizer.weight_decay"):
            self.assertFalse(cdr[path]["editable"])
        cdr_config = load_yaml(recipe_by_id(
            registry["formal_recipe_bindings"]["cdr"], command_console.ROOT
        ).config_path)
        changed_cdr = deepcopy(cdr_config)
        changed_cdr["optimizer"]["momentum"] = 0.9
        with self.assertRaisesRegex(ValueError, "optimizer.momentum"):
            command_console._assert_locked_parameters_unchanged(
                cdr_config, changed_cdr, "cdr"
            )

    def test_noise_rate_updates_derived_method_fields(self):
        from copy import deepcopy
        from lnl_toolbox.catalog import load_yaml

        registry = command_console._parameter_registry()
        for method in ("coteaching", "cnlcu"):
            recipe = registry["formal_recipe_bindings"][method]
            fields = {field["path"]: field for field in command_console._config_schema(recipe)["fields"]}
            for path in registry["runtime_derived_parameter_paths"][method]:
                self.assertFalse(fields[path]["editable"])
            with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
                saved = command_console._save_config({
                    "recipe": recipe,
                    "path": str(Path(directory) / f"{method}-rate.yaml"),
                    "patches": [{"path": "noise.rate", "value": 0.3}],
                    "acknowledge_paper_impact": True,
                })
                config = load_yaml(command_console.ROOT / saved["path"])
                self.assertAlmostEqual(config[method]["noise_rate"], 0.3)
                self.assertAlmostEqual(config[method]["remember_schedule"]["end"], 0.7)
                changed = deepcopy(config)
                changed[method]["remember_schedule"]["end"] = 0.5
                with self.assertRaisesRegex(ValueError, "remember_schedule.end"):
                    command_console._assert_locked_parameters_unchanged(
                        config, changed, method
                    )

    def test_web_save_rejects_model_name_that_would_fail_at_training(self):
        recipe = command_console._parameter_registry()["formal_recipe_bindings"]["coteaching"]
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            destination = str(Path(directory) / "unknown-model.yaml")
            with self.assertRaisesRegex(ValueError, "下拉"):
                command_console._save_config({
                    "recipe": recipe, "path": destination,
                    "patches": [{"path": "model.name", "value": "__unknown_model__"}],
                })

    def test_importance_reweighting_fixed_and_derived_dimensions(self):
        recipe = command_console._parameter_registry()["formal_recipe_bindings"]["importance_reweighting"]
        fields = {field["path"]: field for field in command_console._config_schema(recipe)["fields"]}
        self.assertFalse(fields["data.dimension"]["editable"])
        self.assertFalse(fields["model.in_features"]["editable"])
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            with self.assertRaisesRegex(ValueError, "锁定参数"):
                command_console._save_config({
                    "recipe": recipe, "path": str(Path(directory) / "wrong-dimension.yaml"),
                    "patches": [{"path": "data.dimension", "value": 3}],
                })

    def test_web_save_rejects_unknown_dataset_and_noise_names(self):
        recipe = "gce-cifar10-noise02-reproduction"
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            for path, error in (
                ("data.name", "下拉"),
                ("noise.name", "下拉"),
            ):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, error):
                    command_console._save_config({
                        "recipe": recipe, "path": str(Path(directory) / "invalid.yaml"),
                        "patches": [{"path": path, "value": "__unknown__"}],
                        "acknowledge_paper_impact": True,
                    })

    def test_web_save_checks_changed_optimizer_and_scheduler_builders(self):
        bindings = command_console._parameter_registry()["formal_recipe_bindings"]
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            for recipe, path, expected in (
                (bindings["coteaching"], "optimizer.name", "下拉"),
                (bindings["l2rw"], "scheduler.name", "下拉"),
            ):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, expected):
                    command_console._save_config({
                        "recipe": recipe, "path": str(Path(directory) / "invalid.yaml"),
                        "patches": [{"path": path, "value": "__unknown__"}],
                    })

    def test_loss_correction_matrices_do_not_show_false_paper_deviation(self):
        schema = command_console._config_schema("loss-correction-cifar10-asymmetric04")
        self.assertFalse(schema["modified_from_paper"])
        fields = {field["path"]: field for field in schema["fields"]}
        for path in ("noise.transition_matrix", "pipeline.transition_estimator.matrix"):
            field = fields[path]
            self.assertEqual(field["kind"], "list")
            self.assertEqual(field["value"], field["changed_from"])
            self.assertFalse(field["changed_from_paper"])
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            saved = command_console._save_config({
                "recipe": "loss-correction-cifar10-asymmetric04",
                "path": str(Path(directory) / "unchanged-loss-correction.yaml"),
                "patches": [],
            })
            from lnl_toolbox.catalog import load_yaml

            config = load_yaml(command_console.ROOT / saved["path"])
            self.assertFalse(config["meta"]["web_parameter_record"]["modified_from_paper"])

    def test_adapted_parameter_schema_tolerates_missing_formal_fields(self):
        from copy import deepcopy
        from lnl_toolbox.catalog import load_yaml, recipe_by_id

        config = deepcopy(load_yaml(recipe_by_id("gce-cifar10-noise02-reproduction", command_console.ROOT).config_path))
        del config["loss"]["q"]
        fields, _ = command_console._registry_config_fields(config, "gce", allow_missing=True)
        self.assertNotIn("loss.q", {field["path"] for field in fields})
        recipe, changes = command_console._paper_parameter_changes(config, "gce")
        self.assertEqual(recipe, "gce-cifar10-noise02-reproduction")
        self.assertIn("loss.q", {change["path"] for change in changes})

    def test_quick_start_project_config_keeps_paper_parameter_controls(self):
        from lnl_toolbox.catalog import load_yaml, recipe_by_id

        source = recipe_by_id("gce-cifar10-noise02-reproduction", command_console.ROOT).config_path
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            generated = Path(directory) / "generated.yaml"
            generated.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            schema = command_console._config_schema(
                path_value=str(generated), recipe_hint="gce-cifar10-noise02-reproduction"
            )
            fields = {field["path"]: field for field in schema["fields"]}
            self.assertEqual(fields["loss.q"]["display_group"], "default")
            saved = command_console._save_config({
                "source_path": str(generated),
                "recipe_hint": "gce-cifar10-noise02-reproduction",
                "path": str(Path(directory) / "edited.yaml"),
                "patches": [{"path": "loss.q", "value": 0.5}],
                "acknowledge_paper_impact": True,
            })
            config = load_yaml(command_console.ROOT / saved["path"])
            self.assertEqual(config["loss"]["q"], 0.5)
            self.assertTrue(config["meta"]["web_parameter_record"]["modified_from_paper"])

    def test_paper_parameter_change_requires_acknowledgement_and_is_recorded(self):
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            destination = Path(directory) / "gce-modified.yaml"
            request_payload = {
                "path": str(destination),
                "recipe": "gce-cifar10-noise02-reproduction",
                "patches": [{"path": "loss.q", "value": 0.5}],
            }
            with self.assertRaisesRegex(ValueError, "需要确认复现影响"):
                command_console._save_config(request_payload)
            request_payload["acknowledge_paper_impact"] = True
            saved = command_console._save_config(request_payload)
            from lnl_toolbox.catalog import load_yaml

            config = load_yaml(command_console.ROOT / saved["path"])
            record = config["meta"]["web_parameter_record"]
            self.assertTrue(record["modified_from_paper"])
            self.assertEqual(record["effective_reproduction_status"], "modified_from_paper")
            self.assertEqual(record["paper_parameter_changes"][0]["path"], "loss.q")
            self.assertNotIn("parameter_record", config)

    def test_web_paper_metadata_does_not_replace_training_parameter_record(self):
        source = command_console._config_payload(
            "binary-risk-natarajan-reproduction"
        )["content"]
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            destination = Path(directory) / "binary-copy.yaml"
            saved = command_console._save_config(
                {
                    "path": str(destination),
                    "recipe": "binary-risk-natarajan-reproduction",
                    "content": source,
                }
            )
            from lnl_toolbox.catalog import load_yaml
            from lnl_toolbox.core.hyperparameters import resolve_parameter_sampling

            config = load_yaml(command_console.ROOT / saved["path"])
            resolved, record = resolve_parameter_sampling(config)
        self.assertIsNone(record)
        self.assertNotIn("parameter_record", resolved)
        self.assertEqual(
            resolved["meta"]["web_parameter_record"]["formal_recipe"],
            "binary-risk-natarajan-reproduction",
        )

    def test_complete_yaml_edit_cannot_bypass_locked_parameter_policy(self):
        source = command_console._config_payload("fine-cifar100n-reproduction")
        changed = source["content"].replace("runner: fine", "runner: clean", 1)
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            with self.assertRaisesRegex(ValueError, "锁定参数"):
                command_console._save_config(
                    {
                        "path": str(Path(directory) / "bypass.yaml"),
                        "recipe": "fine-cifar100n-reproduction",
                        "source_path": source["path"],
                        "content": changed,
                        "acknowledge_paper_impact": True,
                    }
                )

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

    def test_yaml_save_rechecks_selected_dataset_compatibility(self):
        compatibility = mock.Mock()
        compatibility.status.value = "incompatible"
        compatibility.reasons = (
            mock.Mock(code="wrong_class_count", message="requires 100 classes"),
        )
        service = mock.Mock()
        service.list_config_compatibility.return_value = (("candidate", compatibility),)
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory, mock.patch(
            "lnl_toolbox.training.service.ExperimentService", return_value=service
        ):
            destination = Path(directory) / "blocked.yaml"
            with self.assertRaisesRegex(ValueError, "与当前配置不兼容"):
                command_console._save_config({
                    "path": str(destination),
                    "recipe": "fine-cifar100n-reproduction",
                    "patches": [],
                    "dataset_alias": "local-cifar10",
                })
            self.assertFalse(destination.exists())

    def test_unknown_command_is_rejected(self):
        with self.assertRaises(KeyError):
            command_console.build_command("not-allowed")

    def test_free_command_is_parsed_without_shell(self):
        command = command_console.parse_free_command(
            'lnl run --recipe "cifar10-clean-smoke" --epochs 1'
        )
        self.assertEqual(command[:3], [sys.executable, "-m", "lnl_toolbox.cli.main"])
        self.assertIn("--epochs", command)
        self.assertIn("1", command)

    def test_free_command_rejects_shell_syntax(self):
        for raw in ("python evil.py", "lnl doctor | more", "lnl doctor > out.txt"):
            with self.assertRaises(ValueError):
                command_console.parse_free_command(raw)

    def test_command_always_uses_current_python(self):
        self.assertEqual(
            command_console.resolve_lnl_command(),
            [sys.executable, "-m", "lnl_toolbox.cli.main"],
        )

    def test_command_builder_does_not_use_shell(self):
        command = command_console.build_command("train-one")
        self.assertEqual(command[:3], [sys.executable, "-m", "lnl_toolbox.cli.main"])
        self.assertIn("--epochs", command)
        self.assertIn("1", command)

    def test_job_error_is_rendered_in_web_output(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('output + "\\n\\n启动错误：" + job.error', page)
        self.assertIn('job.structured != null && !job.error', page)

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
        self.assertIsNone(payload["training"])
        json.dumps(payload)

    def test_terminal_process_poll_releases_main_stop_state(self):
        process = mock.Mock()
        process.poll.return_value = 0
        job = command_console.Job(
            job_id="exited",
            key="doctor",
            command=["lnl", "doctor"],
            display_command="lnl doctor",
            process=process,
            cancel_requested=True,
        )
        payload = command_console._job_payload(job)
        self.assertFalse(payload["running"])
        self.assertEqual(payload["returncode"], 0)

    def test_main_console_exposes_process_list(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        for marker in (
            'id="console-job-list"', 'id="refresh-console-jobs"',
            'id="toggle-console-jobs"', 'aria-expanded="false"',
            'id="console-job-list-items"', 'function setConsoleJobListExpanded', 'function refreshConsoleJobList',
            'function renderConsoleJobList', 'fetch("/api/jobs")',
        ):
            self.assertIn(marker, page)
        self.assertTrue(callable(command_console.job_list_payload))

    def test_training_job_payload_contains_best_effort_snapshot(self):
        from web.training_status import TrainingContext

        job = command_console.Job(
            job_id="training",
            key="custom",
            command=["lnl", "run"],
            display_command="lnl run",
            lines=['{"event":"epoch","epoch":1}'],
            training_context=TrainingContext(None, None, "run", 2),
        )
        payload = command_console._job_payload(job)
        self.assertEqual(payload["training"]["completed_epoch"], 1)
        self.assertEqual(payload["training"]["total_epochs"], 2)

    def test_web_training_without_output_receives_a_unique_run_directory(self):
        with mock.patch.object(command_console.subprocess, "Popen") as popen:
            popen.return_value.stdout = None
            job = command_console._start_process(
                "custom", ["lnl", "run", "--recipe", "cifar10-clean-smoke"], "lnl run --recipe cifar10-clean-smoke"
            )
        self.assertIn("--output-dir", job.command)
        self.assertIn("artifacts/web-runs/", job.display_command.replace("\\", "/"))
        self.assertIsNotNone(job.training_context)

    def test_completed_job_polling_returns_without_reentrant_lock_deadlock(self):
        job = command_console.Job(
            job_id="tutorial-complete",
            key="doctor",
            command=["lnl", "doctor"],
            display_command="lnl doctor",
            lines=["ok"],
            returncode=0,
        )
        with command_console.JOBS_LOCK:
            command_console.JOBS[job.job_id] = job
        server = command_console.ThreadingHTTPServer(
            ("127.0.0.1", 0), command_console.ConsoleHandler
        )
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with request.urlopen(
                f"{base}/api/jobs/{job.job_id}", timeout=2
            ) as response:
                payload = json.loads(response.read())
            self.assertFalse(payload["running"])
            self.assertEqual(payload["returncode"], 0)
            self.assertIsNone(payload["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            with command_console.JOBS_LOCK:
                command_console.JOBS.pop(job.job_id, None)

    def test_tutorial_success_handler_advances_progress_and_unlocks_next_step(self):
        page = (command_console.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn(
            'state.tutorialCompleted[stepId] = !job.error && job.returncode === 0 ? "passed" : "failed";',
            page,
        )
        self.assertIn(
            'state.timer = setInterval(pollJob, 350)',
            page,
        )
        self.assertIn(
            '["passed", "not-needed"].includes(tutorialStatus(active.id))',
            page,
        )
        self.assertIn(
            'const doneCount = steps.filter(function (step) { return ["passed", "not-needed"].includes(tutorialStatus(step.id)); }).length;',
            page,
        )

    def test_cancel_job_terminates_running_web_child(self):
        process = mock.Mock()
        process.poll.return_value = None
        job = command_console.Job(
            job_id="cancel-me",
            key="custom",
            command=["python", "-c", ""],
            display_command="python -c ...",
            process=process,
        )
        with command_console.JOBS_LOCK:
            command_console.JOBS[job.job_id] = job
        try:
            with mock.patch.object(command_console.subprocess, "run") as taskkill:
                result = command_console.cancel_job(job.job_id)
            if command_console.os.name == "nt":
                taskkill.assert_called_once_with(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=command_console.subprocess.DEVNULL,
                    stderr=command_console.subprocess.DEVNULL,
                    check=False,
                )
                process.terminate.assert_not_called()
            else:
                taskkill.assert_not_called()
                process.terminate.assert_called_once_with()
            self.assertTrue(result.cancel_requested)
        finally:
            with command_console.JOBS_LOCK:
                command_console.JOBS.pop(job.job_id, None)


if __name__ == "__main__":
    unittest.main()
