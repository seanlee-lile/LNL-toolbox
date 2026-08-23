from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from lnl_toolbox import catalog as catalog_module
from lnl_toolbox.catalog import (
    discover_recipes,
    default_paper_config,
    find_project_root,
    load_papers,
    load_yaml,
    load_recipe_config,
    recipe_by_id,
    resolve_config_paths,
    select_paper_config,
    validate_config,
)
from lnl_toolbox.cli.main import main
from lnl_toolbox.data.profile import (
    DatasetProfile,
    KnowledgeState,
    Modality,
    NoiseKnowledge,
)
from lnl_toolbox.training.compatibility import (
    CompatibilityReason,
    CompatibilityResult,
    CompatibilityStatus,
)
from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE, DatasetStatusReport
from lnl_toolbox.training.runners import resolve_runner, runner_names


ROOT = Path(__file__).resolve().parents[1]


class RunnerResolutionTest(unittest.TestCase):
    def test_all_public_runners_are_registered(self) -> None:
        self.assertEqual(
            set(runner_names()),
            {
                "binary", "cal", "ca2c", "clean", "coteaching", "cwd", "dld", "dual_t", "fine",
                "importance_reweighting", "instance_transition", "multi_model",
                "l2rw", "lend", "mc_ldce", "pcse", "supervised", "cnlcu", "dividemix",
                "t_revision", "upm", "volmin", "volminnet",
            },
        )

    def test_unknown_method_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown method"):
            resolve_runner({"method": "cotaching", "data": {"name": "cifar10"}})

    def test_dedicated_config_cannot_use_supervised_runner(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_runner(
                {
                    "execution": {"runner": "supervised"},
                    "data": {"name": "cifar100"},
                    "fine": {"warmup_epochs": 1},
                }
            )

    def test_legacy_special_sections_are_inferred(self) -> None:
        self.assertEqual(resolve_runner({"data": {}, "cwd": {}}).name, "cwd")
        self.assertEqual(
            resolve_runner({"data": {}, "algorithm": {"name": "jocor"}}).name,
            "multi_model",
        )


class CatalogTest(unittest.TestCase):
    def test_installed_recipe_uses_distribution_file_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "venv"
            site_packages = prefix / "lib" / "site-packages"
            site_packages.mkdir(parents=True)
            relative = "configs/experiment/cifar10_volminnet_smoke.yaml"
            entry = Path("../..") / "share" / "lnl-toolbox" / relative

            class Distribution:
                files = (entry,)

                @staticmethod
                def locate_file(value: Path) -> Path:
                    return site_packages / value

            with patch.object(
                catalog_module.metadata,
                "distribution",
                return_value=Distribution(),
            ):
                installed = catalog_module._installed_recipe_path(relative)
            self.assertEqual(
                installed,
                (prefix / "share" / "lnl-toolbox" / relative).resolve(),
            )

    def test_every_builtin_recipe_has_explicit_valid_runner(self) -> None:
        recipes = discover_recipes(ROOT)
        self.assertGreaterEqual(len(recipes), 39)
        for recipe in recipes:
            config = load_recipe_config(recipe)
            self.assertIn("execution", config, recipe.id)
            self.assertEqual(validate_config(config).name, recipe.runner, recipe.id)

    def test_paper_catalog_only_references_runnable_recipes(self) -> None:
        recipes = {
            item.id
            for item in discover_recipes(ROOT, include_conditional=True)
        }
        papers = load_papers(ROOT)
        self.assertGreaterEqual(len(papers), 10)
        for paper in papers:
            for item in paper.configs:
                self.assertIn(item.recipe_id, recipes)

    def test_every_paper_has_one_formal_default(self) -> None:
        papers = load_papers(ROOT)
        self.assertEqual(len(papers), 26)
        defaults = [default_paper_config(paper, root=ROOT) for paper in papers]
        self.assertEqual(len(defaults), 26)
        self.assertTrue(all(config.profile == "reproduction" for config, _ in defaults))
        self.assertEqual(len({paper.id for paper in papers}), 26)

    def test_manifest_excludes_local_untracked_yaml_and_conditional_recipes(self) -> None:
        recipes = {item.id for item in discover_recipes(ROOT)}
        self.assertNotIn("cifar10-symmetric40-all-e5", recipes)
        self.assertNotIn("cifar10-symmetric40-small-loss-e5", recipes)
        self.assertFalse(any("mentornet" in recipe for recipe in recipes))
        self.assertNotIn("cifar10-pcse-reproduction", recipes)
        all_recipes = {
            item.id
            for item in discover_recipes(ROOT, include_conditional=True)
        }
        self.assertIn("mentornet-dd-cifar100-symmetric04-smoke", all_recipes)
        self.assertIn("cifar10-pcse-reproduction", all_recipes)

    def test_public_recipe_catalog_is_small_without_hiding_internal_lookup(self) -> None:
        public = discover_recipes(ROOT, public_only=True)
        self.assertEqual(len(public), 4)
        self.assertTrue(all(item.visibility == "public" for item in public))
        public_ids = {item.id for item in public}
        self.assertIn("cifar10-clean-smoke", public_ids)
        self.assertNotIn("fine-cifar100n-reproduction", public_ids)
        self.assertEqual(
            recipe_by_id("fine-cifar100n-reproduction", ROOT).id,
            "fine-cifar100n-reproduction",
        )

    def test_method_specific_preflight_and_conditional_artifact(self) -> None:
        cnlcu = load_recipe_config(
            next(item for item in discover_recipes(ROOT) if item.id == "cifar10-cnlcu-soft-smoke")
        )
        self.assertEqual(validate_config(cnlcu).name, "cnlcu")
        mentor = load_recipe_config(
            next(
                item
                for item in discover_recipes(ROOT, include_conditional=True)
                if item.id == "mentornet-dd-cifar100-symmetric04-smoke"
            )
        )
        mentor["pipeline"]["weight_provider"]["artifact_path"] = str(
            ROOT / "data/mentornet/missing-artifact-for-test.pt"
        )
        with self.assertRaisesRegex(ValueError, "conditional.*MentorArtifact"):
            validate_config(resolve_config_paths(mentor, ROOT))
        pcse = load_recipe_config(
            next(
                item
                for item in discover_recipes(ROOT, include_conditional=True)
                if item.id == "cifar10-pcse-reproduction"
            )
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LNL_PCSE_SOURCE_RUN", None)
            with self.assertRaisesRegex(
                ValueError, "source environment variable is not set"
            ):
                validate_config(resolve_config_paths(pcse, ROOT))

    def test_multiple_paper_variants_require_selection(self) -> None:
        paper = next(item for item in load_papers(ROOT) if item.id == "apl")
        with self.assertRaisesRegex(ValueError, "choose --variant"):
            select_paper_config(paper, profile="reproduction", root=ROOT)

    def test_paths_are_resolved_against_project_not_cwd(self) -> None:
        config = {"data": {"name": "cifar10", "root": "data/cifar10"}, "output_root": "artifacts/runs"}
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                import os

                os.chdir(directory)
                resolved = resolve_config_paths(config, ROOT)
            finally:
                os.chdir(previous)
        self.assertEqual(Path(resolved["data"]["root"]), (ROOT / "data/cifar10").resolve())
        self.assertEqual(Path(resolved["output_root"]), (ROOT / "artifacts/runs").resolve())

    def test_packaged_recipe_without_project_uses_caller_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(find_project_root(), Path(directory).resolve())
            finally:
                os.chdir(previous)


class UnifiedCliTest(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def compatibility_profile() -> DatasetProfile:
        return DatasetProfile(
            dataset="fixture", adapter="fixture", source="fixture-root",
            task="classification", modality=Modality.IMAGE, num_classes=3,
            input_shape=(8, 8, 3), channels=3,
            sample_counts_by_split=(("train", 6), ("test", 3)),
            available_splits=("train", "test"),
            class_names=("a", "b", "c"),
            class_distribution_by_split=(("train", (2, 2, 2)), ("test", (1, 1, 1))),
            observed_train_labels=KnowledgeState.AVAILABLE,
            clean_train_labels=KnowledgeState.UNKNOWN,
            clean_validation_labels=KnowledgeState.UNAVAILABLE,
            stable_indices=KnowledgeState.AVAILABLE,
            dataset_fingerprint="d" * 64,
            split_fingerprints=(("train", "a" * 64), ("test", "b" * 64)),
            noise=NoiseKnowledge(),
        )

    def test_data_inspect_exposes_profile_in_human_and_json_formats(self) -> None:
        profile = self.compatibility_profile()
        report = DatasetStatusReport(
            "fixture", "fixture", "ready", location="fixture-root",
            train_samples=6, test_samples=3, classes=3,
            fingerprint=profile.dataset_fingerprint, profile=profile,
        )
        with patch.object(DEFAULT_DATA_SERVICE, "inspect", return_value=report):
            code, output, error = self.invoke("data", "inspect", "fixture")
            self.assertEqual(code, 0, error)
            self.assertIn("Modality         image", output)
            self.assertIn("Clean labels     unknown", output)
            self.assertIn("Profile fingerprint", output)

            code, output, error = self.invoke(
                "data", "inspect", "fixture", "--format", "json"
            )
        self.assertEqual(code, 0, error)
        value = json.loads(output)
        self.assertEqual(value["profile"]["modality"], "image")
        self.assertEqual(value["profile"]["noise"]["rate"]["status"], "unknown")

    def test_data_declare_uses_persisted_service_contract(self) -> None:
        capabilities = Mock()
        capabilities.clean_train_labels = KnowledgeState.AVAILABLE
        capabilities.noise_status.value = "noisy"
        capabilities.noise_origin.value = "native"
        capabilities.noise_rate.status.value = "estimated"
        with patch.object(
            DEFAULT_DATA_SERVICE, "update_declarations", return_value=capabilities
        ) as update:
            code, output, error = self.invoke(
                "data", "declare", "fixture",
                "--clean-train-labels", "available",
                "--noise-status", "noisy",
                "--noise-origin", "native",
                "--noise-rate", "0.2",
                "--noise-rate-status", "estimated",
                "--noise-rate-provenance", "user-audit",
            )
        self.assertEqual(code, 0, error)
        updates = update.call_args.args[1]
        self.assertEqual(updates["clean_train_labels"], "available")
        self.assertEqual(updates["noise_rate"]["status"], "estimated")
        self.assertEqual(updates["noise_rate"]["provenance"], "user-audit")
        self.assertIn("Declarations updated", output)

    def test_methods_compatible_has_grouped_and_machine_readable_results(self) -> None:
        results = (
            CompatibilityResult(CompatibilityStatus.COMPATIBLE, "upm", "fixture"),
            CompatibilityResult(
                CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS,
                "coteaching", "fixture",
                reasons=(CompatibilityReason("requires_noise_rate_prior", "prior required"),),
                required_user_inputs=("noise_rate_prior",),
            ),
            CompatibilityResult(
                CompatibilityStatus.INCOMPATIBLE,
                "importance_reweighting", "fixture",
                reasons=(CompatibilityReason("unsupported_modality", "image unsupported"),),
            ),
        )
        with patch(
            "lnl_toolbox.cli.main.ExperimentService.list_compatible_methods",
            return_value=results,
        ):
            code, output, error = self.invoke(
                "methods", "compatible", "--dataset", "fixture"
            )
            self.assertEqual(code, 0, error)
            self.assertIn("Compatible:", output)
            self.assertIn("Requires additional input:", output)
            self.assertIn("Unavailable:", output)
            self.assertIn("requires_noise_rate_prior", output)

            code, output, error = self.invoke(
                "methods", "compatible", "--dataset", "fixture", "--format", "json"
            )
        self.assertEqual(code, 0, error)
        value = json.loads(output)
        self.assertEqual(value[1]["required_user_inputs"], ["noise_rate_prior"])
        self.assertEqual(value[2]["reason_codes"], ["unsupported_modality"])

    def test_compatibility_failure_prevents_validate_dry_run_and_run(self) -> None:
        error = ValueError(
            "method 'upm' is not ready for dataset 'heart': incompatible; "
            "unsupported_modality: image required"
        )
        with patch(
            "lnl_toolbox.cli.main.ExperimentService.preflight", side_effect=error
        ), patch("lnl_toolbox.cli.main.ExperimentService.run") as runner:
            code, _, stderr = self.invoke(
                "validate", "--recipe", "cifar10-upm-smoke", "--check-data"
            )
            self.assertEqual(code, 2)
            self.assertIn("unsupported_modality", stderr)
            code, _, stderr = self.invoke(
                "run", "--recipe", "cifar10-upm-smoke", "--dry-run"
            )
            self.assertEqual(code, 2)
            self.assertIn("unsupported_modality", stderr)
            code, _, stderr = self.invoke("run", "--recipe", "cifar10-upm-smoke")
            self.assertEqual(code, 2)
            self.assertIn("unsupported_modality", stderr)
        runner.assert_not_called()

    def test_web_command_starts_main_page_and_supports_no_open(self) -> None:
        with patch("lnl_toolbox.cli.main.subprocess.call", return_value=0) as call:
            code, _output, error = self.invoke(
                "web", "--host", "127.0.0.1", "--port", "9000"
            )
        self.assertEqual(code, 0, error)
        command = call.call_args.args[0]
        self.assertIn("command_console.py", command[1])
        self.assertIn("--open", command)
        self.assertIn("9000", command)

        with patch("lnl_toolbox.cli.main.subprocess.call", return_value=0) as call:
            code, _output, error = self.invoke("web", "--no-open")
        self.assertEqual(code, 0, error)
        self.assertNotIn("--open", call.call_args.args[0])

    def test_local_dataset_registration_and_recipe_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cifar100"
            source.mkdir()
            environment = {"LNL_DATA_CATALOG": str(root / "datasets.json")}
            with patch.dict(os.environ, environment, clear=False):
                code, output, error = self.invoke(
                    "data", "register", "lab-cifar100",
                    "--adapter", "cifar100", "--root", str(source),
                )
                self.assertEqual(code, 0, error)
                self.assertIn("state: registered", output)
                self.assertIn("does not prove trainability", output)

                code, output, error = self.invoke("data", "list")
                self.assertEqual(code, 0, error)
                self.assertIn("lab-cifar100", output)
                self.assertIn("incomplete", output)

                code, output, error = self.invoke("data", "status", "lab-cifar100")
                self.assertEqual(code, 0, error)
                self.assertIn("Status           INCOMPLETE", output)

                code, output, error = self.invoke("data", "path", "lab-cifar100")
                self.assertEqual(code, 0, error)
                self.assertIn(str(source.resolve()), output)

                code, output, error = self.invoke(
                    "run", "cifar10-clean-smoke", "--data", "lab-cifar100",
                    "--dry-run", "--no-check-data",
                )
                self.assertEqual(code, 0, error)
                self.assertIn("Dataset: cifar100", output)

                code, output, error = self.invoke(
                    "data", "remove", "lab-cifar100"
                )
                self.assertEqual(code, 0, error)
                self.assertIn("Removed", output)

    def test_local_dataset_registration_requires_adapter_specific_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = {"LNL_DATA_CATALOG": str(Path(directory) / "datasets.json")}
            with patch.dict(os.environ, environment, clear=False):
                code, _, error = self.invoke(
                    "data", "register", "human-noise", "--adapter", "cifar10n",
                    "--root", directory,
                )
                self.assertEqual(code, 2)
                self.assertIn("requires --labels", error)

                source = Path(directory) / "heart.dat"
                rows = []
                for row in range(40):
                    features = [
                        f"{(row + column) % 11 + column / 10:.1f}"
                        for column in range(13)
                    ]
                    rows.append(" ".join(features + [str(1 + row % 2)]))
                source.write_text("\n".join(rows) + "\n", encoding="utf-8")
                code, output, error = self.invoke(
                    "data", "register", "heart", "--adapter", "uci_binary",
                    "--path", str(source),
                )
                self.assertEqual(code, 0, error)
                self.assertIn("heart: uci_binary", output)

                code, output, error = self.invoke(
                    "data", "inspect", "heart", "--format", "json"
                )
                self.assertEqual(code, 0, error)
                inspected = json.loads(output)
                self.assertEqual(inspected["profile"]["modality"], "tabular")
                self.assertEqual(inspected["profile"]["num_classes"], 2)

                code, output, error = self.invoke(
                    "methods", "compatible", "--dataset", "heart", "--format", "json"
                )
                self.assertEqual(code, 0, error)
                compatibility = {item["method"]: item for item in json.loads(output)}
                self.assertEqual(
                    compatibility["importance_reweighting"]["status"], "compatible"
                )
                self.assertEqual(compatibility["upm"]["status"], "incompatible")
                self.assertIn(
                    "unsupported_modality", compatibility["upm"]["reason_codes"]
                )

                code, output, error = self.invoke(
                    "data", "verify", "heart",
                    "--output-dir", str(Path(directory) / "heart-run"),
                    "--project-root", str(ROOT),
                )
                self.assertEqual(code, 0, error)
                self.assertIn("Training check   VERIFIED", output)
                self.assertIn("Train samples", output)
                self.assertIn("completed one-epoch run", output)

    def test_data_verify_uses_automatic_profile_without_a_recipe(self) -> None:
        record = Mock(alias="fashion", adapter="fashion_mnist", signature="a" * 64)
        report = DatasetStatusReport(
            "fashion",
            "fashion_mnist",
            "ready",
            training_evidence={"run_dir": "verify-run"},
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            DEFAULT_DATA_SERVICE, "record", return_value=record
        ), patch.object(
            DEFAULT_DATA_SERVICE,
            "verify",
            return_value=(report, Path(directory)),
        ) as verify:
            destination = Path(directory) / "run"
            code, output, error = self.invoke(
                "data", "verify", "fashion", "--output-dir", str(destination)
            )
        self.assertEqual(code, 0, error)
        verify.assert_called_once_with(
            "fashion", None, destination, recipe=None
        )
        self.assertIn("automatic dataset profile", output)

    def test_list_and_paper_show_are_user_facing(self) -> None:
        code, output, _ = self.invoke("list", "experiments", "--profile", "smoke")
        self.assertEqual(code, 0)
        self.assertIn("个可运行实验（规模=smoke）", output)
        self.assertIn("数据集：", output)
        self.assertIn("执行器：", output)
        self.assertIn("先预览：", output)
        self.assertNotIn("RECIPE | PROFILE", output)

        code, output, _ = self.invoke("papers", "list")
        self.assertEqual(code, 0)
        self.assertIn("dual-t", output)
        self.assertIn("篇具有可运行配置的论文", output)
        self.assertIn("实现保真度：", output)
        self.assertIn("建议先看：", output)
        self.assertNotIn("ID | PAPER", output)
        code, output, _ = self.invoke("papers", "show", "jocor")
        self.assertEqual(code, 0)
        self.assertIn("Toolbox 生命周期", output)
        self.assertIn("论文概念 -> config -> 实现", output)
        self.assertIn("已知差异与限制", output)
        self.assertIn("Recipe：", output)
        self.assertIn("标签来源：", output)
        self.assertIn("Scheduler：", output)
        self.assertIn("生成 symmetric 噪声", output)
        self.assertIn("由执行器决定", output)

        code, output, _ = self.invoke("papers", "show", "dual-t")
        self.assertEqual(code, 0)
        self.assertIn("posterior_stage=", output)

    def test_experiment_list_supports_tsv_for_scripts(self) -> None:
        code, output, _ = self.invoke(
            "list", "experiments", "--profile", "smoke", "--format", "tsv"
        )
        self.assertEqual(code, 0)
        lines = output.splitlines()
        self.assertEqual(
            lines[0], "recipe\tprofile\tdataset\tnoise\tmethod\trunner\tepochs"
        )
        self.assertTrue(all("\t" in line for line in lines[1:]))

    def test_other_catalogs_are_human_readable_and_support_tsv(self) -> None:
        code, output, _ = self.invoke("list", "components", "--kind", "loss")
        self.assertEqual(code, 0)
        self.assertIn("个可组合组件", output)
        self.assertIn("能力：", output)
        self.assertNotIn("KIND | NAME", output)

        code, output, _ = self.invoke(
            "list", "components", "--kind", "loss", "--format", "tsv"
        )
        self.assertEqual(code, 0)
        self.assertEqual(output.splitlines()[0], "kind\tname\tcapabilities\tpaper")

        code, output, _ = self.invoke("papers", "list", "--format", "tsv")
        self.assertEqual(code, 0)
        self.assertEqual(
            output.splitlines()[0],
            "id\tacronym\ttitle\tvenue\tyear\tprofiles\tfidelity\trunners\trecommended",
        )

    def test_paper_config_resolved_does_not_change_source(self) -> None:
        path = ROOT / "configs/experiment/cifar10_coteaching_smoke.yaml"
        before = path.read_bytes()
        code, output, _ = self.invoke(
            "papers", "config", "coteaching", "--profile", "smoke", "--resolved"
        )
        self.assertEqual(code, 0)
        self.assertEqual(path.read_bytes(), before)
        parsed = yaml.safe_load(output)
        self.assertEqual(validate_config(parsed).name, "coteaching")

    def test_dry_run_does_not_create_output(self) -> None:
        with patch("lnl_toolbox.cli.main.ExperimentService.run") as runner:
            code, output, _ = self.invoke(
                "run",
                "--recipe",
                "cifar10-symmetric-ce-smoke",
                "--dry-run",
                "--no-check-data",
            )
        self.assertEqual(code, 0)
        self.assertIn("runner: supervised", output)
        self.assertIn("Dataset: cifar10", output)
        runner.assert_not_called()

    def test_dry_run_checks_data_unless_explicitly_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "missing-data.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {"name": "cifar10", "root": str(root / "missing")},
                        "execution": {"runner": "supervised"},
                    }
                ),
                encoding="utf-8",
            )
            code, _, error = self.invoke("run", str(config_path), "--dry-run")
            self.assertEqual(code, 2)
            self.assertIn("data path does not exist", error)

            with patch("lnl_toolbox.cli.main.ExperimentService.run") as runner:
                code, output, error = self.invoke(
                    "run", str(config_path), "--dry-run", "--no-check-data"
                )
            self.assertEqual(code, 0, error)
            self.assertIn("runner: supervised", output)
            runner.assert_not_called()

            code, _, error = self.invoke("run", str(config_path), "--no-check-data")
            self.assertEqual(code, 2)
            self.assertIn("only valid together with --dry-run", error)

    def test_run_checks_data_before_invoking_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "missing-data.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {"name": "cifar10", "root": str(root / "missing")},
                        "execution": {"runner": "supervised"},
                    }
                ),
                encoding="utf-8",
            )
            with patch("lnl_toolbox.training.experiment.run_experiment") as runner:
                code, _, error = self.invoke(
                    "run",
                    "--config", str(config_path),
                    "--project-root", str(ROOT),
                )
            self.assertEqual(code, 2)
            self.assertIn("data path does not exist", error)
            runner.assert_not_called()

    def test_run_normalizes_final_result_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            run_dir = root / "run"
            config_path = root / "custom.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {"name": "cifar10", "root": str(data_root)},
                        "loss": {"name": "gce"},
                        "execution": {"runner": "supervised"},
                    }
                ),
                encoding="utf-8",
            )

            def fake_run(_config, output_dir, _resume):
                output = Path(output_dir)
                output.mkdir()
                (output / "metrics.jsonl").write_text(
                    json.dumps({"epoch": 1, "test_accuracy": 0.75}) + "\n",
                    encoding="utf-8",
                )
                return output

            with patch(
                "lnl_toolbox.training.runners.RunnerSpec.invoke", side_effect=fake_run
            ), patch("lnl_toolbox.cli.main.ExperimentService.preflight"):
                code, _, error = self.invoke(
                    "run",
                    "--config", str(config_path),
                    "--project-root", str(ROOT),
                    "--output-dir", str(run_dir),
                )
            self.assertEqual(code, 0, error)
            final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(final["test_accuracy"], 0.75)
            self.assertEqual(final["event"], "final")
            self.assertEqual(final["method"], "gce")
            self.assertEqual(final["runner"], "supervised")
            self.assertEqual(final["status"], "completed")
            self.assertIs(final["completed"], True)

    def test_completed_resume_is_strict_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "data": {"name": "cifar10", "root": "missing-data-is-irrelevant"},
                        "execution": {"runner": "supervised"},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "last.pt").write_bytes(b"checkpoint")
            (run_dir / "metrics.jsonl").write_text("original\n", encoding="utf-8")
            (run_dir / "final_metrics.json").write_text(
                json.dumps({"test_accuracy": 0.5}), encoding="utf-8"
            )
            before = {
                path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in run_dir.iterdir()
            }
            with patch("lnl_toolbox.training.experiment.run_experiment") as runner:
                code, output, error = self.invoke("resume", str(run_dir))
            after = {
                path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in run_dir.iterdir()
            }
            self.assertEqual(code, 0, error)
            self.assertIn("resume complete", output)
            self.assertEqual(after, before)
            runner.assert_not_called()

    def test_compose_lists_compatible_slots_and_dedicated_runners(self) -> None:
        code, output, _ = self.invoke("compose", "list", "--runner", "supervised")
        self.assertEqual(code, 0)
        self.assertIn("可组合结构", output)
        self.assertIn("transition_estimator 必须与 risk_corrector 配对", output)
        self.assertIn("regularizer 虽已注册，但尚未接入", output)

        code, output, _ = self.invoke("compose", "list", "--runner", "dual_t")
        self.assertEqual(code, 0)
        self.assertIn("专用生命周期", output)
        self.assertIn("cifar10-dual-t-smoke", output)

    def test_compose_creates_valid_yaml_without_overwriting(self) -> None:
        source = ROOT / "configs/experiment/cifar10_symmetric_ce_smoke.yaml"
        before = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gce_small_loss.yaml"
            code, text, error = self.invoke(
                "compose", "create",
                "--base", "cifar10-symmetric-ce-smoke",
                "--loss", "gce",
                "--selector", "small_loss",
                "--keep-rate", "0.6",
                "--output", str(output),
            )
            self.assertEqual((code, error), (0, ""))
            self.assertIn("已生成新配置", text)
            config = load_yaml(output)
            self.assertEqual(config["loss"], {"name": "gce"})
            self.assertEqual(config["selector"], {"name": "small_loss", "keep_rate": 0.6})
            self.assertEqual(config["execution"], {"runner": "supervised"})

            code, _, error = self.invoke(
                "compose", "create",
                "--base", "cifar10-symmetric-ce-smoke",
                "--output", str(output),
            )
            self.assertEqual(code, 2)
            self.assertIn("refusing to overwrite", error)
        self.assertEqual(source.read_bytes(), before)

    def test_compose_rejects_incompatible_objective_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.yaml"
            code, _, error = self.invoke(
                "compose", "create",
                "--base", "dss-cifar10-symmetric05-smoke",
                "--selector", "small_loss",
                "--keep-rate", "0.5",
                "--output", str(output),
            )
            self.assertEqual(code, 2)
            self.assertIn("DSS requires selector.name='all'", error)
            self.assertFalse(output.exists())

    def test_compose_copies_dedicated_paper_recipe_without_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "volminnet.yaml"
            code, text, error = self.invoke(
                "compose", "create",
                "--base", "volminnet-cifar10-reproduction",
                "--output", str(output),
            )
            self.assertEqual((code, error), (0, ""))
            self.assertIn("论文生命周期：volminnet", text)
            self.assertEqual(
                load_yaml(output),
                load_yaml(ROOT / "configs/experiment/volminnet_cifar10_reproduction.yaml"),
            )

            rejected = Path(directory) / "invalid.yaml"
            code, _, error = self.invoke(
                "compose", "create",
                "--base", "volminnet-cifar10-reproduction",
                "--loss", "ce",
                "--output", str(rejected),
            )
            self.assertEqual(code, 2)
            self.assertIn("only be copied without component overrides", error)
            self.assertFalse(rejected.exists())

    def test_list_experiments_marks_status_and_hides_conditional(self) -> None:
        code, output, _ = self.invoke("list", "experiments", "--profile", "smoke")
        self.assertEqual(code, 0)
        self.assertIn("IMPLEMENTATION", output)
        self.assertIn("cifar10-clean-smoke", output)
        self.assertNotIn("cifar10-cnlcu-soft-smoke", output)
        self.assertNotIn("mentornet", output)

        code, output, _ = self.invoke(
            "list", "experiments", "--profile", "smoke", "--all"
        )
        self.assertEqual(code, 0)
        self.assertIn("cifar10-cnlcu-soft-smoke", output)
        self.assertNotIn("mentornet", output)

    def test_browsing_does_not_import_optional_sklearn(self) -> None:
        script = r'''
import builtins, contextlib, io
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "sklearn" or name.startswith("sklearn."):
        raise ModuleNotFoundError("blocked sklearn")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from lnl_toolbox.cli.main import main
with contextlib.redirect_stdout(io.StringIO()):
    assert main(["list", "experiments"]) == 0
    assert main(["papers", "list"]) == 0
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_volminnet_is_discoverable_and_paper_mapped(self) -> None:
        code, output, _ = self.invoke(
            "list", "experiments", "--profile", "smoke", "--all"
        )
        self.assertEqual(code, 0)
        self.assertIn("cifar10-volminnet-smoke", output)
        code, output, _ = self.invoke("papers", "show", "volminnet")
        self.assertEqual(code, 0)
        self.assertIn("VolMinNet", output)
        self.assertIn("paper_positive_logdet", output)
        code, output, _ = self.invoke(
            "papers", "config", "volminnet", "--profile", "smoke", "--path-only"
        )
        self.assertEqual(code, 0)
        self.assertTrue(output.strip().endswith("cifar10_volminnet_smoke.yaml"))

    def test_volminnet_validate_dry_run_and_epoch_override(self) -> None:
        code, _, error = self.invoke("validate", "--recipe", "cifar10-volminnet-smoke")
        self.assertEqual(code, 0, error)
        code, output, error = self.invoke(
            "run",
            "--recipe",
            "cifar10-volminnet-smoke",
            "--dry-run",
            "--no-check-data",
            "--epochs",
            "3",
        )
        self.assertEqual(code, 0, error)
        self.assertIn("volminnet", output)
        self.assertIn("3", output)

    def test_dividemix_is_discoverable_and_has_staged_epoch_override(self) -> None:
        code, output, error = self.invoke("papers", "show", "dividemix")
        self.assertEqual(code, 0, error)
        self.assertIn("cross-network GMM", output)
        code, output, error = self.invoke(
            "run",
            "--recipe",
            "cifar10-dividemix-smoke",
            "--dry-run",
            "--no-check-data",
            "--epochs",
            "3",
        )
        self.assertEqual(code, 0, error)
        self.assertIn("1/3/4 (warmup/main/total)", output)
        self.assertIn("Models: 2", output)

    def test_positional_source_and_dotted_override(self) -> None:
        code, output, error = self.invoke(
            "run",
            "cifar10-symmetric-ce-smoke",
            "--set",
            "trainer.epochs=2",
            "--dry-run",
            "--no-check-data",
        )
        self.assertEqual(code, 0, error)
        self.assertIn("Training budget: 2", output)

    def test_positional_yaml_is_accepted(self) -> None:
        path = ROOT / "configs/experiment/cifar10_symmetric_ce_smoke.yaml"
        code, output, error = self.invoke("validate", str(path))
        self.assertEqual(code, 0, error)
        self.assertIn("supervised", output)

    def test_compare_and_report_dispatch_to_evaluation_service(self) -> None:
        summary = {
            "group_by": ["method", "noise.rate", "primary_metric.name"],
            "summaries": [
                {
                    "group": {
                        "method": "ce",
                        "noise.rate": 0.2,
                        "primary_metric.name": "test_accuracy",
                    },
                    "metric": "test_accuracy",
                    "n": 1,
                    "mean": 0.8,
                    "std": 0.0,
                    "median": 0.8,
                    "min": 0.8,
                    "max": 0.8,
                }
            ],
            "compatibility": {"model": "consistent"},
            "warnings": [],
            "excluded_runs": [],
            "failed_runs": [],
        }
        with patch("lnl_toolbox.cli.main.compare_runs", return_value=summary):
            code, output, error = self.invoke(
                "compare", str(ROOT), "--group-by", "method,noise.rate"
            )
        self.assertEqual(code, 0, error)
        self.assertIn("METHOD\tNOISE\tMETRIC", output)
        self.assertIn("test_accuracy", output)
        self.assertIn("Compatibility", output)

        leaked = dict(summary, excluded_runs=[{"run_dir": "x", "reason": "leakage"}])
        with patch("lnl_toolbox.cli.main.compare_runs", return_value=leaked):
            code, _, _ = self.invoke("compare", str(ROOT), "--strict")
        self.assertEqual(code, 1)

        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.cli.main.compare_runs", return_value=summary
        ), patch(
            "lnl_toolbox.cli.main.write_report",
            return_value={"report": Path(directory) / "report.md"},
        ):
            code, output, error = self.invoke("report", str(ROOT), "--output-dir", directory)
        self.assertEqual(code, 0, error)
        self.assertIn("report.md", output)

    def test_matrix_sweep_dry_run_and_status_do_not_invoke_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "sweep-output"
            spec = root / "sweep.yaml"
            spec.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "base": {"recipe": "cifar10-symmetric-ce-smoke"},
                        "matrix": {"trainer.epochs": [1, 2]},
                        "seeds": [1, 2],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            with patch("lnl_toolbox.cli.main.ExperimentService.run") as runner:
                code, output, error = self.invoke(
                    "sweep",
                    str(spec),
                    "--dry-run",
                    "--no-check-data",
                    "--output-dir",
                    str(output_dir),
                )
            self.assertEqual(code, 0, error)
            self.assertIn("Total runs:\n  4", output)
            self.assertIn("trainer.epochs=1", output)
            self.assertFalse(output_dir.exists())
            runner.assert_not_called()

            output_dir.mkdir()
            (output_dir / "sweep_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "sweep_id": "example",
                        "status": "running",
                        "runs": [
                            {"seed": 1, "status": "completed"},
                            {
                                "seed": 2,
                                "status": "failed",
                                "overrides": {"trainer.epochs": 2},
                                "error": "failure",
                            },
                            {"seed": 3, "status": "pending"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            code, output, error = self.invoke("sweep", "status", str(output_dir))
            self.assertEqual(code, 0, error)
            self.assertIn("1 / 3 completed", output)
            self.assertIn("trainer.epochs=2", output)

    def test_cli_matrix_is_typed_and_seed_defaults_to_recipe_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "lnl_toolbox.cli.main.ExperimentService.run"
        ) as runner:
            code, output, error = self.invoke(
                "sweep",
                "--recipe",
                "cifar10-clean-smoke",
                "--matrix",
                "loader.batch_size=[256,512]",
                "--matrix",
                "optimizer.lr=[0.01,0.001]",
                "--output-dir",
                directory,
                "--dry-run",
                "--no-check-data",
            )
        self.assertEqual(code, 0, error)
        self.assertIn("Total runs:\n  4", output)
        self.assertIn("loader.batch_size=256", output)
        runner.assert_not_called()

    def test_cli_matrix_multiplies_optional_seeds_and_rejects_bad_json(self) -> None:
        code, output, error = self.invoke(
            "sweep",
            "--recipe",
            "cifar10-clean-smoke",
            "--matrix",
            "loader.batch_size=[256,512]",
            "--matrix",
            "optimizer.lr=[0.01,0.001]",
            "--seeds",
            "1",
            "2",
            "3",
            "--dry-run",
            "--no-check-data",
        )
        self.assertEqual(code, 0, error)
        self.assertIn("Total runs:\n  12", output)
        code, _, error = self.invoke(
            "sweep", "--recipe", "cifar10-clean-smoke", "--matrix", "optimizer.lr=0.1"
        )
        self.assertEqual(code, 2)
        self.assertIn("JSON array", error)

    def test_table_commands_offer_json_contracts(self) -> None:
        for arguments in (
            ("list", "experiments", "--format", "json"),
            ("list", "components", "--format", "json"),
            ("papers", "list", "--format", "json"),
            ("data", "list", "--format", "json"),
        ):
            code, output, error = self.invoke(*arguments)
            self.assertEqual(code, 0, error)
            self.assertIsInstance(json.loads(output), list)

        summary = {
            "summaries": [], "warnings": [], "excluded_runs": [], "failed_runs": [],
            "group_by": [], "compatibility": {},
        }
        with patch("lnl_toolbox.cli.main.compare_runs", return_value=summary):
            code, output, error = self.invoke("compare", str(ROOT), "--format", "json")
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(output)["summaries"], [])


if __name__ == "__main__":
    unittest.main()
