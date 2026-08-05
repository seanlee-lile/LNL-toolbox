from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import yaml

from lnl_toolbox.catalog import (
    discover_recipes,
    load_papers,
    load_yaml,
    resolve_config_paths,
    select_paper_config,
    validate_config,
)
from lnl_toolbox.cli.main import main
from lnl_toolbox.training.runners import resolve_runner, runner_names


ROOT = Path(__file__).resolve().parents[1]


class RunnerResolutionTest(unittest.TestCase):
    def test_all_public_runners_are_registered(self) -> None:
        self.assertEqual(
            set(runner_names()),
            {
                "binary", "cal", "ca2c", "clean", "coteaching", "cwd", "dld", "dual_t", "fine",
                "importance_reweighting", "instance_transition", "multi_model",
                "l2rw", "mc_ldce", "pcse", "supervised",
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
    def test_every_builtin_recipe_has_explicit_valid_runner(self) -> None:
        recipes = discover_recipes(ROOT)
        self.assertGreaterEqual(len(recipes), 39)
        for recipe in recipes:
            config = load_yaml(recipe.config_path)
            self.assertIn("execution", config, recipe.id)
            self.assertEqual(validate_config(config).name, recipe.runner, recipe.id)

    def test_paper_catalog_only_references_runnable_recipes(self) -> None:
        recipes = {item.id for item in discover_recipes(ROOT)}
        papers = load_papers(ROOT)
        self.assertGreaterEqual(len(papers), 10)
        for paper in papers:
            self.assertTrue(paper.configs)
            for item in paper.configs:
                self.assertIn(item.recipe_id, recipes)

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


class UnifiedCliTest(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

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
        code, output, _ = self.invoke("run", "--recipe", "cifar10-symmetric-ce-smoke", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("执行器: supervised", output)
        self.assertIn("标签来源:", output)

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


if __name__ == "__main__":
    unittest.main()
