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
                "binary", "clean", "coteaching", "cwd", "dual_t", "fine",
                "importance_reweighting", "instance_transition", "multi_model",
                "pcse", "supervised",
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
        code, output, _ = self.invoke("papers", "list")
        self.assertEqual(code, 0)
        self.assertIn("dual-t", output)
        code, output, _ = self.invoke("papers", "show", "jocor")
        self.assertEqual(code, 0)
        self.assertIn("Toolbox 生命周期", output)
        self.assertIn("论文概念 -> config -> 实现", output)
        self.assertIn("已知差异与限制", output)

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


if __name__ == "__main__":
    unittest.main()
