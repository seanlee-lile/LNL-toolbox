from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from lnl_toolbox import toolbox
from lnl_toolbox.catalog import (
    load_papers,
    load_recipe_config,
    paper_by_id,
    recipe_by_id,
    select_paper_config,
    validate_config,
)
from lnl_toolbox.training.dld_experiment import run_dld_experiment
from lnl_toolbox.training.lend_experiment import run_lend_experiment
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner
from lnl_toolbox.training.upm_experiment import run_upm_experiment


class PublicPaperRoutingTest(unittest.TestCase):
    @staticmethod
    def _config(recipe: str):
        return load_recipe_config(recipe_by_id(recipe))

    def test_volmin_and_legacy_alias_share_one_complete_runner(self) -> None:
        canonical = toolbox.get("volmin")
        alias = toolbox.get("volminnet")
        self.assertEqual(canonical.spec.name, "volmin")
        self.assertEqual(alias.spec.name, "volminnet")
        self.assertEqual(canonical.spec.module, alias.spec.module)
        self.assertEqual(canonical.spec.function, alias.spec.function)
        self.assertEqual(
            canonical.spec.module,
            "lnl_toolbox.training.volminnet_experiment",
        )
        self.assertEqual(paper_by_id("volmin").id, "volmin")
        self.assertEqual(paper_by_id("volminnet").id, "volmin")
        selected, recipe = select_paper_config(
            paper_by_id("volminnet"), profile="smoke"
        )
        self.assertEqual(selected.implementation_status, "user_ready")
        self.assertEqual(recipe.id, "volmin-cifar10-smoke")
        config = self._config(recipe.id)
        self.assertEqual(resolve_runner(config).name, "volmin")
        self.assertEqual(validate_config(config).name, "volmin")

    def test_public_recipes_dispatch_to_existing_paper_workflows(self) -> None:
        cases = (
            (
                "upm-cifar10-smoke",
                run_upm_experiment,
                "lnl_toolbox.training.upm_experiment._run_upm_paper_workflow",
                "lnl_toolbox.training.upm_experiment._run_legacy_upm_experiment",
            ),
            (
                "dld-cifar10-smoke",
                run_dld_experiment,
                "lnl_toolbox.training.dld_experiment._run_dld_paper_workflow",
                "lnl_toolbox.training.dld_experiment._run_legacy_dld_experiment",
            ),
            (
                "lend-cifar10-smoke",
                run_lend_experiment,
                "lnl_toolbox.training.lend_experiment._run_lend_paper_workflow",
                "lnl_toolbox.training.lend_experiment._run_legacy_lend_experiment",
            ),
        )
        expected = Path("paper-workflow")
        for recipe_id, entrypoint, paper_target, legacy_target in cases:
            with self.subTest(recipe=recipe_id):
                config = self._config(recipe_id)
                self.assertEqual(validate_config(config).name, config["method"])
                with patch(paper_target, return_value=expected) as paper_run, patch(
                    legacy_target
                ) as legacy_run:
                    self.assertEqual(entrypoint(config), expected)
                paper_run.assert_called_once()
                legacy_run.assert_not_called()

    def test_stage_epoch_overrides_do_not_change_other_stages(self) -> None:
        cases = (
            ("upm-cifar10-smoke", ("upm", "main", "epochs"), ("upm", "stage1", "epochs")),
            ("dld-cifar10-smoke", ("dld", "diffusion", "epochs"), None),
            ("lend-cifar10-smoke", ("lend", "training", "epochs"), None),
            ("cifar10-dividemix-smoke", ("dividemix", "training", "epochs"), ("dividemix", "warmup", "epochs")),
        )
        for recipe_id, target, protected in cases:
            with self.subTest(recipe=recipe_id):
                config = self._config(recipe_id)
                protected_value = self._nested(config, protected) if protected else None
                apply_epoch_override(config, 7)
                self.assertEqual(self._nested(config, target), 7)
                if protected:
                    self.assertEqual(self._nested(config, protected), protected_value)

    def test_five_public_methods_are_user_ready_and_runnable(self) -> None:
        papers = {paper.id: paper for paper in load_papers()}
        for method in ("volmin", "upm", "dld", "dividemix", "lend"):
            self.assertEqual(papers[method].implementation_status, "user_ready")
            self.assertEqual(papers[method].availability, "runnable")

    @staticmethod
    def _nested(config, path):
        value = config
        for key in path:
            value = value[key]
        return value


if __name__ == "__main__":
    unittest.main()
