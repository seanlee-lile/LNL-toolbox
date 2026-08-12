from __future__ import annotations

import inspect
import unittest

from lnl_toolbox.cli.main import _print_plan
from lnl_toolbox.training.runners import apply_epoch_override, resolve_runner


class RunnerPlanningTest(unittest.TestCase):
    def test_supervised_runner_owns_plan_and_budget_override(self) -> None:
        config = {
            "execution": {"runner": "supervised"},
            "data": {"name": "cifar10"},
            "trainer": {"epochs": 10},
            "loss": {"name": "gce"},
        }
        runner = resolve_runner(config)
        self.assertEqual(runner.describe(config).training_budget, "10 epochs")
        apply_epoch_override(config, 3)
        self.assertEqual(config["trainer"]["epochs"], 3)

    def test_multistage_runner_uses_registered_budget_path(self) -> None:
        config = {
            "execution": {"runner": "upm"},
            "method": "upm",
            "data": {"name": "cifar10"},
            "upm": {"main": {"epochs": 100}},
        }
        apply_epoch_override(config, 4)
        self.assertEqual(config["upm"]["main"]["epochs"], 4)

    def test_cli_preview_contains_no_method_specific_dispatch(self) -> None:
        source = inspect.getsource(_print_plan)
        for method in ("upm", "dld", "dividemix", "lend"):
            self.assertNotIn(f'== "{method}"', source)


if __name__ == "__main__":
    unittest.main()
