from __future__ import annotations

import unittest

from lnl_toolbox.training.runners import create_runner_registry


class RunnerContractTest(unittest.TestCase):
    def test_every_builtin_runner_declares_lifecycle_checkpoint_and_smoke(self) -> None:
        registry = create_runner_registry()
        self.assertEqual(len(registry.names()), 23)
        for name in registry.names():
            spec = registry.get(name)
            self.assertTrue(spec.lifecycle, name)
            self.assertIn(spec.checkpoint_unit, {"epoch", "step"}, name)
            self.assertTrue(spec.smoke_recipe, name)

    def test_step_runner_is_reserved_for_step_based_training(self) -> None:
        spec = create_runner_registry().get("l2rw")
        self.assertEqual(spec.checkpoint_unit, "step")


if __name__ == "__main__":
    unittest.main()
