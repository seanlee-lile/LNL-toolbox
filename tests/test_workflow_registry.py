from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from lnl_toolbox.training.experiment import run_experiment
from lnl_toolbox.training.workflows import (
    WorkflowRegistry,
    create_workflow_registry,
    method_name,
    resolve_workflow,
)


class WorkflowRegistryTest(unittest.TestCase):
    def test_builtin_workflows_are_registered_without_main_runner_branches(self) -> None:
        registry = create_workflow_registry()
        self.assertEqual(
            registry.names(),
            (
                "ca2c", "cal", "cnlcu", "coteaching", "dld", "dual_t",
                "importance_reweighting", "l2rw", "lend", "mc_ldce", "pcse", "t_revision", "upm", "volmin",
            ),
        )
        source = inspect.getsource(run_experiment)
        for name in registry.names():
            self.assertNotIn(name, source)

    def test_method_name_supports_scalar_mapping_and_default(self) -> None:
        self.assertEqual(method_name({}), "")
        self.assertEqual(method_name({"method": " Dual_T "}), "dual_t")
        self.assertEqual(method_name({"method": {"name": "PCSE"}}), "pcse")

    def test_unknown_and_renamed_methods_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown method"):
            resolve_workflow({"method": "not_registered"})
        with self.assertRaisesRegex(ValueError, "dual_t_forward.*dual_t"):
            resolve_workflow({"method": "dual_t_forward"})

    def test_registry_rejects_duplicate_names_and_loads_lazily(self) -> None:
        registry = WorkflowRegistry()
        registry.add("example", "pathlib", "Path")
        with self.assertRaisesRegex(KeyError, "already registered"):
            registry.add("example", "pathlib", "Path")
        self.assertIs(registry.resolve("example"), Path)


if __name__ == "__main__":
    unittest.main()
