from __future__ import annotations

import unittest

from lnl_toolbox.scratch import ScratchContext, ScratchExecutionError, execute_recipe
from lnl_toolbox.scratch.registry import block


@block(id="test_explode", name="Explode", category="Test")
def _explode(ctx):
    raise RuntimeError("boom")


class ScratchExecutorTest(unittest.TestCase):
    def test_sequence_loop_and_condition_share_context(self) -> None:
        recipe = {
            "schema_version": 1,
            "name": "toy",
            "steps": [
                {"block": "set_value", "params": {"value": True, "save_as": "enabled"}},
                {
                    "block": "repeat_n",
                    "params": {"count": 3, "index_as": "iteration"},
                    "steps": [{
                        "block": "if_context",
                        "params": {"flag": "enabled"},
                        "steps": [{
                            "block": "set_value",
                            "params": {"value": "ran", "save_as": "result"},
                        }],
                    }],
                },
            ],
        }
        context = execute_recipe(recipe)
        self.assertEqual(context["iteration"], 2)
        self.assertEqual(context["result"], "ran")

    def test_runtime_error_identifies_block(self) -> None:
        recipe = {
            "schema_version": 1,
            "name": "runtime-error",
            "steps": [{"block": "test_explode"}],
        }
        with self.assertRaisesRegex(ScratchExecutionError, "Explode.*boom"):
            execute_recipe(recipe, ScratchContext())


if __name__ == "__main__":
    unittest.main()
