from __future__ import annotations

import unittest

from lnl_toolbox.scratch import ScratchValidationError, validate_recipe


class ScratchValidationTest(unittest.TestCase):
    def test_unknown_block_and_forbidden_code_fail(self) -> None:
        with self.assertRaisesRegex(ScratchValidationError, "unknown Scratch block"):
            validate_recipe({"schema_version": 1, "name": "bad", "steps": [{"block": "missing"}]})
        with self.assertRaisesRegex(ScratchValidationError, "forbidden"):
            validate_recipe({"schema_version": 1, "name": "bad", "steps": [{"block": "set_value", "params": {"value": 1}, "code": "print(1)"}]})

    def test_missing_parameter_and_requirement_are_explained(self) -> None:
        with self.assertRaisesRegex(ScratchValidationError, "missing parameter `value`"):
            validate_recipe({"schema_version": 1, "name": "bad", "steps": [{"block": "set_value"}]})
        with self.assertRaisesRegex(ScratchValidationError, "missing context values: flag"):
            validate_recipe({"schema_version": 1, "name": "bad", "steps": [{"block": "if_context", "params": {"flag": "flag"}, "steps": []}]})

    def test_action_cannot_have_children(self) -> None:
        with self.assertRaisesRegex(ScratchValidationError, "cannot contain"):
            validate_recipe({"schema_version": 1, "name": "bad", "steps": [{"block": "set_value", "params": {"value": 1}, "steps": []}]})


if __name__ == "__main__":
    unittest.main()
