from __future__ import annotations

from copy import deepcopy
import unittest

from lnl_toolbox.core.config_overrides import (
    apply_override,
    apply_override_assignments,
    parse_override_value,
)


class ConfigOverridesTest(unittest.TestCase):
    def test_values_are_typed(self) -> None:
        self.assertIs(parse_override_value("true"), True)
        self.assertEqual(parse_override_value("12"), 12)
        self.assertEqual(parse_override_value("0.25"), 0.25)
        self.assertEqual(parse_override_value("[1, 2]"), [1, 2])
        self.assertEqual(parse_override_value("cifar10"), "cifar10")

    def test_nested_existing_values_are_overridden(self) -> None:
        config = {"trainer": {"epochs": 10}, "seed": 1}
        result = apply_override_assignments(
            config, ["trainer.epochs=20", "seed=7"]
        )
        self.assertEqual(result, {"trainer": {"epochs": 20}, "seed": 7})
        self.assertEqual(config, {"trainer": {"epochs": 10}, "seed": 1})

    def test_typo_fails_without_partial_mutation(self) -> None:
        config = {"trainer": {"epochs": 10}}
        before = deepcopy(config)
        with self.assertRaisesRegex(ValueError, "epochs"):
            apply_override(config, "trainer.epoch", 20)
        self.assertEqual(config, before)

    def test_assignment_requires_equals(self) -> None:
        with self.assertRaisesRegex(ValueError, "path=value"):
            apply_override_assignments({"seed": 1}, ["seed"])


if __name__ == "__main__":
    unittest.main()
