from __future__ import annotations

import unittest

from lnl_toolbox.training.runners import create_runner_registry, resolve_runner


class PaperRunnerRegistryTest(unittest.TestCase):
    def test_new_paper_runners_are_lazy_and_resolvable(self) -> None:
        registry = create_runner_registry()
        for name in ("mc_ldce", "cal", "ca2c", "l2rw"):
            self.assertIn(name, registry.names())
            self.assertEqual(resolve_runner({"method": name}).name, name)

    def test_explicit_runner_must_match_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_runner({"method": "cal", "execution": {"runner": "ca2c"}})


if __name__ == "__main__":
    unittest.main()
