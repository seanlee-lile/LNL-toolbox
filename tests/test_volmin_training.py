from __future__ import annotations

import unittest

from lnl_toolbox import toolbox


class VolMinTrainingTest(unittest.TestCase):
    def test_canonical_and_alias_use_complete_volminnet_workflow(self) -> None:
        canonical = toolbox.get("volmin")
        alias = toolbox.get("volminnet")
        self.assertEqual(canonical.spec.name, "volmin")
        self.assertEqual(alias.spec.name, "volminnet")
        self.assertEqual(canonical.spec.module, "lnl_toolbox.training.volminnet_experiment")
        self.assertEqual(canonical.spec.function, "run_volminnet_experiment")


if __name__ == "__main__":
    unittest.main()
