from __future__ import annotations

import unittest

from lnl_toolbox import toolbox


class UnifiedMethodMatrixTest(unittest.TestCase):
    def test_lifecycle_metadata_is_available_through_the_public_runner(self) -> None:
        expected = {
            "apl": "single_stage", "gce": "single_stage", "dss": "single_stage",
            "cdr": "single_stage", "loss-correction": "single_stage",
            "jocor": "multi_model", "coteaching": "multi_model", "cnlcu": "multi_model",
            "pdl": "staged", "t-revision": "staged", "dual-t": "staged",
            "cal": "staged", "pcse": "staged", "mc-ldce": "staged", "volmin": "staged",
            "upm": "staged", "fine": "two_stage", "ca2c": "staged", "dld": "staged",
            "lend": "staged", "importance-reweighting": "staged", "l2rw": "staged",
            "binary-risk": "single_stage", "mentornet": "single_stage",
        }
        for method, lifecycle in expected.items():
            runner = toolbox.get(method)
            self.assertEqual(runner.spec.lifecycle, lifecycle, method)
            self.assertIn(runner.spec.checkpoint_unit, {"epoch", "step"}, method)


if __name__ == "__main__":
    unittest.main()
