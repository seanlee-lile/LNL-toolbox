from __future__ import annotations

import unittest

from lnl_toolbox.noise.quickstart_catalog import (
    build_noise_config,
    quick_start_noise_spec,
    visible_synthetic_noise_specs,
)


class QuickStartNoiseCatalogTests(unittest.TestCase):
    def test_symmetric_is_human_readable(self) -> None:
        self.assertEqual(quick_start_noise_spec("symmetric").label, "对称噪声")

    def test_external_source_is_hidden(self) -> None:
        self.assertNotIn("external_torch", {item.key for item in visible_synthetic_noise_specs()})

    def test_clean_has_no_generated_config(self) -> None:
        self.assertIsNone(build_noise_config("clean", rate=None, seed=None))

    def test_symmetric_builds_existing_contract(self) -> None:
        self.assertEqual(
            build_noise_config("symmetric", rate=0.2, seed=1),
            {"name": "symmetric", "rate": 0.2, "seed": 1},
        )

    def test_rate_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            build_noise_config("symmetric", rate=-0.1, seed=1)
        with self.assertRaises(ValueError):
            build_noise_config("symmetric", rate=1.1, seed=1)

    def test_rate_is_required(self) -> None:
        with self.assertRaises(ValueError):
            build_noise_config("pairflip", rate=None, seed=1)

    def test_artifact_backed_visible_capabilities_do_not_fabricate_config(self) -> None:
        self.assertIsNone(build_noise_config("class_conditional", rate=0.2, seed=1))
        self.assertIsNone(build_noise_config("instance_dependent", rate=0.2, seed=1))


if __name__ == "__main__":
    unittest.main()
