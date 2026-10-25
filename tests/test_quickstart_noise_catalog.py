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

    def test_every_visible_synthetic_capability_builds_a_config(self) -> None:
        for spec in visible_synthetic_noise_specs():
            with self.subTest(spec=spec.key):
                extra = {'transition_matrix': ((0.8, 0.2), (0.1, 0.9))} if spec.key == 'class_conditional' else {}
                self.assertIsNotNone(build_noise_config(spec.key, rate=0.2, seed=1, **extra))

    def test_explicit_matrix_is_validated(self) -> None:
        matrix = ((0.8, 0.2), (0.1, 0.9))
        self.assertEqual(build_noise_config('class_conditional', rate=0.2, seed=1, transition_matrix=matrix)['transition_matrix'], [list(row) for row in matrix])
        for invalid in (None, (), ((1.0,),), ((0.8, 0.1), (0.1, 0.9)), ((-0.1, 1.1), (0.1, 0.9)), ((float('nan'), 0.2), (0.1, 0.9))):
            with self.subTest(matrix=invalid), self.assertRaises(ValueError):
                build_noise_config('class_conditional', rate=0.2, seed=1, transition_matrix=invalid)

    def test_external_path_is_declaration_not_readiness(self) -> None:
        value = build_noise_config('external_torch', rate=None, seed=3, path=' missing.pt ')
        self.assertEqual(value['path'], 'missing.pt')
        self.assertNotIn('rate', value)
        self.assertEqual(value['clean_key'], 'clean_label_train')
        self.assertEqual(value['noisy_key'], 'noise_label_train')
        with self.assertRaises(ValueError):
            build_noise_config('external_torch', rate=None, seed=3)


if __name__ == "__main__":
    unittest.main()
