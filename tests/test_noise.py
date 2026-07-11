import tempfile
import unittest
from pathlib import Path

import numpy as np

from lnl_toolbox.noise import NoiseManifest, generate_instance_dependent, generate_symmetric


class NoiseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.tile(np.arange(10), 20)

    def test_zero_rate_keeps_labels(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.0, 1, "toy")
        np.testing.assert_array_equal(manifest.clean_targets, manifest.noisy_targets)

    def test_flips_never_keep_original_class(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 1, "toy")
        self.assertTrue(np.all(manifest.noisy_targets[manifest.flip_mask] != self.labels[manifest.flip_mask]))

    def test_seed_is_reproducible_and_manifest_roundtrips(self) -> None:
        first = generate_symmetric(self.labels, 10, 0.4, 7, "toy")
        second = generate_symmetric(self.labels, 10, 0.4, 7, "toy")
        np.testing.assert_array_equal(first.noisy_targets, second.noisy_targets)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noise.npz"
            first.save(path)
            loaded = NoiseManifest.load(path)
        np.testing.assert_array_equal(first.noisy_targets, loaded.noisy_targets)
        self.assertAlmostEqual(first.realized_rate, loaded.realized_rate)

    def test_instance_dependent_manifest_tracks_per_sample_probabilities(self) -> None:
        scores = np.random.default_rng(3).normal(size=(self.labels.size, 10))
        manifest = generate_instance_dependent(self.labels, scores, 0.3, 4, "toy")
        self.assertEqual(manifest.per_sample_transition.shape, (self.labels.size, 10))
        np.testing.assert_allclose(manifest.per_sample_transition.sum(axis=1), 1.0)



if __name__ == "__main__":
    unittest.main()
