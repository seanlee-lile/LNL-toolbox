import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lnl_toolbox.noise import (
    KnownTransition,
    NoiseManifest,
    generate_instance_dependent,
    generate_symmetric,
    validate_transition_matrix,
)
from lnl_toolbox.training.experiment import _load_noise_manifest, _validate_resume_noise


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

    def test_manifest_validates_dataset_identity_and_label_alignment(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 1, "cifar10")
        self.assertIs(manifest.validate_for(self.labels, "CIFAR-10", 10), manifest)
        with self.assertRaisesRegex(ValueError, "dataset"):
            manifest.validate_for(self.labels, "cifar100", 10)
        with self.assertRaisesRegex(ValueError, "length"):
            manifest.validate_for(self.labels[:-1], "cifar10", 10)
        shifted = np.roll(self.labels, 1)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            manifest.validate_for(shifted, "cifar10", 10)

    def test_manifest_rejects_invalid_target_range_and_probability_shape(self) -> None:
        bad_targets = self.labels.copy()
        bad_targets[0] = 10
        manifest = NoiseManifest("cifar10", "fixture", 1, 0.1, self.labels, bad_targets)
        with self.assertRaisesRegex(ValueError, "noisy_targets"):
            manifest.validate_for(self.labels, "cifar10", 10)

        probabilities = np.full((self.labels.size, 9), 1.0 / 9.0)
        manifest = NoiseManifest(
            "cifar10", "fixture", 1, 0.1, self.labels, self.labels,
            per_sample_transition=probabilities,
        )
        with self.assertRaisesRegex(ValueError, "per_sample_transition"):
            manifest.validate_for(self.labels, "cifar10", 10)

    def test_transition_matrix_validation_and_known_provider(self) -> None:
        matrix = np.eye(10, dtype=np.float64) * 0.9
        matrix[np.arange(10), (np.arange(10) + 1) % 10] = 0.1
        provider = KnownTransition(matrix)
        tensor = provider.as_tensor(device="cpu", dtype=torch.float32)
        self.assertEqual(provider.num_classes, 10)
        self.assertEqual(tensor.shape, (10, 10))
        self.assertEqual(tensor.dtype, torch.float32)
        np.testing.assert_allclose(tensor.numpy(), matrix, rtol=1e-6, atol=1e-8)
        self.assertAlmostEqual(float(tensor[0, 1]), 0.1)
        self.assertEqual(float(tensor[1, 0]), 0.0)

        manifest = NoiseManifest(
            "cifar10", "fixture", 1, 0.1, self.labels, self.labels,
            transition_matrix=matrix,
        )
        np.testing.assert_array_equal(KnownTransition.from_manifest(manifest).matrix, matrix)

    def test_transition_matrix_rejects_invalid_probabilities(self) -> None:
        invalid = (
            np.ones((2, 3)),
            np.array([[1.1, -0.1], [0.0, 1.0]]),
            np.array([[0.2, 0.2], [0.0, 1.0]]),
            np.array([[np.nan, 0.0], [0.0, 1.0]]),
        )
        for matrix in invalid:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    validate_transition_matrix(matrix)

    def test_training_manifest_resolution_records_identity_without_labels(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 5, "cifar10")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noise.npz"
            manifest.save(path)
            loaded, resolved, summary = _load_noise_manifest(
                {"manifest": str(path)}, self.labels, "cifar10", 10
            )
        self.assertIsNotNone(loaded)
        self.assertEqual(resolved["manifest"], str(path.resolve()))
        self.assertEqual(len(resolved["manifest_sha256"]), 64)
        self.assertEqual(summary["noise_type"], "symmetric")
        self.assertFalse(summary["uses_known_transition"])
        self.assertNotIn("clean_targets", summary)
        self.assertNotIn("noisy_targets", summary)

    def test_resume_must_use_the_same_manifest_identity(self) -> None:
        _validate_resume_noise(
            {"noise": {"manifest_sha256": "abc"}},
            {"noise": {"manifest_sha256": "ABC"}},
        )
        with self.assertRaisesRegex(ValueError, "different noise manifest"):
            _validate_resume_noise(
                {"noise": {"manifest_sha256": "abc"}},
                {"noise": {"manifest_sha256": "def"}},
            )
        with self.assertRaisesRegex(ValueError, "different noise manifest"):
            _validate_resume_noise({"noise": {"manifest_sha256": "abc"}}, {})



if __name__ == "__main__":
    unittest.main()
