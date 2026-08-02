from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lnl_toolbox.data.torch_cifar import train_validation_split
from lnl_toolbox.noise.generators import generate_symmetric
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.noise.split_manifest import (
    _validate_symmetric_rate,
    generate_split_symmetric_manifest,
)
from scripts.prepare_split_noise_manifest import _prepare_destination


class SplitNoiseManifestTest(unittest.TestCase):
    def test_classwise_legacy_matches_reference_sequence(self) -> None:
        labels = np.repeat(np.arange(3), 6)
        train, validation = train_validation_split(
            labels,
            6,
            7,
            strategy="classwise_legacy",
            rng="numpy_legacy",
        )
        random = np.random.RandomState(7)
        expected_train = []
        expected_validation = []
        for class_index in range(3):
            indices = np.flatnonzero(labels == class_index)
            random.shuffle(indices)
            expected_train.extend(indices[:4])
            expected_validation.extend(indices[4:])
        random.shuffle(expected_train)
        random.shuffle(expected_validation)
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(
            validation,
            expected_validation,
        )

    def test_each_split_restarts_symmetric_noise_rng(self) -> None:
        labels = np.tile(np.arange(3), 12)
        train = np.arange(0, 24)
        validation = np.arange(24, 36)
        manifest = generate_split_symmetric_manifest(
            labels,
            (train, validation),
            num_classes=3,
            rate=0.5,
            seed=4,
            dataset="toy",
            split_names=("train", "validation"),
        )
        expected = [
            generate_symmetric(
                labels[indices],
                3,
                0.5,
                4,
                "toy",
                sampling="transition",
                rng="numpy_legacy",
            ).noisy_targets
            for indices in (train, validation)
        ]
        np.testing.assert_array_equal(
            manifest.noisy_targets,
            np.concatenate(expected),
        )
        np.testing.assert_array_equal(
            manifest.global_indices,
            np.concatenate((train, validation)),
        )
        self.assertEqual(
            manifest.metadata["split_names"],
            ["train", "validation"],
        )
        self.assertEqual(manifest.metadata["rng_scope"], "per_split")

    def test_manifest_roundtrip_preserves_provenance_and_mapping(self):
        labels = np.tile(np.arange(2), 10)
        train = np.arange(0, 12)
        validation = np.arange(12, 20)
        manifest = generate_split_symmetric_manifest(
            labels,
            (train, validation),
            num_classes=2,
            rate=0.3,
            seed=2,
            dataset="toy",
            split_names=("train", "validation"),
        )
        np.testing.assert_allclose(
            1.0 - np.diag(manifest.transition_matrix),
            0.3,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.npz"
            manifest.save(path)
            restored = NoiseManifest.load(path)
        np.testing.assert_array_equal(
            restored.global_indices,
            manifest.global_indices,
        )
        np.testing.assert_array_equal(
            restored.noisy_targets,
            manifest.noisy_targets,
        )
        self.assertEqual(restored.mapping_hash, manifest.mapping_hash)
        self.assertEqual(restored.metadata, manifest.metadata)

    def test_invalid_split_alignment_is_rejected(self) -> None:
        labels = np.arange(12) % 3
        with self.assertRaisesRegex(ValueError, "disjoint"):
            generate_split_symmetric_manifest(
                labels,
                (np.arange(8), np.arange(7, 12)),
                num_classes=3,
                rate=0.2,
                seed=1,
                dataset="toy",
            )
        with self.assertRaisesRegex(ValueError, "split_names"):
            generate_split_symmetric_manifest(
                labels,
                (np.arange(8), np.arange(8, 12)),
                num_classes=3,
                rate=0.2,
                seed=1,
                dataset="toy",
                split_names=("train", "train"),
            )
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            generate_split_symmetric_manifest(
                labels,
                (np.array([0.0, 1.0]), np.arange(2, 12)),
                num_classes=3,
                rate=0.2,
                seed=1,
                dataset="toy",
            )

    def test_rate_mismatch_and_script_overwrite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "implied rate"):
            _validate_symmetric_rate(np.eye(3), 0.2)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "manifest.npz"
            self.assertEqual(
                _prepare_destination(destination),
                destination,
            )
            self.assertTrue(destination.parent.is_dir())
            destination.write_bytes(b"existing")
            with self.assertRaisesRegex(
                FileExistsError,
                "refusing to overwrite",
            ):
                _prepare_destination(destination)


if __name__ == "__main__":
    unittest.main()
