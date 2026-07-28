from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lnl_toolbox.data.torch_cifar import train_validation_split
from lnl_toolbox.noise.generators import generate_symmetric
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.noise.split_manifest import (
    generate_split_symmetric_manifest,
)


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
        np.testing.assert_array_equal(validation, expected_validation)

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
        )
        expected_train = generate_symmetric(
            labels[train], 3, 0.5, 4, "toy",
            sampling="transition", rng="numpy_legacy",
        )
        expected_validation = generate_symmetric(
            labels[validation], 3, 0.5, 4, "toy",
            sampling="transition", rng="numpy_legacy",
        )
        np.testing.assert_array_equal(
            manifest.noisy_targets,
            np.concatenate((
                expected_train.noisy_targets,
                expected_validation.noisy_targets,
            )),
        )
        self.assertEqual(manifest.metadata["rng_scope"], "per_split")

    def test_manifest_roundtrip_preserves_global_mapping(self) -> None:
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
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.npz"
            manifest.save(path)
            restored = NoiseManifest.load(path)
        np.testing.assert_array_equal(
            restored.global_indices, manifest.global_indices
        )
        np.testing.assert_array_equal(
            restored.noisy_targets, manifest.noisy_targets
        )
        self.assertEqual(restored.mapping_hash, manifest.mapping_hash)


if __name__ == "__main__":
    unittest.main()
