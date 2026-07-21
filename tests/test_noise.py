import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from lnl_toolbox.data.noisy_dataset import NoisyTargetDataset
from lnl_toolbox.noise import (
    NoiseManifest,
    generate_instance_dependent,
    generate_pairflip,
    generate_symmetric,
)


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
        np.testing.assert_array_equal(first.global_indices, loaded.global_indices)
        self.assertEqual(first.mapping_hash, loaded.mapping_hash)
        self.assertAlmostEqual(first.realized_rate, loaded.realized_rate)

    def test_legacy_v1_manifest_loads_with_explicit_default_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            metadata = {
                "version": "1.0", "dataset": "toy", "dataset_fingerprint": "",
                "noise_type": "symmetric", "seed": 3, "requested_rate": 0.2,
                "metadata": {},
            }
            noisy = self.labels.copy()
            noisy[:2] = (noisy[:2] + 1) % 10
            np.savez_compressed(
                path, clean_targets=self.labels, noisy_targets=noisy,
                flip_mask=self.labels != noisy, transition_matrix=np.array([]),
                per_sample_transition=np.array([]),
                metadata_json=np.array(json.dumps(metadata)),
            )
            loaded = NoiseManifest.load(path)
        self.assertEqual(loaded.version, "1.0")
        np.testing.assert_array_equal(loaded.global_indices, np.arange(self.labels.size))

    def test_different_seeds_usually_generate_different_noise(self) -> None:
        first = generate_symmetric(self.labels, 10, 0.4, 7, "toy")
        second = generate_symmetric(self.labels, 10, 0.4, 8, "toy")
        self.assertFalse(np.array_equal(first.noisy_targets, second.noisy_targets))

    def test_symmetric_realized_rate_matches_fixed_requested_count(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 7, "toy")
        self.assertAlmostEqual(manifest.actual_rate, 0.4)

    def test_pairflip_only_moves_to_next_class(self) -> None:
        manifest = generate_pairflip(self.labels, 10, 0.5, 3, "toy")
        expected = (manifest.clean_targets[manifest.flip_mask] + 1) % 10
        np.testing.assert_array_equal(manifest.noisy_targets[manifest.flip_mask], expected)

    def test_mapping_hash_covers_indices_targets_and_context(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 7, "toy")
        reordered = NoiseManifest(
            dataset=manifest.dataset,
            split=manifest.split,
            noise_type=manifest.noise_type,
            seed=manifest.seed,
            requested_rate=manifest.requested_rate,
            clean_targets=manifest.clean_targets[::-1],
            noisy_targets=manifest.noisy_targets[::-1],
            global_indices=manifest.global_indices[::-1],
            num_classes=manifest.num_classes,
            dataset_fingerprint=manifest.dataset_fingerprint,
        )
        self.assertNotEqual(manifest.mapping_hash, reordered.mapping_hash)

    def test_noisy_dataset_maps_by_global_index_and_hides_clean_target(self) -> None:
        class IndexedDataset:
            indices = np.array([9, 3, 7], dtype=np.int64)

            def __len__(self) -> int:
                return len(self.indices)

            def __getitem__(self, item: int) -> dict[str, object]:
                index = int(self.indices[item])
                return {"input": torch.tensor([index]), "target": index % 2, "index": index}

        dataset = NoisyTargetDataset(
            IndexedDataset(),
            global_indices=np.array([3, 7, 9]),
            noisy_targets=np.array([1, 2, 4]),
        )
        self.assertEqual(set(dataset[0]), {"input", "target", "index"})
        self.assertEqual(dataset[0]["target"], 4)
        seen = {}
        loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(2))
        for batch in loader:
            seen[int(batch["index"].item())] = int(batch["target"].item())
        self.assertEqual(seen, {3: 1, 7: 2, 9: 4})

    def test_instance_dependent_manifest_tracks_per_sample_probabilities(self) -> None:
        scores = np.random.default_rng(3).normal(size=(self.labels.size, 10))
        manifest = generate_instance_dependent(self.labels, scores, 0.3, 4, "toy")
        self.assertEqual(manifest.per_sample_transition.shape, (self.labels.size, 10))
        np.testing.assert_allclose(manifest.per_sample_transition.sum(axis=1), 1.0)



if __name__ == "__main__":
    unittest.main()
