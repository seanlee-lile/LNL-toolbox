import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from lnl_toolbox.data.noisy_dataset import NoisyTargetDataset
from lnl_toolbox.noise import (
    KnownTransition,
    NoiseManifest,
    generate_class_conditional,
    generate_instance_dependent,
    generate_pairflip,
    generate_symmetric,
    validate_transition_matrix,
)
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    prepare_noise_manifest,
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

    def test_per_class_symmetric_sampling_flips_each_class_at_requested_rate(self) -> None:
        manifest = generate_symmetric(
            self.labels, 10, 0.2, 7, "toy", sampling="per_class"
        )
        for class_index in range(10):
            class_mask = manifest.clean_targets == class_index
            self.assertEqual(
                int(manifest.flip_mask[class_mask].sum()),
                int(round(0.2 * int(class_mask.sum()))),
            )
        self.assertEqual(manifest.metadata["sampling"], "per_class")

    def test_legacy_numpy_per_class_sampling_is_reproducible(self) -> None:
        first = generate_symmetric(
            self.labels, 10, 0.2, 7, "toy", sampling="per_class", rng="numpy_legacy"
        )
        second = generate_symmetric(
            self.labels, 10, 0.2, 7, "toy", sampling="per_class", rng="numpy_legacy"
        )
        np.testing.assert_array_equal(first.noisy_targets, second.noisy_targets)
        self.assertEqual(first.metadata["rng"], "numpy_legacy")

    def test_transition_sampling_matches_per_sample_legacy_multinomial(self) -> None:
        labels = np.arange(10, dtype=np.int64)
        manifest = generate_symmetric(
            labels,
            10,
            0.4,
            7,
            "toy",
            sampling="transition",
            rng="numpy_legacy",
        )
        matrix = np.full((10, 10), 0.4 / 9.0)
        np.fill_diagonal(matrix, 0.6)
        random = np.random.RandomState(7)
        expected = np.array([
            random.multinomial(1, matrix[label], size=1)[0].argmax()
            for label in labels
        ])
        np.testing.assert_array_equal(manifest.noisy_targets, expected)
        self.assertEqual(manifest.metadata["sampling"], "transition")

    def test_pairflip_only_moves_to_next_class(self) -> None:
        manifest = generate_pairflip(self.labels, 10, 0.5, 3, "toy")
        expected = (manifest.clean_targets[manifest.flip_mask] + 1) % 10
        np.testing.assert_array_equal(manifest.noisy_targets[manifest.flip_mask], expected)

    def test_class_conditional_generator_is_reproducible_and_preserves_matrix(self) -> None:
        matrix = np.eye(3, dtype=np.float64)
        matrix[0] = [0.6, 0.4, 0.0]
        matrix[1] = [0.0, 0.6, 0.4]
        labels = np.tile(np.arange(3), 20)
        manifest = generate_class_conditional(labels, matrix, 0.4, 11, "fixture")
        repeated = generate_class_conditional(labels, matrix, 0.4, 11, "fixture")
        np.testing.assert_array_equal(manifest.noisy_targets, repeated.noisy_targets)
        np.testing.assert_allclose(manifest.transition_matrix, matrix)
        self.assertEqual(manifest.noise_type, "class_conditional")

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
        with self.assertRaisesRegex(ValueError, "per_sample_transition"):
            NoiseManifest(
                "cifar10", "fixture", 1, 0.1, self.labels, self.labels,
                per_sample_transition=probabilities,
            )

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

    def test_external_manifest_is_normalized_and_records_identity_without_labels(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 5, "cifar10")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            run_dir = root / "run"
            run_dir.mkdir()
            manifest.save(source)
            loaded, path = prepare_noise_manifest(
                {"noise": {"manifest": str(source)}},
                dataset="cifar10",
                clean_targets=self.labels,
                global_indices=np.arange(self.labels.size),
                num_classes=10,
                run_dir=run_dir,
            )
            summary = checkpoint_noise_metadata(
                loaded, path, run_dir, loaded.actual_rate, mode="external"
            )
        self.assertIsNotNone(loaded)
        self.assertEqual(path.name, "noise_manifest.npz")
        self.assertEqual(len(summary["manifest_sha256"]), 64)
        self.assertEqual(len(summary["source_manifest_sha256"]), 64)
        self.assertEqual(summary["noise_type"], "symmetric")
        self.assertNotIn("clean_targets", summary)
        self.assertNotIn("noisy_targets", summary)

    def test_manifest_can_be_a_superset_of_required_training_indices(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 5, "cifar10")
        required = np.arange(20, 80, dtype=np.int64)
        self.assertIs(
            manifest.validate_for(
                self.labels, "cifar10", 10, required_indices=required
            ),
            manifest,
        )
        with self.assertRaisesRegex(ValueError, "cover"):
            manifest.validate_for(
                self.labels,
                "cifar10",
                10,
                required_indices=np.array([self.labels.size]),
            )



if __name__ == "__main__":
    unittest.main()
