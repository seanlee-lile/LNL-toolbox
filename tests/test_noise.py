"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_noise.py ---
import json

# --- merged from test_noise.py ---
from pathlib import Path

# --- merged from test_noise.py ---
import tempfile

# --- merged from test_noise.py ---
import unittest

# --- merged from test_noise.py ---
import numpy as np

# --- merged from test_noise.py ---
import torch

# --- merged from test_noise.py ---
from torch.utils.data import DataLoader

# --- merged from test_noise.py ---
from lnl_toolbox.data.noisy_dataset import NoisyTargetDataset

# --- merged from test_noise.py ---
from lnl_toolbox.noise import KnownTransition, NoiseManifest, generate_class_conditional, generate_instance_dependent, generate_pairflip, generate_symmetric, validate_transition_matrix

# --- merged from test_noise.py ---
from lnl_toolbox.training.noisy_labels import checkpoint_noise_metadata, prepare_noise_manifest

# --- merged from test_noise.py ---
class _noise_NoiseTest(unittest.TestCase):

    def setUp(self) -> None:
        self.labels = np.tile(np.arange(10), 20)

    def test_zero_rate_keeps_labels(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.0, 1, 'toy')
        np.testing.assert_array_equal(manifest.clean_targets, manifest.noisy_targets)

    def test_flips_never_keep_original_class(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 1, 'toy')
        self.assertTrue(np.all(manifest.noisy_targets[manifest.flip_mask] != self.labels[manifest.flip_mask]))

    def test_seed_is_reproducible_and_manifest_roundtrips(self) -> None:
        first = generate_symmetric(self.labels, 10, 0.4, 7, 'toy')
        second = generate_symmetric(self.labels, 10, 0.4, 7, 'toy')
        np.testing.assert_array_equal(first.noisy_targets, second.noisy_targets)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'noise.npz'
            first.save(path)
            loaded = NoiseManifest.load(path)
        np.testing.assert_array_equal(first.noisy_targets, loaded.noisy_targets)
        np.testing.assert_array_equal(first.global_indices, loaded.global_indices)
        self.assertEqual(first.mapping_hash, loaded.mapping_hash)
        self.assertAlmostEqual(first.realized_rate, loaded.realized_rate)

    def test_legacy_v1_manifest_loads_with_explicit_default_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.npz'
            metadata = {'version': '1.0', 'dataset': 'toy', 'dataset_fingerprint': '', 'noise_type': 'symmetric', 'seed': 3, 'requested_rate': 0.2, 'metadata': {}}
            noisy = self.labels.copy()
            noisy[:2] = (noisy[:2] + 1) % 10
            np.savez_compressed(path, clean_targets=self.labels, noisy_targets=noisy, flip_mask=self.labels != noisy, transition_matrix=np.array([]), per_sample_transition=np.array([]), metadata_json=np.array(json.dumps(metadata)))
            loaded = NoiseManifest.load(path)
        self.assertEqual(loaded.version, '1.0')
        np.testing.assert_array_equal(loaded.global_indices, np.arange(self.labels.size))

    def test_different_seeds_usually_generate_different_noise(self) -> None:
        first = generate_symmetric(self.labels, 10, 0.4, 7, 'toy')
        second = generate_symmetric(self.labels, 10, 0.4, 8, 'toy')
        self.assertFalse(np.array_equal(first.noisy_targets, second.noisy_targets))

    def test_symmetric_realized_rate_matches_fixed_requested_count(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 7, 'toy')
        self.assertAlmostEqual(manifest.actual_rate, 0.4)

    def test_per_class_symmetric_sampling_flips_each_class_at_requested_rate(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.2, 7, 'toy', sampling='per_class')
        for class_index in range(10):
            class_mask = manifest.clean_targets == class_index
            self.assertEqual(int(manifest.flip_mask[class_mask].sum()), int(round(0.2 * int(class_mask.sum()))))
        self.assertEqual(manifest.metadata['sampling'], 'per_class')

    def test_legacy_numpy_per_class_sampling_is_reproducible(self) -> None:
        first = generate_symmetric(self.labels, 10, 0.2, 7, 'toy', sampling='per_class', rng='numpy_legacy')
        second = generate_symmetric(self.labels, 10, 0.2, 7, 'toy', sampling='per_class', rng='numpy_legacy')
        np.testing.assert_array_equal(first.noisy_targets, second.noisy_targets)
        self.assertEqual(first.metadata['rng'], 'numpy_legacy')

    def test_transition_sampling_matches_per_sample_legacy_multinomial(self) -> None:
        labels = np.arange(10, dtype=np.int64)
        manifest = generate_symmetric(labels, 10, 0.4, 7, 'toy', sampling='transition', rng='numpy_legacy')
        matrix = np.full((10, 10), 0.4 / 9.0)
        np.fill_diagonal(matrix, 0.6)
        random = np.random.RandomState(7)
        expected = np.array([random.multinomial(1, matrix[label], size=1)[0].argmax() for label in labels])
        np.testing.assert_array_equal(manifest.noisy_targets, expected)
        self.assertEqual(manifest.metadata['sampling'], 'transition')

    def test_pairflip_only_moves_to_next_class(self) -> None:
        manifest = generate_pairflip(self.labels, 10, 0.5, 3, 'toy')
        expected = (manifest.clean_targets[manifest.flip_mask] + 1) % 10
        np.testing.assert_array_equal(manifest.noisy_targets[manifest.flip_mask], expected)

    def test_transition_sampling_matches_legacy_multinomial(self) -> None:
        labels = np.arange(10, dtype=np.int64)
        manifest = generate_symmetric(labels, 10, 0.4, 7, 'toy', sampling='transition', rng='numpy_legacy')
        matrix = np.full((10, 10), 0.4 / 9.0)
        np.fill_diagonal(matrix, 0.6)
        random = np.random.RandomState(7)
        expected = np.array([random.multinomial(1, matrix[label], size=1)[0].argmax() for label in labels])
        np.testing.assert_array_equal(manifest.noisy_targets, expected)
        np.testing.assert_allclose(manifest.transition_matrix, matrix)
        self.assertEqual(manifest.metadata['sampling'], 'transition')

    def test_noise_config_sampling_contract_and_resume_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {'dataset': 'cifar10', 'clean_targets': self.labels, 'global_indices': np.arange(self.labels.size), 'num_classes': 10, 'dataset_targets': self.labels}
            transition_config = {'seed': 7, 'noise': {'name': 'symmetric', 'rate': 0.4, 'seed': 7, 'sampling': 'transition', 'rng': 'numpy_legacy'}}
            transition_dir = root / 'transition'
            transition_dir.mkdir()
            manifest, path = prepare_noise_manifest(transition_config, run_dir=transition_dir, **common)
            self.assertEqual(manifest.metadata['sampling'], 'transition')
            summary = checkpoint_noise_metadata(manifest, path, transition_dir, manifest.actual_rate, mode='generated')
            resumed, _ = prepare_noise_manifest(transition_config, run_dir=transition_dir, checkpoint_payload={'noise': summary}, **common)
            self.assertEqual(resumed.mapping_hash, manifest.mapping_hash)
            changed = {**transition_config, 'noise': {**transition_config['noise'], 'sampling': 'global'}}
            with self.assertRaisesRegex(ValueError, 'sampling'):
                prepare_noise_manifest(changed, run_dir=transition_dir, checkpoint_payload={'noise': summary}, **common)
            for name, noise, expected in (('default', {'name': 'symmetric', 'rate': 0.2, 'seed': 7}, 'global'), ('per_class', {'name': 'symmetric', 'rate': 0.2, 'seed': 7, 'sampling': 'per_class'}, 'per_class')):
                with self.subTest(name=name):
                    run_dir = root / name
                    run_dir.mkdir()
                    generated, _ = prepare_noise_manifest({'seed': 7, 'noise': noise}, run_dir=run_dir, **common)
                    self.assertEqual(generated.metadata['sampling'], expected)

    def test_invalid_or_non_symmetric_transition_sampling_fails_early(self):
        common = {'dataset': 'cifar10', 'clean_targets': self.labels, 'global_indices': np.arange(self.labels.size), 'num_classes': 10, 'dataset_targets': self.labels}
        cases = ({'noise': {'name': 'pairflip', 'rate': 0.2, 'sampling': 'transition'}}, {'noise': {'name': 'symmetric', 'rate': 0.2, 'sampling': 'invalid'}})
        for config in cases:
            with self.subTest(config=config):
                with tempfile.TemporaryDirectory() as directory:
                    run_dir = Path(directory)
                    with self.assertRaisesRegex(ValueError, 'sampling'):
                        prepare_noise_manifest(config, run_dir=run_dir, **common)
                    self.assertFalse((run_dir / 'noise_manifest.npz').exists())

    def test_mapping_hash_covers_indices_targets_and_context(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 7, 'toy')
        reordered = NoiseManifest(dataset=manifest.dataset, split=manifest.split, noise_type=manifest.noise_type, seed=manifest.seed, requested_rate=manifest.requested_rate, clean_targets=manifest.clean_targets[::-1], noisy_targets=manifest.noisy_targets[::-1], global_indices=manifest.global_indices[::-1], num_classes=manifest.num_classes)
        self.assertNotEqual(manifest.mapping_hash, reordered.mapping_hash)

    def test_noisy_dataset_maps_by_global_index_and_hides_clean_target(self) -> None:

        class IndexedDataset:
            indices = np.array([9, 3, 7], dtype=np.int64)

            def __len__(self) -> int:
                return len(self.indices)

            def __getitem__(self, item: int) -> dict[str, object]:
                index = int(self.indices[item])
                return {'input': torch.tensor([index]), 'target': index % 2, 'index': index}
        dataset = NoisyTargetDataset(IndexedDataset(), global_indices=np.array([3, 7, 9]), noisy_targets=np.array([1, 2, 4]))
        self.assertEqual(set(dataset[0]), {'input', 'target', 'index'})
        self.assertEqual(dataset[0]['target'], 4)
        seen = {}
        loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(2))
        for batch in loader:
            seen[int(batch['index'].item())] = int(batch['target'].item())
        self.assertEqual(seen, {3: 1, 7: 2, 9: 4})

    def test_instance_dependent_manifest_tracks_per_sample_probabilities(self) -> None:
        scores = np.random.default_rng(3).normal(size=(self.labels.size, 10))
        manifest = generate_instance_dependent(self.labels, scores, 0.3, 4, 'toy')
        self.assertEqual(manifest.per_sample_transition.shape, (self.labels.size, 10))
        np.testing.assert_allclose(manifest.per_sample_transition.sum(axis=1), 1.0)

    def test_manifest_validates_dataset_identity_and_label_alignment(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 1, 'cifar10')
        self.assertIs(manifest.validate_for(self.labels, 'CIFAR-10', 10), manifest)
        with self.assertRaisesRegex(ValueError, 'dataset'):
            manifest.validate_for(self.labels, 'cifar100', 10)
        with self.assertRaisesRegex(ValueError, 'index.*outside.*namespace'):
            manifest.validate_for(self.labels[:-1], 'cifar10', 10)
        shifted = np.roll(self.labels, 1)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            manifest.validate_for(shifted, 'cifar10', 10)

    def test_manifest_rejects_invalid_target_range_and_probability_shape(self) -> None:
        bad_targets = self.labels.copy()
        bad_targets[0] = 10
        manifest = NoiseManifest('cifar10', 'fixture', 1, 0.1, self.labels, bad_targets)
        with self.assertRaisesRegex(ValueError, 'noisy_targets'):
            manifest.validate_for(self.labels, 'cifar10', 10)
        probabilities = np.full((self.labels.size, 9), 1.0 / 9.0)
        with self.assertRaisesRegex(ValueError, 'per_sample_transition'):
            NoiseManifest('cifar10', 'fixture', 1, 0.1, self.labels, self.labels, per_sample_transition=probabilities)

    def test_transition_matrix_validation_and_known_provider(self) -> None:
        matrix = np.eye(10, dtype=np.float64) * 0.9
        matrix[np.arange(10), (np.arange(10) + 1) % 10] = 0.1
        provider = KnownTransition(matrix)
        tensor = provider.as_tensor(device='cpu', dtype=torch.float32)
        self.assertEqual(provider.num_classes, 10)
        self.assertEqual(tensor.shape, (10, 10))
        self.assertEqual(tensor.dtype, torch.float32)
        np.testing.assert_allclose(tensor.numpy(), matrix, rtol=1e-06, atol=1e-08)
        self.assertAlmostEqual(float(tensor[0, 1]), 0.1)
        self.assertEqual(float(tensor[1, 0]), 0.0)
        manifest = NoiseManifest('cifar10', 'fixture', 1, 0.1, self.labels, self.labels, transition_matrix=matrix)
        np.testing.assert_array_equal(KnownTransition.from_manifest(manifest).matrix, matrix)

    def test_transition_matrix_rejects_invalid_probabilities(self) -> None:
        invalid = (np.ones((2, 3)), np.array([[1.1, -0.1], [0.0, 1.0]]), np.array([[0.2, 0.2], [0.0, 1.0]]), np.array([[np.nan, 0.0], [0.0, 1.0]]))
        for matrix in invalid:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    validate_transition_matrix(matrix)

    def test_external_manifest_is_normalized_and_records_identity_without_labels(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 5, 'cifar10')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.npz'
            run_dir = root / 'run'
            run_dir.mkdir()
            manifest.save(source)
            loaded, path = prepare_noise_manifest({'noise': {'manifest': str(source)}}, dataset='cifar10', clean_targets=self.labels, global_indices=np.arange(self.labels.size), num_classes=10, run_dir=run_dir)
            summary = checkpoint_noise_metadata(loaded, path, run_dir, loaded.actual_rate, mode='external')
        self.assertIsNotNone(loaded)
        self.assertEqual(path.name, 'noise_manifest.npz')
        self.assertEqual(len(summary['manifest_sha256']), 64)
        self.assertEqual(len(summary['source_manifest_sha256']), 64)
        self.assertEqual(summary['noise_type'], 'symmetric')
        self.assertNotIn('clean_targets', summary)
        self.assertNotIn('noisy_targets', summary)

    def test_manifest_can_be_a_superset_of_required_training_indices(self) -> None:
        manifest = generate_symmetric(self.labels, 10, 0.4, 5, 'cifar10')
        required = np.arange(20, 80, dtype=np.int64)
        self.assertIs(manifest.validate_for(self.labels, 'cifar10', 10, required_indices=required), manifest)
        with self.assertRaisesRegex(ValueError, 'cover'):
            manifest.validate_for(self.labels, 'cifar10', 10, required_indices=np.array([self.labels.size]))

# --- merged from test_split_noise_manifest.py ---
import tempfile

# --- merged from test_split_noise_manifest.py ---
import unittest

# --- merged from test_split_noise_manifest.py ---
from pathlib import Path

# --- merged from test_split_noise_manifest.py ---
import numpy as np

# --- merged from test_split_noise_manifest.py ---
from lnl_toolbox.data.torch_cifar import train_validation_split

# --- merged from test_split_noise_manifest.py ---
from lnl_toolbox.noise.generators import generate_symmetric

# --- merged from test_split_noise_manifest.py ---
from lnl_toolbox.noise.manifest import NoiseManifest

# --- merged from test_split_noise_manifest.py ---
from lnl_toolbox.noise.split_manifest import _validate_symmetric_rate, generate_split_symmetric_manifest

# --- merged from test_split_noise_manifest.py ---
from scripts.prepare_split_noise_manifest import _prepare_destination

# --- merged from test_split_noise_manifest.py ---
class _split_noise_manifest_SplitNoiseManifestTest(unittest.TestCase):

    def test_classwise_legacy_matches_reference_sequence(self) -> None:
        labels = np.repeat(np.arange(3), 6)
        train, validation = train_validation_split(labels, 6, 7, strategy='classwise_legacy', rng='numpy_legacy')
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
        manifest = generate_split_symmetric_manifest(labels, (train, validation), num_classes=3, rate=0.5, seed=4, dataset='toy', split_names=('train', 'validation'))
        expected = [generate_symmetric(labels[indices], 3, 0.5, 4, 'toy', sampling='transition', rng='numpy_legacy').noisy_targets for indices in (train, validation)]
        np.testing.assert_array_equal(manifest.noisy_targets, np.concatenate(expected))
        np.testing.assert_array_equal(manifest.global_indices, np.concatenate((train, validation)))
        self.assertEqual(manifest.metadata['split_names'], ['train', 'validation'])
        self.assertEqual(manifest.metadata['rng_scope'], 'per_split')

    def test_manifest_roundtrip_preserves_provenance_and_mapping(self):
        labels = np.tile(np.arange(2), 10)
        train = np.arange(0, 12)
        validation = np.arange(12, 20)
        manifest = generate_split_symmetric_manifest(labels, (train, validation), num_classes=2, rate=0.3, seed=2, dataset='toy', split_names=('train', 'validation'))
        np.testing.assert_allclose(1.0 - np.diag(manifest.transition_matrix), 0.3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.npz'
            manifest.save(path)
            restored = NoiseManifest.load(path)
        np.testing.assert_array_equal(restored.global_indices, manifest.global_indices)
        np.testing.assert_array_equal(restored.noisy_targets, manifest.noisy_targets)
        self.assertEqual(restored.mapping_hash, manifest.mapping_hash)
        self.assertEqual(restored.metadata, manifest.metadata)

    def test_invalid_split_alignment_is_rejected(self) -> None:
        labels = np.arange(12) % 3
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            generate_split_symmetric_manifest(labels, (np.arange(8), np.arange(7, 12)), num_classes=3, rate=0.2, seed=1, dataset='toy')
        with self.assertRaisesRegex(ValueError, 'split_names'):
            generate_split_symmetric_manifest(labels, (np.arange(8), np.arange(8, 12)), num_classes=3, rate=0.2, seed=1, dataset='toy', split_names=('train', 'train'))
        with self.assertRaisesRegex(ValueError, 'integer dtype'):
            generate_split_symmetric_manifest(labels, (np.array([0.0, 1.0]), np.arange(2, 12)), num_classes=3, rate=0.2, seed=1, dataset='toy')

    def test_rate_mismatch_and_script_overwrite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'implied rate'):
            _validate_symmetric_rate(np.eye(3), 0.2)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'nested' / 'manifest.npz'
            self.assertEqual(_prepare_destination(destination), destination)
            self.assertTrue(destination.parent.is_dir())
            destination.write_bytes(b'existing')
            with self.assertRaisesRegex(FileExistsError, 'refusing to overwrite'):
                _prepare_destination(destination)

# --- merged from test_treatments.py ---
import unittest

# --- merged from test_treatments.py ---
import torch

# --- merged from test_treatments.py ---
from lnl_toolbox.selectors import SelectionInput, SelectionResult

# --- merged from test_treatments.py ---
from lnl_toolbox.treatments import ContributionResult, ReductionSpec, SelectorContributionAdapter, reduce_per_sample_loss, validate_contribution_result

# --- merged from test_treatments.py ---
class _treatments_SampleTreatmentTest(unittest.TestCase):

    def test_selector_adapter_preserves_mask_metrics_and_adds_identity_weights(self):

        class FixedSelector:

            def select(self, selection_input):
                return SelectionResult(selected_mask=torch.tensor([False, True, True]), metrics={'selected_samples': 2.0, 'selected_ratio': 2.0 / 3.0})
        result = SelectorContributionAdapter(FixedSelector()).resolve(SelectionInput(scores=torch.tensor([0.7, 0.2, 0.3]), sample_indices=torch.tensor([8, 2, 5])))
        self.assertEqual(result.selected_mask.tolist(), [False, True, True])
        self.assertTrue(torch.equal(result.sample_weights, torch.ones(3)))
        self.assertEqual(result.metrics['selected_samples'], 2.0)
        self.assertEqual(result.metrics['selected_ratio'], 2.0 / 3.0)

    def test_weight_sum_mean_matches_selected_hard_mask_mean(self):
        losses = torch.tensor([1.0, 2.0, 7.0, 9.0], requires_grad=True)
        contribution = ContributionResult(selected_mask=torch.tensor([True, False, True, False]), sample_weights=torch.ones(4))
        objective = reduce_per_sample_loss(losses, contribution)
        self.assertEqual(objective.item(), losses[[0, 2]].mean().item())
        objective.backward()
        self.assertTrue(torch.equal(losses.grad, torch.tensor([0.5, 0.0, 0.5, 0.0])))

    def test_continuous_weights_and_reduction_modes(self):
        losses = torch.tensor([1.0, 3.0, 8.0])
        contribution = ContributionResult(selected_mask=torch.tensor([True, True, False]), sample_weights=torch.tensor([1.0, 3.0, 100.0]))
        self.assertEqual(reduce_per_sample_loss(losses, contribution).item(), 2.5)
        self.assertAlmostEqual(reduce_per_sample_loss(losses, contribution, ReductionSpec('batch_mean')).item(), 10.0 / 3.0, places=6)
        self.assertEqual(reduce_per_sample_loss(losses, contribution, ReductionSpec('sum')).item(), 10.0)

    def test_reduction_spec_rejects_unknown_normalization(self):
        with self.assertRaisesRegex(ValueError, 'normalization'):
            ReductionSpec('mean')

    def test_contribution_validation_rejects_invalid_weights_and_empty_result(self):
        cases = ((ContributionResult(selected_mask=torch.tensor([True, False]), sample_weights=torch.tensor([1.0, -1.0])), 'non-negative'), (ContributionResult(selected_mask=torch.tensor([True, False]), sample_weights=torch.tensor([float('nan'), 1.0])), 'finite'), (ContributionResult(selected_mask=torch.tensor([False, False]), sample_weights=torch.ones(2)), 'positive contribution'), (ContributionResult(selected_mask=torch.tensor([True, True]), sample_weights=torch.zeros(2)), 'positive contribution'))
        for result, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_contribution_result(result, batch_size=2, device=torch.device('cpu'))

    def test_reducer_rejects_nonfinite_loss_without_fallback(self):
        contribution = ContributionResult(selected_mask=torch.tensor([True, False]), sample_weights=torch.ones(2))
        with self.assertRaisesRegex(ValueError, 'finite'):
            reduce_per_sample_loss(torch.tensor([float('nan'), 2.0]), contribution)
