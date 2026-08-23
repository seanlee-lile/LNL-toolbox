"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_data_adapters.py ---
import gzip

# --- merged from test_data_adapters.py ---
from pathlib import Path

# --- merged from test_data_adapters.py ---
import struct

# --- merged from test_data_adapters.py ---
import tempfile

# --- merged from test_data_adapters.py ---
import unittest

# --- merged from test_data_adapters.py ---
from unittest.mock import patch

# --- merged from test_data_adapters.py ---
import numpy as np

# --- merged from test_data_adapters.py ---
from PIL import Image

# --- merged from test_data_adapters.py ---
import torch

# --- merged from test_data_adapters.py ---
from lnl_toolbox.data.cifar import CifarData

# --- merged from test_data_adapters.py ---
from lnl_toolbox.data.cifar_n import CifarNAdapter

# --- merged from test_data_adapters.py ---
from lnl_toolbox.data.contracts import DataSpec

# --- merged from test_data_adapters.py ---
from lnl_toolbox.data.profile import DatasetSemanticHints, KnowledgeState, Modality, NoiseStatus

# --- merged from test_data_adapters.py ---
from lnl_toolbox.data.registry import DatasetRegistry

# --- merged from test_data_adapters.py ---
from lnl_toolbox.data.mnist import MnistAdapter

# --- merged from test_data_adapters.py ---
from lnl_toolbox.data.real_noise import Animal10NAdapter, Clothing1MAdapter

# --- merged from test_data_adapters.py ---
from lnl_toolbox.data.sources import CifarAdapter, SyntheticAdapter, UciBinaryAdapter

# --- merged from test_data_adapters.py ---
from lnl_toolbox.training.data_service import DataService

# --- merged from test_data_adapters.py ---
def _data_adapters__cifar(size: int, split: str, classes: int=10) -> CifarData:
    labels = np.arange(size, dtype=np.int64) % classes
    return CifarData(np.zeros((size, 32, 32, 3), dtype=np.uint8), labels, tuple(map(str, range(classes))), split, f'cifar{classes}')

# --- merged from test_data_adapters.py ---
def _data_adapters__write_idx(root: Path, split: str, count: int) -> None:
    prefix = 'train' if split == 'train' else 't10k'
    images = np.arange(count * 28 * 28, dtype=np.uint8).reshape(count, 28, 28)
    labels = np.arange(count, dtype=np.uint8) % 10
    with gzip.open(root / f'{prefix}-images-idx3-ubyte.gz', 'wb') as handle:
        handle.write(struct.pack('>IIII', 2051, count, 28, 28) + images.tobytes())
    with gzip.open(root / f'{prefix}-labels-idx1-ubyte.gz', 'wb') as handle:
        handle.write(struct.pack('>II', 2049, count) + labels.tobytes())

# --- merged from test_data_adapters.py ---
def _data_adapters__animal_record(index: int, label: int) -> bytes:
    pixels = np.full(3 * 64 * 64, index % 256, dtype=np.uint8)
    return index.to_bytes(4, 'little') + label.to_bytes(4, 'little') + pixels.tobytes()

# --- merged from test_data_adapters.py ---
class _data_adapters_DataAdapterFixtureTest(unittest.TestCase):

    def test_cifar_and_cifar_n_observed_clean_separation(self) -> None:
        corpus = _data_adapters__cifar(6, 'train')
        with patch('lnl_toolbox.data.sources.load_cifar10', return_value=corpus):
            split = CifarAdapter('cifar10', 10).load(DataSpec('cifar10'), 'train', seed=1)
        np.testing.assert_array_equal(split.observed_targets, split.clean_targets)
        self.assertEqual(split.global_indices.tolist(), list(range(6)))
        registry = DatasetRegistry()
        registry.add(CifarAdapter('cifar10', 10))
        with patch('lnl_toolbox.data.sources.load_cifar10', return_value=corpus):
            profile = DataService(registry).inspect({'data': {'name': 'cifar10'}}).profile
        self.assertEqual(profile.modality, Modality.IMAGE)
        self.assertEqual(profile.input_shape, (32, 32, 3))
        self.assertEqual(profile.channels, 3)
        self.assertEqual(profile.clean_train_labels, KnowledgeState.AVAILABLE)
        self.assertEqual(profile.noise.status, NoiseStatus.CLEAN)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            noisy = np.asarray([1, 1, 2, 3, 4, 5], dtype=np.int64)
            torch.save({'aggre_label': noisy, 'clean_label': corpus.labels.copy()}, root / 'CIFAR-10_human.pt')
            with patch('lnl_toolbox.data.cifar_n.load_cifar10', return_value=corpus):
                split = CifarNAdapter('cifar10n', 10).load(DataSpec('cifar10n', root=root), 'train', seed=1)
            self.assertEqual(split.observed_targets.tolist(), noisy.tolist())
            self.assertEqual(split.clean_targets.tolist(), corpus.labels.tolist())
            self.assertIn('human_annotation', split.source)

    def test_mnist_official_idx_gzip_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _data_adapters__write_idx(root, 'train', 8)
            _data_adapters__write_idx(root, 'test', 4)
            adapter = MnistAdapter('mnist')
            adapter.validate(DataSpec('mnist', root=root))
            split = adapter.load(DataSpec('mnist', root=root), 'train', seed=1)
        self.assertEqual(len(split), 8)
        self.assertEqual(split.inputs.shape, (8, 28, 28))
        self.assertEqual(split.source, 'official_idx_gzip')

    def test_clothing1m_and_animal10n_lazy_file_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'sample.jpg'
            Image.new('RGB', (4, 4), 'white').save(image)
            for name in ('noisy_train_key_list.txt', 'clean_val_key_list.txt', 'clean_test_key_list.txt'):
                (root / name).write_text('sample.jpg\n', encoding='utf-8')
            (root / 'noisy_label_kv.txt').write_text('sample.jpg 3\n', encoding='utf-8')
            (root / 'clean_label_kv.txt').write_text('sample.jpg 3\n', encoding='utf-8')
            clothing = Clothing1MAdapter()
            clothing.validate(DataSpec('clothing1m', root=root))
            train = clothing.load(DataSpec('clothing1m', root=root), 'train', seed=1)
            test = clothing.load(DataSpec('clothing1m', root=root), 'test', seed=1)
            self.assertIsNone(train.clean_targets)
            self.assertEqual(test.clean_targets.tolist(), [3])
            service = DataService(
                DatasetRegistry((clothing,)),
                LocalDatasetCatalog(root / 'catalog.json'),
            )
            service.register('clothing-fixture', 'clothing1m', {'root': root})
            report = service.inspect('clothing-fixture')
            self.assertEqual(report.profile.noise.status, NoiseStatus.NOISY)
            self.assertEqual(report.profile.noise.origin.value, 'native')
            self.assertEqual(report.profile.clean_train_labels, KnowledgeState.UNAVAILABLE)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split_name in ('training', 'testing'):
                split_root = root / split_name
                split_root.mkdir(parents=True)
                for label in range(10):
                    Image.new('RGB', (4, 4), 'white').save(split_root / f'{label}_one.png')
            animal = Animal10NAdapter()
            self.assertIsInstance(clothing.semantic_hints, DatasetSemanticHints)
            self.assertEqual(clothing.semantic_hints.noise.status, NoiseStatus.NOISY)
            self.assertEqual(clothing.semantic_hints.noise.origin.value, 'native')
            self.assertEqual(clothing.semantic_hints.clean_train_labels, KnowledgeState.UNAVAILABLE)
            self.assertEqual(animal.semantic_hints.clean_train_labels, KnowledgeState.UNAVAILABLE)
            train = animal.load(DataSpec('animal10n', root=root), 'train', seed=1)
            test = animal.load(DataSpec('animal10n', root=root), 'test', seed=1)
            self.assertIsNone(train.clean_targets)
            self.assertEqual(len(test), 10)
            self.assertIsNotNone(test.clean_targets)

    def test_animal10n_official_binary_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data_batch_1.bin').write_bytes(_data_adapters__animal_record(0, 2) + _data_adapters__animal_record(1, 7))
            (root / 'test_batch.bin').write_bytes(_data_adapters__animal_record(2, 4))
            adapter = Animal10NAdapter()
            adapter.validate(DataSpec('animal10n', root=root))
            train = adapter.load(DataSpec('animal10n', root=root), 'train', seed=1)
            test = adapter.load(DataSpec('animal10n', root=root), 'test', seed=1)
            self.assertEqual(train.inputs.shape, (2, 64, 64, 3))
            self.assertEqual(train.observed_targets.tolist(), [2, 7])
            self.assertEqual(test.clean_targets.tolist(), [4])

    def test_uci_fits_preprocessing_on_training_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'heart.csv'
            source.write_text('value,color,label\n' + '\n'.join((f"{index},{('red' if index % 2 else 'blue')},{index % 2}" for index in range(20))) + '\n', encoding='utf-8')
            spec = DataSpec('uci_binary', path=source, options={'preprocessing': {'format': 'csv', 'target_column': 'label', 'has_header': True, 'standardize': True}, 'split': {'validation_fraction': 0.2, 'test_fraction': 0.2, 'seed': 4}})
            adapter = UciBinaryAdapter()
            train = adapter.load(spec, 'train', seed=4)
            validation = adapter.load(spec, 'validation', seed=4)
            test = adapter.load(spec, 'test', seed=4)
            self.assertEqual(len(train) + len(validation) + len(test), 20)
            self.assertFalse(set(train.global_indices) & set(validation.global_indices))
            self.assertTrue(np.isfinite(validation.inputs).all())
            registry = DatasetRegistry()
            registry.add(adapter)
            profile = DataService(registry).inspect({'data': {'name': 'uci_binary', 'path': str(source), **dict(spec.options)}}).profile
            self.assertEqual(profile.modality, Modality.TABULAR)
            self.assertEqual(profile.input_shape, (3,))
            self.assertEqual(profile.available_splits, ('test', 'train', 'validation'))

    def test_synthetic_splits_have_disjoint_stable_indices(self) -> None:
        spec = DataSpec('synthetic_multiclass', options={'num_classes': 3, 'dimension': 3, 'train_size': 9, 'validation_size': 6, 'test_size': 6})
        adapter = SyntheticAdapter('synthetic_multiclass')
        train = adapter.load(spec, 'train', seed=7)
        validation = adapter.load(spec, 'validation', seed=7)
        test = adapter.load(spec, 'test', seed=7)
        self.assertFalse(set(train.global_indices) & set(validation.global_indices))
        self.assertFalse(set(validation.global_indices) & set(test.global_indices))
        repeated = adapter.load(spec, 'train', seed=7)
        np.testing.assert_array_equal(train.inputs, repeated.inputs)

# --- merged from test_cifar_reader.py ---
import pickle

# --- merged from test_cifar_reader.py ---
import tempfile

# --- merged from test_cifar_reader.py ---
import unittest

# --- merged from test_cifar_reader.py ---
from pathlib import Path

# --- merged from test_cifar_reader.py ---
import numpy as np

# --- merged from test_cifar_reader.py ---
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100, summarize_cifar

# --- merged from test_cifar_reader.py ---
def _cifar_reader_dump(path: Path, payload: dict) -> None:
    with path.open('wb') as handle:
        pickle.dump(payload, handle)

# --- merged from test_cifar_reader.py ---
class _cifar_reader_CifarReaderTest(unittest.TestCase):

    def test_cifar10_pickle_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flat = np.arange(2 * 3072, dtype=np.uint8).reshape(2, 3072)
            for index in range(1, 6):
                _cifar_reader_dump(root / f'data_batch_{index}', {b'data': flat, b'labels': [0, 1]})
            _cifar_reader_dump(root / 'test_batch', {b'data': flat, b'labels': [1, 0]})
            _cifar_reader_dump(root / 'batches.meta', {b'label_names': [b'zero', b'one']})
            data = load_cifar10(root, 'train')
        self.assertEqual(data.images.shape, (10, 32, 32, 3))
        self.assertEqual(data.class_names, ('zero', 'one'))

    def test_cifar100_pickle_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flat = np.zeros((3, 3072), dtype=np.uint8)
            _cifar_reader_dump(root / 'train', {b'data': flat, b'fine_labels': [0, 1, 1]})
            _cifar_reader_dump(root / 'meta', {b'fine_label_names': [b'zero', b'one']})
            data = load_cifar100(root, 'train')
            summary = summarize_cifar(data)
        self.assertEqual(data.images.shape, (3, 32, 32, 3))
        self.assertEqual(summary['class_count_max'], 2)

# --- merged from test_data_service.py ---
import gzip

# --- merged from test_data_service.py ---
import json

# --- merged from test_data_service.py ---
from pathlib import Path

# --- merged from test_data_service.py ---
import struct

# --- merged from test_data_service.py ---
import tempfile

# --- merged from test_data_service.py ---
import unittest

# --- merged from test_data_service.py ---
import numpy as np

# --- merged from test_data_service.py ---
import torch

# --- merged from test_data_service.py ---
from lnl_toolbox.data import DataRequirements, DataRole, DataSpec, DatasetRegistry, IndexedDatasetView, LocalDatasetCatalog, RawDatasetSplit

# --- merged from test_data_service.py ---
from lnl_toolbox.training.checkpoint import atomic_save, read_checkpoint

# --- merged from test_data_service.py ---
from lnl_toolbox.training.data_service import DATASETS, DataService, prepare_experiment_data

# --- merged from test_data_service.py ---
from lnl_toolbox.data.profile import DatasetDeclarations, KnowledgeState, Modality, NoiseRateInfo, NoiseRateStatus, NoiseStatus

# --- merged from test_data_service.py ---
class _data_service__FixtureAdapter:
    name = 'fixture'
    aliases = ('fixture-data',)

    def __init__(self, *, fail: bool=False) -> None:
        self.fail = fail

    def validate(self, spec: DataSpec) -> None:
        if self.fail:
            raise ValueError('broken fixture layout')
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError('fixture root missing')

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        self.validate(spec)
        count = 8 if split == 'train' else 4
        return RawDatasetSplit(inputs=np.arange(count * 2, dtype=np.float32).reshape(count, 2), observed_targets=np.arange(count, dtype=np.int64) % 2, global_indices=np.arange(count, dtype=np.int64), dataset=self.name, split=split, num_classes=2, source=str(spec.root))

# --- merged from test_data_service.py ---
class _data_service__NativeNoisyFixtureAdapter:
    name = 'native_noisy_fixture'
    aliases = ()

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError('native fixture root missing')

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        self.validate(spec)
        targets = {'train': np.asarray([5, 6], dtype=np.int64), 'validation': np.asarray([8, 9], dtype=np.int64), 'test': np.asarray([0, 1], dtype=np.int64)}[split]
        return RawDatasetSplit(inputs=np.arange(targets.size * 2, dtype=np.float32).reshape(targets.size, 2), observed_targets=targets, global_indices=np.arange(targets.size, dtype=np.int64), dataset=self.name, split=split, num_classes=10, clean_targets=None if split == 'train' else targets.copy(), source=str(spec.root))


class _data_service__CrossSplitCollisionAdapter:
    name = 'cross_split_collision_fixture'
    aliases = ()

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise FileNotFoundError('collision fixture root missing')

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        self.validate(spec)
        observed = {'train': [1, 2], 'validation': [8, 9], 'test': [3, 4]}[split]
        targets = np.asarray(observed, dtype=np.int64)
        return RawDatasetSplit(
            inputs=np.arange(targets.size * 2, dtype=np.float32).reshape(targets.size, 2),
            observed_targets=targets,
            global_indices=np.arange(targets.size, dtype=np.int64),
            dataset=self.name,
            split=split,
            num_classes=10,
            clean_targets=targets.copy(),
            source=str(spec.root),
        )

# --- merged from test_data_service.py ---
def _data_service__config() -> dict:
    return {'seed': 9, 'data': {'name': 'synthetic_multiclass', 'num_classes': 3, 'dimension': 4, 'train_size': 30, 'validation_size': 12, 'test_size': 12}, 'noise': {'name': 'clean', 'rate': 0.0, 'seed': 9}, 'loader': {'batch_size': 6, 'num_workers': 0, 'drop_last': False}}

# --- merged from test_data_service.py ---
def _data_service__write_fashion_idx(root: Path, split: str, count: int) -> None:
    prefix = 'train' if split == 'train' else 't10k'
    images = np.arange(count * 28 * 28, dtype=np.uint8).reshape(count, 28, 28)
    labels = np.arange(count, dtype=np.uint8) % 10
    with gzip.open(root / f'{prefix}-images-idx3-ubyte.gz', 'wb') as handle:
        handle.write(struct.pack('>IIII', 2051, count, 28, 28) + images.tobytes())
    with gzip.open(root / f'{prefix}-labels-idx1-ubyte.gz', 'wb') as handle:
        handle.write(struct.pack('>II', 2049, count) + labels.tobytes())

# --- merged from test_data_service.py ---
class _data_service_DataServiceTest(unittest.TestCase):

    def test_native_validation_noisy_targets_use_validation_split_mapping(self) -> None:
        requirements = DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}), validation_targets='noisy', needs_noise_manifest=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = prepare_experiment_data({'data': {'name': 'native_noisy_fixture', 'root': str(root)}, 'loader': {'batch_size': 2, 'num_workers': 0}}, requirements=requirements, run_dir=root / 'run', seed=3, registry=DatasetRegistry((_data_service__NativeNoisyFixtureAdapter(),)))
            validation = prepared.dataset_for(DataRole.NOISY_VALIDATION)
            self.assertEqual([int(validation[index]['target']) for index in range(2)], [8, 9])
            self.assertEqual(prepared.train_split.observed_targets.tolist(), [5, 6])

    def test_native_noisy_data_fails_before_manifest_required_training(self) -> None:
        requirements = DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}), validation_targets='noisy', needs_noise_manifest=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'native observed noisy labels without train clean targets'):
                prepare_experiment_data({'data': {'name': 'native_noisy_fixture', 'root': str(root)}, 'loader': {'batch_size': 2, 'num_workers': 0}}, requirements=requirements, run_dir=root / 'run', seed=3, registry=DatasetRegistry((_data_service__NativeNoisyFixtureAdapter(),)))

    def test_synthetic_noise_manifest_and_validation_mapping_are_preserved(self) -> None:
        requirements = DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}), validation_targets='noisy')
        config = _data_service__config()
        config['noise'] = {'name': 'symmetric', 'rate': 0.4, 'seed': 9}
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_experiment_data(config, requirements=requirements, run_dir=directory, seed=9)
            self.assertIsNotNone(prepared.manifest)
            assert prepared.manifest is not None
            noisy_by_index = {int(index): int(target) for index, target in zip(prepared.manifest.global_indices, prepared.manifest.noisy_targets, strict=True)}
            train = prepared.dataset_for(DataRole.TRAIN)
            validation = prepared.dataset_for(DataRole.NOISY_VALIDATION)
            for offset in range(len(train)):
                sample = train[offset]
                self.assertEqual(int(sample['target']), noisy_by_index[int(sample['index'])])
            self.assertEqual(
                [int(validation[offset]['target']) for offset in range(len(validation))],
                prepared.validation_split.observed_targets.tolist(),
            )
            self.assertEqual(set(prepared.train_indices) & set(prepared.validation_indices), set())

    def test_independent_validation_indices_do_not_override_train_manifest(self) -> None:
        requirements = DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
            validation_targets='noisy',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = prepare_experiment_data(
                {
                    'data': {'name': 'cross_split_collision_fixture', 'root': str(root)},
                    'noise': {'name': 'symmetric', 'rate': 0.5, 'seed': 7},
                    'loader': {'batch_size': 2, 'num_workers': 0},
                },
                requirements=requirements,
                run_dir=root / 'run',
                seed=7,
                registry=DatasetRegistry((_data_service__CrossSplitCollisionAdapter(),)),
            )
            assert prepared.manifest is not None
            self.assertEqual(prepared.manifest.split, 'train')
            self.assertEqual(prepared.manifest.global_indices.tolist(), [0, 1])
            noisy_by_index = dict(zip(prepared.manifest.global_indices.tolist(), prepared.manifest.noisy_targets.tolist()))
            train = prepared.dataset_for(DataRole.TRAIN)
            validation = prepared.dataset_for(DataRole.NOISY_VALIDATION)
            self.assertEqual([int(train[index]['target']) for index in range(2)], [noisy_by_index[0], noisy_by_index[1]])
            self.assertEqual([int(validation[index]['target']) for index in range(2)], [8, 9])

    def test_raw_split_sample_keys_include_the_split_namespace(self) -> None:
        train = RawDatasetSplit(np.zeros((1, 2)), np.array([0]), np.array([0]), 'fixture', 'train', 2)
        validation = RawDatasetSplit(np.zeros((1, 2)), np.array([1]), np.array([0]), 'fixture', 'validation', 2)
        self.assertNotEqual(train.sample_key(0), validation.sample_key(0))
        with self.assertRaisesRegex(KeyError, 'outside split'):
            train.sample_key(1)

    def test_indexed_view_rejects_targets_from_another_namespace(self) -> None:
        split = RawDatasetSplit(np.zeros((1, 2)), np.array([0]), np.array([0]), 'fixture', 'train', 2)
        with self.assertRaisesRegex(KeyError, 'outside'):
            IndexedDatasetView(split, targets_by_index={1: 0})

    def test_management_status_path_and_real_split_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = DatasetRegistry((_data_service__FixtureAdapter(),))
            catalog = LocalDatasetCatalog(root / 'catalog.json')
            service = DataService(registry, catalog)
            self.assertEqual(service.status('fixture').status, 'missing')
            registered = service.register('lab-fixture', 'fixture-data', {'root': root})
            self.assertEqual(registered.status, 'incomplete')
            self.assertEqual(service.path('lab-fixture'), root)
            inspected = service.inspect('lab-fixture')
            self.assertEqual(inspected.status, 'ready')
            self.assertEqual(inspected.train_samples, 8)
            self.assertEqual(inspected.test_samples, 4)
            self.assertEqual(inspected.classes, 2)
            self.assertEqual(len(inspected.fingerprint or ''), 64)
            self.assertEqual(service.status('lab-fixture').status, 'ready')

    def test_failed_adapter_is_never_reported_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = DataService(DatasetRegistry((_data_service__FixtureAdapter(fail=True),)), LocalDatasetCatalog(root / 'catalog.json'))
            service.register('broken', 'fixture', {'root': root})
            report = service.inspect('broken')
            self.assertEqual(report.status, 'incomplete')
            self.assertIn('broken fixture layout', report.error or '')
            self.assertNotEqual(service.status('broken').status, 'ready')

    def test_portable_config_resolves_unique_local_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = DataService(DatasetRegistry((_data_service__FixtureAdapter(),)), LocalDatasetCatalog(root / 'catalog.json'))
            service.register('lab', 'fixture', {'root': root})
            resolved = service.resolve_config({'data': {'name': 'fixture'}})
            self.assertEqual(Path(resolved['data']['root']), root)
            self.assertEqual(resolved['local_dataset']['alias'], 'lab')

    def test_portable_config_rejects_ambiguous_local_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = DataService(DatasetRegistry((_data_service__FixtureAdapter(),)), LocalDatasetCatalog(root / 'catalog.json'))
            service.register('first', 'fixture', {'root': root})
            service.register('second', 'fixture', {'root': root})
            with self.assertRaisesRegex(ValueError, 'multiple local registrations'):
                service.resolve_config({'data': {'name': 'fixture'}})

    def test_registry_alias_and_unknown_dataset(self) -> None:
        self.assertEqual(DATASETS.get('cifar-10').name, 'cifar10')
        self.assertEqual(DATASETS.get('fashionmnist').name, 'fashion_mnist')
        with self.assertRaisesRegex(ValueError, 'unknown dataset'):
            DATASETS.get('cifar11')

    def test_spec_preserves_legacy_options(self) -> None:
        spec = DataSpec.from_mapping({'name': 'cifar-10', 'root': 'data', 'num_val': 7})
        self.assertEqual(spec.name, 'cifar_10')
        self.assertEqual(spec.options['num_val'], 7)

    def test_roles_batch_schema_indices_views_and_loader_seed(self) -> None:
        requirements = DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.TRAIN_EVAL, DataRole.CLEAN_VALIDATION, DataRole.TEST}), views=('weak', 'strong'))
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_experiment_data(_data_service__config(), requirements=requirements, run_dir=directory, seed=9)
            batch = next(iter(prepared.loader(DataRole.TRAIN, epoch=2)))
            self.assertEqual(set(batch), {'input', 'target', 'index', 'views', 'strong_input'})
            self.assertNotIn('clean_target', batch)
            self.assertEqual(set(batch['views']), {'weak', 'strong'})
            first = torch.cat([value['index'] for value in prepared.loader(DataRole.TRAIN, epoch=4)])
            repeated = torch.cat([value['index'] for value in prepared.loader(DataRole.TRAIN, epoch=4)])
            another = torch.cat([value['index'] for value in prepared.loader(DataRole.TRAIN, epoch=5)])
            self.assertTrue(torch.equal(first, repeated))
            self.assertFalse(torch.equal(first, another))
            self.assertEqual(set(first.tolist()), set(prepared.train_indices.tolist()))
            chosen = prepared.train_indices[::2]
            probabilities = {int(index): float(offset) for offset, index in enumerate(chosen)}
            dynamic = prepared.dynamic_dataset(chosen, overlays={'clean_probability': probabilities})
            self.assertEqual(dynamic.indices.tolist(), chosen.tolist())
            self.assertIn('clean_probability', dynamic[0])

    def test_official_fashion_idx_enters_unified_service(self) -> None:
        requirements = DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.TEST}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, run_root = (root / 'fashion', root / 'run')
            data_root.mkdir()
            _data_service__write_fashion_idx(data_root, 'train', 20)
            _data_service__write_fashion_idx(data_root, 'test', 10)
            config = {'data': {'name': 'fashion_mnist', 'root': str(data_root), 'validation_size': 0}, 'noise': {'name': 'clean', 'rate': 0.0, 'seed': 3}, 'loader': {'batch_size': 5, 'num_workers': 0}}
            prepared = prepare_experiment_data(config, requirements=requirements, run_dir=run_root, seed=3)
            batch = next(iter(prepared.loader(DataRole.TRAIN, epoch=0)))
            self.assertEqual(set(batch), {'input', 'target', 'index'})
            self.assertNotIn('clean_target', batch)
            self.assertEqual(tuple(batch['input'].shape[1:]), (1, 28, 28))

    def test_registered_fashion_mnist_uses_automatic_training_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'FashionMNIST' / 'raw'
            data_root.mkdir(parents=True)
            _data_service__write_fashion_idx(data_root, 'train', 40)
            _data_service__write_fashion_idx(data_root, 'test', 20)
            service = DataService(DATASETS, LocalDatasetCatalog(root / 'catalog.json'))
            service.register('fashion', 'fashion_mnist', {'root': root})
            report, run_dir = service.verify('fashion', None, root / 'verify')
            self.assertEqual(report.status, 'ready')
            self.assertEqual(report.training_evidence['verification_profile'], 'automatic')
            self.assertIsNone(report.training_evidence['recipe'])
            final = json.loads((run_dir / 'final_metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(final['completed_epochs'], 1)
            manifest = json.loads((run_dir / 'data_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['dataset'], 'fashion_mnist')

    def test_manifest_checkpoint_round_trip_and_tamper_failure(self) -> None:
        requirements = DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.TEST}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = prepare_experiment_data(_data_service__config(), requirements=requirements, run_dir=root, seed=9)
            atomic_save({'model': {}}, root / 'last.pt')
            payload = read_checkpoint(root / 'last.pt')
            self.assertEqual(payload['data'], prepared.state_dict())
            prepare_experiment_data(_data_service__config(), requirements=requirements, run_dir=root, seed=9, checkpoint_payload=payload)
            manifest_path = root / 'data_manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['loader']['batch_size'] += 1
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                read_checkpoint(root / 'last.pt')

    def test_checkpoint_state_rejects_different_data(self) -> None:
        requirements = DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.TEST}))
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            original = prepare_experiment_data(_data_service__config(), requirements=requirements, run_dir=first, seed=9)
            changed = _data_service__config()
            changed['data']['train_size'] = 33
            current = prepare_experiment_data(changed, requirements=requirements, run_dir=second, seed=9)
            with self.assertRaisesRegex(ValueError, 'data identity mismatch'):
                current.load_state_dict(original.state_dict())

    def test_all_experiment_runners_use_only_unified_entry(self) -> None:
        root = Path(__file__).resolve().parents[1] / 'src' / 'lnl_toolbox' / 'training'
        runners = ('experiment.py', 'multi_model_experiment.py', 'binary_experiment.py', 'importance_reweighting_experiment.py', 'cwd_experiment.py', 'dual_t_experiment.py', 'dual_t_evidence_experiment.py', 'instance_transition_experiment.py', 't_revision_experiment.py', 'volminnet_experiment.py', 'upm_experiment.py', 'pcse_experiment.py', 'mc_ldce_experiment.py', 'cal_experiment.py', 'volmin_experiment.py', 'coteaching_experiment.py', 'cnlcu_experiment.py', 'lend_experiment.py', 'fine_experiment.py', 'dld_experiment.py', 'ca2c_experiment.py', 'dividemix_experiment.py', 'l2rw_experiment.py')
        forbidden = ('from lnl_toolbox.data.cifar', 'from lnl_toolbox.data.torch_cifar', 'DataLoader(', 'prepare_noisy_classification(')
        for filename in runners:
            source = (root / filename).read_text(encoding='utf-8')
            self.assertIn('prepare_experiment_data', source, filename)
            for pattern in forbidden:
                self.assertNotIn(pattern, source, f'{filename}: {pattern}')

    def test_inspect_generates_deterministic_profile_and_persists_it(self) -> None:
        registry = DatasetRegistry()
        registry.add(_data_service__FixtureAdapter())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = LocalDatasetCatalog(root / 'catalog.json')
            service = DataService(registry, catalog)
            service.register('fixture-local', 'fixture', {'root': root})
            first = service.inspect('fixture-local', seed=7)
            second = service.inspect('fixture-local', seed=7)
            self.assertIsNotNone(first.profile)
            self.assertEqual(first.profile, second.profile)
            self.assertEqual(first.profile.modality, Modality.TABULAR)
            self.assertEqual(first.profile.input_shape, (2,))
            self.assertEqual(first.profile.available_splits, ('test', 'train', 'validation'))
            self.assertEqual(first.profile.observed_train_labels, KnowledgeState.AVAILABLE)
            self.assertEqual(first.profile.clean_train_labels, KnowledgeState.UNKNOWN)
            self.assertEqual(first.profile.noise.status, NoiseStatus.UNKNOWN)
            record = catalog.get('fixture-local')
            self.assertEqual(record.profile_fingerprint, first.profile.fingerprint)
            self.assertEqual(record.profile['fingerprint'], first.profile.fingerprint)

    def test_missing_clean_targets_do_not_imply_noisy_dataset(self) -> None:
        registry = DatasetRegistry()
        registry.add(_data_service__NativeNoisyFixtureAdapter())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = DataService(registry, LocalDatasetCatalog(root / 'catalog.json'))
            service.register('native', 'native_noisy_fixture', {'root': root})
            report = service.inspect('native')
            self.assertEqual(report.profile.clean_train_labels, KnowledgeState.UNKNOWN)
            self.assertEqual(report.profile.noise.status, NoiseStatus.UNKNOWN)
            capabilities = service.set_declarations('native', DatasetDeclarations(clean_train_labels=KnowledgeState.UNAVAILABLE, method_noise_rate_prior=NoiseRateInfo(NoiseRateStatus.KNOWN, 0.2, 'user')))
            self.assertEqual(capabilities.clean_train_labels, KnowledgeState.UNAVAILABLE)
            self.assertIsNone(service.capabilities('native').method_noise_rate_prior.value)

    def test_old_local_catalog_record_without_profile_remains_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'catalog.json'
            path.write_text(json.dumps({'version': 1, 'datasets': {'legacy': {'adapter': 'fixture', 'data': {'name': 'fixture', 'root': str(root)}, 'state': 'registered', 'evidence': None, 'error': None}}}), encoding='utf-8')
            record = LocalDatasetCatalog(path).get('legacy')
            self.assertIsNone(record.profile)
            self.assertIsNone(record.declarations)
            self.assertIsNone(record.profile_fingerprint)

# --- merged from test_dataset_training_fixtures.py ---
import gzip

# --- merged from test_dataset_training_fixtures.py ---
import json

# --- merged from test_dataset_training_fixtures.py ---
from pathlib import Path

# --- merged from test_dataset_training_fixtures.py ---
import struct

# --- merged from test_dataset_training_fixtures.py ---
import tempfile

# --- merged from test_dataset_training_fixtures.py ---
import unittest

# --- merged from test_dataset_training_fixtures.py ---
import numpy as np

# --- merged from test_dataset_training_fixtures.py ---
from PIL import Image

# --- merged from test_dataset_training_fixtures.py ---
from lnl_toolbox.training.service import ExperimentService

# --- merged from test_dataset_training_fixtures.py ---
def _dataset_training_fixtures__write_idx(root: Path, split: str, count: int) -> None:
    """Write the official MNIST/Fashion-MNIST IDX.GZ pair."""
    prefix = 'train' if split == 'train' else 't10k'
    images = np.arange(count * 28 * 28, dtype=np.uint8).reshape(count, 28, 28)
    labels = np.arange(count, dtype=np.uint8) % 10
    with gzip.open(root / f'{prefix}-images-idx3-ubyte.gz', 'wb') as handle:
        handle.write(struct.pack('>IIII', 2051, count, 28, 28))
        handle.write(images.tobytes())
    with gzip.open(root / f'{prefix}-labels-idx1-ubyte.gz', 'wb') as handle:
        handle.write(struct.pack('>II', 2049, count))
        handle.write(labels.tobytes())

# --- merged from test_dataset_training_fixtures.py ---
def _dataset_training_fixtures__write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((40, 40, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode='RGB').save(path)

# --- merged from test_dataset_training_fixtures.py ---
def _dataset_training_fixtures__write_clothing1m(root: Path) -> None:
    train = [f'images/train/{index}.jpg' for index in range(8)]
    validation = [f'images/val/{index}.jpg' for index in range(4)]
    test = [f'images/test/{index}.jpg' for index in range(4)]
    for offset, key in enumerate(train + validation + test):
        _dataset_training_fixtures__write_rgb(root / key, 16 + offset * 8)
    (root / 'noisy_train_key_list.txt').write_text('\n'.join(train) + '\n', encoding='utf-8')
    (root / 'clean_val_key_list.txt').write_text('\n'.join(validation) + '\n', encoding='utf-8')
    (root / 'clean_test_key_list.txt').write_text('\n'.join(test) + '\n', encoding='utf-8')
    (root / 'noisy_label_kv.txt').write_text('\n'.join((f'{key} {index % 2}' for index, key in enumerate(train))) + '\n', encoding='utf-8')
    clean = validation + test
    (root / 'clean_label_kv.txt').write_text('\n'.join((f'{key} {index % 2}' for index, key in enumerate(clean))) + '\n', encoding='utf-8')

# --- merged from test_dataset_training_fixtures.py ---
def _dataset_training_fixtures__animal_record(identifier: int, label: int) -> bytes:
    pixel_value = (16 + identifier * 7) % 256
    pixels = np.full((3, 64, 64), pixel_value, dtype=np.uint8)
    return struct.pack('<II', identifier, label) + pixels.tobytes()

# --- merged from test_dataset_training_fixtures.py ---
def _dataset_training_fixtures__write_animal10n(root: Path) -> None:
    (root / 'data_batch_1.bin').write_bytes(b''.join((_dataset_training_fixtures__animal_record(index, index % 2) for index in range(8))))
    (root / 'test_batch.bin').write_bytes(b''.join((_dataset_training_fixtures__animal_record(100 + index, index % 2) for index in range(4))))

# --- merged from test_dataset_training_fixtures.py ---
def _dataset_training_fixtures__write_uci_heart(path: Path) -> None:
    rows = []
    for index in range(40):
        features = [f'{(index + column) % 11 + column / 10:.1f}' for column in range(13)]
        rows.append(' '.join(features + [str(1 + index % 2)]))
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')

# --- merged from test_dataset_training_fixtures.py ---
def _dataset_training_fixtures__image_config(name: str, root: Path) -> dict:
    return {'seed': 13, 'data': {'name': name, 'root': str(root), 'validation_size': 4, 'augment': False, 'image_size': 32}, 'loader': {'batch_size': 64, 'num_workers': 0, 'pin_memory': False}, 'model': {'name': 'tiny_cnn', 'width': 2}, 'loss': {'name': 'ce'}, 'optimizer': {'name': 'adam', 'lr': 0.001}, 'scheduler': {'name': 'none'}, 'trainer': {'epochs': 1, 'device': 'cpu', 'progress': False}, 'execution': {'runner': 'clean'}}

# --- merged from test_dataset_training_fixtures.py ---
class _dataset_training_fixtures_DatasetTrainingFixturesTest(unittest.TestCase):

    def _assert_trained(self, config: dict, run_dir: Path) -> None:
        result = ExperimentService().run(config, run_dir)
        final = json.loads((result / 'final_metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(final['completed_epochs'], 1)
        self.assertTrue(np.isfinite(float(final['test_accuracy'])))
        self.assertTrue((result / 'data_manifest.json').is_file())
        epoch_rows = []
        for line in (result / 'metrics.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('event') == 'epoch':
                epoch_rows.append(row)
        self.assertEqual(len(epoch_rows), 1)

    def test_official_mnist_idx_gzip_trains_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'MNIST' / 'raw'
            data_root.mkdir(parents=True)
            _dataset_training_fixtures__write_idx(data_root, 'train', 40)
            _dataset_training_fixtures__write_idx(data_root, 'test', 20)
            self._assert_trained(_dataset_training_fixtures__image_config('mnist', root), root / 'run')

    def test_official_fashion_mnist_idx_gzip_trains_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'FashionMNIST' / 'raw'
            data_root.mkdir(parents=True)
            _dataset_training_fixtures__write_idx(data_root, 'train', 40)
            _dataset_training_fixtures__write_idx(data_root, 'test', 20)
            self._assert_trained(_dataset_training_fixtures__image_config('fashion_mnist', root), root / 'run')

    def test_official_clothing1m_lists_and_label_maps_train_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset_training_fixtures__write_clothing1m(root)
            self._assert_trained(_dataset_training_fixtures__image_config('clothing1m', root), root / 'run')

    def test_official_animal10n_binary_records_train_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset_training_fixtures__write_animal10n(root)
            self._assert_trained(_dataset_training_fixtures__image_config('animal10n', root), root / 'run')

    def test_official_uci_heart_whitespace_rows_train_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'heart.dat'
            run_dir = root / 'run'
            _dataset_training_fixtures__write_uci_heart(source)
            config = {'seed': 13, 'data': {'name': 'uci_binary', 'path': str(source), 'preprocessing': {'format': 'whitespace', 'target_column': -1, 'standardize': False, 'label_values': ['1', '2']}, 'split': {'validation_fraction': 0.2, 'test_fraction': 0.2, 'seed': 13}}, 'loader': {'batch_size': 64, 'num_workers': 0}, 'model': {'name': 'linear'}, 'learning_rate': 0.01, 'epochs': 1, 'execution': {'runner': 'binary'}}
            result = ExperimentService().run(config, run_dir)
            rows = json.loads((result / 'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 1)
            self.assertTrue(np.isfinite(float(rows[0]['test_accuracy'])))
            self.assertTrue((result / 'data_manifest.json').is_file())

# --- merged from test_local_data_catalog.py ---
from pathlib import Path

# --- merged from test_local_data_catalog.py ---
import tempfile

# --- merged from test_local_data_catalog.py ---
import unittest

# --- merged from test_local_data_catalog.py ---
from lnl_toolbox.data.local_catalog import LocalDatasetCatalog

# --- merged from test_local_data_catalog.py ---
class _local_data_catalog_LocalDatasetCatalogTest(unittest.TestCase):

    def test_registration_merge_states_and_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'dataset'
            source.mkdir()
            (source / 'train').write_text('fixture', encoding='utf-8')
            catalog = LocalDatasetCatalog(root / 'catalog.json')
            record = catalog.register('My-CIFAR', 'cifar10', {'root': source})
            self.assertEqual(record.effective_state, 'registered')
            config = catalog.apply({'data': {'name': 'cifar100', 'root': 'stale', 'validation_size': 5, 'augment': False}}, 'my-cifar')
            self.assertEqual(config['data']['name'], 'cifar10')
            self.assertEqual(config['data']['root'], str(source.resolve()))
            self.assertEqual(config['data']['validation_size'], 5)
            self.assertEqual(config['local_dataset']['alias'], 'my-cifar')
            self.assertEqual(catalog.mark_layout_validated('my-cifar').effective_state, 'layout_validated')
            verified = catalog.mark_training_verified('my-cifar', {'run_dir': 'run', 'data_fingerprint': 'abc'})
            self.assertEqual(verified.effective_state, 'training_verified')
            (source / 'changed').write_text('changed', encoding='utf-8')
            self.assertEqual(catalog.get('my-cifar').effective_state, 'verification_stale')
            catalog.remove('my-cifar')
            self.assertEqual(catalog.records(), ())

    def test_reregister_and_failure_clear_old_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = (root / 'first', root / 'second')
            first.mkdir()
            second.mkdir()
            catalog = LocalDatasetCatalog(root / 'catalog.json')
            catalog.register('data', 'cifar10', {'root': first})
            catalog.mark_training_verified('data', {'run_dir': 'run'})
            record = catalog.register('data', 'cifar100', {'root': second})
            self.assertEqual(record.adapter, 'cifar100')
            self.assertEqual(record.effective_state, 'registered')
            failed = catalog.mark_failed('data', 'bad labels')
            self.assertEqual(failed.effective_state, 'failed')
            self.assertEqual(failed.error, 'bad labels')

# --- merged from test_reliability_selection_adapter.py ---
import unittest

# --- merged from test_reliability_selection_adapter.py ---
import torch

# --- merged from test_reliability_selection_adapter.py ---
from lnl_toolbox.estimators import DivideMixGMMCleanProbabilityEstimator, DivideMixGMMLossInput, ReliabilityResult, ReliabilityToSelectionInputAdapter

# --- merged from test_reliability_selection_adapter.py ---
from lnl_toolbox.selectors import SmallLossSelector

# --- merged from test_reliability_selection_adapter.py ---
from lnl_toolbox.treatments import SelectorContributionAdapter

# --- merged from test_reliability_selection_adapter.py ---
class _reliability_selection_adapter_ReliabilitySelectionAdapterTest(unittest.TestCase):

    def setUp(self) -> None:
        self.adapter = ReliabilityToSelectionInputAdapter()

    def test_dataset_result_is_aligned_to_requested_batch_order(self):
        result = ReliabilityResult(sample_indices=torch.tensor([90, 7, 41, 300]), scores=torch.tensor([0.2, 0.9, 0.5, 0.8]))
        expected = torch.tensor([300, 7, 90])
        selection_input = self.adapter.adapt(result, expected_sample_indices=expected, metadata={'epoch': 2})
        self.assertTrue(torch.equal(selection_input.sample_indices, expected))
        self.assertTrue(torch.equal(selection_input.scores, torch.tensor([-0.8, -0.9, -0.2])))
        self.assertEqual(selection_input.metadata, {'epoch': 2})

    def test_result_and_expected_permutations_preserve_index_score_mapping(self):
        indices = torch.tensor([101, -5, 800, 42])
        scores = torch.tensor([0.1, 0.7, 0.4, 0.9])
        first = self.adapter.adapt(ReliabilityResult(indices, scores), expected_sample_indices=torch.tensor([42, 101, 800]))
        permutation = torch.tensor([2, 0, 3, 1])
        second = self.adapter.adapt(ReliabilityResult(indices[permutation], scores[permutation]), expected_sample_indices=torch.tensor([800, 42, 101]))
        first_by_index = {int(index): float(score) for index, score in zip(first.sample_indices.tolist(), first.scores.tolist())}
        second_by_index = {int(index): float(score) for index, score in zip(second.sample_indices.tolist(), second.scores.tolist())}
        self.assertEqual(first_by_index, second_by_index)

    def test_missing_and_duplicate_expected_indices_are_rejected(self):
        result = ReliabilityResult(sample_indices=torch.tensor([2, 8, 50]), scores=torch.tensor([0.2, 0.8, 0.5]))
        with self.assertRaisesRegex(ValueError, 'absent'):
            self.adapter.adapt(result, expected_sample_indices=torch.tensor([8, 99]))
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.adapter.adapt(result, expected_sample_indices=torch.tensor([8, 8]))

    def test_invalid_expected_index_contract_is_rejected(self):
        result = ReliabilityResult(sample_indices=torch.tensor([2, 8]), scores=torch.tensor([0.2, 0.8]))
        cases = ((torch.tensor([[2, 8]]), 'one-dimensional'), (torch.tensor([], dtype=torch.long), 'must not be empty'), (torch.tensor([2.0, 8.0]), 'integer dtype'))
        for expected, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.adapter.adapt(result, expected_sample_indices=expected)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_expected_indices_must_share_result_device(self):
        result = ReliabilityResult(sample_indices=torch.tensor([2, 8], device='cuda'), scores=torch.tensor([0.2, 0.8], device='cuda'))
        with self.assertRaisesRegex(ValueError, 'result device'):
            self.adapter.adapt(result, expected_sample_indices=torch.tensor([8]))

    def test_invalid_reliability_result_is_rejected(self):
        cases = ((ReliabilityResult(sample_indices=torch.tensor([1, 2]), scores=torch.tensor([0.1, float('nan')])), 'finite'), (ReliabilityResult(sample_indices=torch.tensor([1, 2]), scores=torch.tensor([0.1, float('inf')])), 'finite'), (ReliabilityResult(sample_indices=torch.tensor([1, 2]), scores=torch.tensor([0.1, 0.2], requires_grad=True)), 'detached'), (ReliabilityResult(sample_indices=torch.tensor([1, 1]), scores=torch.tensor([0.1, 0.2])), 'unique'))
        for result, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.adapter.adapt(result, expected_sample_indices=torch.tensor([1]))

    def test_high_reliability_is_preferred_by_small_loss_selector(self):
        result = ReliabilityResult(sample_indices=torch.tensor([70, 10, 90, 30]), scores=torch.tensor([0.1, 0.95, 0.4, 0.8]))
        expected = torch.tensor([90, 70, 30, 10])
        selection_input = self.adapter.adapt(result, expected_sample_indices=expected)
        selection = SmallLossSelector(0.5).select(selection_input)
        selected_indices = expected[selection.selected_mask]
        self.assertEqual(set(selected_indices.tolist()), {10, 30})

    def test_output_can_continue_through_existing_contribution_adapter(self):
        result = ReliabilityResult(sample_indices=torch.tensor([20, 4, 80, 9]), scores=torch.tensor([0.3, 0.9, 0.1, 0.7]))
        selection_input = self.adapter.adapt(result, expected_sample_indices=torch.tensor([9, 80, 4, 20]))
        contribution = SelectorContributionAdapter(SmallLossSelector(0.5)).resolve(selection_input)
        self.assertEqual(contribution.selected_mask.tolist(), [True, False, True, False])
        self.assertTrue(torch.equal(contribution.sample_weights, torch.ones(4)))

    def test_adapter_does_not_modify_result_or_add_treatment_behavior(self):
        indices = torch.tensor([6, 2, 99])
        scores = torch.tensor([0.6, 0.2, 0.9])
        result = ReliabilityResult(sample_indices=indices, scores=scores, metrics={'score_mean': float(scores.mean().item())})
        original_indices = indices.clone()
        original_scores = scores.clone()
        original_metrics = dict(result.metrics)
        selection_input = self.adapter.adapt(result, expected_sample_indices=torch.tensor([99, 6]))
        self.assertTrue(torch.equal(result.sample_indices, original_indices))
        self.assertTrue(torch.equal(result.scores, original_scores))
        self.assertEqual(result.metrics, original_metrics)
        for absent_attribute in ('selected_mask', 'sample_weights', 'threshold', 'labels', 'split'):
            self.assertFalse(hasattr(selection_input, absent_attribute))

    def test_dividemix_component_output_becomes_batch_ranking_input_only(self):
        estimator_input = DivideMixGMMLossInput(per_sample_losses=torch.tensor([0.1, 0.13, 0.16, 1.8, 2.0, 2.2]), sample_indices=torch.tensor([40, 10, 70, 20, 90, 30]))
        reliability = DivideMixGMMCleanProbabilityEstimator(random_seed=17).estimate(estimator_input)
        expected = torch.tensor([90, 10, 30])
        selection_input = self.adapter.adapt(reliability, expected_sample_indices=expected)
        self.assertTrue(torch.equal(selection_input.sample_indices, expected))
        expected_scores = -torch.stack([reliability.scores[reliability.sample_indices == index].squeeze(0) for index in expected])
        self.assertTrue(torch.equal(selection_input.scores, expected_scores))

# --- merged from test_trusted_supervision.py ---
import tempfile

# --- merged from test_trusted_supervision.py ---
from pathlib import Path

# --- merged from test_trusted_supervision.py ---
import unittest

# --- merged from test_trusted_supervision.py ---
import numpy as np

# --- merged from test_trusted_supervision.py ---
import torch

# --- merged from test_trusted_supervision.py ---
from torch.utils.data import Dataset

# --- merged from test_trusted_supervision.py ---
from lnl_toolbox.data.trusted import TrustedSupervisionManifest, TrustedValidationProvider

# --- merged from test_trusted_supervision.py ---
class _trusted_supervision__Dataset(Dataset):

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return {'input': torch.tensor([float(item)]), 'target': 99, 'index': (8, 3)[item]}

# --- merged from test_trusted_supervision.py ---
class _trusted_supervision_TrustedSupervisionTest(unittest.TestCase):

    def test_manifest_roundtrip_and_provider_replaces_only_explicit_targets(self) -> None:
        manifest = TrustedSupervisionManifest(np.array([3, 8]), np.array([1, 0]), 'fixture', 'trusted_validation', 'audited_manifest', True, {'reviewer': 'unit-test'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trusted.npz'
            manifest.save(path)
            loaded = TrustedSupervisionManifest.load(path)
        provider = TrustedValidationProvider(_trusted_supervision__Dataset(), loaded)
        batch = next(iter(provider.loader(batch_size=2, shuffle=False, seed=1)))
        self.assertEqual(batch['target'].tolist(), [0, 1])
        self.assertEqual(provider.fingerprint, manifest.fingerprint)

    def test_ordinary_validation_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'explicitly audited'):
            TrustedSupervisionManifest(np.array([0]), np.array([0]), 'fixture', 'trusted_validation', 'validation', True)

    def test_test_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'trusted_validation'):
            TrustedSupervisionManifest(np.array([0]), np.array([0]), 'fixture', 'test', 'audited_manifest', True)
