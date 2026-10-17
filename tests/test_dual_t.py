from __future__ import annotations
import inspect
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from lnl_toolbox.algorithms.dual_t.evidence import build_transition_evidence, realized_empirical_transition, transition_matrix_error
from lnl_toolbox.noise.estimators import DualTransitionEstimator, PosteriorSnapshot
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.runtime import seed_everything
from lnl_toolbox.training.checkpoint import capture_rng_state
from lnl_toolbox.training.dual_t_evidence_experiment import _run_final_arm, _tensor_state_hash, run_dual_t_evidence_experiment
from lnl_toolbox.training.experiment import build_model

class _dual_t_evidence__FixedImageDataset(torch.utils.data.Dataset):

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(17)
        self.inputs = torch.rand(12, 3, 32, 32, generator=generator)
        self.targets = torch.arange(12) % 3

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, item: int):
        return {'input': self.inputs[item], 'target': self.targets[item], 'index': torch.tensor(100 + item * 5)}

class _dual_t_evidence__PreparedFixture:

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def loader(self, role, *, generator_seed, shuffle=True):
        del role
        generator = torch.Generator().manual_seed(int(generator_seed))
        return torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=shuffle, num_workers=0, generator=generator)

class _dual_t_evidence_DualTEvidenceContractTest(unittest.TestCase):

    @staticmethod
    def _snapshot() -> PosteriorSnapshot:
        return PosteriorSnapshot(noisy_probabilities=np.asarray([[0.9, 0.05, 0.05], [0.6, 0.2, 0.2], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0.05, 0.05, 0.9], [0.2, 0.2, 0.6]]), noisy_targets=np.asarray([0, 1, 1, 2, 2, 0]), global_indices=np.asarray([30, 20, 10, 40, 50, 60]), dataset='fixture', split='train')

    @staticmethod
    def _manifest(transition: np.ndarray) -> NoiseManifest:
        return NoiseManifest(dataset='fixture', noise_type='synthetic_fixture', seed=9, requested_rate=0.25, clean_targets=np.asarray([0, 0, 1, 1, 2, 2]), noisy_targets=np.asarray([0, 1, 1, 2, 2, 0]), transition_matrix=transition, num_classes=3, global_indices=np.asarray([10, 20, 30, 40, 50, 60]))

    def test_matrix_error_matches_elementwise_hand_calculation(self) -> None:
        truth = np.asarray([[0.8, 0.2], [0.1, 0.9]])
        estimate = np.asarray([[0.7, 0.3], [0.2, 0.8]])
        error = transition_matrix_error(estimate, truth)
        self.assertAlmostEqual(error.l1_total, 0.4)
        self.assertAlmostEqual(error.l1_mean, 0.1)
        self.assertAlmostEqual(error.max_absolute_error, 0.1)
        np.testing.assert_allclose(error.row_l1, (0.2, 0.2))

    def test_ground_truth_is_the_manifest_matrix_and_snapshot_is_shared(self) -> None:
        snapshot = self._snapshot()
        dual = DualTransitionEstimator().estimate(snapshot)
        manifest = self._manifest(dual.matrix)
        evidence = build_transition_evidence(snapshot=snapshot, manifest=manifest)
        np.testing.assert_array_equal(evidence.ground_truth_matrix, manifest.transition_matrix)
        self.assertEqual(evidence.anchor_artifact.source_snapshot_hash, snapshot.snapshot_hash)
        self.assertEqual(evidence.dual_t_artifact.source_snapshot_hash, snapshot.snapshot_hash)
        self.assertEqual(evidence.dual_t_error.l1_total, 0.0)
        self.assertGreater(evidence.anchor_error.l1_total, 0.0)

    def test_manifest_without_ground_truth_transition_is_rejected(self) -> None:
        manifest = self._manifest(np.eye(3))
        manifest.transition_matrix = None
        with self.assertRaisesRegex(ValueError, 'NoiseManifest.transition_matrix'):
            build_transition_evidence(snapshot=self._snapshot(), manifest=manifest)

    def test_realized_matrix_is_an_offline_index_aligned_diagnostic(self) -> None:
        manifest = self._manifest(np.eye(3))
        realized = realized_empirical_transition(manifest, np.asarray([60, 10, 50, 20, 40, 30]))
        expected = np.asarray([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
        np.testing.assert_allclose(realized, expected)


    def test_final_arm_contract_has_no_ground_truth_or_clean_targets(self) -> None:
        parameters = inspect.signature(_run_final_arm).parameters
        self.assertNotIn('manifest', parameters)
        self.assertNotIn('ground_truth', parameters)
        self.assertNotIn('clean_targets', parameters)

class _dual_t_evidence_DualTEvidenceFairnessTest(unittest.TestCase):

    def test_sequential_arms_share_initial_state_order_and_input_tensors(self) -> None:
        seed_everything(23)
        model_config = {'name': 'tiny_cnn', 'width': 4}
        reference = build_model(model_config, 3)
        initial_state = {name: value.detach().cpu().clone() for name, value in reference.state_dict().items()}
        initial_hash = _tensor_state_hash(initial_state)
        rng_state = capture_rng_state()
        dataset = _dual_t_evidence__FixedImageDataset()
        arguments = {'initial_state': initial_state, 'model_config': model_config, 'optimizer_config': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.0}, 'scheduler_config': {'name': 'none'}, 'epochs': 2, 'num_classes': 3, 'prepared_data': _dual_t_evidence__PreparedFixture(dataset), 'sampler_seed': 101, 'rng_state': rng_state, 'device': torch.device('cpu'), 'transition': None}
        with tempfile.TemporaryDirectory() as directory:
            first = _run_final_arm(name='first', run_dir=Path(directory) / 'first', **arguments)
            second = _run_final_arm(name='second', run_dir=Path(directory) / 'second', **arguments)
            first_rows = [json.loads(line) for line in (Path(directory) / 'first' / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            second_rows = [json.loads(line) for line in (Path(directory) / 'second' / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(first.initial_state_hash, initial_hash)
        self.assertEqual(second.initial_state_hash, initial_hash)
        self.assertEqual(first.sampler_seed, second.sampler_seed)
        self.assertEqual(first.batch_index_hashes, second.batch_index_hashes)
        self.assertEqual(first.input_tensor_hashes, second.input_tensor_hashes)
        self.assertEqual(len(first.batch_index_hashes), 2)
        self.assertEqual([row['event'] for row in first_rows], ['epoch', 'epoch'])
        self.assertEqual([row['epoch'] for row in first_rows], [1, 2])
        self.assertEqual(
            [{key: value for key, value in row.items() if key != 'arm'} for row in first_rows],
            [{key: value for key, value in row.items() if key != 'arm'} for row in second_rows],
        )


import hashlib
import json
from unittest.mock import patch
from torch import nn
from torch.utils.data import DataLoader
from lnl_toolbox.algorithms.dual_t import DualTConfig, DualTAlgorithm, DualTPhase, DualTState
from lnl_toolbox.algorithms.dual_t.algorithm import _train_supervised_epoch
from lnl_toolbox.core import RunState
from lnl_toolbox.noise import DualTransitionEstimator, PosteriorSnapshot
from lnl_toolbox.noise.transition import TransitionArtifact
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.experiment import run_experiment

class _dual_t_forward__IndexedClassifier(nn.Module):

    def __init__(self, classes: int=3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(classes) * 5.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight.t()

def _dual_t_forward__records(*, clean_offset: int=0):
    noisy_targets = (0, 1, 1, 2, 2, 0)
    records = []
    for index, noisy_target in enumerate(noisy_targets):
        latent_class = index // 2
        records.append({'input': torch.nn.functional.one_hot(torch.tensor(latent_class), num_classes=3).to(dtype=torch.float32), 'target': torch.tensor(noisy_target), 'index': torch.tensor(100 + index * 7), 'clean_target': torch.tensor((latent_class + clean_offset) % 3)})
    return records

def _dual_t_forward__validation_records():
    return [{'input': torch.nn.functional.one_hot(torch.tensor(index), num_classes=3).to(dtype=torch.float32), 'target': torch.tensor(index), 'index': torch.tensor(500 + index)} for index in range(3)]

def _dual_t_forward__test_records():
    return [{'input': torch.nn.functional.one_hot(torch.tensor(index), num_classes=3).to(dtype=torch.float32), 'target': torch.tensor(index), 'index': torch.tensor(800 + index)} for index in range(3)]

def _dual_t_forward__config(*, posterior_epochs: int=2, final_epochs: int=2):
    stage = {'model': {'name': 'fixture'}, 'optimizer': {'name': 'sgd', 'lr': 0.0}, 'scheduler': {'name': 'none'}, 'loss': {'name': 'ce'}}
    return {'method': 'dual_t', 'seed': 3, 'data': {'name': 'fixture'}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 5, 'validation_targets': 'noisy', 'mapping_hash': 'a' * 64}, 'loader': {'batch_size': 3}, 'evaluation': {'selection_split': 'validation'}, 'posterior_stage': {**stage, 'epochs': posterior_epochs, 'checkpoint_selection': 'noisy_validation_accuracy'}, 'transition_stage': {}, 'final_stage': {**stage, 'epochs': final_epochs, 'fresh_model': True}}

def _dual_t_forward__algorithm(directory: str | Path, *, posterior_epochs: int=2, final_epochs: int=2, clean_offset: int=0) -> DualTAlgorithm:
    posterior_model = _dual_t_forward__IndexedClassifier()
    final_model = _dual_t_forward__IndexedClassifier()
    posterior_optimizer = torch.optim.SGD(posterior_model.parameters(), lr=0.0)
    final_optimizer = torch.optim.SGD(final_model.parameters(), lr=0.0)
    loss = nn.CrossEntropyLoss(reduction='none')
    return DualTAlgorithm(posterior_model=posterior_model, posterior_optimizer=posterior_optimizer, posterior_scheduler=None, final_model=final_model, final_optimizer=final_optimizer, final_scheduler=None, posterior_loss=loss, final_loss=nn.CrossEntropyLoss(reduction='none'), train_loader=DataLoader(_dual_t_forward__records(clean_offset=clean_offset), batch_size=3, shuffle=True), noisy_validation_loader=DataLoader(_dual_t_forward__validation_records(), batch_size=3), clean_test_loader=DataLoader(_dual_t_forward__test_records(), batch_size=3), device=torch.device('cpu'), run_dir=directory, config=_dual_t_forward__config(posterior_epochs=posterior_epochs, final_epochs=final_epochs), dataset='fixture', noise_metadata={'manifest_sha256': 'b' * 64, 'mapping_hash': 'a' * 64})

class DualTPaperProtocolTest(unittest.TestCase):

    def test_formal_defaults_and_metadata(self):
        from lnl_toolbox.catalog import load_yaml
        from lnl_toolbox.training.experiment import build_optimizer
        root = Path(__file__).resolve().parents[1]
        config = load_yaml(root / 'configs/reproduction/cifar10_dual_t_sym20.yaml')
        metadata = load_yaml(root / 'web/lnl_parameter_metadata_registry_revised.yaml')['methods']['dual_t']
        for stage in ('posterior_stage', 'final_stage'):
            build_optimizer(nn.Linear(3, 3), config[stage]['optimizer'])
            self.assertEqual(config[stage]['model']['variant'], 'imagenet_standard')
            self.assertEqual(metadata['parameters']['paper'][stage + '.epochs']['current'], config[stage]['epochs'])
        for level, fields in metadata['parameters'].items():
            self.assertEqual(metadata['coverage']['by_level'][level], len(fields))
        self.assertEqual(metadata['coverage']['parameter_count'], sum(len(v) for v in metadata['parameters'].values()))

    def test_standard_resnet_stem_and_unchanged_cifar_default(self):
        for name, blocks in (('resnet18', [2, 2, 2, 2]), ('resnet34', [3, 4, 6, 3])):
            model = build_model({'name': name, 'variant': 'imagenet_standard'}, 10).eval()
            self.assertEqual(model.conv1.kernel_size, (7, 7))
            self.assertEqual(model.conv1.stride, (2, 2))
            self.assertEqual(model.maxpool.kernel_size, 3)
            self.assertEqual([len(getattr(model, f'layer{i}')) for i in range(1, 5)], blocks)
            self.assertEqual(model(torch.zeros(2, 3, 32, 32)).shape, (2, 10))
            cifar = build_model({'name': name}, 10)
            self.assertEqual(cifar.stem[0].kernel_size, (3, 3))
            self.assertEqual(cifar.stem[0].stride, (1, 1))
        with self.assertRaisesRegex(ValueError, 'base_width'):
            build_model({'name': 'resnet18', 'variant': 'imagenet_standard', 'base_width': 16}, 10)

    def test_forward_epsilon_loss_and_gradient_preserve_default_behavior(self):
        from lnl_toolbox.algorithms.dual_t.algorithm import _build_final_risk_corrector
        from lnl_toolbox.algorithms.transition_risk import ForwardRiskCorrector
        matrix = TransitionArtifact(np.eye(2), estimator='fixture')
        logits = torch.tensor([[0., -90.]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1])
        probability = logits.softmax(1)[:, 1]
        loss = nn.CrossEntropyLoss(reduction='none')
        actual = _build_final_risk_corrector('forward').per_sample_risk(
            logits=logits, noisy_targets=targets, base_loss=loss, transition=matrix)
        expected = -(probability + 1e-12).log()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
            torch.autograd.grad(expected.sum(), logits, retain_graph=True)[0])
        default = ForwardRiskCorrector().per_sample_risk(
            logits=logits, noisy_targets=targets, base_loss=loss, transition=matrix)
        torch.testing.assert_close(default, -probability.log())
        self.assertGreater(default.item(), actual.item())
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ForwardRiskCorrector(log_epsilon=value)

    def test_snapshot_uses_deterministic_loader_with_aligned_labels(self):
        from lnl_toolbox.training.snapshots import collect_posterior_snapshot
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _dual_t_forward__algorithm(directory, posterior_epochs=1, final_epochs=1)
            algorithm.train_posterior()
            records = _dual_t_forward__records()
            deterministic = DataLoader(records, batch_size=3, shuffle=False)
            algorithm.posterior_loader = deterministic
            algorithm.train_loader = object()  # Iteration would fail if the old path is used.
            with patch('lnl_toolbox.algorithms.dual_t.algorithm.collect_posterior_snapshot',
                    wraps=collect_posterior_snapshot) as collect:
                algorithm.estimate_transition()
            self.assertIs(collect.call_args.args[1], deterministic)
            self.assertEqual(algorithm.snapshot.global_indices.tolist(), [int(r['index']) for r in records])
            self.assertEqual(algorithm.snapshot.noisy_targets.tolist(), [int(r['target']) for r in records])

    def test_production_runner_uses_train_eval_and_completes_short_training(self):
        from types import SimpleNamespace
        from lnl_toolbox.data import DataRole
        from lnl_toolbox.training.runners import resolve_data_requirements
        from lnl_toolbox.training.dual_t_experiment import run_dual_t_experiment
        config = _dual_t_forward__config(posterior_epochs=1, final_epochs=1)
        requirements = resolve_data_requirements(config, expected_runner='dual_t')
        self.assertIn(DataRole.TRAIN_EVAL, requirements.roles)
        loaders = {
            DataRole.TRAIN: DataLoader(_dual_t_forward__records(), batch_size=3),
            DataRole.TRAIN_EVAL: DataLoader(_dual_t_forward__records(), batch_size=3),
            DataRole.NOISY_VALIDATION: DataLoader(_dual_t_forward__validation_records(), batch_size=3),
            DataRole.TEST: DataLoader(_dual_t_forward__test_records(), batch_size=3),
        }
        calls = []
        def loader(role, **kwargs):
            calls.append((role, kwargs))
            return loaders[role]
        prepared = SimpleNamespace(dataset='fixture', num_classes=3,
            manifest=object(), manifest_path=Path('unused.npz'), train_indices=[],
            validation_indices=[], input_spec=SimpleNamespace(channels=None), loader=loader)
        from lnl_toolbox.training.snapshots import collect_posterior_snapshot
        prefix = 'lnl_toolbox.training.dual_t_experiment.'
        with tempfile.TemporaryDirectory() as directory, \
                patch(prefix + 'prepare_experiment_data', return_value=prepared), \
                patch(prefix + 'effective_subset_actual_rate', return_value=0.2), \
                patch(prefix + 'checkpoint_noise_metadata', return_value={
                    'mode': 'generated', 'manifest_path': 'noise_manifest.npz',
                    'manifest_sha256': 'b' * 64, 'mapping_hash': 'a' * 64,
                    'manifest_actual_rate': 0.2, 'effective_train_subset_actual_rate': 0.2,
                    'validation_targets': 'noisy', 'effective_validation_subset_actual_rate': 0.2}), \
                patch(prefix + 'build_model', side_effect=lambda *_: _dual_t_forward__IndexedClassifier()), \
                patch('lnl_toolbox.algorithms.dual_t.algorithm.collect_posterior_snapshot',
                    wraps=collect_posterior_snapshot) as collect:
            run_dual_t_experiment(config, output_dir=directory)
            self.assertIn((DataRole.TRAIN_EVAL, {'shuffle': False}), calls)
            self.assertIs(collect.call_args.args[1], loaders[DataRole.TRAIN_EVAL])
            self.assertTrue(Path(directory, 'final_metrics.json').is_file())
            self.assertTrue(Path(directory, 'best.pt').is_file())

    def test_checkpoint_selection_uses_corrected_predictions(self):
        from lnl_toolbox.algorithms.dual_t.algorithm import _evaluate_forward_classification
        from lnl_toolbox.evaluation.classification import evaluate_classification
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _dual_t_forward__algorithm(directory, posterior_epochs=1, final_epochs=1)
            algorithm.train_posterior()
            algorithm.estimate_transition()
            algorithm.start_final_training()
            # Rotate classes: raw predictions are all correct; corrected ones are all wrong.
            algorithm.transition = TransitionArtifact(np.roll(np.eye(3), 1, axis=1), estimator='fixture')
            model = algorithm.final_algorithm.model
            raw = evaluate_classification(model, algorithm.noisy_validation_loader,
                algorithm.final_algorithm.loss, algorithm.device)
            corrected = _evaluate_forward_classification(model, algorithm.noisy_validation_loader,
                algorithm.final_algorithm.loss, algorithm.device, algorithm.transition)
            self.assertEqual(raw['accuracy'], 1)
            self.assertEqual(corrected['accuracy'], 0)
            algorithm.train_final()
            self.assertEqual(algorithm.state.best_final_validation_accuracy, corrected['accuracy'])
            self.assertEqual(read_checkpoint(algorithm.best_path, 'cpu')['dual_t_state']['best_final_validation_accuracy'], 0)


class _dual_t_forward_DualTMathTest(unittest.TestCase):

    @staticmethod
    def _snapshot() -> PosteriorSnapshot:
        return PosteriorSnapshot(noisy_probabilities=np.asarray([[0.9, 0.05, 0.05], [0.6, 0.2, 0.2], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0.05, 0.05, 0.9], [0.2, 0.2, 0.6]]), noisy_targets=np.asarray([0, 1, 1, 2, 2, 0]), global_indices=np.asarray([30, 20, 10, 40, 50, 60]), dataset='fixture', split='train')

    def test_t_club_times_t_spade_and_not_reverse(self) -> None:
        artifact = DualTransitionEstimator().estimate(self._snapshot())
        t_club = np.asarray(artifact.metadata['t_club'])
        t_spade = np.asarray(artifact.metadata['t_spade'])
        np.testing.assert_allclose(artifact.matrix, t_club @ t_spade)
        self.assertFalse(np.allclose(t_club @ t_spade, t_spade @ t_club))

    def test_stable_index_permutation_does_not_change_estimate(self) -> None:
        original = self._snapshot()
        order = np.asarray([4, 0, 5, 2, 1, 3])
        permuted = PosteriorSnapshot(original.noisy_probabilities[order], original.noisy_targets[order], original.global_indices[order], original.dataset, original.split)
        first = DualTransitionEstimator().estimate(original)
        second = DualTransitionEstimator().estimate(permuted)
        np.testing.assert_allclose(first.matrix, second.matrix)
        self.assertEqual(first.metadata['anchor_global_indices'], second.metadata['anchor_global_indices'])

    def test_empty_intermediate_class_fails(self) -> None:
        snapshot = PosteriorSnapshot(np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]), np.asarray([0, 1]), np.asarray([3, 9]), 'fixture', 'train')
        with self.assertRaisesRegex(ValueError, 'empty intermediate classes'):
            DualTransitionEstimator().estimate(snapshot)

class _dual_t_forward_DualTStateTest(unittest.TestCase):

    def test_only_sequential_phase_transitions_are_allowed(self) -> None:
        state = DualTState()
        with self.assertRaisesRegex(ValueError, 'illegal Dual-T phase'):
            state.advance(DualTPhase.TRANSITION_READY)
        state.best_posterior_epoch = 0
        state.posterior_completed_epochs = 1
        state.best_posterior_checkpoint_sha256 = 'a' * 64
        state.advance(DualTPhase.POSTERIOR_READY)
        state.posterior_snapshot_hash = 'b' * 64
        state.transition_artifact_hash = 'c' * 64
        state.advance(DualTPhase.TRANSITION_READY)
        state.advance(DualTPhase.FINAL_TRAINING)
        state.best_final_epoch = 0
        state.final_completed_epochs = 1
        state.advance(DualTPhase.COMPLETED)
        restored = DualTState.from_state_dict(state.state_dict())
        self.assertIs(restored.phase, DualTPhase.COMPLETED)

    def test_config_rejects_test_selection_and_clean_validation(self) -> None:
        test_selection = _dual_t_forward__config()
        test_selection['evaluation'] = {'selection_split': 'test'}
        with self.assertRaisesRegex(ValueError, 'validation, not test'):
            DualTConfig.from_mapping(test_selection)
        clean_validation = _dual_t_forward__config()
        clean_validation['noise']['validation_targets'] = 'clean'
        with self.assertRaisesRegex(ValueError, 'validation_targets: noisy'):
            DualTConfig.from_mapping(clean_validation)

    def test_config_defaults_to_forward_and_accepts_explicit_forward(self) -> None:
        implicit = DualTConfig.from_mapping(_dual_t_forward__config())
        self.assertEqual(implicit.classifier_backend, 'forward')
        explicit_values = _dual_t_forward__config()
        explicit_values['final_stage']['classifier'] = 'forward'
        explicit = DualTConfig.from_mapping(explicit_values)
        self.assertEqual(explicit.classifier_backend, 'forward')

    def test_config_rejects_other_classifier_backends(self) -> None:
        values = _dual_t_forward__config()
        values['final_stage']['classifier'] = 'revision'
        with self.assertRaisesRegex(NotImplementedError, "only supports classifier backend 'forward'"):
            DualTConfig.from_mapping(values)

    def test_old_method_name_has_an_explicit_rename_error(self) -> None:
        values = _dual_t_forward__config()
        values['method'] = 'dual_t_forward'
        with self.assertRaisesRegex(ValueError, "dual_t_forward.*renamed to 'dual_t'"):
            DualTConfig.from_mapping(values)
        with self.assertRaisesRegex(ValueError, "dual_t_forward.*renamed to 'dual_t'"):
            run_experiment(values)

    def test_transition_stage_does_not_expose_internal_composition(self) -> None:
        values = _dual_t_forward__config()
        values['transition_stage'] = {'estimator': 'dual_t', 'correction': 'forward'}
        with self.assertRaisesRegex(ValueError, 'internal method fields'):
            DualTConfig.from_mapping(values)


class _dual_t_forward_DualTAlgorithmLifecycleTest(unittest.TestCase):

    def test_extracted_epoch_helper_preserves_production_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            torch.manual_seed(31)
            first = _dual_t_forward__algorithm(first_dir)
            torch.manual_seed(31)
            second = _dual_t_forward__algorithm(second_dir)
            first_state = RunState(phase='posterior_train')
            second_state = RunState(phase='posterior_train')
            production = first._train_epoch(first.posterior_algorithm, first_state, 0)
            extracted = _train_supervised_epoch(second.posterior_algorithm, second.train_loader, second_state, 0)
            self.assertEqual(production, extracted)
            self.assertEqual(first_state.step, second_state.step)

    def test_tiny_cpu_end_to_end_uses_fresh_models_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _dual_t_forward__algorithm(directory, posterior_epochs=1, final_epochs=1)
            self.assertIsNot(algorithm.posterior_algorithm.model, algorithm.final_algorithm.model)
            self.assertIsNot(algorithm.posterior_algorithm.optimizer, algorithm.final_algorithm.optimizer)
            final = algorithm.run()
            self.assertIs(algorithm.state.phase, DualTPhase.COMPLETED)
            self.assertEqual(final['method'], 'dual_t')
            self.assertEqual(final['completed_posterior_epochs'], 1)
            self.assertEqual(final['completed_final_epochs'], 1)
            for filename in ('posterior_best.pt', 'posterior_snapshot.npz', 'transition_artifact.npz', 'last.pt', 'best.pt', 'metrics.jsonl', 'final_metrics.json'):
                self.assertTrue(Path(directory, filename).is_file(), filename)
            self.assertEqual(algorithm.transition.metadata['composition'], 't_club @ t_spade')
            self.assertEqual(algorithm.transition.metadata['posterior_best_checkpoint_sha256'], algorithm.state.best_posterior_checkpoint_sha256)
            self.assertEqual(algorithm.transition.metadata['method'], 'dual_t')
            metric_rows = [json.loads(line) for line in Path(directory, 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            final_epoch = next((row for row in metric_rows if row.get('stage') == 'final' and row.get('event') == 'epoch'))
            self.assertIn('validation_forward_loss', final_epoch)
            self.assertNotIn('validation_observed_ce_loss', final_epoch)
            self.assertNotIn('validation_loss', final_epoch)
            actual_best_sha = hashlib.sha256(Path(directory, 'posterior_best.pt').read_bytes()).hexdigest()
            best_payload = read_checkpoint(Path(directory, 'posterior_best.pt'), 'cpu')
            last_payload = read_checkpoint(Path(directory, 'last.pt'), 'cpu')
            self.assertNotEqual(best_payload['dual_t_state']['best_posterior_checkpoint_sha256'], actual_best_sha)
            self.assertEqual(last_payload['dual_t_state']['best_posterior_checkpoint_sha256'], actual_best_sha)
            snapshot_mtime = Path(directory, 'posterior_snapshot.npz').stat().st_mtime_ns
            transition_mtime = Path(directory, 'transition_artifact.npz').stat().st_mtime_ns
            restored = _dual_t_forward__algorithm(directory, posterior_epochs=1, final_epochs=1)
            restored.resume(Path(directory, 'last.pt'))
            restored.run()
            self.assertEqual(Path(directory, 'posterior_snapshot.npz').stat().st_mtime_ns, snapshot_mtime)
            self.assertEqual(Path(directory, 'transition_artifact.npz').stat().st_mtime_ns, transition_mtime)

    def test_transition_snapshot_is_loaded_from_best_not_current_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _dual_t_forward__algorithm(directory, posterior_epochs=1, final_epochs=1)
            algorithm.train_posterior()
            with torch.no_grad():
                algorithm.posterior_algorithm.model.weight.zero_()
                algorithm.posterior_algorithm.model.weight[2, 0] = 10.0
            algorithm.estimate_transition()
            expected = torch.softmax(torch.eye(3) * 5.0, dim=1).numpy()
            first_by_class = algorithm.snapshot.noisy_probabilities[[0, 2, 4]]
            np.testing.assert_allclose(first_by_class, expected, atol=1e-07)

    def test_clean_target_is_not_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            torch.manual_seed(11)
            first = _dual_t_forward__algorithm(first_dir, posterior_epochs=1, final_epochs=1, clean_offset=0)
            first.train_posterior()
            first.estimate_transition()
            torch.manual_seed(11)
            second = _dual_t_forward__algorithm(second_dir, posterior_epochs=1, final_epochs=1, clean_offset=2)
            second.train_posterior()
            second.estimate_transition()
            self.assertEqual(first.snapshot.snapshot_hash, second.snapshot.snapshot_hash)
            np.testing.assert_allclose(first.transition.matrix, second.transition.matrix)

    def test_posterior_training_interruption_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _dual_t_forward__algorithm(directory)
            first.train_posterior(max_epochs=1)
            self.assertIs(first.state.phase, DualTPhase.POSTERIOR_TRAINING)
            self.assertEqual(first.state.posterior_completed_epochs, 1)
            restored = _dual_t_forward__algorithm(directory)
            restored.resume(Path(directory) / 'last.pt')
            final = restored.run()
            self.assertEqual(final['completed_posterior_epochs'], 2)
            self.assertEqual(final['posterior_global_step'], 4)

    def test_transition_ready_resume_requires_and_loads_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _dual_t_forward__algorithm(directory)
            first.train_posterior()
            first.estimate_transition()
            self.assertIs(first.state.phase, DualTPhase.TRANSITION_READY)
            restored = _dual_t_forward__algorithm(directory)
            restored.resume(Path(directory) / 'last.pt')
            self.assertIsNotNone(restored.transition)
            self.assertEqual(restored.transition.artifact_hash, first.transition.artifact_hash)

    def test_transition_ready_resume_rejects_missing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _dual_t_forward__algorithm(directory)
            first.train_posterior()
            first.estimate_transition()
            Path(directory, 'posterior_snapshot.npz').unlink()
            restored = _dual_t_forward__algorithm(directory)
            with self.assertRaisesRegex(FileNotFoundError, 'artifact missing'):
                restored.resume(Path(directory) / 'last.pt')

    def test_transition_save_failure_does_not_advance_phase_or_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _dual_t_forward__algorithm(directory)
            algorithm.train_posterior()
            self.assertIs(algorithm.state.phase, DualTPhase.POSTERIOR_READY)
            last_before = Path(directory, 'last.pt').read_bytes()
            with patch.object(TransitionArtifact, 'save', side_effect=OSError('simulated artifact save failure')):
                with self.assertRaisesRegex(OSError, 'simulated'):
                    algorithm.estimate_transition()
            self.assertIs(algorithm.state.phase, DualTPhase.POSTERIOR_READY)
            self.assertEqual(algorithm.state.posterior_snapshot_hash, '')
            self.assertEqual(algorithm.state.transition_artifact_hash, '')
            self.assertEqual(Path(directory, 'last.pt').read_bytes(), last_before)

    def test_transition_ready_resume_rejects_artifact_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _dual_t_forward__algorithm(directory)
            first.train_posterior()
            first.estimate_transition()
            transition_path = Path(directory, 'transition_artifact.npz')
            with np.load(transition_path, allow_pickle=False) as data:
                matrix = data['matrix'].copy()
                metadata_json = data['metadata_json'].copy()
            matrix[0] = np.roll(matrix[0], 1)
            np.savez_compressed(transition_path, matrix=matrix, metadata_json=metadata_json)
            restored = _dual_t_forward__algorithm(directory)
            with self.assertRaisesRegex(ValueError, 'hash'):
                restored.resume(Path(directory) / 'last.pt')

    def test_resume_rejects_posterior_best_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _dual_t_forward__algorithm(directory)
            first.train_posterior(max_epochs=1)
            with Path(directory, 'posterior_best.pt').open('ab') as stream:
                stream.write(b'tamper')
            restored = _dual_t_forward__algorithm(directory)
            with self.assertRaisesRegex(ValueError, 'checkpoint hash mismatch'):
                restored.resume(Path(directory) / 'last.pt')

    def test_final_training_interruption_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _dual_t_forward__algorithm(directory)
            first.train_posterior()
            first.estimate_transition()
            first.start_final_training()
            first.train_final(max_epochs=1)
            self.assertIs(first.state.phase, DualTPhase.FINAL_TRAINING)
            self.assertEqual(first.state.final_completed_epochs, 1)
            restored = _dual_t_forward__algorithm(directory)
            restored.resume(Path(directory) / 'last.pt')
            final = restored.run()
            self.assertEqual(final['completed_final_epochs'], 2)
            self.assertEqual(final['final_global_step'], 4)

    def test_posterior_best_cannot_be_used_as_run_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _dual_t_forward__algorithm(directory)
            first.train_posterior(max_epochs=1)
            restored = _dual_t_forward__algorithm(directory)
            with self.assertRaisesRegex(ValueError, 'run-state checkpoint'):
                restored.resume(Path(directory) / 'posterior_best.pt')
