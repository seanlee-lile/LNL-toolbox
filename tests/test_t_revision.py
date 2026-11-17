from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from lnl_toolbox.algorithms.t_revision import AdditiveTransitionRevision, RevisedTransitionArtifact, TRevisionPhase, TRevisionState, validate_revision_optimizer
from lnl_toolbox.noise.transition import TransitionArtifact



class _t_revision_algorithm_TRevisionTransitionTest(unittest.TestCase):

    def test_delta_starts_zero_and_raw_addition_is_unprojected(self) -> None:
        initial = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        revision = AdditiveTransitionRevision(initial)
        torch.testing.assert_close(revision(), initial)
        with torch.no_grad():
            revision.delta.copy_(torch.tensor([[0.4, -0.5], [0.0, 0.2]]))
        expected = torch.tensor([[1.2, -0.3], [0.1, 1.1]])
        torch.testing.assert_close(revision(), expected)
        diagnostics = revision.diagnostics()
        self.assertIs(diagnostics['non_negative'], False)
        self.assertIs(diagnostics['row_stochastic'], False)

    def test_optimizer_must_contain_model_and_delta_exactly_once(self) -> None:
        model = nn.Linear(2, 2)
        revision = AdditiveTransitionRevision(torch.eye(2))
        optimizer = torch.optim.Adam(list(model.parameters()) + [revision.delta], lr=0.001)
        validate_revision_optimizer(optimizer, model.parameters(), revision)
        missing = torch.optim.Adam(model.parameters(), lr=0.001)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            validate_revision_optimizer(missing, model.parameters(), revision)
        duplicate = torch.optim.Adam(list(model.parameters()) + [revision.delta], lr=0.001)
        duplicate.param_groups[0]['params'].append(revision.delta)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_revision_optimizer(duplicate, model.parameters(), revision)

    def test_one_step_updates_model_and_delta(self) -> None:
        model = nn.Linear(2, 2)
        revision = AdditiveTransitionRevision(torch.tensor([[0.8, 0.2], [0.2, 0.8]]))
        optimizer = torch.optim.Adam(list(model.parameters()) + [revision.delta], lr=0.01)
        model_before = model.weight.detach().clone()
        delta_before = revision.delta.detach().clone()
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        logits = model(inputs)
        from lnl_toolbox.algorithms.t_revision import t_revision_reweight_objective
        objective = t_revision_reweight_objective(logits, torch.tensor([0, 1]), revision(), denominator_floor=1e-12).objective
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        self.assertFalse(torch.equal(model_before, model.weight.detach()))
        self.assertFalse(torch.equal(delta_before, revision.delta.detach()))

    def test_revised_artifact_is_not_a_transition_artifact_and_roundtrips(self) -> None:
        artifact = RevisedTransitionArtifact(initial_transition=np.asarray([[0.8, 0.2], [0.1, 0.9]]), delta=np.asarray([[0.3, -0.4], [0.0, 0.1]]), source_initial_artifact_hash='a' * 64, stage2a_best_checkpoint_sha256='b' * 64, best_noisy_validation_accuracy=0.75)
        self.assertNotIsInstance(artifact, TransitionArtifact)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'revised.npz'
            artifact.save(path)
            restored = RevisedTransitionArtifact.load(path)
        self.assertEqual(restored.artifact_hash, artifact.artifact_hash)
        np.testing.assert_allclose(restored.revised_transition, artifact.revised_transition)

class _t_revision_algorithm_TRevisionStateTest(unittest.TestCase):

    def test_only_forward_phase_transitions_are_allowed(self) -> None:
        state = TRevisionState()
        with self.assertRaisesRegex(ValueError, 'illegal T-Revision'):
            state.advance(TRevisionPhase.TRANSITION_INITIALIZED)
        state.stage1_completed_epochs = 1
        state.stage1_best_epoch = 0
        state.stage1_best_hash = 'a' * 64
        state.advance(TRevisionPhase.STAGE1_READY)
        state.snapshot_hash = 'b' * 64
        state.initial_transition_hash = 'c' * 64
        state.advance(TRevisionPhase.TRANSITION_INITIALIZED)
        state.advance(TRevisionPhase.CLASSIFIER_INITIALIZATION)
        state.stage2a_completed_epochs = 1
        state.stage2a_best_epoch = 0
        state.stage2a_best_hash = 'd' * 64
        state.advance(TRevisionPhase.CLASSIFIER_READY)
        state.advance(TRevisionPhase.REVISION_TRAINING)
        state.revision_completed_epochs = 1
        state.revision_best_epoch = 0
        state.revision_best_checkpoint_hash = 'f' * 64
        state.revised_transition_hash = 'e' * 64
        state.advance(TRevisionPhase.COMPLETED)
        restored = TRevisionState.from_state_dict(state.state_dict())
        self.assertIs(restored.phase, TRevisionPhase.COMPLETED)

import math
from torch.nn import functional as F
from lnl_toolbox.algorithms.t_revision import t_revision_reweight_objective

class _t_revision_objective_TRevisionObjectiveTest(unittest.TestCase):

    def test_matches_vectorized_equation_three_by_hand(self) -> None:
        logits = torch.tensor([[1.2, -0.4], [0.1, 0.8]], requires_grad=True)
        targets = torch.tensor([0, 1])
        transition = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
        result = t_revision_reweight_objective(logits, targets, transition, denominator_floor=1e-12)
        clean = torch.softmax(logits, dim=1)
        noisy = clean @ transition
        expected_weights = torch.stack([clean[0, 0] / noisy[0, 0], clean[1, 1] / noisy[1, 1]])
        expected = (expected_weights * F.cross_entropy(logits, targets, reduction='none')).mean()
        torch.testing.assert_close(result.objective, expected)
        self.assertAlmostEqual(result.weight_mean, float(expected_weights.mean()))

    def test_non_commuting_direction_is_clean_probability_times_transition(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
        targets = torch.tensor([1])
        transition = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]])
        result = t_revision_reweight_objective(logits, targets, transition, denominator_floor=0.0)
        clean = torch.softmax(logits, 1)
        correct = clean[0, 1] / (clean @ transition)[0, 1]
        reverse = clean[0, 1] / (clean @ transition.t())[0, 1]
        self.assertAlmostEqual(result.weight_mean, float(correct))
        self.assertNotAlmostEqual(result.weight_mean, float(reverse))

    def test_identity_transition_reduces_to_cross_entropy(self) -> None:
        logits = torch.randn(5, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0, 2])
        result = t_revision_reweight_objective(logits, targets, torch.eye(3), denominator_floor=0.0)
        torch.testing.assert_close(result.objective, F.cross_entropy(logits, targets))
        self.assertAlmostEqual(result.weight_min, 1.0)
        self.assertAlmostEqual(result.weight_max, 1.0)

    def test_classifier_and_transition_receive_gradients(self) -> None:
        logits = torch.tensor([[1.0, 0.2], [-0.4, 0.8]], requires_grad=True)
        transition = torch.tensor([[0.85, 0.15], [0.25, 0.75]], requires_grad=True)
        result = t_revision_reweight_objective(logits, torch.tensor([0, 1]), transition, denominator_floor=1e-12)
        result.objective.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(transition.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(transition.grad.abs().sum()), 0.0)


    def test_invalid_denominator_and_nonfinite_values_fail(self) -> None:
        logits = torch.tensor([[0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, 'strictly greater'):
            t_revision_reweight_objective(logits, torch.tensor([0]), torch.zeros(2, 2), denominator_floor=0.0)
        with self.assertRaisesRegex(ValueError, 'transition must be finite'):
            t_revision_reweight_objective(logits, torch.tensor([0]), torch.tensor([[math.nan, 0.0], [0.0, 1.0]]), denominator_floor=0.0)
        with self.assertRaisesRegex(ValueError, 'logits must be finite'):
            t_revision_reweight_objective(torch.tensor([[math.inf, 0.0]]), torch.tensor([0]), torch.eye(2), denominator_floor=0.0)

    def test_raw_negative_entries_are_not_clipped_when_denominator_is_valid(self) -> None:
        logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
        transition = torch.tensor([[1.1, -0.1], [0.2, 0.8]], requires_grad=True)
        result = t_revision_reweight_objective(logits, torch.tensor([0]), transition, denominator_floor=0.0)
        self.assertTrue(torch.isfinite(result.objective))
        result.objective.backward()
        self.assertIsNotNone(transition.grad)

import hashlib
import json
import os
import subprocess
import sys
from torch.utils.data import DataLoader
from lnl_toolbox.algorithms.t_revision import TRevisionAlgorithm, TRevisionConfig
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.experiment import run_experiment
from lnl_toolbox.training.t_revision_experiment import _preflight_model_output

class _t_revision_workflow__Classifier(nn.Module):

    def __init__(self, classes: int=3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(classes) * 2.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight.t()

def _t_revision_workflow__records(*, clean_offset: int=0):
    records = []
    for position, target in enumerate((0, 0, 1, 1, 2, 2)):
        records.append({'input': torch.nn.functional.one_hot(torch.tensor(target), num_classes=3).float(), 'target': torch.tensor(target), 'index': torch.tensor(100 + position * 11), 'clean_target': torch.tensor((target + clean_offset) % 3)})
    return records

def _t_revision_workflow__config(*, stage1_epochs: int=1, stage2a_epochs: int=1, revision_epochs: int=1):
    return {'method': 't_revision', 'seed': 5, 'data': {'name': 'fixture', 'width': 3, 'validation_size': 3}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 9, 'validation_targets': 'noisy', 'mapping_hash': 'a' * 64}, 'loader': {'batch_size': 3}, 'trainer': {'device': 'cpu'}, 'evaluation': {'selection_split': 'validation'}, 't_revision': {'objective': 'reweight', 'fidelity': 'paper_experiment_raw_additive', 'stage1': {'epochs': stage1_epochs, 'model': {'name': 'fixture'}, 'optimizer': {'name': 'sgd', 'lr': 0.01}, 'scheduler': {'name': 'none'}, 'best_metric': 'noisy_validation_accuracy'}, 'transition_initialization': {'method': 'pseudo_anchor_max_posterior', 'posterior_split': 'train', 'extraction_augmentation': False, 'tie_break': 'stable_sample_index'}, 'classifier_initialization': {'epochs': stage2a_epochs, 'optimizer': {'name': 'sgd', 'lr': 0.01}, 'scheduler': {'name': 'none'}, 'start_from': 'stage1_best', 'best_metric': 'revised_noisy_validation_accuracy'}, 'revision': {'epochs': revision_epochs, 'transition_mode': 'paper_experiment_raw_additive', 'delta_initialization': 'zeros', 'start_from': 'classifier_initialization_best', 'optimizer': {'name': 'adam', 'lr': 0.01}, 'scheduler': {'name': 'none'}, 'ratio': {'detach': False, 'clamp': 'none', 'denominator_floor': 1e-12}, 'best_metric': 'revised_noisy_validation_accuracy'}}}

def _t_revision_workflow__algorithm(directory: str | Path, *, stage1_epochs: int=1, stage2a_epochs: int=1, revision_epochs: int=1, clean_offset: int=0, device: str='cpu', stage_early_stopping=None):
    torch_device = torch.device(device)
    model = _t_revision_workflow__Classifier().to(torch_device)
    stage1_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def classifier_optimizer(value):
        return torch.optim.SGD(value.parameters(), lr=0.01)

    def revision_optimizer(value, revision):
        return torch.optim.Adam(list(value.parameters()) + [revision.delta], lr=0.01)
    generator = torch.Generator().manual_seed(5)
    train = DataLoader(_t_revision_workflow__records(clean_offset=clean_offset), batch_size=3, shuffle=True, generator=generator)
    posterior = DataLoader(_t_revision_workflow__records(clean_offset=clean_offset), batch_size=3, shuffle=False)
    validation = DataLoader(_t_revision_workflow__records(clean_offset=2), batch_size=3, shuffle=False)
    test = DataLoader(_t_revision_workflow__records(clean_offset=1), batch_size=3, shuffle=False)
    config = _t_revision_workflow__config(stage1_epochs=stage1_epochs, stage2a_epochs=stage2a_epochs, revision_epochs=revision_epochs)
    if stage_early_stopping is not None:
        config['early_stopping'] = {'scope': 'stages', **stage_early_stopping}
    return TRevisionAlgorithm(model=model, stage1_optimizer=stage1_optimizer, stage1_scheduler=None, classifier_optimizer_factory=classifier_optimizer, classifier_scheduler_factory=lambda optimizer: None, revision_optimizer_factory=revision_optimizer, revision_scheduler_factory=lambda optimizer: None, loss=nn.CrossEntropyLoss(reduction='none'), train_loader=train, posterior_loader=posterior, noisy_validation_loader=validation, clean_test_loader=test, device=torch_device, run_dir=directory, config=config, dataset='fixture', num_classes=3, noise_metadata={'manifest_sha256': 'b' * 64, 'mapping_hash': 'a' * 64})

def _t_revision_workflow__sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class _t_revision_workflow_TRevisionWorkflowTest(unittest.TestCase):

    def test_each_stage_stops_after_exactly_fifteen_bad_epochs_and_completes(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _t_revision_workflow__algorithm(
                directory, stage1_epochs=20, stage2a_epochs=20, revision_epochs=20,
                stage_early_stopping={'patience': 15})
            from lnl_toolbox.core.config_schema import normalize_experiment_config
            normalize_experiment_config(algorithm.config)
            stable = {'accuracy': .5, 'loss': 1.}
            with patch('lnl_toolbox.algorithms.t_revision.algorithm.evaluate_classification', return_value=stable), patch.object(algorithm, '_evaluate_noisy', return_value=stable):
                algorithm.run()
            self.assertIs(algorithm.state.phase, TRevisionPhase.COMPLETED)
            self.assertEqual(algorithm.state.stage1_completed_epochs, 16)
            self.assertEqual(algorithm.state.stage2a_completed_epochs, 16)
            self.assertEqual(algorithm.state.revision_completed_epochs, 16)
            self.assertTrue((Path(directory) / 'final_metrics.json').is_file())
            for stopper in algorithm.stage_stoppers.values():
                self.assertEqual(stopper.bad_epochs, 15)


    def test_config_rejects_unsupported_variants_and_test_selection(self) -> None:
        values = _t_revision_workflow__config()
        values['t_revision']['objective'] = 'forward'
        with self.assertRaisesRegex(NotImplementedError, 'objective: reweight'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        values['t_revision']['revision']['transition_mode'] = 'softmax'
        with self.assertRaisesRegex(ValueError, 'transition_mode'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        values['evaluation']['selection_split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'must use validation'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        values['data']['validation_size'] = 0
        with self.assertRaisesRegex(ValueError, 'validation_size'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        values['noise']['name'] = 'instance_dependent'
        with self.assertRaisesRegex(ValueError, 'class-dependent'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        del values['loader']
        with self.assertRaisesRegex(TypeError, 'loader'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        del values['trainer']['device']
        with self.assertRaisesRegex(ValueError, 'trainer.device'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        values['t_revision']['classifier_initialization']['start_from'] = 'last'
        with self.assertRaisesRegex(ValueError, 'start_from'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        values['t_revision']['revision']['start_from'] = 'stage1_best'
        with self.assertRaisesRegex(ValueError, 'start_from'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        values['t_revision']['revision']['delta_initialization'] = 'random'
        with self.assertRaisesRegex(ValueError, 'delta_initialization'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        values['t_revision']['revision']['ratio']['detach'] = True
        with self.assertRaisesRegex(ValueError, 'detach'):
            TRevisionConfig.from_mapping(values)
        values = _t_revision_workflow__config()
        values['t_revision']['revision']['ratio']['clamp'] = 'zero'
        with self.assertRaisesRegex(ValueError, 'clamp'):
            TRevisionConfig.from_mapping(values)

    def test_model_preflight_rejects_class_mismatch_and_preserves_backward(self) -> None:
        loader = [{'input': torch.ones(2, 3)}]
        wrong = nn.Linear(3, 2)
        with self.assertRaisesRegex(ValueError, 'class dimension'):
            _preflight_model_output(wrong, loader, torch.device('cpu'), 3)
        model = nn.Linear(3, 3)
        _preflight_model_output(model, loader, torch.device('cpu'), 3)
        model(torch.ones(2, 3)).sum().backward()
        self.assertIsNotNone(model.weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_model_preflight_does_not_create_inference_parameters(self) -> None:
        device = torch.device('cuda')
        model = nn.Linear(3, 3).to(device)
        _preflight_model_output(model, [{'input': torch.ones(2, 3)}], device, 3)
        model(torch.ones(2, 3, device=device)).sum().backward()
        self.assertIsNotNone(model.weight.grad)




    def test_t_revision_import_does_not_require_sklearn(self) -> None:
        code = "\nimport builtins\nreal_import = builtins.__import__\ndef guarded(name, *args, **kwargs):\n    if name == 'sklearn' or name.startswith('sklearn.'):\n        raise ModuleNotFoundError('blocked sklearn')\n    return real_import(name, *args, **kwargs)\nbuiltins.__import__ = guarded\nfrom lnl_toolbox.algorithms.t_revision import TRevisionAlgorithm\nfrom lnl_toolbox.training.t_revision_experiment import run_t_revision_experiment\nprint(TRevisionAlgorithm.__name__, run_t_revision_experiment.__name__)\n"
        environment = os.environ.copy()
        environment['PYTHONPATH'] = 'src'
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('TRevisionAlgorithm', result.stdout)



    def test_snapshot_and_anchor_transition_come_from_stage1_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _t_revision_workflow__algorithm(directory, stage1_epochs=2)
            algorithm.train_stage1()
            self.assertEqual(algorithm.state.stage1_best_epoch, 0)
            best = read_checkpoint(algorithm.stage1_best_path, 'cpu')
            best_weight = best['model']['weight'].clone()
            with torch.no_grad():
                algorithm.model.weight.add_(100.0 * torch.randn_like(algorithm.model.weight))
            algorithm.initialize_transition()
            torch.testing.assert_close(algorithm.model.weight.cpu(), best_weight)
            self.assertEqual(algorithm.initial_transition.metadata['stage1_best_checkpoint_sha256'], algorithm.state.stage1_best_hash)
            self.assertEqual(algorithm.initial_transition.source_snapshot_hash, algorithm.snapshot.snapshot_hash)

    def test_transition_initialization_rejects_snapshot_missing_target_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _t_revision_workflow__algorithm(directory)
            algorithm.train_stage1()
            records = [record for record in _t_revision_workflow__records() if int(record['target']) < 2]
            algorithm.posterior_loader = DataLoader(records, batch_size=2, shuffle=False)
            with self.assertRaisesRegex(ValueError, 'missing noisy target classes'):
                algorithm.initialize_transition()

    def test_resume_from_directory_continues_next_epoch(self) -> None:
        import yaml
        from unittest.mock import patch
        from lnl_toolbox.data.cifar import CifarData
        from lnl_toolbox.algorithms.t_revision import algorithm as impl
        from lnl_toolbox.training.service import ExperimentService
        config = yaml.safe_load(Path('configs/smoke/cifar10_t_revision_smoke.yaml').read_text(encoding='utf-8'))
        config['data'].update(root='unused', validation_size=10, max_train_samples=20, max_validation_samples=10, max_test_samples=10)
        config['loader']['batch_size'] = 10
        config['t_revision']['stage1']['epochs'] = 1
        config['t_revision']['stage1']['model']['width'] = 4
        config['t_revision']['classifier_initialization']['epochs'] = 1
        config['t_revision']['revision']['epochs'] = 2
        def source(_root, split):
            count = 40 if split == 'train' else 20
            return CifarData(np.zeros((count, 32, 32, 3), dtype=np.uint8), np.arange(count) % 10, tuple(map(str, range(10))), split, 'cifar10')
        save = impl.atomic_save
        def interrupt(payload, path):
            save(payload, path)
            if Path(path).name == 'last.pt' and payload['t_revision_state']['revision_completed_epochs'] == 1:
                raise RuntimeError('simulated interruption')
        with tempfile.TemporaryDirectory() as directory, patch('lnl_toolbox.data.sources.load_cifar10', side_effect=source):
            run_dir = Path(directory) / 'run'
            with patch.object(impl, 'atomic_save', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    ExperimentService().run(config, run_dir)
            self.assertEqual(read_checkpoint(run_dir / 'last.pt', 'cpu')['t_revision_state']['revision_completed_epochs'], 1)
            ExperimentService().resume(run_dir)
            self.assertEqual(read_checkpoint(run_dir / 'last.pt', 'cpu')['t_revision_state']['revision_completed_epochs'], 2)

    def test_stage2a_fixes_transition_and_revision_rebuilds_adam(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _t_revision_workflow__algorithm(directory)
            algorithm.train_stage1()
            algorithm.initialize_transition()
            transition_before = algorithm.initial_transition.matrix.copy()
            algorithm.start_classifier_initialization()
            self.assertIsInstance(algorithm.optimizer, torch.optim.SGD)
            algorithm.train_classifier_initialization()
            torch.testing.assert_close(torch.as_tensor(algorithm.initial_transition.matrix.copy()), torch.as_tensor(transition_before))
            algorithm.start_revision()
            self.assertIsInstance(algorithm.optimizer, torch.optim.Adam)
            self.assertEqual(len(algorithm.optimizer.state), 0)
            self.assertTrue(torch.equal(algorithm.revision.delta, torch.zeros_like(algorithm.revision.delta)))

    def test_clean_target_is_not_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            torch.manual_seed(17)
            first = _t_revision_workflow__algorithm(first_dir, clean_offset=0)
            first.train_stage1()
            first.initialize_transition()
            torch.manual_seed(17)
            second = _t_revision_workflow__algorithm(second_dir, clean_offset=2)
            second.train_stage1()
            second.initialize_transition()
            self.assertEqual(first.state.snapshot_hash, second.state.snapshot_hash)
            self.assertEqual(first.state.initial_transition_hash, second.state.initial_transition_hash)



    def test_dispatch_is_lazy_and_available(self) -> None:
        with unittest.mock.patch('lnl_toolbox.training.t_revision_experiment.run_t_revision_experiment', return_value=Path('fixture')) as runner:
            result = run_experiment(_t_revision_workflow__config())
        self.assertEqual(result, Path('fixture'))
        runner.assert_called_once()

    def test_unknown_method_fails_instead_of_supervised_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Unsupported training method'):
            run_experiment({'method': 'not_a_method'})

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_three_stage_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _t_revision_workflow__algorithm(directory, device='cuda')
            algorithm.run()
            self.assertIs(algorithm.state.phase, TRevisionPhase.COMPLETED)
