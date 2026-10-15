"""Merged unit tests; source modules were consolidated without changing assertions."""

# --- merged from test_fine.py ---
import unittest

# --- merged from test_fine.py ---
import torch

# --- merged from test_fine.py ---
from lnl_toolbox.algorithms.fine import FINERegularizer

# --- merged from test_fine.py ---
from lnl_toolbox.selectors.sed import SEDSelector, SelfAdaptiveClassSelector, SelfAdaptiveConfidenceReweighting

# --- merged from test_fine.py ---
from lnl_toolbox.selectors.base import SelectionInput

# --- merged from test_fine.py ---
class _fine_FINETest(unittest.TestCase):

    def test_regularizer_matches_paper_two_terms_and_state_round_trip(self) -> None:
        regularizer = FINERegularizer(beta=0.001, gamma=0.1, seed=4)
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        targets = torch.tensor([0, 0])
        pseudo = torch.tensor([0, 1])
        value = regularizer(logits, targets, rejected_mask=torch.ones(2, dtype=torch.bool), pseudo_labels=pseudo)
        probabilities = torch.softmax(logits, dim=1)
        expected = 0.1 * -torch.log(1.0 + 1e-07 - probabilities[1, 1]) / 2 + 0.001 * torch.log_softmax(logits, dim=1)[1, 0] / 2
        self.assertTrue(torch.allclose(value, expected))
        value.backward()
        state = regularizer.state_dict()
        restored = FINERegularizer(beta=0.001, gamma=0.1, seed=4)
        restored.load_state_dict(state)
        self.assertTrue(torch.isfinite(value))

    def test_nsnl_uses_complementary_class_and_one_over_c(self) -> None:
        regularizer = FINERegularizer(beta=0.0, gamma=1.0, seed=4)
        logits = torch.log(torch.tensor([[0.8, 0.1, 0.1]]))
        value = regularizer(
            logits,
            torch.tensor([0]),
            rejected_mask=torch.tensor([True]),
            pseudo_labels=torch.tensor([1]),
        )
        expected = -torch.log(torch.tensor(0.9 + 1.0e-7)) / 3
        torch.testing.assert_close(value, expected)

    def test_paper_coefficients_loss_and_gradient_for_three_and_hundred_classes(self) -> None:
        # Paper Eqs. (2)-(5), not SED_FINE.py's conflicting implementation.
        for classes in (3, 100):
            with self.subTest(classes=classes):
                logits = torch.linspace(-2., 2., 4*classes, dtype=torch.float64).reshape(4,classes).requires_grad_()
                targets = torch.tensor([0,1,2,0])
                rejected = torch.tensor([False,True,True,True])
                pseudo = torch.tensor([1,2,2,2])
                mask = rejected & pseudo.ne(targets)
                generator = torch.Generator().manual_seed(17)
                complementary = (targets + torch.randint(1,classes,targets.shape,generator=generator))%classes
                regularizer = FINERegularizer(seed=17)
                self.assertEqual((regularizer.beta,regularizer.gamma),(.001,.1))
                actual = regularizer(logits,targets,rejected_mask=rejected,pseudo_labels=pseudo)
                forgetting = torch.log_softmax(logits,1).gather(1,targets[:,None]).squeeze(1)/classes
                suppression = -torch.log(1+regularizer.probability_floor-torch.softmax(logits,1).gather(1,complementary[:,None]).squeeze(1))/classes
                expected = .001*forgetting[mask].mean()+.1*suppression[mask].mean()
                torch.testing.assert_close(actual,expected)
                actual_grad = torch.autograd.grad(actual,logits,retain_graph=True)[0]
                expected_grad = torch.autograd.grad(expected,logits)[0]
                torch.testing.assert_close(actual_grad,expected_grad)
                self.assertEqual(actual_grad[~mask].abs().sum().item(),0.)

    def test_beta_alone_weights_forgetting_with_paper_scaling(self) -> None:
        logits = torch.tensor([[1.,-1.,0.]],dtype=torch.float64,requires_grad=True)
        targets = torch.tensor([0])
        actual = FINERegularizer(beta=1.,gamma=0.)(logits,targets,
            rejected_mask=torch.tensor([True]),pseudo_labels=torch.tensor([1]))
        expected = torch.log_softmax(logits,1)[0,0]/3
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits)[0])

    def test_complementary_rng_resumes_exactly(self) -> None:
        logits = torch.tensor([[2.0, 1.0, -1.0]])
        targets = torch.tensor([0])
        kwargs = {
            "rejected_mask": torch.tensor([True]),
            "pseudo_labels": torch.tensor([1]),
        }
        regularizer = FINERegularizer(beta=0.0, gamma=1.0, seed=9)
        regularizer(logits, targets, **kwargs)
        state = regularizer.state_dict()
        expected = regularizer(logits, targets, **kwargs)
        restored = FINERegularizer(beta=0.0, gamma=1.0, seed=9)
        restored.load_state_dict(state)
        actual = restored(logits, targets, **kwargs)
        torch.testing.assert_close(actual, expected)

    def test_sed_keeps_at_least_one_sample(self) -> None:
        result = SEDSelector(threshold=0.0).select(SelectionInput(torch.tensor([0.2, 0.4]), torch.tensor([3, 4])))
        self.assertTrue(bool(result.selected_mask.any()))
        self.assertTrue(torch.equal(result.rejected_mask, ~result.selected_mask))

    def test_regularizer_uses_rejected_subset_and_empty_subset_is_zero(self) -> None:
        regularizer = FINERegularizer(beta=0.001, gamma=0.1, seed=4)
        logits = torch.randn(3, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2])
        rejected = torch.tensor([False, True, True])
        value = regularizer(logits, targets, rejected_mask=rejected, pseudo_labels=torch.tensor([1, 0, 0]))
        value.backward()
        self.assertEqual(float(logits.grad[0].abs().sum()), 0.0)
        empty_logits = torch.randn(2, 3, requires_grad=True)
        empty_value = regularizer(empty_logits, torch.tensor([0, 1]), rejected_mask=torch.zeros(2, dtype=torch.bool))
        empty_value.backward()
        self.assertEqual(float(empty_value), 0.0)

    def test_scs_and_scr_are_stateful_and_finite(self) -> None:
        probabilities = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([0, 1, 1, 0])
        scs = SelfAdaptiveClassSelector(2, momentum=0.0, quantile=None)
        mask = scs.select_epoch(probabilities, targets)
        self.assertEqual(mask.shape, targets.shape)
        restored_scs = SelfAdaptiveClassSelector(2, momentum=0.0, quantile=None)
        restored_scs.load_state_dict(scs.state_dict())
        self.assertTrue(torch.equal(restored_scs.local, scs.local))
        scr = SelfAdaptiveConfidenceReweighting(2, momentum=0.0)
        weights = scr.weights(probabilities)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(bool(((weights > 0.0) & (weights <= 1.0)).all()))
        restored_scr = SelfAdaptiveConfidenceReweighting(2, momentum=0.0)
        restored_scr.load_state_dict(scr.state_dict())
        self.assertTrue(torch.equal(restored_scr.mean, scr.mean))

# --- merged from test_fine_training.py ---
import tempfile

import json

# --- merged from test_fine_training.py ---
import unittest

# --- merged from test_fine_training.py ---
import pickle

# --- merged from test_fine_training.py ---
from pathlib import Path

# --- merged from test_fine_training.py ---
from unittest.mock import patch

# --- merged from test_fine_training.py ---
import numpy as np

# --- merged from test_fine_training.py ---
import torch

# --- merged from test_fine_training.py ---
from PIL import Image

# --- merged from test_fine_training.py ---
from lnl_toolbox.data.cifar import CifarData

# --- merged from test_fine_training.py ---
from lnl_toolbox.data.multi_view import IndexedMultiViewCifarDataset, build_strong_cifar_transform

# --- merged from test_fine_training.py ---
from lnl_toolbox.models.fine_cnn import FineSevenCNN

# --- merged from test_fine_training.py ---
from lnl_toolbox.training.fine_experiment import run_fine_experiment

# --- merged from test_fine_training.py ---
from lnl_toolbox.training.model_ema import ModelEMA

# --- merged from test_fine_training.py ---
def _fine_training__cifar(split: str, samples_per_class: int) -> CifarData:
    labels = np.repeat(np.arange(100, dtype=np.int64), samples_per_class)
    images = np.zeros((labels.size, 32, 32, 3), dtype=np.uint8)
    images[:, 0, 0, 0] = labels.astype(np.uint8)
    return CifarData(images, labels, tuple(map(str, range(100))), split, 'cifar100')

# --- merged from test_fine_training.py ---
class _fine_training_FINETrainingTest(unittest.TestCase):

    def test_seven_cnn_exposes_logits_and_features(self) -> None:
        model = FineSevenCNN(100, base_width=2)
        output = model.forward_with_features(torch.randn(2, 3, 32, 32))
        self.assertEqual(output.logits.shape, (2, 100))
        self.assertEqual(output.features.shape, (2, 16))
        outputs = model.forward_outputs(torch.randn(2, 3, 32, 32))
        self.assertEqual(outputs['logits'].shape, (2, 100))
        self.assertEqual(outputs['prob'].shape, (2, 100))

    def test_seven_cnn_initialization_matches_official_scope(self) -> None:
        model = FineSevenCNN(100, base_width=64)
        linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
        convolution_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        self.assertEqual(len(linear_layers), 4)
        self.assertEqual(len(convolution_layers), 6)
        for layer in linear_layers:
            self.assertTrue(torch.isfinite(layer.weight).all())
            self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))

    def test_model_ema_round_trip(self) -> None:
        model = torch.nn.Linear(2, 2)
        ema = ModelEMA(model, 0.9)
        with torch.no_grad():
            model.weight.add_(1.0)
        ema.update(model)
        restored = ModelEMA(model, 0.9)
        restored.load_state_dict(ema.state_dict())
        self.assertTrue(torch.equal(restored.model.weight, ema.model.weight))

    def test_fine_ema_matches_official_parameter_only_updates(self) -> None:
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
        ema = ModelEMA(model, 0.5, update_buffers=False)
        initial_mean = ema.model[0].running_mean.clone()
        with torch.no_grad():
            model[0].running_mean.add_(3.0)
            model[1].weight.add_(2.0)
        before = ema.model[1].weight.clone()
        ema.update(model)
        self.assertTrue(torch.equal(ema.model[0].running_mean, initial_mean))
        torch.testing.assert_close(ema.model[1].weight, 0.5 * before + 0.5 * model[1].weight)

    def test_multi_view_preserves_global_index_and_target(self) -> None:
        data = _fine_training__cifar('train', 1)
        transform = lambda image: torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
        dataset = IndexedMultiViewCifarDataset(data, np.asarray([7]), weak_transform=transform, strong_transform=transform, targets_by_index={7: 3})
        sample = dataset[0]
        self.assertEqual(sample['index'], 7)
        self.assertEqual(sample['target'], 3)
        self.assertEqual(sample['input'].shape, sample['strong_input'].shape)

    def test_official_cifar_strong_policy_returns_tensor(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        transform = build_strong_cifar_transform(policy='official_cifar10')
        result = transform(Image.fromarray(image, mode='RGB'))
        self.assertEqual(tuple(result.shape), (3, 32, 32))
        pickle.dumps(transform)

    def test_two_epoch_cli_lifecycle_writes_resumable_outputs(self) -> None:
        train = _fine_training__cifar('train', 2)
        test = _fine_training__cifar('test', 1)
        config = {'seed': 3, 'data': {'name': 'cifar100', 'root': 'unused', 'validation_size': 100, 'max_test_samples': 100, 'augment': False, 'strong_magnitude': 1}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 3}, 'loader': {'batch_size': 100, 'num_workers': 0, 'pin_memory': False}, 'model': {'name': 'tiny_cnn', 'width': 2}, 'optimizer': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.0}, 'scheduler': {'eta_min': 0.0005}, 'trainer': {'epochs': 2, 'device': 'cpu'}, 'fine': {'warmup_epochs': 1, 'warmup_lr': 0.01, 'ema_momentum': 0.9, 'momentum_scs': 0.9, 'momentum_scr': 0.9, 'beta': 0.1, 'gamma': 0.002, 'alpha': 1.0}}
        with tempfile.TemporaryDirectory() as directory:
            with patch('lnl_toolbox.data.sources.load_cifar100', side_effect=lambda _root, split: train if split == 'train' else test):
                result = run_fine_experiment(config, directory)
            self.assertTrue((result / 'last.pt').is_file())
            self.assertTrue((result / 'metrics.jsonl').is_file())
            self.assertTrue((result / 'training_curves.svg').is_file())
            payload = torch.load(result / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(payload['completed_epoch'], 1)
            self.assertIn('ema', payload)
            self.assertIn('scs', payload)
            self.assertIn('scr', payload)
            rows = [json.loads(line) for line in (result / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([row['event'] for row in rows], ['epoch', 'epoch', 'final'])
            self.assertIn('test_loss', rows[0])
            self.assertIn('validation_loss', rows[0])
            self.assertEqual(rows[-1]['selection_split'], 'none')
            self.assertFalse(rows[-1]['test_selection_leakage'])
            self.assertEqual(rows[-1]['epochs_averaged'], 2)
            self.assertAlmostEqual(rows[-1]['last_10_test_accuracy'], sum(row['test_accuracy'] for row in rows[:-1])/2)

    def test_no_validation_reports_test_per_epoch_without_retesting_completed_resume(self) -> None:
        train = _fine_training__cifar('train', 1)
        test = _fine_training__cifar('test', 1)
        config = {'seed': 3, 'data': {'name': 'cifar100', 'root': 'unused', 'validation_size': 0, 'max_test_samples': 100, 'augment': False, 'strong_magnitude': 1}, 'noise': {'name': 'symmetric', 'rate': 0.2, 'seed': 3}, 'loader': {'batch_size': 100, 'num_workers': 0, 'pin_memory': False}, 'model': {'name': 'tiny_cnn', 'width': 2}, 'optimizer': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.0}, 'scheduler': {'eta_min': 0.0005}, 'trainer': {'epochs': 1, 'device': 'cpu'}, 'fine': {'warmup_epochs': 1, 'warmup_lr': 0.01, 'ema_momentum': 0.9, 'momentum_scs': 0.9, 'momentum_scr': 0.9, 'beta': 0.1, 'gamma': 0.002, 'alpha': 1.0}}
        calls = []

        def evaluate(_model, loader, _criterion, _device):
            calls.append(loader)
            return {'loss': 1.0, 'accuracy': 0.25, 'samples': float(len(loader.dataset))}

        with tempfile.TemporaryDirectory() as directory:
            with patch('lnl_toolbox.data.sources.load_cifar100', side_effect=lambda _root, split: train if split == 'train' else test), patch('lnl_toolbox.training.fine_experiment.evaluate_classification', side_effect=evaluate):
                result = run_fine_experiment(config, directory)
                run_fine_experiment(config, resume=result / 'last.pt')
            rows = [json.loads(line) for line in (result / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertEqual([row['event'] for row in rows], ['epoch', 'final'])
        self.assertEqual(sum(row['event'] == 'final' for row in rows), 1)
        self.assertNotIn('validation_loss', rows[0])
        self.assertIn('test_loss', rows[0])
        self.assertEqual(rows[-1]['test_loss'], 1.0)

    def test_last_ten_report_uses_only_tail_without_model_selection(self) -> None:
        train, test = _fine_training__cifar('train',1), _fine_training__cifar('test',1)
        config = {'seed':3,'data':{'name':'cifar100','root':'unused','validation_size':0,'augment':False},
            'noise':{'name':'symmetric','rate':.2},'loader':{'batch_size':100,'num_workers':0},
            'model':{'name':'tiny_cnn','width':1},'optimizer':{'name':'sgd','lr':.01},
            'scheduler':{'eta_min':.0005},'trainer':{'epochs':12,'device':'cpu'},'fine':{'warmup_epochs':12,'warmup_lr':.01}}
        evaluations = [{'loss':1.,'accuracy':i/100} for i in range(12)]
        with tempfile.TemporaryDirectory() as directory, \
             patch('lnl_toolbox.data.sources.load_cifar100',side_effect=lambda _root,split:train if split=='train' else test), \
             patch('lnl_toolbox.training.fine_experiment._train_warmup',return_value=(1.,.1)), \
             patch('lnl_toolbox.training.fine_experiment.evaluate_classification',side_effect=evaluations) as evaluate:
            root=run_fine_experiment(config,directory)
            rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(evaluate.call_count,12)
            self.assertEqual(rows[-1]['epochs_averaged'],10)
            self.assertAlmostEqual(rows[-1]['last_10_test_accuracy'],sum(range(2,12))/1000)
            self.assertEqual(rows[-1]['selection_split'],'none')
            self.assertFalse((root/'best.pt').exists())

    def test_augmented_snapshot_and_exact_interrupted_resume(self) -> None:
        from lnl_toolbox.training.data_service import PreparedData
        from lnl_toolbox.training.checkpoint import atomic_save as save
        train, test = _fine_training__cifar('train',2), _fine_training__cifar('test',1)
        config = {'seed':3,'data':{'name':'cifar100','root':'unused','validation_size':0,'augment':True,'strong_magnitude':1},
            'noise':{'name':'symmetric','rate':.2},'loader':{'batch_size':100,'num_workers':0,'pin_memory':False},
            'model':{'name':'tiny_cnn','width':2},'optimizer':{'name':'sgd','lr':.01,'momentum':.0},
            'scheduler':{'eta_min':.0005},'trainer':{'epochs':3,'device':'cpu'},'fine':{'warmup_epochs':1,'warmup_lr':.01}}
        with tempfile.TemporaryDirectory() as directory, \
             patch('lnl_toolbox.data.sources.load_cifar100',side_effect=lambda _root,split:train if split=='train' else test):
            root=Path(directory)
            original=PreparedData.dynamic_dataset
            with patch.object(PreparedData,'dynamic_dataset',autospec=True,side_effect=original) as dynamic:
                complete=run_fine_experiment(config,root/'complete')
                calls=[call for call in dynamic.call_args_list if call.kwargs.get('views')==('weak',)]
                self.assertTrue(calls)
                self.assertTrue(all(call.kwargs['training'] for call in calls))
            def interrupt(payload,path):
                save(payload,path)
                if payload['completed_epoch']==1:
                    raise RuntimeError('test interruption')
            with patch('lnl_toolbox.training.fine_experiment.atomic_save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'test interruption'):
                    run_fine_experiment(config,root/'interrupted')
            resumed_path=root/'interrupted'/'last.pt'
            run_fine_experiment(config,resume=resumed_path)
            complete_state=torch.load(complete/'last.pt',weights_only=False)
            resumed=torch.load(resumed_path,weights_only=False)
            self.assertEqual(resumed['completed_epoch'],2)
            for name,value in complete_state['model'].items():
                torch.testing.assert_close(resumed['model'][name],value,rtol=0,atol=0)
            self.assertEqual(resumed['metrics'],complete_state['metrics'])
            final=[json.loads(line) for line in (resumed_path.parent/'metrics.jsonl').read_text().splitlines()][-1]
            self.assertAlmostEqual(final['last_10_test_accuracy'],sum(row['test_accuracy'] for row in resumed['metrics'])/3)

    def test_formal_scratch_and_web_defaults_follow_paper(self) -> None:
        import yaml
        from web.command_console import _config_schema
        config=yaml.safe_load(Path('configs/experiment/fine_cifar100n_reproduction.yaml').read_text(encoding='utf-8'))
        self.assertEqual((config['fine']['beta'],config['fine']['gamma']),(.001,.1))
        scratch=yaml.safe_load(Path('src/lnl_toolbox/scratch/recipes/papers/fine.yaml').read_text(encoding='utf-8'))
        state=next(step['params'] for step in scratch['steps'] if step['block']=='create_fine_state')
        self.assertEqual((state['beta'],state['gamma']),(.001,.1))
        refresh=next(step for step in scratch['steps'] if step['block']=='epoch_loop')['steps']
        snapshot_loader=next(step['params'] for step in refresh if step['block']=='refresh_epoch_loader' and step['params']['save_as']=='train_eval_loader')
        self.assertEqual(snapshot_loader['role'],'train')
        registry=yaml.safe_load(Path('web/lnl_parameter_metadata_registry_revised.yaml').read_text(encoding='utf-8'))
        parameters=registry['methods']['fine']['parameters']
        self.assertEqual(parameters['basic']['noise.seed']['current'],23)
        self.assertEqual(parameters['advanced']['noise.rng']['current'],'numpy_legacy')
        schema=_config_schema('fine-cifar100n-reproduction')
        fields={field['path']:field for field in schema['fields']}
        self.assertFalse(fields['model.dropout']['visible'])
        self.assertEqual(fields['fine.beta']['value'],.001)
        self.assertEqual(fields['fine.gamma']['value'],.1)
