import math
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from lnl_toolbox.algorithms.cdr import CDRUpdatePolicy, critical_parameter_masks, official_code_parameter_masks
from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.algorithms.update_policy import ParameterUpdateInput
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.losses import CrossEntropyLoss
from lnl_toolbox.plugins.builtin import build_builtin_parameter_update_policy, create_builtin_catalog
from lnl_toolbox.selectors import SmallLossSelector
from lnl_toolbox.training.checkpoint import load_checkpoint, save_checkpoint
from lnl_toolbox.training.experiment import _validate_resume_config, _validate_supervised_config

class _cdr__VectorModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor([2.0, -1.0, 0.0, 4.0]))

    def forward(self) -> torch.Tensor:
        return self.values

class _cdr_CDRUpdatePolicyTest(unittest.TestCase):

    def test_author_mode_preserves_reporting_forward_and_batchnorm_updates(self) -> None:
        for mode, forwards in (("paper", 1), ("official_code", 2)):
            with self.subTest(mode=mode):
                torch.manual_seed(7)
                model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
                reference = copy.deepcopy(model)
                optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
                policy = CDRUpdatePolicy(.4, 0., compatibility_mode=mode)
                inputs = torch.tensor([[1., 2.], [2., 1.], [-1., 0.], [0., -1.]])
                targets = torch.tensor([0, 1, 0, 1])
                if mode == "official_code":
                    reference(inputs)
                objective = CrossEntropyLoss()(reference(inputs), targets).mean()
                CDRUpdatePolicy(.4, 0., compatibility_mode=mode).update(
                    ParameterUpdateInput(objective, reference, reference_optimizer, RunState()))
                algorithm = SupervisedClassificationAlgorithm(
                    model, optimizer, CrossEntropyLoss(), torch.device('cpu'), update_policy=policy)
                algorithm.step(Batch({'input': inputs, 'target': targets, 'index': torch.arange(4)}), RunState())
                self.assertEqual(int(model[0].num_batches_tracked), forwards)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, reference.state_dict()[name])

    def test_criticality_and_exact_top_k_match_equations_three_and_four(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0, 0.0, 4.0]))
        parameter.grad = torch.tensor([1.0, 3.0, 9.0, -0.25])
        result = critical_parameter_masks([('weight', parameter)], 0.5)
        self.assertEqual(result.eligible_parameters, 4)
        self.assertEqual(result.critical_parameters, 2)
        self.assertTrue(torch.equal(result.masks['weight'], torch.tensor([True, True, False, False])))

    def test_ties_use_parameter_name_then_flat_offset(self) -> None:
        first = torch.nn.Parameter(torch.ones(2))
        second = torch.nn.Parameter(torch.ones(2))
        first.grad = torch.ones(2)
        second.grad = torch.ones(2)
        result = critical_parameter_masks([('z_weight', second), ('a_weight', first)], 0.5)
        self.assertTrue(torch.equal(result.masks['a_weight'], torch.tensor([True, True])))
        self.assertTrue(torch.equal(result.masks['z_weight'], torch.tensor([False, False])))

    def test_cdr_update_matches_equations_five_and_six(self) -> None:
        model = _cdr__VectorModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        coefficients = torch.tensor([1.0, 3.0, 9.0, -0.25])
        objective = (model() * coefficients).sum()
        before = model.values.detach().clone()
        result = CDRUpdatePolicy(noise_rate=0.5, l1_decay=0.1).update(ParameterUpdateInput(objective=objective, model=model, optimizer=optimizer, run_state=RunState()))
        expected_gradient = torch.tensor([0.5 * 1.0 + 0.1, 0.5 * 3.0 - 0.1, 0.0, 0.1])
        self.assertTrue(torch.allclose(model.values, before - 0.2 * expected_gradient, atol=0.0, rtol=0.0))
        self.assertEqual(result.metrics['update_eligible_parameters'], 4.0)
        self.assertEqual(result.metrics['update_critical_parameters'], 2.0)
        self.assertEqual(result.metrics['update_critical_ratio'], 0.5)

    def test_noncritical_parameters_receive_only_l1_sign_update(self) -> None:
        model = _cdr__VectorModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        objective = (model() * torch.tensor([1.0, 0.5, 8.0, 0.1])).sum()
        before = model.values.detach().clone()
        CDRUpdatePolicy(noise_rate=0.75, l1_decay=0.2).update(ParameterUpdateInput(objective, model, optimizer, RunState()))
        delta = (before - model.values.detach()) / 0.1
        self.assertAlmostEqual(delta[1].item(), -0.2, places=6)
        self.assertEqual(delta[2].item(), 0.0)
        self.assertAlmostEqual(delta[3].item(), 0.2, places=6)

    def test_critical_then_noncritical_parameter_has_only_l1_second_step(self) -> None:
        model = _cdr__VectorModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0)
        policy = CDRUpdatePolicy(noise_rate=0.75, l1_decay=0.2)
        policy.update(ParameterUpdateInput((model() * torch.tensor([10.0, 0.0, 0.0, 0.0])).sum(), model, optimizer, RunState()))
        before_second_step = model.values.detach().clone()
        policy.update(ParameterUpdateInput((model() * torch.tensor([0.0, 10.0, 0.0, 0.0])).sum(), model, optimizer, RunState()))
        actual_delta = before_second_step[0] - model.values.detach()[0]
        expected_l1_delta = 0.1 * 0.2 * before_second_step[0].sign()
        torch.testing.assert_close(actual_delta, expected_l1_delta)

    def test_invalid_parameters_optimizer_and_gradients_fail(self) -> None:
        with self.assertRaises(ValueError):
            CDRUpdatePolicy(noise_rate=-0.1, l1_decay=0.1)
        with self.assertRaises(ValueError):
            CDRUpdatePolicy(noise_rate=1.0, l1_decay=0.1)
        with self.assertRaises(ValueError):
            CDRUpdatePolicy(noise_rate=0.2, l1_decay=-0.1)
        with self.assertRaises(ValueError):
            CDRUpdatePolicy(noise_rate=0.2, l1_decay=0.1, critical_scope='weights_only')
        model = _cdr__VectorModel()
        objective = model().sum()
        with self.assertRaisesRegex(TypeError, 'SGD'):
            CDRUpdatePolicy(0.2, 0.1).update(ParameterUpdateInput(objective, model, torch.optim.Adam(model.parameters()), RunState()))
        model = _cdr__VectorModel()
        objective = model().sum()
        with self.assertRaisesRegex(ValueError, 'weight_decay=0'):
            CDRUpdatePolicy(0.2, 0.1).update(ParameterUpdateInput(objective, model, torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01), RunState()))
        model = _cdr__VectorModel()
        objective = model().sum()
        with self.assertRaisesRegex(ValueError, 'non-critical.*L1'):
            CDRUpdatePolicy(0.2, 0.1).update(ParameterUpdateInput(objective, model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9), RunState()))
        parameter = torch.nn.Parameter(torch.tensor([float('nan')]))
        parameter.grad = torch.ones(1)
        with self.assertRaisesRegex(ValueError, 'finite'):
            critical_parameter_masks([('bad', parameter)], 0.2)
        embedding = torch.nn.Embedding(4, 2, sparse=True)
        sparse_objective = embedding(torch.tensor([0, 1])).sum()
        sparse_objective.backward()
        with self.assertRaisesRegex(ValueError, 'sparse'):
            critical_parameter_masks(embedding.named_parameters(), 0.2)

    def test_count_uses_ceil_and_all_trainable_dimensions_participate(self) -> None:
        weight = torch.nn.Parameter(torch.ones(4, 2))
        bias = torch.nn.Parameter(torch.ones(3))
        weight.grad = torch.ones_like(weight)
        bias.grad = torch.ones_like(bias)
        result = critical_parameter_masks([('weight', weight), ('bias', bias)], noise_rate=0.4)
        self.assertEqual(result.eligible_parameters, 11)
        self.assertEqual(result.critical_parameters, math.ceil(0.6 * 11))
        self.assertIn('bias', result.masks)

    def test_default_mode_is_identical_to_explicit_paper_mode(self) -> None:
        results = []
        metrics = []
        for mode in (None, 'paper'):
            model = _cdr__VectorModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
            objective = (model() * torch.tensor([1.0, 3.0, 9.0, -0.25])).sum()
            keyword = {} if mode is None else {'compatibility_mode': mode}
            result = CDRUpdatePolicy(noise_rate=0.5, l1_decay=0.1, **keyword).update(ParameterUpdateInput(objective, model, optimizer, RunState()))
            results.append(model.values.detach().clone())
            metrics.append(result.metrics)
        self.assertTrue(torch.equal(results[0], results[1]))
        self.assertEqual(metrics[0], metrics[1])

    def test_official_code_scope_threshold_ties_and_l2_step(self) -> None:
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, -1.0], [0.0, 4.0]]))
            model.bias.copy_(torch.tensor([1.0, -1.0]))
        weight_before = model.weight.detach().clone()
        bias_before = model.bias.detach().clone()
        coefficients = torch.tensor([[1.0, 3.0], [9.0, -0.25]])
        bias_coefficients = torch.tensor([2.0, -3.0])
        objective = (model.weight * coefficients).sum() + (model.bias * bias_coefficients).sum()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1)
        result = CDRUpdatePolicy(noise_rate=0.5, l1_decay=0.0, compatibility_mode='official_code').update(ParameterUpdateInput(objective, model, optimizer, RunState()))
        expected_weight_gradient = torch.tensor([[0.5, 1.5], [0.0, 0.0]]) + 0.1 * weight_before
        expected_bias_gradient = bias_coefficients + 0.1 * bias_before
        torch.testing.assert_close(model.weight, weight_before - 0.1 * expected_weight_gradient)
        torch.testing.assert_close(model.bias, bias_before - 0.1 * expected_bias_gradient)
        self.assertEqual(result.metrics['update_eligible_parameters'], 4.0)
        self.assertEqual(result.metrics['update_critical_parameters'], 2.0)
        tied = torch.nn.Parameter(torch.ones(4, 1))
        tied.grad = torch.ones_like(tied)
        masks = official_code_parameter_masks([('weight', tied)], 0.5)
        self.assertEqual(masks.eligible_parameters, 4)
        self.assertEqual(masks.critical_parameters, 4)

    def test_modes_are_validated_immutable_and_checkpoint_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, 'compatibility_mode'):
            CDRUpdatePolicy(0.4, 0.0, compatibility_mode='unknown')
        with self.assertRaisesRegex(ValueError, 'l1_decay=0'):
            CDRUpdatePolicy(0.4, 0.001, compatibility_mode='official_code')
        with self.assertRaisesRegex(ValueError, 'critical_scope'):
            CDRUpdatePolicy(0.4, 0.0, critical_scope='all_trainable', compatibility_mode='official_code')
        policy = CDRUpdatePolicy(0.4, 0.001)
        with self.assertRaises(AttributeError):
            policy.compatibility_mode = 'official_code'
        self.assertEqual(policy.state_dict(), {'compatibility_mode': 'paper'})
        policy.load_state_dict({})
        with self.assertRaisesRegex(ValueError, 'changed'):
            policy.load_state_dict({'compatibility_mode': 'official_code'})
        official = CDRUpdatePolicy(0.4, 0.0, compatibility_mode='official_code')
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            official.load_state_dict({})

    def test_selector_and_cdr_compose_without_clean_label_input(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        algorithm = SupervisedClassificationAlgorithm(model, optimizer, CrossEntropyLoss(), torch.device('cpu'), selector=SmallLossSelector(keep_rate=0.5), update_policy=CDRUpdatePolicy(noise_rate=0.5, l1_decay=0.0))
        algorithm.setup(ExperimentContext(Path.cwd()))
        result = algorithm.step(Batch({'input': torch.eye(2), 'target': torch.tensor([0, 1]), 'index': torch.tensor([9, 3])}), RunState())
        self.assertEqual(result.metrics['selected_ratio'], 0.5)
        self.assertEqual(result.metrics['update_critical_ratio'], 0.5)

    def test_plugin_kind_and_builder_are_distinct(self) -> None:
        catalog = create_builtin_catalog()
        self.assertEqual([item.name for item in catalog.find(kind='parameter_update_policy')], ['cdr', 'standard', 'step_milestone'])
        self.assertEqual(build_builtin_parameter_update_policy(None, catalog).name, 'standard')
        cdr = build_builtin_parameter_update_policy({'name': 'cdr', 'noise_rate': 0.4, 'l1_decay': 0.001, 'critical_scope': 'all_trainable'}, catalog)
        self.assertIsInstance(cdr, CDRUpdatePolicy)
        official = build_builtin_parameter_update_policy({'name': 'cdr', 'noise_rate': 0.4, 'l1_decay': 0.0, 'compatibility_mode': 'official_code'}, catalog)
        self.assertEqual(official.compatibility_mode, 'official_code')
        with self.assertRaises(ValueError):
            build_builtin_parameter_update_policy({'name': 'unknown'}, catalog)
        with self.assertRaises(TypeError):
            build_builtin_parameter_update_policy('cdr', catalog)

    def test_checkpoint_roundtrip_binds_policy_identity_and_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
            algorithm = SupervisedClassificationAlgorithm(model, optimizer, CrossEntropyLoss(), torch.device('cpu'), update_policy=CDRUpdatePolicy(0.4, 0.001))
            algorithm.setup(ExperimentContext(Path(directory)))
            state = RunState()
            algorithm.step(Batch({'input': torch.eye(2), 'target': torch.tensor([0, 1]), 'index': torch.tensor([4, 2])}), state)
            saved_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
            path = Path(directory) / 'last.pt'
            config = {'parameter_update': {'name': 'cdr', 'noise_rate': 0.4, 'l1_decay': 0.001}}
            save_checkpoint(path, algorithm, state, 0, config)
            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.05, momentum=0.0)
            restored_algorithm = SupervisedClassificationAlgorithm(restored_model, restored_optimizer, CrossEntropyLoss(), torch.device('cpu'), update_policy=CDRUpdatePolicy(0.4, 0.001))
            restored_algorithm.setup(ExperimentContext(Path(directory)))
            restored_state, epoch, payload = load_checkpoint(path, restored_algorithm, torch.device('cpu'))
            self.assertEqual(restored_state.step, 1)
            self.assertEqual(epoch, 0)
            self.assertEqual(payload['parameter_update_policy'], {'name': 'cdr', 'state': {'compatibility_mode': 'paper'}})
            for name, value in restored_model.named_parameters():
                self.assertTrue(torch.equal(value, saved_parameters[name]))
            self.assertEqual(restored_optimizer.param_groups[0]['momentum'], 0.0)
            self.assertFalse(restored_optimizer.state)
            standard_model = torch.nn.Linear(2, 2)
            standard_algorithm = SupervisedClassificationAlgorithm(standard_model, torch.optim.SGD(standard_model.parameters(), lr=0.05), CrossEntropyLoss(), torch.device('cpu'))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                load_checkpoint(path, standard_algorithm, torch.device('cpu'))

    def test_web_noise_prior_is_independent_and_l1_explanation_covers_both_groups(self) -> None:
        from web import command_console
        from lnl_toolbox.catalog import load_yaml
        recipe = 'cifar10-symmetric-cdr-reproduction'
        fields = {field['path']: field for field in command_console._config_schema(recipe)['fields']}
        self.assertEqual(fields['parameter_update.noise_rate']['linked_fields'], [])
        self.assertEqual(fields['noise.rate']['linked_fields'], [])
        for edited, other, value in (
            ('parameter_update.noise_rate', 'noise.rate', .2),
            ('noise.rate', 'parameter_update.noise_rate', .3),
        ):
            with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
                saved = command_console._save_config({
                    'recipe': recipe, 'path': str(Path(directory) / 'cdr.yaml'),
                    'patches': [{'path': edited, 'value': value}],
                    'acknowledge_paper_impact': True,
                })
                config = load_yaml(command_console.ROOT / saved['path'])
                section, key = edited.split('.')
                self.assertEqual(config[section][key], value)
                section, key = other.split('.')
                self.assertEqual(config[section][key], .4)
                _validate_supervised_config(config)

    def test_runner_validates_cdr_optimizer_and_noise_contracts(self) -> None:
        base = {'optimizer': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.0, 'weight_decay': 0.0}, 'noise': {'name': 'symmetric', 'rate': 0.4}, 'parameter_update': {'name': 'cdr', 'noise_rate': 0.4, 'l1_decay': 0.001}}
        _validate_supervised_config(base)
        equivalent_rate = {**base, 'noise': {'name': 'symmetric', 'rate': 0.40000000001}}
        _validate_supervised_config(equivalent_rate)
        with self.assertRaisesRegex(ValueError, "optimizer.name='sgd'"):
            _validate_supervised_config({**base, 'optimizer': {'name': 'adamw', 'lr': 0.001}})
        with self.assertRaisesRegex(ValueError, 'non-critical.*L1'):
            invalid_momentum = {**base, 'optimizer': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0}}
            _validate_supervised_config(invalid_momentum)
        self.assertEqual(invalid_momentum['optimizer']['momentum'], 0.9)
        with self.assertRaisesRegex(ValueError, 'weight_decay=0'):
            _validate_supervised_config({**base, 'optimizer': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.0, 'weight_decay': 0.1}})
        independent_rates = {**base, 'noise': {'name': 'symmetric', 'rate': 0.2}}
        _validate_supervised_config(independent_rates)
        self.assertEqual(independent_rates['noise']['rate'], 0.2)
        self.assertEqual(independent_rates['parameter_update']['noise_rate'], 0.4)
        with self.assertRaisesRegex(ValueError, 'parameter_update'):
            _validate_resume_config(base, {**base, 'parameter_update': {'name': 'cdr', 'noise_rate': 0.2, 'l1_decay': 0.001}})
        official = {**base, 'optimizer': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.1}, 'parameter_update': {'name': 'cdr', 'noise_rate': 0.4, 'l1_decay': 0.0, 'compatibility_mode': 'official_code'}}
        _validate_supervised_config(official)
        with self.assertRaisesRegex(ValueError, 'parameter_update'):
            _validate_resume_config(official, {**base, 'optimizer': official['optimizer']})

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cpu_and_cuda_masks_and_updates_match(self) -> None:
        coefficients = torch.tensor([1.0, 3.0, 9.0, -0.25])
        results = []
        for device in (torch.device('cpu'), torch.device('cuda')):
            model = _cdr__VectorModel().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
            objective = (model() * coefficients.to(device)).sum()
            CDRUpdatePolicy(0.5, 0.1).update(ParameterUpdateInput(objective, model, optimizer, RunState()))
            results.append(model.values.detach().cpu())
        self.assertTrue(torch.equal(results[0], results[1]))


import yaml
from lnl_toolbox.training.experiment import build_model, build_optimizer

class _cdr_reproduction_CDRReproductionTest(unittest.TestCase):

    def test_resnet50_builder_has_expected_structure_and_options(self) -> None:
        model = build_model({'name': 'resnet50', 'base_width': 8, 'stem_padding': 0, 'initialization': 'torch_default'}, 10)
        self.assertEqual(tuple((len(layer) for layer in (model.layer1, model.layer2, model.layer3, model.layer4))), (3, 4, 6, 3))
        self.assertEqual(model.stem[0].padding, (0, 0))
        self.assertEqual(tuple(model(torch.randn(1, 3, 32, 32)).shape), (1, 10))
        featured = model.forward_with_features(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(featured.logits.shape), (1, 10))
        self.assertEqual(tuple(featured.features.shape), (1, 256))

    def test_existing_resnet18_defaults_are_not_changed(self) -> None:
        model = build_model({'name': 'resnet18', 'base_width': 8}, 10)
        self.assertEqual(model.stem[0].padding, (1, 1))
        self.assertEqual(tuple((len(layer) for layer in (model.layer1, model.layer2, model.layer3, model.layer4))), (2, 2, 2, 2))


    def test_scratch_author_updates_match_runner_and_resume_with_momentum(self) -> None:
        from copy import deepcopy
        from lnl_toolbox.scratch.context import ScratchContext
        from lnl_toolbox.scratch.blocks.parameter import parameter_criticality_mask, masked_gradient_update
        runner = torch.nn.Linear(2, 2)
        with torch.no_grad():
            runner.weight.copy_(torch.tensor([[2., -1.], [0., 4.]]))
            runner.bias.copy_(torch.tensor([1., -1.]))
        scratch = deepcopy(runner)
        models = [runner, scratch]
        optimizers = [torch.optim.SGD(m.parameters(), lr=.01, momentum=.9, weight_decay=.001) for m in models]
        policy = CDRUpdatePolicy(.4, 0., compatibility_mode='official_code')
        saved = None
        for step in range(3):
            weights = torch.tensor([[1., 3.], [9., -.25]]) if step != 1 else torch.tensor([[.01, .01], [9., 8.]])
            def objective(model):
                return (model.weight * weights).sum() + (model.bias * torch.tensor([2., -3.])).sum()
            policy.update(ParameterUpdateInput(objective(runner), runner, optimizers[0], RunState()))
            optimizers[1].zero_grad(set_to_none=True)
            objective(scratch).backward()
            ctx = ScratchContext(model=scratch)
            parameter_criticality_mask(ctx, noise_rate=.4, compatibility_mode='official_code')
            masked_gradient_update(ctx, scale=.6, l1_decay=0.)
            optimizers[1].step()
            for left, right in zip(runner.parameters(), scratch.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            if step == 0:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'resume.pt'
                    torch.save({'model': runner.state_dict(), 'optimizer': optimizers[0].state_dict(), 'policy': policy.state_dict()}, path)
                    saved = torch.load(path, weights_only=False)
                resumed = torch.nn.Linear(2, 2)
                resumed.load_state_dict(saved['model'])
                resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=.01, momentum=.9, weight_decay=.001)
                resumed_optimizer.load_state_dict(saved['optimizer'])
                resumed_policy = CDRUpdatePolicy(.4, 0., compatibility_mode='official_code')
                resumed_policy.load_state_dict(saved['policy'])
            else:
                resumed_policy.update(ParameterUpdateInput(objective(resumed), resumed, resumed_optimizer, RunState()))
                for left, right in zip(runner.parameters(), resumed.parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)



    def test_scratch_cdr_recipe_applies_explicit_normalization(self) -> None:
        from lnl_toolbox.scratch.context import ScratchContext
        from lnl_toolbox.scratch.data_runtime import ScratchSample, ScratchSplit
        from lnl_toolbox.scratch.blocks.data import configure_preprocessing, configure_views
        root = Path(__file__).resolve().parents[1]
        recipe = yaml.safe_load((root / 'src/lnl_toolbox/scratch/recipes/papers/cdr.yaml').read_text(encoding='utf-8'))
        config = yaml.safe_load((root / 'configs/experiment/cifar10_symmetric_cdr_reproduction.yaml').read_text(encoding='utf-8'))
        params = next(step['params'] for step in recipe['steps'] if step['block'] == 'configure_preprocessing')
        self.assertEqual(params['options']['normalization'], config['data']['normalization'])
        source = ScratchSplit('cifar10', 'train', (ScratchSample(torch.zeros(3, 32, 32), 0, 0, 0),), 10)
        ctx = ScratchContext(train_split=source)
        configure_preprocessing(ctx, **params)
        operations = ctx['preprocessing_transform'].transforms
        self.assertEqual([type(op).__name__ for op in operations], ['RandomCrop', 'RandomHorizontalFlip', 'Lambda', 'Normalize'])
        configure_views(ctx, views=['weak'])
        expected = (-torch.tensor(config['data']['normalization']['mean']) / torch.tensor(config['data']['normalization']['std']))[:, None, None].expand(3, 32, 32)
        torch.testing.assert_close(ctx['preprocessing_transform'](source.samples[0].input), expected)
        torch.testing.assert_close(ctx['view_transforms']['weak'](source.samples[0].input), expected)

    def test_scratch_standard_default_normalization_is_unchanged(self) -> None:
        from lnl_toolbox.scratch.data_runtime import ScratchSample, ScratchSplit, build_transforms
        value = torch.full((3, 32, 32), .5)
        source = ScratchSplit('cifar10', 'train', (ScratchSample(value, 0, 0, 0),), 10)
        transform, _ = build_transforms({'name': 'standard', 'augment': False}, source)
        expected = ((.5 - torch.tensor([.49139968, .48215827, .44653124])) / torch.tensor([.24703233, .24348505, .26158768]))[:, None, None].expand_as(value)
        torch.testing.assert_close(transform(value), expected)

    def test_scratch_rejects_invalid_explicit_normalization(self) -> None:
        from lnl_toolbox.scratch.data_runtime import ScratchSample, ScratchSplit, build_transforms
        source = ScratchSplit('cifar10', 'train', (ScratchSample(torch.zeros(3, 32, 32), 0, 0, 0),), 10)
        for settings in [[], {}, {'mean': [], 'std': []}, {'mean': [0, 0], 'std': [1]}, {'mean': [float('nan')], 'std': [1]}, {'mean': [0], 'std': [0]}, {'mean': [0], 'std': [-1]}, {'mean': [0], 'std': [float('inf')]}]:
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, 'normalization'):
                build_transforms({'name': 'standard', 'normalization': settings}, source)
