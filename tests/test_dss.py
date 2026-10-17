"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_dss.py ---
import unittest

# --- merged from test_dss.py ---
from pathlib import Path

# --- merged from test_dss.py ---
import torch

# --- merged from test_dss.py ---
import yaml

# --- merged from test_dss.py ---
from lnl_toolbox.algorithms.dss import DSSObjective

# --- merged from test_dss.py ---
from lnl_toolbox.algorithms.masked_risk import candidate_masked_cross_entropy

# --- merged from test_dss.py ---
from lnl_toolbox.core import RunState

# --- merged from test_dss.py ---
from lnl_toolbox.core.objectives import ObjectiveResult

# --- merged from test_dss.py ---
from lnl_toolbox.plugins.builtin.catalog import build_builtin_objective_consumer

# --- merged from test_dss.py ---
from lnl_toolbox.selectors.dss import DSSSelectorState

# --- merged from test_dss.py ---
from lnl_toolbox.training.experiment import _resolve_dss_epoch_contract

# --- merged from test_dss.py ---
from lnl_toolbox.training.pipeline import StandardNoisyERMPipeline

# --- merged from test_dss.py ---
class DSSScratchProtocolTest(unittest.TestCase):
    def test_scratch_author_model_matches_formal_weights_outputs_and_gradients(self) -> None:
        from lnl_toolbox.models.cifar_resnet import preact_resnet18
        from lnl_toolbox.scratch.blocks.models import create_model
        from lnl_toolbox.scratch.context import ScratchContext
        torch.manual_seed(17)
        formal = preact_resnet18(num_classes=10, base_width=4, variant='dss_author')
        torch.manual_seed(17)
        ctx = ScratchContext(device=torch.device('cpu'))
        create_model(ctx, model='preact_resnet18', base_width=4, variant='dss_author')
        scratch = ctx['model']
        self.assertEqual(set(formal.state_dict()), set(scratch.state_dict()))
        for name, value in formal.state_dict().items():
            torch.testing.assert_close(value, scratch.state_dict()[name], rtol=0, atol=0)
        inputs = torch.randn(2, 3, 32, 32)
        left, right = formal(inputs), scratch(inputs)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        left.square().mean().backward()
        right.square().mean().backward()
        for name, parameter in formal.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(scratch.named_parameters())[name].grad)
        with self.assertRaisesRegex(ValueError, 'variant'):
            create_model(ctx, model='resnet50', variant='dss_author')

    def test_scratch_classwise_split_matches_formal_indices(self) -> None:
        import numpy as np
        from lnl_toolbox.data.torch_cifar import train_validation_split
        from lnl_toolbox.scratch.data_runtime import ScratchSample, ScratchSplit, split_source
        from lnl_toolbox.scratch.blocks.data import create_dataset_split
        from lnl_toolbox.scratch.context import ScratchContext
        labels = np.repeat(np.arange(3), 10)
        source = ScratchSplit('cifar10', 'train', tuple(ScratchSample(torch.zeros(3, 32, 32), i + 100, int(y), int(y)) for i, y in enumerate(labels)), 3)
        ctx = ScratchContext(train_source=source)
        create_dataset_split(ctx, validation_size=6, split_seed=4, split_strategy='classwise_legacy', split_rng='numpy_legacy')
        expected_train, expected_validation = train_validation_split(labels, 6, 4, strategy='classwise_legacy', rng='numpy_legacy')
        self.assertEqual([s.index for s in ctx['train_split'].samples], (expected_train + 100).tolist())
        self.assertEqual([s.index for s in ctx['validation_split'].samples], (expected_validation + 100).tolist())
        with self.assertRaisesRegex(ValueError, 'numpy_legacy'):
            split_source(source, None, {'split': {'validation_size': 6, 'strategy': 'classwise_legacy'}})
        # Unspecified RNG/strategy retains the previous default_rng split.
        actual, val = split_source(source, None, {'split': {'validation_size': 6, 'seed': 4}})
        expected = np.random.default_rng(4).choice(30, size=6, replace=False).tolist()
        self.assertEqual([s.index - 100 for s in val.samples], expected)

    def test_scratch_evaluation_roles_are_deterministic_without_training_augmentation(self) -> None:
        from lnl_toolbox.scratch.data_runtime import ScratchSample, ScratchSplit, build_role_datasets
        from lnl_toolbox.scratch.blocks.data import configure_preprocessing, configure_views
        from lnl_toolbox.scratch.context import ScratchContext
        image = torch.linspace(0, 1, 3 * 32 * 32).reshape(3, 32, 32)
        source = ScratchSplit('cifar10', 'train', (ScratchSample(image, 0, 0, 0),), 10)
        recipe = yaml.safe_load(Path('src/lnl_toolbox/scratch/recipes/papers/dss.yaml').read_text(encoding='utf-8'))
        params = next(s['params'] for s in recipe['steps'] if s['block'] == 'configure_preprocessing')
        ctx = ScratchContext(train_split=source)
        configure_preprocessing(ctx, **params)
        configure_views(ctx, views=['weak'])
        roles = build_role_datasets(train_split=source, noisy_train=source, validation_split=source, test_split=source,
                                    roles=['train', 'train_eval', 'noisy_validation', 'test'], data_config=ctx['data_plan']['data'],
                                    preprocessing_transform=ctx['preprocessing_transform'], view_transforms=ctx['view_transforms'])
        expected = (image - torch.tensor(params['options']['normalization']['mean'])[:, None, None]) / torch.tensor(params['options']['normalization']['std'])[:, None, None]
        self.assertIn('RandomCrop', [type(t).__name__ for t in roles['train'].transform.transforms])
        for role in ['train_eval', 'noisy_validation', 'test']:
            self.assertEqual([type(t).__name__ for t in roles[role].transform.transforms], ['Lambda', 'Normalize'])
            for _ in range(3):
                torch.testing.assert_close(roles[role][0]['inputs'], expected)
                torch.testing.assert_close(roles[role][0]['views']['weak'], expected)
        legacy = build_role_datasets(train_split=source, noisy_train=source, validation_split=source, test_split=source,
                                     roles=['train', 'test'], data_config={}, preprocessing_transform=ctx['preprocessing_transform'], view_transforms=ctx['view_transforms'])
        self.assertIs(legacy['test'].transform, legacy['train'].transform)

    def test_dss_scratch_recipe_short_training_restores_validation_best_model(self) -> None:
        from copy import deepcopy
        from lnl_toolbox.scratch import ScratchContext, execute_recipe, validate_recipe
        recipe = yaml.safe_load(Path('src/lnl_toolbox/scratch/recipes/papers/dss.yaml').read_text(encoding='utf-8'))
        validate_recipe(recipe)
        recipe = deepcopy(recipe)
        for step in recipe['steps']:
            params = step.get('params', {})
            if step['block'] == 'create_dataset_split': params['validation_size'] = 10
            elif step['block'] == 'configure_loader': params.update(batch_size=5, num_workers=0)
            elif step['block'] == 'create_model': params['base_width'] = 4
            elif step['block'] == 'create_dss_state': params.update(total_epochs=2, warmup_epochs=1)
            elif step['block'] == 'epoch_loop':
                params['epochs'] = 2
                for nested in step['steps']:
                    if nested['block'] == 'refresh_epoch_loader': nested['params']['batch_size'] = 5
        train = [{'inputs': torch.rand(3, 32, 32), 'targets': i % 10, 'indices': i} for i in range(20)]
        test = [{'inputs': torch.rand(3, 32, 32), 'targets': i % 10, 'indices': i} for i in range(10)]
        context = execute_recipe(recipe, context=ScratchContext(dataset_catalog={'cifar10': {'train': train, 'test': test}}))
        self.assertEqual(context['dss_state'].current_epoch, 1)
        self.assertTrue(bool(context['dss_state'].observed[context['dss_state'].labels >= 0].all()))
        self.assertIn(context['best_epoch'], [0, 1])
        for name, value in context['model'].state_dict().items():
            torch.testing.assert_close(value, context['best_model_state'][name], rtol=0, atol=0)
        self.assertTrue(0. <= context['test_accuracy'] <= 1.)


class DSSAuthorModelTest(unittest.TestCase):
    def test_author_variant_structure_initialization_and_builder(self) -> None:
        from torch import nn
        from lnl_toolbox.models.cifar_resnet import DSSAuthorPreActBlock, PreActBlock, preact_resnet18
        from lnl_toolbox.training.experiment import build_model
        config = yaml.safe_load(Path('configs/experiment/dss_cifar10_symmetric05_reproduction.yaml').read_text(encoding='utf-8'))
        self.assertEqual(config['model']['variant'], 'dss_author')
        torch.manual_seed(17)
        expected_stem = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        torch.manual_seed(17)
        model = build_model(config['model'], 10)
        self.assertIsInstance(model.stem, nn.Sequential)
        self.assertIsInstance(model.stem[1], nn.BatchNorm2d)
        self.assertIsInstance(model.stem[2], nn.ReLU)
        self.assertIsInstance(model.layer1[0], DSSAuthorPreActBlock)
        self.assertIsInstance(model.final_bn, nn.Identity)
        torch.testing.assert_close(model.stem[0].weight, expected_stem.weight)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            features = model.layer4(model.layer3(model.layer2(model.layer1(model.stem(x)))))
            expected = model.classifier(nn.functional.avg_pool2d(features, 4).flatten(1))
            torch.testing.assert_close(model(x), expected)
        standard = preact_resnet18(base_width=4)
        self.assertIsInstance(standard.stem, nn.Conv2d)
        self.assertIs(type(standard.layer1[0]), PreActBlock)
        self.assertIsInstance(standard.final_bn, nn.BatchNorm2d)
        with self.assertRaisesRegex(ValueError, 'variant'):
            preact_resnet18(variant='unknown')

    def test_author_identity_shortcut_uses_bn_relu_input(self) -> None:
        from lnl_toolbox.models.cifar_resnet import DSSAuthorPreActBlock, PreActBlock
        block = DSSAuthorPreActBlock(2, 2).eval()
        standard = PreActBlock(2, 2).eval()
        standard.load_state_dict(block.state_dict())
        for layer in (block, standard):
            layer.conv1.weight.data.zero_()
            layer.conv2.weight.data.zero_()
        x = torch.tensor([[[[-2., 1.]], [[3., -4.]]]])
        torch.testing.assert_close(block(x), torch.relu(block.bn1(x)))
        torch.testing.assert_close(standard(x), x)

    def test_empty_selected_loss_is_finite_and_normal_average_is_unchanged(self) -> None:
        from lnl_toolbox.training.experiment import _selected_loss_mean
        self.assertEqual(_selected_loss_mean(0., 0.), 0.)
        self.assertEqual(_selected_loss_mean(6., 3.), 2.)

    def test_author_variant_config_is_valid_and_rejects_unknown_variant(self) -> None:
        from lnl_toolbox.catalog import validate_config
        config = yaml.safe_load(Path('configs/experiment/dss_cifar10_symmetric05_reproduction.yaml').read_text(encoding='utf-8'))
        validate_config(config)
        config['model']['variant'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'model.variant'):
            validate_config(config)

    def test_web_author_variant_has_choices_explanation_and_correct_defaults(self) -> None:
        from web import command_console
        fields = {field['path']: field for field in command_console._config_schema(
            'dss-cifar10-symmetric05-reproduction')['fields']}
        variant = fields['model.variant']
        self.assertEqual(variant['value'], 'dss_author')
        self.assertEqual({choice['value'] for choice in variant['choices']}, {'standard', 'dss_author'})
        self.assertIn('默认初始化', variant['note'])
        self.assertEqual(fields['data.normalization.std']['value'], [.2023, .1994, .201])
        self.assertEqual(fields['loader.batch_size']['value'], 128)
        self.assertEqual(fields['optimizer.lr']['value'], .02)
        self.assertEqual(fields['trainer.epochs']['value'], 150)


class _dss_DSSRiskTest(unittest.TestCase):

    def test_candidate_masked_ce_matches_manual_denominator(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        targets = torch.tensor([0])
        excluded = torch.tensor([[False, True, False]])
        loss = candidate_masked_cross_entropy(logits, targets, excluded)
        expected = torch.logsumexp(logits[:, [0, 2]], dim=1) - logits[:, 0]
        torch.testing.assert_close(loss, expected)
        loss.sum().backward()
        self.assertEqual(float(logits.grad[0, 1]), 0.0)

    def test_invalid_mask_contract_is_rejected(self) -> None:
        logits = torch.zeros(1, 3)
        targets = torch.tensor([1])
        with self.assertRaisesRegex(ValueError, 'target class'):
            candidate_masked_cross_entropy(logits, targets, torch.tensor([[False, True, False]]))
        with self.assertRaisesRegex(ValueError, 'shape'):
            candidate_masked_cross_entropy(logits, targets, torch.zeros(1, 2, dtype=torch.bool))

# --- merged from test_dss.py ---
class _dss_DSSSelectorStateTest(unittest.TestCase):

    def test_selection_is_finalized_at_epoch_boundary(self) -> None:
        selector = DSSSelectorState(2, 2, 3, warmup_epochs=1, mda=False, ccs=False)
        indices = torch.tensor([0, 1])
        targets = torch.tensor([0, 1])
        selector.on_cycle_start(0)
        before, _ = selector.masks(indices, targets)
        selector.observe(indices, targets, torch.tensor([[0.9, 0.1], [0.8, 0.2]]), 0)
        during, _ = selector.masks(indices, targets)
        selector.on_cycle_end(0)
        after, _ = selector.masks(indices, targets)
        self.assertTrue(bool(before.all()))
        self.assertTrue(bool(during.all()))
        torch.testing.assert_close(after, torch.tensor([True, False]))

    def test_mda_uses_batch_ema_and_renormalization(self) -> None:
        selector = DSSSelectorState(2, 2, 2, warmup_epochs=2, prior_decay=0.5, mda=True, ccs=False)
        selector.on_cycle_start(0)
        probabilities = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
        selector.observe(torch.tensor([0, 1]), torch.tensor([0, 1]), probabilities, 0)
        expected_marginal = torch.tensor([0.6, 0.4])
        adjusted = probabilities / (2 * expected_marginal)
        adjusted = adjusted / adjusted.sum(dim=1, keepdim=True)
        torch.testing.assert_close(selector.marginal, expected_marginal)
        torch.testing.assert_close(selector.current_prediction[:2], adjusted)

    def test_ccs_excludes_significantly_rising_non_target_class(self) -> None:
        selector = DSSSelectorState(1, 2, 4, warmup_epochs=1, mda=False, ccs=True)
        for epoch, probability in enumerate((0.1, 0.2, 0.3, 0.4)):
            selector.on_cycle_start(epoch)
            selector.observe(torch.tensor([0]), torch.tensor([0]), torch.tensor([[1.0 - probability, probability]]), epoch)
            selector.on_cycle_end(epoch)
        _, excluded = selector.masks(torch.tensor([0]), torch.tensor([0]))
        torch.testing.assert_close(excluded, torch.tensor([[False, True]]))

    def test_state_roundtrip_restores_history_and_masks(self) -> None:
        selector = DSSSelectorState(2, 2, 3, warmup_epochs=1, mda=False)
        selector.on_cycle_start(0)
        selector.observe(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([[0.9, 0.1], [0.7, 0.3]]), 0)
        selector.on_cycle_end(0)
        restored = DSSSelectorState(2, 2, 3, warmup_epochs=1, mda=False)
        restored.load_state_dict(selector.state_dict())
        self.assertEqual(restored.current_epoch, 0)
        torch.testing.assert_close(restored.selected, selector.selected)
        torch.testing.assert_close(restored.history.values, selector.history.values)

    def test_stable_index_and_epoch_contracts_fail_explicitly(self) -> None:
        selector = DSSSelectorState(3, 2, 2, warmup_epochs=1)
        selector.on_cycle_start(0)
        with self.assertRaisesRegex(ValueError, 'unique'):
            selector.masks(torch.tensor([0, 0]), torch.tensor([0, 0]))
        with self.assertRaisesRegex(ValueError, 'integer'):
            selector.masks(torch.tensor([0.0]), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, 'prior epoch'):
            selector.current_epoch = 1
            selector.observe(torch.tensor([1]), torch.tensor([0]), torch.tensor([[0.6, 0.4]]), 1)

# --- merged from test_dss.py ---
class _dss_DSSObjectiveTest(unittest.TestCase):

    def test_objective_uses_batch_mean_but_reports_selected_mean(self) -> None:
        objective = DSSObjective(2, 2, 2, warmup_epochs=0, mda=False, ccs=False)
        objective.selector.selected[:] = torch.tensor([True, False])
        objective.on_cycle_start(RunState(cycle=0))
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        result = objective.compute(logits=logits, noisy_targets=torch.tensor([0, 1]), sample_indices=torch.tensor([0, 1]), base_loss=None, metadata={'epoch': 0})
        self.assertIsInstance(result, ObjectiveResult)
        per_sample = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]), reduction='none')
        torch.testing.assert_close(result.objective, per_sample[0] / 2)
        torch.testing.assert_close(result.reporting_loss, per_sample[0])
        result.objective.backward()
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.0)

    def test_plugin_builds_dss_and_pipeline_rejects_composition(self) -> None:
        objective_config = {'name': 'dss', 'num_samples': 8, 'num_classes': 2, 'total_epochs': 3, 'warmup_epochs': 1}
        objective = build_builtin_objective_consumer(objective_config)
        self.assertIsInstance(objective, DSSObjective)
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            StandardNoisyERMPipeline.from_config({'objective_consumer': objective_config, 'weight_provider': {'name': 'missing'}})

    def test_trainer_epochs_are_the_only_resolved_horizon(self) -> None:
        config = {'pipeline': {'objective_consumer': {'name': 'dss', 'num_samples': 8, 'num_classes': 2}}}
        _resolve_dss_epoch_contract(config, 4)
        self.assertEqual(config['pipeline']['objective_consumer']['total_epochs'], 4)
        conflicting = {'pipeline': {'objective_consumer': {'name': 'dss', 'total_epochs': 3}}}
        with self.assertRaisesRegex(ValueError, 'trainer.epochs'):
            _resolve_dss_epoch_contract(conflicting, 4)
        for section, value in (('loss', {'name': 'gce'}), ('selector', {'name': 'small_loss', 'keep_rate': 0.5}), ('parameter_update', {'name': 'cdr'})):
            invalid = {section: value, 'pipeline': {'objective_consumer': {'name': 'dss'}}}
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, 'DSS requires'):
                _resolve_dss_epoch_contract(invalid, 4)

    def test_pipeline_component_state_roundtrip_is_strict(self) -> None:
        config = {'objective_consumer': {'name': 'dss', 'num_samples': 2, 'num_classes': 2, 'total_epochs': 2, 'warmup_epochs': 1, 'mda': False, 'ccs': False}}
        pipeline = StandardNoisyERMPipeline.from_config(config)
        objective = pipeline.objective_consumer
        objective.on_cycle_start(RunState(cycle=0))
        objective.compute(logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]), noisy_targets=torch.tensor([0, 1]), sample_indices=torch.tensor([0, 1]), base_loss=None, metadata={'epoch': 0})
        objective.on_cycle_end(RunState(cycle=0))
        states = pipeline.component_state_dict()
        restored = StandardNoisyERMPipeline.from_config(config)
        restored.load_component_states(states)
        torch.testing.assert_close(restored.objective_consumer.selector.history.values, objective.selector.history.values)
        with self.assertRaisesRegex(ValueError, 'missing'):
            restored.load_component_states({})

    def test_configs_do_not_duplicate_total_epochs(self) -> None:
        for filename, expected_epochs in (('dss_cifar10_symmetric05_smoke.yaml', 2), ('dss_cifar10_symmetric05_reproduction.yaml', 150)):
            config = yaml.safe_load((Path('configs/experiment') / filename).read_text(encoding='utf-8'))
            objective = config['pipeline']['objective_consumer']
            self.assertNotIn('total_epochs', objective)
            _resolve_dss_epoch_contract(config, int(config['trainer']['epochs']))
            self.assertEqual(objective.get('total_epochs'), None)
            self.assertEqual(config['pipeline']['objective_consumer']['total_epochs'], expected_epochs)
            self.assertEqual(config['evaluation']['selection_split'], 'validation')
