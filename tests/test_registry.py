"""Merged unit tests; source modules were consolidated without changing assertions."""
from __future__ import annotations

# --- merged from test_registry.py ---
import unittest

# --- merged from test_registry.py ---
from lnl_toolbox.registry import Registry

# --- merged from test_registry.py ---
class _registry_RegistryTest(unittest.TestCase):

    def test_register_and_build(self) -> None:
        registry = Registry('loss')

        @registry.register('demo')
        def build(value: int) -> int:
            return value * 2
        self.assertEqual(registry.build('demo', value=3), 6)
        self.assertEqual(registry.names(), ('demo',))

# --- merged from test_runner_registry.py ---
import unittest

# --- merged from test_runner_registry.py ---
from lnl_toolbox.training.runners import create_runner_registry, resolve_runner

# --- merged from test_runner_registry.py ---
class _runner_registry_PaperRunnerRegistryTest(unittest.TestCase):

    def test_new_paper_runners_are_lazy_and_resolvable(self) -> None:
        registry = create_runner_registry()
        for name in ('mc_ldce', 'cal', 'ca2c', 'l2rw'):
            self.assertIn(name, registry.names())
            self.assertEqual(resolve_runner({'method': name}).name, name)

    def test_explicit_runner_must_match_method(self) -> None:
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            resolve_runner({'method': 'cal', 'execution': {'runner': 'ca2c'}})

# --- merged from test_compatibility.py ---
import unittest

# --- merged from test_compatibility.py ---
from unittest.mock import Mock, patch

# --- merged from test_compatibility.py ---
from lnl_toolbox.catalog import default_paper_config, load_papers, load_recipe_config, recipe_by_id

# --- merged from test_compatibility.py ---
from lnl_toolbox.data.profile import DatasetDeclarationConflict, DatasetDeclarations, DatasetProfile, KnowledgeState, Modality, NoiseKnowledge, NoiseOrigin, NoiseRateInfo, NoiseRateStatus, NoiseStatus, resolve_dataset_capabilities
from lnl_toolbox.data.contracts import DataRequirements, DataRole

# --- merged from test_compatibility.py ---
from lnl_toolbox.training.compatibility import ConfigInputRequirement, CompatibilityReason, CompatibilityResult, CompatibilityStatus, MethodRequirements, resolve_compatibility
from test_t_revision import _t_revision_workflow__algorithm as _algorithm

# --- merged from test_compatibility.py ---
from lnl_toolbox.training.runners import RunnerSpec, create_runner_registry, resolve_runner, runner_names

# --- merged from test_compatibility.py ---
from lnl_toolbox.training.service import ExperimentService

# --- merged from test_compatibility.py ---
def _compatibility__profile(*, modality: Modality=Modality.IMAGE, classes: int=10, clean: KnowledgeState=KnowledgeState.UNKNOWN, clean_validation: KnowledgeState=KnowledgeState.UNKNOWN, noise: NoiseKnowledge | None=None) -> DatasetProfile:
    return DatasetProfile(dataset='fixture', adapter='fixture', source='fixture-root', task='classification', modality=modality, num_classes=classes, input_shape=(32, 32, 3) if modality is Modality.IMAGE else (12,), channels=3 if modality is Modality.IMAGE else None, sample_counts_by_split=(('train', 20), ('test', 10)), available_splits=('train', 'test'), class_names=tuple((str(index) for index in range(classes))), class_distribution_by_split=(('train', tuple((20 // classes for _ in range(classes)))), ('test', tuple((10 // classes for _ in range(classes))))), observed_train_labels=KnowledgeState.AVAILABLE, clean_train_labels=clean, clean_validation_labels=clean_validation, stable_indices=KnowledgeState.AVAILABLE, dataset_fingerprint='d' * 64, split_fingerprints=(('train', 'a' * 64), ('test', 'b' * 64)), noise=noise or NoiseKnowledge())

# --- merged from test_compatibility.py ---
def _compatibility__method(**overrides) -> MethodRequirements:
    values = {
        'method': 'fixture_method',
        'supported_modalities': frozenset({Modality.IMAGE}),
        'data_requirements': DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.TEST}),
            needs_noise_manifest=False,
        ),
    }
    values.update(overrides)
    if values.get('requires_method_noise_prior') and 'method_noise_prior_paths' not in values:
        values['method_noise_prior_paths'] = (('noise', 'rate'),)
    return MethodRequirements(**values)

# --- merged from test_compatibility.py ---
class _compatibility_NoiseKnowledgeTest(unittest.TestCase):

    def test_rate_states_and_round_trip(self) -> None:
        known = NoiseRateInfo(NoiseRateStatus.KNOWN, 0.2, 'user')
        estimated = NoiseRateInfo(NoiseRateStatus.ESTIMATED, 0.3, 'estimator:v1')
        self.assertEqual(NoiseRateInfo.from_dict(known.to_dict()), known)
        self.assertEqual(NoiseRateInfo.from_dict(estimated.to_dict()), estimated)
        self.assertIsNone(NoiseRateInfo().value)
        self.assertIsNone(NoiseRateInfo(NoiseRateStatus.NOT_APPLICABLE).value)

    def test_invalid_rate_state_combinations_fail(self) -> None:
        for constructor in (lambda: NoiseRateInfo(NoiseRateStatus.KNOWN), lambda: NoiseRateInfo(NoiseRateStatus.ESTIMATED, 0.2), lambda: NoiseRateInfo(NoiseRateStatus.UNKNOWN, 0.2), lambda: NoiseRateInfo(NoiseRateStatus.KNOWN, -0.1)):
            with self.assertRaises(ValueError):
                constructor()

    def test_profile_round_trip_and_fingerprint_are_deterministic(self) -> None:
        profile = _compatibility__profile()
        restored = DatasetProfile.from_dict(profile.to_dict())
        self.assertEqual(restored, profile)
        self.assertEqual(restored.fingerprint, profile.fingerprint)
        self.assertEqual(len(profile.fingerprint), 64)

    def test_user_declarations_cannot_override_hard_facts(self) -> None:
        profile = _compatibility__profile(clean=KnowledgeState.AVAILABLE, noise=NoiseKnowledge(NoiseStatus.CLEAN, NoiseOrigin.UNKNOWN, NoiseRateInfo(NoiseRateStatus.NOT_APPLICABLE)))
        with self.assertRaises(DatasetDeclarationConflict):
            resolve_dataset_capabilities(profile, DatasetDeclarations(clean_train_labels=KnowledgeState.UNAVAILABLE))
        with self.assertRaises(DatasetDeclarationConflict):
            resolve_dataset_capabilities(profile, DatasetDeclarations(noise_status=NoiseStatus.NOISY))

# --- merged from test_compatibility.py ---
class _compatibility_CompatibilityResolverTest(unittest.TestCase):

    def test_image_method_and_tabular_mismatch(self) -> None:
        image = resolve_dataset_capabilities(_compatibility__profile(clean=KnowledgeState.AVAILABLE))
        result = resolve_compatibility(image, _compatibility__method())
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE)
        tabular = resolve_dataset_capabilities(_compatibility__profile(modality=Modality.TABULAR, classes=2, clean=KnowledgeState.AVAILABLE))
        result = resolve_compatibility(tabular, _compatibility__method())
        self.assertEqual(result.status, CompatibilityStatus.INCOMPATIBLE)
        self.assertIn('unsupported_modality', {reason.code for reason in result.reasons})

    def test_class_count_constraints(self) -> None:
        dataset = resolve_dataset_capabilities(_compatibility__profile(classes=3))
        exact_binary = _compatibility__method(exact_classes=frozenset({2}), max_classes=2)
        result = resolve_compatibility(dataset, exact_binary)
        self.assertEqual(result.status, CompatibilityStatus.INCOMPATIBLE)
        self.assertIn('wrong_class_count', {reason.code for reason in result.reasons})

    def test_clean_label_unknown_is_not_unavailable(self) -> None:
        requirement = _compatibility__method(requires_clean_train_labels=True)
        unknown = resolve_dataset_capabilities(_compatibility__profile(clean=KnowledgeState.UNKNOWN))
        self.assertEqual(resolve_compatibility(unknown, requirement).status, CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS)
        unavailable = resolve_dataset_capabilities(_compatibility__profile(clean=KnowledgeState.UNKNOWN), DatasetDeclarations(clean_train_labels=KnowledgeState.UNAVAILABLE))
        self.assertEqual(resolve_compatibility(unavailable, requirement).status, CompatibilityStatus.INCOMPATIBLE)
        available = resolve_dataset_capabilities(_compatibility__profile(clean=KnowledgeState.UNKNOWN), DatasetDeclarations(clean_train_labels=KnowledgeState.AVAILABLE))
        self.assertEqual(resolve_compatibility(available, requirement).status, CompatibilityStatus.COMPATIBLE)

    def test_dataset_true_rate_is_distinct_from_method_prior(self) -> None:
        profile = _compatibility__profile(clean=KnowledgeState.AVAILABLE)
        true_rate_only = resolve_dataset_capabilities(profile, DatasetDeclarations(noise_rate=NoiseRateInfo(NoiseRateStatus.KNOWN, 0.2, 'user')))
        method = _compatibility__method(
            requires_method_noise_prior=True,
            method_noise_prior_paths=(('noise', 'rate'),),
        )
        result = resolve_compatibility(true_rate_only, method)
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS)
        self.assertEqual(result.required_user_inputs, ('noise_rate_prior',))
        self.assertEqual(result.required_input_paths, (('noise_rate_prior', (('noise', 'rate'),)),))
        with_prior = resolve_dataset_capabilities(profile, DatasetDeclarations(method_noise_rate_prior=NoiseRateInfo(NoiseRateStatus.KNOWN, 0.2, 'user')))
        self.assertEqual(
            resolve_compatibility(
                with_prior,
                method,
                method_noise_rate_prior=NoiseRateInfo(NoiseRateStatus.KNOWN, 0.2, 'selected recipe'),
            ).status,
            CompatibilityStatus.COMPATIBLE,
        )
        true_rate_method = _compatibility__method(requires_dataset_true_noise_rate=True)
        self.assertEqual(resolve_compatibility(true_rate_only, true_rate_method).status, CompatibilityStatus.COMPATIBLE)
        unknown = resolve_dataset_capabilities(_compatibility__profile())
        self.assertIn('dataset_noise_rate', resolve_compatibility(unknown, true_rate_method).required_user_inputs)

    def test_native_noise_and_pretrained_requirements(self) -> None:
        native = resolve_dataset_capabilities(_compatibility__profile(clean=KnowledgeState.UNKNOWN, noise=NoiseKnowledge(NoiseStatus.NOISY, NoiseOrigin.NATIVE, NoiseRateInfo())))
        result = resolve_compatibility(native, _compatibility__method())
        self.assertEqual(result.status, CompatibilityStatus.INCOMPATIBLE)
        self.assertIn('unsupported_native_noise', {reason.code for reason in result.reasons})
        native_supported = _compatibility__method(supports_native_noisy_labels=True)
        self.assertEqual(resolve_compatibility(native, native_supported).status, CompatibilityStatus.COMPATIBLE)
        dataset = resolve_dataset_capabilities(_compatibility__profile(clean=KnowledgeState.AVAILABLE))
        result = resolve_compatibility(dataset, _compatibility__method(required_pretrained_roles=('upm_main_best',)))
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS)
        self.assertIn('pretrained:upm_main_best', result.required_user_inputs)

    def test_machine_result_uses_stable_reason_codes(self) -> None:
        result = CompatibilityResult(CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS, 'method', 'dataset', reasons=(CompatibilityReason('unknown_noise_rate', 'rate is unknown'),), required_user_inputs=('dataset_noise_rate',))
        value = result.to_dict()
        self.assertEqual(value['status'], 'compatible_with_requirements')
        self.assertEqual(value['reason_codes'], ['unknown_noise_rate'])
        self.assertEqual(value['required_user_inputs'], ['dataset_noise_rate'])

# --- merged from test_compatibility.py ---
class _compatibility_RunnerRequirementsTest(unittest.TestCase):

    def test_twelve_method_requirements_are_attached_to_runner_registry(self) -> None:
        registry = create_runner_registry()
        expected = {'upm': ('upm', Modality.IMAGE, 2), 'coteaching': ('coteaching', Modality.IMAGE, 2), 'lend': ('lend', Modality.IMAGE, 2), 'dividemix': ('dividemix', Modality.IMAGE, 2), 'pcse': ('pcse', Modality.IMAGE, 3), 'importance_reweighting': ('importance_reweighting', Modality.TABULAR, 2), 't_revision': ('t_revision', Modality.IMAGE, 2), 'cnlcu': ('cnlcu', Modality.IMAGE, 2), 'dld': ('dld', Modality.IMAGE, 2), 'ca2c': ('ca2c', Modality.IMAGE, 2), 'instance_transition': ('pdl', Modality.IMAGE, 2), 'volminnet': ('volminnet', Modality.IMAGE, 3)}
        for runner, (method, modality, minimum) in expected.items():
            with self.subTest(runner=runner):
                requirements = registry.get(runner).requirements({})
                self.assertIsNotNone(requirements)
                self.assertEqual(requirements.method, method)
                self.assertIn(modality, requirements.supported_modalities)
                self.assertEqual(requirements.min_classes, minimum)
                self.assertTrue(requirements.requires_noise_manifest)

    def test_rate_prior_and_external_source_metadata(self) -> None:
        registry = create_runner_registry()
        for runner in ('coteaching', 'cnlcu', 'dividemix'):
            requirement = registry.get(runner).requirements({})
            self.assertTrue(requirement.requires_method_noise_prior)
            self.assertFalse(requirement.requires_dataset_true_noise_rate)
        pcse = registry.get('pcse').requirements({'pretraining_stage': {'mode': 'external_checkpoint'}})
        self.assertEqual(pcse.required_pretrained_roles, ('upm_main_best',))
        dld = registry.get('dld').requirements({'dld': {'feature_extractor': {'source': 'external_checkpoint'}}})
        self.assertEqual(dld.required_pretrained_roles, ('upm_main_best',))

    def test_all_formal_papers_publish_config_specific_requirements(self) -> None:
        expected = {'binary-risk': 'binary', 'importance-reweighting': 'importance_reweighting', 'loss-correction': 'loss_correction', 'coteaching': 'coteaching', 'gce': 'gce', 'l2rw': 'l2rw', 'mentornet': 'mentornet', 't-revision': 't_revision', 'apl': 'apl', 'dividemix': 'dividemix', 'dual-t': 'dual_t', 'jocor': 'jocor', 'pdl': 'pdl', 'cal': 'cal', 'cdr': 'cdr', 'upm': 'upm', 'volminnet': 'volminnet', 'cnlcu': 'cnlcu', 'cwd': 'cwd', 'lend': 'lend', 'mc-ldce': 'mc_ldce', 'pcse': 'pcse', 'ca2c': 'ca2c', 'dld': 'dld', 'dss': 'dss', 'fine': 'fine'}
        papers = load_papers()
        self.assertEqual(len(papers), 26)
        for paper in papers:
            with self.subTest(paper=paper.id):
                _, recipe = default_paper_config(paper)
                config = load_recipe_config(recipe)
                requirements = resolve_runner(config).requirements(config)
                self.assertIsNotNone(requirements)
                self.assertEqual(requirements.method, expected[paper.id])
                self.assertIsInstance(requirements.data_requirements, DataRequirements)
                if paper.id == 'cal':
                    self.assertFalse(requirements.requires_noise_manifest)
                else:
                    self.assertTrue(requirements.requires_noise_manifest)

    def test_requirements_separate_algorithm_needs_from_implementation_limits(self) -> None:
        registry = create_runner_registry()
        binary = registry.get('binary').requirements({'risk': {'name': 'natarajan_unbiased'}})
        self.assertEqual(binary.exact_classes, frozenset({2}))
        self.assertEqual(binary.supported_modalities, frozenset({Modality.IMAGE, Modality.TABULAR}))
        self.assertEqual(binary.implemented_variant, 'binary_risk')
        self.assertEqual(binary.required_config_inputs[0].mode, 'all')
        fine = registry.get('fine').requirements({})
        self.assertEqual(fine.exact_classes, frozenset())
        self.assertFalse(fine.requires_clean_validation)
        self.assertEqual(fine.implementation_limits, frozenset({'modality', 'strong_view'}))
        cal = registry.get('cal').requirements({})
        self.assertFalse(cal.requires_clean_train_labels)
        self.assertFalse(cal.requires_aligned_clean_noisy_targets)
        self.assertTrue(cal.supports_native_noisy_labels)
        l2rw = registry.get('l2rw').requirements({'trusted_validation': {'source': 'official_generated'}})
        self.assertTrue(l2rw.requires_clean_train_labels)
        audited = registry.get('l2rw').requirements({'trusted_validation': {'source': 'audited_manifest'}})
        self.assertFalse(audited.requires_clean_train_labels)
        self.assertEqual({item.code for item in audited.required_config_inputs}, {'requires_trusted_validation', 'requires_trusted_manifest'})
        noisy_only = ('coteaching', 'dual_t', 'upm', 'dividemix', 'cnlcu', 't_revision')
        for runner_name in noisy_only:
            with self.subTest(runner=runner_name):
                requirements = registry.get(runner_name).requirements({})
                self.assertIn(
                    'requires_noisy_training_labels',
                    {item.code for item in requirements.required_config_inputs},
                )
        external_pcse = registry.get('pcse').requirements(
            {'pretraining_stage': {'mode': 'external_checkpoint'}}
        )
        self.assertEqual(external_pcse.exact_classes, frozenset({10}))
        train_pcse = registry.get('pcse').requirements(
            {'pretraining_stage': {'mode': 'train'}}
        )
        self.assertEqual(train_pcse.exact_classes, frozenset())

    def test_preserved_implementation_limits_have_the_correct_origin(self) -> None:
        registry = create_runner_registry()
        class_cases = (
            ('cwd', 3),
            ('importance_reweighting', 3),
            ('pcse', 2),
            ('volminnet', 2),
            ('mc_ldce', 2),
        )
        for runner_name, classes in class_cases:
            with self.subTest(runner=runner_name):
                capabilities = resolve_dataset_capabilities(
                    _compatibility__profile(classes=classes)
                )
                result = resolve_compatibility(
                    capabilities, registry.get(runner_name).requirements({})
                )
                reason = next(item for item in result.reasons if item.code == 'wrong_class_count')
                self.assertEqual(reason.origin, 'implemented_variant_limit')

        tabular = resolve_dataset_capabilities(
            _compatibility__profile(modality=Modality.TABULAR)
        )
        modality_requirements = {
            'fine': registry.get('fine').requirements({}),
            'dld': registry.get('dld').requirements({}),
            'mentornet': registry.get('supervised').requirements(
                {'pipeline': {'weight_provider': {'name': 'mentornet'}}}
            ),
        }
        for runner_name, requirements in modality_requirements.items():
            with self.subTest(runner=runner_name):
                result = resolve_compatibility(tabular, requirements)
                reason = next(item for item in result.reasons if item.code == 'unsupported_modality')
                self.assertEqual(reason.origin, 'implemented_variant_limit')

        binary = resolve_compatibility(
            resolve_dataset_capabilities(_compatibility__profile(classes=3)),
            registry.get('binary').requirements({}),
        )
        reason = next(item for item in binary.reasons if item.code == 'wrong_class_count')
        self.assertEqual(reason.origin, 'algorithm_requirement')

    def test_zero_validation_dedicated_requirements_do_not_publish_validation_roles(self) -> None:
        registry = create_runner_registry()
        cases = (
            ('mc_ldce', {'data': {'validation_size': 0}}),
            ('ca2c', {'data': {'validation_size': 0}}),
        )
        for runner_name, config in cases:
            with self.subTest(runner=runner_name):
                requirements = registry.get(runner_name).requirements(config)
                self.assertNotIn(DataRole.CLEAN_VALIDATION, requirements.data_requirements.roles)
                self.assertNotIn(DataRole.NOISY_VALIDATION, requirements.data_requirements.roles)

    def test_l2rw_uses_one_modality_contract_and_keeps_trusted_supervision(self) -> None:
        registry = create_runner_registry()
        smoke = load_recipe_config(recipe_by_id('l2rw-cifar10-smoke'))
        smoke_requirements = registry.get('l2rw').requirements(smoke)
        self.assertEqual(
            smoke_requirements.supported_modalities,
            frozenset({Modality.IMAGE, Modality.TABULAR}),
        )
        service = ExperimentService()
        service.preflight(smoke, check_data=True)
        self.assertEqual(
            service.last_compatibility.status,
            CompatibilityStatus.COMPATIBLE,
        )

        reproduction = load_recipe_config(recipe_by_id('l2rw-cifar10-reproduction'))
        reproduction_requirements = registry.get('l2rw').requirements(reproduction)
        self.assertEqual(
            reproduction_requirements.supported_modalities,
            frozenset({Modality.IMAGE, Modality.TABULAR}),
        )

        unknown_tabular = {
            'data': {'name': 'another_tabular_dataset'},
            'model': {'name': 'feature_mlp'},
            'trusted_validation': {'source': 'synthetic_fixture'},
        }
        self.assertEqual(
            registry.get('l2rw').requirements(unknown_tabular).supported_modalities,
            frozenset({Modality.IMAGE, Modality.TABULAR}),
        )

    def test_cal_mc_ldce_and_ca2c_preserve_their_synthetic_smokes(self) -> None:
        registry = create_runner_registry()
        service = ExperimentService()
        cases = (
            ('cal-cifar10-smoke', 'cal'),
            ('mc-ldce-cifar10-smoke', 'mc_ldce'),
            ('ca2c-cifar10-smoke', 'ca2c'),
        )
        for recipe_id, runner_name in cases:
            with self.subTest(recipe=recipe_id):
                config = load_recipe_config(recipe_by_id(recipe_id))
                requirements = registry.get(runner_name).requirements(config)
                expected = frozenset({Modality.IMAGE, Modality.TABULAR})
                self.assertEqual(
                    requirements.supported_modalities,
                    expected,
                )
                service.preflight(config, check_data=True)
                self.assertEqual(
                    service.last_compatibility.status,
                    CompatibilityStatus.COMPATIBLE,
                )

    def test_cal_mc_ldce_and_ca2c_are_dataset_neutral(self) -> None:
        registry = create_runner_registry()
        formal_cases = (
            ('cal-cifar10-reproduction', 'cal'),
            ('mc-ldce-cifar10-reproduction', 'mc_ldce'),
            ('ca2c-cifar10-reproduction', 'ca2c'),
        )
        for recipe_id, runner_name in formal_cases:
            with self.subTest(recipe=recipe_id):
                config = load_recipe_config(recipe_by_id(recipe_id))
                requirements = registry.get(runner_name).requirements(config)
                expected = frozenset({Modality.IMAGE, Modality.TABULAR})
                self.assertEqual(requirements.supported_modalities, expected)

        cal = registry.get('cal').requirements(
            load_recipe_config(recipe_by_id('cal-cifar10-reproduction'))
        )
        self.assertEqual(
            {item.code for item in cal.required_config_inputs},
            set(),
        )
        mc_ldce = registry.get('mc_ldce').requirements({
            'data': {'name': 'synthetic_multiclass'},
            'model': {'name': 'feature_mlp'},
            'transition': {'estimator': 'known_smoke'},
        })
        self.assertEqual(
            {item.code for item in mc_ldce.required_config_inputs},
            {'requires_transition_source'},
        )

        tabular = resolve_dataset_capabilities(
            _compatibility__profile(
                modality=Modality.TABULAR,
                classes=3,
                clean=KnowledgeState.AVAILABLE,
                clean_validation=KnowledgeState.AVAILABLE,
            )
        )
        unknown_configs = {
            'cal': {
                'data': {'name': 'unknown_tabular'},
                'model': {'name': 'feature_mlp'},
                'noise': {'name': 'symmetric'},
                'cal': {'confidence_weight': 1.0},
            },
            'mc_ldce': {
                'data': {'name': 'unknown_tabular'},
                'model': {'name': 'feature_mlp'},
                'transition': {
                    'estimator': 'known_smoke',
                    'matrix': [[1.0, 0.0, 0.0]] * 3,
                },
            },
            'ca2c': {
                'data': {'name': 'unknown_tabular'},
                'model': {'name': 'feature_mlp'},
                'ca2c': {
                    'warmup_epochs': 1,
                    'candidate_k': 1,
                    'hard_weight': 0.5,
                },
            },
        }
        for runner_name, config in unknown_configs.items():
            with self.subTest(runner=runner_name):
                requirements = registry.get(runner_name).requirements(config)
                expected = CompatibilityStatus.COMPATIBLE
                self.assertEqual(
                    resolve_compatibility(tabular, requirements).status,
                    expected,
                )

    def test_shared_runner_detection_is_component_driven(self) -> None:
        registry = create_runner_registry()
        supervised = registry.get('supervised')
        plain_ce = supervised.requirements({})
        self.assertEqual(plain_ce.method, 'ce')
        self.assertEqual(plain_ce.supported_modalities, frozenset({Modality.IMAGE, Modality.TABULAR}))
        self.assertFalse(plain_ce.requires_noise_manifest)
        self.assertTrue(plain_ce.supports_native_noisy_labels)
        cases = (({'loss': {'name': 'gce'}}, 'gce'), ({'loss': {'name': 'apl'}}, 'apl'), ({'parameter_update': {'name': 'cdr'}}, 'cdr'), ({'pipeline': {'objective_consumer': {'name': 'dss'}}}, 'dss'), ({'pipeline': {'weight_provider': {'name': 'mentornet'}}}, 'mentornet'), ({'pipeline': {'risk_corrector': {'name': 'forward'}}}, 'loss_correction'))
        for config, expected in cases:
            with self.subTest(expected=expected):
                requirements = supervised.requirements(config)
                self.assertEqual(requirements.method, expected)
                if expected in {'gce', 'apl'}:
                    self.assertEqual(
                        requirements.supported_modalities,
                        frozenset({Modality.IMAGE, Modality.TABULAR}),
                    )
                    self.assertFalse(requirements.requires_noise_manifest)
                    self.assertTrue(requirements.supports_native_noisy_labels)
                if expected in {'cdr', 'dss'}:
                    self.assertEqual(
                        requirements.supported_modalities,
                        frozenset({Modality.IMAGE, Modality.TABULAR}),
                    )
        volminnet = registry.get('volminnet').requirements({})
        self.assertEqual(
            volminnet.supported_modalities,
            frozenset({Modality.IMAGE, Modality.TABULAR}),
        )
        self.assertEqual(volminnet.min_classes, 3)
        self.assertEqual(volminnet.exact_classes, frozenset())
        multi_model = registry.get('multi_model')
        self.assertIsNone(multi_model.requirements({}))
        self.assertEqual(multi_model.requirements({'algorithm': {'name': 'jocor'}}).method, 'jocor')

    def test_remaining_papers_reject_observed_only_native_noise(self) -> None:
        native_noise = NoiseKnowledge(status=NoiseStatus.NOISY, origin=NoiseOrigin.NATIVE, rate=NoiseRateInfo())
        papers = {paper.id: paper for paper in load_papers()}
        for paper_id in {'binary-risk', 'loss-correction', 'l2rw', 'mentornet', 'dual-t', 'jocor', 'cdr', 'cwd', 'mc-ldce', 'dss', 'fine'}:
            with self.subTest(paper=paper_id):
                _, recipe = default_paper_config(papers[paper_id])
                config = load_recipe_config(recipe)
                requirements = resolve_runner(config).requirements(config)
                modality = next(iter(requirements.supported_modalities))
                classes = next(iter(requirements.exact_classes)) if requirements.exact_classes else max(2, requirements.min_classes or 2)
                capabilities = resolve_dataset_capabilities(_compatibility__profile(modality=modality, classes=classes, clean=KnowledgeState.UNAVAILABLE, clean_validation=KnowledgeState.UNAVAILABLE, noise=native_noise))
                result = resolve_compatibility(capabilities, requirements)
                self.assertEqual(result.status, CompatibilityStatus.INCOMPATIBLE)

    def test_cal_accepts_observed_only_native_noise(self) -> None:
        native_noise = NoiseKnowledge(status=NoiseStatus.NOISY, origin=NoiseOrigin.NATIVE, rate=NoiseRateInfo())
        capabilities = resolve_dataset_capabilities(_compatibility__profile(modality=Modality.TABULAR, classes=3, clean=KnowledgeState.UNAVAILABLE, noise=native_noise))
        requirements = create_runner_registry().get('cal').requirements({})
        self.assertEqual(resolve_compatibility(capabilities, requirements).status, CompatibilityStatus.COMPATIBLE)

    def test_runner_invoke_passes_the_canonical_data_requirement_object(self) -> None:
        requirement = _compatibility__method()
        provider = Mock(return_value=requirement)
        runner = Mock(return_value=Path('run'))
        spec = RunnerSpec('fixture', 'unused', 'unused', requirements_provider=provider)
        with patch.object(RunnerSpec, 'load', return_value=runner):
            spec.invoke({'seed': 1})
        self.assertIs(runner.call_args.kwargs['requirements'], requirement.data_requirements)

    def test_data_service_does_not_import_runner_or_method_registry(self) -> None:
        import inspect
        import lnl_toolbox.training.data_service as data_service_module
        source = inspect.getsource(data_service_module)
        self.assertNotIn('training.runners', source)
        self.assertNotIn('resolve_runner', source)

    def test_loss_only_methods_accept_native_noise_without_clean_labels(self) -> None:
        native_noise = NoiseKnowledge(
            status=NoiseStatus.NOISY,
            origin=NoiseOrigin.NATIVE,
            rate=NoiseRateInfo(),
        )
        capabilities = resolve_dataset_capabilities(
            _compatibility__profile(
                modality=Modality.TABULAR,
                classes=4,
                clean=KnowledgeState.UNAVAILABLE,
                clean_validation=KnowledgeState.UNAVAILABLE,
                noise=native_noise,
            )
        )
        supervised = create_runner_registry().get('supervised')
        for loss in ('gce', 'apl'):
            with self.subTest(loss=loss):
                requirements = supervised.requirements(
                    {'loss': {'name': loss}, 'noise': {'validation_targets': 'noisy'}}
                )
                self.assertEqual(
                    resolve_compatibility(capabilities, requirements).status,
                    CompatibilityStatus.COMPATIBLE,
                )

# --- merged from test_compatibility.py ---
class _compatibility_ExperimentCompatibilityServiceTest(unittest.TestCase):

    def test_mentor_artifact_is_reported_as_an_implementation_limit(self) -> None:
        capabilities = resolve_dataset_capabilities(
            _compatibility__profile(clean=KnowledgeState.AVAILABLE)
        )
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        service = ExperimentService(data_service=data_service)
        config = {
            'execution': {'runner': 'supervised'},
            'data': {'name': 'fixture'},
            'pipeline': {'weight_provider': {'name': 'mentornet'}},
        }
        result = service.resolve_method_compatibility(config, config)
        reason = next(
            item for item in result.reasons
            if item.code == 'requires_mentor_artifact'
        )
        self.assertEqual(reason.origin, 'implemented_variant_limit')
        serialized = result.to_dict()
        serialized_reason = next(
            item for item in serialized['reasons']
            if item['code'] == 'requires_mentor_artifact'
        )
        self.assertEqual(serialized_reason['origin'], 'implemented_variant_limit')

    def test_l2rw_trusted_supervision_remains_an_algorithm_requirement(self) -> None:
        capabilities = resolve_dataset_capabilities(
            _compatibility__profile(clean=KnowledgeState.AVAILABLE)
        )
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        service = ExperimentService(data_service=data_service)
        config = {
            'execution': {'runner': 'l2rw'},
            'data': {'name': 'fixture'},
        }
        result = service.resolve_method_compatibility(config, config)
        reason = next(
            item for item in result.reasons
            if item.code == 'requires_trusted_validation'
        )
        self.assertEqual(reason.origin, 'algorithm_requirement')

    def test_service_and_direct_resolver_agree(self) -> None:
        capabilities = resolve_dataset_capabilities(_compatibility__profile(modality=Modality.TABULAR, classes=2))
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        service = ExperimentService(data_service=data_service)
        config = {
            'method': 'importance_reweighting',
            'execution': {'runner': 'importance_reweighting'},
            'data': {'name': 'fixture'},
            'noise': {'rho_positive': 0.2, 'rho_negative': 0.1},
        }
        result = service.resolve_method_compatibility(config, config)
        direct = resolve_compatibility(
            capabilities,
            create_runner_registry().get('importance_reweighting').requirements(config),
        )
        self.assertEqual(result, direct)
        data_service.capabilities.assert_called_once_with(config, seed=0, persist=False)

    def test_config_prior_is_not_treated_as_dataset_true_rate(self) -> None:
        capabilities = resolve_dataset_capabilities(_compatibility__profile(clean=KnowledgeState.AVAILABLE))
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        service = ExperimentService(data_service=data_service)
        config = {'method': 'coteaching', 'execution': {'runner': 'coteaching'}, 'data': {'name': 'fixture'}, 'noise': {'name': 'symmetric', 'rate': 0.2}, 'coteaching': {'noise_rate': 0.2}}
        result = service.resolve_method_compatibility(config, config)
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE)
        self.assertEqual(capabilities.noise_rate.status, NoiseRateStatus.UNKNOWN)

    def test_incompatible_preflight_stops_before_runner_invocation(self) -> None:
        capabilities = resolve_dataset_capabilities(_compatibility__profile(modality=Modality.TABULAR, classes=2))
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        runner = Mock(name='runner')
        runner.name = 'upm'
        runner.requirements.return_value = _compatibility__method()
        service = ExperimentService(data_service=data_service)
        config = {'method': 'upm', 'execution': {'runner': 'upm'}, 'data': {'name': 'fixture'}}
        with patch('lnl_toolbox.catalog.validate_config', return_value=runner):
            with self.assertRaisesRegex(ValueError, 'missing_noise_manifest'):
                service.preflight(config, check_data=True)
        runner.invoke.assert_not_called()

    def test_preflight_requires_method_prior_but_not_dataset_true_rate(self) -> None:
        capabilities = resolve_dataset_capabilities(_compatibility__profile(clean=KnowledgeState.AVAILABLE))
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        runner = create_runner_registry().get('coteaching')
        service = ExperimentService(data_service=data_service)
        base = {'schema_version': 1, 'kind': 'experiment', 'method': 'coteaching', 'execution': {'runner': 'coteaching'}, 'data': {'name': 'fixture'}, 'noise': {'name': 'symmetric', 'rate': 0.2}}
        with patch('lnl_toolbox.catalog.validate_config', return_value=runner):
            with self.assertRaisesRegex(ValueError, 'requires_noise_rate_prior'):
                service.preflight(base, check_data=True)
            configured = {**base, 'coteaching': {'noise_rate': 0.2}}
            self.assertIs(service.preflight(configured, check_data=True), runner)
        self.assertEqual(service.last_compatibility.status, CompatibilityStatus.COMPATIBLE)
        self.assertEqual(capabilities.noise_rate.status, NoiseRateStatus.UNKNOWN)

    def test_unknown_dataset_rate_passes_when_method_does_not_require_it(self) -> None:
        capabilities = resolve_dataset_capabilities(_compatibility__profile(clean=KnowledgeState.AVAILABLE))
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        service = ExperimentService(data_service=data_service)
        config = {
            'method': 'upm',
            'execution': {'runner': 'upm'},
            'data': {'name': 'fixture'},
            'noise': {'name': 'symmetric', 'rate': 0.2},
        }
        result = service.resolve_method_compatibility(config, config)
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE)

    def test_method_discovery_uses_every_central_runner(self) -> None:
        capabilities = resolve_dataset_capabilities(_compatibility__profile(clean=KnowledgeState.AVAILABLE))
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        results = ExperimentService(data_service=data_service).list_compatible_methods('fixture')
        self.assertEqual(len(results), len(runner_names()))
        by_method = {result.method: result for result in results}
        self.assertIn('upm', by_method)
        self.assertNotIn('requirements_unavailable', {reason.code for reason in by_method['binary'].reasons})

    def test_required_config_inputs_support_all_and_any_modes(self) -> None:
        capabilities = resolve_dataset_capabilities(_compatibility__profile(modality=Modality.TABULAR, classes=2, clean=KnowledgeState.AVAILABLE))
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        service = ExperimentService(data_service=data_service)
        missing = service.resolve_method_compatibility({'data': {'name': 'fixture'}, 'execution': {'runner': 'binary'}, 'risk': {'name': 'natarajan_unbiased'}}, {'data': {'name': 'fixture'}, 'execution': {'runner': 'binary'}, 'risk': {'name': 'natarajan_unbiased'}})
        self.assertIn('requires_binary_noise_prior', {reason.code for reason in missing.reasons})
        configured = {'data': {'name': 'fixture'}, 'execution': {'runner': 'binary'}, 'risk': {'name': 'natarajan_unbiased', 'rho_positive': 0.2, 'rho_negative': 0.3}}
        result = service.resolve_method_compatibility(configured, configured)
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE)
        requirement = ConfigInputRequirement(code='source', paths=(('a',), ('b',)), mode='any')
        self.assertEqual(requirement.mode, 'any')

# --- merged from test_cnlcu_readiness.py ---
from pathlib import Path

# --- merged from test_cnlcu_readiness.py ---
import unittest

# --- merged from test_cnlcu_readiness.py ---
import yaml

# --- merged from test_cnlcu_readiness.py ---
from lnl_toolbox.algorithms.cnlcu import CNLCUConfig

# --- merged from test_cnlcu_readiness.py ---
_cnlcu_readiness_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_cnlcu_readiness.py ---
_cnlcu_readiness_CONFIG = _cnlcu_readiness_ROOT / 'configs/reproduction/cifar10_cnlcu_soft_sym20_short.yaml'

# --- merged from test_cnlcu_readiness.py ---
_cnlcu_readiness_FORMAL_CONFIG = _cnlcu_readiness_ROOT / 'configs/experiment/cnlcu_cifar10_reproduction.yaml'

# --- merged from test_cnlcu_readiness.py ---
class _cnlcu_readiness_CNLCUReadinessTest(unittest.TestCase):

    def test_formal_config_matches_cifar10_paper_protocol(self) -> None:
        config = yaml.safe_load(_cnlcu_readiness_FORMAL_CONFIG.read_text(encoding='utf-8'))
        method = CNLCUConfig.from_mapping(config)
        self.assertEqual(config['configuration_fidelity'], 'paper_protocol')
        self.assertEqual(config['model'], {'name': 'cnlcu_cnn9'})
        self.assertEqual(config['optimizer']['lr'], 0.001)
        self.assertEqual(config['scheduler'], {'name': 'linear_after', 'start_epoch': 80, 'end_epoch': 200})
        self.assertEqual(config['loader']['batch_size'], 128)
        self.assertEqual(config['trainer']['epochs'], 200)
        self.assertEqual(method.variant, 'soft')

    def test_short_config_is_full_data_cnlcu_soft_sym20(self) -> None:
        config = yaml.safe_load(_cnlcu_readiness_CONFIG.read_text(encoding='utf-8'))
        method = CNLCUConfig.from_mapping(config)
        self.assertEqual(config['method'], 'cnlcu')
        self.assertEqual(config['execution']['runner'], 'cnlcu')
        self.assertEqual(config['configuration_fidelity'], 'engineering')
        self.assertEqual(config['data']['name'], 'cifar10')
        for key in ('max_train_samples', 'max_validation_samples', 'max_test_samples'):
            self.assertNotIn(key, config['data'])
        self.assertEqual(config['noise']['name'], 'symmetric')
        self.assertEqual(config['noise']['rate'], 0.2)
        self.assertEqual(config['noise']['validation_targets'], 'noisy')
        self.assertEqual(config['model']['name'], 'cifar_cnn8')
        self.assertEqual(config['trainer']['epochs'], 15)
        self.assertEqual(method.variant, 'soft')
        self.assertEqual(method.window_size, 5)
        self.assertEqual(method.rate_at(0), 1.0)
        self.assertEqual(method.rate_at(10), 0.8)

# --- merged from test_dld_readiness.py ---
import os

# --- merged from test_dld_readiness.py ---
from pathlib import Path

# --- merged from test_dld_readiness.py ---
import tempfile

# --- merged from test_dld_readiness.py ---
import unittest

# --- merged from test_dld_readiness.py ---
from unittest import mock

# --- merged from test_dld_readiness.py ---
import numpy as np

# --- merged from test_dld_readiness.py ---
import torch

# --- merged from test_dld_readiness.py ---
import yaml

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.algorithms.dld import DLDConfig

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.noise.generators import generate_symmetric

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.noise.manifest import fingerprint_labels

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.training.checkpoint import atomic_save

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.training.dld_pretrained import load_upm_main_best_feature_source

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.training.experiment import build_model

# --- merged from test_dld_readiness.py ---
from lnl_toolbox.training.noisy_labels import file_sha256

# --- merged from test_dld_readiness.py ---
_dld_readiness_ROOT = Path(__file__).resolve().parents[1]

# --- merged from test_dld_readiness.py ---
def _dld_readiness__source(directory: Path) -> tuple[dict, torch.nn.Module]:
    model_config = {'name': 'resnet18', 'base_width': 16}
    model = build_model(model_config, 10)
    labels = np.arange(20, dtype=np.int64) % 10
    manifest = generate_symmetric(labels, 10, 0.4, 1, 'cifar10', sampling='per_class')
    manifest.global_indices = np.arange(labels.size, dtype=np.int64)
    manifest.dataset_fingerprint = fingerprint_labels(labels)
    manifest_path = directory / 'noise_manifest.npz'
    manifest.save(manifest_path)
    noise = {'dataset': 'cifar10', 'num_classes': 10, 'mapping_hash': manifest.mapping_hash, 'dataset_fingerprint': manifest.dataset_fingerprint, 'manifest_sha256': file_sha256(manifest_path)}
    payload = {'method': 'upm', 'checkpoint_role': 'main_best', 'config': {'upm': {'main': {'model': model_config}}}, 'noise': noise, 'upm_state': {'main_completed_epochs': 3, 'main_global_step': 9, 'main_best_epoch': 1, 'main_best_validation_accuracy': 0.5}, 'best_main_model_state': model.state_dict()}
    checkpoint_path = directory / 'best.pt'
    atomic_save(payload, checkpoint_path)
    return ({'adapter': 'upm_main_best', 'run_directory_env': 'DLD_TEST_SOURCE', 'checkpoint_sha256': file_sha256(checkpoint_path), 'manifest_sha256': file_sha256(manifest_path), 'mapping_hash': manifest.mapping_hash, 'dataset_fingerprint': manifest.dataset_fingerprint, 'model': model_config}, model)

# --- merged from test_dld_readiness.py ---
class _dld_readiness_DLDReadinessTest(unittest.TestCase):

    def test_formal_config_uses_current_contract_and_full_budget(self) -> None:
        path = _dld_readiness_ROOT / 'configs/experiment/dld_cifar10_reproduction.yaml'
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        parsed = DLDConfig.from_mapping(config)
        self.assertEqual(config['configuration_fidelity'], 'paper_oriented')
        self.assertEqual(config['loader']['batch_size'], 200)
        self.assertEqual(parsed.precorrection['k_neighbors'], 50)
        self.assertEqual(parsed.diffusion['timesteps'], 1000)
        self.assertEqual(parsed.epochs, 200)

    def test_real_short_config_is_full_data_external_sym20(self) -> None:
        path = _dld_readiness_ROOT / 'configs' / 'reproduction' / 'cifar10_dld_sym20_short.yaml'
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        parsed = DLDConfig.from_mapping(config)
        self.assertEqual(config['method'], 'dld')
        self.assertEqual(config['data']['name'], 'cifar10')
        for name in ('max_train_samples', 'max_validation_samples', 'max_test_samples'):
            self.assertNotIn(name, config['data'])
        self.assertEqual(config['noise']['rate'], 0.2)
        self.assertEqual(parsed.fidelity['name'], 'paper_oriented_v2_cosine_similarity')
        self.assertEqual(parsed.fidelity['neighbor_metric'], 'cosine_similarity')
        self.assertEqual(parsed.fidelity['neighbor_weighting'], 'inverse_neighbor_value')
        self.assertEqual(parsed.feature_extractor['source'], 'external_checkpoint')
        self.assertEqual(parsed.precorrection['query_chunk_size'], 64)
        self.assertEqual(parsed.epochs, 15)
        legacy = yaml.safe_load(path.read_text(encoding='utf-8'))
        legacy['dld']['fidelity']['name'] = 'paper_oriented_v1'
        legacy['dld']['fidelity']['neighbor_metric'] = 'cosine_distance'
        legacy['dld']['fidelity'].pop('neighbor_weighting')
        with self.assertRaisesRegex(ValueError, 'fidelity'):
            DLDConfig.from_mapping(legacy)

    def test_external_source_is_strict_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _dld_readiness__source(root)
            with mock.patch.dict(os.environ, {'DLD_TEST_SOURCE': directory}):
                source = load_upm_main_best_feature_source(config, build_model(config['model'], 10), num_classes=10)
                self.assertEqual(source.provenance['adapter'], 'upm_main_best')
                source.assert_unchanged()
                original = (root / 'noise_manifest.npz').read_bytes()
                (root / 'noise_manifest.npz').write_bytes(original + b'changed')
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    source.assert_unchanged()

    def test_external_source_rejects_identity_and_role_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = _dld_readiness__source(root)
            with mock.patch.dict(os.environ, {'DLD_TEST_SOURCE': directory}):
                invalid = {**config, 'checkpoint_sha256': 'f' * 64}
                with self.assertRaisesRegex(ValueError, 'checkpoint SHA-256'):
                    load_upm_main_best_feature_source(invalid, build_model(config['model'], 10), num_classes=10)
                payload = torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
                payload['checkpoint_role'] = 'last'
                atomic_save(payload, root / 'best.pt')
                role_config = {**config, 'checkpoint_sha256': file_sha256(root / 'best.pt')}
                with self.assertRaisesRegex(ValueError, 'main_best'):
                    load_upm_main_best_feature_source(role_config, build_model(config['model'], 10), num_classes=10)

# --- merged from test_t_revision_readiness.py ---
import json

# --- merged from test_t_revision_readiness.py ---
from pathlib import Path

# --- merged from test_t_revision_readiness.py ---
import tempfile

# --- merged from test_t_revision_readiness.py ---
import unittest

# --- merged from test_t_revision_readiness.py ---
import numpy as np

# --- merged from test_t_revision_readiness.py ---
import torch

# --- merged from test_t_revision_readiness.py ---
from torch.nn import functional as F

# --- merged from test_t_revision_readiness.py ---
from lnl_toolbox.algorithms.t_revision import t_revision_reweight_objective

# --- merged from test_t_revision_readiness.py ---
from lnl_toolbox.catalog import load_recipe_config, paper_by_id, recipe_by_id, validate_config

# --- merged from test_t_revision_readiness.py ---
# --- merged from test_t_revision_readiness.py ---
class _t_revision_readiness_TRevisionReadinessTest(unittest.TestCase):

    def test_short_recipe_is_full_data_sym20_resnet18(self) -> None:
        recipe = recipe_by_id('cifar10-t-revision-sym20-short')
        self.assertEqual(recipe.profile, 'reproduction')
        self.assertEqual(recipe.runner, 't_revision')
        self.assertEqual(recipe.configuration_fidelity, 'engineering')
        config = load_recipe_config(recipe)
        self.assertEqual(validate_config(config).name, 't_revision')
        self.assertEqual(config['data']['name'], 'cifar10')
        for limit in ('max_train_samples', 'max_validation_samples', 'max_test_samples'):
            self.assertNotIn(limit, config['data'])
        self.assertEqual(config['data']['validation_size'], 5000)
        self.assertIs(config['data']['augment'], True)
        self.assertEqual(config['noise']['rate'], 0.2)
        self.assertEqual(config['noise']['sampling'], 'transition')
        self.assertEqual(config['noise']['validation_targets'], 'noisy')
        self.assertEqual(config['loader']['batch_size'], 128)
        self.assertEqual(config['t_revision']['stage1']['model']['name'], 'resnet18')
        self.assertEqual(config['t_revision']['stage1']['epochs'], 15)
        self.assertEqual(config['t_revision']['classifier_initialization']['epochs'], 15)
        self.assertEqual(config['t_revision']['revision']['epochs'], 20)
        paper = paper_by_id('t-revision')
        self.assertIn(recipe.id, {item.recipe_id for item in paper.configs})

    def test_objective_exposes_detached_diagnostics_without_changing_loss(self) -> None:
        logits = torch.tensor([[1.2, -0.2], [0.1, 0.7]], requires_grad=True)
        targets = torch.tensor([0, 1])
        transition = torch.tensor([[0.8, 0.2], [0.1, 0.9]], requires_grad=True)
        result = t_revision_reweight_objective(logits, targets, transition, denominator_floor=1e-12)
        clean = torch.softmax(logits, dim=1)
        denominator = (clean @ transition).gather(1, targets[:, None]).squeeze(1)
        expected_weights = clean.gather(1, targets[:, None]).squeeze(1) / denominator
        torch.testing.assert_close(result.sample_weights, expected_weights.detach())
        torch.testing.assert_close(result.sample_denominators, denominator.detach())
        self.assertFalse(result.sample_weights.requires_grad)
        self.assertFalse(result.sample_denominators.requires_grad)
        result.objective.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(transition.grad)

    def test_stage2a_detaches_ratio_without_changing_equation_three(self) -> None:
        logits = torch.tensor([[1.2, -0.2], [0.1, 0.7]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([0, 1])
        transition = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float64, requires_grad=True)
        result = t_revision_reweight_objective(logits, targets, transition, denominator_floor=1e-12, detach_ratio=True)
        clean = torch.softmax(logits, dim=1)
        numerator = clean.gather(1, targets[:, None]).squeeze(1)
        denominator = (clean @ transition).gather(1, targets[:, None]).squeeze(1)
        expected_ratio = numerator / denominator
        expected = (expected_ratio.detach() * F.cross_entropy(logits, targets, reduction='none')).mean()
        torch.testing.assert_close(result.sample_weights, expected_ratio.detach())
        torch.testing.assert_close(result.objective, expected)
        result.objective.backward()
        self.assertIsNone(transition.grad)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_stage2a_and_revision_share_ratio_values_but_not_gradient_ownership(self) -> None:
        logits_stage2a = torch.tensor([[0.3, 1.1, -0.4], [1.2, -0.1, 0.2]], dtype=torch.float64, requires_grad=True)
        logits_revision = logits_stage2a.detach().clone().requires_grad_(True)
        targets = torch.tensor([1, 0])
        transition_stage2a = torch.tensor([[0.75, 0.2, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]], dtype=torch.float64, requires_grad=True)
        transition_revision = transition_stage2a.detach().clone().requires_grad_(True)
        stage2a = t_revision_reweight_objective(logits_stage2a, targets, transition_stage2a, denominator_floor=1e-12, detach_ratio=True)
        revision = t_revision_reweight_objective(logits_revision, targets, transition_revision, denominator_floor=1e-12, detach_ratio=False)
        torch.testing.assert_close(stage2a.sample_weights, revision.sample_weights)
        torch.testing.assert_close(stage2a.objective, revision.objective)
        stage2a.objective.backward()
        revision.objective.backward()
        self.assertIsNone(transition_stage2a.grad)
        self.assertIsNotNone(transition_revision.grad)
        self.assertTrue(torch.isfinite(transition_revision.grad).all())
        self.assertGreater(float(transition_revision.grad.abs().sum()), 0.0)
        for gradient in (logits_stage2a.grad, logits_revision.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_denominator_fail_fast_is_unchanged_when_ratio_is_detached(self) -> None:
        logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'strictly greater'):
            t_revision_reweight_objective(logits, torch.tensor([0]), torch.zeros(2, 2), denominator_floor=0.0, detach_ratio=True)

    def test_epoch_telemetry_covers_weights_updates_and_transition_health(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            algorithm = _algorithm(directory)
            algorithm.diagnostic_transition = np.eye(3, dtype=np.float64)
            final = algorithm.run()
            rows = [json.loads(line) for line in (Path(directory) / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            transition_row = next((row for row in rows if row['event'] == 'transition_initialization'))
            self.assertEqual(transition_row['posterior_finite_count'], transition_row['posterior_value_count'])
            self.assertLessEqual(transition_row['posterior_row_sum_max_error'], 1e-06)
            self.assertEqual(len(transition_row['pseudo_anchor_indices']), 3)
            for stage in ('classifier_initialization', 'revision'):
                row = next((value for value in rows if value.get('event') == 'epoch' and value.get('stage') == stage))
                for field in ('weight_p50', 'weight_p90', 'weight_p95', 'weight_p99', 'weight_ess', 'weight_ess_fraction', 'gradient_norm', 'parameter_norm', 'update_norm', 'relative_update_norm', 'optimizer_step_count', 'denominator_min'):
                    self.assertIn(field, row)
                    self.assertTrue(np.isfinite(row[field]), field)
                self.assertEqual(row['weight_negative_count'], 0)
                self.assertEqual(row['weight_nonfinite_count'], 0)
                self.assertGreater(row['optimizer_step_count'], 0)
            revision = next((row for row in rows if row.get('event') == 'epoch' and row.get('stage') == 'revision'))
            self.assertIn('initial_transition', revision)
            self.assertIn('delta_transition', revision)
            self.assertIn('revised_transition', revision)
            self.assertIn('revised_transition_negative_entry_count', revision)
            self.assertGreater(revision['delta_l1'], 0.0)
            self.assertIn('initial_true_T_relative_L1_error', final)
            self.assertIn('revised_true_T_relative_L1_error', final)

# --- merged from test_workflow_registry.py ---
import inspect

# --- merged from test_workflow_registry.py ---
from pathlib import Path

# --- merged from test_workflow_registry.py ---
import unittest

# --- merged from test_workflow_registry.py ---
from lnl_toolbox.training.experiment import run_experiment

# --- merged from test_workflow_registry.py ---
from lnl_toolbox.training.workflows import WorkflowRegistry, create_workflow_registry, method_name, resolve_workflow

# --- merged from test_workflow_registry.py ---
class _workflow_registry_WorkflowRegistryTest(unittest.TestCase):

    def test_builtin_workflows_are_registered_without_main_runner_branches(self) -> None:
        registry = create_workflow_registry()
        self.assertEqual(registry.names(), ('ca2c', 'cal', 'cnlcu', 'coteaching', 'dividemix', 'dld', 'dual_t', 'importance_reweighting', 'l2rw', 'lend', 'mc_ldce', 'pcse', 't_revision', 'upm', 'volmin', 'volminnet'))
        source = inspect.getsource(run_experiment)
        for name in registry.names():
            self.assertNotIn(name, source)

    def test_method_name_supports_scalar_mapping_and_default(self) -> None:
        self.assertEqual(method_name({}), '')
        self.assertEqual(method_name({'method': ' Dual_T '}), 'dual_t')
        self.assertEqual(method_name({'method': {'name': 'PCSE'}}), 'pcse')
        self.assertEqual(method_name({'method': 'VolMinNet'}), 'volminnet')

    def test_unknown_and_renamed_methods_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, 'unknown method'):
            resolve_workflow({'method': 'not_registered'})
        with self.assertRaisesRegex(ValueError, 'dual_t_forward.*dual_t'):
            resolve_workflow({'method': 'dual_t_forward'})

    def test_registry_rejects_duplicate_names_and_loads_lazily(self) -> None:
        registry = WorkflowRegistry()
        registry.add('example', 'pathlib', 'Path')
        with self.assertRaisesRegex(KeyError, 'already registered'):
            registry.add('example', 'pathlib', 'Path')
        self.assertIs(registry.resolve('example'), Path)
