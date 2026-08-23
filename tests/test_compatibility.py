from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from lnl_toolbox.data.profile import (
    DatasetDeclarationConflict,
    DatasetDeclarations,
    DatasetProfile,
    KnowledgeState,
    Modality,
    NoiseKnowledge,
    NoiseOrigin,
    NoiseRateInfo,
    NoiseRateStatus,
    NoiseStatus,
    resolve_dataset_capabilities,
)
from lnl_toolbox.training.compatibility import (
    CompatibilityReason,
    CompatibilityResult,
    CompatibilityStatus,
    MethodRequirements,
    resolve_compatibility,
)
from lnl_toolbox.training.runners import create_runner_registry, runner_names
from lnl_toolbox.training.service import ExperimentService


def _profile(
    *,
    modality: Modality = Modality.IMAGE,
    classes: int = 10,
    clean: KnowledgeState = KnowledgeState.UNKNOWN,
    clean_validation: KnowledgeState = KnowledgeState.UNKNOWN,
    noise: NoiseKnowledge | None = None,
) -> DatasetProfile:
    return DatasetProfile(
        dataset="fixture", adapter="fixture", source="fixture-root",
        task="classification", modality=modality, num_classes=classes,
        input_shape=(32, 32, 3) if modality is Modality.IMAGE else (12,),
        channels=3 if modality is Modality.IMAGE else None,
        sample_counts_by_split=(("train", 20), ("test", 10)),
        available_splits=("train", "test"),
        class_names=tuple(str(index) for index in range(classes)),
        class_distribution_by_split=(
            ("train", tuple(20 // classes for _ in range(classes))),
            ("test", tuple(10 // classes for _ in range(classes))),
        ),
        observed_train_labels=KnowledgeState.AVAILABLE,
        clean_train_labels=clean,
        clean_validation_labels=clean_validation,
        stable_indices=KnowledgeState.AVAILABLE,
        dataset_fingerprint="d" * 64,
        split_fingerprints=(("train", "a" * 64), ("test", "b" * 64)),
        noise=noise or NoiseKnowledge(),
    )


def _method(**overrides) -> MethodRequirements:
    values = {
        "method": "fixture_method",
        "supported_modalities": frozenset({Modality.IMAGE}),
        "requires_noise_manifest": False,
    }
    values.update(overrides)
    return MethodRequirements(**values)


class NoiseKnowledgeTest(unittest.TestCase):
    def test_rate_states_and_round_trip(self) -> None:
        known = NoiseRateInfo(NoiseRateStatus.KNOWN, 0.2, "user")
        estimated = NoiseRateInfo(NoiseRateStatus.ESTIMATED, 0.3, "estimator:v1")
        self.assertEqual(NoiseRateInfo.from_dict(known.to_dict()), known)
        self.assertEqual(NoiseRateInfo.from_dict(estimated.to_dict()), estimated)
        self.assertIsNone(NoiseRateInfo().value)
        self.assertIsNone(NoiseRateInfo(NoiseRateStatus.NOT_APPLICABLE).value)

    def test_invalid_rate_state_combinations_fail(self) -> None:
        for constructor in (
            lambda: NoiseRateInfo(NoiseRateStatus.KNOWN),
            lambda: NoiseRateInfo(NoiseRateStatus.ESTIMATED, 0.2),
            lambda: NoiseRateInfo(NoiseRateStatus.UNKNOWN, 0.2),
            lambda: NoiseRateInfo(NoiseRateStatus.KNOWN, -0.1),
        ):
            with self.assertRaises(ValueError):
                constructor()

    def test_profile_round_trip_and_fingerprint_are_deterministic(self) -> None:
        profile = _profile()
        restored = DatasetProfile.from_dict(profile.to_dict())
        self.assertEqual(restored, profile)
        self.assertEqual(restored.fingerprint, profile.fingerprint)
        self.assertEqual(len(profile.fingerprint), 64)

    def test_user_declarations_cannot_override_hard_facts(self) -> None:
        profile = _profile(
            clean=KnowledgeState.AVAILABLE,
            noise=NoiseKnowledge(
                NoiseStatus.CLEAN,
                NoiseOrigin.UNKNOWN,
                NoiseRateInfo(NoiseRateStatus.NOT_APPLICABLE),
            ),
        )
        with self.assertRaises(DatasetDeclarationConflict):
            resolve_dataset_capabilities(
                profile,
                DatasetDeclarations(clean_train_labels=KnowledgeState.UNAVAILABLE),
            )
        with self.assertRaises(DatasetDeclarationConflict):
            resolve_dataset_capabilities(
                profile,
                DatasetDeclarations(noise_status=NoiseStatus.NOISY),
            )


class CompatibilityResolverTest(unittest.TestCase):
    def test_image_method_and_tabular_mismatch(self) -> None:
        image = resolve_dataset_capabilities(_profile(clean=KnowledgeState.AVAILABLE))
        result = resolve_compatibility(image, _method())
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE)

        tabular = resolve_dataset_capabilities(
            _profile(modality=Modality.TABULAR, classes=2, clean=KnowledgeState.AVAILABLE)
        )
        result = resolve_compatibility(tabular, _method())
        self.assertEqual(result.status, CompatibilityStatus.INCOMPATIBLE)
        self.assertIn("unsupported_modality", {reason.code for reason in result.reasons})

    def test_class_count_constraints(self) -> None:
        dataset = resolve_dataset_capabilities(_profile(classes=3))
        exact_binary = _method(exact_classes=frozenset({2}), max_classes=2)
        result = resolve_compatibility(dataset, exact_binary)
        self.assertEqual(result.status, CompatibilityStatus.INCOMPATIBLE)
        self.assertIn("wrong_class_count", {reason.code for reason in result.reasons})

    def test_clean_label_unknown_is_not_unavailable(self) -> None:
        requirement = _method(requires_clean_train_labels=True)
        unknown = resolve_dataset_capabilities(_profile(clean=KnowledgeState.UNKNOWN))
        self.assertEqual(
            resolve_compatibility(unknown, requirement).status,
            CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS,
        )
        unavailable = resolve_dataset_capabilities(
            _profile(clean=KnowledgeState.UNKNOWN),
            DatasetDeclarations(clean_train_labels=KnowledgeState.UNAVAILABLE),
        )
        self.assertEqual(
            resolve_compatibility(unavailable, requirement).status,
            CompatibilityStatus.INCOMPATIBLE,
        )
        available = resolve_dataset_capabilities(
            _profile(clean=KnowledgeState.UNKNOWN),
            DatasetDeclarations(clean_train_labels=KnowledgeState.AVAILABLE),
        )
        self.assertEqual(
            resolve_compatibility(available, requirement).status,
            CompatibilityStatus.COMPATIBLE,
        )

    def test_dataset_true_rate_is_distinct_from_method_prior(self) -> None:
        profile = _profile(clean=KnowledgeState.AVAILABLE)
        true_rate_only = resolve_dataset_capabilities(
            profile,
            DatasetDeclarations(
                noise_rate=NoiseRateInfo(NoiseRateStatus.KNOWN, 0.2, "user")
            ),
        )
        method = _method(requires_method_noise_prior=True)
        result = resolve_compatibility(true_rate_only, method)
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS)
        self.assertEqual(result.required_user_inputs, ("noise_rate_prior",))

        with_prior = resolve_dataset_capabilities(
            profile,
            DatasetDeclarations(
                method_noise_rate_prior=NoiseRateInfo(
                    NoiseRateStatus.KNOWN, 0.2, "user"
                )
            ),
        )
        self.assertEqual(
            resolve_compatibility(with_prior, method).status,
            CompatibilityStatus.COMPATIBLE,
        )

        true_rate_method = _method(requires_dataset_true_noise_rate=True)
        self.assertEqual(
            resolve_compatibility(true_rate_only, true_rate_method).status,
            CompatibilityStatus.COMPATIBLE,
        )
        unknown = resolve_dataset_capabilities(_profile())
        self.assertIn(
            "dataset_noise_rate",
            resolve_compatibility(unknown, true_rate_method).required_user_inputs,
        )

    def test_native_noise_and_pretrained_requirements(self) -> None:
        native = resolve_dataset_capabilities(
            _profile(
                clean=KnowledgeState.UNKNOWN,
                noise=NoiseKnowledge(
                    NoiseStatus.NOISY, NoiseOrigin.NATIVE, NoiseRateInfo()
                ),
            )
        )
        result = resolve_compatibility(native, _method())
        self.assertEqual(result.status, CompatibilityStatus.INCOMPATIBLE)
        self.assertIn("unsupported_native_noise", {reason.code for reason in result.reasons})
        native_supported = _method(supports_native_noisy_labels=True)
        self.assertEqual(
            resolve_compatibility(native, native_supported).status,
            CompatibilityStatus.COMPATIBLE,
        )

        dataset = resolve_dataset_capabilities(_profile(clean=KnowledgeState.AVAILABLE))
        result = resolve_compatibility(
            dataset, _method(required_pretrained_roles=("upm_main_best",))
        )
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS)
        self.assertIn("pretrained:upm_main_best", result.required_user_inputs)

    def test_machine_result_uses_stable_reason_codes(self) -> None:
        result = CompatibilityResult(
            CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS,
            "method",
            "dataset",
            reasons=(CompatibilityReason("unknown_noise_rate", "rate is unknown"),),
            required_user_inputs=("dataset_noise_rate",),
        )
        value = result.to_dict()
        self.assertEqual(value["status"], "compatible_with_requirements")
        self.assertEqual(value["reason_codes"], ["unknown_noise_rate"])
        self.assertEqual(value["required_user_inputs"], ["dataset_noise_rate"])


class RunnerRequirementsTest(unittest.TestCase):
    def test_twelve_method_requirements_are_attached_to_runner_registry(self) -> None:
        registry = create_runner_registry()
        expected = {
            "upm": ("upm", Modality.IMAGE, 2),
            "coteaching": ("coteaching", Modality.IMAGE, 2),
            "lend": ("lend", Modality.IMAGE, 2),
            "dividemix": ("dividemix", Modality.IMAGE, 2),
            "pcse": ("pcse", Modality.IMAGE, 3),
            "importance_reweighting": (
                "importance_reweighting", Modality.TABULAR, 2
            ),
            "t_revision": ("t_revision", Modality.IMAGE, 2),
            "cnlcu": ("cnlcu", Modality.IMAGE, 2),
            "dld": ("dld", Modality.IMAGE, 2),
            "ca2c": ("ca2c", Modality.IMAGE, 2),
            "instance_transition": ("pdl", Modality.IMAGE, 2),
            "volminnet": ("volminnet", Modality.IMAGE, 3),
        }
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
        for runner in ("coteaching", "cnlcu", "dividemix"):
            requirement = registry.get(runner).requirements({})
            self.assertTrue(requirement.requires_method_noise_prior)
            self.assertFalse(requirement.requires_dataset_true_noise_rate)

        pcse = registry.get("pcse").requirements(
            {"pretraining_stage": {"mode": "external_checkpoint"}}
        )
        self.assertEqual(pcse.required_pretrained_roles, ("upm_main_best",))
        dld = registry.get("dld").requirements(
            {"dld": {"feature_extractor": {"source": "external_checkpoint"}}}
        )
        self.assertEqual(dld.required_pretrained_roles, ("upm_main_best",))


class ExperimentCompatibilityServiceTest(unittest.TestCase):
    def test_service_and_direct_resolver_agree(self) -> None:
        capabilities = resolve_dataset_capabilities(
            _profile(modality=Modality.TABULAR, classes=2)
        )
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        service = ExperimentService(data_service=data_service)
        result = service.resolve_method_compatibility(
            "fixture", "importance_reweighting"
        )
        direct = resolve_compatibility(
            capabilities,
            create_runner_registry().get("importance_reweighting").requirements({}),
        )
        self.assertEqual(result, direct)
        data_service.capabilities.assert_called_once_with(
            "fixture", seed=0, persist=False
        )

    def test_config_prior_is_not_treated_as_dataset_true_rate(self) -> None:
        capabilities = resolve_dataset_capabilities(
            _profile(clean=KnowledgeState.AVAILABLE)
        )
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        service = ExperimentService(data_service=data_service)
        config = {
            "method": "coteaching",
            "execution": {"runner": "coteaching"},
            "data": {"name": "fixture"},
            "coteaching": {"noise_rate": 0.2},
        }
        result = service.resolve_method_compatibility(config, config)
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE)
        self.assertEqual(capabilities.noise_rate.status, NoiseRateStatus.UNKNOWN)

    def test_incompatible_preflight_stops_before_runner_invocation(self) -> None:
        capabilities = resolve_dataset_capabilities(
            _profile(modality=Modality.TABULAR, classes=2)
        )
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        runner = Mock(name="runner")
        runner.name = "upm"
        runner.requirements.return_value = _method()
        service = ExperimentService(data_service=data_service)
        config = {
            "method": "upm",
            "execution": {"runner": "upm"},
            "data": {"name": "fixture"},
        }
        with patch("lnl_toolbox.catalog.validate_config", return_value=runner):
            with self.assertRaisesRegex(ValueError, "unsupported_modality"):
                service.preflight(config, check_data=True)
        runner.invoke.assert_not_called()

    def test_preflight_requires_method_prior_but_not_dataset_true_rate(self) -> None:
        capabilities = resolve_dataset_capabilities(
            _profile(clean=KnowledgeState.AVAILABLE)
        )
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        runner = create_runner_registry().get("coteaching")
        service = ExperimentService(data_service=data_service)
        base = {
            "schema_version": 1,
            "kind": "experiment",
            "method": "coteaching",
            "execution": {"runner": "coteaching"},
            "data": {"name": "fixture"},
        }
        with patch("lnl_toolbox.catalog.validate_config", return_value=runner):
            with self.assertRaisesRegex(ValueError, "requires_noise_rate_prior"):
                service.preflight(base, check_data=True)
            configured = {**base, "coteaching": {"noise_rate": 0.2}}
            self.assertIs(service.preflight(configured, check_data=True), runner)
        self.assertEqual(
            service.last_compatibility.status, CompatibilityStatus.COMPATIBLE
        )
        self.assertEqual(capabilities.noise_rate.status, NoiseRateStatus.UNKNOWN)

    def test_unknown_dataset_rate_passes_when_method_does_not_require_it(self) -> None:
        capabilities = resolve_dataset_capabilities(
            _profile(clean=KnowledgeState.AVAILABLE)
        )
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        service = ExperimentService(data_service=data_service)
        result = service.resolve_method_compatibility("fixture", "upm")
        self.assertEqual(result.status, CompatibilityStatus.COMPATIBLE)

    def test_method_discovery_uses_every_central_runner(self) -> None:
        capabilities = resolve_dataset_capabilities(
            _profile(clean=KnowledgeState.AVAILABLE)
        )
        data_service = Mock()
        data_service.capabilities.return_value = capabilities
        results = ExperimentService(data_service=data_service).list_compatible_methods(
            "fixture"
        )
        self.assertEqual(len(results), len(runner_names()))
        by_method = {result.method: result for result in results}
        self.assertIn("upm", by_method)
        self.assertIn("requirements_unavailable", {
            reason.code for reason in by_method["binary"].reasons
        })


if __name__ == "__main__":
    unittest.main()
