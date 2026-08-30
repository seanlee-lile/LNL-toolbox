from __future__ import annotations

import unittest

from lnl_toolbox.training.prerequisites import (
    PrerequisiteRegistry,
    ReadinessLevel,
    ReadinessStatus,
    SourceDescriptor,
    ValidationMetadata,
)


class PrerequisiteContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = SourceDescriptor(
            key="fixture",
            kind="classifier",
            name="Fixture classifier",
            requirement="a trained classifier",
            obtain="finish a supported producer run",
            provide="select its run directory",
            validator="fixture",
            supported_sources=("ce",),
            config_paths=(("source",),),
        )

    def test_ready_requires_every_readiness_layer(self) -> None:
        value = ValidationMetadata.ready(
            self.descriptor, "validated", provenance={"method": "ce"}
        )
        self.assertEqual(value.status, ReadinessStatus.READY)
        self.assertEqual(value.level, ReadinessLevel.CONSUMER_VALIDATED)
        self.assertTrue(value.configured)
        self.assertTrue(value.path_exists)
        self.assertTrue(value.schema_valid)
        self.assertTrue(value.identity_compatible)
        self.assertTrue(value.consumer_validated)
        self.assertEqual(value.to_dict()["provenance"]["method"], "ce")

    def test_path_exists_does_not_mean_ready(self) -> None:
        value = ValidationMetadata.invalid(
            self.descriptor,
            ReadinessLevel.PATH_EXISTS,
            "checkpoint schema is invalid",
        )
        self.assertEqual(value.status, ReadinessStatus.INVALID)
        self.assertTrue(value.path_exists)
        self.assertFalse(value.schema_valid)
        self.assertFalse(value.consumer_validated)

    def test_registry_delegates_without_consumer_knowledge(self) -> None:
        registry = PrerequisiteRegistry()
        registry.register(
            "fixture",
            lambda descriptor, config: ValidationMetadata.ready(
                descriptor, "validated", provenance={"value": config["source"]}
            ),
        )
        result = registry.evaluate(self.descriptor, {"source": "run"})
        self.assertEqual(result.provenance["value"], "run")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry.register("fixture", lambda *_: result)


if __name__ == "__main__":
    unittest.main()
