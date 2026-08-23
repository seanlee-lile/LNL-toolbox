from __future__ import annotations

from pathlib import Path
import unittest

from lnl_toolbox.catalog import discover_recipes, load_yaml
from lnl_toolbox.core.config_schema import (
    normalize_experiment_config,
    runtime_experiment_config,
)


class ConfigSchemaTest(unittest.TestCase):
    def test_binary_legacy_aliases_normalize_to_shared_sections(self) -> None:
        value = normalize_experiment_config({
            "execution": {"runner": "binary"},
            "data": {"name": "synthetic_binary_2d"},
            "batch_size": 8,
            "learning_rate": 0.02,
            "epochs": 3,
        })
        self.assertEqual(value["loader"]["batch_size"], 8)
        self.assertEqual(value["optimizer"]["lr"], 0.02)
        self.assertEqual(value["trainer"]["epochs"], 3)
        for key in ("batch_size", "learning_rate", "epochs"):
            self.assertNotIn(key, value)

    def test_noise_type_is_publicly_removed_but_runtime_compatible(self) -> None:
        value = normalize_experiment_config({
            "execution": {"runner": "supervised"},
            "data": {"name": "synthetic_multiclass"},
            "noise": {"type": "symmetric", "rate": 0.2},
        })
        self.assertEqual(value["noise"]["name"], "symmetric")
        self.assertNotIn("type", value["noise"])
        self.assertEqual(runtime_experiment_config(value)["noise"]["type"], "symmetric")

    def test_unknown_top_level_field_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown top-level"):
            normalize_experiment_config({
                "execution": {"runner": "clean"},
                "data": {"name": "synthetic_multiclass"},
                "trianer": {"epochs": 1},
            })

    def test_every_active_yaml_is_versioned_and_canonical(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "configs").rglob("*.yaml"))
        self.assertEqual(len(paths), 94)
        for path in paths:
            value = load_yaml(path)
            self.assertEqual(value["schema_version"], 1, path)
            self.assertIn(value["kind"], {"experiment", "fragment", "mentor_artifact"})
            if value["kind"] == "experiment":
                self.assertIn("runner", value["execution"], path)
                self.assertNotIn("root", value["data"], path)
                self.assertNotIn("type", value.get("noise", {}), path)
        self.assertEqual(len(discover_recipes(root, include_conditional=True)), 67)


if __name__ == "__main__":
    unittest.main()
