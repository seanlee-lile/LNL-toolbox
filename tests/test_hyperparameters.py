import unittest

from lnl_toolbox.core.hyperparameters import (
    ParameterRecord,
    resolve_parameter_sampling,
    sample_parameters,
)


class HyperparameterTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_does_not_touch_global_rng(self):
        import random

        candidates = {
            "lr": (0.01, 0.05),
            "depth": (14, 32),
        }
        random.seed(91)
        before = random.getstate()
        first = sample_parameters(
            "loss_correction",
            7,
            candidates,
        )
        after = random.getstate()
        second = sample_parameters(
            "loss_correction",
            7,
            candidates,
        )
        self.assertEqual(before, after)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertIn(first.parameters["lr"], candidates["lr"])
        self.assertIn(first.parameters["depth"], candidates["depth"])

    def test_parameter_record_round_trip(self) -> None:
        record = sample_parameters(
            "fixture",
            3,
            {"beta": (0.001,)},
            sources={"beta": "fixture source"},
        )
        self.assertEqual(
            ParameterRecord.from_dict(record.to_dict()),
            record,
        )

    def test_resolution_does_not_mutate_input(self) -> None:
        config = {
            "seed": 4,
            "nested": {"value": [1, 2]},
            "parameter_sampling": {
                "paper": "fixture",
                "seed": 9,
                "candidates": {"alpha": [0.1, 0.2]},
            },
        }
        original_nested = config["nested"]
        resolved, record = resolve_parameter_sampling(config)
        self.assertIsNotNone(record)
        self.assertNotIn("parameter_record", config)
        self.assertIs(config["nested"], original_nested)
        self.assertIn(
            resolved["resolved_parameters"]["alpha"],
            (0.1, 0.2),
        )

    def test_invalid_candidates_and_non_json_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            sample_parameters("fixture", 1, {"alpha": ()})
        with self.assertRaisesRegex(TypeError, "JSON-compatible"):
            sample_parameters(
                "fixture",
                1,
                {"alpha": (object(),)},
            )


if __name__ == "__main__":
    unittest.main()
