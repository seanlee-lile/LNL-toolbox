import unittest

from lnl_toolbox.core.hyperparameters import ParameterRecord, sample_parameters


class HyperparameterTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_within_candidates(self) -> None:
        candidates = {"lr": (0.01, 0.05), "depth": (14, 32)}
        first = sample_parameters("loss_correction", 7, candidates)
        second = sample_parameters("loss_correction", 7, candidates)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertIn(first.parameters["lr"], candidates["lr"])
        self.assertIn(first.parameters["depth"], candidates["depth"])

    def test_parameter_record_round_trip(self) -> None:
        record = sample_parameters("fine", 3, {"beta": (0.001,)})
        self.assertEqual(ParameterRecord.from_dict(record.to_dict()), record)


if __name__ == "__main__":
    unittest.main()
