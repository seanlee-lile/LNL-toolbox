import unittest

from lnl_toolbox.scratch.web.server import _job_error_code


class ScratchRunJobGuidanceTest(unittest.TestCase):
    def test_dataset_model_mismatch_is_not_masked_by_context_path(self):
        message = (
            "Scratch error: step 15 — Create Model failed. Available context: "
            "['_progress_path', 'dataset']. Original error: model num_classes=10 "
            "does not match the loaded dataset num_classes=2"
        )
        self.assertEqual(_job_error_code(message), "dataset-model-class-mismatch")

    def test_missing_dataset_cases_have_actionable_codes(self):
        self.assertEqual(_job_error_code("load_dataset requires a dataset name"), "missing-dataset-selection")
        self.assertEqual(_job_error_code("dataset path is required"), "missing-dataset-path")
        self.assertEqual(_job_error_code("dataset file not found"), "dataset-unavailable")


if __name__ == "__main__":
    unittest.main()
