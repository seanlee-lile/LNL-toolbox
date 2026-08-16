from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from lnl_toolbox.data.binary_benchmarks import BinaryBenchmarkTensorDataset
from lnl_toolbox.training.experiment import run_experiment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/reproduction/uci_heart_importance_reweighting.yaml"


def _heart_fixture(path: Path, samples: int = 60) -> str:
    rows: list[str] = []
    for index in range(samples):
        label = 1 if index % 2 == 0 else 2
        center = -1.0 if label == 1 else 1.0
        features = [center + 0.03 * ((index + column) % 7) for column in range(13)]
        rows.append(" ".join([*(f"{value:.6f}" for value in features), str(label)]))
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _config(source: Path, sha256: str, epochs: int) -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["data"]["path"] = str(source)
    config["data"]["sha256"] = sha256
    config["data"]["expected_samples"] = 60
    config["posterior_stage"].update({
        "max_centers": 8,
        "max_iterations": 20,
        "learning_rate": 0.01,
    })
    config["loader"]["batch_size"] = 12
    config["trainer"].update({"epochs": epochs, "device": "cpu"})
    return config


class ImportanceReweightingUCIAdapterTest(unittest.TestCase):
    def test_training_view_never_exposes_clean_target(self) -> None:
        from lnl_toolbox.data.binary_benchmarks import BinaryBenchmark

        benchmark = BinaryBenchmark(
            np.zeros((4, 13), dtype=np.float32),
            np.asarray([0, 1, 0, 1]),
            "fixture",
            global_indices=np.asarray([5, 7, 11, 13]),
        )
        item = BinaryBenchmarkTensorDataset(
            benchmark, np.asarray([1, 1, 0, 0])
        )[0]
        self.assertEqual(set(item), {"input", "target", "index"})

    def test_real_adapter_resume_preserves_data_and_method_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "heart.dat"
            sha256 = _heart_fixture(source)
            run_dir = root / "run"
            run_experiment(_config(source, sha256, 2), output_dir=run_dir)

            artifact_names = (
                "noise_manifest.npz",
                "posterior_snapshot.npz",
                "noise_rate_artifact.npz",
                "preprocessing_state.json",
                "split_manifest.json",
            )
            before = {
                name: (hashlib.sha256((run_dir / name).read_bytes()).hexdigest(),
                       (run_dir / name).stat().st_mtime_ns)
                for name in artifact_names
            }
            run_experiment(
                _config(source, sha256, 3),
                resume=run_dir / "last.pt",
            )
            after = {
                name: (hashlib.sha256((run_dir / name).read_bytes()).hexdigest(),
                       (run_dir / name).stat().st_mtime_ns)
                for name in artifact_names
            }
            self.assertEqual(before, after)
            final = json.loads((run_dir / "final_metrics.json").read_text())
            self.assertEqual(final["completed_epochs"], 3)
            self.assertEqual(final["posterior"]["value_count"], 72)
            self.assertEqual(final["posterior"]["finite_count"], 72)
            self.assertLess(final["posterior"]["row_sum_max_error"], 1e-10)
            self.assertEqual(final["weights"]["negative_count"], 0)
            self.assertEqual(final["weights"]["nonfinite_count"], 0)
            self.assertGreater(final["weights"]["ess"], 0.0)
            self.assertFalse(final["optimization"]["optimizer_stall"])

    def test_raw_hash_and_resume_split_identity_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "heart.dat"
            sha256 = _heart_fixture(source)
            bad = _config(source, "0" * 64, 1)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                run_experiment(bad, output_dir=root / "bad")

            run_dir = root / "run"
            run_experiment(_config(source, sha256, 1), output_dir=run_dir)
            changed = _config(source, sha256, 2)
            changed["data"]["split"]["seed"] += 1
            with self.assertRaisesRegex(ValueError, "identity mismatch|configuration mismatch"):
                run_experiment(changed, resume=run_dir / "last.pt")


if __name__ == "__main__":
    unittest.main()
