from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.importance_reweighting import (
    ImportanceReweightingConfig,
    KLIEPBinaryNoisyPosteriorEstimator,
    PaperRawMinNoiseRateEstimator,
)
from lnl_toolbox.data.binary_synthetic import (
    generate_synthetic_binary_high_dim,
)
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.experiment import run_experiment


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiment"
    / "importance_reweighting_binary_high_dim_smoke.yaml"
)


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def build_estimator() -> KLIEPBinaryNoisyPosteriorEstimator:
    return KLIEPBinaryNoisyPosteriorEstimator(
        bandwidth=4.0,
        max_centers=16,
        max_iterations=150,
        learning_rate=0.02,
        tolerance=1.0e-7,
        epsilon=1.0e-12,
        seed=37,
    )


class KLIEPPosteriorTest(unittest.TestCase):
    def test_high_dimensional_data_is_balanced_and_deterministic(self) -> None:
        first = generate_synthetic_binary_high_dim(
            40, 20, 5, split="train"
        )
        second = generate_synthetic_binary_high_dim(
            40, 20, 5, split="train"
        )
        self.assertEqual(first.features.shape, (40, 20))
        self.assertEqual(first.dataset, "synthetic_binary_high_dim")
        np.testing.assert_array_equal(np.bincount(first.labels), [20, 20])
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.global_indices, second.global_indices)

    def test_posterior_is_valid_deterministic_and_index_aligned(self) -> None:
        data = generate_synthetic_binary_high_dim(
            80, 20, 9, start_index=100, split="train"
        )
        estimator = build_estimator()
        first = estimator.fit_predict(
            data.features,
            data.labels,
            data.global_indices,
            dataset=data.dataset,
            split=data.split,
        )
        second = estimator.fit_predict(
            data.features,
            data.labels,
            data.global_indices,
            dataset=data.dataset,
            split=data.split,
        )
        self.assertEqual(first.noisy_probabilities.shape, (80, 2))
        self.assertTrue(np.isfinite(first.noisy_probabilities).all())
        self.assertTrue((first.noisy_probabilities >= 0.0).all())
        np.testing.assert_allclose(
            first.noisy_probabilities.sum(axis=1),
            np.ones(80),
            rtol=1e-8,
            atol=1e-10,
        )
        np.testing.assert_array_equal(
            first.global_indices, np.sort(data.global_indices)
        )
        np.testing.assert_array_equal(
            first.noisy_probabilities,
            second.noisy_probabilities,
        )

    def test_input_permutation_preserves_stable_index_mapping(self) -> None:
        data = generate_synthetic_binary_high_dim(
            64, 20, 11, start_index=17, split="train"
        )
        permutation = np.random.default_rng(99).permutation(64)
        estimator = build_estimator()
        original = estimator.fit_predict(
            data.features,
            data.labels,
            data.global_indices,
            dataset=data.dataset,
            split=data.split,
        )
        permuted = estimator.fit_predict(
            data.features[permutation],
            data.labels[permutation],
            data.global_indices[permutation],
            dataset=data.dataset,
            split=data.split,
        )
        np.testing.assert_array_equal(
            original.global_indices, permuted.global_indices
        )
        np.testing.assert_allclose(
            original.noisy_probabilities,
            permuted.noisy_probabilities,
            rtol=0.0,
            atol=0.0,
        )

    def test_density_ratio_satisfies_empirical_normalization(self) -> None:
        data = generate_synthetic_binary_high_dim(
            60, 20, 13, split="train"
        )
        estimator = build_estimator()
        ordered = np.argsort(data.global_indices, kind="stable")
        values = data.features[ordered]
        targets = data.labels[ordered]
        fit = estimator._fit_ratio(
            values[targets == 0],
            values,
            class_index=0,
        )
        ratio = estimator._kernel(values, fit.centers) @ fit.coefficients
        self.assertAlmostEqual(float(ratio.mean()), 1.0, places=10)
        self.assertTrue(np.isfinite(ratio).all())
        self.assertTrue((ratio >= 0.0).all())

    def test_invalid_shapes_labels_and_indices_are_rejected(self) -> None:
        data = generate_synthetic_binary_high_dim(
            20, 20, 17, split="train"
        )
        estimator = build_estimator()
        cases = (
            (data.features[:, :2], data.labels, data.global_indices, "D > 2"),
            (
                data.features,
                np.zeros(20, dtype=np.int64),
                data.global_indices,
                "both binary classes",
            ),
            (
                data.features,
                np.where(data.labels == 1, 2, 0),
                data.global_indices,
                "only 0 and 1",
            ),
            (
                data.features,
                data.labels,
                np.zeros(20, dtype=np.int64),
                "unique",
            ),
        )
        for features, targets, indices, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    estimator.fit_predict(
                        features,
                        targets,
                        indices,
                        dataset=data.dataset,
                        split=data.split,
                    )

    def test_raw_min_rates_from_kliep_snapshot_are_legal(self) -> None:
        data = generate_synthetic_binary_high_dim(
            80, 20, 23, split="train"
        )
        snapshot = build_estimator().fit_predict(
            data.features,
            data.labels,
            data.global_indices,
            dataset=data.dataset,
            split=data.split,
        )
        rates = PaperRawMinNoiseRateEstimator().estimate(snapshot)
        self.assertGreaterEqual(rates.rho_positive, 0.0)
        self.assertGreaterEqual(rates.rho_negative, 0.0)
        self.assertLess(rates.rho_positive + rates.rho_negative, 1.0)


class KLIEPWorkflowTest(unittest.TestCase):
    def test_config_requires_matching_high_dimension(self) -> None:
        config = load_config()
        parsed = ImportanceReweightingConfig.from_mapping(config)
        self.assertEqual(parsed.data["dimension"], 20)
        self.assertEqual(parsed.posterior_stage["name"], "kliep")

        invalid = deepcopy(config)
        invalid["model"]["in_features"] = 19
        with self.assertRaisesRegex(ValueError, "in_features"):
            ImportanceReweightingConfig.from_mapping(invalid)
        invalid = deepcopy(config)
        invalid["num_classes"] = 10
        with self.assertRaisesRegex(ValueError, "num_classes"):
            ImportanceReweightingConfig.from_mapping(invalid)

    def test_high_dimensional_smoke_and_resume_preserve_artifacts(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            result = run_experiment(config, run_dir)
            snapshot = result / "posterior_snapshot.npz"
            rates = result / "noise_rate_artifact.npz"
            snapshot_before = snapshot.read_bytes()
            rates_before = rates.read_bytes()
            first = read_checkpoint(result / "last.pt", "cpu")
            self.assertEqual(first["method_state"]["final_completed_epochs"], 2)
            self.assertEqual(first["method_state"]["final_global_step"], 8)
            self.assertEqual(
                first["posterior_backend_identity"]["name"], "kliep"
            )
            self.assertEqual(
                first["posterior_backend_identity"]["feature_dimension"], 20
            )
            self.assertEqual(
                first["posterior_backend_hash"],
                first["method_state"]["posterior_backend_hash"],
            )

            config["trainer"]["epochs"] = 3
            run_experiment(config, resume=result / "last.pt")
            final = read_checkpoint(result / "last.pt", "cpu")
            self.assertEqual(final["method_state"]["final_completed_epochs"], 3)
            self.assertEqual(final["method_state"]["final_global_step"], 12)
            self.assertEqual(snapshot.read_bytes(), snapshot_before)
            self.assertEqual(rates.read_bytes(), rates_before)
            metrics = yaml.safe_load(
                (result / "final_metrics.json").read_text(encoding="utf-8")
            )
            for name in (
                "test_accuracy",
                "test_loss",
                "rho_positive_hat",
                "rho_negative_hat",
            ):
                self.assertTrue(np.isfinite(metrics[name]))
            self.assertEqual(metrics["reduction"], "batch_mean")

    def test_resume_rejects_backend_and_parameter_mismatch(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_experiment(config, run_dir)
            checkpoint = run_dir / "last.pt"
            changed = deepcopy(config)
            changed["posterior_stage"]["bandwidth"] = 3.5
            changed["trainer"]["epochs"] = 3
            with self.assertRaisesRegex(ValueError, "backend identity"):
                run_experiment(changed, resume=checkpoint)

            changed = deepcopy(config)
            changed["posterior_stage"]["name"] = "kde"
            changed["posterior_stage"].pop("max_centers")
            changed["posterior_stage"].pop("max_iterations")
            changed["posterior_stage"].pop("learning_rate")
            changed["posterior_stage"].pop("tolerance")
            changed["posterior_stage"].pop("epsilon")
            changed["posterior_stage"].pop("seed")
            changed["trainer"]["epochs"] = 3
            with self.assertRaisesRegex(ValueError, "backend identity"):
                run_experiment(changed, resume=checkpoint)

            changed = deepcopy(config)
            changed["data"]["dimension"] = 19
            changed["model"]["in_features"] = 19
            changed["trainer"]["epochs"] = 3
            with self.assertRaisesRegex(ValueError, "data_manifest|data identity"):
                run_experiment(changed, resume=checkpoint)

    def test_resume_rejects_checkpoint_backend_hash_corruption(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_experiment(config, run_dir)
            payload = read_checkpoint(run_dir / "last.pt", "cpu")
            payload["posterior_backend_hash"] = "f" * 64
            bad = run_dir / "bad-backend.pt"
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, "backend hash"):
                run_experiment(config, resume=bad)


if __name__ == "__main__":
    unittest.main()
