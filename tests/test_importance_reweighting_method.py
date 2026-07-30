from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.importance_reweighting import (
    ImportanceReweightingAlgorithm,
    ImportanceReweightingConfig,
    ImportanceReweightingPhase,
    ImportanceReweightingState,
    IndexedBinaryRCNWeightProvider,
    KDEBinaryNoisyPosteriorEstimator,
    NoiseRateArtifact,
    PaperRawMinNoiseRateEstimator,
    validate_binary_posterior_snapshot,
)
from lnl_toolbox.data.binary_synthetic import (
    BinaryTensorDataset,
    SyntheticBinaryData,
    generate_synthetic_binary_2d,
    validate_zero_one_labels,
)
from lnl_toolbox.noise.binary_rcn import (
    generate_binary_asymmetric_rcn,
    validate_binary_rcn_manifest,
)
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.noise.estimators import PosteriorSnapshot
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.experiment import run_experiment
from lnl_toolbox.treatments import SupervisedWeightInput


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment/importance_reweighting_binary_smoke.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


class BinaryMethodBoundaryTest(unittest.TestCase):
    def test_config_rejects_multiclass_and_wrong_model_output(self) -> None:
        for classes in (3, 10):
            config = load_config()
            config["num_classes"] = classes
            with self.assertRaisesRegex(ValueError, "num_classes"):
                ImportanceReweightingConfig.from_mapping(config)
        config = load_config()
        config["model"]["num_classes"] = 3
        with self.assertRaisesRegex(ValueError, "2 classes"):
            ImportanceReweightingConfig.from_mapping(config)

    def test_label_validation_rejects_nonbinary_and_single_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            validate_zero_one_labels(np.array([0, 2]), owner="test")
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            validate_zero_one_labels(
                np.zeros(3, dtype=np.int64),
                owner="test",
                require_both_classes=True,
            )

    def test_dataset_requires_two_features(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[N, 2\]"):
            SyntheticBinaryData(
                np.zeros((4, 3)),
                np.array([0, 1, 0, 1]),
                np.arange(4),
                "train",
            )

    def test_training_dataset_exposes_no_clean_truth(self) -> None:
        data = generate_synthetic_binary_2d(4, 3, split="train")
        item = BinaryTensorDataset(data, 1 - data.labels)[0]
        self.assertEqual(set(item), {"input", "target", "index"})
        self.assertNotIn("clean_target", item)

    def test_manifest_binary_shape_and_identity(self) -> None:
        data = generate_synthetic_binary_2d(20, 1, split="train")
        manifest = generate_binary_asymmetric_rcn(
            data.labels, data.global_indices,
            rho_positive=0.2, rho_negative=0.1, seed=2,
        )
        validate_binary_rcn_manifest(manifest)
        manifest.transition_matrix = np.eye(3)
        with self.assertRaisesRegex(ValueError, r"\[2, 2\]"):
            validate_binary_rcn_manifest(manifest)
        three_class = NoiseManifest(
            dataset="fixture",
            noise_type="binary_asymmetric_rcn",
            seed=1,
            requested_rate=0.1,
            clean_targets=np.array([0, 1, 2]),
            noisy_targets=np.array([0, 1, 2]),
            transition_matrix=np.eye(3),
            metadata={
                "rho_positive": 0.2,
                "rho_negative": 0.1,
                "label_convention": "zero_one",
            },
            num_classes=3,
        )
        with self.assertRaisesRegex(ValueError, "num_classes"):
            validate_binary_rcn_manifest(three_class)

    def test_three_class_snapshot_is_rejected(self) -> None:
        snapshot = PosteriorSnapshot(
            np.full((3, 3), 1 / 3),
            np.array([0, 1, 2]),
            np.arange(3),
            "fixture",
            "train",
        )
        with self.assertRaisesRegex(ValueError, r"\[N, 2\]"):
            validate_binary_posterior_snapshot(snapshot)


class EstimationAndWeightLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = PosteriorSnapshot(
            noisy_probabilities=np.array([
                [0.8, 0.2],
                [0.3, 0.7],
                [0.6, 0.4],
            ]),
            noisy_targets=np.array([0, 1, 0]),
            global_indices=np.array([10, 30, 20]),
            dataset="synthetic_binary_2d",
            split="train",
        )
        self.rates = PaperRawMinNoiseRateEstimator().estimate(self.snapshot)

    def test_raw_min_uses_paper_direction(self) -> None:
        self.assertAlmostEqual(self.rates.rho_positive, 0.3)
        self.assertAlmostEqual(self.rates.rho_negative, 0.2)
        self.assertEqual(self.rates.positive_extreme_global_index, 30)
        self.assertEqual(self.rates.negative_extreme_global_index, 10)

    def test_lookup_uses_stable_index_and_detached_posterior(self) -> None:
        provider = IndexedBinaryRCNWeightProvider(self.snapshot, self.rates)
        result = provider.compute(SupervisedWeightInput(
            logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            noisy_targets=torch.tensor([0, 1]),
            sample_indices=torch.tensor([20, 30]),
            per_sample_loss=torch.ones(2),
        ))
        self.assertEqual(result.sample_weights.shape, (2,))
        self.assertIs(result.sample_weights.requires_grad, False)
        self.assertTrue(torch.isfinite(result.sample_weights).all())

    def test_lookup_rejects_bad_logits_missing_duplicate_and_misalignment(self) -> None:
        provider = IndexedBinaryRCNWeightProvider(self.snapshot, self.rates)
        base = dict(
            logits=torch.zeros(2, 2),
            noisy_targets=torch.tensor([0, 1]),
            sample_indices=torch.tensor([10, 30]),
            per_sample_loss=torch.ones(2),
        )
        for shape in ((2, 1), (2, 3)):
            values = dict(base)
            values["logits"] = torch.zeros(shape)
            with self.assertRaisesRegex(ValueError, r"\[B, 2\]"):
                provider.compute(SupervisedWeightInput(**values))
        values = dict(base)
        values["sample_indices"] = torch.tensor([10, 99])
        with self.assertRaisesRegex(ValueError, "missing"):
            provider.compute(SupervisedWeightInput(**values))
        values["sample_indices"] = torch.tensor([10, 10])
        with self.assertRaisesRegex(ValueError, "unique"):
            provider.compute(SupervisedWeightInput(**values))
        values["sample_indices"] = torch.tensor([10, 30])
        values["noisy_targets"] = torch.tensor([1, 1])
        with self.assertRaisesRegex(ValueError, "align"):
            provider.compute(SupervisedWeightInput(**values))

    def test_rate_artifact_rejects_bad_sum_and_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum"):
            NoiseRateArtifact(
                0.6, 0.4, 1, 2, "a" * 64,
                "synthetic_binary_2d", "train",
            )
        wrong = NoiseRateArtifact(
            0.2, 0.1, 1, 2, "b" * 64,
            "synthetic_binary_2d", "train",
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            IndexedBinaryRCNWeightProvider(self.snapshot, wrong)

    def test_kde_snapshot_is_binary_and_index_aligned(self) -> None:
        data = generate_synthetic_binary_2d(30, 7, split="train")
        snapshot = KDEBinaryNoisyPosteriorEstimator(0.2).fit_predict(
            data.features, data.labels, data.global_indices,
            dataset=data.dataset, split="train",
        )
        self.assertEqual(snapshot.noisy_probabilities.shape, (30, 2))
        np.testing.assert_array_equal(
            snapshot.global_indices, np.sort(data.global_indices)
        )


class StateAndSmokeTest(unittest.TestCase):
    @staticmethod
    def _assert_no_pending_artifacts(run_dir: Path) -> None:
        pending = list(run_dir.glob(".*.pending.npz"))
        if pending:
            raise AssertionError(f"temporary artifacts were not cleaned: {pending}")

    def test_method_forces_paper_batch_mean_reduction(self) -> None:
        self.assertEqual(
            ImportanceReweightingAlgorithm.FINAL_REDUCTION.normalization,
            "batch_mean",
        )
        self.assertNotEqual(
            ImportanceReweightingAlgorithm.FINAL_REDUCTION.normalization,
            "weight_sum_mean",
        )

    def test_phase_transitions_and_extension(self) -> None:
        state = ImportanceReweightingState()
        with self.assertRaisesRegex(ValueError, "illegal"):
            state.advance(ImportanceReweightingPhase.RATE_READY)
        state.posterior_snapshot_hash = "a" * 64
        state.advance(ImportanceReweightingPhase.POSTERIOR_READY)
        state.noise_rate_artifact_hash = "b" * 64
        state.advance(ImportanceReweightingPhase.RATE_READY)
        state.advance(ImportanceReweightingPhase.FINAL_TRAINING)
        state.final_completed_epochs = 2
        state.best_final_epoch = 0
        state.advance(ImportanceReweightingPhase.COMPLETED)
        state.reopen_final_training(3)
        self.assertEqual(
            state.phase, ImportanceReweightingPhase.FINAL_TRAINING
        )

    def test_tiny_smoke_and_resume_preserve_artifacts(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            result = run_experiment(config, run_dir)
            snapshot = result / "posterior_snapshot.npz"
            rates = result / "noise_rate_artifact.npz"
            before_snapshot = snapshot.read_bytes()
            before_rates = rates.read_bytes()
            first = read_checkpoint(result / "last.pt", "cpu")
            self.assertEqual(
                first["method_state"]["final_completed_epochs"], 2
            )
            self.assertEqual(first["method_state"]["final_global_step"], 8)
            self.assertEqual(
                first["method_state"]["phase"], "completed"
            )
            self.assertEqual(
                read_checkpoint(result / "last.pt", "cpu")["config"]["num_classes"],
                2,
            )
            config["trainer"]["epochs"] = 3
            run_experiment(config, resume=result / "last.pt")
            final = read_checkpoint(result / "last.pt", "cpu")
            self.assertEqual(
                final["method_state"]["final_completed_epochs"], 3
            )
            self.assertEqual(final["method_state"]["final_global_step"], 12)
            self.assertEqual(snapshot.read_bytes(), before_snapshot)
            self.assertEqual(rates.read_bytes(), before_rates)
            self.assertEqual(final["posterior_snapshot_hash"],
                             first["posterior_snapshot_hash"])
            self.assertEqual(final["noise_rate_artifact_hash"],
                             first["noise_rate_artifact_hash"])
            self.assertEqual(final["config"]["trainer"]["epochs"], 3)
            metrics = yaml.safe_load(
                (result / "final_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["reduction"], "batch_mean")
            self.assertTrue(np.isfinite(metrics["test_loss"]))

    def test_resume_rejects_class_and_convention_corruption(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_experiment(config, run_dir)
            original = read_checkpoint(run_dir / "last.pt", "cpu")
            for field, value, pattern in (
                ("num_classes", 3, "num_classes"),
                ("label_convention", "minus_plus", "convention"),
            ):
                corrupted = deepcopy(original)
                corrupted[field] = value
                path = run_dir / f"bad-{field}.pt"
                torch.save(corrupted, path)
                with self.assertRaisesRegex(ValueError, pattern):
                    run_experiment(config, resume=path)

    def test_resume_rejects_rate_artifact_provenance_mismatch(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_experiment(config, run_dir)
            payload = read_checkpoint(run_dir / "last.pt", "cpu")
            original = NoiseRateArtifact.load(
                run_dir / "noise_rate_artifact.npz"
            )
            wrong = NoiseRateArtifact(
                rho_positive=original.rho_positive,
                rho_negative=original.rho_negative,
                positive_extreme_global_index=(
                    original.positive_extreme_global_index
                ),
                negative_extreme_global_index=(
                    original.negative_extreme_global_index
                ),
                source_snapshot_hash="f" * 64,
                dataset=original.dataset,
                split=original.split,
            )
            wrong.save(run_dir / "noise_rate_artifact.npz")
            payload["noise_rate_artifact_hash"] = wrong.artifact_hash
            payload["method_state"]["noise_rate_artifact_hash"] = (
                wrong.artifact_hash
            )
            bad = run_dir / "bad-provenance.pt"
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, "source snapshot"):
                run_experiment(config, resume=bad)

    def test_posterior_temporary_write_failure_preserves_formal_path(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            formal = run_dir / "posterior_snapshot.npz"
            formal.write_bytes(b"previous-valid-placeholder")
            with mock.patch.object(
                PosteriorSnapshot,
                "save",
                side_effect=OSError("simulated posterior write failure"),
            ):
                with self.assertRaisesRegex(OSError, "write failure"):
                    run_experiment(config, run_dir)
            self.assertEqual(formal.read_bytes(), b"previous-valid-placeholder")
            self.assertFalse((run_dir / "last.pt").exists())
            self._assert_no_pending_artifacts(run_dir)

    def test_posterior_temporary_reload_failure_preserves_formal_path(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            formal = run_dir / "posterior_snapshot.npz"
            formal.write_bytes(b"previous-valid-placeholder")
            with mock.patch.object(
                PosteriorSnapshot,
                "load",
                side_effect=ValueError("simulated posterior validation failure"),
            ):
                with self.assertRaisesRegex(ValueError, "validation failure"):
                    run_experiment(config, run_dir)
            self.assertEqual(formal.read_bytes(), b"previous-valid-placeholder")
            self.assertFalse((run_dir / "last.pt").exists())
            self._assert_no_pending_artifacts(run_dir)

    def test_rate_temporary_write_failure_keeps_posterior_ready_checkpoint(
        self,
    ) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            formal = run_dir / "noise_rate_artifact.npz"
            formal.write_bytes(b"previous-valid-placeholder")
            with mock.patch.object(
                NoiseRateArtifact,
                "save",
                side_effect=OSError("simulated rate write failure"),
            ):
                with self.assertRaisesRegex(OSError, "write failure"):
                    run_experiment(config, run_dir)
            checkpoint = read_checkpoint(run_dir / "last.pt", "cpu")
            self.assertEqual(
                checkpoint["method_state"]["phase"], "posterior_ready"
            )
            self.assertEqual(
                checkpoint["method_state"]["noise_rate_artifact_hash"], ""
            )
            self.assertEqual(formal.read_bytes(), b"previous-valid-placeholder")
            self._assert_no_pending_artifacts(run_dir)

    def test_rate_temporary_reload_failure_keeps_posterior_ready_checkpoint(
        self,
    ) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            formal = run_dir / "noise_rate_artifact.npz"
            formal.write_bytes(b"previous-valid-placeholder")
            with mock.patch.object(
                NoiseRateArtifact,
                "load",
                side_effect=ValueError("simulated rate validation failure"),
            ):
                with self.assertRaisesRegex(ValueError, "validation failure"):
                    run_experiment(config, run_dir)
            checkpoint = read_checkpoint(run_dir / "last.pt", "cpu")
            self.assertEqual(
                checkpoint["method_state"]["phase"], "posterior_ready"
            )
            self.assertEqual(
                checkpoint["method_state"]["noise_rate_artifact_hash"], ""
            )
            self.assertEqual(formal.read_bytes(), b"previous-valid-placeholder")
            self._assert_no_pending_artifacts(run_dir)

    def test_checkpoint_is_written_only_after_formal_artifact_is_valid(
        self,
    ) -> None:
        config = load_config()
        observations: list[tuple[str, bool]] = []
        original = ImportanceReweightingAlgorithm._save_last

        def record_then_save(owner: ImportanceReweightingAlgorithm) -> None:
            phase = owner.state.phase.value
            if phase == "posterior_ready":
                loaded = PosteriorSnapshot.load(owner.snapshot_path)
                valid = (
                    loaded.snapshot_hash
                    == owner.state.posterior_snapshot_hash
                )
            elif phase == "rate_ready":
                loaded_rate = NoiseRateArtifact.load(owner.rate_path)
                valid = (
                    loaded_rate.artifact_hash
                    == owner.state.noise_rate_artifact_hash
                )
            else:
                valid = True
            observations.append((phase, valid))
            original(owner)

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                ImportanceReweightingAlgorithm,
                "_save_last",
                record_then_save,
            ):
                run_experiment(config, Path(temporary) / "run")
        self.assertIn(("posterior_ready", True), observations)
        self.assertIn(("rate_ready", True), observations)

    def test_resume_rejects_corrupted_formal_snapshot_without_rebuilding(
        self,
    ) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_experiment(config, run_dir)
            snapshot = run_dir / "posterior_snapshot.npz"
            snapshot.write_bytes(b"corrupted")
            corrupted = snapshot.read_bytes()
            with self.assertRaises(ValueError):
                run_experiment(config, resume=run_dir / "last.pt")
            self.assertEqual(snapshot.read_bytes(), corrupted)


if __name__ == "__main__":
    unittest.main()
