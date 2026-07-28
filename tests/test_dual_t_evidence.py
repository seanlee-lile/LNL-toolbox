from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lnl_toolbox.algorithms.dual_t.evidence import (
    build_transition_evidence,
    realized_empirical_transition,
    transition_matrix_error,
)
from lnl_toolbox.noise.estimators import (
    DualTransitionEstimator,
    PosteriorSnapshot,
)
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.runtime import seed_everything
from lnl_toolbox.training.checkpoint import capture_rng_state
from lnl_toolbox.training.dual_t_evidence_experiment import (
    _run_final_arm,
    _tensor_state_hash,
    run_dual_t_evidence_experiment,
)
from lnl_toolbox.training.experiment import build_model


class _FixedImageDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(17)
        self.inputs = torch.rand(12, 3, 32, 32, generator=generator)
        self.targets = torch.arange(12) % 3

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, item: int):
        return {
            "input": self.inputs[item],
            "target": self.targets[item],
            "index": torch.tensor(100 + item * 5),
        }


class DualTEvidenceContractTest(unittest.TestCase):
    @staticmethod
    def _snapshot() -> PosteriorSnapshot:
        return PosteriorSnapshot(
            noisy_probabilities=np.asarray([
                [0.90, 0.05, 0.05],
                [0.60, 0.20, 0.20],
                [0.10, 0.80, 0.10],
                [0.20, 0.60, 0.20],
                [0.05, 0.05, 0.90],
                [0.20, 0.20, 0.60],
            ]),
            noisy_targets=np.asarray([0, 1, 1, 2, 2, 0]),
            global_indices=np.asarray([30, 20, 10, 40, 50, 60]),
            dataset="fixture",
            split="train",
        )

    @staticmethod
    def _manifest(transition: np.ndarray) -> NoiseManifest:
        return NoiseManifest(
            dataset="fixture",
            noise_type="synthetic_fixture",
            seed=9,
            requested_rate=0.25,
            clean_targets=np.asarray([0, 0, 1, 1, 2, 2]),
            noisy_targets=np.asarray([0, 1, 1, 2, 2, 0]),
            transition_matrix=transition,
            num_classes=3,
            global_indices=np.asarray([10, 20, 30, 40, 50, 60]),
        )

    def test_matrix_error_matches_elementwise_hand_calculation(self) -> None:
        truth = np.asarray([[0.8, 0.2], [0.1, 0.9]])
        estimate = np.asarray([[0.7, 0.3], [0.2, 0.8]])
        error = transition_matrix_error(estimate, truth)
        self.assertAlmostEqual(error.l1_total, 0.4)
        self.assertAlmostEqual(error.l1_mean, 0.1)
        self.assertAlmostEqual(error.max_absolute_error, 0.1)
        np.testing.assert_allclose(error.row_l1, (0.2, 0.2))

    def test_ground_truth_is_the_manifest_matrix_and_snapshot_is_shared(
        self,
    ) -> None:
        snapshot = self._snapshot()
        dual = DualTransitionEstimator().estimate(snapshot)
        manifest = self._manifest(dual.matrix)
        evidence = build_transition_evidence(
            snapshot=snapshot,
            manifest=manifest,
        )
        np.testing.assert_array_equal(
            evidence.ground_truth_matrix,
            manifest.transition_matrix,
        )
        self.assertEqual(
            evidence.anchor_artifact.source_snapshot_hash,
            snapshot.snapshot_hash,
        )
        self.assertEqual(
            evidence.dual_t_artifact.source_snapshot_hash,
            snapshot.snapshot_hash,
        )
        self.assertEqual(evidence.dual_t_error.l1_total, 0.0)
        self.assertGreater(evidence.anchor_error.l1_total, 0.0)

    def test_manifest_without_ground_truth_transition_is_rejected(self) -> None:
        manifest = self._manifest(np.eye(3))
        manifest.transition_matrix = None
        with self.assertRaisesRegex(
            ValueError,
            "NoiseManifest.transition_matrix",
        ):
            build_transition_evidence(
                snapshot=self._snapshot(),
                manifest=manifest,
            )

    def test_realized_matrix_is_an_offline_index_aligned_diagnostic(
        self,
    ) -> None:
        manifest = self._manifest(np.eye(3))
        realized = realized_empirical_transition(
            manifest,
            np.asarray([60, 10, 50, 20, 40, 30]),
        )
        expected = np.asarray([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ])
        np.testing.assert_allclose(realized, expected)

    def test_runner_collects_snapshot_explicitly_without_estimate_transition(
        self,
    ) -> None:
        source = inspect.getsource(run_dual_t_evidence_experiment)
        self.assertIn("collect_posterior_snapshot(", source)
        self.assertIn("posterior_best_path", source)
        self.assertNotIn(".estimate_transition(", source)

    def test_final_arm_contract_has_no_ground_truth_or_clean_targets(
        self,
    ) -> None:
        parameters = inspect.signature(_run_final_arm).parameters
        self.assertNotIn("manifest", parameters)
        self.assertNotIn("ground_truth", parameters)
        self.assertNotIn("clean_targets", parameters)


class DualTEvidenceFairnessTest(unittest.TestCase):
    def test_sequential_arms_share_initial_state_order_and_input_tensors(
        self,
    ) -> None:
        seed_everything(23)
        model_config = {"name": "tiny_cnn", "width": 4}
        reference = build_model(model_config, 3)
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in reference.state_dict().items()
        }
        initial_hash = _tensor_state_hash(initial_state)
        rng_state = capture_rng_state()
        dataset = _FixedImageDataset()
        arguments = {
            "initial_state": initial_state,
            "model_config": model_config,
            "optimizer_config": {
                "name": "sgd",
                "lr": 0.01,
                "momentum": 0.0,
            },
            "scheduler_config": {"name": "none"},
            "epochs": 2,
            "num_classes": 3,
            "train_dataset": dataset,
            "noisy_validation_dataset": dataset,
            "clean_test_dataset": dataset,
            "loader_config": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
            },
            "sampler_seed": 101,
            "rng_state": rng_state,
            "device": torch.device("cpu"),
            "transition": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            first = _run_final_arm(
                name="first",
                run_dir=Path(directory) / "first",
                **arguments,
            )
            second = _run_final_arm(
                name="second",
                run_dir=Path(directory) / "second",
                **arguments,
            )
        self.assertEqual(first.initial_state_hash, initial_hash)
        self.assertEqual(second.initial_state_hash, initial_hash)
        self.assertEqual(first.sampler_seed, second.sampler_seed)
        self.assertEqual(first.batch_index_hashes, second.batch_index_hashes)
        self.assertEqual(first.input_tensor_hashes, second.input_tensor_hashes)
        self.assertEqual(len(first.batch_index_hashes), 2)


if __name__ == "__main__":
    unittest.main()
