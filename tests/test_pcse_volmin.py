from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.algorithms.pcse import PCSEAlgorithm, PCSEPhase
from lnl_toolbox.algorithms.pcse.volmin import (
    DiagonallyDominantTransition,
    build_volmin_optimizer,
    validate_trainable_transition,
    volmin_objective,
)
from lnl_toolbox.data.multiclass_synthetic import (
    MulticlassTensorDataset,
    generate_synthetic_multiclass,
)
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.noise.transition import TransitionArtifact
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.pcse_experiment import (
    _PCSEMultilayerPerceptron,
    run_pcse_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
VOLMIN_CONFIG = (
    ROOT / "configs/experiment/pcse_multiclass_volmin_smoke.yaml"
)


def _config() -> dict:
    return yaml.safe_load(VOLMIN_CONFIG.read_text(encoding="utf-8"))


def _algorithm(run_dir: Path) -> PCSEAlgorithm:
    config = _config()
    config["pretraining_stage"]["epochs"] = 1
    config["transition_stage"]["epochs"] = 2
    config["ensemble_stage"]["epochs"] = 1
    train = generate_synthetic_multiclass(
        90, 6, 3, 501, start_index=0, split="train"
    )
    validation = generate_synthetic_multiclass(
        30, 6, 3, 502, start_index=90, split="validation"
    )
    test = generate_synthetic_multiclass(
        30, 6, 3, 503, start_index=120, split="test"
    )
    train_loader = DataLoader(
        MulticlassTensorDataset(train),
        batch_size=30,
        shuffle=False,
    )
    validation_loader = DataLoader(
        MulticlassTensorDataset(validation),
        batch_size=30,
        shuffle=False,
    )
    torch.manual_seed(47)
    model = _PCSEMultilayerPerceptron(6, 12, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return PCSEAlgorithm(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss=CrossEntropyLoss(),
        train_loader=train_loader,
        statistics_loader=train_loader,
        noisy_validation_loader=validation_loader,
        clean_test_loader=DataLoader(
            MulticlassTensorDataset(test), batch_size=30, shuffle=False
        ),
        device=torch.device("cpu"),
        run_dir=run_dir,
        config=config,
        dataset="synthetic_multiclass",
        num_classes=3,
        noise_metadata={},
    )


class PCSEVolMinPrimitiveTest(unittest.TestCase):
    def test_parameterization_is_safe_deterministic_and_row_stochastic(
        self,
    ) -> None:
        first = DiagonallyDominantTransition(
            3,
            initial_flip_mass=0.1,
            max_flip_mass=0.49,
            temperature=1.0,
            seed=7,
        )
        second = DiagonallyDominantTransition(
            3,
            initial_flip_mass=0.1,
            max_flip_mass=0.49,
            temperature=1.0,
            seed=7,
        )
        transition = first.matrix()
        torch.testing.assert_close(transition, second.matrix())
        self.assertTrue(bool((transition >= 0.0).all()))
        torch.testing.assert_close(
            transition.sum(dim=1), torch.ones(3, dtype=torch.float64)
        )
        self.assertTrue(bool((torch.diagonal(transition) > 0.5).all()))
        sign, _ = torch.linalg.slogdet(transition)
        self.assertGreater(float(sign), 0.0)
        validate_trainable_transition(
            transition,
            determinant_tolerance=1e-8,
            condition_limit=1e6,
        )

    def test_asymmetric_clean_to_noisy_direction_and_objective_sign(
        self,
    ) -> None:
        clean = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float64
        )
        transition = torch.tensor(
            [
                [0.75, 0.20, 0.05],
                [0.10, 0.80, 0.10],
                [0.15, 0.05, 0.80],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        targets = torch.tensor([1, 2])
        objective, metrics = volmin_objective(
            torch.log(clean),
            targets,
            transition,
            lambda_volume=0.2,
            determinant_tolerance=1e-8,
            condition_limit=1e6,
        )
        noisy = clean @ transition
        expected_nll = -torch.log(noisy[torch.arange(2), targets]).mean()
        expected = expected_nll + 0.2 * torch.logdet(transition)
        torch.testing.assert_close(objective, expected)
        self.assertAlmostEqual(
            metrics["classification_loss"], float(expected_nll), places=10
        )

    def test_model_and_transition_receive_finite_gradients_and_update(
        self,
    ) -> None:
        torch.manual_seed(13)
        model = nn.Linear(4, 3)
        transition_model = DiagonallyDominantTransition(
            3,
            initial_flip_mass=0.05,
            max_flip_mass=0.49,
            temperature=1.0,
            seed=17,
        )
        optimizer = build_volmin_optimizer(
            model,
            transition_model,
            {
                "name": "adamw",
                "model_lr": 0.01,
                "transition_lr": 0.02,
                "weight_decay": 0.0,
            },
        )
        model_before = model.weight.detach().clone()
        transition_before = transition_model.matrix().detach().clone()
        objective, _ = volmin_objective(
            model(torch.randn(8, 4)).to(torch.float64),
            torch.tensor([0, 1, 2, 0, 1, 2, 1, 0]),
            transition_model.matrix(),
            lambda_volume=0.001,
            determinant_tolerance=1e-8,
            condition_limit=1e6,
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        parameters = list(model.parameters()) + list(
            transition_model.parameters()
        )
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))
        self.assertTrue(
            all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
        )
        optimizer.step()
        self.assertFalse(torch.equal(model_before, model.weight.detach()))
        self.assertFalse(
            torch.equal(transition_before, transition_model.matrix().detach())
        )

    def test_illegal_singular_or_wrong_sign_transition_fails(self) -> None:
        singular = torch.tensor(
            [
                [0.6, 0.2, 0.2],
                [0.6, 0.2, 0.2],
                [0.1, 0.1, 0.8],
            ],
            dtype=torch.float64,
        )
        with self.assertRaisesRegex(
            ValueError, "diagonally dominant|singular"
        ):
            validate_trainable_transition(
                singular,
                determinant_tolerance=1e-8,
                condition_limit=1e6,
            )
        negative_sign = torch.tensor(
            [
                [0.1, 0.8, 0.1],
                [0.8, 0.1, 0.1],
                [0.1, 0.1, 0.8],
            ],
            dtype=torch.float64,
        )
        with self.assertRaisesRegex(
            ValueError, "diagonally dominant|determinant"
        ):
            validate_trainable_transition(
                negative_sign,
                determinant_tolerance=1e-8,
                condition_limit=1e6,
            )


class PCSEVolMinWorkflowTest(unittest.TestCase):
    def test_transition_training_interruption_resume_and_model_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = _algorithm(run_dir)
            first.train_pretraining()
            source_hash = first.state.pretrained_checkpoint_sha256
            first.start_transition_training()
            first.train_transition(max_epochs=1)
            self.assertEqual(
                first.state.phase, PCSEPhase.TRANSITION_TRAINING
            )
            self.assertEqual(first.state.transition_completed_epochs, 1)
            initial_step = first.state.transition_global_step
            interrupted_payload = read_checkpoint(
                run_dir / "last.pt", "cpu"
            )
            self.assertIsInstance(
                interrupted_payload["transition_model"], dict
            )
            self.assertIsInstance(
                interrupted_payload["transition_optimizer"], dict
            )
            self.assertIn("rng_state", interrupted_payload)
            first.close()

            resumed = _algorithm(run_dir)
            resumed.resume(run_dir / "last.pt")
            self.assertEqual(resumed.state.transition_completed_epochs, 1)
            self.assertTrue(
                bool(resumed.transition_optimizer.state_dict()["state"])
            )
            resumed.train_transition()
            self.assertEqual(resumed.state.phase, PCSEPhase.TRANSITION_READY)
            self.assertGreater(resumed.state.transition_global_step, initial_step)
            self.assertEqual(
                resumed.transition.metadata[
                    "source_pretrained_checkpoint_sha256"
                ],
                source_hash,
            )
            self.assertEqual(
                resumed.transition.metadata[
                    "feature_model_checkpoint_sha256"
                ],
                resumed.state.volmin_final_checkpoint_sha256,
            )
            self.assertNotEqual(
                resumed.state.feature_model_checkpoint_sha256, source_hash
            )
            statistics = resumed.estimate_statistics()
            self.assertEqual(
                statistics.provenance["feature_model_checkpoint_sha256"],
                resumed.state.volmin_final_checkpoint_sha256,
            )
            resumed.close()

    def test_publish_validation_failure_preserves_formal_files_and_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            algorithm = _algorithm(run_dir)
            algorithm.train_pretraining()
            algorithm.start_transition_training()
            final_path = run_dir / "volmin_final.pt"
            transition_path = run_dir / "transition_artifact.npz"
            final_path.write_bytes(b"existing-final")
            transition_path.write_bytes(b"existing-transition")
            previous_final = final_path.read_bytes()
            previous_transition = transition_path.read_bytes()
            before_last = (run_dir / "last.pt").read_bytes()
            with mock.patch.object(
                TransitionArtifact,
                "load",
                side_effect=ValueError("injected transition validation"),
            ):
                with self.assertRaisesRegex(ValueError, "injected"):
                    algorithm.train_transition()
            self.assertEqual(
                algorithm.state.phase, PCSEPhase.TRANSITION_TRAINING
            )
            self.assertEqual(
                algorithm.state.volmin_final_checkpoint_sha256, ""
            )
            self.assertEqual(algorithm.state.transition_artifact_hash, "")
            self.assertEqual(final_path.read_bytes(), previous_final)
            self.assertEqual(
                transition_path.read_bytes(), previous_transition
            )
            self.assertNotEqual((run_dir / "last.pt").read_bytes(), before_last)
            algorithm.close()

    def test_tiny_workflow_completed_resume_keeps_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            config = _config()
            run_pcse_experiment(config, output_dir=run_dir)
            final = json.loads(
                (run_dir / "final_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final["transition_backend"], "paper_volmin")
            names = (
                "volmin_final.pt",
                "transition_artifact.npz",
                "pcse_statistics.npz",
                "pcse_gda.npz",
                "pcse_ensemble.npz",
            )
            before = {
                name: (
                    (run_dir / name).read_bytes(),
                    (run_dir / name).stat().st_mtime_ns,
                )
                for name in names
            }
            run_pcse_experiment(config, resume=run_dir / "last.pt")
            after = {
                name: (
                    (run_dir / name).read_bytes(),
                    (run_dir / name).stat().st_mtime_ns,
                )
                for name in names
            }
            self.assertEqual(before, after)

            mismatch = deepcopy(config)
            mismatch["transition_stage"]["lambda_volume"] *= 2.0
            with self.assertRaisesRegex(ValueError, "method settings"):
                run_pcse_experiment(
                    mismatch, resume=run_dir / "last.pt"
                )


if __name__ == "__main__":
    unittest.main()
