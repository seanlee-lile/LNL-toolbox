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
from torch.utils.data import DataLoader, Dataset
import yaml

from lnl_toolbox.algorithms.pcse import (
    PCSEAlgorithm,
    PCSEConfig,
    PCSEPhase,
    PCSEState,
)
from lnl_toolbox.algorithms.pcse.config import PCSEFeatureLayerConfig
from lnl_toolbox.algorithms.pcse.features import collect_pcse_features
from lnl_toolbox.algorithms.pcse.artifacts import (
    PCSEEnsembleArtifact,
    persist_npz_atomically,
)
from lnl_toolbox.data.multiclass_synthetic import (
    MulticlassTensorDataset,
    generate_synthetic_multiclass,
)
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.training.pcse_experiment import (
    _PCSEMultilayerPerceptron,
    run_pcse_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs/experiment/pcse_multiclass_smoke.yaml"


def _load_smoke_config() -> dict:
    return yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))


def _pretraining_algorithm(
    run_dir: Path,
    *,
    pretraining_epochs: int = 2,
    fixed_model: bool = False,
) -> PCSEAlgorithm:
    config = _load_smoke_config()
    config["pretraining_stage"]["epochs"] = pretraining_epochs
    torch.manual_seed(19)
    data = generate_synthetic_multiclass(
        90, 6, 3, 111, start_index=0, split="train"
    )
    validation = generate_synthetic_multiclass(
        30, 6, 3, 112, start_index=90, split="validation"
    )
    test = generate_synthetic_multiclass(
        30, 6, 3, 113, start_index=120, split="test"
    )
    train_set = MulticlassTensorDataset(data)
    validation_set = MulticlassTensorDataset(validation)
    test_set = MulticlassTensorDataset(test)
    train_loader = DataLoader(train_set, batch_size=30, shuffle=False)
    validation_loader = DataLoader(
        validation_set, batch_size=30, shuffle=False
    )
    model = (
        _FixedPCSEModel()
        if fixed_model
        else _PCSEMultilayerPerceptron(6, 12, 3)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0 if fixed_model else 0.01
    )
    return PCSEAlgorithm(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss=CrossEntropyLoss(),
        train_loader=train_loader,
        statistics_loader=train_loader,
        noisy_validation_loader=validation_loader,
        clean_test_loader=DataLoader(test_set, batch_size=30, shuffle=False),
        device=torch.device("cpu"),
        run_dir=run_dir,
        config=config,
        dataset="synthetic_multiclass",
        num_classes=3,
        noise_metadata={},
    )


class _FeatureDataset(Dataset):
    def __init__(self, clean_values: torch.Tensor) -> None:
        self.clean_values = clean_values
        self.inputs = torch.tensor(
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
        )
        self.targets = torch.tensor([0, 1, 2])
        self.indices = torch.tensor([30, 10, 20])

    def __len__(self) -> int:
        return 3

    def __getitem__(self, item: int):
        return {
            "input": self.inputs[item],
            "target": self.targets[item],
            "index": self.indices[item],
            "clean_target": self.clean_values[item],
        }


class _HookModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden1 = nn.Identity()
        self.hidden2 = nn.Linear(3, 3, bias=False)
        self.classifier = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.hidden2.weight.copy_(torch.eye(3))
            self.classifier.weight.copy_(torch.eye(3))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.hidden2(self.hidden1(values)))


class _FixedPCSEModel(nn.Module):
    """Near-oracle noisy-posterior fixture without reading clean targets."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden1 = nn.Identity()
        self.hidden2 = nn.Linear(6, 6, bias=False)
        self.classifier = nn.Linear(6, 3, bias=False)
        with torch.no_grad():
            self.hidden2.weight.copy_(torch.eye(6))
            self.classifier.weight.zero_()
            self.classifier.weight[:, :3].copy_(50.0 * torch.eye(3))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.hidden2(self.hidden1(values)))


class PCSEWorkflowTest(unittest.TestCase):
    def test_config_requires_multilayer_and_valid_backend(self) -> None:
        config = _load_smoke_config()
        parsed = PCSEConfig.from_mapping(config)
        self.assertEqual(parsed.transition_backend, "dual_t")
        self.assertEqual(
            tuple(layer.name for layer in parsed.feature_layers),
            ("hidden1", "hidden2"),
        )
        single = deepcopy(config)
        single["feature_stage"]["layers"] = [{"name": "hidden1"}]
        with self.assertRaisesRegex(ValueError, "at least two"):
            PCSEConfig.from_mapping(single)
        binary = deepcopy(config)
        binary["data"]["num_classes"] = 2
        with self.assertRaisesRegex(ValueError, "num_classes >= 3"):
            PCSEConfig.from_mapping(binary)
        paper_backend = deepcopy(config)
        paper_backend["transition_stage"] = {
            "name": "paper_volmin",
            "epochs": 2,
            "lambda_volume": 0.001,
            "optimizer": {
                "name": "adamw",
                "model_lr": 0.001,
                "transition_lr": 0.01,
                "weight_decay": 0.0,
            },
            "scheduler": {"name": "none"},
            "parameterization": {
                "name": "diagonal_dominant",
                "initial_flip_mass": 0.05,
                "max_flip_mass": 0.49,
                "temperature": 1.0,
                "seed": 3,
            },
            "determinant_tolerance": 1e-6,
            "condition_limit": 1e6,
        }
        parsed_paper = PCSEConfig.from_mapping(paper_backend)
        self.assertEqual(parsed_paper.transition_backend, "paper_volmin")
        self.assertEqual(
            parsed_paper.transition_backend_config["lambda_volume"], 0.001
        )

    def test_external_checkpoint_config_is_strict_and_train_mode_unchanged(self) -> None:
        config = _load_smoke_config()
        self.assertEqual(PCSEConfig.from_mapping(config).pretraining.mode, "train")
        external = deepcopy(config)
        external["pretraining_stage"]["mode"] = "external_checkpoint"
        external["pretraining_stage"]["epochs"] = 0
        model = dict(external["pretraining_stage"]["model"])
        external["pretraining_stage"]["source"] = {
            "adapter": "upm_main_best",
            "run_directory_env": "LNL_PCSE_SOURCE_RUN",
            "checkpoint_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "mapping_hash": "c" * 64,
            "dataset_fingerprint": "d" * 64,
            "model": model,
        }
        parsed = PCSEConfig.from_mapping(external)
        self.assertEqual(parsed.pretraining.mode, "external_checkpoint")
        invalid = deepcopy(external)
        invalid["pretraining_stage"]["source"]["adapter"] = "generic"
        with self.assertRaisesRegex(ValueError, "upm_main_best"):
            PCSEConfig.from_mapping(invalid)

    def test_external_adoption_persists_provenance_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _load_smoke_config()
            config["pretraining_stage"]["mode"] = "external_checkpoint"
            config["pretraining_stage"]["epochs"] = 0
            model = dict(config["pretraining_stage"]["model"])
            config["pretraining_stage"]["source"] = {
                "adapter": "upm_main_best",
                "run_directory_env": "PCSE_TEST_SOURCE",
                "checkpoint_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "mapping_hash": "c" * 64,
                "dataset_fingerprint": "d" * 64,
                "model": model,
            }
            source = {
                "adapter": "upm_main_best",
                "checkpoint": {"sha256": "a" * 64},
            }
            algorithm = _pretraining_algorithm(Path(directory))
            algorithm.config = config
            algorithm.method_config = PCSEConfig.from_mapping(config)
            algorithm.external_source_provenance = source
            algorithm.adopt_external_pretrained(
                completed_epochs=3,
                global_step=9,
                best_epoch=1,
                validation_accuracy=0.5,
                validation_loss=1.0,
            )
            self.assertEqual(algorithm.state.phase, PCSEPhase.PRETRAINED)
            checkpoint = torch.load(
                Path(directory) / "last.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["external_source_provenance"], source)
            algorithm.close()

    def test_phase_machine_rejects_illegal_transition(self) -> None:
        state = PCSEState()
        with self.assertRaisesRegex(ValueError, "illegal PCSE phase"):
            state.advance(PCSEPhase.TRANSITION_READY)

    def test_feature_extraction_preserves_layer_order_and_stable_indices(
        self,
    ) -> None:
        model = _HookModel()
        loader = DataLoader(
            _FeatureDataset(torch.tensor([2, 2, 2])),
            batch_size=2,
            shuffle=False,
        )
        result = collect_pcse_features(
            model,
            loader,
            "cpu",
            dataset="synthetic",
            split="train",
            layers=(
                PCSEFeatureLayerConfig("hidden2", "global_average"),
                PCSEFeatureLayerConfig("hidden1", "global_average"),
            ),
        )
        self.assertEqual(result.layer_names, ("hidden2", "hidden1"))
        np.testing.assert_array_equal(
            result.snapshots[0].global_indices, np.array([10, 20, 30])
        )
        np.testing.assert_array_equal(
            result.snapshots[0].global_indices,
            result.snapshots[1].global_indices,
        )
        self.assertEqual(len(model.hidden1._forward_hooks), 0)
        self.assertEqual(len(model.hidden2._forward_hooks), 0)

    def test_logits_cannot_be_used_as_hidden_features(self) -> None:
        model = _HookModel()
        loader = DataLoader(
            _FeatureDataset(torch.tensor([0, 1, 2])), batch_size=3
        )
        with self.assertRaisesRegex(ValueError, "model logits output"):
            collect_pcse_features(
                model,
                loader,
                "cpu",
                dataset="synthetic",
                split="train",
                layers=(
                    PCSEFeatureLayerConfig(
                        "classifier", "global_average"
                    ),
                    PCSEFeatureLayerConfig("hidden2", "global_average"),
                ),
            )
        self.assertEqual(len(model.classifier._forward_hooks), 0)

    def test_feature_extraction_does_not_read_clean_label_field(self) -> None:
        model = _HookModel()
        layers = (
            PCSEFeatureLayerConfig("hidden1", "global_average"),
            PCSEFeatureLayerConfig("hidden2", "global_average"),
        )
        first = collect_pcse_features(
            model,
            DataLoader(
                _FeatureDataset(torch.tensor([0, 1, 2])), batch_size=3
            ),
            "cpu",
            dataset="synthetic",
            split="train",
            layers=layers,
        )
        second = collect_pcse_features(
            model,
            DataLoader(
                _FeatureDataset(torch.tensor([2, 2, 2])), batch_size=3
            ),
            "cpu",
            dataset="synthetic",
            split="train",
            layers=layers,
        )
        for left, right in zip(first.snapshots, second.snapshots):
            np.testing.assert_array_equal(left.features, right.features)
            np.testing.assert_array_equal(
                left.noisy_targets, right.noisy_targets
            )

    def test_pretraining_interruption_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _pretraining_algorithm(Path(directory))
            first.train_pretraining(max_epochs=1)
            self.assertEqual(first.state.phase, PCSEPhase.PRETRAINING)
            self.assertEqual(first.state.pretraining_completed_epochs, 1)
            first.close()

            resumed = _pretraining_algorithm(Path(directory))
            resumed.resume(Path(directory) / "last.pt")
            self.assertEqual(resumed.state.pretraining_completed_epochs, 1)
            initial_step = resumed.state.pretraining_global_step
            resumed.train_pretraining()
            self.assertEqual(resumed.state.phase, PCSEPhase.PRETRAINED)
            self.assertEqual(resumed.state.pretraining_completed_epochs, 2)
            self.assertGreater(
                resumed.state.pretraining_global_step, initial_step
            )
            resumed.close()

    def test_transition_persistence_failure_does_not_advance_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            algorithm = _pretraining_algorithm(
                run_dir, pretraining_epochs=1, fixed_model=True
            )
            algorithm.train_pretraining()
            before_last = (run_dir / "last.pt").read_bytes()
            with mock.patch(
                "lnl_toolbox.algorithms.pcse.algorithm._persist_transition",
                side_effect=OSError("injected transition save failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    algorithm.estimate_transition()
            self.assertEqual(algorithm.state.phase, PCSEPhase.PRETRAINED)
            self.assertEqual(algorithm.state.posterior_snapshot_hash, "")
            self.assertEqual(algorithm.state.transition_artifact_hash, "")
            self.assertEqual((run_dir / "last.pt").read_bytes(), before_last)
            algorithm.close()

    def test_atomic_artifact_failure_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ensemble.npz"
            original = PCSEEnsembleArtifact(
                ("h1", "h2"), np.array([0.5, 0.5]), {"version": 1}
            )
            persist_npz_atomically(
                original, destination, PCSEEnsembleArtifact.load
            )
            before = destination.read_bytes()
            replacement = PCSEEnsembleArtifact(
                ("h1", "h2"), np.array([0.6, 0.4]), {"version": 2}
            )
            with self.assertRaisesRegex(ValueError, "injected validation"):
                persist_npz_atomically(
                    replacement,
                    destination,
                    lambda _path: (_ for _ in ()).throw(
                        ValueError("injected validation failure")
                    ),
                )
            self.assertEqual(destination.read_bytes(), before)

    def test_ensemble_interruption_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = _pretraining_algorithm(
                run_dir, pretraining_epochs=1, fixed_model=True
            )
            first.train_pretraining()
            first.estimate_transition()
            first.estimate_statistics()
            first.build_gda()
            first.start_ensemble_training()
            first.train_ensemble(max_epochs=1)
            self.assertEqual(first.state.phase, PCSEPhase.ENSEMBLE_TRAINING)
            self.assertEqual(first.state.ensemble_completed_epochs, 1)
            first.close()

            resumed = _pretraining_algorithm(
                run_dir, pretraining_epochs=1, fixed_model=True
            )
            resumed.resume(run_dir / "last.pt")
            self.assertEqual(resumed.state.ensemble_completed_epochs, 1)
            resumed.train_ensemble()
            self.assertEqual(resumed.state.phase, PCSEPhase.COMPLETED)
            self.assertEqual(resumed.state.ensemble_completed_epochs, 3)
            resumed.close()

    def test_two_layer_workflow_completed_resume_and_artifact_damage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            result = run_pcse_experiment(
                _load_smoke_config(), output_dir=run_dir
            )
            self.assertEqual(result, run_dir.resolve())
            final = json.loads(
                (run_dir / "final_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final["method"], "pcse")
            self.assertEqual(final["transition_backend"], "dual_t")
            self.assertEqual(len(final["ensemble_weights"]), 2)
            self.assertTrue(all(value > 0 for value in final["ensemble_weights"]))
            self.assertAlmostEqual(sum(final["ensemble_weights"]), 1.0)
            with np.load(
                run_dir / "posterior_snapshot.npz", allow_pickle=False
            ) as snapshot:
                posterior = snapshot["noisy_probabilities"]
            self.assertEqual(posterior.shape[1], 3)
            self.assertTrue(np.isfinite(posterior).all())
            np.testing.assert_allclose(
                posterior.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8
            )
            artifact_names = (
                "posterior_snapshot.npz",
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
                for name in artifact_names
            }
            run_pcse_experiment(
                _load_smoke_config(),
                resume=run_dir / "last.pt",
            )
            after = {
                name: (
                    (run_dir / name).read_bytes(),
                    (run_dir / name).stat().st_mtime_ns,
                )
                for name in artifact_names
            }
            self.assertEqual(before, after)

            transition_path = run_dir / "transition_artifact.npz"
            valid_transition = transition_path.read_bytes()
            transition_path.write_bytes(b"damaged")
            with self.assertRaises((ValueError, OSError)):
                run_pcse_experiment(
                    _load_smoke_config(),
                    resume=run_dir / "last.pt",
                )
            transition_path.write_bytes(valid_transition)

            mismatch = _load_smoke_config()
            mismatch["feature_stage"]["layers"].reverse()
            with self.assertRaisesRegex(ValueError, "method settings"):
                run_pcse_experiment(
                    mismatch,
                    resume=run_dir / "last.pt",
                )


if __name__ == "__main__":
    unittest.main()
