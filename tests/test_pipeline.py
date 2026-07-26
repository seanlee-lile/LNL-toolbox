from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.algorithms.transition_risk import ForwardRiskCorrector
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.noise import KnownTransition
from lnl_toolbox.plugins.builtin import build_builtin_pipeline
from lnl_toolbox.treatments import WeightInput, WeightResult
from lnl_toolbox.training.pipeline import StandardNoisyERMPipeline


class _WeightProvider:
    def __init__(self) -> None:
        self.inputs: list[WeightInput] = []

    def compute(self, weight_input: WeightInput) -> WeightResult:
        self.inputs.append(weight_input)
        return WeightResult(
            sample_weights=torch.full(
                (weight_input.noisy_targets.numel(),),
                0.5,
                device=weight_input.noisy_targets.device,
            ),
            metrics={"provider_seen": 1.0},
        )


class _IndexedModel(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.eye(classes) * 5.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits[inputs.view(-1).long()]


class PipelineIntegrationTest(unittest.TestCase):
    def test_supervised_algorithm_composes_risk_and_weight_provider(self) -> None:
        model = nn.Linear(2, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        provider = _WeightProvider()
        algorithm = SupervisedClassificationAlgorithm(
            model,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            risk_corrector=ForwardRiskCorrector(),
            transition=KnownTransition(np.eye(2)),
            weight_provider=provider,
        )
        algorithm.setup(ExperimentContext(Path(".")))
        result = algorithm.step(
            Batch({
                "input": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "target": torch.tensor([0, 1]),
                "index": torch.tensor([4, 9]),
            }),
            RunState(phase="train"),
        )
        self.assertEqual(len(provider.inputs), 1)
        self.assertTrue(torch.equal(provider.inputs[0].sample_indices, torch.tensor([4, 9])))
        self.assertEqual(result.metrics["selected_samples"], 2.0)
        self.assertEqual(result.metrics["treatment_provider_seen"], 1.0)

    def test_dual_t_pipeline_persists_snapshot_and_transition(self) -> None:
        classes = 4
        model = _IndexedModel(classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.arange(classes).view(-1, 1)
        loader = DataLoader(
            [
                {"input": inputs[i], "target": torch.tensor(i), "index": torch.tensor(i)}
                for i in range(classes)
            ],
            batch_size=2,
            shuffle=False,
        )
        pipeline = build_builtin_pipeline({
            "name": "standard_noisy_erm",
            "warmup_epochs": 0,
            "transition_estimator": {"name": "dual_t"},
            "risk_corrector": {"name": "forward"},
        })
        self.assertIsInstance(pipeline, StandardNoisyERMPipeline)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = pipeline.prepare_transition(
                model=model,
                optimizer=optimizer,
                loader=loader,
                device=torch.device("cpu"),
                dataset="synthetic",
                split="train",
                run_dir=directory,
            )
            self.assertIsNotNone(artifacts.snapshot)
            self.assertIsNotNone(artifacts.transition)
            self.assertTrue(Path(directory, "posterior_snapshot.npz").is_file())
            self.assertTrue(Path(directory, "transition_artifact.npz").is_file())
            restored = build_builtin_pipeline({
                "name": "standard_noisy_erm",
                "transition_estimator": {"name": "dual_t"},
            })
            self.assertTrue(restored.load_artifacts(directory))
            self.assertEqual(
                restored.artifacts.snapshot.snapshot_hash,
                artifacts.snapshot.snapshot_hash,
            )

    def test_builtin_weight_provider_accepts_generic_pipeline_input(self) -> None:
        pipeline = build_builtin_pipeline({
            "name": "standard_noisy_erm",
            "weight_provider": {
                "name": "binary_rcn_importance",
                "rho_positive": 0.1,
                "rho_negative": 0.2,
            },
        })
        result = pipeline.weight_provider.compute(WeightInput(
            logits=torch.tensor([[2.0, 0.0]]),
            noisy_targets=torch.tensor([0]),
            sample_indices=torch.tensor([3]),
            per_sample_loss=torch.tensor([0.1]),
            posterior_probabilities=torch.tensor([[0.9, 0.1]]),
        ))
        self.assertEqual(tuple(result.sample_weights.shape), (1,))
        self.assertTrue(torch.isfinite(result.sample_weights).all())


if __name__ == "__main__":
    unittest.main()
