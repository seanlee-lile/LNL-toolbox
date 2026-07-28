import math
import tempfile
import unittest
from pathlib import Path

import torch

from lnl_toolbox.algorithms.cdr import (
    CDRUpdatePolicy,
    critical_parameter_masks,
    official_code_parameter_masks,
)
from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.algorithms.update_policy import ParameterUpdateInput
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.losses import CrossEntropyLoss
from lnl_toolbox.plugins.builtin import (
    build_builtin_parameter_update_policy,
    create_builtin_catalog,
)
from lnl_toolbox.selectors import SmallLossSelector
from lnl_toolbox.training.checkpoint import load_checkpoint, save_checkpoint
from lnl_toolbox.training.experiment import (
    _validate_resume_config,
    _validate_supervised_config,
)


class _VectorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor([2.0, -1.0, 0.0, 4.0]))

    def forward(self) -> torch.Tensor:
        return self.values


class CDRUpdatePolicyTest(unittest.TestCase):
    def test_criticality_and_exact_top_k_match_equations_three_and_four(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0, 0.0, 4.0]))
        parameter.grad = torch.tensor([1.0, 3.0, 9.0, -0.25])
        result = critical_parameter_masks([("weight", parameter)], 0.5)
        self.assertEqual(result.eligible_parameters, 4)
        self.assertEqual(result.critical_parameters, 2)
        self.assertTrue(torch.equal(
            result.masks["weight"],
            torch.tensor([True, True, False, False]),
        ))

    def test_ties_use_parameter_name_then_flat_offset(self) -> None:
        first = torch.nn.Parameter(torch.ones(2))
        second = torch.nn.Parameter(torch.ones(2))
        first.grad = torch.ones(2)
        second.grad = torch.ones(2)
        result = critical_parameter_masks(
            [("z_weight", second), ("a_weight", first)],
            0.5,
        )
        self.assertTrue(torch.equal(
            result.masks["a_weight"], torch.tensor([True, True])
        ))
        self.assertTrue(torch.equal(
            result.masks["z_weight"], torch.tensor([False, False])
        ))

    def test_cdr_update_matches_equations_five_and_six(self) -> None:
        model = _VectorModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        coefficients = torch.tensor([1.0, 3.0, 9.0, -0.25])
        objective = (model() * coefficients).sum()
        before = model.values.detach().clone()

        result = CDRUpdatePolicy(noise_rate=0.5, l1_decay=0.1).update(
            ParameterUpdateInput(
                objective=objective,
                model=model,
                optimizer=optimizer,
                run_state=RunState(),
            )
        )

        expected_gradient = torch.tensor([
            0.5 * 1.0 + 0.1,
            0.5 * 3.0 - 0.1,
            0.0,
            0.1,
        ])
        self.assertTrue(torch.allclose(
            model.values,
            before - 0.2 * expected_gradient,
            atol=0.0,
            rtol=0.0,
        ))
        self.assertEqual(result.metrics["update_eligible_parameters"], 4.0)
        self.assertEqual(result.metrics["update_critical_parameters"], 2.0)
        self.assertEqual(result.metrics["update_critical_ratio"], 0.5)

    def test_noncritical_parameters_receive_only_l1_sign_update(self) -> None:
        model = _VectorModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        objective = (model() * torch.tensor([1.0, 0.5, 8.0, 0.1])).sum()
        before = model.values.detach().clone()
        CDRUpdatePolicy(noise_rate=0.75, l1_decay=0.2).update(
            ParameterUpdateInput(objective, model, optimizer, RunState())
        )
        delta = (before - model.values.detach()) / 0.1
        self.assertAlmostEqual(delta[1].item(), -0.2, places=6)
        self.assertEqual(delta[2].item(), 0.0)
        self.assertAlmostEqual(delta[3].item(), 0.2, places=6)

    def test_invalid_parameters_optimizer_and_gradients_fail(self) -> None:
        with self.assertRaises(ValueError):
            CDRUpdatePolicy(noise_rate=-0.1, l1_decay=0.1)
        with self.assertRaises(ValueError):
            CDRUpdatePolicy(noise_rate=1.0, l1_decay=0.1)
        with self.assertRaises(ValueError):
            CDRUpdatePolicy(noise_rate=0.2, l1_decay=-0.1)
        with self.assertRaises(ValueError):
            CDRUpdatePolicy(
                noise_rate=0.2,
                l1_decay=0.1,
                critical_scope="weights_only",
            )

        model = _VectorModel()
        objective = model().sum()
        with self.assertRaisesRegex(TypeError, "SGD"):
            CDRUpdatePolicy(0.2, 0.1).update(ParameterUpdateInput(
                objective, model, torch.optim.Adam(model.parameters()), RunState()
            ))

        model = _VectorModel()
        objective = model().sum()
        with self.assertRaisesRegex(ValueError, "weight_decay=0"):
            CDRUpdatePolicy(0.2, 0.1).update(ParameterUpdateInput(
                objective,
                model,
                torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01),
                RunState(),
            ))

        parameter = torch.nn.Parameter(torch.tensor([float("nan")]))
        parameter.grad = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "finite"):
            critical_parameter_masks([("bad", parameter)], 0.2)

        embedding = torch.nn.Embedding(4, 2, sparse=True)
        sparse_objective = embedding(torch.tensor([0, 1])).sum()
        sparse_objective.backward()
        with self.assertRaisesRegex(ValueError, "sparse"):
            critical_parameter_masks(embedding.named_parameters(), 0.2)

    def test_count_uses_ceil_and_all_trainable_dimensions_participate(self) -> None:
        weight = torch.nn.Parameter(torch.ones(4, 2))
        bias = torch.nn.Parameter(torch.ones(3))
        weight.grad = torch.ones_like(weight)
        bias.grad = torch.ones_like(bias)
        result = critical_parameter_masks(
            [("weight", weight), ("bias", bias)],
            noise_rate=0.4,
        )
        self.assertEqual(result.eligible_parameters, 11)
        self.assertEqual(result.critical_parameters, math.ceil(0.6 * 11))
        self.assertIn("bias", result.masks)

    def test_official_code_mode_matches_released_scope_ties_and_l2_step(self) -> None:
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, -1.0], [0.0, 4.0]]))
            model.bias.copy_(torch.tensor([1.0, -1.0]))
        weight_before = model.weight.detach().clone()
        bias_before = model.bias.detach().clone()
        coefficients = torch.tensor([[1.0, 3.0], [9.0, -0.25]])
        bias_coefficients = torch.tensor([2.0, -3.0])
        objective = (
            (model.weight * coefficients).sum()
            + (model.bias * bias_coefficients).sum()
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, weight_decay=0.1
        )
        result = CDRUpdatePolicy(
            noise_rate=0.5,
            l1_decay=0.0,
            critical_scope="matrix_and_convolution_weights",
            compatibility_mode="official_code",
        ).update(ParameterUpdateInput(
            objective, model, optimizer, RunState()
        ))
        expected_weight_gradient = torch.tensor([
            [0.5, 1.5],
            [0.0, 0.0],
        ]) + 0.1 * weight_before
        expected_bias_gradient = bias_coefficients + 0.1 * bias_before
        torch.testing.assert_close(
            model.weight,
            weight_before - 0.1 * expected_weight_gradient,
        )
        torch.testing.assert_close(
            model.bias,
            bias_before - 0.1 * expected_bias_gradient,
        )
        self.assertEqual(result.metrics["update_eligible_parameters"], 4.0)
        self.assertEqual(result.metrics["update_critical_parameters"], 2.0)

        tied = torch.nn.Parameter(torch.ones(4, 1))
        tied.grad = torch.ones_like(tied)
        masks = official_code_parameter_masks([("weight", tied)], 0.5)
        self.assertEqual(masks.critical_parameters, 4)

    def test_official_code_mode_rejects_paper_l1_or_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "l1_decay=0"):
            CDRUpdatePolicy(
                0.4,
                0.001,
                compatibility_mode="official_code",
                critical_scope="matrix_and_convolution_weights",
            )
        with self.assertRaisesRegex(ValueError, "critical_scope"):
            CDRUpdatePolicy(
                0.4,
                0.0,
                compatibility_mode="official_code",
                critical_scope="all_trainable",
            )

    def test_selector_and_cdr_compose_without_clean_label_input(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        algorithm = SupervisedClassificationAlgorithm(
            model,
            optimizer,
            CrossEntropyLoss(),
            torch.device("cpu"),
            selector=SmallLossSelector(keep_rate=0.5),
            update_policy=CDRUpdatePolicy(noise_rate=0.5, l1_decay=0.0),
        )
        algorithm.setup(ExperimentContext(Path.cwd()))
        result = algorithm.step(
            Batch({
                "input": torch.eye(2),
                "target": torch.tensor([0, 1]),
                "index": torch.tensor([9, 3]),
            }),
            RunState(),
        )
        self.assertEqual(result.metrics["selected_ratio"], 0.5)
        self.assertEqual(result.metrics["update_critical_ratio"], 0.5)
        self.assertEqual(
            set(Batch({
                "input": torch.eye(2),
                "target": torch.tensor([0, 1]),
                "index": torch.tensor([9, 3]),
            }).payload),
            {"input", "target", "index"},
        )

    def test_plugin_kind_and_builder_are_distinct(self) -> None:
        catalog = create_builtin_catalog()
        self.assertEqual(
            [
                item.name
                for item in catalog.find(kind="parameter_update_policy")
            ],
            ["cdr", "standard"],
        )
        self.assertEqual(
            build_builtin_parameter_update_policy(None, catalog).name,
            "standard",
        )
        cdr = build_builtin_parameter_update_policy({
            "name": "cdr",
            "noise_rate": 0.4,
            "l1_decay": 0.001,
            "critical_scope": "all_trainable",
        }, catalog)
        self.assertIsInstance(cdr, CDRUpdatePolicy)
        with self.assertRaises(ValueError):
            build_builtin_parameter_update_policy({"name": "unknown"}, catalog)
        with self.assertRaises(TypeError):
            build_builtin_parameter_update_policy("cdr", catalog)  # type: ignore[arg-type]

    def test_checkpoint_roundtrip_binds_policy_identity_and_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.05, momentum=0.9
            )
            algorithm = SupervisedClassificationAlgorithm(
                model,
                optimizer,
                CrossEntropyLoss(),
                torch.device("cpu"),
                update_policy=CDRUpdatePolicy(0.4, 0.001),
            )
            algorithm.setup(ExperimentContext(Path(directory)))
            state = RunState()
            algorithm.step(Batch({
                "input": torch.eye(2),
                "target": torch.tensor([0, 1]),
                "index": torch.tensor([4, 2]),
            }), state)
            saved_parameters = {
                name: value.detach().clone()
                for name, value in model.named_parameters()
            }
            path = Path(directory) / "last.pt"
            config = {
                "parameter_update": {
                    "name": "cdr",
                    "noise_rate": 0.4,
                    "l1_decay": 0.001,
                }
            }
            save_checkpoint(path, algorithm, state, 0, config)

            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.SGD(
                restored_model.parameters(), lr=0.05, momentum=0.9
            )
            restored_algorithm = SupervisedClassificationAlgorithm(
                restored_model,
                restored_optimizer,
                CrossEntropyLoss(),
                torch.device("cpu"),
                update_policy=CDRUpdatePolicy(0.4, 0.001),
            )
            restored_algorithm.setup(ExperimentContext(Path(directory)))
            restored_state, epoch, payload = load_checkpoint(
                path, restored_algorithm, torch.device("cpu")
            )
            self.assertEqual(restored_state.step, 1)
            self.assertEqual(epoch, 0)
            self.assertEqual(
                payload["parameter_update_policy"],
                {"name": "cdr", "state": {}},
            )
            for name, value in restored_model.named_parameters():
                self.assertTrue(torch.equal(value, saved_parameters[name]))
            self.assertTrue(restored_optimizer.state)

            standard_model = torch.nn.Linear(2, 2)
            standard_algorithm = SupervisedClassificationAlgorithm(
                standard_model,
                torch.optim.SGD(standard_model.parameters(), lr=0.05),
                CrossEntropyLoss(),
                torch.device("cpu"),
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_checkpoint(path, standard_algorithm, torch.device("cpu"))

    def test_runner_does_not_own_cdr_optimizer_validation(self) -> None:
        base = {
            "optimizer": {"name": "sgd", "lr": 0.01, "weight_decay": 0.0},
            "parameter_update": {
                "name": "cdr",
                "noise_rate": 0.4,
                "l1_decay": 0.001,
            },
        }
        _validate_supervised_config(base)
        _validate_supervised_config({
            **base,
            "optimizer": {"name": "adamw", "lr": 0.001},
        })
        _validate_supervised_config({
            **base,
            "optimizer": {"name": "sgd", "lr": 0.01, "weight_decay": 0.1},
        })
        with self.assertRaisesRegex(ValueError, "parameter_update"):
            _validate_resume_config(
                base,
                {
                    **base,
                    "parameter_update": {
                        "name": "cdr",
                        "noise_rate": 0.2,
                        "l1_decay": 0.001,
                    },
                },
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_and_cuda_masks_and_updates_match(self) -> None:
        coefficients = torch.tensor([1.0, 3.0, 9.0, -0.25])
        results = []
        for device in (torch.device("cpu"), torch.device("cuda")):
            model = _VectorModel().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
            objective = (model() * coefficients.to(device)).sum()
            CDRUpdatePolicy(0.5, 0.1).update(ParameterUpdateInput(
                objective, model, optimizer, RunState()
            ))
            results.append(model.values.detach().cpu())
        self.assertTrue(torch.equal(results[0], results[1]))


if __name__ == "__main__":
    unittest.main()
