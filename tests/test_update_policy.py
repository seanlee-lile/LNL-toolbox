import unittest

import torch

from lnl_toolbox.algorithms.update_policy import (
    ParameterUpdateInput,
    ParameterUpdateResult,
    StandardUpdatePolicy,
    restore_update_policy,
    serialize_update_policy,
)
from lnl_toolbox.core import RunState


class ParameterUpdatePolicyTest(unittest.TestCase):
    def test_standard_policy_matches_ordinary_torch_update(self) -> None:
        policy_model = torch.nn.Linear(3, 2)
        reference_model = torch.nn.Linear(3, 2)
        reference_model.load_state_dict(policy_model.state_dict())
        inputs = torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]])
        targets = torch.tensor([0, 1])

        policy_optimizer = torch.optim.SGD(
            policy_model.parameters(), lr=0.1, momentum=0.9
        )
        reference_optimizer = torch.optim.SGD(
            reference_model.parameters(), lr=0.1, momentum=0.9
        )
        policy_objective = torch.nn.functional.cross_entropy(
            policy_model(inputs), targets
        )
        result = StandardUpdatePolicy().update(ParameterUpdateInput(
            objective=policy_objective,
            model=policy_model,
            optimizer=policy_optimizer,
            run_state=RunState(),
        ))

        reference_optimizer.zero_grad(set_to_none=True)
        reference_objective = torch.nn.functional.cross_entropy(
            reference_model(inputs), targets
        )
        reference_objective.backward()
        reference_optimizer.step()

        self.assertEqual(result.metrics, {})
        for actual, expected in zip(
            policy_model.parameters(), reference_model.parameters()
        ):
            self.assertTrue(torch.equal(actual, expected))
        policy_state = policy_optimizer.state_dict()
        reference_state = reference_optimizer.state_dict()
        self.assertEqual(policy_state["param_groups"], reference_state["param_groups"])
        for actual, expected in zip(
            policy_state["state"].values(),
            reference_state["state"].values(),
        ):
            self.assertEqual(set(actual), set(expected))
            for key in actual:
                if torch.is_tensor(actual[key]):
                    self.assertTrue(torch.equal(actual[key], expected[key]))
                else:
                    self.assertEqual(actual[key], expected[key])

    def test_update_input_requires_a_finite_scalar_with_gradients(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        state = RunState()
        with self.assertRaisesRegex(ValueError, "scalar"):
            ParameterUpdateInput(torch.ones(2, requires_grad=True), model, optimizer, state)
        with self.assertRaisesRegex(ValueError, "require gradients"):
            ParameterUpdateInput(torch.tensor(1.0), model, optimizer, state)
        with self.assertRaisesRegex(ValueError, "finite"):
            ParameterUpdateInput(
                torch.tensor(float("nan"), requires_grad=True),
                model,
                optimizer,
                state,
            )

    def test_update_result_rejects_nonfinite_metrics(self) -> None:
        self.assertEqual(ParameterUpdateResult({"steps": 1}).metrics, {"steps": 1.0})
        with self.assertRaisesRegex(ValueError, "finite"):
            ParameterUpdateResult({"bad": float("inf")})

    def test_policy_checkpoint_identity_is_strict(self) -> None:
        policy = StandardUpdatePolicy()
        payload = serialize_update_policy(policy)
        self.assertEqual(payload, {"name": "standard", "state": {}})
        restore_update_policy(policy, payload)
        restore_update_policy(policy, None)
        with self.assertRaisesRegex(ValueError, "does not match"):
            restore_update_policy(policy, {"name": "other", "state": {}})
        with self.assertRaisesRegex(TypeError, "mapping"):
            restore_update_policy(policy, {"name": "standard", "state": []})

    def test_standard_policy_gradient_clipping_matches_torch(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        policy = StandardUpdatePolicy(max_grad_norm=0.25)
        objective = (model(torch.ones(1, 2)) - 10.0).square().mean()
        request = ParameterUpdateInput(objective, model, optimizer, RunState())
        policy.update(request)
        self.assertAlmostEqual(float(model.weight.grad.norm()), 0.25, places=5)
        self.assertEqual(policy.state_dict(), {"max_grad_norm": 0.25})


if __name__ == "__main__":
    unittest.main()
