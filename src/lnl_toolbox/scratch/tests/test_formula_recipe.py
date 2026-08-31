import unittest

import torch

from lnl_toolbox.scratch import ScratchContext, execute_recipe, load_recipe, validate_recipe
from lnl_toolbox.scratch.formula import register_formula, unregister_formula


class FormulaRecipeIntegrationTest(unittest.TestCase):
    def tearDown(self):
        try:
            unregister_formula("user/recipe_formula")
        except KeyError:
            pass

    def test_formula_is_a_normal_recipe_step(self):
        register_formula({
            "id": "user/recipe_formula",
            "name": "Recipe Formula",
            "inputs": {"x": {}},
            "steps": [{"id": "out", "block": "detach", "bindings": {"input": "x"}}],
            "outputs": {"value": {"source": "out"}},
        })
        recipe = {
            "schema_version": 1,
            "name": "formula_recipe",
            "settings": {"input_tensor": torch.ones(2, requires_grad=True)},
            "steps": [{"block": "formula/user/recipe_formula", "params": {"x": "input_tensor", "save_as": "result"}}],
        }
        validate_recipe(recipe)
        context = execute_recipe(recipe)
        self.assertIn("result", context)
        self.assertFalse(context["result"].requires_grad)

    def test_new_algorithm_example_runs_without_python_paper_block(self):
        recipe = load_recipe("src/lnl_toolbox/scratch/recipes/examples/custom_formula_method.yaml")
        validate_recipe(recipe)
        context = execute_recipe(recipe, runtime_limits={"max_epochs": 1, "max_batches": 1, "skip_final_test": True})
        self.assertTrue(context.get("metrics"))
        self.assertIn("_formula_provenance", context)
