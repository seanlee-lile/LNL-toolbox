from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json
import shutil
import subprocess
import unittest

import torch

from lnl_toolbox.scratch import ScratchContext, execute_recipe, load_recipe, validate_recipe
from lnl_toolbox.scratch.blocks.selection import select_lowest_scores


ROOT = Path(__file__).resolve().parents[1]


class ScratchSmokeTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js required to generate the actual UI template')
    def test_single_model_template_clears_gradients_each_batch(self) -> None:
        result = subprocess.run([
            shutil.which('node'), '-e', r"""
const fs = require('node:fs'), vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const code = source.slice(source.indexOf('function makeRecipeStep('), source.indexOf('function dualSkeletonRecipe('));
const sandbox = {newStep: block => ({block, params: {}})};
vm.createContext(sandbox);
console.log(JSON.stringify(vm.runInContext(code + '\nsingleSkeletonRecipe()', sandbox)));
""", str(ROOT / 'web' / 'scratch.js')], capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        template = json.loads(result.stdout)
        epoch = next(step for step in template['steps'] if step['block'] == 'epoch_loop')
        batch = next(step for step in epoch['steps'] if step['block'] == 'batch_loop')
        recipe = {'schema_version': 1, 'name': 'single model gradient regression', 'steps': [
            {'block': 'epoch_loop', 'params': {'epochs': 2}, 'steps': [batch]},
        ]}
        with torch.random.fork_rng():
            torch.manual_seed(12)
            actual = torch.nn.Linear(4, 2)
            expected = deepcopy(actual)
            batches = [{'inputs': torch.randn(3, 4), 'targets': torch.tensor([0, 1, 0]),
                        'indices': torch.arange(3)} for _ in range(2)]
        settings = next(step['params'] for step in template['steps'] if step['block'] == 'create_optimizer')
        optimizer = torch.optim.SGD(actual.parameters(), lr=settings['lr'], momentum=settings['momentum'])
        reference = torch.optim.SGD(expected.parameters(), lr=settings['lr'], momentum=settings['momentum'])
        # Include stale gradients before the first batch, then exercise both
        # batch and epoch boundaries with the UI's actual generated steps.
        for model in (actual, expected):
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
        context = execute_recipe(recipe, ScratchContext({
            'model': actual, 'optimizer': optimizer, 'train_loader': batches, 'device': 'cpu',
        }))
        for _ in range(2):
            for sample in batches:
                reference.zero_grad()
                torch.nn.functional.cross_entropy(expected(sample['inputs']), sample['targets']).backward()
                reference.step()
        self.assertEqual(context['global_step'], 4)
        for observed, wanted in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(observed, wanted)
            torch.testing.assert_close(observed.grad, wanted.grad)

    def test_ce_and_gce_recipes_execute_one_epoch(self) -> None:
        for filename in ("ce_synthetic.yaml", "gce_small_loss_synthetic.yaml"):
            recipe = load_recipe(ROOT / "recipes" / "examples" / filename)
            validate_recipe(recipe)
            context = execute_recipe(recipe)
            self.assertIn("model", context)
            self.assertTrue(context["loss"].ndim == 0)

    def test_common_selection_uses_string_slots(self) -> None:
        context = ScratchContext({"a": torch.tensor([3.0, 1.0, 2.0, 4.0]), "b": torch.tensor([1.0, 4.0, 2.0, 3.0]), "rate": 0.5})
        select_lowest_scores(context, scores="a", keep_fraction="rate", save_as="selected_a")
        select_lowest_scores(context, scores="b", keep_fraction="rate", save_as="selected_b")
        self.assertEqual(context["selected_a"].tolist(), [1, 2])
        self.assertEqual(context["selected_b"].tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
