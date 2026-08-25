# LNL Scratch

Scratch is a small recipe interpreter for assembling noisy-label training as an ordered list of semantic blocks.

```text
Recipe = ordered algorithm steps
Block = one Python algorithm step
Context = shared string-keyed workbench
Executor = top-to-bottom interpreter
```

Run it independently from the legacy CLI:

```bash
python -m lnl_toolbox.scratch.cli list-blocks
python -m lnl_toolbox.scratch.cli validate src/lnl_toolbox/scratch/recipes/examples/ce_synthetic.yaml
python -m lnl_toolbox.scratch.cli run src/lnl_toolbox/scratch/recipes/examples/gce_small_loss_synthetic.yaml
python -m lnl_toolbox.scratch.web.server
```

The runtime imports only Python's standard library, PyTorch/torchvision, NumPy, PyYAML, and Scratch's own modules. Legacy algorithms are consulted during development but are not runtime dependencies.

The smoke coverage includes CE, formula-level GCE, APL, Co-teaching peer exchange, DivideMix warmup/co-divide/refinement, and legacy catalog shape/value paths. The GCE paper recipe is now a formal CIFAR-10 path; its structural tests validate it without launching the 120-epoch run. The separate `recipes/examples/gce_formula_smoke.yaml` remains the fast one-epoch check.

## Formula status

Scratch Paper templates:

- GCE: the formal recipe includes GCE-2018 data preparation, symmetric-0.2 noise, ResNet-34, SGD, MultiStepLR, validation model selection, and the formula chain expanded into forward, probability, target-class probability, q-formula, mean, backward, and optimizer steps.
- Co-teaching: the dual-peer remember-rate, per-sample losses, small-loss sets, cross-selected losses, and peer updates are expanded.
- The other 24 Paper Recipes are available as editable `template-ready` Scratch templates. Their formal recipe structure and data-particle composition remain visible, while their paper-specific formulas are not claimed to be fully expanded.
