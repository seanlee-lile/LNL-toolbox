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

The current smoke coverage includes CE, GCE, APL, Co-teaching peer exchange, and the DivideMix warmup/co-divide/refinement lifecycle. The paper recipe directory is intentionally marked in `REFERENCE_MAP.md`; a recipe is not considered a numerical reproduction until its equations, defaults, lifecycle, and tensor shapes have been checked against the paper and the catalogued implementation.
