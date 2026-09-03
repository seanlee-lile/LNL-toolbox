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

Data and statistics use one Scratch-native contract: `inputs`, `targets` (always
the observed label), `indices`, and optional `clean_targets`.  The public
posterior/feature snapshot blocks collect detached values with their targets
and stable indices; fitting a transition or other estimator is a separate
operation.  Model names in formal recipes construct their corresponding
Scratch-owned topology (the bounded fixture limits data/steps only, never the
architecture).

The Scratch data picker is linked to the same local dataset catalog used by the
main data-registration page.  After registering a source there, open the
`load_dataset` block and choose its alias from the `dataset` selector.  If the
Scratch page was already open, click **同步已登记数据集** in the dataset
inspector to refresh the list.  The selected alias carries its registered
adapter and path into the Scratch-native loader; the loader still performs the
real layout/file checks and reports a precise error when the source is not
usable.  The shared catalog is `%LOCALAPPDATA%\\lnl-toolbox\\datasets.json`
on Windows (or the path in `LNL_DATA_CATALOG`).

The WebUI starts a run as a background Scratch job.  The **运行** panel polls
the job for the current block, epoch/batch position, and progress fraction;
**停止运行** requests cancellation and keeps the partial artifact directory.
When a dataset is missing, unregistered, unavailable, or incompatible with the
model class count, the same panel opens a concrete guidance card that points
back to the `load_dataset` or `create_model` parameters instead of showing only
the raw traceback.  The standalone service exposes these controls through
`/api/run`, `/api/jobs/<id>`, and `/api/jobs/<id>/cancel`; the mounted console
uses the corresponding `/api/scratch/...` routes.

The smoke coverage includes CE, formula-level GCE, APL, Co-teaching peer exchange, DivideMix warmup/co-divide/refinement, and legacy catalog shape/value paths. The GCE paper recipe is now a formal CIFAR-10 path; its structural tests validate it without launching the 120-epoch run. The separate `recipes/examples/gce_formula_smoke.yaml` remains the fast one-epoch check.

## Formula status

Scratch Paper templates:

- GCE: the formal recipe includes GCE-2018 data preparation, symmetric-0.2 noise, ResNet-34, SGD, MultiStepLR, validation model selection, and the formula chain expanded into forward, probability, target-class probability, q-formula, mean, backward, and optimizer steps.
- Co-teaching: the dual-peer remember-rate, per-sample losses, small-loss sets, cross-selected losses, and peer updates are expanded.
- The other 24 Paper Recipes are available as editable `template-ready` Scratch templates. Their formal recipe structure and data-particle composition remain visible, while their paper-specific formulas are not claimed to be fully expanded.

The DSS and LEND formal-semantics gates compare Scratch state/graph outputs to
their legacy implementations in tests only; those legacy modules are never
loaded by Scratch production code.

User-created Recipes are saved outside the package source tree under
`$LNL_SCRATCH_WORKSPACE/recipes/` (or `~/.lnl_toolbox/scratch/recipes/` when the
environment variable is unset). System Paper Recipes and bundled examples
remain read-only package resources.

## Research Formula Composer

Click **新建公式** in the Scratch page to save a user formula as YAML under
the Scratch user workspace (`$LNL_SCRATCH_WORKSPACE/formulas/`, or
`~/.lnl_toolbox/scratch/formulas/`). A formula contains declared inputs and
parameters, an ordered list of existing `formula_safe` Registry operations,
and named outputs. It is validated, registered immediately as a normal
`formula/user/...` Scratch Block, and can be referenced by any Recipe or by a
nested formula. No Python, `eval`, implicit detach, reduction, state mutation,
model construction, or loader creation is permitted inside a formula.

Five built-in YAML examples are shipped in `scratch/formula/examples`: standard
cross entropy, GCE, weighted cross entropy, transition-corrected risk, and a
per-sample confidence score. Run artifacts record each formula's content hash
and YAML snapshot for reproducibility.
