# UPM reproduction status

## Scope

The toolbox implements the complete two-stage UPM method workflow: noisy-CE
pretraining, deterministic observed-label probability estimation, per-sample
confusing probabilities, paper Eq. (8), explicit Eq. (11)-(12) PGA, fixed-q
soft-target training, strict artifacts, and checkpoint/resume.

The available smoke recipe is a workflow check:

```powershell
lnl papers show upm
lnl papers config upm --profile smoke --path-only
lnl run --recipe cifar10-upm-smoke
```

The catalog also exposes `upm-cifar10-reproduction`, a full-budget CIFAR-10
engineering profile. Its availability does not change the current
`reproduction_status: not_run`: neither a recipe, Dry-run, nor a completed
single workflow proves the paper's numerical results.

## Fidelity boundaries

Paper-specified mechanism includes scalar observed-class `psi`, sample-specific
`eta`, detached predicting posterior `q`, and projected eta updates. The paper's
CIFAR experiments use ResNet-32, 160 epochs, batch size 256, momentum 0.9,
weight decay 1e-4, classifier LR 0.05 decayed every 40 epochs, and eta updates
every five epochs beginning at the 35th epoch.

The deterministic best-checkpoint snapshot is a toolbox reproducibility
choice: the paper does not specify best-versus-last snapshot selection or
augmentation during collection. A fresh Stage-2 model matches the released
Clothing1M code behavior, although the paper does not state this explicitly.

The smoke's TinyCNN, small subsets, short epochs, and symmetric noise are
engineering choices. The paper's CIFAR-100 MLP relabeling, CIFAR-10
ResNet-50/k-means relabeling, Clothing1M experiment, five-seed statistics, and
paper tables remain unexecuted. Therefore smoke results must not be described
as numerical reproduction of the paper.

## Producer boundary

UPM publishes `stage1_best.pt`, `psi_snapshot.npz`, main-stage checkpoints,
manifest provenance, and stable-index identity for its own workflow. PCSE and
a compatible DLD source adapter may consume a UPM source only after validating
the declared role, checkpoint schema, model/dataset identity, manifest,
mapping, and hashes. The presence of an arbitrary UPM checkpoint is not
evidence that it satisfies a consumer contract.

## Resume acceptance

Resume validates method/config identity, NoiseManifest identity, stable sample
mapping, Stage-1 best checkpoint identity, psi snapshot hash and provenance,
and complete eta/update-count state. Missing or damaged psi after `PSI_READY`
fails explicitly. Increasing only `upm.main.epochs` is supported; decreasing
epochs or changing the eta schedule is rejected. `lnl resume` itself does not
accept an epoch override; an increased target requires an updated YAML and an
explicit `lnl run --config ... --resume <run>/last.pt` path after Dry-run.
