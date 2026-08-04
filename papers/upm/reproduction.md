# UPM reproduction status

## Scope

The toolbox implements the complete two-stage UPM method workflow: noisy-CE
pretraining, deterministic observed-label probability estimation, per-sample
confusing probabilities, paper Eq. (8), explicit Eq. (11)-(12) PGA, fixed-q
soft-target training, strict artifacts, and checkpoint/resume.

The available recipe is a workflow smoke:

```powershell
lnl papers show upm
lnl papers config upm --profile smoke --path-only
lnl run --recipe cifar10-upm-smoke
```

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

## Resume acceptance

Resume validates method/config identity, NoiseManifest identity, stable sample
mapping, Stage-1 best checkpoint identity, psi snapshot hash and provenance,
and complete eta/update-count state. Missing or damaged psi after `PSI_READY`
fails explicitly. Increasing only `upm.main.epochs` is supported; decreasing
epochs or changing the eta schedule is rejected.
