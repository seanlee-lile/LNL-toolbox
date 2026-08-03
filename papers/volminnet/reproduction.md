# VolMinNet reproduction status

Paper: *Provably End-to-end Label-noise Learning without Anchor Points*
(ICML 2021).

## Current status

- Method workflow: implemented.
- Unified CLI and smoke recipe: implemented.
- Checkpoint/resume and transition provenance: implemented.
- Paper numerical reproduction: not run.

The smoke configuration uses CIFAR-10 with small subsets, two epochs, disabled
augmentation, and TinyCNN. Those are smoke-only engineering choices. The
paper-oriented mechanism retained by the smoke is the fixed-diagonal,
sigmoid-off-diagonal row-stochastic transition; joint classifier/transition
optimization; noisy-label NLL; `lambda=1e-4`; and noisy-validation selection.

The paper's CIFAR-10 experiment uses ResNet-18, batch size 128, SGD, initial
learning rate 0.01, momentum 0.9, learning-rate drops after epochs 30 and 60,
and 150 epochs. A formal reproduction configuration and repeated-seed results
remain future work. The author's released code uses `log(abs(det(T)))`, while
this implementation deliberately follows the paper-facing positive-logdet
contract and rejects non-positive determinants.

## Commands

```powershell
lnl validate --recipe cifar10-volminnet-smoke
lnl run --recipe cifar10-volminnet-smoke --dry-run
lnl run --recipe cifar10-volminnet-smoke
lnl resume <run-directory>
lnl papers show volminnet
```

Do not interpret smoke accuracy or transition error as reproduction of the
paper tables. Synthetic ground-truth transition matrices are diagnostic only
and never enter training or checkpoint selection.
