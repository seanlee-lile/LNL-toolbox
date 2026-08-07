# DivideMix reproduction maintenance

## Status

The toolbox provides a user-ready, paper/official-oriented DivideMix workflow.
The bundled configuration is a small workflow smoke, not a numerical
reproduction of the 300-epoch CIFAR experiments.

## Fidelity boundary

Implemented mechanisms include independent peer warmup, asymmetric-noise
confidence penalty, full-dataset epoch loss snapshots, the existing
two-component GMM clean-probability estimator, cross co-divide, co-refinement,
co-guessing, minibatch-scalar MixUp, the three-term MixMatch objective, paired
validation selection, and phase-aware resume. Deterministic GMM seeding,
stable-index alignment, artifact validation, and noisy-validation checkpoint
selection are toolbox safeguards. The released CIFAR code does not provide the
same strict artifact/resume lifecycle and evaluates test data every epoch.
For the official 90%-noise five-epoch history policy, the averaged normalized
loss is min-max normalized once more to satisfy the existing shared estimator
contract; this is a documented implementation choice rather than an author-code
exact detail.

The smoke uses TinyCNN, small CIFAR subsets, no augmentation, one warmup epoch,
and two main epochs. It must not be described as reproducing the paper's
reported accuracy. A future formal profile should use PreAct ResNet-18, the
paper optimizer and augmentation, 300 total epochs, and multi-seed reporting.

## Commands

```powershell
lnl papers show dividemix
lnl validate --recipe cifar10-dividemix-smoke
lnl run --recipe cifar10-dividemix-smoke --dry-run
lnl run --recipe cifar10-dividemix-smoke
lnl resume <run-directory>
```

## Run record

```text
Commit:
Config:
Device:
Warmup/main epochs:
Manifest mapping hash:
Last co-divide artifact hash:
Best noisy-validation ensemble accuracy:
Clean-test ensemble accuracy:
Resume boundary:
Completed no-op verified:
NaN/Inf:
Result:
Notes:
```
