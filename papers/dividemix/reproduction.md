# DivideMix CIFAR-10 Reproduction and Maintenance Guide

This repository exposes DivideMix through the standard LNL Toolbox workflow.
The formal public profile is a full-data, single-seed CIFAR-10 usability
configuration for symmetric 20% label noise. It is intended for transition
sanity checks and long-running engineering validation; it is not evidence of
paper-exact numerical reproduction.

## Public workflow

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.main papers show dividemix
python -m lnl_toolbox.cli.main validate --recipe cifar10-dividemix-sym20
python -m lnl_toolbox.cli.main run --recipe cifar10-dividemix-sym20 --dry-run
python -m lnl_toolbox.cli.main run --recipe cifar10-dividemix-sym20
python -m lnl_toolbox.cli.main resume <run-directory>
```

For the bounded warmup-to-main GPU transition check, override only the main
stage:

```powershell
python -m lnl_toolbox.cli.main run `
  --recipe cifar10-dividemix-sym20 `
  --epochs 1 `
  --output-dir <temporary-run-directory>
```

This executes 10 warmup epochs followed by one main DivideMix epoch. Resume
the same run to two completed main epochs with:

```powershell
python -m lnl_toolbox.cli.main resume <temporary-run-directory> --epochs 2
```

Generated checkpoints, manifests, metrics, and per-epoch co-divide artifacts
belong in the run directory and must not be committed.

## Formal configuration

`configs/reproduction/cifar10_dividemix_sym20.yaml` fixes:

- full CIFAR-10 input with no sample-count limits;
- a 5,000-example noisy-validation split and clean test split;
- symmetric 20% label noise with seed 1;
- two independently initialized PreActResNet-18 peers with base width 64;
- SGD at learning rate 0.02, momentum 0.9, and weight decay 5e-4;
- minibatch size 128;
- 10 warmup epochs and 300 main-stage epochs;
- a MultiStep scheduler with milestone 150 and gamma 0.1;
- two augmentations, sharpening temperature 0.5, MixUp alpha 4;
- clean-probability threshold 0.5, lambda-u 25, and lambda-r 1.

The public `--epochs N` option changes only
`dividemix.training.epochs`. It never changes the 10-epoch warmup budget.
Consequently, the stored formal profile represents 10 warmup plus 300 main
epochs: 310 Toolbox lifecycle epochs. The paper describes a 300-epoch budget;
this separate-stage representation is an explicit engineering lifecycle
difference and must not be hidden.

The current runner steps both schedulers after every warmup epoch and every
main epoch. Therefore milestone 150 is a global scheduler-step milestone: with
10 warmup epochs it is reached after main epoch 140, not main epoch 150. The
configuration preserves the requested/official milestone value rather than
silently changing it to 160. This is another known fidelity boundary.

## Fidelity boundaries

### Architecture

The Toolbox model is a genuine 18-layer pre-activation residual architecture
with `[2, 2, 2, 2]` blocks and base width 64. Its stem and final normalization
are not bit-for-bit identical to the released DivideMix `PreResNet.py` model.
Architecture fidelity is therefore **CLOSE / paper-compatible**, not
official-code exact.

### Noise

The Toolbox `sampling: global` symmetric generator selects a fixed number of
examples and forces each selected label to change class. This matches the
paper's separately reported no-true-label-retention criterion, but not the
main-table random-replacement process where a sampled label may remain the
original class. The formal profile is **Appendix-compatible**, not main-table
noise-exact.

### Validation and reporting

The runner reserves 5,000 original training examples for noisy-validation
selection. It selects a paired A/B checkpoint using noisy-validation ensemble
accuracy and evaluates the clean test set only after training. This protects
the clean-test boundary but differs from the paper's best-test and last-ten
reporting protocol. No paper table value, multi-seed statistic, or last-ten
result has been reproduced by this configuration.

## Pipeline and artifacts

The complete workflow is:

```text
independent A/B warmup
-> full-training-set per-sample CE snapshots
-> two-component GMM for each peer
-> cross co-divide (A selects for B; B selects for A)
-> label co-refinement and two-network co-guessing
-> MixMatch objective
-> sequential A/B updates
-> paired noisy-validation selection
-> clean-test logits-sum ensemble
```

Inspect at least:

```text
resolved_config.yaml
environment.json
noise_manifest.npz
noise_summary.json
metrics.jsonl
dividemix_epoch_*.npz
last.pt
best.pt
final_metrics.json
```

Each co-divide artifact binds stable sample indices, normalized losses, clean
probabilities, masks, model hashes, configuration hash, manifest mapping hash,
epoch, and the cross producer/consumer direction. Resume validates these
identities and restores both peers, optimizers, schedulers, RNG state, phase,
and bounded loss histories. Exact recovery is supported at persisted phase and
epoch boundaries; mid-minibatch exact resume is not claimed.

## Claim boundary

The repository may claim a tested, user-ready DivideMix method workflow after
the full-data GPU transition sanity passes. It must not claim paper-exact
architecture, main-table noise generation, 300-total-epoch semantics, reported
accuracy reproduction, or multi-seed statistics without separate evidence.

## Sources

- Junnan Li, Richard Socher, and Steven C. H. Hoi. *DivideMix: Learning with
  Noisy Labels as Semi-supervised Learning*. ICLR 2020.
- Paper: <https://openreview.net/forum?id=HJgExaVtwr>
- Released implementation: <https://github.com/LiJunnan1992/DivideMix>
