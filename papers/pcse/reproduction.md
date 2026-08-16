# PCSE real-CIFAR maintenance workflow

The public `cifar10-pcse-reproduction` recipe is an engineering real-data
workflow. It consumes an immutable full-budget UPM CIFAR-10 checkpoint and
runs the existing PCSE `paper_volmin` transition, statistic recovery,
multi-layer GDA, noisy-validation ensemble fitting, and clean-test inference.
It is not a paper-exact numerical reproduction.

## Source contract

Set `LNL_PCSE_SOURCE_RUN` to the UPM run directory containing `best.pt` and
`noise_manifest.npz`. The adapter supports only the explicit
`upm_main_best` schema. It validates the configured checkpoint SHA-256,
manifest SHA-256, mapping hash, dataset fingerprint, method/role, ResNet-18
base-width 16 architecture, ten classes, and a strict state-dict load.
The source files' SHA-256, size, and mtime are checked again after the run.

```powershell
$env:LNL_PCSE_SOURCE_RUN = "C:\Users\win11\AppData\Local\Temp\lnl-upm-full-mainprime-8904368-20260812-v1"
python -m lnl_toolbox.cli.main validate --recipe cifar10-pcse-reproduction
python -m lnl_toolbox.cli.main run --recipe cifar10-pcse-reproduction --dry-run
python -m lnl_toolbox.cli.main run --recipe cifar10-pcse-reproduction
python -m lnl_toolbox.cli.main resume <run-directory>
```

## Data boundaries

The source manifest fixes a full CIFAR-10 symmetric-40 mapping. The 45,000
noisy training examples feed VolMin and PCSE statistics; the independent 5,000
noisy validation examples fit ensemble weights; clean CIFAR-10 test labels are
read only during final source-backbone and PCSE evaluation. `layer3` and
`layer4` are globally averaged to 64 and 128 dimensions, respectively.

The source model initializes the method-local PCSE model. `paper_volmin` then
updates that local model and transition jointly, as already defined by PCSE's
existing lifecycle. It never mutates the UPM checkpoint. The diagonal-dominant
transition parameterization and covariance ridge are explicit engineering
choices and are not claimed as paper-original hyperparameters.

## Artifacts and resume

Inspect `pretrained_best.pt`, `volmin_final.pt`, `transition_artifact.npz`,
`pcse_statistics.npz`, `pcse_gda.npz`, `pcse_ensemble.npz`, `last.pt`, and
`final_metrics.json`. Resume validates method configuration, source identity,
transition and feature-model provenance, and all artifact hashes. A completed
resume is expected to be a strict no-op.

This profile establishes real-data engineering usability only. Paper-exact
protocol, numerical table reproduction, and multi-seed aggregation remain
unverified.
