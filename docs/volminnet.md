# VolMinNet

VolMinNet is exposed through the Toolbox's unified CLI as `method: volminnet`.
It jointly trains a clean-posterior classifier and a global class-conditional
transition matrix without anchor or pseudo-anchor samples.

The implementation uses the canonical Toolbox convention
`T[i,j] = P(noisy=j | clean=i)` and computes `noisy_prob = clean_prob @ T`.
Its transition follows the paper parameterization: the unnormalized diagonal
is fixed to one, off-diagonal entries are sigmoids of trainable values, and
rows are normalized. The volume fidelity is `paper_positive_logdet`: a
non-positive or non-finite determinant fails explicitly rather than silently
using `log(abs(det(T)))`.

Run the smoke profile through the normal interface:

```powershell
lnl validate --recipe cifar10-volminnet-smoke
lnl run --recipe cifar10-volminnet-smoke --dry-run
lnl run --recipe cifar10-volminnet-smoke
lnl resume <run-directory>
```

The equivalent source-checkout entry is `python -m lnl_toolbox.cli.main`.
`--epochs N` changes `trainer.epochs`, because VolMinNet has one joint stage.

The run directory contains `resolved_config.yaml`, `environment.json`, the
noise manifest, `metrics.jsonl`, `final_metrics.json`, `last.pt`, `best.pt`,
and initial/best/last transition artifacts. Clean test labels are evaluation
only; noisy validation loss selects the paired classifier/transition best
checkpoint. The smoke profile establishes workflow behavior, not paper-level
numerical reproduction.
