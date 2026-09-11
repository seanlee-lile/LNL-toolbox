# UPM method

`method: upm` implements the two-stage Universal Probabilistic Model workflow
from Wang et al. (AAAI 2021). Stage 1 trains a naive classifier on noisy labels
and publishes a deterministic train-only posterior snapshot from its best
noisy-validation checkpoint. For sample `i`, UPM uses only the scalar
`psi_i = P(observed_noisy_label_i | x_i)` gathered from that snapshot.

Stage 2 creates a fresh classifier and a stable-indexed `eta[N]` state. Each
batch computes detached true-label posterior `q` with paper Eq. (8), optionally
updates the batch's `eta` values using explicit projected gradient ascent from
Eq. (11)-(12), and updates the classifier using the same fixed `q` as a soft
target. Clean train or validation labels are not consumed; clean test labels
are used only for final evaluation.

## User workflow

```powershell
lnl validate --recipe cifar10-upm-smoke
lnl run --recipe cifar10-upm-smoke --dry-run
lnl run --recipe cifar10-upm-smoke --output-dir <run-dir>
lnl resume <run-dir>
```

The module entry `python -m lnl_toolbox.cli.main` is equivalent. For UPM,
`--epochs N` changes only `upm.main.epochs`; it never changes Stage 1.

`last.pt` is the sole resume checkpoint. The run also writes
`stage1_best.pt`, `psi_snapshot.npz`, `eta_initial.npz`, `eta_best.npz`,
`eta_last.npz`, `best.pt`, `metrics.jsonl`, and `final_metrics.json`.
`lnl resume <run-dir>` only resumes the already resolved configuration; it has
no `--epochs` override. A completed run is therefore a no-op under that
configuration. To increase only `upm.main.epochs`, copy or update the YAML,
run `lnl run --config <updated.yaml> --dry-run`, then resume explicitly with
`lnl run --config <updated.yaml> --resume <run-dir>/last.pt`. Stage 1 and the
identity settings must remain compatible.

The built-in smoke uses small CIFAR subsets and symmetric synthetic noise. It
validates lifecycle and integration only; it is not the paper's CIFAR IDN data
generation or a numerical reproduction.

## Formal engineering profile and producer boundary

`upm-cifar10-reproduction` is the current full-budget CIFAR-10 engineering
profile. It is a runnable formal workflow, but its catalog
`reproduction_status` remains `not_run`; recipe availability, Dry-run, or a
successful smoke must not be described as numerical reproduction.

UPM artifacts can be consumed by PCSE or a compatible DLD source adapter only
when the consumer validates an explicit role, checkpoint schema, model/dataset
identity, NoiseManifest, stable mapping, and recorded digests. An arbitrary
UPM checkpoint is not a valid source merely because it has a `.pt` suffix.
