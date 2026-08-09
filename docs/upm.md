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
Completed resume is a no-op unless `upm.main.epochs` is explicitly increased.

The built-in smoke uses small CIFAR subsets and symmetric synthetic noise. It
validates lifecycle and integration only; it is not the paper's CIFAR IDN data
generation or a numerical reproduction.
