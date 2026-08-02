# Importance Reweighting Reproduction and Maintenance Guide

This guide fixes two maintained workflows for Li et al.,
*Classification with Noisy Labels by Importance Reweighting*. The method
uses a noisy-label posterior estimator appropriate to the data dimension:
KDE for the low-dimensional route and KLIEP density-ratio estimation for
the high-dimensional route. Both backends produce the same
`PosteriorSnapshot [N, 2]` contract and feed the same paper raw-min
noise-rate estimator and importance-weighted empirical-risk minimization.

The current goal is to validate both complete processing chains. These
synthetic configurations do not reproduce all datasets, baselines, or
numerical tables from the paper.

## Supported Scope

The current implementation supports:

- binary classification only, with labels encoded as `{0, 1}`;
- asymmetric binary random classification noise;
- low-dimensional KDE noisy-posterior estimation;
- high-dimensional KLIEP density-ratio posterior estimation;
- sample-aligned `PosteriorSnapshot [N, 2]` artifacts;
- the paper raw-min estimates of `rho_positive` and `rho_negative`;
- the paper-exact binary importance-weight formula;
- weighted per-sample cross entropy reduced by explicit `batch_mean`;
- stable global sample indices throughout posterior lookup and training;
- atomic posterior and noise-rate artifact publication;
- strict checkpoint and resume validation;
- checkpoint selection on observed/noisy validation labels and final
  evaluation on a clean test split.

Clean labels are not supplied to the training algorithm. The current runner
does not use clean validation targets: validation labels are noisy by
contract, while the test labels remain clean.

## Not Yet Covered

This version does not include:

- UCI datasets;
- hinge loss;
- multiclass importance reweighting;
- automatic KDE or KLIEP bandwidth selection;
- the full set of paper baselines;
- formal paper tables or paper-level numerical reproduction;
- multi-seed mean and standard deviation;
- a claim that the synthetic distribution exactly reproduces a paper
  dataset.

Smoke accuracy demonstrates only that the pipeline is executable. A single
maintained long run does not establish superiority over a baseline.

## Data Flows

The low-dimensional flow is:

```text
synthetic_binary_2d
-> KDE noisy-posterior producer
-> PosteriorSnapshot [N, 2]
-> paper raw-min noise rates
-> exact binary importance weights
-> batch-mean weighted cross entropy
-> noisy-validation checkpoint selection
-> clean-test evaluation
```

The high-dimensional flow is:

```text
synthetic_binary_high_dim
-> KLIEP density-ratio noisy-posterior producer
-> PosteriorSnapshot [N, 2]
-> paper raw-min noise rates
-> exact binary importance weights
-> batch-mean weighted cross entropy
-> noisy-validation checkpoint selection
-> clean-test evaluation
```

KDE and KLIEP replace only the posterior producer. The raw-min estimator,
`NoiseRateArtifact`, indexed weight provider, exact importance-weight
formula, reducer, final training step, and checkpoint lifecycle are shared.

For KLIEP, the two class-specific density ratios are estimated independently.
The implementation multiplies each ratio by its empirical noisy-class prior
and normalizes the two resulting scores row-wise. This finite-sample
normalization is an explicit repository implementation choice.

## Maintained Configurations

| Workflow | Configuration |
|---|---|
| low-dimensional KDE | `configs/reproduction/importance_reweighting_binary_low_dim.yaml` |
| high-dimensional KLIEP | `configs/reproduction/importance_reweighting_binary_high_dim.yaml` |

Both maintained configurations use 4,096 training examples, 1,024 noisy
validation examples, 1,024 clean test examples, 50 final-training epochs,
fixed data/noise seeds, and a linear two-logit classifier. They are larger
than the tiny smoke configurations but remain suitable for a routine
single-machine maintenance run.

The following KLIEP values are implementation choices and must not be
described as paper-specified hyperparameters:

| Parameter | Maintained value |
|---|---:|
| `bandwidth` | 4.0 |
| `max_centers` | 24 |
| `max_iterations` | 200 |
| `learning_rate` | 0.02 |
| `tolerance` | 1e-7 |
| `epsilon` | 1e-12 |
| `seed` | 37 |

The synthetic sample counts, optimizer settings, batch size, training
budget, device selection, and fixed seeds are also repository maintenance
choices. They make the workflow reproducible but are not paper-table
hyperparameters.

## Fresh Runs

Low-dimensional KDE:

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.train `
  --config configs/reproduction/importance_reweighting_binary_low_dim.yaml `
  --output-dir artifacts/reproductions/importance-reweighting-low-dim-seed17
```

High-dimensional KLIEP:

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.train `
  --config configs/reproduction/importance_reweighting_binary_high_dim.yaml `
  --output-dir artifacts/reproductions/importance-reweighting-high-dim-seed17
```

Generated artifacts under `artifacts/` are local experiment outputs and
must not be committed.

## Resume

Resume an interrupted low-dimensional run:

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.train `
  --config configs/reproduction/importance_reweighting_binary_low_dim.yaml `
  --resume artifacts/reproductions/importance-reweighting-low-dim-seed17/last.pt
```

Resume an interrupted high-dimensional run:

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.train `
  --config configs/reproduction/importance_reweighting_binary_high_dim.yaml `
  --resume artifacts/reproductions/importance-reweighting-high-dim-seed17/last.pt
```

`--epochs <total>` may be used only when deliberately extending the total
final-training budget. It denotes the target total epoch count, not the
number of additional epochs.

## Artifacts and Provenance

At minimum, inspect:

- `last.pt`: the only resumable checkpoint;
- `posterior_snapshot.npz`: stable-index noisy posterior values;
- `noise_rate_artifact.npz`: raw-min rates and source-snapshot identity;
- `final_metrics.json`: completed epochs, global step, best validation
  metric, clean-test metrics, estimated rates, and artifact hashes.

The checkpoint records:

- `posterior_backend_identity`, including backend name, feature dimension,
  effective estimator parameters, and implementation version;
- `posterior_backend_hash`, the canonical hash of that identity;
- `posterior_snapshot_hash`, binding posterior values, noisy targets, stable
  indices, dataset, and split;
- `noise_rate_artifact_hash`;
- the rate artifact's `source_snapshot_hash`;
- method phase, completed epochs, and global step.

The source-snapshot hash proves that the raw-min rates came from the
persisted posterior snapshot. It does not prove paper-level numerical
reproduction.

## Resume Acceptance

A valid resume must satisfy all of the following:

- completed epochs and global step continue rather than restart;
- `posterior_snapshot.npz` is loaded, not regenerated;
- `noise_rate_artifact.npz` is loaded, not regenerated;
- artifact content hashes and modification times remain unchanged;
- backend identity and backend hash remain unchanged;
- rate-artifact provenance still points to the snapshot hash;
- a KDE checkpoint cannot resume with a KLIEP configuration;
- changed KLIEP bandwidth, centers, iteration, optimizer, tolerance,
  epsilon, seed, or feature dimension is rejected;
- missing or corrupted artifacts cause an explicit failure and are not
  silently rebuilt.

For a maintenance run, record file SHA-256 and modification times before
and after resume:

```powershell
Get-FileHash <run>/posterior_snapshot.npz -Algorithm SHA256
Get-FileHash <run>/noise_rate_artifact.npz -Algorithm SHA256
Get-Item <run>/posterior_snapshot.npz, <run>/noise_rate_artifact.npz |
  Select-Object Name, Length, LastWriteTimeUtc
```

## Long-Test Record Template

```text
Commit:
Config:
Device:
Start time:
End time:
Fresh run / Resume:
Completed epochs:
Global step:
Backend:
Backend hash:
Snapshot hash:
Rate artifact hash:
Validation metric:
Test metric:
NaN/Inf:
Artifact unchanged after resume:
Result:
Notes:
```

## Result Interpretation

- Tiny smoke accuracy proves only that the implementation and lifecycle run.
- A single longer synthetic run cannot establish that Importance
  Reweighting outperforms noisy CE or another baseline.
- These maintained configurations do not reproduce UCI experiments.
- No paper table or multi-seed mean/std is currently required or available.
- Report the workflows as tested binary KDE and KLIEP processing chains,
  not as a complete reproduction of every experiment in the paper.

## Source

- Tongliang Liu and Dacheng Tao. *Classification with Noisy Labels by
  Importance Reweighting*. IEEE Transactions on Pattern Analysis and
  Machine Intelligence, 2016.
- arXiv: <https://arxiv.org/abs/1411.7718>
