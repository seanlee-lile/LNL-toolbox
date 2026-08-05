# T-Revision Reweight-R

## When to use it

Use `method: t_revision` for multiclass classification when label noise is
well described by one class-dependent, instance-independent transition matrix,
only noisy training/validation labels are available, and an Anchor estimate is
expected to benefit from a learned additive revision. A non-empty noisy
validation split is required because all three best checkpoints use noisy
validation accuracy; clean test labels are evaluation-only.

Do not use this workflow for instance-dependent or sample-specific transition
noise, open-set noise, multilabel targets, regression, missing training
classes, or clean-validation checkpoint selection. Use Forward Correction when
a trustworthy fixed transition matrix is already available, Dual-T when the
goal is factorized transition estimation without additive joint revision, and
Co-teaching/CNLCU when sample selection is more appropriate than a fixed
class-transition assumption. No method is universally best for every noise
process.

## Public entry point

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.train `
  --config configs/experiment/cifar10_t_revision_smoke.yaml `
  --output-dir artifacts/runs/t-revision-smoke
```

Resume only from the run-local `last.pt`:

```powershell
python -m lnl_toolbox.cli.train `
  --config configs/experiment/cifar10_t_revision_smoke.yaml `
  --resume artifacts/runs/t-revision-smoke/last.pt
```

`--epochs N` changes only the final revision-stage total for T-Revision. It
does not retrain Stage 1 or Stage 2A. The same completed configuration is a
strict no-op; a larger revision target continues from saved model, Adam,
scheduler, RNG, and delta state. Reducing the target or changing another
identity field is rejected. A cosine scheduler must set `t_max` explicitly if
the revision target may be extended.

## Required configuration

The maintained example is
`configs/experiment/cifar10_t_revision_smoke.yaml`. These choices must remain
explicit:

- `method: t_revision`;
- CIFAR-10 or CIFAR-100 data with positive `validation_size`;
- symmetric, pairflip, or an external fixed class-dependent noise manifest;
- `noise.validation_targets: noisy`;
- `objective: reweight` and `fidelity: paper_experiment_raw_additive`;
- every stage's positive epoch target, optimizer, and scheduler;
- deterministic train pseudo-anchor initialization;
- Stage 2A starts from `stage1_best`;
- Stage 2B starts from `classifier_initialization_best`, uses zero delta and
  Adam over exactly the classifier parameters plus delta;
- ratio policy `detach: false`, `clamp: none`, and an explicit non-negative
  `denominator_floor`;
- `evaluation.selection_split: validation`.

Model width, learning rates, weight decay, schedules, batch size, augmentation,
seed, and device are experiment choices. The smoke configuration validates the
workflow; it is not a paper-number reproduction preset.

## Automatic lifecycle

```text
noisy CE Stage 1
-> noisy-validation Stage 1 best
-> deterministic train PosteriorSnapshot
-> pseudo-anchor T_hat
-> fixed-T Reweight-R classifier initialization (Stage 2A)
-> noisy-validation Stage 2A best
-> joint classifier + raw additive delta revision (Stage 2B)
-> noisy-validation revision best
-> clean-test evaluation
```

The convention is `T[i,j] = P(noisy=j | clean=i)` and
`p_noisy = p_clean @ T`. Reweight-R uses the undetached ratio
`p_clean[y] / (p_clean @ T)[y]`. Stage 2B deliberately follows the released
experiment lifecycle's unconstrained `T_hat + delta`: it is not projected,
clipped, normalized, or guaranteed to be a probability transition matrix.

## Outputs and interpretation

A completed run contains:

- `resolved_config.yaml`, `environment.json`, `noise_manifest.npz`, and
  `noise_summary.json`;
- `stage1_best.pt`, `posterior_snapshot.npz`, and `transition_initial.npz`;
- `stage2a_best.pt`, `best.pt`, `last.pt`, and `transition_revised.npz`;
- `metrics.jsonl` and `final_metrics.json`.

`last.pt` is the only resume checkpoint. The three best files are evaluation
and provenance artifacts. `final_metrics.json` records completion phase, all
best epochs and noisy-validation metrics, best-model clean-test accuracy,
artifact paths/hashes, pseudo-anchor indices, initial matrix diagnostics,
delta norms, and explicit finite/non-negative/row-stochastic flags for the raw
revised matrix. Large matrices remain in NPZ artifacts. For synthetic noise,
`true_T_relative_L1_error` is diagnostic only and never affects training or
checkpoint selection.

## Resume and failure boundaries

Resume validates method/config identity, manifest mapping, best checkpoint
hashes, posterior/transition provenance, optimizer/scheduler state, delta, RNG,
and revised-artifact integrity. Existing initial artifacts are not silently
regenerated. A corrupt or missing artifact fails explicitly.

Preflight rejects unsupported datasets, empty validation, overlapping splits,
missing observed training classes, sample-specific transition manifests,
manifest/model class-count mismatch, invalid model output shape, clean
validation, and unsupported objective/transition/ratio policies. During
training, non-finite logits, transition values, denominators, weights, delta,
or objective fail at the consuming stage.

## Fidelity boundary

This is a complete callable Reweight-R workflow with corrected vectorized Eq.
(3) and explicit released-code lifecycle choices. It must not be described as
paper-exact T-Revision, a Forward-R implementation, a projected-transition
variant, or a formal long-run/multi-seed numerical reproduction.
