# DLD workflow

The toolbox exposes Directional Label Diffusion (DLD) through the standard
`doctor -> list -> validate -> dry-run -> run -> resume` interface. The current
implementation is a **paper-oriented workflow**, not a paper numerical or
released-code exact reproduction.

The built-in smoke performs:

```text
noisy CIFAR train
-> frozen repository feature model with weak/strong views
-> cosine-distance weighted KNN distributions
-> KL(p_s || p_w) and deterministic two-component GMM partition
-> paper-oriented y0 and yn, with yd = yn - y0
-> independent direction/noise predictors and optimizers
-> deterministic five-step reverse sampling
-> noisy-validation checkpoint selection
-> clean-test reporting
```

Run it with:

```powershell
python -m lnl_toolbox.cli.main validate --recipe cifar10-dld-smoke
python -m lnl_toolbox.cli.main run --recipe cifar10-dld-smoke --dry-run
python -m lnl_toolbox.cli.main run --recipe cifar10-dld-smoke
python -m lnl_toolbox.cli.main resume <run-directory>
```

`--epochs N` changes only `dld.diffusion.epochs`. The run publishes
`dld_precorrection.npz`, `last.pt`, `best.pt`, `metrics.jsonl`, and
`final_metrics.json`. Resume validates the manifest, stable sample mapping,
frozen-feature identity, transformations, pre-correction artifact, schedule,
inference policy, and fidelity identity. A completed resume with the same epoch
target is a no-op.

The zero inference initialization is an explicit engineering policy. The smoke
uses a randomly initialized frozen TinyCNN and synthetic symmetric noise only to
exercise lifecycle correctness; it does not reproduce the paper's representation
backend, instance-dependent noise protocol, or reported accuracy.
