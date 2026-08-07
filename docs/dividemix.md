# DivideMix workflow

The public `method: dividemix` runner implements a paper/official-oriented
two-network workflow. Both peers are warmed up independently. At the start of
each main epoch, both full-training loss snapshots are frozen before either
peer is updated. The existing two-component GMM estimator produces continuous
clean probabilities. Probabilities produced by A divide data for B, and those
produced by B divide data for A; self-divide is forbidden.

Labeled targets use co-refinement, unlabeled targets use predictions from both
peers, and the method-private MixMatch objective combines soft-target CE,
unlabeled probability MSE, and the uniform-prior regularizer. The official
CIFAR inference policy is `logits_A + logits_B`.

Artifacts named `dividemix_epoch_XXXX.npz` are written to a temporary file,
reloaded and validated, then atomically published. `last.pt` owns both models,
optimizers, schedulers, RNG, loss histories and the current artifact identity.
Resume is exact at warmup epoch, co-divide, network-A-ready, and full-epoch
boundaries; arbitrary minibatch resume is not claimed.

```powershell
lnl validate --recipe cifar10-dividemix-smoke
lnl run --recipe cifar10-dividemix-smoke --dry-run
lnl run --recipe cifar10-dividemix-smoke
lnl resume <run-directory>
```

`--epochs N` changes only `dividemix.training.epochs`; it does not change the
warmup budget. Noisy validation selects the paired best checkpoint. Clean test
labels are used only in the final evaluation.
