# Scratch reference map

This is a development audit trail, not a runtime import map. Scratch blocks are independently implemented even when the listed legacy source was inspected.

## Implemented smoke recipes

| Method | Old sources inspected | Scratch blocks / recipe | Status |
|---|---|---|---|
| GCE | `src/lnl_toolbox/losses/torch_losses.py`; `src/lnl_toolbox/training/experiment.py` | `gce_loss`, `mean_loss`; `recipes/papers/gce.yaml` | smoke checked |
| APL | `src/lnl_toolbox/losses/torch_losses.py`; `src/lnl_toolbox/training/experiment.py` | `apl_loss`, `mean_loss`; `recipes/papers/apl.yaml` | smoke checked |
| Co-teaching | `src/lnl_toolbox/algorithms/coteaching`; `src/lnl_toolbox/training/coteaching_experiment.py` | `peer_exchange`, `select_by_indices`; `recipes/papers/coteaching.yaml` | short lifecycle smoke |
| DivideMix | `src/lnl_toolbox/algorithms/dividemix`; `src/lnl_toolbox/training/dividemix_experiment.py` | `warmup`, `fit_gmm`, `split_clean_noisy`, `co_refine`, `mixmatch_step`; `recipes/papers/dividemix.yaml` | short lifecycle smoke |

## Public building blocks

Runtime/data/model/forward/loss/selection/optimization/evaluation blocks are registered dynamically from Python. The first public loss set includes CE, GCE, MAE, NCE, RCE, APL, Binary Risk, Forward Correction, and Backward Correction.

## Remaining catalog work

The catalog currently lists these methods and their source-of-truth paths. They still require paper-by-paper equation/default/lifecycle review before being advertised as complete Scratch recipes:

`PDL`, `JoCoR`, `DSS`, `CDR`, `MentorNet`, `Loss Correction`, `Dual-T`, `Importance Reweighting`, `CWD`, `PCSE`, `FINE`, `CNLCU`, `T-Revision`, `DLD`, `VolMinNet`, `UPM`, `LEND`, `CAL`, `MC-LDCE`, `CA2C`, and `L2RW`.

No Scratch block imports these legacy modules at runtime. Their paths are recorded in `src/lnl_toolbox/paper_catalog.json` and should be inspected before each future recipe is added.
