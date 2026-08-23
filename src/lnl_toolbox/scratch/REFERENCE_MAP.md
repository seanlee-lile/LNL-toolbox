# Scratch reference map

This is a development audit trail, not a runtime import map. Scratch blocks are independently implemented even when the listed legacy source was inspected.

## Implemented smoke recipes

| Method | Old sources inspected | Scratch blocks / recipe | Status |
|---|---|---|---|
| GCE | `src/lnl_toolbox/losses/torch_losses.py`; `src/lnl_toolbox/training/experiment.py`; `src/lnl_toolbox/training/data_service.py`; `src/lnl_toolbox/models/cifar_resnet.py` | `prepare_gce_cifar10`, `resnet34`, `forward`, `softmax_probability`, `gather_target_probability`, `gce_q_formula`, `mean_loss`, `backward`, `optimizer_step`, validation selection, and MultiStepLR; `recipes/papers/gce.yaml` | complete formal CIFAR-10 path; `recipes/examples/gce_formula_smoke.yaml` remains the one-epoch smoke |
| APL | `src/lnl_toolbox/losses/torch_losses.py`; `src/lnl_toolbox/training/experiment.py` | `apl_loss`, `mean_loss`; `recipes/papers/apl.yaml` | smoke checked |
| Co-teaching | `src/lnl_toolbox/algorithms/coteaching/`; `src/lnl_toolbox/training/coteaching_experiment.py` | `remember_rate_formula`, `small_loss_indices`, `cross_select_loss_a_from_b`, `cross_select_loss_b_from_a`, plus shared forward/CE/backward/update blocks; `recipes/papers/coteaching.yaml` | formula-ready; short lifecycle smoke |
| DivideMix | `src/lnl_toolbox/algorithms/dividemix`; `src/lnl_toolbox/training/dividemix_experiment.py` | `warmup`, `fit_gmm`, `split_clean_noisy`, `co_refine`, `mixmatch_step`; `recipes/papers/dividemix.yaml` | short lifecycle smoke |

## Public building blocks

Runtime/data/model/forward/loss/selection/optimization/evaluation blocks are registered dynamically from Python. The first public loss set includes CE, GCE, MAE, NCE, RCE, APL, Binary Risk, Forward Correction, and Backward Correction.

## Formula-ready mappings

### GCE

- Legacy implementation: `src/lnl_toolbox/losses/torch_losses.py::generalized_cross_entropy` and `src/lnl_toolbox/training/experiment.py`.
- Paper source: `papers/manifest.json` entry `gce` and the corresponding local paper artifact.
- `p = softmax(z)` → `softmax_probability`.
- `p_y = f_y(x)` → `gather_target_probability`.
- `L_q(f(x), y) = (1 - p_y^q) / q` → `gce_q_formula`.
- `L = mean_i l_i` → `mean_loss`; optimization lifecycle → `backward` and `optimizer_step`.

### Co-teaching

- Legacy implementation: `src/lnl_toolbox/algorithms/coteaching/algorithm.py`, `selection.py`, `legacy.py`, and `src/lnl_toolbox/training/coteaching_experiment.py`.
- Paper source: `papers/manifest.json` entry `coteaching` and `02_sample_selection/09_coteaching_neurips2018.pdf`.
- `R(epoch) = 1 - min(epoch / T_k × noise_rate, noise_rate)` → `remember_rate_formula`.
- `selected = lowest_loss(loss_per_sample, floor(R(T) × batch_size))` → `small_loss_indices`.
- `L_A = mean(loss_A[selected_B])` → `cross_select_loss_a_from_b`; `L_B = mean(loss_B[selected_A])` → `cross_select_loss_b_from_a`.
- The repository's implementation and paper mapping use peer cross-update; the separate implementation audit documents a JoCoR shared-joint-selection path. These are kept distinct rather than silently merged.

## Catalog coverage

Every catalog entry now has a recipe under `recipes/papers/`. The GCE entry is wired to the formal local CIFAR-10 reproduction contract; the remaining legacy entries are ordered Scratch smoke recipes. One-epoch execution checks belong under `recipes/examples/`.

| Catalog id | Recipe | Scratch semantic blocks | Legacy sources inspected |
|---|---|---|---|
| `pdl` | `pdl.yaml` | `pdl_instance_transition` | `training/instance_transition_experiment.py`; `algorithms/instance_transition.py` |
| `jocor` | `jocor.yaml` | `jocor_agreement`, `small_loss` | `algorithms/jocor.py`; `training/multi_model_experiment.py` |
| `dss` | `dss.yaml` | `dss_evidence`, `top_k_confidence` | `algorithms/dss.py`; `selectors/dss.py` |
| `cdr` | `cdr.yaml` | `cdr_parameter_mask` | `algorithms/cdr.py`; `training/experiment.py` |
| `mentornet` | `mentornet.yaml` | `mentor_weight` | `training/mentor_learning.py`; `models/mentornet.py` |
| `coteaching` | `coteaching.yaml` | `peer_exchange`, `select_by_indices` | `algorithms/coteaching/`; `training/coteaching_experiment.py` |
| `loss-correction` | `loss_correction.yaml` | `estimate_transition`, `forward_correction` | `noise/estimators.py`; `algorithms/transition_risk.py` |
| `apl` | `apl.yaml` | `apl_loss`, `mean_loss` | `losses/torch_losses.py`; `training/experiment.py` |
| `gce` | `gce.yaml` | `prepare_gce_cifar10`, ResNet-34, GCE formula blocks, validation selection, MultiStepLR | `losses/torch_losses.py`; `training/experiment.py`; `training/data_service.py`; `models/cifar_resnet.py` |
| `dual-t` | `dual_t.yaml` | `compose_transition`, `forward_correction` | `algorithms/dual_t/`; `training/dual_t_experiment.py` |
| `importance-reweighting` | `importance_reweighting.yaml` | `importance_reweight`, `mean_loss` | `algorithms/importance_reweighting/`; `training/importance_reweighting_experiment.py` |
| `cwd` | `cwd.yaml` | `cwd_statistics`, `mean_loss` | `estimators/cwd.py`; `training/cwd_experiment.py` |
| `pcse` | `pcse.yaml` | `pcse_statistics` | `algorithms/pcse/`; `training/pcse_experiment.py` |
| `fine` | `fine.yaml` | `fine_feature_filter` | `algorithms/fine.py`; `training/fine_experiment.py` |
| `cnlcu` | `cnlcu.yaml` | `cnlcu_uncertainty` | `algorithms/cnlcu/`; `training/cnlcu_experiment.py` |
| `t-revision` | `t_revision.yaml` | `revise_transition` | `algorithms/t_revision/`; `training/t_revision_experiment.py` |
| `dld` | `dld.yaml` | `dld_label_diffusion` | `algorithms/dld/`; `training/dld_experiment.py` |
| `binary-risk` | `binary_risk.yaml` | `binary_risk`, `mean_loss` | `algorithms/binary_risk.py`; `training/binary_experiment.py` |
| `volminnet` | `volminnet.yaml` | `volminnet_objective` | `algorithms/volminnet/`; `training/volminnet_experiment.py` |
| `upm` | `upm.yaml` | `upm_eta_update` | `algorithms/upm/`; `training/upm_experiment.py` |
| `dividemix` | `dividemix.yaml` | `warmup`, `fit_gmm`, `split_clean_noisy`, `co_refine`, `mixmatch_step` | `algorithms/dividemix/`; `training/dividemix_experiment.py` |
| `lend` | `lend.yaml` | `lend_label_dilution` | `algorithms/lend/`; `training/lend_experiment.py` |
| `cal` | `cal.yaml` | `cal_second_order_risk` | `algorithms/cal.py`; `training/cal_experiment.py` |
| `mc-ldce` | `mc_ldce.yaml` | `mc_ldce_centroid_risk` | `estimators/mc_ldce.py`; `training/mc_ldce_experiment.py` |
| `ca2c` | `ca2c.yaml` | `ca2c_candidate_memory` | `algorithms/ca2c.py`; `training/ca2c_experiment.py` |
| `l2rw` | `l2rw.yaml` | `l2rw_meta_weight` | `algorithms/l2rw.py`; `training/l2rw_experiment.py` |

No Scratch block imports these legacy modules at runtime. The source paths above are reference-only and correspond to the `implementation_paths` in `src/lnl_toolbox/paper_catalog.json`.
