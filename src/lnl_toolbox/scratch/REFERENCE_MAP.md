# Scratch reference map

This is a development audit trail, not a runtime import map. Scratch blocks are independently implemented even when the listed legacy source was inspected.

## Implemented smoke recipes

| Method | Old sources inspected | Scratch blocks / recipe | Status |
|---|---|---|---|
| GCE | `src/lnl_toolbox/losses/torch_losses.py`; `src/lnl_toolbox/training/experiment.py`; `src/lnl_toolbox/training/data_service.py`; `src/lnl_toolbox/models/cifar_resnet.py` | Scratch-native `load_dataset → inspect_dataset_semantics → create_dataset_split → select_label_source → apply_noise → build_noise_manifest → configure_preprocessing → configure_views → assign_data_roles → configure_loader → build_prepared_data → build_loaders`, followed by `resnet34`, formula, optimization, and selection blocks; `recipes/papers/gce.yaml` | complete formal CIFAR-10 data plan; legacy data service is reference-only |
| APL | `src/lnl_toolbox/losses/torch_losses.py`; `src/lnl_toolbox/training/experiment.py` | `apl_loss`, `mean_loss`; `recipes/papers/apl.yaml` | smoke checked |
| Co-teaching | `src/lnl_toolbox/algorithms/coteaching/`; `src/lnl_toolbox/training/coteaching_experiment.py` | `linear_rate_schedule`, `select_lowest_scores`, `select_by_indices`, `mean_by_indices`, plus shared forward/CE/backward/update blocks; `recipes/papers/coteaching.yaml` | formula-ready; short lifecycle smoke |
| DivideMix | `src/lnl_toolbox/algorithms/dividemix`; `src/lnl_toolbox/training/dividemix_experiment.py` | `fit_gmm`, `threshold_mask`, `mask_to_indices`, `one_hot`, `weighted_blend`, `sharpen_distribution`, `soft_target_cross_entropy`, `mean_squared_error`, `prior_kl`, `weighted_sum`; `recipes/papers/dividemix.yaml` | formal lifecycle with public tensor/loss operations |

## Public building blocks

Runtime/data/model/forward/loss/selection/optimization/evaluation blocks are registered dynamically from Python. The first public loss set includes CE, GCE, MAE, NCE, RCE, APL, Binary Risk, Forward Correction, and Backward Correction.

### Training lifecycle

- `step_milestone_update` is a public optimizer operation for explicit global-step learning-rate milestones and is shared by L2RW and MentorNet.
- `track_best_state` / `restore_best_state` are public multi-module checkpoint operations. T-Revision, VolMinNet, Co-teaching, JoCoR, and CNL-CU pass their auxiliary revision/transition/peer module explicitly; the former paper-named wrappers were removed.
- FINE EMA/SED state, DSS cycle hooks, and MC-LDCE feature freezing remain paper-specific because they mutate structured algorithm state or enforce a paper-defined model contract.

## Formula-ready mappings

### GCE

- Legacy implementation: `src/lnl_toolbox/losses/torch_losses.py::generalized_cross_entropy` and `src/lnl_toolbox/training/experiment.py`.
- Paper source: `papers/manifest.json` entry `gce` and the corresponding local paper artifact.
- `p = softmax(z)` → `softmax`.
- `p_y = f_y(x)` → `gather_by_label` → explicit `clamp_min`.
- `L_q(f(x), y) = (1 - p_y^q) / q` → `elementwise_power` → `affine_transform`.
- `L = mean_i l_i` → `mean_loss`; optimization lifecycle → `backward` and `optimizer_step`.

### Co-teaching

- Legacy implementation: `src/lnl_toolbox/algorithms/coteaching/algorithm.py`, `selection.py`, `legacy.py`, and `src/lnl_toolbox/training/coteaching_experiment.py`.
- Paper source: `papers/manifest.json` entry `coteaching` and `02_sample_selection/09_coteaching_neurips2018.pdf`.
- `R(epoch) = 1 - min(epoch / T_k × noise_rate, noise_rate)` → `linear_rate_schedule`.
- `selected = lowest_loss(loss_per_sample, floor(R(T) × batch_size))` → `select_lowest_scores`.
- `L_A = mean(loss_A[selected_B])` → `select_by_indices` → `mean_by_indices`; same independently for peer B.
- The repository's implementation and paper mapping use peer cross-update; the separate implementation audit documents a JoCoR shared-joint-selection path. These are kept distinct rather than silently merged.

## Catalog coverage

Every catalog entry now has a recipe under `recipes/papers/`. The GCE entry is wired to the formal local CIFAR-10 reproduction contract; the remaining legacy entries are ordered Scratch smoke recipes. One-epoch execution checks belong under `recipes/examples/`.

| Catalog id | Recipe | Scratch semantic blocks | Legacy sources inspected |
|---|---|---|---|
| `pdl` | `pdl.yaml` | `pdl_instance_transition`, `transition_corrected_risk`, `compose_revision_transition` | `training/instance_transition_experiment.py`; `algorithms/instance_transition.py` |
| `jocor` | `jocor.yaml` | `symmetric_kl`, `select_lowest_scores`, `select_by_indices`, `mean_by_indices` | `algorithms/jocor.py`; `training/multi_model_experiment.py` |
| `dss` | `dss.yaml` | `dss_evidence`, `top_k_confidence` | `algorithms/dss.py`; `selectors/dss.py` |
| `cdr` | `cdr.yaml` | `parameter_criticality_mask`, `masked_gradient_update` | `algorithms/cdr.py`; `training/experiment.py` |
| `mentornet` | `mentornet.yaml` | `create_mentor_provider`, `mentor_compute_weights`, `step_milestone_update` | `training/mentor_learning.py`; `models/mentornet.py` |
| `coteaching` | `coteaching.yaml` | `linear_rate_schedule`, `select_lowest_scores`, `select_by_indices`, `mean_by_indices`, `track_best_state`, `restore_best_state` | `algorithms/coteaching/`; `training/coteaching_experiment.py` |
| `loss-correction` | `loss_correction.yaml` | `estimate_transition`, `forward_correction` | `noise/estimators.py`; `algorithms/transition_risk.py` |
| `apl` | `apl.yaml` | `apl_loss`, `mean_loss` | `losses/torch_losses.py`; `training/experiment.py` |
| `gce` | `gce.yaml` | Scratch data particles, ResNet-34, GCE formula blocks, validation selection, MultiStepLR | `losses/torch_losses.py`; `training/experiment.py`; `training/data_service.py`; `models/cifar_resnet.py` |
| `dual-t` | `dual_t.yaml` | `compose_transition`, `forward_correction` | `algorithms/dual_t/`; `training/dual_t_experiment.py` |
| `importance-reweighting` | `importance_reweighting.yaml` | `importance_reweight`, `mean_loss` | `algorithms/importance_reweighting/`; `training/importance_reweighting_experiment.py` |
| `cwd` | `cwd.yaml` | `cwd_statistics`, `mean_loss` | `estimators/cwd.py`; `training/cwd_experiment.py` |
| `pcse` | `pcse.yaml` | `create_volmin_transition`, `materialize_transition`, `softmax`, `apply_transition`, `gather_by_label`, `negative_log`, `volmin_positive_logdet`, `pcse_statistics` | `algorithms/pcse/`; `training/pcse_experiment.py` |
| `fine` | `fine.yaml` | `create_fine_state`, `fine_snapshot_predictions`, `fine_scs_select`, `fine_scr_reweight`, `select_batch_view`, `fine_prepare_batch_targets`, `fine_warmup_loss`, `fine_robust_loss`, `fine_ema_update` | `algorithms/fine.py`; `training/fine_experiment.py` |
| `cnlcu` | `cnlcu.yaml` | `create_cnlcu_history`, `append_cnlcu_history`, `cnlcu_soft_robust_mean`, `cnlcu_soft_score`, `select_lowest_scores`, `indices_to_mask`, `track_best_state`, `restore_best_state` | `algorithms/cnlcu/`; `training/cnlcu_experiment.py` |
| `t-revision` | `t_revision.yaml` | `revise_transition`, `track_best_state`, `restore_best_state` | `algorithms/t_revision/`; `training/t_revision_experiment.py` |
| `dld` | `dld.yaml` | `one_hot`, `softmax`, `subtract`, `sample_timestep_noise`, `dld_sample_forward_state`, `mean_squared_error`, `weighted_sum` | DLD diffusion predictor lifecycle remains a paper-level operation; tensor/loss composition is Scratch-native. |
| `binary-risk` | `binary_risk.yaml` | `binary_risk`, `mean_loss` | `algorithms/binary_risk.py`; `training/binary_experiment.py` |
| `volminnet` | `volminnet.yaml` | `softmax`, `apply_transition`, `gather_by_label`, `negative_log`, `mean_loss`, `volmin_positive_logdet`, `track_best_state`, `restore_best_state` | `algorithms/volminnet/`; `training/volminnet_experiment.py` |
| `upm` | `upm.yaml` | `create_upm_state`, `upm_clean_posterior`, `upm_update_eta`, `soft_target_cross_entropy` | `algorithms/upm/`; `training/upm_experiment.py` |
| `dividemix` | `dividemix.yaml` | `fit_gmm`, `threshold_mask`, `mask_to_indices`, `one_hot`, `weighted_blend`, `sharpen_distribution`, `soft_target_cross_entropy`, `mean_squared_error`, `prior_kl`, `weighted_sum` | `algorithms/dividemix/`; `training/dividemix_experiment.py` |
| `lend` | `lend.yaml` | `create_indexed_history`, `pairwise_similarity`, `topk_neighborhood`, `normalize_graph`, `propagate_labels`, `indexed_ema`, `agreement_mask`, `masked_mean` | `algorithms/lend/`; `training/lend_experiment.py` |
| `cal` | `cal.yaml` | `piecewise_rate_schedule`, `cal_warmup_objective`, `cal_prepare_proxy_batch`, `weighted_sum` | `algorithms/cal.py`; `training/cal_experiment.py` |
| `mc-ldce` | `mc_ldce.yaml` | `mc_ldce_prepare_statistic`, `mc_ldce_objective` | `estimators/mc_ldce.py`; `training/mc_ldce_experiment.py` |
| `ca2c` | `ca2c.yaml` | `per_sample_ce`, `mean_loss`, `weighted_sum`, `ca2c_cross_guidance`, `ca2c_partial_label_loss`, `ca2c_negative_label_loss` | `algorithms/ca2c.py`; `training/ca2c_experiment.py` |
| `l2rw` | `l2rw.yaml` | `get_next_batch`, `zeros_like`, `elementwise_multiply`, `sum_values`, `l2rw_virtual_update`, `l2rw_trusted_meta_loss`, `l2rw_epsilon_gradient`, `nonnegative_projection`, `normalize_nonnegative_weights` | `algorithms/l2rw.py`; `training/l2rw_experiment.py` |

Scratch data particles do not import the legacy data service at runtime. The source paths above are reference-only; algorithm/model blocks retain their existing implementation audits and are outside this data-particle migration.
