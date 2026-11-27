# 配置目录说明

这里的 YAML 使用版本化公共合同。每个文件必须显式写出：

```yaml
schema_version: 1
kind: experiment  # 或 fragment / mentor_artifact
```


## 文件名迁移（旧路径 → 新路径）

只改了文件名，未改变 YAML、recipe ID、profile 或训练合同。正式 recipe 推荐使用 `lnl run --recipe <id>`，用 `lnl papers config <paper> --profile <profile> --variant <variant> --path-only` 查询当前 Paper 配置位置。对仍用旧 `--config`、Sweep `base.config` 或 `mentor prepare/train --config` 的命令，按下表替换完整路径；路径均相对于 `configs/`。不要改已有 run 目录或外部 wizard 的 `configs/experiment/cifar10_smoke.yaml`、`configs/experiment/cifar10_clean_smoke.yaml`。

<details>
<summary>查看完整旧→新路径（79 项）</summary>

| 旧路径 | 新路径 |
| --- | --- |
| `algorithm/cnlcu_hard.yaml` | `algorithm/cnlcu-hard.yaml` |
| `methods/apl/apl_cifar10_noise02_reproduction.yaml` | `methods/apl/cifar10-sym20-reproduction.yaml` |
| `methods/apl/apl_cifar10_noise02_smoke.yaml` | `methods/apl/cifar10-sym20-smoke.yaml` |
| `methods/apl/apl_cifar10_noise04_reproduction.yaml` | `methods/apl/cifar10-sym40-reproduction.yaml` |
| `methods/binary-risk/binary_risk_natarajan_1epoch.yaml` | `methods/binary-risk/synthetic-binary-2d-rcn20-10-1epoch.yaml` |
| `methods/binary-risk/binary_risk_natarajan_reproduction.yaml` | `methods/binary-risk/synthetic-binary-2d-rcn40-40-reproduction.yaml` |
| `methods/ca2c/ca2c_cifar100_reproduction.yaml` | `methods/ca2c/cifar100-sym20-reproduction.yaml` |
| `methods/ca2c/ca2c_cifar10_smoke.yaml` | `methods/ca2c/synthetic-multiclass-sym10-smoke.yaml` |
| `methods/cal/cal_cifar10_reproduction.yaml` | `methods/cal/cifar10-idn20-external-reproduction.yaml` |
| `methods/cal/cal_cifar10_smoke.yaml` | `methods/cal/synthetic-multiclass-sym10-smoke.yaml` |
| `methods/cdr/cifar10_symmetric_cdr_reproduction.yaml` | `methods/cdr/cifar10-sym40-official-code-reproduction.yaml` |
| `methods/cdr/cifar10_symmetric_cdr_smoke.yaml` | `methods/cdr/cifar10-sym40-paper-smoke.yaml` |
| `methods/clean/cifar10_clean.yaml` | `methods/clean/cifar10-tinycnn-train.yaml` |
| `methods/clean/cifar10_clean_baseline.yaml` | `methods/clean/cifar10-preact-resnet18-baseline.yaml` |
| `methods/clean/cifar10_clean_smoke.yaml` | `methods/clean/cifar10-tinycnn-smoke.yaml` |
| `methods/cnlcu/cifar10_cnlcu_hard_smoke.yaml` | `methods/cnlcu/cifar10-sym40-hard-smoke.yaml` |
| `methods/cnlcu/cifar10_cnlcu_soft_smoke.yaml` | `methods/cnlcu/cifar10-sym40-soft-smoke.yaml` |
| `methods/cnlcu/cnlcu_cifar10_reproduction.yaml` | `methods/cnlcu/cifar10-sym20-soft-reproduction.yaml` |
| `methods/cnlcu/support/cifar10_cnlcu_soft_sym20_short.yaml` | `methods/cnlcu/support/cifar10-sym20-soft-short.yaml` |
| `methods/coteaching/cifar10_coteaching_reproduction.yaml` | `methods/coteaching/cifar10-sym20-reproduction.yaml` |
| `methods/coteaching/cifar10_coteaching_smoke.yaml` | `methods/coteaching/cifar10-sym40-smoke.yaml` |
| `methods/cwd/cwd_cifar10_reproduction.yaml` | `methods/cwd/cifar10-airplane-automobile-rcn20-20-scalar-reproduction.yaml` |
| `methods/cwd/cwd_cifar10_smoke.yaml` | `methods/cwd/cifar10-airplane-automobile-rcn20-20-smoke.yaml` |
| `methods/cwd/support/cwd_cifar10_1epoch.yaml` | `methods/cwd/support/cifar10-airplane-automobile-rcn20-20-1epoch.yaml` |
| `methods/dividemix/cifar10_dividemix_smoke.yaml` | `methods/dividemix/cifar10-sym20-smoke.yaml` |
| `methods/dividemix/cifar10_dividemix_sym20.yaml` | `methods/dividemix/cifar10-sym20-reproduction.yaml` |
| `methods/dld/cifar10_dld_smoke.yaml` | `methods/dld/cifar10-sym20-tinycnn-smoke.yaml` |
| `methods/dld/dld_cifar10_reproduction.yaml` | `methods/dld/cifar10-sym20-resnet34-external-reproduction.yaml` |
| `methods/dld/support/cifar10_dld_sym20_short.yaml` | `methods/dld/support/cifar10-sym20-resnet34-external-short.yaml` |
| `methods/dss/dss_cifar10_symmetric05_reproduction.yaml` | `methods/dss/cifar10-sym50-reproduction.yaml` |
| `methods/dss/dss_cifar10_symmetric05_smoke.yaml` | `methods/dss/cifar10-sym50-smoke.yaml` |
| `methods/dual-t/cifar10_dual_t_smoke.yaml` | `methods/dual-t/cifar10-sym20-smoke.yaml` |
| `methods/dual-t/cifar10_dual_t_sym20.yaml` | `methods/dual-t/cifar10-sym20-reproduction.yaml` |
| `methods/dual-t/evidence/cifar10_dual_t_evidence_smoke.yaml` | `methods/dual-t/evidence/cifar10-sym20-evidence-smoke.yaml` |
| `methods/dual-t/evidence/cifar10_dual_t_sym20_evidence.yaml` | `methods/dual-t/evidence/cifar10-sym20-evidence-reproduction.yaml` |
| `methods/fine/fine_cifar100n_reproduction.yaml` | `methods/fine/cifar100-sym20-reproduction.yaml` |
| `methods/fine/fine_cifar100n_smoke.yaml` | `methods/fine/cifar100-sym20-smoke.yaml` |
| `methods/fine/support/fine_cifar100n_1epoch.yaml` | `methods/fine/support/cifar100-sym20-warmup-1epoch.yaml` |
| `methods/gce/gce_cifar10_noise02_reproduction.yaml` | `methods/gce/cifar10-sym20-reproduction.yaml` |
| `methods/gce/gce_cifar10_noise02_smoke.yaml` | `methods/gce/cifar10-sym20-smoke.yaml` |
| `methods/importance-reweighting/importance_reweighting_binary_high_dim.yaml` | `methods/importance-reweighting/synthetic-binary-high-dim-rcn20-10-kliep-reproduction.yaml` |
| `methods/importance-reweighting/importance_reweighting_binary_high_dim_smoke.yaml` | `methods/importance-reweighting/synthetic-binary-high-dim-rcn20-10-kliep-smoke.yaml` |
| `methods/importance-reweighting/importance_reweighting_binary_low_dim.yaml` | `methods/importance-reweighting/synthetic-binary-2d-rcn20-10-kde-reproduction.yaml` |
| `methods/importance-reweighting/importance_reweighting_binary_smoke.yaml` | `methods/importance-reweighting/synthetic-binary-2d-rcn20-10-kde-smoke.yaml` |
| `methods/importance-reweighting/uci_heart_importance_reweighting.yaml` | `methods/importance-reweighting/uci-statlog-heart-rcn30-10-kliep-experiment.yaml` |
| `methods/jocor/jocor_cifar10_symmetric05_reproduction.yaml` | `methods/jocor/cifar10-sym50-reproduction.yaml` |
| `methods/jocor/jocor_cifar10_symmetric05_smoke.yaml` | `methods/jocor/cifar10-sym50-smoke.yaml` |
| `methods/l2rw/l2rw_cifar10_reproduction.yaml` | `methods/l2rw/cifar10-uniformflip40-reproduction.yaml` |
| `methods/l2rw/l2rw_cifar10_smoke.yaml` | `methods/l2rw/synthetic-multiclass-sym20-smoke.yaml` |
| `methods/l2rw/support/l2rw_cifar10_1epoch.yaml` | `methods/l2rw/support/cifar10-sym40-audited-trusted-1epoch.yaml` |
| `methods/lend/cifar10_lend_smoke.yaml` | `methods/lend/cifar10-sym20-smoke.yaml` |
| `methods/lend/lend_cifar10_reproduction.yaml` | `methods/lend/cifar10-sym40-reproduction.yaml` |
| `methods/loss-correction/loss_correction_cifar10_asymmetric04.yaml` | `methods/loss-correction/cifar10-class-conditional40-known-t-reproduction.yaml` |
| `methods/mc-ldce/mc_ldce_cifar10_reproduction.yaml` | `methods/mc-ldce/cifar10-sym20-reproduction.yaml` |
| `methods/mc-ldce/mc_ldce_cifar10_smoke.yaml` | `methods/mc-ldce/synthetic-multiclass-sym10-smoke.yaml` |
| `methods/mc-ldce/support/mc_ldce_cifar10_1epoch.yaml` | `methods/mc-ldce/support/cifar10-sym20-estimated-t-1epoch.yaml` |
| `methods/mentornet/mentornet_dd_cifar100_symmetric04_reproduction.yaml` | `methods/mentornet/cifar100-sym40-student-reproduction.yaml` |
| `methods/mentornet/mentornet_dd_cifar100_symmetric04_smoke.yaml` | `methods/mentornet/cifar100-sym40-student-smoke.yaml` |
| `methods/mentornet/producer/mentornet_dd_teacher_cifar10_symmetric04.yaml` | `methods/mentornet/producer/cifar10-sym40-teacher-producer.yaml` |
| `methods/mentornet/support/mentornet_cifar10_symmetric04_1epoch.yaml` | `methods/mentornet/support/cifar10-sym40-student-1epoch.yaml` |
| `methods/pcse/cifar10_pcse_reproduction.yaml` | `methods/pcse/cifar10-sym40-paper-volmin-reproduction.yaml` |
| `methods/pcse/pcse_multiclass_smoke.yaml` | `methods/pcse/synthetic-multiclass-sym10-dual-t-smoke.yaml` |
| `methods/pcse/pcse_multiclass_volmin_smoke.yaml` | `methods/pcse/synthetic-multiclass-sym10-paper-volmin-smoke.yaml` |
| `methods/pdl/pdl_cifar10_reproduction.yaml` | `methods/pdl/cifar10-pdl40-reproduction.yaml` |
| `methods/pdl/pdl_cifar10_smoke.yaml` | `methods/pdl/cifar10-pdl40-smoke.yaml` |
| `methods/supervised-baselines/cifar10_foundation_smoke.yaml` | `methods/supervised-baselines/cifar10-sym40-anchor-smoke.yaml` |
| `methods/supervised-baselines/cifar10_smoke.yaml` | `methods/supervised-baselines/cifar10-clean-ce-smoke.yaml` |
| `methods/supervised-baselines/cifar10_symmetric_ce_smoke.yaml` | `methods/supervised-baselines/cifar10-sym40-ce-smoke.yaml` |
| `methods/supervised-baselines/cifar10_symmetric_small_loss_linear_smoke.yaml` | `methods/supervised-baselines/cifar10-sym40-small-loss-linear-smoke.yaml` |
| `methods/supervised-baselines/cifar10_symmetric_small_loss_smoke.yaml` | `methods/supervised-baselines/cifar10-sym40-small-loss-fixed-smoke.yaml` |
| `methods/t-revision/cifar10_t_revision_smoke.yaml` | `methods/t-revision/cifar10-sym40-smoke.yaml` |
| `methods/t-revision/cifar10_t_revision_sym20_reproduction.yaml` | `methods/t-revision/cifar10-sym20-reproduction.yaml` |
| `methods/t-revision/cifar10_t_revision_sym20_short.yaml` | `methods/t-revision/cifar10-sym20-short.yaml` |
| `methods/upm/cifar10_upm_smoke.yaml` | `methods/upm/cifar10-sym20-smoke.yaml` |
| `methods/upm/upm_cifar10_reproduction.yaml` | `methods/upm/cifar10-sym40-reproduction.yaml` |
| `methods/volminnet/cifar10_volminnet_smoke.yaml` | `methods/volminnet/cifar10-sym40-smoke.yaml` |
| `methods/volminnet/legacy/volmin_cifar10_reproduction.yaml` | `methods/volminnet/legacy/cifar10-sym40-reproduction.yaml` |
| `methods/volminnet/legacy/volmin_cifar10_smoke.yaml` | `methods/volminnet/legacy/synthetic-multiclass-sym20-smoke.yaml` |
| `methods/volminnet/volminnet_cifar10_reproduction.yaml` | `methods/volminnet/cifar10-sym20-reproduction.yaml` |

</details>

## v1.0 术语

| 术语 | 在 Toolbox 中的含义 |
| --- | --- |
| Paper | 论文及其实现说明。Paper catalog 使用稳定的 Paper ID，并通过 `recipe_id` 关联一个或多个内置配方。 |
| Method / runner | Method 是用户选择或看到的方法，runner 是 `execution.runner` 指定的执行器。多个方法可以复用同一 runner；例如 GCE 使用 `supervised` runner 和 `loss.name: gce`。 |
| Config / recipe | Config 是配置内容。内置 recipe 是 catalog 登记且可用公开 ID 查找的完整实验配置。用户也可按路径运行未登记的完整 YAML。 |
| Profile | `smoke` 用于短预算链路检查，`reproduction` 用于论文相关正式实验，`experiment` 用于其他配置。内置 recipe 的发现 profile 来自 manifest，不取决于 YAML 文件名或所在目录。 |
| Variant | Paper catalog 的 `variant` 是选择某个数据集、噪声或协议配置的标签。实现变体由方法专用 YAML 字段选择，实际含义由 runner 决定。 |
| Fidelity | `configuration_fidelity` 描述配置的协议或实现对齐边界。内置 recipe 优先使用 YAML 的显式值；未声明时，按 discovery profile `smoke` / `reproduction` / `experiment` 分别得到 `smoke` / `paper_oriented` / `engineering`，不读取文件名或目录。Paper 关联可覆盖该标签，并继续兼容旧字段 `fidelity`。 |
| Reproduction status | `reproduction_status` 记录复现证据状态，默认 `not_run`。正式 profile、fidelity 标签和 smoke 通过均不能证明论文数值已复现。 |

## 新增内置 recipe

1. 将完整实验写为 `kind: experiment`，使用 `snake_case` 的字段、注册键和 YAML 文件名，沿用现有 runner、方法和组件的精确注册名。参考片段用 `kind: fragment`，MentorNet 等生产者配置按各自入口使用，不能当作普通训练 recipe 登记。
2. 为新配方选择全局唯一的稳定 ID，使用小写字母、数字和连字符。文件名可用 `_smoke`、`_reproduction` 后缀帮助读者辨识，但不定义 ID、profile 或 fidelity。不要因重命名 YAML 而改变已发布的 ID。
3. 在 `src/lnl_toolbox/cli/data/recipe_catalog.json` 的 `recipes` 中登记恰含 `id`、`path`、`profile` 的对象，例如现有 GCE smoke 条目：`{"id": "gce-cifar10-noise02-smoke", "path": "configs/methods/gce/cifar10-sym20-smoke.yaml", "profile": "smoke"}`。不接受裸路径字符串。同步 `pyproject.toml` 的 `tool.setuptools.data-files`，按需以相同路径登记 `conditional` 或精选 `public` 条目。文件存在或被打包均不会自动登记 recipe。
4. 若配方应出现在 Paper 页面，在 `src/lnl_toolbox/paper_catalog.json` 补齐 `recipe_id`、`profile` 和用于选择的 `variant`（新标签用连字符分隔）。保留该 Paper 的正式默认顺序，代码取第一个 `reproduction` 条目。新增条目的 Paper profile 应与发现 profile 一致。已有 Loss Correction 条目的 Paper profile 为 `reproduction`，发现 profile 为 `experiment`，属于保留的历史例外。
5. 若配方替换正式参数编辑基准或新增可编辑字段，检查 `web/lnl_parameter_metadata_registry_revised.yaml` 中的正式绑定与字段路径。运行配置、catalog 关联和 Web 参数展示分别承担执行、导航和编辑职责。

移动内置 YAML 时，仍须同步 manifest 的资源路径、打包声明和适用的 `public` / `conditional` 路径引用，并核对直接 `--config` 示例、旧交互菜单及 packaged-only producer 等特定路径消费者。公开菜单和条件配方仍按路径登记，在主 `recipes` 顺序上做筛选。

旧 CLI 交互向导会重写训练参数，内置菜单只从源码 catalog 按顺序提供适合编辑的 CIFAR-10/100 模板。Clean 向导接受无噪声、模型受支持的 clean/supervised 配置；通用向导要求 supervised runner 和显式顶层 TinyCNN；两者均不自动纳入 conditional 配方。默认 ID 仍是 `cifar10-clean-smoke` 和 `cifar10-smoke`。完整论文及多阶段实验请用 `lnl run --recipe <id>`，不要用向导代替原配方。

向导另从当前项目的 `configs/experiment/*.yaml` 发现本地文件，保留独立用户项目的同名优先与无候选时的手输路径。源码根目录中的未登记文件只有符合向导资格才会显示为本地项；这避免将专用工作流和测试配置混入普通模板。Producer 用专用入口运行；packaged-only 辅助配置、开发例与任意自定义 YAML 可按路径使用。独立安装不自动从包内填充向导菜单；`--recipe` 和显式 `--config` 不受影响。

**GCE（现有 loss 和 runner）。** 参考 `methods/gce/cifar10-sym20-reproduction.yaml`。新增同类配方时沿用 `supervised` 和 `loss.name: gce`，完成 YAML、manifest 和打包路径登记，再按需关联 GCE Paper。沿用现有字段和正式编辑基准时，无需修改 Web 元数据。

**DLD（专用生命周期与前置条件）。** 参考 `methods/dld/cifar10-sym20-resnet34-external-reproduction.yaml`，保留 `dld` runner 及其特征提取、扩散和推理阶段。完成相同的路径登记及 Paper 关联；如果所用正式配置依赖外部特征来源，还需登记 `conditional`。只有正式参数基准或可编辑字段变化时，才同步 Web 绑定。

## 配置参考

下列字段都放在完整的 `kind: experiment` 配置中；从对应完整 recipe 开始，再按需要修改。历史示例值不等于论文推荐值。

- **基础 loss 与噪声。** 在 `loss` 下用 `name: ce` 选择 CE；在 `noise` 下用 `name: symmetric`、`rate` 和 `seed` 分别指定噪声类型、翻转率和随机种子。见 [CE + symmetric 完整示例](methods/supervised-baselines/cifar10-sym40-ce-smoke.yaml)。
- **APL / NCE。** `loss.name: apl` 组合正权重 `alpha` / `beta` 与 `active.name: nce` / `passive.name: rce`。例如自定义 `loss.passive.log_zero: -4.0` 并不是[现行 APL 正式值](methods/apl/cifar10-sym20-reproduction.yaml) `log(1e-4)`。NCE 也可作为注册的独立 loss：使用 `loss.name: nce`，按需设置 `loss.eps`，字段见 [NCE 片段](algorithm/nce.yaml)。
- **CDR。** `cdr.compatibility_mode` 区分 [`paper`](methods/cdr/cifar10-sym40-paper-smoke.yaml) 和 [`official_code`](methods/cdr/cifar10-sym40-official-code-reproduction.yaml)；结合 recipe 检查 `cdr.noise_rate`、`cdr.l1_decay` 和 `cdr.critical_scope`，不能把 smoke 参数当作正式协议。
- **CNLCU。** 把 [CNLCU-H 片段](algorithm/cnlcu-hard.yaml)的有效参数用在完整配置的 `cnlcu` 节中；窗口、最少观测数、fixed loss bound 和 corrected LOF 共同定义 hard 路径。相对地，[CNLCU-S](methods/cnlcu/cifar10-sym40-soft-smoke.yaml)使用固定长度 float32 loss history、soft uncertainty、线性 remember 日程以及 stable sample index 的并列裁决。
- **Co-teaching / JoCoR。** 从 [Co-teaching 完整配置](methods/coteaching/cifar10-sym40-smoke.yaml)检查实际噪声率与 remember schedule 的配对；改噪声设置时需重新核对 样本选择比例，而不是把 `noise.rate` 当成 keep rate。 [JoCoR](methods/jocor/cifar10-sym50-smoke.yaml) 的 `algorithm.name: jocor` / `algorithm.lambda` 与 `selector.name: small_loss` / `selector.keep_rate` 配套；floor 是选择计数规则。
- **DSS / GCE。** [DSS 完整配置](methods/dss/cifar10-sym50-reproduction.yaml)包含 warmup、MDA 和 CCS/masking 阶段，单个 loss 片段不能表达该流程。[GCE 完整配置](methods/gce/cifar10-sym20-reproduction.yaml)使用 `loss.name: gce` 和 `loss.q` 控制目标形状；标准 GCE 不隐含截断阈值。
- **Loss Correction。** [已知 T 的 forward 正式配置](methods/loss-correction/cifar10-class-conditional40-known-t-reproduction.yaml)使用预先指定的转移矩阵，不进行预训练；未知 T 的 [Anchor 两阶段 smoke](methods/supervised-baselines/cifar10-sym40-anchor-smoke.yaml)先以正的 warmup 预算训练估计模型，再用 `pipeline.transition_estimator` 估计 T 并进入校正训练。
- **MentorNet。** `pipeline.weight_provider` 配置 Student 消费的 MentorArtifact 及运行日程，如 percentile、decay、burn-in 和 dropout。旧自定义示例的 decay `0.5` 不等价于[正式 Student 配置](methods/mentornet/cifar100-sym40-student-reproduction.yaml)的 `0.95`；artifact 必须满足该 Student 的身份要求。
- **IDN / PDL。** 通用 IDN 的入口是 Python `lnl_toolbox.noise.generators.generate_instance_dependent(labels, class_scores, rate, seed, dataset=...)`，其中分数形状为 `[N, C]`；旧 `score_weighted_idn` YAML 片段不是现行训练 schema 的噪声类型。PDL 的 Algorithm 2 噪声生成产生 noisy-label manifest，和 [PDL 训练配置](methods/pdl/cifar10-pdl40-reproduction.yaml)的实例转移估计/校正不是同一步。

CNLCU-H 和 NCE 两份保留片段不是完整实验，也不是完整算法配置库。它们不参与自动组合，不会改变正式 recipe；引用其中参数时须放入上面说明的配置上下文。

## 未登记的科研支持配置

以下完整配置可按新路径使用 `--config`，不属于 recipe discovery 或发布包，仍须准备原配置所需数据和资源：

| 配置 | 验证边界 |
| --- | --- |
| [CWD 1 epoch](methods/cwd/support/cifar10-airplane-automobile-rcn20-20-1epoch.yaml) | CIFAR-10 airplane/automobile，五折中的 fold 0，非 `binary_scalar`；一次运行不覆盖全部五折。 |
| [FINE 1 epoch](methods/fine/support/cifar100-sym20-warmup-1epoch.yaml) | CIFAR-100 合成对称噪声；只走 warmup，不消费真实 CIFAR-100N 标签，也不验证 robust 阶段。 |
| [L2RW 1 epoch](methods/l2rw/support/cifar10-sym40-audited-trusted-1epoch.yaml) | 需要配置指定的 audited trusted manifest；用来检查 bilevel 更新，不等同于完整实验。 |
| [MC-LDCE 1 epoch](methods/mc-ldce/support/cifar10-sym20-estimated-t-1epoch.yaml) | 包含真实 T 估计到主训练的交接；不以测试矩阵代替该阶段。 |
| [MentorNet CIFAR-10 Student](methods/mentornet/support/cifar10-sym40-student-1epoch.yaml) | 使用配置所指的冻结 MentorArtifact；不要假定与正式 CIFAR-100 Student 的 artifact 可互换。 |
| [VolMin smoke](methods/volminnet/legacy/synthetic-multiclass-sym20-smoke.yaml) / [reproduction](methods/volminnet/legacy/cifar10-sym40-reproduction.yaml) | 独立 `volmin` runner，联合学习模型和 T，以 clean validation 选择 checkpoint，保留原 `last.pt` / resume 合同。现行 `volminnet` 使用 noisy validation 并导出 best/last portable T；旧 VolMin 不是其前置步骤。 |

`lnl doctor` 检查正在运行的源码或安装包中的 **68 份已登记配置资源**，包括 conditional 定义；这不检查方法是否具备外部数据或 artifact。三份只打包、不登记的 YAML 由打包验证另行核验。工作目录无需包含旧 `configs/experiment/`；用户仍可在该目录保存自己的配置。用显式 `--config` 检查指定配置，加 `--check-data` 才请求该配置的数据检查。

历史配置见[归档索引](../archive/configs/README.md)。支持配置迁移后，按旧完整路径运行的命令需更新；自定义 YAML 位置约定不变。

## 用户可见配方

活动 YAML 是内部可复现资产，不等于都需要出现在用户菜单中。
`src/lnl_toolbox/cli/data/recipe_catalog.json` 的 `public` 清单维护默认公开模板：

- `lnl list experiments` 只显示少量公开模板；
- `lnl list experiments --all` 显示全部内置实验配置；
- Web 新手模式只显示公开模板；
- Web“新建 YAML”默认按 26 篇论文各显示一份正式配置；“通用监督组合”模式继续使用少量公开模板；
- Web 的 Recipe 编辑页仍可访问全部内部配置。

论文正式配置模式只复制论文目录指定的完整 YAML，不把专用 runner 改写成通用监督配置，
也不允许在复制时混入 loss/selector 等跨生命周期覆盖。生成后可在 Recipe 编辑页审阅和修改。
界面同时显示 `paper_protocol`、`paper_oriented` 或 `engineering` 等 fidelity，避免把完整预算
误称为已完成论文数值复现。

Smoke 配置会缩小数据、模型或训练预算以检查链路；正式配置还定义模型、优化器、
scheduler、数据规模和论文生命周期，不能只通过修改 epoch 将 Smoke 当作正式复现。

`kind: experiment` 表示可直接交给 `lnl run` 的完整配置；`kind: fragment` 仅保存参数
片段，不能独立运行。历史版本原样保存在
`archive/configs-legacy-2026-08-21/`，并由 SHA-256 清单保护。

完整实验统一使用 `execution.runner`、`data.name`、`loader`、`model`、`optimizer` 和
`trainer`。论文特有阶段仍保留独立 mapping，不得为了外观统一而改变训练生命周期。
`noise.type` 以及顶层 `epochs/batch_size/learning_rate` 已废弃。

`data.name` 选择数据适配器；本机位置由 `lnl data register` 登记。完整 recipe 通常不写
`data.root`。当同一适配器只有一个本地登记时会自动解析；存在多个登记时必须用
`lnl run ... --data <alias>` 明确选择。

训练实验通过顶层 `loss` mapping 选择逐样本 PyTorch objective，缺省时兼容为 CE：

```yaml
loss:
  name: gce
  q: 0.7
```

`configs/algorithm/` 保存标记为 `kind: fragment` 的参考片段；当前入口未实现 Hydra
defaults 合并，因此完整实验 YAML 必须显式包含所需参数。

顶层 `selector` mapping 选择单 batch 的 hard sample selector。缺省时使用 `all`，与原有全样本均值训练一致：

```yaml
selector:
  name: small_loss
  keep_rate: 0.5
```

`small_loss` 的固定浮点配置保持兼容，也可显式写为 constant schedule：

```yaml
selector:
  name: small_loss
  keep_rate:
    name: constant
    value: 0.8
```

linear schedule 使用零基 epoch：epoch 0 为 `start`，epoch `warmup_epochs` 达到 `end`，之后固定为 `end`。

```yaml
selector:
  name: small_loss
  keep_rate:
    name: linear
    start: 1.0
    end: 0.6
    warmup_epochs: 10
```

所有 rate 必须位于 `(0, 1]`，`warmup_epochs` 必须为正整数。keep rate 不从 noise rate 推导。当前通用训练路径只提供无状态的 `all` 和 `small_loss`。Selector 只接收 detached 的逐样本 loss、stable global index 和当前零基 epoch；不读取 clean label，也不管理优化器或训练生命周期。Co-teaching 的双网络 peer exchange 仍属于独立 Algorithm/Pipeline，不使用该配置替代。

通用训练入口支持两种互斥噪声来源。运行开始时生成：

```yaml
noise:
  name: symmetric       # 或 pairflip
  rate: 0.4
  seed: 17
  manifest_filename: noise_manifest.npz
```

读取已经生成的 Noise Manifest：

```yaml
noise:
  manifest: data/noise/cifar10-symmetric-0.4-seed1.npz
  manifest_filename: noise_manifest.npz
```

`manifest` 不能与 `name/rate/seed` 混用。缺少 `noise` 时保持干净标签训练；配置后仅训练集按 global index 使用噪声标签，validation/test 仍使用干净标签。最终映射固定保存到运行目录，恢复时不重新生成、也不重新依赖外部源文件。`lnl-clean-train` 是 clean-only 包装，会拒绝 `noise`；噪声实验使用 `lnl-train`。

## Conditional recipes / prerequisites

有些 recipe 是 conditional：配置文件存在不表示当前机器已经能够运行。PCSE、DLD、CAL 和 MentorNet 等流程可能分别需要 checkpoint/source run、真实 pretrained feature extractor、对齐的 external clean/noisy label artifact 或已验证的 MentorArtifact。

来源可由 YAML 中的路径、run directory/environment、external label artifact 或 MentorArtifact 提供。readiness 至少会经过配置、schema、identity compatibility 和 consumer validation；仅 `PATH_EXISTS` 不等于 `READY`。不要把任意 `.pt` 文件、随机冻结模型或未对齐标签文件当作正式来源。

本地数据登记同样按 adapter source contract 进行：单文件 adapter 使用 `lnl data register ... --path <file>`，目录 adapter 使用 `lnl data register ... --root <directory>`。不要同时传递无关的 source 参数。

这里仅说明配置合同；Web 操作、DatasetProfile 和具体方法的 prerequisite 指引请见 [Toolbox 使用手册](../docs/toolbox-usage-manual.md)。

