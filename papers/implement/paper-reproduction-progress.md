# 论文复现进度与参数台账

> 归档位置与维护规则见本目录的 `README.md`。

本文件是论文复现的唯一进度入口。`paper-implementation-guideline.md` 说明算法应如何实现；本文件只记录“是否已经实际运行”、使用了哪些参数、产物在哪里以及结果与原文相差多少。

状态定义：

- `未开始`：只有论文阅读或接口规划；
- `组件完成`：核心公式或组件通过测试，但没有完整论文实验；
- `Smoke 通过`：论文配置的小样本闭环通过；
- `单次复现`：至少完成一次论文规模实验；
- `完整复现`：覆盖论文要求的重复次数、设置和汇总指标。

## 总表

| # | 论文/方法 | 当前组件 | 复现状态 | 下一缺口 |
|---:|---|---|---|---|
| 1 | UPM | 指南 | 未开始 | InstanceNoiseModel、PosteriorRefiner、Pipeline |
| 2 | CAL | 指南 | 未开始 | StatisticEstimator、RiskCorrector |
| 3 | PDL | 指南 | 未开始 | 实例级 `T(x)` 与 Pipeline |
| 4 | JoCoR | Selector 基础 | 未开始 | 双网络 Algorithm |
| 5 | DSS | BASE/MDA/CCS、masked risk、Objective lifecycle、split-aware manifest | Smoke 通过 | 唯一一次 150-epoch 正式实验与论文曲线比较 |
| 6 | CDR | 论文/官方双模式 ParameterUpdatePolicy、完整 noisy-validation Pipeline | Smoke 通过 | 对齐后单次 100-epoch 正式实验与曲线比较 |
| 7 | CNLCU | Selector 基础 | 未开始 | loss history、不确定性与双网络 |
| 8 | MentorNet | WeightProvider 基础 | 未开始 | Mentor/Student Pipeline |
| 9 | Co-teaching | small-loss/交换 helper | 未开始 | 双网络训练与恢复 |
| 10 | Loss Correction | Anchor/known estimator、Forward/Backward RiskCorrector、通用 Pipeline | 单次复现 | 多 seed、其他噪声设置 |
| 11 | Normalized Loss/APL | NCE、MAE、RCE、APL | 单次复现 | 如需完整复现，再补多 seed 与其他噪声设置 |
| 12 | GCE | 标准 GCE | 单次复现 | 后续如需完整复现，再补 5 次重复与其他噪声设置 |
| 13 | VolMinNet | Transition 基础 | 未开始 | 可训练 NoiseModel |
| 14 | Natarajan Risk | Weight/Risk 基础 | 未开始 | 二分类 RiskCorrector |
| 15 | T-Revision | Anchor/Weight 基础 | 未开始 | 三阶段 Pipeline |
| 16 | Dual-T | TransitionEstimator、PosteriorSnapshot、通用 Pipeline | Pipeline smoke 通过 | 论文实验与参数核对 |
| 17 | MC-LDCE | 指南 | 未开始 | StatisticEstimator、global objective |
| 18 | Importance Reweighting | Binary RCN WeightProvider、通用权重接入 | Pipeline smoke 通过 | posterior/rate estimator 与论文实验 |
| 19 | CWD | 指南 | 未开始 | class-wise StatisticEstimator |
| 20 | PCSE | 指南 | 未开始 | 特征统计与 post-processing |
| 21 | DLD | 指南 | 未开始 | diffusion label Pipeline |
| 22 | FINE | Selector 基础 | 未开始 | forgetting/negative regularizer |
| 23 | CA2C | 双网络接口待建 | 未开始 | asymmetric Pipeline |
| 24 | DivideMix | Selector 基础 | 未开始 | GMM、MixMatch、双网络 Pipeline |
| 25 | L2RW | WeightProvider 基础 | 未开始 | clean meta-batch 与 MetaUpdater |
| 26 | LEND | 指南 | 未开始 | FeatureGraph 与 indexed label state |

## 复现增量与复用审计表

该表用于约束“菜谱组合原料”的实现方式。`计划中` 只表示已经审阅并保存实施计划，不表示代码已经存在。

| 论文 | 复用原料 | 新增/扩展通用原料 | 论文专属菜谱 | 新增依据 | 后续使用者 | 状态 |
|---|---|---|---|---|---|---|
| GCE | CIFAR Dataset、Noise Manifest、SGD/scheduler、统一 runner、checkpoint、逐样本 Loss 协议 | `GeneralizedCrossEntropyLoss`；CIFAR ResNet-34；`gce2018` per-pixel mean preprocessing；可选 noisy validation；终端进度与 SVG 曲线 | `gce_cifar10_noise02_smoke.yaml`、`gce_cifar10_noise02_reproduction.yaml` | GCE 公式和论文实验模型/预处理此前不在生产闭环；进度能力为所有长实验共用 | 后续 robust loss、transition correction 及其他 CIFAR 长实验 | 单次复现；详见 [GCE 结果](gce/result.md) |
| APL | CIFAR Dataset、Noise Manifest、NCE、RCE、APL、SGD/cosine、统一 runner、checkpoint、metrics | 通用 `CifarCnn8`；symmetric sampling；StandardUpdatePolicy gradient clipping；显式 model-selection split | `apl_cifar10_noise02_reproduction.yaml`、`apl_cifar10_noise04_reproduction.yaml` | 已新增通用 `numpy_legacy + per_class` 官方噪声策略；Normalize 已改为作者精确常数；模型源码仍需官方 `models.py` 可得后逐行确认 | 其他 CIFAR loss 论文、使用 symmetric noise 或梯度裁剪的实验 | 旧配置两次单次复现；20% `79.25%`，40% `66.61%`；官方噪声对齐后待重跑 |

## 单篇复现记录

- GCE：[单次复现结果](gce/result.md)
- APL：[单次复现计划](apl/plan.md)；已完成 CIFAR-10 symmetric 0.2、NCE+RCE、seed 1 的一次 120 epochs 运行，best epoch 113，test accuracy `79.25%`，test selection leakage 已明确记录。
- APL 对照修正：标准 CIFAR-10 Normalize 已采用作者代码精确常数；40% 实验使用 global symmetric sampling，以减少与作者数据生成逻辑的偏差；`CifarCnn8` 当前结构保持不变，待官方模型源码可取得后再做逐层确认。
- APL 40%：完成 CIFAR-10 symmetric 0.4、NCE+RCE、seed 1、120 epochs；best epoch 119，test accuracy `66.61%`；产物位于 `artifacts/reproductions/20260726-144152/`，test selection leakage 已明确记录。
- APL 官方噪声对齐：已实现通用 `sampling: per_class` + `rng: numpy_legacy`，未运行实验；此前 40% 结果使用旧 global/default_rng 配置，不作为对齐后结果。
- 共享代码整理：CDR optimizer 约束已移回 `CDRUpdatePolicy`；新增通用 `loss_for_all_targets`、`RiskCorrector`、Forward/Backward 校正器、WeightProvider、Pipeline、统一 warm-up、FeatureSnapshot、TargetProvider、IndexedHistory、EarlyStopping、MultiModel/PeerExchange、Instance/Trainable Transition、Statistic/Neighbor Artifact 和 checkpoint RNG 状态；Dual-T 已通过 warm-up → snapshot → artifact → correction 闭环 smoke，GCE 保持通用 loss。
## 本轮底座维护记录（2026-07-27）

- 目标范围：Binary Risk、Loss Correction、CWD、FINE；APL 不在本轮复现范围内。
- 状态：四个目标的通用组件已实现并通过 focused/smoke tests；尚未运行正式论文实验。
- 参数规则：每篇论文只抽取一组候选参数；抽样 seed、候选集合、来源和解析配置由 `ParameterRecord` 保存。
- 曲线规则：正式训练结束后读取 `metrics.jsonl`，通过 `curve_comparison.py` 生成叠加曲线、差值 CSV 和比较摘要。
- 测试：本轮相关测试通过；完整测试中既有 DivideMix GMM 测试受 conda 环境 SciPy `_propack` DLL 问题影响，未归因于本轮改动。
- 下一步：下一轮先确定四篇论文各自的一组论文原始参数和数据切片，再逐篇运行，不重复实验。

## Loss Correction 单次复现记录（2026-07-28）

- 配置：`configs/experiment/loss_correction_cifar10_asymmetric04.yaml`；CIFAR-10、官方 class-conditional asymmetric `eta=0.4`、32-layer ResNet、Forward correction、seed 1。
- 训练：120 epochs；SGD `lr=0.01`、momentum `0.9`、weight decay `1e-4`；第 40/80 epoch 后学习率分别降至 `0.001/0.0001`；batch size 128。
- 结果：best epoch 48，noisy validation accuracy `72.78%`，clean test accuracy `85.34%`；实际训练集 corruption rate `19.796%`（官方映射仅作用于受影响类别）。
- 论文对照：Patrini et al.（CVPR 2017，Table 2）在 CIFAR-10 32-layer ResNet、asymmetric `N=0.2`（与本配置约 20% 实际翻转率对应）的 Forward correction clean test accuracy 为 `89.9%`；本次低 `4.56` 个百分点。论文该组为单次运行，无标准差。
- 对照结论：训练通路已完成，但结果尚未精确对齐论文；`72.78%` 为 noisy validation，不能直接与论文 clean test 比较，后续需继续核查复现差异。
- 产物：`artifacts/reproductions/20260728-010054/`，包含 `metrics.jsonl`、`best.pt`、`last.pt`、`posterior_snapshot.npz`、`transition_artifact.npz` 和 `training_curves.svg`。
- 验证：Loss Correction、pipeline、noise、transition estimator、plugin focused tests 通过；完整 unittest 为 241/249 通过，8 项仍为既有 DivideMix/Scipy `_propack` DLL 环境错误。

## CDR 单次复现记录（2026-07-28）

- 配置：`configs/experiment/cifar10_symmetric_cdr_reproduction.yaml`；CIFAR-10、symmetric 40%、ResNet-50、seed 1、10% noisy validation。
- 论文参数：batch size 64、SGD、lr `0.01`、momentum `0.9`、论文 L1 decay `0.001`、optimizer weight decay `0`、100 epochs、milestones `[40, 80]`、gamma `0.1`。
- 说明：论文正文规定 CIFAR 实验 100 epochs；官方仓库对 CIFAR 默认覆盖为 200 epochs，本次按论文正文执行。论文 Table 1 的 CDR symmetric-40% CIFAR-10 参照为 `62.72 ± 0.38%`（5 次运行均值±标准差），本次仅运行 1 次，不计算标准差。
- 状态：已完成通用 CIFAR ResNet-50 构造、正式配置和 focused tests；正式训练已从同一 checkpoint 一次性完成到 epoch 30/100，尚未完成论文规定的 100 epochs。
- 阶段结果：best epoch 18，noisy validation accuracy `16.32%`，clean test accuracy `21.54%`；仅作 30-epoch 阶段记录，不作为最终复现结果。
- 历史产物目录：`artifacts/reproductions/20260728-125547/`；保留 `last.pt`、`best.pt`、`metrics.jsonl` 和 `training_curves.svg`。后续对齐检查确认该运行配置与官方数据/模型通路不同，不能继续用于正式复现。

### CDR 对齐维护记录（2026-07-28）

- 当前任务：对齐 CDR 论文、官方代码与 Toolbox 通路；当前为 detached HEAD，
  基线提交 `ccc9146`。
- 完成度：5/5（100%）：官方流程核对、通用数据对齐、模型对齐、CDR 双模式、
  测试与文档维护。
- 通用增量：random/stratified split、可配置 CIFAR mean/std、legacy transition
  sampling、ResNet stem padding/初始化选项；均保留原默认行为。
- CDR 增量：`paper` 模式保持 Eq. (3)-(6) 与 L1；`official_code` 模式复现
  二维/四维 scope、threshold ties 和 optimizer L2；runner 无论文名称分支。
- 配置：正式配置采用论文公式与 100 epochs，同时按官方数据/模型代码设置
  random split、seed 1 transition noise、Normalize、stem padding 0 和默认初始化。
- 验证：相关 focused 78 项通过；完整 unittest `259/259` 通过；Conda `pytorch`
  环境未复现 SciPy `_propack` DLL 错误；CUDA smoke 位于
  `artifacts/runs/20260728-175427/`。
- 旧 30-epoch 记录使用 global fixed-count noise、不同 noise seed、旧 Normalize
  与 stem，不能与新配置 resume，保留为历史诊断产物，不计作对齐后正式实验。
- 阻塞：无代码阻塞；尚未运行对齐后的正式 100 epochs。
- 精确下一步：使用当前正式配置仅运行一次 100 epochs，完成后生成论文曲线比较；
  当前修改尚未提交，也未准备推送。

## DSS 底座与 Smoke 记录（2026-07-28）

- 当前任务：实现基础 DSS 单网络复现通路；当前为 detached HEAD，基线提交
  `ccc9146`。
- 精确目标：BASE + MDA + CCS、candidate masked CE、epoch-boundary 状态、
  checkpoint/resume、官方数据划分/噪声 RNG 和单次正式配置。
- 完成度：6/7（85.71%）：论文/官方通路核对、通用 Objective lifecycle、
  DSS 数学与状态、split-aware manifest、配置与参数抽样、测试/Smoke 和文档维护
  已完成；唯一一次 150-epoch 正式训练与曲线比较未执行。
- 抽样记录：sampling seed `20260728`；从论文出现的 CIFAR-10
  symmetric-50%、asymmetric-40%、IDN-50% 中抽中 symmetric-50%；官方 seeds
  1–5 中抽中训练 seed 4。
- 正式参数：PreActResNet-18、batch 128、SGD lr `0.02`、momentum `0.9`、
  weight decay `0.001`、150 epochs、milestone 80、gamma `0.1`、warm-up 30、
  MDA decay `0.99`、CCS alpha `0.10`、noisy validation。
- 模块化边界：`training/experiment.py` 和 `training/pipeline.py` 本任务未修改；
  DSS 通过 `objective_consumer/dss` 插件接入，Pipeline 复用既有 component-state
  checkpoint。
- 数据对齐：新增 `classwise_legacy` split 和 per-split RNG Noise Manifest；
  训练与验证分别以 seed 4 执行 symmetric transition sampling，clean label 不进入
  Objective/selector。
- Smoke：`artifacts/test-runs/dss-cifar10-smoke/` 完成 2 epochs；epoch 1
  selected ratio 1.0，epoch 2 为 0.078125；checkpoint 恢复成功。
- 测试：DSS focused 9 项、split-manifest 3 项和受影响模块测试通过；完整
  unittest `271/271` 通过。
- 文件维护：已更新 `docs/file-map.md`、`docs/data-flow-guide.md`、
  `papers/lnl-26-paper-module-coverage.md` 和 implementation guideline。
- 本地 checkpoint commits：无；history cleanup 尚不需要；当前未准备 push。
- 正式训练状态：按用户要求跳过全部训练流程，当前无训练进程。此前两个正式启动目录
  `artifacts/reproductions/dss-cifar10-sym05-seed4/` 和
  `artifacts/reproductions/dss-cifar10-sym05-seed4-valid/` 均误用了环境中旧版安装包，
  checkpoint 不含 DSS component state，因此仅作为诊断产物，不计入复现结果；
  `dss-cifar10-sym05-seed4-current-source/` 未形成有效训练产物。
- 阻塞：无代码或数据阻塞；正式训练由用户主动跳过。
- 精确下一步：保持底座与 Smoke 状态，不运行正式实验；仅在用户重新明确授权训练后，
  先以前台导入校验确认当前工作区源码，再执行唯一一次 150-epoch 正式运行。
