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
| 3 | PDL | Algorithm 2、Eq. 1/2/4、实例转移 Algorithm 与独立多阶段 runner | Smoke 通过 | 单次正式论文实验与曲线比较 |
| 4 | JoCoR | 双网络 Algorithm、共同 small-loss、通用 multi-model runner | 正式实验中断（161/200） | 续训至 200 epochs 后完成最终论文比较 |
| 5 | DSS | BASE/MDA/CCS、masked risk、Objective lifecycle、split-aware manifest | 单次正式复现完成 | 论文曲线比较 |
| 6 | CDR | 论文/官方双模式 ParameterUpdatePolicy、完整 noisy-validation Pipeline | Smoke 通过 | 对齐后单次 100-epoch 正式实验与曲线比较 |
| 7 | CNLCU | Selector 基础 | 未开始 | loss history、不确定性与双网络 |
| 8 | MentorNet | MentorArtifact、bi-LSTM Mentor、状态化 WeightProvider、step 调度、ResNet-101、5000-sample trusted data | 正式数据与 smoke 通过 | 唯一一次 CIFAR-100 Sym-40 39k-step 训练 |
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
| 19 | CWD | Eq. 19/21--30 estimator、global objective、独立 CIFAR-binary runner | 组件完成 | 正式 200-epoch 单 fold 运行与论文曲线比较 |
| 20 | PCSE | 指南 | 未开始 | 特征统计与 post-processing |
| 21 | DLD | 指南 | 未开始 | diffusion label Pipeline |
| 22 | FINE | EMA、SCS/SCR、强增强、两项 regularizer、独立两阶段 runner | 组件完成 | 正式 300-epoch 单次运行与论文曲线比较 |
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
- 完成度：7/7（100%）：论文/官方通路核对、通用 Objective lifecycle、DSS 数学与状态、
  split-aware manifest、配置与参数抽样、测试/Smoke、唯一一次 150-epoch 正式训练和文档维护
  已完成；论文曲线比较待补。
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
- 正式复现：`artifacts/reproductions/dss-cifar10-sym05-seed4-managed/` 完成 150 epochs；
  best epoch `121`，best noisy-validation accuracy `45.84%`，clean test accuracy `88.50%`；
  `test_selection_leakage=false`，checkpoint 为 format v2 并包含 `objective_consumer` 状态。
- 文件维护：已更新 `docs/file-map.md`、`docs/data-flow-guide.md`、
  `papers/lnl-26-paper-module-coverage.md` 和 implementation guideline。
- 本地 checkpoint commits：无；history cleanup 尚不需要；当前未准备 push。
- 历史无效启动状态：此前两个正式启动目录
  `artifacts/reproductions/dss-cifar10-sym05-seed4/` 和
  `artifacts/reproductions/dss-cifar10-sym05-seed4-valid/` 均误用了环境中旧版安装包，
  checkpoint 不含 DSS component state，因此仅作为诊断产物，不计入复现结果；
  `dss-cifar10-sym05-seed4-current-source/` 未形成有效训练产物；均不计入结果。
- 阻塞：无代码或数据阻塞；正式训练已完成。
- 精确下一步：使用 `curve_comparison.py` 补充 DSS 与论文曲线的对齐比较。

## PDL 底座与 Smoke 记录（2026-08-01）

- 当前任务：模块化实现 PDL，并建立可供 UPM 等方法复用的实例转移训练入口；
  当前分支 `codex/lnl-reproduction-foundations`，基线提交 `4dd0746`。
- 完成度：7/7（100%）：Algorithm 2 generator、anchor candidates、Eq. (1)/(2)/(4)、
  紧凑 artifact、通用实例转移 Algorithm、独立多阶段 runner、配置/测试/文档均完成。
- 模块化边界：`training/experiment.py`、`training/pipeline.py` 和
  `algorithms/supervised.py` 未因 PDL 修改；全局 `[C,C]` TransitionArtifact 语义不变。
  PDL 使用独立 `instance_transition_estimator` 与 `instance_transition_algorithm` plugin kinds。
- 论文配置：CIFAR-10、PDL IDN rate `0.4`、ResNet-34、无 augmentation、batch 128、
  SGD lr `0.01`、momentum `0.9`、weight decay `1e-4`、milestones 40/80、
  noisy validation 5000、20 parts；正式配置尚未运行。
- Smoke：`artifacts/test-runs/pdl-cifar10-smoke/` 完成 warm-up 1 epoch 与 corrected
  training 2 epochs；生成 posterior/feature snapshot、instance artifact、best/last checkpoint、
  metrics 与曲线；resume 成功且 artifact hash
  `2c729d2c317cdcd981522b6facb964e2bf4beb956873bc99fcafa23e59e27574` 保持一致。
- Smoke 指标仅验证通路：100 个 test 样本 accuracy `10.0%`，不作为论文结果。
- 测试：PDL `7/7`、Plugin `12/12`、受影响 transition `20/20`、完整 unittest
  `292/292` 通过；fresh smoke 与 checkpoint resume 均成功。
- 尚未实现：共享 T-Revision；它不是 PDL 基础 Forward/Reweight 通路的阻塞项。
- 当前修改未提交、未推送；shared plugin catalog、file-map 和两份论文文档存在协作者冲突风险。
- 精确下一步：在用户确认随机抽取的论文配置后，仅运行一次正式实验，再生成论文曲线比较。

## JoCoR 底座与 Smoke 记录（2026-07-29）

- 当前任务：模块化实现 JoCoR，并为后续 Co-teaching、CNLCU 提供通用多模型入口；
  当前分支 `codex/lnl-reproduction-foundations`，基线提交 `4dd0746`。
- 完成度：6/7（85.71%）：论文/官方代码核对、通用模型组、JoCoR 数学、
  multi-model runner、checkpoint/resume、配置/测试/文档均完成；正式实验已运行至
  161/200，但两次训练进程均在未触发 early stopping 的情况下中断。
- 模块化边界：新增 `multi_model_algorithm` plugin kind 和独立
  `training/multi_model_experiment.py`；`training/experiment.py` 与
  `training/pipeline.py` 未修改，JoCoR 不使用 Co-teaching peer exchange。
- 论文对齐：官方六卷积 CNN 两份、同一 batch、两份 CE 加双向 KL、
  同一 joint small-loss 集合、一个联合 Adam optimizer；SmallLossSelector 通过
  显式 `rounding: floor` 对齐官方 `int(...)`，旧默认 `ceil` 不变。
- 抽样配置：sampling seed `20260728` 从 CIFAR-10 Sym-20%、Sym-50%、
  Sym-80%、Asym-40% 中抽中 Sym-50%；训练 seed 1；λ=0.9。
- 正式参数：tensor-only、无 augmentation/Normalize、batch 128、Adam
  `lr=0.001`、β=(0.9,0.999)、200 epochs、epoch 80 后线性衰减并将 β1 改为
  0.1、num_gradual 10、末 10 epoch 两个成员准确率均值。
- Smoke：`artifacts/test-runs/jocor-cifar10-smoke/` 完成 2 epochs；
  CUDA 运行、双模型 format-v2 checkpoint、恢复、成员/ensemble 指标和曲线均正常。
- 测试：JoCoR focused `7/7`、Selector `14/14`、Plugin `11/11`、
  Foundation `9/9` 通过；完整 unittest `278/278` 通过。
- 正式产物：`artifacts/reproductions/jocor-cifar10-sym05-seed1/`；
  `last.pt` 可恢复，最近 10 个已完成 epoch（152--161）的双成员平均测试准确率
  `79.12%`，ensemble 平均为 `79.55%`，最佳 ensemble 为 `79.78%`（epoch 139）。
- 暂定论文比较：论文 CIFAR-10 symmetric-50% 报告 `79.41 ± 0.25%`；按论文双成员
  平均口径当前差 `-0.29` 个百分点。该结果仅为暂定值，因为尚未完成 200 epochs，
  且本轮只有一个 seed，不能替代论文的多次重复统计。
- 当前修改未提交、未推送；高冲突文件仅为 plugin catalog、file-map 和两份共享论文文档。
- 精确下一步：如需完成 JoCoR，使用同一 `last.pt` 续训至 200 epochs，再以末 10
  epoch 两模型均值与论文同设置结果比较；否则保留当前结果为中断的暂定记录。

## CWD/FINE 最小侵入补齐记录（2026-08-02）

- 当前任务：使 CWD 与 FINE 在用户提供 YAML 后可由独立 CLI 直接训练；当前分支
  `codex/lnl-reproduction-foundations`，基线提交 `4dd0746`。
- 完成度：4/4（100%）：CWD 论文统计公式、CWD 独立训练入口、FINE 官方损失与
  selection 语义、FINE warm-up/EMA/SCS/SCR/强增强两阶段入口均已实现。
- 模块化边界：未修改 `training/experiment.py`、`training/pipeline.py`、
  `plugins/builtin/catalog.py` 或 checkpoint；CWD/FINE 生命周期分别位于独立 runner，
  模型、EMA、多视图、selector、estimator 和 objective 均可单独复用。
- CWD：按 Eq. 19、21--30 保存 class prior、virtual flip matrix、系数矩阵、伪逆、
  centroid 与 source snapshot hash；CIFAR airplane/automobile 合并官方 train/test 后执行
  五折中的固定 fold 0，每个 epoch 刷新 feature/statistic artifact。
- CWD 单次配置：symmetric `(eta_P, eta_N)=(0.2,0.2)`、ResNet-34、Adam lr `0.05`、
  weight decay `1e-4`、milestones 40/120、200 epochs；只运行一个 fold/seed，不重复。
- FINE：附加项仅作用于 `rejected_mask & (pseudo_label != noisy_label)`；训练入口保存
  EMA、SCS、SCR、regularizer、optimizer/scheduler 与 RNG 状态。
- FINE 单次配置：CIFAR-100 symmetric 20%、七卷积 StudentNet、batch 128、SGD、
  warm-up lr `0.1`、robust lr `0.05`、warm-up 200/total 300 epochs、cosine eta-min
  `5e-4`、alpha/beta/gamma=`1.0/0.1/0.002`；只运行一个 seed。
- 已执行测试：CWD focused/training `7/7`，FINE focused/training `8/8`，Pipeline
  `12/12`，完整 unittest `299/299`，均通过；conda 启动仍输出已有 OpenCL vendor
  `temp.txt` 警告，但不影响测试结果。
- Smoke：CWD 与 FINE 两份 smoke YAML 均完成并通过 checkpoint resume；产物分别位于
  `artifacts/test-runs/cwd-cifar10-minimal-invasion/` 和
  `artifacts/test-runs/fine-cifar100-minimal-invasion/`。指标只验证调用链，不作论文比较。
- 文件新增：CWD/FINE 两个独立 runner 与 CLI、通用 EMA、多视图、七卷积模型、
  两个训练测试和四份 smoke/reproduction YAML；file-map 已同步维护。
- 当前修改未提交、未推送；`docs/file-map.md` 与本文件已有协作者内容，只做局部追加，
  仍属于高冲突共享文件。
- 精确下一步：由用户选择 CWD 或 FINE 后，只运行对应单次 reproduction YAML；完成后
  数字化论文曲线并比较。本轮未运行正式论文实验。
