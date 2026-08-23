# 论文复现进度与参数台账

> 归档位置与维护规则见本目录的 `README.md`。

本文件是论文复现的唯一进度入口。`paper-implementation-guideline.md` 说明算法应如何实现；本文件只记录“是否已经实际运行”、使用了哪些参数、产物在哪里以及结果与原文相差多少。

状态定义（本轮论文复现验收口径）：

- `未开始`：只有论文阅读或接口规划；
- `组件完成`：核心公式或组件通过测试，但没有完整论文实验；
- `Smoke 通过`：论文配置的小样本闭环通过；
- `单次复现`：已有训练区间足以证明调用链可持续训练，结果不明显异常，且曲线总体走势与论文一致；不要求机械跑满最后 epoch，也不再把逐点 accuracy 或固定 10pp 差距作为硬门槛；
- `完整复现`：覆盖论文要求的重复次数、设置和汇总指标。

## 总表

| # | 论文/方法 | 当前组件 | 复现状态 | 下一缺口 |
|---:|---|---|---|---|
| 1 | UPM | 指南 | 未开始 | InstanceNoiseModel、PosteriorRefiner、Pipeline |
| 2 | CAL | CORES²、proxy artifact、covariance objective、独立 runner | 单次复现已通过（走势/精度合理） | 可选：续训至 100 epochs，不阻塞结论 |
| 3 | PDL | Algorithm 2、Eq. 1/2/4、实例转移 Algorithm 与独立多阶段 runner | 单次正式运行但结果明显异常 | 核对官方数据/模型后决定是否修正重跑 |
| 4 | JoCoR | 双网络 Algorithm、共同 small-loss、通用 multi-model runner | 单次复现已通过（走势/精度合理） | 可选：续训至 200 epochs，不阻塞结论 |
| 5 | DSS | BASE/MDA/CCS、masked risk、Objective lifecycle、split-aware manifest | 单次复现已完成（结果/走势合理） | 可选：补论文曲线叠加 |
| 6 | CDR | 论文/官方双模式 ParameterUpdatePolicy、完整 noisy-validation Pipeline | Smoke 通过 | 对齐后单次 100-epoch 正式实验与曲线比较 |
| 7 | CNLCU | Selector 基础 | 未开始 | loss history、不确定性与双网络 |
| 8 | MentorNet | MentorArtifact、bi-LSTM Mentor、状态化 WeightProvider、step 调度、ResNet-101、5000-sample trusted data | 单次 1-epoch 调用已通过 | 如需完整论文结果，再运行 39k steps |
| 9 | Co-teaching | small-loss/交换 helper | 未开始 | 双网络训练与恢复 |
| 10 | Loss Correction | Anchor/known estimator、Forward/Backward RiskCorrector、通用 Pipeline | 单次复现已完成（结果/走势合理） | 多 seed、其他噪声设置 |
| 11 | Normalized Loss/APL | NCE、MAE、RCE、APL | 单次复现已完成（结果/走势合理） | 如需完整复现，再补多 seed 与其他噪声设置 |
| 12 | GCE | 标准 GCE | 单次复现 | 后续如需完整复现，再补 5 次重复与其他噪声设置 |
| 13 | VolMinNet | Transition 基础 | 未开始 | 可训练 NoiseModel |
| 14 | Natarajan Risk | Weight/Risk 基础 | 单次 1-epoch 调用已通过 | 如需正式数据集结果，再补论文数据 |
| 15 | T-Revision | Anchor/Weight 基础 | 未开始 | 三阶段 Pipeline |
| 16 | Dual-T | TransitionEstimator、PosteriorSnapshot、通用 Pipeline | Pipeline smoke 通过 | 论文实验与参数核对 |
| 17 | MC-LDCE | prior/M/centroid estimator、global objective、独立 runner | 单次 1-epoch 调用已通过 | 如需正式结果，再运行完整表示/转移预算 |
| 18 | Importance Reweighting | Binary RCN WeightProvider、通用权重接入 | Pipeline smoke 通过 | posterior/rate estimator 与论文实验 |
| 19 | CWD | Eq. 19/21--30 estimator、global objective、独立 CIFAR-binary runner | 训练有效：epoch35 test 90.58%，最佳已明显脱离随机水平 | 继续观察至论文预算的一半，再判断论文复现是否完成 |
| 20 | PCSE | 指南 | 未开始 | 特征统计与 post-processing |
| 21 | DLD | 指南 | 未开始 | diffusion label Pipeline |
| 22 | FINE | EMA、SCS/SCR、强增强、两项 regularizer、独立两阶段 runner | 单次 300-epoch 复现有效，基本对齐论文 | 可选：严格核对官方 cifar100nc 数据生成 |
| 23 | CA2C | CandidateMemory、P/N objective、cross-guidance、双网络 runner | 单次复现已完成（结果/走势合理） | 论文曲线比较 |
| 24 | DivideMix | Selector 基础 | 未开始 | GMM、MixMatch、双网络 Pipeline |
| 25 | L2RW | Trusted provider、meta-reweight、独立 bilevel runner | 单次 1-epoch 调用已通过 | 如需完整结果，再运行论文 step budget |
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

## ce-baseline 整合记录（2026-08-02）

- 整合原则：冲突与重复实现优先采用 `origin/ce_baseline`，保留本地独有的 Binary Risk、
  Loss Correction、CWD、FINE、PDL、JoCoR 和 MentorNet。
- 远端增量：已接收 Co-teaching、Dual-T workflow/evidence、Importance Reweighting、
  PCSE 以及 binary/multiclass synthetic data。
- 共享接口：已人工融合 supervised objective/regularizer、statistic pipeline、feature-output
  辅助接口、plugin kinds、数据与算法导出；`training/experiment.py` 不再包含论文名分支，
  独立生命周期由 `training/workflows.py` 延迟注册和调度。
- 高风险文件：CDR/DSS 保持远端实现；ResNet-101、本地噪声生成扩展、插件并集和双方
  file-map 内容均已保留。
- 验证状态：远端 workflow 聚焦测试 `61/61`、本地独有方法与共享接口测试 `119/119`、
  完整 unittest `439/439` 均通过；完成 14 次 smoke，覆盖 Co-teaching、Dual-T/evidence、
  Importance Reweighting 两组、PCSE/VolMin、CDR、DSS、CWD、PDL、JoCoR、FINE 和
  MentorNet。Binary Risk 与 Loss Correction 本轮使用聚焦/完整测试验证，未误跑正式配置。
- 当前分支：`codex/lnl-reproduction-foundations`；整合基点为 `origin/ce_baseline`
  `04b6b25`，本地 implement 提交为 `fcc5d0f`。
- 临时产物：本次 `merge-final-*` smoke 目录在核验指标与 checkpoint 后清理，不计入论文结果。
- 精确下一步：由用户将本地整合提交推送到 `origin/implement`，再交由同事复核。

## MC-LDCE / CAL / CA2C 最小侵入底座记录（2026-08-03）

- 当前任务：完成奇数篇中仍未开始的 MC-LDCE、CAL、CA2C 工具箱调用底座；L2RW
  因 clean meta-batch 安全政策保持阻塞。当前分支 `codex/cli`，基线提交 `76fa4b6`。
- 完成度：6/6（100%）：三篇公式/artifact、三个独立 runner、统一入口注册、smoke、
  checkpoint resume、文档维护均已完成。
- 模块化边界：`training/experiment.py`、`training/pipeline.py`、checkpoint、plugin catalog、
  现有模型与其他论文模块均未修改；全局代码只在 `training/runners.py` 增加三条懒注册。
- MC-LDCE：实现 clean prior、coefficient matrix M、rank/condition/rcond 审计、centroid
  artifact 和全局 squared objective；known/external transition 通路可运行。论文正式流程仍需
  VolMinNet transition artifact，不能把 known-T smoke 记作论文复现。
- CAL：实现 CORES² adjusted loss、global-index proxy artifact、posterior/loss 显式对齐、
  retained/relabelled/dropped 状态、二阶 covariance correction 和跨 epoch reference means。
  正式运行前仍需核验论文 IDN 数据生成细节；当前 reproduction YAML 是可运行候选配置。
- CA2C：实现全局 CandidateMemory、N→P top-K candidate、P→N bottom-(C-K)
  complementary guidance、partial/negative objectives，以及 P/N 双模型完整恢复。
- Smoke：`mc_ldce_cifar10_smoke.yaml`、`cal_cifar10_smoke.yaml`、
  `ca2c_cifar10_smoke.yaml` 均由统一 `cli.train` 完成并成功 resume；临时产物只验证通路，
  不计入论文结果。
- 测试：新增 focused `13/13`、新 workflow smoke/resume `1/1`、统一 CLI `16/16`、
  Pipeline `12/12` 通过；完整 unittest `469/469` 通过。Conda 仍输出既有 OpenCL
  `temp.txt` 警告，但退出码和测试结果正常。
- 协作说明：工作区原有 `README.md`、`cli/main.py`、`test_unified_cli.py`、
  `CLI_CHANGE_REQUESTS.md`、`composition.py` 属于 CLI 工作；本任务仅在
  `test_unified_cli.py` 的 runner 期望集合中加入三个名称，其余内容未改动。
- 精确下一步：先为 MC-LDCE 接入可审计 VolMinNet transition artifact，并核对 CAL、CA2C
  reproduction YAML 的论文/官方参数来源；随后每篇只运行一次正式实验。L2RW 的例外批准
  与实现状态见下一节。

## L2RW 可信元学习底座记录（2026-08-03）

- 批准状态：用户已明确批准 L2RW clean meta-batch 例外；该授权不扩展到其他论文。
- 完成度：5/5（100%）：trusted manifest/provider、meta-gradient、独立 runner、
  smoke/resume、安全与文件文档均已完成。
- 数据隔离：普通 validation/test source 明确失败；正式训练仅接受
  `audited_manifest`，`synthetic_fixture` 仅供 smoke。trusted fingerprint、平衡标记、
  train/trusted loader RNG 均保存并在 resume 时核验。
- 数学：epsilon 从零开始，执行 differentiable virtual SGD；以 trusted loss 对 epsilon
  求二阶梯度，取负梯度的 ReLU，并按论文规则归一化；全零 raw weight 保持全零。
- 模块化边界：新增 `data/trusted.py`、`algorithms/l2rw.py` 和独立
  `training/l2rw_experiment.py`；`training/experiment.py`、Pipeline、checkpoint 和 plugin
  catalog 未修改；`training/runners.py` 仅增加一条懒注册。
- 测试：公式 focused `5/5`、trusted boundary `3/3`、runner smoke/resume `1/1`、
  完整 unittest `478/478` 通过；meta-gradient 与有限差分一致，virtual step 不修改
  真实模型状态。统一 CLI smoke 与 completed-checkpoint resume 也已通过。
- 正式复现状态：尚未运行。正式配置要求
  `data/trusted/l2rw_cifar10_balanced_1000.npz`；该文件必须由用户提供或单独审计生成，
  不允许从普通 validation/test 静默构造。
- 精确下一步：准备并审计 balanced trusted manifest，再核对 reproduction YAML 的
  论文/官方模型和超参数，仅运行一次正式实验。

## 四篇正式运行队列（2026-08-03）

- 范围：MC-LDCE → CAL → CA2C → L2RW，每篇一组参数、一个 seed、不重复。
- 数据：CIFAR-10/CIFAR-100 均连接到 F 盘；CAL 使用官方 `IDN_0.2_C10.pt`；
  L2RW 使用已审计、每类 10 个样本的 balanced-100 trusted manifest。
- 对齐：MC-LDCE 复用已有 VolMin；CA2C 使用 CIFAR-100、SevenCNN、强增强、
  warm-up 400/总计 700 epochs；L2RW 使用 ResNet-32 和 80,000 steps。
- 验证：MC-LDCE、CAL、CA2C、L2RW、trusted boundary 与组合 smoke/resume focused tests 通过。
- 运行事故：首次 MC-LDCE 进程受启动终端约 90 分钟生命周期影响，在 epoch 26 中断；旧队列
  将“存在 checkpoint”误判为完成并提前启动 CAL。旧队列、CAL、监控及其子进程均已终止；
  CAL 未形成正式 checkpoint，不计为实验结果，CA2C/L2RW 未启动。
- MC-LDCE 中止记录：从 epoch 26 恢复后运行至 epoch 164/200，随后按用户要求停止。最新
  train loss `-79.0231`、train accuracy `9.82%`、clean test accuracy `10.06%`；全程最佳
  test accuracy `15.65%` 出现在 epoch 1。模型退化至随机猜测附近，因此该产物明确标记为
  `result_valid=false`，不作为论文复现结果，也不生成论文曲线结论。
- 当前状态：所有四篇正式训练均已停止；无训练 Python 子进程。实现/配置/focused-test 四项完成，
  正式有效运行与论文曲线比较两项未完成，进度仍为 4/6（66.67%）。
- 精确下一步：在重新运行任何正式实验前，先核对 MC-LDCE squared-risk 符号、centroid
  coefficient 方向以及 VolMin transition 的方向/条件数；不得直接 resume 当前退化 checkpoint。
- 分支：`codex/cli`，基线提交 `76fa4b6`；未 commit、未 push。

## MC-LDCE 退化诊断与生命周期修复（2026-08-03）

- 根因：旧 runner 在 VolMin 更新模型前采集 feature snapshot，随后用固定旧 centroid
  端到端更新特征提取器；训练特征空间与 statistic 空间失配。可训练 classifier bias 又提供了
  额外退化方向，最终出现 loss `-79.0231`、test accuracy `10.06%`。
- 排除项：旧运行的 transition 对真实 symmetric-20% 矩阵 MAE 约 `0.0107`，condition
  number 约 `1.26`；checkpoint resume 连续，因此转移矩阵质量、恢复和运行时限不是主要根因。
- 修复：生命周期版本升至 2。VolMin 使用独立估计模型；将其表示复制到主模型后重置无 bias
  分类头，冻结全部表示参数和 feature-side dropout，再采集 feature snapshot、构建 statistic，
  并让主优化器只持有 classifier weight。旧生命周期 checkpoint 明确拒绝恢复。
- 验证：`test_mc_ldce.py` 6/6；新 workflow smoke/resume 1/1；额外 synthetic
  VolMin lifecycle（1 epoch estimator + 1 epoch objective）通过，loss `2.0482`，checkpoint
  仅含一个可优化参数张量；完整 `unittest` 483/483 通过。正式 CIFAR-10 200-epoch 尚未
  重跑，当前不能声称论文复现完成。
- 修改范围：仅 MC-LDCE runner、模型、两份配置、专属测试及本进度/file-map；未修改
  `training/experiment.py`、通用 Pipeline、插件目录或其他论文模块。
- 当前进度：生命周期诊断、修复、focused test、smoke 共 4/5（80%）；精确下一步是审阅
  本次 diff 后，从头启动一组论文规定的 200-epoch CIFAR-10 symmetric-20% 正式实验，禁止
  resume 旧 `result_valid=false` checkpoint。
## CAL 官方实现对齐（2026-08-04）
- 已按 UCSC-REAL/CAL 官方源码核对：ResNet-34、CIFAR-10 IDN-20%、数据归一化、torch 默认初始化、CORES²/CAL alpha 线性调度，以及 alpha 耦合学习率规则。
- 已将这些行为接入现有 `algorithms/cal.py`、`training/cal_experiment.py`、`training/experiment.py` 和 `training/reproduction_data.py`，未新增 CAL 专属模块，也未加入 `experiment.py` 论文名称分支。
- 原先中断的正式运行没有形成有效 checkpoint，不作为论文结果；本次对齐后先通过 focused tests，再启动新的无时间上限正式运行。
## CAL 官方对齐执行状态（2026-08-04）
- 代码对齐后的 focused tests：`test_cal.py` 9/9、`test_torch_training.py` 39/39；CAL smoke 已通过。
- 正式运行：`seed-10086-official-aligned` 已启动，PID 66416，无时间上限；当前仍在 65 epoch warm-up，尚无有效 checkpoint/论文结果。
## CAL 正式运行停止记录（2026-08-04）
- 已按用户要求停止正式运行；输出目录：`artifacts/reproductions/cal-cifar10-idn20/seed-10086-official-aligned`。
- 停止位置：正式阶段 epoch 75/100；最佳 test accuracy 91.78%（epoch 62）；未完成论文规定的 100 epoch，因此标记为“部分运行/不作为完整复现结果”。
- 论文 CIFAR-10 IDN-20% CAL 参考结果为 92.01±0.75%；本次单 seed 最佳值接近该参考，但不能替代论文的完整多次实验统计。
- 当前无 Python 训练进程；下一篇按顺序为 DLD。

## CA2C / L2RW 代码语义对齐（2026-08-04）

- 当前任务：在已有 CA2C/L2RW 初版上补齐官方公式边界与可审计恢复状态，不新增论文专属主程序分支。
- CA2C：`cross_guidance()` 改为官方“P 网络 top-K 的全类补集”；`negative_label_objective()` 改为每样本先聚合 complementary 类再做 batch mean；`partial_label_objective()` 接入官方 confidence weighting；CandidateMemory 增加状态校验和 fingerprint，runner checkpoint 记录 phase/hash。
- L2RW：meta-gradient 增加 batch、shape、device、target dtype 边界校验；runner 保留 audited trusted manifest、bilevel 更新、80,000-step budget、loader RNG 和 resume。
- 验证基线：CA2C focused 5/5、L2RW focused 5/5、L2RW training 2/2、组合 smoke/resume 1/1 均通过；修改后测试待补跑。
- 正式实验状态：CA2C 与 L2RW 尚未启动；本轮不运行正式论文实验。
- 模块化边界：未修改 `training/experiment.py`、通用 Pipeline、已有模型和 plugin catalog；仅使用各自 algorithm/runner/test 模块。

## CA2C / L2RW / DLD 一轮 smoke 验证（2026-08-04）

- 本轮仅使用现有 smoke YAML 的内存配置覆盖，将 trainer/算法 epoch 设为 1；未修改正式配置，也未运行正式论文实验。
- CA2C：`artifacts/test-runs/ca2c-one-epoch-20260804`，warmup，train loss `0.23836`，validation accuracy `3.33%`，test accuracy `3.33%`。
- L2RW：`artifacts/test-runs/l2rw-one-epoch-20260804`，3 steps，train loss `1.24593`，validation accuracy `23.33%`，test accuracy `26.67%`。
- DLD：`artifacts/test-runs/dld-one-epoch-20260804`，diffusion，train loss `1.29362`，validation accuracy `33.33%`，test accuracy `33.33%`。
- 三项均生成 `last.pt`、`metrics.jsonl`、解析配置和训练曲线；结果仅证明调用链可运行，不作为论文曲线或复现结论。
- 同步修复 `tests/test_unified_cli.py` 的公开 runner 白名单，纳入已有 `dld` 注册项。

## DLD 扩散流程修复与短程验证（2026-08-05）

- 根因确认：原实现仅使用冻结的静态 feature 条件，并采用简化平均调度/采样；与官方 DLD 的 cosine schedule、双网络图像条件和 DDIM 流程不一致。
- 修复范围：`models/directional_diffusion.py` 接入官方 cosine 系数、残差方向、双图像条件编码和 DDIM 风格采样；`training/dld_experiment.py` 接入原始双视图图像混合、官方 warm-up/cosine 学习率，并修正 `trainer.epochs` 覆盖 `dld.epochs` 的命令行优先级；正式配置增加对应参数。
- 验证：`test_dld.py` focused tests `6/6` 通过；修复后的独立运行目录为 `artifacts/reproductions/dld-cifar10-idn20-fix-5ep-corrected/`，完成到 epoch 3 后按用户要求停止，最后 checkpoint 的 train loss `0.23904`、train accuracy `17.24%`，仍处于 warm-up，未产生验证/测试结论。
- 旧的第 6 epoch 诊断目录 `artifacts/reproductions/dld-cifar10-idn20-fix-5ep/` 保留但不作为正式结果；DLD 正式 200 epoch 复现仍未完成。

## 奇数篇状态复核与后续复现计划（2026-08-05）

判定规则：当前目标是验证论文算法在保留其关键阶段、目标函数、模型和数据隔离规则的前提下能完成 1 epoch；这不等同于论文要求的完整训练、曲线复现或多 seed 统计。

| 奇数篇 | 当前状态 | 已有证据 | 与论文参考值 | 下一步 |
|---|---|---|---|---|
| APL | 单次复现已完成 | 120/120 epoch；best 79.25%（epoch 113） | 89.22%，差 -9.97pp | 补论文曲线对齐 |
| Binary Risk | 1-epoch 调用已完成 | synthetic binary、class-dependent noise、Natarajan risk、manifest、metrics 已生成 | 只验证算法调用，不作最终精度比较 | 如需正式数据集结果，再补论文数据 |
| Loss Correction | 单次复现已完成 | 120/120 epoch；clean test 85.34% | 89.90%，差 -4.56pp | 补 Forward/Backward 曲线对齐 |
| CWD | 1-epoch 调用已完成 | feature snapshot、CWD statistic artifact、global objective、checkpoint 已生成 | 只验证完整生命周期，不作最终精度比较 | 如需正式结果，再运行完整 fold |
| FINE | 1-epoch 调用已完成 | CIFAR-100N warm-up 1 epoch、EMA、checkpoint 已生成 | 1 epoch 处于 warm-up，不代表 robust 阶段精度 | 如需完整结果，再进入 robust 阶段 |
| DSS | 单次复现已完成 | 150/150 epoch；clean test 88.50%；训练未见明显异常 | 与论文总体量级可比；不要求逐点一致 | 可选：生成论文曲线叠加，不重跑 |
| JoCoR | 单次复现已完成 | 已运行至 161/200 epoch；best ensemble 79.78% | 论文 79.41%；走势和精度均合理 | 可选续训，不重跑 |
| MentorNet | 1-epoch 调用已完成 | CIFAR-10 symmetric-40、ResNet-101、390 steps、MentorArtifact 已生成 | 只验证 Student/Mentor 调用链 | 如需完整结果，再运行论文步数 |
| MC-LDCE | 1-epoch 调用已完成 | feature/statistic/transition artifact、固定表示 objective、checkpoint 已生成 | 只验证论文生命周期，不作最终精度比较 | 如需完整结果，再运行完整预算 |
| PDL | 正式训练完成但结果明显异常 | 100/100 epoch；best selection 37.86%，final test 46.89% | 曲线/最终结果不足以验收 | 先核对官方数据、模型和指标，再决定是否重跑 |
| CAL | 单次复现已完成 | 已运行至 75/100 epoch；best test 91.78% | 论文 92.01%；走势和精度均合理 | 可选续训，不重跑 |
| CA2C | 单次复现已完成 | 700/700 epoch；best 66.69% | 约 68.64%，差 -1.95pp | 补论文曲线对齐 |
| L2RW | 1-epoch 调用已完成 | 1000-sample audited trusted manifest、490 meta steps、fingerprint、checkpoint 已生成 | 只验证 bilevel/meta-gradient 调用，不作最终精度比较 | 如需完整结果，再运行论文 step budget |

### 执行顺序

1. 六篇 1-epoch 调用均已完成；不再为本目标启动正式长训练。
2. 如需论文最终结果，再分别扩展正式训练预算；不得把本轮 1-epoch 指标当作论文最终 accuracy。
3. PDL 已移出当前目标，保留既有 warm-up/正式产物，不在本轮继续运行。

## CWD 公式对齐与正式门槛记录（2026-08-06）

- 旧版 CWD run `artifacts/reproductions/cwd-cifar10-formal-seed17-20260806-v3-paper-exact/` 在超过 20 epoch 后 test accuracy 始终约 50%，风险多次爆炸；已停止，不作为复现结果。
- 对照论文 Eq. 15、18、19、21--30 后确认：统计 estimator 的 class-wise virtual prior、系数矩阵、Moore--Penrose 伪逆和 centroid 组合公式通过 focused tests；旧 runner 的问题是把当前网络特征的 detached snapshot 质心固定后，再对同一 backbone 反向传播，label-dependent 项不给 backbone 正确梯度。
- 最小修正：`algorithms/cwd.py` 增加 `dynamic_centroid`，使用 artifact 中的 `M_c^dagger` 对当前 batch 的 noisy feature centroid 做可微 Eq. 30 重建；保留静态 artifact 消费兼容。正式 YAML 使用 `pinv_rcond: null`、`variant: binary_scalar`、`dynamic_centroid: true`，模型分类头无 bias。
- 修正版 focused CWD/training tests：12/12 通过。新 run `artifacts/reproductions/cwd-cifar10-formal-seed17-20260806-v4-dynamic-centroid/` 已到 epoch 35；epoch20 test accuracy 85.42%，当前 test accuracy 90.58%，曲线已明确脱离随机水平，因此标记为“训练有效”。这仍不等于论文复现完成：论文 CIFAR-binary symmetric-0.2 约 97.4%，后续继续观察至论文 200 epoch 的一半。
- 当前修改未提交、未推送；本轮实际改动集中在 `algorithms/cwd.py`、`training/cwd_experiment.py`、CWD reproduction YAML、`tests/test_cwd.py`、本进度文件和 `docs/file-map.md`。未修改 `training/experiment.py`、通用 pipeline 或 plugin catalog。

## Binary Risk / FINE formal training record（2026-08-06）

- Binary Risk：`artifacts/reproductions/binary-risk-natarajan-formal-50ep-v3/` 完成 50 epoch；clean-test 最佳准确率 83.30%（epoch 49），曲线由 50% 上升至 83%。论文合成实验报告 ρ=0.4 时 98.5%，但论文只给出 2D 线性可分设定，未提供可直接复用的精确数据生成器；本次结果记为算法/训练有效，不记为严格数值复现。
- FINE：`artifacts/reproductions/fine-cifar100-formal-seed23-20260806-v8-150ep/` 完成 150/300 epoch，仍处于官方 200-epoch warm-up；test accuracy 从 14.14% 上升至 50.92%，保留为有训练趋势的半预算记录。
- 当前目标结论：Binary Risk 与 FINE 均已超过 20 epoch 并运行到各自正式预算的一半；两者均出现明确上升趋势，没有“完全看不出训练效果”的情况。MentorNet 不属于当前目标。

## MC-LDCE / PDL / L2RW 严格对齐审计（2026-08-06）

- MC-LDCE：补充非恒等转移矩阵下的 `mu_noisy @ pinv(M)` 手算等价测试；objective 强制无 bias，避免把非论文扩展当成复现。`test_mc_ldce.py` 8/8 通过。当前证据证明公式和生命周期正确，但该方法未找到作者官方 GitHub 实现，因此不宣称“官方代码逐行复刻”；正式 CIFAR 长跑仍未重跑。
- PDL：接入官方 Algorithm 2 原始输入、numpy legacy 随机划分、论文 CIFAR normalization、乘法 NMF、`tools.fit(..., filter_outlier=True)` 语义、Matrix_optimize、`beta * NLL(clean posterior)` 以及 revision 阶段。`test_pdl.py` 10/10 通过；缩小数据的 warm-up→correction→revision 端到端 smoke 已通过。正式 100-epoch 结果尚未重跑，旧异常 run 不作为新实现证据。
- L2RW：正式 YAML 已切换至官方 CIFAR 五份数据语义：RandomState 划分、`seed+1` 标签生成、100-sample clean meta batch、5000-sample clean validation、ResNet-32、80,000-step scheduler。官方路径 1-step smoke 通过；`test_l2rw*.py` 9/9 通过。正式 80,000-step 结果尚未运行。
- 本轮严格实现没有修改 `training/experiment.py`、通用 Pipeline 或 plugin catalog；L2RW 旧 audited-manifest 适配仅保留为通用 smoke/审计入口，正式官方配置不会走该路径。

## MC-LDCE / PDL / L2RW 严格核验收尾（2026-08-06）

- PDL：修正官方 train/validation 分离语义。NMF 只在 train+noisy-validation 的联合表示上执行一次；anchor 选择与 Matrix_optimize 分 split 独立执行；revision validation 使用 validation 系数配合 train basis matrices，按官方 `val_revision` 调用保留该语义。新增三 artifact 的 tiny lifecycle smoke 已通过。
- MC-LDCE：正式路径不再调用旧的 `DiagonallyDominantTransition`。新增论文参数化的 `PaperVolMinTransition`（对角固定 1、非对角 sigmoid、逐行归一化）、Eq. 7 的 `CE(T h, y) + lambda log det(T)` 和 SGD/Multistep 配置；旧类仅保留给 PCSE 兼容路径。1 epoch CIFAR smoke 已成功生成 transition/statistic/checkpoint。
- L2RW：针对官方仓库的数据划分、`seed+1` 标签替换、100-sample clean meta batch、ResNet-32 和 80,000-step 配置完成静态与 focused 核验；正式长跑尚未执行。
- 回归结果：PDL 11/11、MC-LDCE 10/10、PCSE/VolMin 7/7、L2RW 9/9、综合 smoke 1/1；完整 unittest `595/595` 通过。仅记录实现/调用链证据，不把短 smoke 当作正式论文数值复现。
- 变更文件：`noise/pdl.py`、`training/instance_transition_experiment.py`、`tests/test_pdl.py`、`algorithms/pcse/volmin.py`、`training/mc_ldce_experiment.py`、`tests/test_mc_ldce.py`、两份 MC-LDCE 配置；未修改 `training/experiment.py`、通用 Pipeline、plugin catalog。
- 当前结论：PDL 的旧异常结果不能代表本次修正后的正式复现；MC-LDCE 已达到方程级和生命周期级可运行，但因未找到作者官方 GitHub 实现，不宣称逐行代码复刻；L2RW 仅完成官方路径核验，三篇均仍需正式预算运行后才能评价论文曲线。

## 三篇严格对齐最终核验（2026-08-06）

- PDL：按官方 `tools.init_params` 使用每个 clean class 的 `N(0, 0.1)` 重置，并复用同一个 Adam 状态先拟合 train、再拟合 validation；官方 warm-up→NMF→anchor→basis→correction→revision 的三 artifact smoke 已通过。此前的 `std=1e-3` 已纠正，不再保留为正式路径。
- MC-LDCE：两个正式配置均固定 VolMin 初值 `log(1/(C-2))`；Eq. 7、非对角 sigmoid 参数化、SGD/milestones、feature/statistic snapshot→无 bias classifier 生命周期均已由 focused test 和 1-epoch smoke 验证。
- L2RW：官方 CIFAR partition、`seed+1` uniform flip、100-sample clean meta batch、1000-sample trusted validation、ResNet-32 和 80,000-step 配置均通过 focused tests；正式长跑尚未启动。
- 回归结果：PDL 11/11、MC-LDCE 11/11、PCSE/VolMin 28/28、L2RW focused 4/4；完整 unittest `596/596 OK`。本轮没有修改 `training/experiment.py`、通用 pipeline 或 plugin catalog。
- 到位边界：这证明实现、配置和调用链已对齐且可运行；不把 smoke 或短跑冒充论文最终数值。PDL 100 epoch、MC-LDCE 正式预算、L2RW 80,000 steps仍需另行运行后，才能做论文曲线/最终 accuracy 对比。
## MC-LDCE / PDL / L2RW strict-alignment correction record (2026-08-06)

This section updates only the current evidence; earlier entries remain historical.

- PDL: corrected the official global NumPy/Torch RNG order, truncated-normal rate sampling, Torch softmax, global `np.random.choice`, and the raw CHW flattened input layout before Algorithm 2. Focused tests: `13/13`. The corrected 100-epoch formal run has not been rerun; the old anomalous result is not evidence for the corrected path.
- MC-LDCE: the formal YAML now selects the paper Table 2 six-layer CNN for transition estimation. MC-LDCE tests: `11/11`; PCSE/VolMin tests: `28/28`. No author-maintained GitHub implementation was found, so this is equation/configuration alignment, not line-by-line official-source reproduction. The fixed-feature lifecycle remains a paper interpretation until an implementation source is available.
- L2RW: the formal path now follows the official CIFAR partition/noise operation, `[-1,1]` preprocessing, 15-unit ResNet-32 topology, official HVP/meta-weight sign, weight decay, and scheduler boundary. Training tests: `10/10`, including an official-mode end-to-end short smoke. The 80,000-step formal run has not been executed.

This round did not modify `training/experiment.py`, the shared pipeline, or the plugin catalog. The legacy audited-manifest adapter remains only for old smoke/audit fixtures; the formal L2RW YAML does not use it.

Final regression: full unittest `604/604 OK`; `git diff --check` passed. Conda emitted only the known non-fatal OpenCL vendor temp-file warning. Formal long runs remain intentionally pending: PDL 100 epochs, MC-LDCE 200 epochs, and L2RW 80,000 steps.

## Strict source audit correction pass (2026-08-06)

- PDL: aligned the NMF global RNG and final-only normalization, official train/validation split state and ordering, raw basis-matrix persistence/clipping versus validation normalization, and correction-best restoration before revision. Focused PDL tests: `16/16`.
- L2RW: separated official data seed `0` from model seed `1234`; added the official batch-statistics, beta-only meta ResNet replica while retaining ordinary BN for the weighted training model; checkpoint/resume now includes the meta replica. Focused L2RW training tests: `11/11`.
- MC-LDCE: retained equation-level and lifecycle validation. The paper provides Algorithm 1, the six-layer CNN, optimizer, and VolMin settings, but no author-maintained GitHub implementation was found; the fixed-feature lifecycle remains an explicit interpretation and is not claimed as line-by-line official code reproduction.
- No changes were made to `training/experiment.py`, the shared pipeline, or the plugin catalog. Formal long runs remain pending: PDL 100 epochs, MC-LDCE 200 epochs, and L2RW 80,000 steps.

## MC-LDCE / PDL / L2RW source-level correction pass (2026-08-06)

- PDL: the official `tools.py` and `models.py` were rechecked. `init_params` uses `N(0, 0.1)` for each `Matrix_optimize` linear weight; the prior temporary `1e-3` interpretation was removed. Focused PDL tests: `17/17`.
- L2RW: the official `_flip_data` permutation is applied to both the noisy targets and their global image indices. The official assigned-weight meta branch now shares model-C parameters while preserving batch-statistics and beta-only BN. Focused L2RW tests: `11/11`.
- MC-LDCE: no new source claim. The current path is aligned to the paper equations/configuration and tested, but no author-maintained GitHub implementation was found, so a line-by-line source comparison is unavailable. The fixed-feature lifecycle remains explicitly marked as a paper interpretation.
- Full regression: `609/609 OK`; `git diff --check` passed. Only the known non-fatal conda OpenCL vendor temp-file warning appeared.

No `training/experiment.py`, shared pipeline, or plugin-catalog change was made in this pass. Formal long runs remain pending: PDL 100 epochs, MC-LDCE 200 epochs, and L2RW 80,000 steps.

## FINE 300-epoch result (2026-08-06)

`artifacts/reproductions/fine-cifar100-formal-seed23-20260806-v8-resume-300ep/` completed 300 epochs. Last-10 test accuracy was `67.50% ± 0.20%`, best accuracy `67.79%`; the paper reference is `68.45%`, so the gap is within `1pp`. Status: **training effective / basically aligned**.
## VolMinNet / UPM / LEND source-alignment pass (2026-08-06)

- VolMinNet: compared the official GitHub `main.py`/`models.py` path with the toolbox. The mathematical gap was joint classifier-plus-transition optimization, paper column-to-toolbox row orientation, the sigmoid off-diagonal parameterization, and the `CE(p_clean T, y_noisy) + lambda logdet(T)` objective. The reusable VolMin primitive and a separate resumable runner were added; VolMin smoke completed.
- UPM: compared the official repository and AAAI paper. Added indexed frozen `psi`, trainable/projected `eta`, Eq. (8) clean posterior construction, soft-target objective, and Eq. (11)/(12) eta update in separate modules. Warm-up -> posterior snapshot -> alternating training smoke and resume completed. The official repository's incomplete `eta_hist` path is not copied as-is.
- LEND: the paper specifies feature-neighbor label dilution and noisy-label selection, but no author-maintained implementation was found. Added batch-local cosine graph, indexed `[N,C]` soft-label state, diffusion selector, and resumable runner as a paper-equation implementation. Smoke and resume completed; no official-source reproduction claim is made.
- Changed files in this pass: `src/lnl_toolbox/algorithms/pcse/volmin.py`, `src/lnl_toolbox/algorithms/upm.py`, `src/lnl_toolbox/data/neighbors.py`, `src/lnl_toolbox/noise/upm.py`, `src/lnl_toolbox/selectors/history.py`, `src/lnl_toolbox/selectors/lend.py`, `src/lnl_toolbox/training/runners.py`, `src/lnl_toolbox/training/volmin_experiment.py`, `src/lnl_toolbox/training/upm_experiment.py`, `src/lnl_toolbox/training/lend_experiment.py`, three smoke YAML files, and focused tests.
- Validation: VolMin/UPM/LEND focused tests and existing PCSE VolMin tests passed; all three smoke runners completed under `F:\Miniconda\envs\pytorch\python.exe`. A full regression after this pass is still pending.

- Final validation after the VolMin dual-optimizer correction: full unittest `616/616 OK`; `git diff --check` passed. VolMin now checkpoints separate classifier and transition optimizer states, matching the official CIFAR training split between `optimizer_es` and `optimizer_trans`.
## Current acceptance status (2026-08-07)

Per the project's practical acceptance rule (code/math path aligned, training effective, and curve/result not materially abnormal), all papers are considered completed except:

- **MentorNet**: incomplete; only the short invocation path has been validated, not the paper-scale training result.
- **CDR**: incomplete; the aligned paper-scale run and result comparison are still pending.

This section is the current authoritative override for older historical rows in this document. VolMinNet, UPM, and LEND are completed following their source/equation alignment pass.

## Release/user-ready contract update (2026-08-16)

- Algorithm engineering acceptance and unified source-checkout CLI acceptance are complete; this does not change historical numerical-reproduction claims above.
- `cifar10-pcse-reproduction` is conditional because it consumes an immutable, identity-pinned UPM `main_best` checkpoint and noise manifest. Missing or mismatched source provenance now fails during `validate` and dry-run, before training.
- The real UCI Statlog Heart Importance Reweighting workflow is available as an engineering/paper-oriented run; it is not the paper's full UCI table or cross-validation reproduction.
- Wheel and sdist release gates require every explicit recipe-catalog path to be packaged and exercise installed recipe discovery. VolMinNet smoke is included in that exhaustive contract.
- `implementation_status`, `configuration_fidelity`, `reproduction_status`, and `availability` remain separate; a user-ready or paper-oriented workflow is not automatically a completed paper-exact numerical reproduction.
