# 代码文件职责说明

本表覆盖当前项目中的源代码、配置、测试和主要文档。`papers/` 下的 PDF、CIFAR 二进制文件和自动生成的 `__pycache__/` 不逐个解释。

统一 CLI 的执行事实源是 `training/runners.py`；`training/workflows.py` 只保留兼容
API。`cli/data/recipe_catalog.json` 是显式内置 recipe 清单，配置文件通过 package
data 安装，避免用户本地 YAML 污染 catalog。

## 1. 仓库根目录

| 文件 | 作用 |
|---|---|
| `README.md` | 项目入口，说明当前定位、核心能力和文档导航。 |
| `pyproject.toml` | Python 包信息、依赖、可选训练依赖、命令行入口及 `src` 包发现规则。 |
| `CLI_CHANGE_REQUESTS.md` | 记录统一 CLI 实际试用中发现的体验问题、处理方案和验收命令。 |
| `toolbox-architecture.md` | LNL toolbox 的长期设计、实验公平性、Noise Manifest 和阶段路线；其中具体算法是参考插件规划。 |
| `.gitignore` | 忽略缓存、虚拟环境、构建产物、运行输出和临时文件。 |

## 2. 通用核心 `src/lnl_toolbox/core/`

| 文件 | 作用 |
|---|---|
| `core/__init__.py` | 汇总并公开核心类型，调用方可从 `lnl_toolbox.core` 统一导入。 |
| `core/component.py` | 定义最小 `Component` 生命周期和可选 `Stateful` checkpoint 协议。 |
| `core/context.py` | 定义 `ExperimentContext`，保存工作目录、配置、seed 和外部服务。 |
| `core/batch.py` | 定义通用 `Batch`；payload 不透明，不强制图像或标签格式。 |
| `core/algorithm.py` | 定义任务无关 `Algorithm` 生命周期协议。 |
| `core/state.py` | 定义 Runner 管理的 `RunState`，记录 cycle、step、phase、指标和元数据。 |
| `core/result.py` | 定义 `StepResult` 和 `Artifact`，统一算法步骤的返回格式。 |
| `core/objectives.py` | 定义通用 `ObjectiveConsumer` 与结构化 `ObjectiveResult`；可同时返回优化标量、样本 mask、报告 loss 和诊断指标。 |
| `core/evaluator.py` | 定义 evaluator 的 `update/compute/reset` 协议，不预设具体指标。 |
| `core/storage.py` | 定义 `ArtifactSink`、`CheckpointStore`、`Checkpoint` 和 `ArtifactRef` 存储边界。 |

## 3. 插件系统 `src/lnl_toolbox/plugins/`

| 文件 | 作用 |
|---|---|
| `plugins/__init__.py` | 公开 `PluginCatalog` 和 `PluginSpec`。 |
| `plugins/catalog.py` | 实现按 kind/name 注册、构建及 capability 查询的插件目录。 |
| `plugins/builtin/__init__.py` | 公开内置示例插件目录构造函数。 |
| `plugins/builtin/catalog.py` | 分 kind 注册 PyTorch/NumPy loss、噪声生成器、通用 `batch_selector`、`multi_model_algorithm`、Anchor/Dual-T `transition_estimator`、ParameterUpdatePolicy 及旧 Co-teaching helper，并提供各自独立的配置构造入口。 |
| `registry.py` | 早期的单类型轻量 Registry；暂时保留以兼容已有代码，长期可由 PluginCatalog 取代。 |

## 4. Runner 与算法接口

| 文件 | 作用 |
|---|---|
| `engine/__init__.py` | 公开 `run_cycles` 和兼容名称 `run_epochs`。 |
| `engine/runner.py` | 执行 setup、run、cycle、step、evaluator 和 close；不处理模型或梯度。 |
| `algorithms/base.py` | 将旧 `Algorithm/TrainState` 导入映射到新的通用核心，保持兼容。 |
| `algorithms/coteaching/` | 完整双模型 `CoTeachingAlgorithm`、配置/状态、stable-index small-loss 选择，以及原 NumPy helper 的兼容迁移。 |
| `algorithms/cnlcu/` | CNLCU-S Eq. (2)/(3)/(7) 与 CNLCU-H Eq. (4)/(8)、corrected LOF、按 epoch 划窗且 history/count 同步重置的 peer state、稳定索引选择、双模型交叉更新及可恢复状态。 |
| `algorithms/t_revision/` | T-Revision Reweight-R 的三阶段状态机、corrected vectorized Eq. (3)、raw additive transition revision、方法专属 artifact 与严格 resume；不把 raw revised matrix 伪装成通用 `TransitionArtifact`。 |
| `algorithms/update_policy.py` | 定义通用 ParameterUpdateInput/Result/Policy、普通 StandardUpdatePolicy，以及 policy checkpoint 身份协议。 |
| `algorithms/cdr.py` | 实现 CDR 论文 Eq. (3)-(6) 的全参数精确 top-k 模式，以及官方代码二维/四维权重、阈值并列和 L2-compatible 模式。 |
| `algorithms/supervised.py` | 单模型监督训练步骤；兼容普通逐样本归约与结构化 Objective，并通用转发 Objective 生命周期钩子。 |
| `algorithms/masked_risk.py` | 通用 candidate-class masked cross entropy；显式接收 `[B,C]` 类别排除 mask。 |
| `algorithms/dss.py` | DSS 的插件化 Objective consumer；组合 BASE、MDA、CCS 和官方 batch-mean 优化语义。 |
| `algorithms/__init__.py` | 汇总算法兼容接口、Co-teaching 函数和可选 PyTorch ParameterUpdatePolicy。 |
| `models/cifar_six_conv.py` | 通用六卷积 CIFAR 分类器；保持 JoCoR 官方 64/64/128/128/196/16 通道及 256→C 头。 |

## 5. 数据层

| 文件/目录 | 作用 |
|---|---|
| `data/contracts.py` | 早期 LNL 分类样本结构，包含 image、target、index 和可选干净标签；属于具体任务协议。 |
| `data/cifar.py` | 读取 CIFAR-10/100 官方 pickle，转换为 `[N,32,32,3]` uint8 图像并验证标签。 |
| `data/torch_cifar.py` | 提供分层、随机及 `classwise_legacy` 可复现划分、可配置 mean/std 的标准变换、GCE 2018 preprocessing 和稳定 `input/target/index`。 |
| `data/noisy_dataset.py` | 按显式 global-index mapping 包装训练 Dataset 并替换 target；不会向 batch 暴露 clean label。 |
| `data/__init__.py` | 公开 CIFAR 读取函数、干净 Dataset 和 noisy wrapper。 |
| `data/cifar-10-batches-py/` | 用户放入的 CIFAR-10 官方 Python 数据。 |
| `data/cifar-100-python/` | 用户放入的 CIFAR-100 官方 Python 数据。 |

## 6. LNL 示例能力

### PDL / 通用实例转移链路（2026-08-01）

| 文件 | 作用 |
|---|---|
| `noise/pdl.py` | PDL Eq. (1)/(4)、紧凑 `PartTransitionArtifact` 与按 global index 查询的 `[B,C,C]` provider。 |
| `algorithms/instance_transition.py` | 不绑定论文名称的实例转移校正分类 Algorithm，支持 Forward 与 importance correction。 |
| `training/instance_transition_experiment.py` | 独立多阶段 runner：noisy warm-up、posterior/feature snapshot、实例转移估计、校正训练、checkpoint/resume。 |
| `cli/instance_transition_train.py` | 通用实例转移实验命令行入口。 |
| `configs/noise/pdl.yaml` | PDL Algorithm 2 噪声 manifest 参数片段。 |
| `configs/experiment/pdl_cifar10_smoke.yaml` | PDL 小样本多阶段 smoke。 |
| `configs/experiment/pdl_cifar10_reproduction.yaml` | 论文 CIFAR-10 单次正式配置；尚未运行。 |
| `tests/test_pdl.py` | generator、anchor、Eq. (1)/(2)/(4)、artifact、Forward/Reweight、checkpoint identity 测试。 |

| 文件 | 作用 |
|---|---|
| `noise/manifest.py` | `NoiseManifest` 的数据结构、标签 fingerprint、NPZ 保存/加载，以及数据集、长度、标签范围和概率的训练前校验。 |
| `noise/generators.py` | symmetric 固定全局/逐类/逐样本 transition 采样、pairflip、class-conditional transition 和基于 class score 的示例 IDN 生成器。 |
| `noise/split_manifest.py` | 对互斥数据 split 分别重启 RNG 后生成一个 global-index Noise Manifest，复用 external-manifest 通路。 |
| `noise/transition.py` | 验证 `T[i,j]=P(noisy=j|clean=i)` 行随机矩阵；提供 `KnownTransition`、版本化 `TransitionArtifact`、NPZ roundtrip 和哈希篡改检测。 |
| `noise/estimators.py` | 定义无 clean-label 的 `PosteriorSnapshot`、`TransitionEstimator` Protocol，并实现 Anchor、Known 与 Dual-T 离线 estimator。 |
| `noise/__init__.py` | 公开 Noise Manifest、生成器、后验快照、estimator 和转移矩阵产物协议。 |
| `losses/numpy_losses.py` | NumPy 版逐样本 CE 与 GCE，用于数学验证，不执行神经网络反向传播。 |
| `losses/torch_losses.py` | PyTorch 版逐样本 CE、标准 GCE、NCE、MAE、RCE、严格 P0 APL，以及 `[B]` 输出合同校验。 |
| `losses/__init__.py` | 公开 NumPy 参考函数；安装 PyTorch 时同时公开可训练 loss。 |
| `losses/loss板块第一轮.md` | Loss 实现简报及 Config、Algorithm、Selector、Evaluator 间的统一调用协议。 |
| `selectors/base.py` | 定义单 batch 的 `SelectionInput`、`SelectionResult`、无状态 `Selector` Protocol 及输入输出校验。 |
| `selectors/basic.py` | 实现选择全部样本的 `AllSelector` 和 schedule-driven、stable-index tie-break 的 `SmallLossSelector`；支持显式 `ceil/floor` 数量取整，默认仍为 `ceil`。 |
| `selectors/schedules.py` | 定义无状态 keep-rate schedule，支持固定浮点、显式 constant 和零基 epoch linear 配置。 |
| `selectors/history.py` | 提供 scalar indexed history 与容量/epoch/class 有界的 `IndexedTensorHistory`。 |
| `selectors/dss.py` | 保存 DSS posterior 历史、MDA 边际、Mann–Kendall score、样本 mask 和类别排除 mask。 |
| `selectors/__init__.py` | 公开通用 Selector 合同、基础实现和边界校验。 |
| `estimators/base.py` | 定义 sample-aligned `ReliabilityResult`、泛型 `ReliabilityEstimator[InputT]` 和轻量 `StatisticResult[StatisticT]`；reliability score 固定为越大越可靠，不直接兼容低分优先的 `SmallLossSelector`。 |
| `estimators/__init__.py` | 公开 Reliability/Statistic estimation 合同及边界验证；不接入 plugin、配置或训练生命周期。 |
| `estimators/dividemix_gmm.py` | 实现 DivideMix 中独立的 epoch-level two-Gaussian clean-probability 子模块；CPU float64 拟合并按 stable index 返回高分代表更可靠的证据，不包含完整 DivideMix Pipeline。 |
| `estimators/selection_adapter.py` | 按 stable index 将 dataset-level reliability 查找、抽取并重排为 batch 输入，再固定取负转换为低分优先的 `SelectionInput`；不调用 Selector 或决定 threshold/split。 |
| `treatments/base.py` | 定义内部 `ContributionResult`，统一表达 hard mask、连续非负样本权重和标量统计。 |
| `treatments/reduction.py` | 定义 `ReductionSpec`，按 weight-sum mean、batch mean 或 sum 归约逐样本 loss。 |
| `treatments/selector_adapter.py` | 将现有 hard Selector 适配为 mask 加全一权重，保持旧配置和数值行为。 |
| `treatments/weights.py` | 定义泛型 WeightProvider、通用 WeightResult 和不依赖具体输入字段的 adapter；BinaryRCNWeightInput 与对应 provider 实现二分类 asymmetric-RCN 的论文精确 importance-weight 公式，不负责 posterior 或噪声率估计。 |
| `algorithms/mentornet.py` | MentorNet 的移动分位数、burn-in/dropout 和状态化连续权重 Provider；只消费 noisy Student loss 与冻结 MentorArtifact。 |
| `models/mentornet.py` | 可复用 bi-LSTM curriculum model；不拥有 StudentNet 或训练循环。 |
| `training/mentor_artifacts.py` | 冻结 Mentor 模型的结构、特征 schema、来源和哈希校验。 |
| `training/mentor_learning.py` | 从隔离的 trusted curriculum feature 数据离线训练 MentorArtifact。 |
| `data/curriculum.py` | trusted curriculum feature 数据合同，与目标 noisy Student run 隔离。 |
| `cli/mentor_prepare.py` | 在 `data/mentornet/` 下确定性生成 trusted indices、noise manifest、Student-feedback features 与元数据。 |
| `treatments/__init__.py` | 公开内部 sample-treatment、reducer、Selector adapter 和连续权重合同。 |
| `evaluation/metrics.py` | NumPy 版 accuracy 和选样 precision/recall。 |
| `evaluation/classification.py` | 单模型分类评测，以及通用命名模型组各成员与 mean-logit ensemble 的单遍评测。 |
| `evaluation/__init__.py` | 公开当前示例指标。 |

## 7. CLI

| 文件 | 作用 |
|---|---|
| `cli/__init__.py` | 共享中文 `PromptSession`、实验模板发现、Loss 选择、clean/generated/external 标签模式和最终确认；不包含训练数学。 |
| `cli/main.py` | `lnl` 统一入口；提供环境检查、实验/组件/论文浏览、配置校验、运行、恢复及组合配置生成。 |
| `composition.py` | 定义 supervised 组件组合的深层兼容校验、显式槽位覆盖和不覆盖式 YAML 写入；专用 runner 不在此伪装成自由组合。 |
| `cli/train.py` | 通用训练入口；无参数进入向导，有参数时保持 argparse/YAML 调用。 |
| `cli/multi_train.py` | 独立的配置化多模型训练入口；供 JoCoR、Co-teaching、CNLCU 等共用，不修改单模型主入口。 |
| `cli/clean_train.py` | Clean baseline 入口；交互选择模型、scheduler、恢复或多 seed。 |
| `cli/make_noise.py` | 交互或参数化地从 `.npy` 标签生成 symmetric/pairflip Noise Manifest。 |
| `cli/inspect_data.py` | 交互或参数化地验证 CIFAR-10/100 并输出划分摘要。 |

## 8. 训练编排

| 文件 | 作用 |
|---|---|
| `training/experiment.py` | 唯一监督训练器；统一构造模型、Loss、batch Selector、ParameterUpdatePolicy、optimizer、scheduler、clean/noisy Dataset、可显式选择 clean/noisy validation、评测和产物。 |
| `training/coteaching_experiment.py` | `method: coteaching` 的 CIFAR 双模型生命周期；负责独立 peer 初始化、epoch-seeded loader、双 peer 评测和 checkpoint/resume。 |
| `training/cnlcu_experiment.py` | `method: cnlcu` 的 CIFAR-10/100 双模型生命周期；按 `variant: soft|hard` 组装 scorer，共享严格 peer cross-update、noisy-validation best selection 和 epoch-boundary resume。 |
| `training/t_revision_experiment.py` | `method: t_revision` 的 CIFAR-10/100 Reweight-R 生命周期；组装 noisy CE、train-only Anchor snapshot、固定-T classifier initialization 和 classifier/delta joint revision。 |
| `training/progress.py` | 无第三方依赖的 batch 终端进度显示和逐 epoch `training_curves.svg` 生成器；不参与训练决策。 |
| `training/clean_baseline.py` | clean-only 包装和多 seed 汇总；检测到 noise 配置立即拒绝。 |
| `training/noisy_labels.py` | 生成或导入 manifest，规范化为 run-local v2，校验恢复身份，并分别记录 train/validation 的标签来源与实际噪声率。 |
| `training/checkpoint.py` | 保存 checkpoint v2 的模型、优化器和 ParameterUpdatePolicy 身份/私有状态，并安全读取旧格式。 |
| `training/snapshots.py` | 在 inference mode 下收集 noisy posterior、target 和稳定 global index，排序后构造 `PosteriorSnapshot`；不执行 warm-up 训练。 |

## 9. 配置

| 文件 | 作用 |
|---|---|
| `configs/README.md` | 说明 YAML 是 LNL 示例配置，核心只接收 mapping，不依赖 YAML/Hydra。 |
| `configs/algorithm/ce.yaml` | CE 示例参数。 |
| `configs/algorithm/gce.yaml` | 标准 GCE 的 `q` 参数；不包含隐式截断阈值。 |
| `configs/algorithm/nce.yaml` | NCE 的数值稳定参数。 |
| `configs/algorithm/apl.yaml` | 严格正权重及 NCE + RCE 嵌套配置。 |
| `configs/algorithm/coteaching.yaml` | Co-teaching 示例参数。 |
| `configs/algorithm/cnlcu_soft.yaml` | CNLCU-S 的线性 remember schedule、fixed-window float32 history、soft uncertainty 和 stable-index selection 参数。 |
| `configs/algorithm/cnlcu_hard.yaml` | CNLCU-H paper formulas、fixed loss bound 和 corrected-LOF fidelity identity。 |
| `configs/algorithm/cdr.yaml` | 显式声明 paper-mode CDR 的噪声率、L1 系数、参数范围和 compatibility mode。 |
| `configs/algorithm/dss.yaml` | DSS Objective 的论文参数片段：150 epochs、30 warm-up、MDA 0.99 与 CCS α=0.10。 |
| `configs/algorithm/jocor.yaml` | JoCoR 联合目标和官方 floor small-loss 日程片段。 |
| `configs/noise/symmetric.yaml` | symmetric noise 示例参数。 |
| `configs/noise/instance_dependent.yaml` | 示例 IDN 参数。 |
| `configs/experiment/cifar10_symmetric_ce_smoke.yaml` | symmetric 0.4 + CE 的统一 noisy runner smoke 配置。 |
| `configs/experiment/cifar10_symmetric_small_loss_smoke.yaml` | symmetric 0.4 + CE + 固定 0.5 keep-rate SmallLossSelector 的单模型 smoke 配置。 |
| `configs/experiment/cifar10_symmetric_small_loss_linear_smoke.yaml` | symmetric 0.4 + CE + 从 1.0 线性变化到 0.5 的 SmallLossSelector smoke 配置。 |
| `configs/experiment/cifar10_symmetric_cdr_smoke.yaml` | symmetric 0.4 + CE + paper-mode CDR ParameterUpdatePolicy 的 smoke 配置。 |
| `configs/experiment/cifar10_coteaching_smoke.yaml` | 双 TinyCNN、symmetric 0.4、两轮 CPU Co-teaching smoke 配置。 |
| `configs/experiment/cifar10_cnlcu_soft_smoke.yaml` | 双 TinyCNN、symmetric 0.4、两轮 CPU CNLCU-S smoke 配置。 |
| `configs/experiment/cifar10_cnlcu_hard_smoke.yaml` | 双 TinyCNN、symmetric 0.4、corrected LOF 的 CPU CNLCU-H smoke 配置。 |
| `configs/experiment/cifar10_t_revision_smoke.yaml` | TinyCNN、symmetric 0.4、Stage 1/2A/2B 各两轮的 CPU T-Revision Reweight-R smoke 配置。 |
| `configs/experiment/gce_cifar10_noise02_smoke.yaml` | GCE 论文设置的 CIFAR-10、symmetric 0.2、ResNet-34 小样本 CUDA smoke。 |
| `configs/experiment/gce_cifar10_noise02_reproduction.yaml` | GCE 论文设置的 CIFAR-10、symmetric 0.2、单次 120 epoch 正式配置。 |
| `configs/experiment/loss_correction_cifar10_asymmetric04.yaml` | Loss Correction 论文设置的 CIFAR-10、官方 class-conditional asymmetric 0.4、Forward、单次 120 epoch 正式配置。 |
| `configs/experiment/dss_cifar10_symmetric05_smoke.yaml` | DSS 两 epoch 小样本闭环配置，仅用于 pipeline/checkpoint 验证。 |
| `configs/experiment/dss_cifar10_symmetric05_reproduction.yaml` | 单次 DSS CIFAR-10 symmetric-50% 正式配置：seed 4、PreActResNet-18、150 epochs。 |
| `configs/experiment/jocor_cifar10_symmetric05_smoke.yaml` | JoCoR 两模型、共同选样、checkpoint 和 CUDA/CPU 闭环 smoke 配置。 |
| `configs/experiment/jocor_cifar10_symmetric05_reproduction.yaml` | JoCoR CIFAR-10 symmetric-50% 单次正式配置：官方 CNN、Adam、λ=0.9、200 epochs 和末 10 epoch 双成员均值。 |
| 实验 YAML 顶层 `selector` | 省略时为 `all`；当前支持 `all` 和固定 keep-rate `small_loss`。 |
| 实验 YAML 顶层 `noise` | 省略时 clean；`name/rate/seed` 生成 symmetric、pairflip 或 class-conditional 噪声；symmetric 可选 fixed-global、per-class 或 transition 采样；也可用 `manifest` 导入外部映射。 |
| 实验 YAML 的 `data.validation_split` / `data.normalization` | 通用配置 random/stratified split、RNG 实现以及三通道 mean/std；默认保持原有分层划分与 Toolbox 标准常数。 |

## 10. 测试

| 文件 | 验证内容 |
|---|---|
| `tests/test_core.py` | 通用 Runner 生命周期、状态推进和 close 行为。 |
| `tests/test_plugins.py` | capability 查询、loss/`batch_selector`/旧 Co-teaching/transition/update-policy kind 隔离，以及 Anchor/Known/Dual-T builder 与各独立调用回路。 |
| `tests/test_registry.py` | 旧 Registry 注册和构建。 |
| `tests/test_cifar_reader.py` | 用小型临时 pickle 验证 CIFAR-10/100 解码逻辑。 |
| `tests/test_noise.py` | 噪声生成（含 legacy transition sampling）、manifest roundtrip/身份校验、非法标签与概率拒绝、KnownTransition 和恢复训练 manifest 身份约束。 |
| `tests/test_transition_estimators.py` | Snapshot/collector 合同、Anchor/Known/Dual-T 数学和顺序不变性、Artifact roundtrip/篡改检测及 Tensor 转换。 |
| `tests/test_estimators.py` | Reliability/Statistic 基础结果合同、stable-index 对齐、score 方向和 metrics 校验。 |
| `tests/test_dividemix_gmm.py` | DivideMix GMM clean-probability 子组件的拟合、确定性、退化输入、可选依赖与输出合同。 |
| `tests/test_reliability_selection_adapter.py` | dataset-level reliability 按 stable index 抽取、重排并显式转换为低分优先 SelectionInput。 |
| `tests/test_losses.py` | P0 loss 公式、GCE 极低概率梯度、逐样本 shape、极端数值和 APL 论文约束。 |
| `tests/test_coteaching.py` | legacy Co-teaching exchange 与旧导入路径回归。 |
| `tests/test_coteaching_algorithm.py` | peer cross-update、初始化、schedule、floor/tie-break 和双 peer state 合同。 |
| `tests/test_coteaching_workflow.py` | YAML dispatch、双模型 checkpoint、fresh/resume/completed-resume 和配置漂移。 |
| `tests/test_cnlcu_estimators.py` | CNLCU-S Eq. (2)/(3)/(7)、CNLCU-H Eq. (4)/(8)、corrected LOF、无 ReLU score、fixed-window history 和 state validation。 |
| `tests/test_cnlcu_algorithm.py` | 当前 loss 先写 history、uncertainty-aware selection、严格 peer cross-update、A/B identity 和配置隔离。 |
| `tests/test_cnlcu_workflow.py` | CIFAR-10/100 dispatch、fresh/resume/completed no-op、history checkpoint 与连续/恢复训练等价性。 |
| `tests/test_t_revision_objective.py` | Reweight-R Eq. (3) 手算、矩阵方向、ratio、梯度、raw transition 和数值失败边界。 |
| `tests/test_t_revision_algorithm.py` | additive delta、optimizer 参数所有权、方法专属 revised artifact 和阶段状态机。 |
| `tests/test_t_revision_workflow.py` | 三阶段运行、best provenance、阶段内 resume、artifact/manifest drift、completed no-op、CPU/CUDA。 |
| `docs/t-revision.md` | T-Revision Reweight-R 的用户入口、适用边界、配置、三阶段生命周期、输出解释与 resume 规则。 |
| `tests/test_update_policy.py` | 通用 ParameterUpdatePolicy 输入输出、Standard 更新等价性和 checkpoint 身份协议。 |
| `tests/test_cdr.py` | CDR Eq. (3)-(6)、官方代码 scope/threshold/L2 模式、稳定 top-k、失败边界、Selector、checkpoint 和 CPU/CUDA 一致性。 |
| `tests/test_cdr_reproduction.py` | CDR 正式配置的 100-epoch 合同、官方 ResNet stem/初始化选项和通用 builder 接入。 |
| `tests/test_cli.py` | Prompt 重试/取消、GCE/APL 配置、APL 正权重输入和交互/参数模式兼容。 |
| `tests/test_noisy_ce_baseline.py` | 统一 runner 的 generated noise、默认 clean/显式 noisy validation、manifest 元数据、Loss 与 Selector 配置接入。 |
| `tests/test_training_progress.py` | 终端进度节流、关闭行为、非法输入和 SVG 曲线产物。 |
| `tests/test_selectors.py` | 通用 Selector 输入输出合同、固定比例、最少选择、stable-index tie-break 和失败边界。 |
| `tests/test_treatments.py` | ContributionResult、ReductionSpec、Selector adapter 和显式 loss 归约合同。 |
| `tests/test_importance_reweighting.py` | Binary asymmetric-RCN 权重公式、通用 WeightProvider adapter、detach 和 batch-mean 梯度。 |
| `tests/test_dss.py` | MDA、BASE、CCS、masked CE、batch-mean objective、状态恢复和插件构建。 |
| `tests/test_jocor.py` | 双向 KL、联合分数、floor 稳定选样、共同更新、官方模型、插件、配置和双模型 checkpoint roundtrip。 |
| `tests/test_split_noise_manifest.py` | classwise legacy split、per-split RNG 重启和 manifest roundtrip。 |

## 11. 调研脚本与文档

| 文件 | 作用 |
|---|---|
| `scripts/download_papers.ps1` | 下载论文并检查 PDF 文件头，生成来源 manifest。 |
| `scripts/extract_papers.py` | 从论文 PDF 抽取文本供摘要整理使用。 |
| `scripts/prepare_split_noise_manifest.py` | 为 CIFAR 生成 classwise-legacy split-aware external Noise Manifest。 |
| `docs/paper-summaries.md` | 26 篇 LNL 论文的中文摘要、代码链接与伪代码。 |
| `docs/usage-guide.md` | 面向使用者的当前功能、命令和下一步测试说明。 |
| `docs/architecture.md` | 通用核心、插件、Runner 与 CIFAR 数据流的架构图。 |
| `docs/file-map.md` | 本文档，解释每个代码文件的职责。 |
| `papers/README.md` | 论文目录、下载状态和缺失 PDF 说明。 |
| `papers/implement/README.md` | 论文实现资料、共享原料和单篇复现目录的维护规则。 |
| `papers/implement/paper-implementation-guideline.md` | 26 篇论文到唯一 Toolbox 模块、接口和调用顺序的实现映射。 |
| `papers/implement/transition-estimator-audit.md` | Anchor 与 Dual-T transition estimator 的论文依据、公式、实现边界、差异和验证状态。 |
| `papers/implement/paper-reproduction-progress.md` | 26 篇论文的复现状态总表、复现增量与复用审计表，以及单篇记录索引。 |
| `papers/implement/gce/result.md` | GCE 单次正式复现的参数、产物、结果和原文差异。 |
| `papers/implement/apl/plan.md` | APL NCE+RCE 单次复现的原料复用、必要增量、固定配置和验收计划。 |
## 12. 本轮四篇论文复现底座

| 文件 | 作用 |
|---|---|
| `core/hyperparameters.py` | 论文候选参数的单次确定性抽样、来源记录和解析配置恢复。 |
| `evaluation/curve_comparison.py` | 读取训练指标和论文曲线，输出叠加曲线、差值和摘要。 |
| `algorithms/binary_risk.py` | Natarajan 二分类无偏风险和 label-dependent cost risk。 |
| `data/binary_benchmarks.py` | UCI/NPZ 二分类数据、稳定 index、分层划分、噪声 manifest 和 CIFAR 二分类视图。 |
| `training/binary_experiment.py` | 通用二分类 Dataset、训练、评测和单次实验入口。 |
| `cli/binary_train.py` | 二分类实验的 YAML/argparse 入口。 |
| `estimators/cwd.py` | 按 CWD Eq. 19、21--30 恢复 binary/multiclass class-wise virtual auxiliary prior、系数矩阵、伪逆和 centroid artifact。 |
| `algorithms/cwd.py` | 只消费 feature/statistic artifact 的 CWD squared global objective；支持论文质心伪逆的静态消费和当前 batch 的可微重建；不绑定 runner。 |
| `training/cwd_experiment.py` | CWD CIFAR airplane/automobile 单 fold 生命周期：五折切片、噪声、逐 epoch feature/statistic artifact、论文 Adam 配置、可微质心重建和 checkpoint/resume。 |
| `cli/cwd_train.py` | CWD 独立 YAML 训练入口。 |
| `models/fine_cnn.py` | 可复用的七卷积 CIFAR StudentNet，公开 logits/features，不包含 FINE 生命周期。 |
| `algorithms/fine.py` | FINE suppression/active-forgetting 两项损失，仅消费 rejected 且伪标签改变的样本。 |
| `selectors/sed.py` | 独立 SED selector，以及可恢复的 SCS 选择和 SCR 连续权重。 |
| `training/model_ema.py` | 通用 EMA 模型及 checkpoint 状态。 |
| `data/multi_view.py` | 稳定 sample index 的 weak/strong CIFAR 多视图数据包装与强增强。 |
| `training/fine_experiment.py` | 独立 warm-up → EMA/SCS/SCR → strong-view robust training 生命周期。 |
| `cli/fine_train.py` | FINE 独立 YAML 训练入口。 |
| `tests/test_cwd.py`、`tests/test_cwd_training.py` | CWD 公式、失败边界、artifact/pipeline 和独立训练闭环。 |
| `tests/test_fine.py`、`tests/test_fine_training.py` | FINE loss/mask、SCS/SCR、EMA、多视图、七卷积模型和两阶段闭环。 |
| `configs/experiment/cwd_cifar10_{smoke,reproduction}.yaml` | CWD smoke 与单次 200-epoch、五折中 fold 0 的论文配置；reproduction 使用 Moore--Penrose `pinv` 和 dynamic centroid。 |
| `configs/experiment/fine_cifar100n_{smoke,reproduction}.yaml` | FINE smoke 与单次 300-epoch、200-epoch warm-up 配置。 |
| `configs/experiment/binary_risk_natarajan_1epoch.yaml` | Natarajan 二分类无偏风险的一组 synthetic 论文算法 1-epoch 验证配置。 |
| `configs/experiment/cwd_cifar10_1epoch.yaml` | CWD 五折 fold 0、CIFAR airplane/automobile、1-epoch 完整生命周期配置。 |
| `configs/experiment/fine_cifar100n_1epoch.yaml` | FINE CIFAR-100N、1-epoch warm-up 生命周期配置；`warmup_epochs == trainer.epochs` 表示只验证首个论文阶段。 |
| `configs/experiment/mentornet_cifar10_symmetric04_1epoch.yaml` | MentorNet CIFAR-10 symmetric-40、ResNet-101、冻结 MentorArtifact 的 1-epoch 配置。 |
| `configs/experiment/mc_ldce_cifar10_1epoch.yaml` | MC-LDCE（仓库中 MC-PCDE 的对应命名）1-epoch 表示/转移估计与全局 objective 配置。 |
| `configs/experiment/l2rw_cifar10_1epoch.yaml` | L2RW 1-epoch、1000-sample audited trusted manifest 与 bilevel 更新配置。 |
| `training/experiment.py` | 仅接入参数记录、ResNet 深度构造和通用 regularizer，不含论文名称分支。 |
| `training/binary_experiment.py` | 保持 CSV/NPZ 兼容，并接入现有 synthetic binary generator、class-dependent noise manifest 与 Natarajan risk。 |
| `training/workflows.py` | 独立 workflow 的延迟注册与通用调度；主实验入口不再按论文名称分支。 |
| `training/pipeline.py` | regularizer、warm-up、artifact 和组件状态生命周期编排。 |
| `training/checkpoint.py` | 参数抽样记录及算法/组件可恢复状态。 |
| `training/progress.py` | 标准 epoch 字段校验和兼容的训练曲线产物。 |
| `tests/test_workflow_registry.py` | workflow 注册、延迟加载、重命名提示及主入口模块化边界。 |

## 13. MC-LDCE、CAL 与 CA2C 最小侵入接入

| 文件 | 作用 |
|---|---|
| `estimators/mc_ldce.py` | MC-LDCE clean prior、label-imputation coefficient matrix、秩/条件数检查和 clean centroid statistic artifact。 |
| `algorithms/mc_ldce.py` | 消费 centroid artifact 的 MC-LDCE 全局平方风险目标。 |
| `training/mc_ldce_experiment.py` | 独立 VolMin 表示/转移估计、固定 feature snapshot、无偏置线性头、transition/statistic artifact、全局目标训练与严格恢复。 |
| `noise/cal.py` | CAL proxy label/status artifact，按稳定 global index 查询并校验来源哈希。 |
| `algorithms/cal.py` | CORES² adjusted loss、transition indicator、二阶 covariance correction 与 CAL objective；保留官方稳定 softmax 数值项。 |
| `training/cal_experiment.py` | CORES² warm-up、posterior/adjusted-loss 对齐、proxy 构建、CAL 训练和恢复；使用可复用的 alpha 耦合 LR 生命周期。 |
| `algorithms/ca2c.py` | CA2C CandidateMemory、N 网络 top-K 与 P 网络 top-K 全类补集 cross-guidance、partial-label 与 negative-learning objective。 |
| `training/ca2c_experiment.py` | P/N 双网络 warm-up、交叉指导、memory、评测和双网络恢复。 |
| `training/reproduction_data.py` | 三个独立 runner 共用的 synthetic/CIFAR noisy-data、稳定 index、loader 与 feature-model 组装；支持论文指定 CIFAR 归一化。 |
| `training/runners.py` | 仅增加 `mc_ldce`、`cal`、`ca2c` 三条懒加载注册；`experiment.py` 不含论文分支。 |
| `tests/test_mc_ldce.py`、`tests/test_cal.py`、`tests/test_ca2c.py` | 三篇论文公式、artifact、索引、梯度和失败边界。 |
| `tests/test_new_paper_training.py`、`tests/test_runner_registry.py` | 三个统一入口 smoke/resume 和 runner 注册边界。 |
| `configs/experiment/{mc_ldce,cal,ca2c}_cifar10_{smoke,reproduction}.yaml` | 三篇论文的 smoke 与正式运行候选配置；正式运行前仍需完成下述论文保真度核验。 |

## 14. L2RW 可信元学习通路

| 文件 | 作用 |
|---|---|
| `data/trusted.py` | 显式 trusted supervision manifest/provider、来源限制、fingerprint 和稳定 index 校验。 |
| `algorithms/l2rw.py` | differentiable virtual SGD、trusted meta-loss、epsilon 二阶梯度和非负归一化权重。 |
| `training/l2rw_experiment.py` | noisy batch 与 trusted batch 分离、真实加权更新、loader RNG/checkpoint/resume。 |
| `configs/experiment/l2rw_cifar10_smoke.yaml` | 只允许 `synthetic_fixture` 的一轮确定性 smoke。 |
| `configs/experiment/l2rw_cifar10_reproduction.yaml` | 要求外部 audited balanced trusted manifest 的正式候选配置。 |
| `tests/test_trusted_supervision.py` | 拒绝普通 validation/test 提升、manifest roundtrip 和显式 target 映射。 |
| `tests/test_l2rw.py`、`tests/test_l2rw_training.py` | 有限差分、权重退化规则、模型不变性及 runner smoke/resume。 |

## 15. 四篇正式复现对齐

| 文件 | 作用 |
|---|---|
| `models/mc_ldce_cnn.py` | MC-LDCE 论文六卷积 CIFAR 模型、显式 feature 输出及固定表示阶段（含 dropout 冻结）。 |
| `models/ca2c_cnn.py` | CA2C 官方结构 SevenCNN 分类器与 projector。 |
| `training/reproduction_data.py` | 可选外部噪声标签、全训练集模式、CIFAR 强视图及专属模型构造。 |
| `training/mc_ldce_experiment.py` | 复用现有 PCSE VolMin 原语但使用独立估计模型；复制估计表示、重置无 bias 分类头并冻结表示后，才采集 snapshot 和构建 statistic。 |
| `training/cal_experiment.py` | 官方固定 IDN 标签下的 proxy 构建与 CAL 二阶段训练。 |
| `training/ca2c_experiment.py` | warm-up candidate 累积、非对称双目标与强视图 consistency。 |
| `training/l2rw_experiment.py` | audited trusted 子集消费及论文 step-budget 调度。 |
| 四份 `*_reproduction.yaml` | 每篇仅一组正式参数；CA2C 保留旧文件名但数据集已改为 CIFAR-100。 |

本轮未修改 `training/experiment.py`、通用 Pipeline、checkpoint、plugin catalog 和现有
`algorithms/pcse/volmin.py`。

## 16. CA2C / L2RW 语义对齐维护（2026-08-04）

| 文件 | 本轮职责 |
|---|---|
| `algorithms/ca2c.py` | 官方 top-K complementary mask、按样本聚合的 negative-label objective、CandidateMemory 校验与 fingerprint。 |
| `training/ca2c_experiment.py` | 复用 CA2C objectives，记录 warm-up/robust phase 和 memory hash，严格恢复双网络状态。 |
| `algorithms/l2rw.py` | 对 noisy/trusted batch、输入输出形状、设备和 target dtype 做 meta-gradient 边界校验。 |
| `training/l2rw_experiment.py` | 保留 trusted manifest 隔离、bilevel 更新、step budget、实时 epoch 输出和 checkpoint/resume。 |
| `tests/test_ca2c.py`、`tests/test_l2rw.py`、`tests/test_l2rw_training.py` | 论文公式、tie 行为、权重梯度和训练恢复测试。 |

本轮仍未修改 `training/experiment.py`、通用 Pipeline、模型和 plugin catalog。

## 17. 六篇论文 1-epoch 调用验收（2026-08-05）

| 方法 | 1-epoch 产物 | 结果 |
|---|---|---|
| Binary Risk | `artifacts/reproductions/binary-risk-natarajan-1epoch/` | Natarajan risk、class-dependent noise、manifest、metrics 均生成。 |
| CWD | `artifacts/reproductions/cwd-cifar10-1epoch/20260805-205014/` | feature snapshot、statistic artifact、global objective、checkpoint 和曲线均生成。 |
| FINE | `artifacts/reproductions/fine-cifar100n-1epoch/20260805-205111/` | 1 个 warm-up epoch 完成；允许 `warmup_epochs == trainer.epochs`，不伪造 robust 阶段。 |
| MentorNet | `artifacts/reproductions/mentornet-cifar10-symmetric04-1epoch/20260805-205208/` | ResNet-101、390 steps、冻结 MentorArtifact、checkpoint 和 final metrics 均生成。 |
| MC-LDCE | `artifacts/reproductions/mc-ldce-cifar10-1epoch/20260805-205506/` | feature/statistic/transition artifact、固定表示 objective 和 checkpoint 均生成。 |
| L2RW | `artifacts/reproductions/l2rw-cifar10-1epoch/20260805-205825/` | 1000-sample trusted manifest、490 meta steps、checkpoint 和 fingerprint 均生成。 |

`data/trusted/l2rw_cifar10_balanced_1000.npz` 是由官方 CIFAR-10 训练标签按固定 seed 生成的本地 audited manifest，属于运行数据，不提交到仓库。

## 18. Binary Risk / FINE formal training record (2026-08-06)

| Method | Artifact | Result |
|---|---|---|
| Binary Risk | `artifacts/reproductions/binary-risk-natarajan-formal-50ep-v3/` | 50 epochs completed; best clean-test accuracy 83.30% at epoch 49; synthetic protocol recorded as implementation validation, not exact numerical reproduction. |
| FINE | `artifacts/reproductions/fine-cifar100-formal-seed23-20260806-v8-150ep/` | 150/300 epochs completed in official warm-up; test accuracy rose from 14.14% to 50.92%. |

Formal entries used here: `configs/experiment/binary_risk_natarajan_reproduction.yaml`, `src/lnl_toolbox/algorithms/binary_risk.py`, `src/lnl_toolbox/training/binary_experiment.py`, and the existing FINE independent runner/configuration.

## 19. MC-LDCE / PDL / L2RW strict-alignment audit (2026-08-06)

| File | Strict-alignment responsibility |
|---|---|
| `src/lnl_toolbox/estimators/mc_ldce.py` | Paper coefficient matrix `M`, clean-prior recovery, Moore--Penrose centroid reconstruction, and auditable convention metadata. |
| `src/lnl_toolbox/algorithms/mc_ldce.py` | Bias-free squared-risk objective only; the paper-inconsistent classifier-bias extension is rejected. |
| `src/lnl_toolbox/algorithms/instance_transition.py` | PDL `beta * NLL(clean posterior)` objective, row-normalized revision matrices, and corrected-posterior evaluation. |
| `src/lnl_toolbox/noise/pdl.py`、`src/lnl_toolbox/noise/generators.py` | Official PDL Algorithm 2, multiplicative NMF representation, percentile anchor selection, and Matrix_optimize basis fitting. |
| `src/lnl_toolbox/training/instance_transition_experiment.py` | PDL raw-input noise generation, official split/normalization, warm-up → correction → revision lifecycle, and artifact-aware resume. |
| `src/lnl_toolbox/training/l2rw_experiment.py` | Official L2RW CIFAR partition/noise generator, noisy train/clean meta/validation/test loaders, and step-level resume. |
| `configs/experiment/pdl_cifar10_reproduction.yaml`、`configs/experiment/l2rw_cifar10_reproduction.yaml` | Paper/official-source reproduction settings; formal configs do not select the legacy audited-manifest path. |
| `tests/test_mc_ldce.py`、`tests/test_pdl.py`、`tests/test_l2rw_training.py` | Non-identity MC-LDCE matrix recovery, PDL beta-NLL, official PDL/L2RW data operations, and resume/smoke coverage. |

The legacy L2RW audited-manifest path remains only as a reusable non-official data adapter for existing smoke/audit fixtures. It is not used by the official reproduction YAML and is not evidence for the official numerical result.

## 20. Strict alignment follow-up: MC-LDCE / PDL / L2RW

| File | Current responsibility |
|---|---|
| `src/lnl_toolbox/noise/pdl.py` | Shared-representation NMF plus split-local anchor/basis artifacts, including the official revision-validation lineage. |
| `src/lnl_toolbox/training/instance_transition_experiment.py` | Official PDL train/validation/revision-validation artifact lifecycle and hash-aware resume; generic instance-transition path remains unchanged. |
| `src/lnl_toolbox/algorithms/pcse/volmin.py` | `PaperVolMinTransition` and Eq. 7 objective for MC-LDCE's VolMin estimation; legacy `DiagonallyDominantTransition` remains only for PCSE compatibility. |
| `src/lnl_toolbox/training/mc_ldce_experiment.py` | Separate paper-style VolMin estimator model, SGD/milestone schedule, then fixed-feature MC-LDCE objective. |
| `configs/experiment/mc_ldce_cifar10_reproduction.yaml` | One paper-aligned MC-LDCE configuration: 150 VolMin epochs, SGD, milestones 30/60, lambda `1e-4`, sigmoid off-diagonal parameterization. |
| `tests/test_pdl.py`, `tests/test_mc_ldce.py` | Split artifact semantics, non-identity matrix recovery, paper VolMin parameterization, differentiability, and failure boundaries. |

The formal MC-LDCE path no longer selects the old PCSE transition parameterization. No global experiment runner or plugin registry change was needed.

## 21. Strict-alignment verification closeout (2026-08-06)

| Method | Verification completed | Remaining boundary |
|---|---|---|
| PDL | Official `Matrix_optimize` reset (`N(0, 0.1)`) and shared Adam lifecycle across train then validation are wired; official three-artifact smoke passed. | Full 100-epoch numerical comparison was not rerun in this audit. |
| MC-LDCE | Formal path uses paper VolMin parameterization with `initial_weight = log(1/(C-2))`, Eq. 7 objective, SGD/milestones, and fixed-feature classifier; 1-epoch smoke passed. | No author-maintained GitHub implementation was found, so this is paper-equation alignment, not line-by-line source matching. |
| L2RW | Official partition/noise/meta-batch/ResNet-32/step-budget path and resume semantics passed focused tests. | Full 80,000-step numerical comparison was not run. |

Validation: PDL 11/11, MC-LDCE 11/11, PCSE/VolMin 28/28, L2RW focused 4/4, full unittest 596/596. No `training/experiment.py`, global pipeline, or plugin-registry change was required.

## 22. Strict-alignment corrections and current evidence (2026-08-06)

This section supersedes only the verification counts and path details above; prior entries are retained as history.

| Method | Corrected formal-path evidence | Not yet claimed |
|---|---|---|
| PDL | `noise/generators.py` now follows the official global NumPy/Torch seed order, truncated-normal rate sampling, Torch softmax, and global `np.random.choice`; `training/instance_transition_experiment.py` converts HWC images back to the official raw CHW flattened layout before Algorithm 2. Focused PDL tests: 13/13. | The corrected 100-epoch formal result has not been rerun, so the old anomalous result is not used as evidence. |
| MC-LDCE | `mc_ldce_cifar10_reproduction.yaml` now selects the paper's six-layer CNN for the transition-estimation path; focused MC-LDCE tests: 11/11, PCSE/VolMin tests: 28/28. | No author-maintained GitHub implementation was found; this is equation/configuration alignment, not line-by-line official-source reproduction. The fixed-feature lifecycle remains a paper interpretation until an implementation source is available. |
| L2RW | The formal path now uses the official CIFAR partition/noise operation, `[-1,1]` preprocessing, 15-unit official ResNet-32 topology, source HVP/meta-weight sign, weight-decay term, and pre-boundary scheduler semantics. Focused L2RW training tests: 10/10, including an official-mode end-to-end short smoke. | The 80,000-step formal result has not been run. |

The official formal paths do not use the legacy audited-manifest adapter. No `training/experiment.py`, global pipeline, or plugin-catalog change was made for these corrections.

Final regression after the official-mode L2RW smoke: full unittest `604/604 OK`; `git diff --check` passed. Conda emitted only the known non-fatal OpenCL vendor temp-file warning.

## 23. Strict source audit correction pass (2026-08-06)

This entry records the source-level corrections made after the previous audit; older entries remain historical.

| Method | Current source-aligned behavior | Evidence boundary |
|---|---|---|
| PDL | `fit_part_representation(..., seed=None)` reproduces `tools.train_m` global NumPy state and final-only normalization. The official split preserves the global `np.random.choice` state and train-then-validation order. Basis fitting keeps the official raw post-`optimizer.step()` train/validation weights; train matrices are clipped only where `main.py` clips them, while validation/revision evaluation applies `tools.norm`. Correction best state is restored before revision. | PDL focused tests: 16/16. Formal 100-epoch rerun is still pending. |
| L2RW | The formal YAML separates model seed `1234` from official data seed `0`. The official meta replica now uses batch-statistics BN with beta-only affine state, while the weighted training model keeps ordinary BN; the meta replica is checkpointed for resume. | L2RW focused tests: 11/11. Formal 80,000-step rerun is still pending. |
| MC-LDCE | Paper equations, six-layer CNN, VolMin parameterization, and artifact lifecycle remain covered. | The paper states the CNN/optimizer setup and Algorithm 1, but no author-maintained GitHub implementation was found; the fixed-feature lifecycle therefore cannot be called an official line-by-line reproduction. |

Relevant primary sources: [PDL official repository](https://github.com/xiaoboxia/Part-dependent-label-noise), [L2RW official repository](https://github.com/uber-research/learning-to-reweight-examples), and [MC-LDCE paper PDF](https://gcatnjust.github.io/ChenGong/paper/ding_sdm22.pdf).

## 24. Source-level correction pass (2026-08-06)

| Method | Maintained source-level fact | Evidence boundary |
|---|---|---|
| PDL | Rechecked the official `tools.py`/`models.py`: `init_params` resets each `Matrix_optimize` linear weight with `N(0, 0.1)`, not `N(0, 0.001)`. The local implementation and regression test now use `0.1`. | PDL focused tests: `17/17`; the corrected 100-epoch run is still pending. |
| L2RW | The official `_flip_data` permutation is now applied to both noisy targets and global image indices, preventing image/label misalignment. The assigned-weight meta replica shares model-C parameters while retaining official batch-statistics, beta-only BN behavior. | L2RW focused tests: `11/11`; the 80,000-step run is still pending. |
| MC-LDCE | Equation-level implementation, six-layer CNN configuration, VolMin path, and artifact lifecycle remain covered. | No author-maintained GitHub implementation was found; therefore this is not claimed as line-by-line source reproduction. |

Full regression after this pass: `609/609 OK`; `git diff --check` passed. No `training/experiment.py`, shared pipeline, or plugin-catalog change was required.
## 25. VolMinNet / UPM / LEND modular additions (2026-08-06)

| Method | Added or changed modules | Evidence boundary |
|---|---|---|
| VolMinNet | `algorithms/pcse/volmin.py` now permits the paper's general nonsingular row-stochastic matrix; `training/volmin_experiment.py` performs joint classifier/transition updates and resumable checkpoints. | Official VolMinNet repository: [xuefeng-li1/Provably-end-to-end-label-noise-learning-without-anchor-points](https://github.com/xuefeng-li1/Provably-end-to-end-label-noise-learning-without-anchor-points). Current runner smoke uses the existing synthetic batch contract; an official CIFAR long run is not claimed. |
| UPM | `noise/upm.py` owns frozen `psi` and indexed `eta`; `algorithms/upm.py` implements Eq. 8, soft-target CE, and projected Eq. 11/12 update; `training/upm_experiment.py` separates noisy warm-up from alternating training. | Official repository: [QizhouWang/instance-dependent-label-noise](https://github.com/QizhouWang/instance-dependent-label-noise). The repository contains incomplete training state code, so the local implementation follows the paper equations and validates the source-compatible parts. |
| LEND | `selectors/lend.py` implements batch-local feature diffusion and `SelectionResult`; `selectors/history.py` adds indexed `Z[N,C]` state; `data/neighbors.py` adds batch graph construction; `training/lend_experiment.py` provides a resumable lifecycle. | No author-maintained implementation was found. This is explicitly a paper-equation implementation, not an official-source line-by-line reproduction. |

The three new runners are registered in `training/runners.py`; no `training/experiment.py`, shared pipeline, model, or plugin-catalog branch was added. Focused smoke paths use `configs/experiment/{volmin_cifar10_smoke,upm_cifar10_smoke,lend_cifar10_smoke}.yaml`.
## Current paper acceptance status (2026-08-07)

The current reproduction acceptance status is maintained in `papers/implement/paper-reproduction-progress.md`. Only MentorNet and CDR remain incomplete; VolMinNet, UPM, and LEND are completed after source/equation alignment and validation.
