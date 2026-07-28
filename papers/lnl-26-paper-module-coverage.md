# 26 篇 LNL 论文与 Toolbox 模块覆盖总览

本文是 LNL Toolbox 的长期论文—taxonomy—代码对齐记录。它回答三个问题：

1. 论文中的哪些能力适合成为通用原语；
2. 当前仓库实现了论文的哪一部分；
3. 哪些方法必须保留为独立 `Algorithm` / `Pipeline`。

本文不把“类型兼容”“plugin 可构造”或“单元测试可拼接”视为完整论文实现。只有模型数量、目标函数、状态、更新顺序和训练生命周期都符合原文时，才允许声明完整实现。

## 项目总目标与实现原则

本项目的一切接口设计、模块实现、论文复现、训练接入和工程测试，最终都服务于本文定义的目标。本文同时定义可复用原语与独立 Algorithm/Pipeline 的边界，记录代码实际覆盖层级，并约束可以和禁止对外声明的能力。

### 最终目标

- 为可复用论文机制提供语义清晰、边界稳定的通用原语；
- 为具有独立模型数量、目标函数、状态和更新顺序的方法实现完整 Algorithm/Pipeline；
- 将原语接入真实训练、checkpoint 和 resume 生命周期；
- 为论文方法提供原文对应的 acceptance tests；
- 准确区分完整实现、精确子组件、通用工程原语、近似实现和未实现；
- 不以类型兼容、plugin 注册或组件拼接夸大完成度。

### 功能保护原则

“不得撤销已有功能”只保护与论文公式或已批准工程语义一致、职责清晰、已有测试、具有合法数据来源且不妨碍其他论文实现空间的能力。违反论文语义的临时接入、宽泛万能输入、没有合法 producer/consumer 的伪链路、silent failure、样本错位、错误 resume 和仅用于证明可构造的 plugin 不属于应原样保留的功能，应通过兼容适配、局部重构或显式弃用修正。

必须保护的已验证能力包括：

- CE、GCE、NCE、MAE、RCE、APL 的逐样本 Loss 与训练消费链；
- AllSelector、SmallLossSelector、keep-rate schedule；
- ContributionResult、ReductionSpec 和统一 reducer；
- Binary RCN importance-weight 精确公式；
- ReliabilityResult、DivideMix GMM 和 stable-index adapter；
- PosteriorSnapshot、Anchor/Dual-T estimator 和 TransitionArtifact；
- StandardUpdatePolicy、CDRUpdatePolicy；
- 合法的 plugin、配置、checkpoint/resume 路径；
- Co-teaching legacy exchange primitive；
- 已有测试证明的 backward compatibility。

使用当前分类器 softmax 冒充独立 noisy-label posterior、未经 stable-index 校验应用 target result、resume 缺失 artifact 时静默重估、保存 state 但不恢复，以及用通用原语冒充完整论文方法，均属于必须修正的过渡行为。

### 开发优先级

1. 完成 Forward/Backward Loss Correction 的 RiskCorrector 和训练消费链；
2. 建立完整 Co-teaching 双模型 Algorithm；
3. 补齐 CDR paper preset 和 early-learning lifecycle；
4. 完成 Importance Reweighting 的 posterior/rate producer 与 Pipeline；
5. 选择 MC-LDCE、CWD 或 PCSE 完成一个 statistic vertical slice；
6. 将 DivideMix 实现为独立双网络 SSL Pipeline；
7. 再逐步补充其他论文完整实现。

主线之外不继续增加无目标容器、空接口或没有 consumer 的 plugin。

### 完整论文方法的完成标准

只有同时满足下列条件，才能标记为“完整实现”：

- 模型数量和 objective 与原文一致；
- 必需的跨 batch、跨 epoch 或全数据状态已经实现；
- 更新顺序和阶段生命周期与原文一致；
- checkpoint/resume 覆盖全部必要状态；
- 具有论文专属配置和 acceptance tests；
- 完成可重复的端到端实验验证；
- 文档准确记录实现偏差与适用范围。

只完成 loss、selector、estimator、update rule 或其他单一模块时，只能声明为“精确子组件”或“通用工程原语”。

### 合并与验收原则

分支合入 `ce_baseline` 前必须说明：补充了哪项覆盖能力；属于原语、精确子组件还是完整方法；是否存在真实 producer/consumer；是否进入训练、checkpoint/resume；是否改变接口或配置；论文忠实性风险；测试是否充分；以及是否需要同步更新本文。不能回答这些问题的代码不应直接进入稳定基线。

## 1. 阅读范围与版本

审计基线为 `ce_baseline` 的 commit
`d53ac44d0ab6d441c5985746da3f5135e9d69994`。论文清单来自
`papers/implement/paper-baseline-taxonomy.md` 和
`papers/implement/paper-implementation-guideline.md`，实现判断来自当前源码与测试。

实际论文根目录是 `D:\ABhomework\科研\label-noise\papers`。审计时该目录只有
24 个 PDF。以下两篇没有本地 PDF，使用在线官方原文核验：

- CDR / Robust Early-Learning：ICLR 2021
  [OpenReview 原文 PDF](https://openreview.net/pdf?id=Eql5b1_hTE4)；
- MC-LDCE：SDM 2022，[arXiv v1](https://arxiv.org/abs/2203.10858)。

其余 24 篇使用本地 PDF，并以论文官方页面核对版本。因而
`papers/README.md` 中“本地 26/26 篇 PDF”不是本次审计时的真实状态。

本文记录的是方法结构和工程边界，不是实验复现报告。论文中的数据集、网络宽度、
训练轮数和最终精度只有在影响模块边界时才会出现。

## 2. Toolbox taxonomy

本文使用 method-first 的公共视角：用户最终选择论文方法，内部 Algorithm 再组合原语，
而不是让用户自由拼装出一个可能不忠实于论文的方法。

| 原语 | 当前统一语义 | 不能承担的职责 |
|---|---|---|
| `Loss` | `logits, targets -> per-sample loss [B]` | 模型协调、样本历史、optimizer step |
| `Selector` | detached batch scores + stable indices -> hard mask | 可靠性估计、双网络交换、标签修改 |
| `WeightProvider` | method-specific input -> non-negative detached weights | meta virtual update、posterior/noise-rate估计 |
| `ReliabilityEstimator` | method-specific evidence -> sample-aligned reliability；越大越可靠 | hard selection、dataset split |
| `StatisticResult[T]` | method-specific statistic 的轻量结果容器 | 统一规定所有 estimator 的 fit/compute 生命周期 |
| `TransitionEstimator` | noisy posterior snapshot -> class-conditional transition artifact | 联合训练 transition、使用 transition 修正风险 |
| `ContributionResult` | hard mask + continuous non-negative weights | signed risk、soft label、多目标训练 |
| `ReductionSpec` | contribution 加权后的显式归约 | 论文专属目标构造 |
| `ParameterUpdatePolicy` | 已构造 scalar objective 后的 backward/update 规则 | 双网络生命周期、meta-objective |
| `Stateful` | 可选 `state_dict/load_state_dict` 合同 | 自动获得 checkpoint ownership |
| Adapter | 显式完成方向、对齐或结果合同转换 | 隐式决定论文策略 |

## 3. 完成度定义

覆盖矩阵中的状态只能按以下定义理解：

- **完整实现**：论文要求的模型、objective、状态、更新顺序和生命周期均已实现并测试。
- **精确子组件**：公式或算法步骤忠实于原文，但只是完整方法的一部分。
- **通用工程原语**：可被论文复用，但不是论文专属实现。
- **近似实现**：有意采用与原文不完全相同的可运行近似。
- **接口已存在但无实现**：只有适配合同或容器，没有论文对应的具体 producer。
- **未实现**：当前没有忠实实现。
- **不适合模块化**：论文核心必须由完整 Algorithm/Pipeline 承担。
- **证据不足**：现有原文证据不足以做可靠判断。

另外必须分开记录：

1. 类型能否连接；
2. 是否有跨组件单元测试；
3. 是否接入真实训练入口；
4. 是否完成整篇论文。

## 4. 总覆盖矩阵

| # | 论文 | 可模块化功能 | taxonomy 模块 | 当前代码位置 | 状态 | 训练接入 | 完整论文 | 主要剩余工作 |
|---:|---|---|---|---|---|---|---|---|
| 01 | UPM | instance noise posterior | NoiseModel / LabelRefiner | 无 | 未实现 | 否 | 否 | latent noise model、交替优化 |
| 02 | CAL | second-order statistics | Statistic / RiskCorrector | `src/lnl_toolbox/estimators/base.py` 仅容器 | 接口已存在但无实现 | 否 | 否 | covariance estimator、CAL risk |
| 03 | PDL | part-dependent transition | NoiseModel | 无论文实现 | 未实现 | 否 | 否 | basis transitions、instance mixing |
| 04 | JoCoR | joint loss 中的逐样本 score | Algorithm / Selector primitive | 通用 Loss、Selector | 未实现 | 否 | 否 | 双网络、co-regularization、联合更新 |
| 05 | DSS | class/instance debias evidence | Reliability / LabelRefiner | 通用合同 | 未实现 | 否 | 否 | MDA、CCS 历史与趋势检验 |
| 06 | CDR | critical parameter update | ParameterUpdatePolicy | `src/lnl_toolbox/algorithms/cdr.py` | 精确子组件 | 是 | 否 | 完整 early-learning lifecycle |
| 07 | CNLCU | loss confidence bounds | Reliability / Selector | 通用合同 | 未实现 | 否 | 否 | 历史状态、区间估计、双网络流程 |
| 08 | MentorNet | learned sample weights | WeightProvider | 泛型合同 | 接口已存在但无实现 | 否 | 否 | Mentor model 与 student lifecycle |
| 09 | Co-teaching | peer small-loss exchange | Algorithm | `src/lnl_toolbox/algorithms/coteaching.py` | 精确子组件 | 否 | 否 | 双模型 Algorithm/checkpoint |
| 10 | Loss Correction | Anchor estimate、Forward/Backward corrected risk | TransitionEstimator / RiskCorrector | `src/lnl_toolbox/noise/estimators.py`、`src/lnl_toolbox/algorithms/transition_risk.py` | 精确子组件 | 是 | 否 | 论文 preset、acceptance experiment |
| 11 | Normalized Losses / APL | NCE、MAE、RCE、APL | Loss | `src/lnl_toolbox/losses/torch_losses.py` | 精确子组件 | 是 | 否 | 论文全部组合与复现实验 |
| 12 | GCE | standard \(L_q\) | Loss | `src/lnl_toolbox/losses/torch_losses.py` | 精确子组件 | 是 | 否 | truncated \(L_q\) 更新 |
| 13 | VolMinNet | trainable transition | NoiseModel / Pipeline | 无 | 未实现 | 否 | 否 | volume regularizer、联合优化 |
| 14 | Natarajan | unbiased binary risk | RiskCorrector | 无 | 未实现 | 否 | 否 | signed corrected risk |
| 15 | T-Revision | Anchor initialization | Transition primitive | `src/lnl_toolbox/noise/estimators.py` | 通用工程原语 | 否 | 否 | trainable slack、revision stages |
| 16 | Dual-T | factorized transition estimate | TransitionEstimator | `src/lnl_toolbox/noise/estimators.py` | 精确子组件 | 是 | 否 | 论文 preset、正式复现实验 |
| 17 | MC-LDCE | clean centroid statistic | Statistic / RiskCorrector | `StatisticResult` | 接口已存在但无实现 | 否 | 否 | centroid estimator、global risk |
| 18 | Importance Reweighting | binary asymmetric-RCN weight | WeightProvider | `src/lnl_toolbox/treatments/weights.py` | 精确子组件 | 否 | 否 | posterior/rate estimation 与构造入口 |
| 19 | CWD | class-wise centroid | Statistic / RiskCorrector | `StatisticResult` | 接口已存在但无实现 | 否 | 否 | auxiliary sets、risk consumer |
| 20 | PCSE | per-class statistics | Statistic / PostProcessor | `StatisticResult` | 接口已存在但无实现 | 否 | 否 | producer 与 inference consumer |
| 21 | DLD | directional label diffusion | LabelRefiner / Pipeline | 无 | 不适合模块化 | 否 | 否 | 完整生成式训练流程 |
| 22 | FINE | noisy-subset objectives | Treatment consumer / Pipeline | 通用 contribution | 未实现 | 否 | 否 | forgetting、negative learning |
| 23 | CA2C | asymmetric co-learning | dual-network Pipeline | 无 | 不适合模块化 | 否 | 否 | 两类模型、cross-guidance |
| 24 | DivideMix | loss-GMM clean probability | ReliabilityEstimator | `src/lnl_toolbox/estimators/dividemix_gmm.py` | 精确子组件 | 否 | 否 | co-divide、MixMatch、双网络 |
| 25 | L2RW | meta-gradient weights | MetaUpdater / Pipeline | 无忠实接口 | 未实现 | 否 | 否 | virtual update、clean meta batch |
| 26 | LEND | embedding graph evidence | Reliability / LabelRefiner | 通用合同 | 未实现 | 否 | 否 | graph、label dilution、indexed state |

矩阵没有把任何一篇标记为完整论文实现。多个组件已经论文精确，但完整方法所要求的
其余目标、状态或生命周期仍不存在。

## 5. Loss 类论文

当前 `Loss` 合同和普通监督消费链已经稳定：

```text
logits + observed targets
-> detached copy to Selector
-> original per-sample loss keeps autograd
-> ContributionResult
-> explicit reducer
-> scalar objective
```

CE、GCE、NCE、MAE、RCE、APL 均输出逐样本 `[B]` 并接入 plugin 和监督训练。
但这不覆盖依赖 transition 的 Forward/Backward、允许负风险的 Natarajan、
或依赖全局统计的 MC-LDCE/CWD。后几类不应硬塞进普通非负 WeightProvider。

## 6. Selection / Weighting 类论文

`AllSelector`、`SmallLossSelector`、constant/linear keep-rate schedule 和
`SelectorContributionAdapter` 已进入真实单模型训练。它们只解决单 batch、
单 score 向量的 hard selection。

Co-teaching、JoCoR 需要双网络和固定 exchange/update 顺序；CNLCU 需要跨 epoch
历史；DSS 会改变候选监督；DivideMix 会把数据分成 labeled/unlabeled；
这些都不能通过增加一个 Selector 名称忠实实现。

`WeightProvider` 能表达已经计算好的连续非负权重。它不能独自实现 MentorNet 的
teacher lifecycle，也不能实现 L2RW 的 virtual update 和 clean-validation
meta-gradient。

## 7. Reliability / Estimation 类论文

`ReliabilityResult(sample_indices, scores, metrics)` 固定“越大越可靠”。
`ReliabilityToSelectionInputAdapter` 使用 stable index 从 dataset `[N]` 结果抽取
batch `[B]`，重排后显式取负，适配“越小越优先”的 `SmallLossSelector`。

DivideMix GMM 已有精确的 loss normalization、二分量 GMM 和低均值 component
clean probability；它可通过 adapter、Selector、Contribution 和 reducer 被测试消费。
这只是 dataset reliability 到 batch ranking 的组件链，不是 DivideMix split。

CNLCU、DSS、LEND 的具体 evidence producer 尚未实现。它们即使输出 scalar
reliability，也仍需要未来 Algorithm 管理历史、特征或监督状态。

## 8. Transition 类论文

当前实现包括：

- `PosteriorSnapshot`：noisy posterior、observed targets、global indices；
- `AnchorTransitionEstimator`：Patrini 等人的 anchor 估计；
- `DualTransitionEstimator`：`T_club @ T_spade`；
- `TransitionArtifact`：矩阵方向、校验、hash、保存和加载；
- `collect_posterior_snapshot`：按 stable index 采集 noisy posterior；
- `ForwardRiskCorrector` / `BackwardRiskCorrector`：消费冻结的 TransitionArtifact 并输出逐样本 corrected risk；
- `StandardNoisyERMPipeline`：编排 warm-up、snapshot、transition estimation、artifact 恢复和监督训练消费。

这些组件通过 `pipeline.transition_estimator` 与 `pipeline.risk_corrector` 的嵌套配置进入训练；resume 必须加载并核验已有 artifact，不允许静默重新估计。该链路仍只是 Loss Correction 和 Dual-T 的精确组件组合，尚无论文专属 preset、acceptance experiment 或完整复现声明。

PDL、VolMinNet、T-Revision 的 transition 是 trainable model/lifecycle 的一部分，
不能伪装成一次性离线 estimator。

## 9. Parameter Update 类论文

`StandardUpdatePolicy` 和 `CDRUpdatePolicy` 已接入
`SupervisedClassificationAlgorithm`、plugin、checkpoint 和 resume。
CDR 组件按每次更新计算 `abs(gradient * parameter)`，选择 critical scalars，
并采用论文约束的 SGD、梯度遮罩与 L1 更新。

它仍只是 Robust Early-Learning 的参数更新组件。完整论文声明还需要明确复现
early-learning 的训练/停止生命周期和论文实验配方。

## 10. Multi-network / SSL / Meta-learning 类论文

以下方法必须拥有独立 Algorithm/Pipeline：

- Co-teaching：两个网络分别选样本并交叉更新；
- JoCoR：两个网络的 joint supervised + co-regularization loss；
- MentorNet：Mentor/Student 两类模型及课程状态；
- DivideMix：双网络 co-divide、label refinement、MixMatch；
- L2RW：virtual model step、clean meta batch、高阶梯度；
- CA2C：不同范式双网络、partial/negative supervision；
- DLD：生成式 label diffusion；
- UPM、VolMinNet：噪声模型和分类器联合或交替优化。

这些方法可以复用 Loss、reducer、artifact 或 stable index，但不能注册同名小组件来
代表论文。

## 11. 逐篇映射（一）：P01–P12

### 映射前模块汇总

当前已完成或基本完成的原语：

- 逐样本 CE、GCE、NCE、MAE、RCE 和 APL Loss，以及 Loss plugin/监督训练消费链；
- All/SmallLoss Selector、keep-rate schedule、Selector adapter 和统一 Treatment/Reducer；
- 泛型 WeightProvider、Binary RCN importance-weight 精确子组件及 contribution adapter；
- ReliabilityResult、DivideMix GMM clean-probability 精确子组件及 stable-index selection adapter；
- PosteriorSnapshot、Anchor/Dual-T TransitionEstimator 和可校验、可 roundtrip 的 TransitionArtifact；
- Forward/Backward RiskCorrector 及其 StandardNoisyERMPipeline 训练、artifact、checkpoint/resume 消费链；
- Standard/CDR ParameterUpdatePolicy，以及 CDR 的训练、checkpoint 和 resume 接入。

当前明显缺失的原语或消费链：

- Natarajan、CAL、MC-LDCE 和 CWD 所需的 RiskCorrector；
- 具体 StatisticEstimator，以及 StatisticResult 的生产者和训练/推理消费者；
- UPM、PDL、VolMinNet 和 T-Revision 所需的 trainable NoiseModel；
- DSS、CNLCU、LEND 所需的 stateful reliability/history producer；
- label refinement、dataset split、post-processing 和通用 component-state checkpoint ownership；
- WeightProvider 和 ReliabilityEstimator 的公开 YAML/plugin/训练构造入口。

### P01 — UPM

- **版本与原文**：AAAI 2021，[官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/17221)，本地 `01_instance_dependent/01_upm_aaai2021.pdf`。
- **研究问题**：在 instance-dependent noise 下显式建模 noisy label 生成过程。
- **核心机制**：confusing/unconfusing latent variable、实例相关概率和交替优化。
- **关键输入/输出**：输入样本与 noisy label；输出 latent/noise posterior 和分类模型。
- **可模块化部分**：NoiseModel 输出、posterior/label-refinement artifact。
- **必须 Algorithm 化的部分**：多个参数块的交替估计和监督更新顺序。
- **当前实现/状态**：无对应实现；**未实现**。
- **代码与测试位置**：无；现有 `src/lnl_toolbox/noise/generators.py` 只生成实验噪声，不是 UPM。
- **当前缺失**：模型、objective、state、checkpoint 和 pipeline。
- **可以声明**：Toolbox 有 IDN 数据生成器和通用训练基础。
- **禁止声明**：不得称 IDN generator、ReliabilityResult 或 noisy CE 为 UPM。

### P02 — CAL

- **版本与原文**：CVPR 2021，[官方页面](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_A_Second-Order_Approach_to_Learning_With_Instance-Dependent_Label_Noise_CVPR_2021_paper.html)，本地 `01_instance_dependent/02_cal_cvpr2021.pdf`。
- **研究问题**：用二阶统计修正 heterogeneous IDN 导致的风险偏差。
- **核心机制**：估计 noise/Bayes-label covariance 项，把 IDN risk 转成更易处理的形式。
- **关键输入/输出**：全局预测/标签统计；输出 covariance statistics 和 corrected risk。
- **可模块化部分**：CAL 专用 StatisticEstimator、RiskCorrector。
- **必须 Algorithm 化的部分**：统计估计与 corrected objective 的训练编排。
- **当前实现/状态**：只有 `src/lnl_toolbox/estimators/base.py::StatisticResult` 容器；**接口已存在但无实现**。
- **代码与测试位置**：容器测试 `tests/test_estimators.py`；无 CAL 测试。
- **当前缺失**：具体 statistic producer、CAL risk consumer。
- **可以声明**：已有轻量 statistic result 合同。
- **禁止声明**：不得因存在 `StatisticResult` 声称已实现 CAL。

### P03 — PDL

- **版本与原文**：NeurIPS 2020，[官方页面](https://proceedings.neurips.cc/paper/2020/hash/5607fe8879e4fd269e88387e8cb30b7e-Abstract.html)，本地 `01_instance_dependent/03_pdl_neurips2020.pdf`。
- **研究问题**：用 part-dependent structure 逼近 instance-dependent transition \(T(x)\)。
- **核心机制**：多个 part-level transition basis 与 instance-dependent mixing。
- **关键输入/输出**：样本特征；输出每个样本的 transition matrix。
- **可模块化部分**：basis transition parameterization、instance transition output contract。
- **必须 Algorithm 化的部分**：特征网络、basis/mixing 和 noisy objective 的联合训练。
- **当前实现/状态**：现有 Anchor/Dual-T 只输出全局 class-conditional T；**未实现**。
- **代码与测试位置**：共享基础在 `src/lnl_toolbox/noise/transition.py`；无 PDL 测试。
- **当前缺失**：\(T(x)\)、part basis、联合 optimizer/state。
- **可以声明**：已有 class-conditional transition artifact。
- **禁止声明**：不得把 Anchor/Dual-T 或 IDN generator 称为 PDL。

### P04 — JoCoR

- **版本与原文**：CVPR 2020，[官方页面](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.html)，本地 `02_sample_selection/04_jocor_cvpr2020.pdf`。
- **研究问题**：通过双网络 agreement 降低 noisy-label memorization。
- **核心机制**：两个网络的 supervised loss 加 co-regularization，按 joint loss 选样本并同时更新。
- **关键输入/输出**：双网络 logits；输出 joint per-sample loss、mask 和双模型更新。
- **可模块化部分**：agreement loss、small-loss ranking primitive。
- **必须 Algorithm 化的部分**：双网络、joint objective、同步更新和 checkpoint。
- **当前实现/状态**：只有通用 Loss/SmallLossSelector；**未实现**。
- **代码与测试位置**：`src/lnl_toolbox/selectors/basic.py`、`tests/test_selectors.py` 不是 JoCoR 测试。
- **当前缺失**：JoCoR objective、双模型 lifecycle。
- **可以声明**：small-loss 是可复用原语。
- **禁止声明**：不得把 SmallLossSelector 或 Co-teaching helper 称为 JoCoR。

### P05 — DSS

- **版本与原文**：CVPR 2026，[官方页面](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.html)，本地 `02_sample_selection/05_dss_cvpr2026.pdf`。
- **研究问题**：同时缓解 small-loss selection 的 class bias 与 instance bias。
- **核心机制**：Marginal Distribution Adjustment、候选类预测历史和 trend-based Candidate Class Selection。
- **关键输入/输出**：epoch-level prediction history；输出调整后的预测/候选监督。
- **可模块化部分**：history statistic、reliability evidence。
- **必须 Algorithm 化的部分**：历史维护、监督空间修改及每轮更新顺序。
- **当前实现/状态**：Reliability/Selector 仅提供合同；**未实现**。
- **代码与测试位置**：`src/lnl_toolbox/estimators/base.py`、`src/lnl_toolbox/selectors/base.py`；无 DSS 测试。
- **当前缺失**：MDA、CCS、趋势检验和 state。
- **可以声明**：stable-index reliability 合同可承载部分证据。
- **禁止声明**：不得把 scalar Top-K/small-loss 称为 DSS。

### P06 — CDR / Robust Early-Learning

- **版本与原文**：ICLR 2021，[官方 PDF](https://openreview.net/pdf?id=Eql5b1_hTE4)；本地 PDF 缺失。
- **研究问题**：在网络开始记忆噪声前保护与 clean generalization 相关的 critical parameters。
- **核心机制**：按 \(|g_j\theta_j|\) 排序参数，critical 与 non-critical 参数采用不同更新。
- **关键输入/输出**：scalar objective、参数和梯度；输出 optimizer update 及参数比例指标。
- **可模块化部分**：CriticalParameterMasks、ParameterUpdatePolicy。
- **必须 Algorithm 化的部分**：完整 early-learning 训练/停止配方。
- **当前实现/状态**：`src/lnl_toolbox/algorithms/cdr.py::CDRUpdatePolicy`；**精确子组件**。
- **代码与测试位置**：`src/lnl_toolbox/algorithms/update_policy.py`、`tests/test_cdr.py`、`tests/test_update_policy.py`。
- **当前缺失**：对完整论文 lifecycle 的正式复现声明。
- **可以声明**：paper-mode CDR parameter-update component 已接入训练/checkpoint/resume。
- **禁止声明**：不得仅凭 update policy 声称完整 Robust Early-Learning。

### P07 — CNLCU

- **版本与原文**：ICLR 2022，[官方页面](https://openreview.net/forum?id=xENf4QUL4LW)，本地 `02_sample_selection/07_uncertainty_selection.pdf`。
- **研究问题**：避免把高 loss 但欠代表的 clean 样本长期排除。
- **核心机制**：跨时间 loss 的置信区间/lower bound 和探索式选择。
- **关键输入/输出**：indexed loss history；输出 uncertainty-adjusted selection evidence。
- **可模块化部分**：stateful ReliabilityEstimator、区间估计器。
- **必须 Algorithm 化的部分**：历史采集、选择/试用反馈和论文网络协作。
- **当前实现/状态**：无具体 producer；**未实现**。
- **代码与测试位置**：通用合同 `src/lnl_toolbox/estimators/base.py`；无 CNLCU 测试。
- **当前缺失**：history state、bound 公式、selection lifecycle。
- **可以声明**：ReliabilityResult 的方向适合表达其可靠性结果。
- **禁止声明**：不得把当前 epoch 的 SmallLossSelector 称为 CNLCU。

### P08 — MentorNet

- **版本与原文**：ICML 2018，[PMLR 页面](https://proceedings.mlr.press/v80/jiang18c.html)，本地 `02_sample_selection/08_mentornet_icml2018.pdf`。
- **研究问题**：学习一个数据驱动 curriculum，为 StudentNet 动态赋权。
- **核心机制**：MentorNet 根据 loss、epoch、label/history 等信号生成 sample weights。
- **关键输入/输出**：mentor features/state；输出连续权重。
- **可模块化部分**：Mentor-specific WeightProvider 输出边界。
- **必须 Algorithm 化的部分**：Mentor 训练/加载、Student 更新及课程生命周期。
- **当前实现/状态**：只有泛型 `WeightProvider[InputT]`；**接口已存在但无实现**。
- **代码与测试位置**：`src/lnl_toolbox/treatments/weights.py`、`tests/test_importance_reweighting.py` 仅测通用 adapter。
- **当前缺失**：Mentor model、input schema、state 和 pipeline。
- **可以声明**：统一 reducer 能消费预先算好的 Mentor weights。
- **禁止声明**：不得把固定 keep-rate 或任意 WeightProvider 称为 MentorNet。

### P09 — Co-teaching

- **版本与原文**：NeurIPS 2018，[官方页面](https://papers.nips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html)，本地 `02_sample_selection/09_coteaching_neurips2018.pdf`。
- **研究问题**：用两个网络互相提供 small-loss 样本，减少自我确认偏差。
- **核心机制**：每个网络独立排序 loss，并用 peer 选出的样本更新自己。
- **关键输入/输出**：两组 per-sample loss；输出交叉 index sets。
- **可模块化部分**：peer exchange helper、keep-rate schedule。
- **必须 Algorithm 化的部分**：两个模型/optimizer、交叉更新顺序和 checkpoint。
- **当前实现/状态**：`src/lnl_toolbox/algorithms/coteaching.py::coteaching_exchange`；**精确子组件**。
- **代码与测试位置**：`tests/test_coteaching.py`、legacy plugin regression in `tests/test_plugins.py`。
- **当前缺失**：可训练双网络 Algorithm/Pipeline。
- **可以声明**：已实现 legacy Co-teaching exchange primitive。
- **禁止声明**：不得声称 Toolbox 已完成 Co-teaching 训练。

### P10 — Forward / Backward Loss Correction

- **版本与原文**：CVPR 2017，[官方页面](https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html)，本地 `03_robust_loss/10_loss_correction_cvpr2017.pdf`。
- **研究问题**：在 class-conditional T 已知或估计时构造一致的 corrected risk。
- **核心机制**：Forward 在 probability side 乘 T；Backward 用 \(T^{-1}\) 修正 loss vector；并给出 anchor 估计。
- **关键输入/输出**：logits、targets、T；输出 corrected per-sample risk。
- **可模块化部分**：AnchorTransitionEstimator、Forward/Backward RiskCorrector。
- **必须 Algorithm 化的部分**：snapshot/estimate/freeze/correct/train 编排。
- **当前实现/状态**：Anchor estimator、Forward/Backward RiskCorrector 和 staged training consumer 已连接；**精确子组件**。
- **代码与测试位置**：`src/lnl_toolbox/noise/estimators.py`、`src/lnl_toolbox/algorithms/transition_risk.py`、`src/lnl_toolbox/training/pipeline.py`、`tests/test_transition_estimators.py`、`tests/test_transition_risk.py`、`tests/test_pipeline.py`。
- **当前缺失**：论文专属 preset、完整 acceptance experiment 和参考结果复现。
- **可以声明**：实现并接通 Patrini Anchor + Forward/Backward corrected-risk component chain。
- **禁止声明**：不得仅凭通用 pipeline 组件链声称已完整复现 Loss Correction 论文。

### P11 — Normalized Losses / APL

- **版本与原文**：ICML 2020，[PMLR 页面](https://proceedings.mlr.press/v119/ma20c.html)，本地 `03_robust_loss/11_normalized_losses_icml2020.pdf`。
- **研究问题**：通过 normalized active loss 与 passive loss 组合兼顾鲁棒性和学习能力。
- **核心机制**：NCE 等 normalized loss，加 MAE/RCE passive loss 形成 APL。
- **关键输入/输出**：logits、observed targets；输出逐样本 loss。
- **可模块化部分**：NCE、MAE、RCE、APL loss modules。
- **必须 Algorithm 化的部分**：普通单模型训练即可；正式复现仍需论文配置。
- **当前实现/状态**：上述组件及 plugin 已有；**精确子组件**。
- **代码与测试位置**：`src/lnl_toolbox/losses/torch_losses.py`、`tests/test_losses.py`、`tests/test_plugins.py`。
- **当前缺失**：论文覆盖的全部 active/passive 组合及正式 benchmark。
- **可以声明**：支持 NCE+MAE/RCE 的 paper-faithful P0 APL。
- **禁止声明**：不得声称已复现论文全部组合和结果。

### P12 — GCE

- **版本与原文**：NeurIPS 2018，[官方页面](https://papers.nips.cc/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html)，本地 `03_robust_loss/12_gce_neurips2018.pdf`。
- **研究问题**：在 CE 的学习效率与 MAE 的噪声鲁棒性之间插值。
- **核心机制**：\(L_q=(1-p_y^q)/q\)，并讨论 truncated \(L_q\)。
- **关键输入/输出**：logits、target、q；输出逐样本 \(L_q\)。
- **可模块化部分**：standard GCE loss。
- **必须 Algorithm 化的部分**：truncated 版本的样本权重交替更新。
- **当前实现/状态**：standard `GeneralizedCrossEntropyLoss`；**精确子组件**。
- **代码与测试位置**：`src/lnl_toolbox/losses/torch_losses.py`、`tests/test_losses.py`。
- **当前缺失**：truncated \(L_q\) 及其更新状态。
- **可以声明**：standard GCE \(L_q\) 可端到端训练。
- **禁止声明**：不得声称实现 truncated GCE 或整篇所有变体。

## 12. 逐篇映射（二）：P13–P20

### P13 — VolMinNet

- **版本与原文**：ICML 2021，[PMLR 页面](https://proceedings.mlr.press/v139/li21l.html)，本地 `04_statistic_estimation/Provably End-to-end Label-noise Learning without Anchor Points.pdf`。
- **研究问题**：不依赖 anchor point，联合识别 clean posterior 与 transition。
- **核心机制**：noisy CE 与 transition simplex volume regularization 同时优化。
- **关键输入/输出**：clean posterior network 与 trainable T；输出 classifier 和 T。
- **可模块化部分**：trainable transition parameterization、volume regularizer。
- **必须 Algorithm 化的部分**：两个 objective/optimizer 的端到端联合训练。
- **当前实现/状态**：离线 artifact 不能表达 trainable T；**未实现**。
- **代码与测试位置**：无 VolMinNet；`src/lnl_toolbox/noise/transition.py` 仅为共享 artifact。
- **当前缺失**：NoiseModel、regularizer、lifecycle、checkpoint。
- **可以声明**：Toolbox 有 transition artifact 基础。
- **禁止声明**：不得把 Anchor/Dual-T estimator 称为 VolMinNet。

### P14 — Natarajan Unbiased Risk

- **版本与原文**：NeurIPS 2013，[官方页面](https://papers.nips.cc/paper/2013/hash/3871bd64012152bfb53fdf04b401193f-Abstract.html)，本地 `04_statistic_estimation/learning-with-noisy-labels-Paper.pdf`。
- **研究问题**：在已知二分类 class-conditional flip rates 下构造无偏风险。
- **核心机制**：组合正/反标签 loss，除以 \(1-\rho_+-\rho_-\)，单样本 corrected risk 可为负。
- **关键输入/输出**：binary losses 和两类 noise rates；输出 signed corrected risk。
- **可模块化部分**：BinaryRiskCorrector。
- **必须 Algorithm 化的部分**：普通优化可消费，但 reducer 必须允许 signed risk。
- **当前实现/状态**：无对应实现；**未实现**。
- **代码与测试位置**：无；`src/lnl_toolbox/treatments/base.py` 禁止负 sample weights。
- **当前缺失**：RiskCorrector 合同与 signed-risk 测试。
- **可以声明**：现有二分类 RCN rate 语义可作为未来输入参考。
- **禁止声明**：不得把 Binary RCN Importance WeightProvider 称为 Natarajan risk。

### P15 — T-Revision

- **版本与原文**：NeurIPS 2019，[官方页面](https://papers.nips.cc/paper/2019/hash/9308b0d6e5898366a4a986bc33f3d3e7-Abstract.html)，本地 `04_statistic_estimation/Are-anchor-points-really-indispensable-in-label-noise-learning.pdf`。
- **研究问题**：无精确 anchor 时修订初始 transition estimate。
- **核心机制**：high-posterior 初始化 T，再学习 slack revision 并进行 importance reweighting。
- **关键输入/输出**：initial T、trainable slack、classifier posterior；输出 revised T 和 weighted objective。
- **可模块化部分**：Anchor initialization、revision layer、专用 weight calculation。
- **必须 Algorithm 化的部分**：初始化、joint revision、validation/训练阶段顺序。
- **当前实现/状态**：只有可复用 Anchor primitive；**通用工程原语**。
- **代码与测试位置**：`src/lnl_toolbox/noise/estimators.py`、`tests/test_transition_estimators.py`。
- **当前缺失**：slack、revision optimizer、importance stage。
- **可以声明**：Anchor component 可作为 T-Revision 前置原语。
- **禁止声明**：不得把一次 Anchor estimate 注册为 T-Revision。

### P16 — Dual-T

- **版本与原文**：NeurIPS 2020，[官方页面](https://papers.nips.cc/paper/2020/hash/512c5cad6c37edb98ae91c8a76c3a291-Abstract.html)，本地 `04_statistic_estimation/dual-t-reducing-estimation-error-for-transition-matrix-in-label-noise-learning.pdf`。
- **研究问题**：通过中间标签分解降低直接 transition estimation error。
- **核心机制**：估计 \(T^{club}\) 和 \(T^{spade}\)，再计算二者乘积。
- **关键输入/输出**：noisy posterior、observed labels、stable indices；输出 TransitionArtifact。
- **可模块化部分**：完整 Dual-T estimator。
- **必须 Algorithm 化的部分**：warm-up snapshot 和 corrected classifier consumer。
- **当前实现/状态**：`DualTransitionEstimator`；**精确子组件**。
- **代码与测试位置**：`src/lnl_toolbox/noise/estimators.py`、`tests/test_transition_estimators.py`、`tests/test_plugins.py`。
- **当前缺失**：论文专属 preset、acceptance experiment 和正式参考结果复现。
- **可以声明**：Dual-T transition-estimation component 已实现、可保存，并可由通用 corrected-risk pipeline 消费。
- **禁止声明**：不得声称已完整复现 Dual-T 论文训练和实验结果。

### P17 — MC-LDCE

- **版本与原文**：SDM 2022，[arXiv v1](https://arxiv.org/abs/2203.10858)；本地 PDF 缺失。
- **研究问题**：把多分类 loss 的 label-dependent 部分转为 clean-centroid estimation。
- **核心机制**：loss decomposition、定义 multi-class centroid、从 noisy observations 无偏估计并重建风险。
- **关键输入/输出**：全局 feature/label statistics；输出 centroid 和 corrected global risk。
- **可模块化部分**：MC-LDCE StatisticEstimator、RiskCorrector。
- **必须 Algorithm 化的部分**：全数据统计采集与 classifier objective。
- **当前实现/状态**：只有 `StatisticResult[T]`；**接口已存在但无实现**。
- **代码与测试位置**：`src/lnl_toolbox/estimators/base.py`、`tests/test_estimators.py` 只验证容器。
- **当前缺失**：centroid payload、estimator、risk consumer。
- **可以声明**：已有不约束 payload 内部结构的 statistic container。
- **禁止声明**：不得称 StatisticResult 为 MC-LDCE。

### P18 — Importance Reweighting

- **版本与原文**：IEEE TPAMI 2016，[arXiv 原文](https://arxiv.org/abs/1411.7718)，本地 `04_statistic_estimation/importance_reweighting.pdf`。
- **研究问题**：用 density-ratio/importance weight 从 noisy distribution 恢复 clean risk。
- **核心机制**：二分类 asymmetric RCN 下由 noisy posterior 和已知 flip rates 计算 \(\beta(x,\tilde y)\)。
- **关键输入/输出**：\(P(\tilde Y\mid X)\)、observed binary target、\(\rho_+,\rho_-\)；输出 detached weights。
- **可模块化部分**：BinaryRCNImportanceWeightProvider。
- **必须 Algorithm 化的部分**：posterior 与 noise-rate estimation、完整实验流程。
- **当前实现/状态**：公式、rho 方向、q=0 和负权重校验均已有；**精确子组件**。
- **代码与测试位置**：`src/lnl_toolbox/treatments/weights.py`、`tests/test_importance_reweighting.py`。
- **当前缺失**：producer、plugin/YAML、监督训练构造入口。
- **可以声明**：paper-exact binary asymmetric-RCN importance-weight component。
- **禁止声明**：不得声称完整 Importance Reweighting 或多分类支持。

### P19 — CWD

- **版本与原文**：IEEE TPAMI 2022，[作者原文](https://gcatnjust.github.io/ChenGong/paper/gong_tpami22.pdf)，本地 `04_statistic_estimation/cwd.pdf`。
- **研究问题**：通过 class-wise denoising 降低 clean-centroid estimator variance。
- **核心机制**：构造正/负虚拟 auxiliary sets，分别处理 false positive/negative 并重建风险。
- **关键输入/输出**：class-wise data statistics 和 noise rates；输出 centroids/corrected risk。
- **可模块化部分**：CWD StatisticEstimator、RiskCorrector。
- **必须 Algorithm 化的部分**：全局统计采集和 objective integration。
- **当前实现/状态**：只有 statistic 容器；**接口已存在但无实现**。
- **代码与测试位置**：`src/lnl_toolbox/estimators/base.py`；无 CWD 测试。
- **当前缺失**：auxiliary sets、centroid formula、consumer。
- **可以声明**：StatisticResult 可承载未来 CWD payload。
- **禁止声明**：不得把 generic class statistics 称为 CWD。

### P20 — PCSE

- **版本与原文**：IEEE TPAMI 2024/正式卷期 2025，[作者论文页面](https://randydkx.github.io/)，本地 `04_statistic_estimation/pcse.pdf`。
- **研究问题**：从 noisy-class 分组恢复每类 mean、covariance 和 prior，用于训练后分类。
- **核心机制**：预训练 feature extractor 上估计 per-class statistics，构造 generative postprocessor。
- **关键输入/输出**：固定 feature、noisy class statistics；输出 clean per-class statistics/predictions。
- **可模块化部分**：PCSE StatisticEstimator、PostProcessor。
- **必须 Algorithm 化的部分**：预训练、指定 feature layer、统计恢复和 inference stage。
- **当前实现/状态**：只有 `StatisticResult[T]`；**接口已存在但无实现**。
- **代码与测试位置**：`src/lnl_toolbox/estimators/base.py`；无 PCSE 测试。
- **当前缺失**：具体 payload、estimator、postprocessor。
- **可以声明**：通用容器不会阻碍未来 PCSE statistics。
- **禁止声明**：不得把训练期 ReliabilityEstimator 称为 PCSE。

## 13. 必须实现为 Algorithm/Pipeline 的方法

### P21 — DLD

- **版本与原文**：CVPR 2025，[官方页面](https://openaccess.thecvf.com/content/CVPR2025/html/Hou_Directional_Label_Diffusion_Model_for_Learning_from_Noisy_Labels_CVPR_2025_paper.html)，本地 `05_others/Directional_Label_Diffusion_Model_for_Learning_from_Noisy_Labels.pdf`。
- **研究问题**：从生成式 label diffusion 视角恢复 robust labels。
- **核心机制**：directional/random diffusion 两条路径，并以 feature-based pre-correction 提供噪声知识。
- **关键输入/输出**：图像特征、预修正标签、diffusion state；输出生成的 label distribution。
- **可模块化部分**：pre-correction artifact、diffusion schedule/denoiser 内部组件。
- **必须 Algorithm 化的部分**：生成模型训练、采样、classifier integration。
- **当前实现/状态**：核心不可由现有小原语表达；**不适合模块化**。
- **代码与测试位置**：无 DLD 实现或测试。
- **当前缺失**：完整 generative pipeline。
- **可以声明**：未来可复用 stable index 和 artifact storage。
- **禁止声明**：不得把 noise generator、LabelRefiner 草案或普通 diffusion 名称称为 DLD。

### P22 — FINE / Active Forgetting

- **版本与原文**：CVPR 2026，[官方页面](https://openaccess.thecvf.com/CVPR2026)，本地 `05_others/Revisiting_Learning_with_Noisy_Labels_Active_Forgetting_and_Noise_Suppression.pdf`。
- **研究问题**：对 baseline 已识别的 noisy subset 主动遗忘并抑制噪声监督。
- **核心机制**：clean subset 保留 baseline objective；noisy subset 使用 active forgetting/negative learning 等专属目标。
- **关键输入/输出**：外部 clean/noisy partition；输出多分支 objective。
- **可模块化部分**：subset-specific regularizer/objective。
- **必须 Algorithm 化的部分**：partition consumer、阶段调度和多项 loss 聚合。
- **当前实现/状态**：ContributionResult 只是共享基础；**未实现**。
- **代码与测试位置**：`src/lnl_toolbox/treatments/base.py`、`src/lnl_toolbox/treatments/reduction.py` 不是 FINE。
- **当前缺失**：FINE objectives、partition lifecycle。
- **可以声明**：统一 treatment 可承载 hard subset contribution。
- **禁止声明**：不得把 Selector 或 reducer 称为 FINE。

### P23 — CA2C

- **版本与原文**：ICCV 2025，[官方页面](https://openaccess.thecvf.com/content/ICCV2025/html/Sheng_CA2C_A_Prior-Knowledge-Free_Approach_for_Robust_Label_Noise_Learning_via_ICCV_2025_paper.html)，本地 `05_others/CA2C_A_Prior-Knowledge-Free_Approach_for_Robust_Label_Noise_Learning.pdf`。
- **研究问题**：无需 noise-rate/threshold prior 的 asymmetric co-learning/co-training。
- **核心机制**：partial-label P-model 与 negative-learning N-model 使用不同范式并 cross-guide。
- **关键输入/输出**：双模型预测、candidate/negative labels；输出互相生成的监督和双模型状态。
- **可模块化部分**：confidence reweighting、candidate-label representation。
- **必须 Algorithm 化的部分**：异构双模型、asymmetric update 和 cross-guidance。
- **当前实现/状态**：核心是完整双网络范式；**不适合模块化**。
- **代码与测试位置**：无 CA2C 实现或测试。
- **当前缺失**：全部 Algorithm/Pipeline。
- **可以声明**：现有 Loss/Treatment 可作为未来内部原语。
- **禁止声明**：不得把 Co-teaching exchange 或双 Selector 称为 CA2C。

### P24 — DivideMix

- **版本与原文**：ICLR 2020，[OpenReview 页面](https://openreview.net/forum?id=qBbAV8Wg90K)，本地 `05_others/Learning with Noisy Labels as Semi-supervised Learning.pdf`。
- **研究问题**：把 noisy-label learning 转为 labeled/unlabeled 半监督学习。
- **核心机制**：全数据 loss min-max normalization、二 GMM clean probability、双网络 co-divide、co-refinement/co-guessing、MixMatch。
- **关键输入/输出**：每网络的 dataset losses；输出 clean probability、split、refined labels 和双模型更新。
- **可模块化部分**：GMM clean-probability ReliabilityEstimator。
- **必须 Algorithm 化的部分**：threshold split、双网络、SSL batches 和完整 epoch lifecycle。
- **当前实现/状态**：`DivideMixGMMCleanProbabilityEstimator` 和显式 selection adapter；**精确子组件**。
- **代码与测试位置**：`src/lnl_toolbox/estimators/dividemix_gmm.py`、`tests/test_dividemix_gmm.py`、`tests/test_reliability_selection_adapter.py`。
- **当前缺失**：最近多轮 loss 聚合、split、MixMatch、双网络 checkpoint。
- **可以声明**：paper-faithful DivideMix GMM clean-probability component。
- **禁止声明**：不得称 GMM+SmallLossSelector 为 DivideMix。

### P25 — L2RW

- **版本与原文**：ICML 2018，[PMLR 页面](https://proceedings.mlr.press/v80/ren18a.html)，本地 `05_others/Learning to Reweight Examples for Robust Deep Learning.pdf`。
- **研究问题**：用少量 clean balanced validation data 在线学习当前 batch 的权重。
- **核心机制**：以临时 epsilon 构造 virtual model update，再对 clean validation loss 求 meta-gradient 并归一化权重。
- **关键输入/输出**：noisy train batch、clean meta batch、model/optimizer state；输出当前 step 权重和真实更新。
- **可模块化部分**：meta weight result；但 producer 不是纯 detached function。
- **必须 Algorithm 化的部分**：virtual step、高阶梯度、clean batch 和 optimizer sequencing。
- **当前实现/状态**：泛型 WeightProvider 不足以忠实表达；**未实现**。
- **代码与测试位置**：无 L2RW；`src/lnl_toolbox/treatments/weights.py` 仅处理已计算权重。
- **当前缺失**：MetaUpdater/Pipeline 及 clean-validation ownership。
- **可以声明**：统一 reducer 可消费最终权重。
- **禁止声明**：不得把任意 continuous weights 或 Binary RCN provider 称为 L2RW。

### P26 — LEND

- **版本与原文**：Machine Learning 2022，[arXiv 原文](https://arxiv.org/abs/2206.13025)，本地 `05_others/Towards Harnessing Feature Embedding for Robust Learning with Noisy Labels.pdf`。
- **研究问题**：利用早期 feature embedding 的邻域结构稀释错误标签监督。
- **核心机制**：batch/feature KNN similarity、label propagation/dilution、用 diluted supervision 继续训练。
- **关键输入/输出**：features、indexed labels/graph；输出 diluted label state 和训练 objective。
- **可模块化部分**：feature-graph builder、reliability/label-refinement artifact。
- **必须 Algorithm 化的部分**：graph 构建时机、indexed label state 和迭代训练。
- **当前实现/状态**：只有可承载 scalar evidence 的合同；**未实现**。
- **代码与测试位置**：无 LEND 实现或测试。
- **当前缺失**：graph、dilution formula、state/checkpoint 和 pipeline。
- **可以声明**：stable index 与 ReliabilityResult 可供未来部分复用。
- **禁止声明**：不得把 feature similarity score 或普通 Selector 称为 LEND。

## 14. 推荐后续路线

建议停止无目标地增加容器，改为由真实消费者驱动原语建设：

1. **Forward/Backward loss correction 验收**：在现有 RiskCorrector 和训练消费链上补论文 preset、严格 resume 与 acceptance experiment。
2. **完整 Co-teaching**：建立第一个双模型 Algorithm，严格复用但不改写 legacy exchange。
3. **CDR paper preset/lifecycle**：在已有精确 update policy 上补完整运行定义。
4. **Importance Reweighting Pipeline**：补 noisy posterior 与 noise-rate producers。
5. **一个 statistic vertical slice**：MC-LDCE、CWD、PCSE 三选一，同时实现 producer 和唯一 consumer。
6. **DivideMix**：作为独立大型 Pipeline，不再用更多通用 Selector 冒充进度。

进入任何论文同名 method 前，先写清模型数量、状态 ownership、step/epoch 顺序、
checkpoint 内容和论文专属 acceptance tests。

## 15. 论文级错误声明黑名单

以下说法禁止出现在 README、plugin metadata、配置名或完成报告中：

- “SmallLossSelector implements Co-teaching/JoCoR/CNLCU/DSS。”
- “DivideMix GMM implements DivideMix。”
- “Anchor/Dual-T artifact implements Forward/Backward training。”
- “Generic WeightProvider implements MentorNet or L2RW。”
- “Binary RCN importance weights fully implement Importance Reweighting。”
- “StatisticResult implements MC-LDCE/CWD/PCSE。”
- “CDRUpdatePolicy fully implements Robust Early-Learning。”
- “IDN generator implements UPM/PDL/CAL。”
- “A plugin registration proves a component is connected to training。”
- “A type-compatible adapter proves the paper lifecycle is complete。”
- “A component test is an end-to-end paper reproduction。”

允许的声明必须带范围，例如“paper-exact binary asymmetric-RCN importance-weight
component”“DivideMix GMM clean-probability subcomponent”或
“Dual-T transition-estimation component”。

## 16. 代码、测试和配置位置索引

| 能力 | 生产代码 | 主要测试 | plugin/训练状态 |
|---|---|---|---|
| per-sample losses | `src/lnl_toolbox/losses/torch_losses.py` | `tests/test_losses.py` | plugin + training |
| Selector contracts | `src/lnl_toolbox/selectors/base.py` | `tests/test_selectors.py` | plugin + training |
| All/SmallLoss | `src/lnl_toolbox/selectors/basic.py` | `tests/test_selectors.py` | plugin + training |
| keep-rate schedule | `src/lnl_toolbox/selectors/schedules.py` | `tests/test_selectors.py` | selector config/resume |
| Contribution/reducer | `src/lnl_toolbox/treatments/base.py`, `src/lnl_toolbox/treatments/reduction.py` | `tests/test_treatments.py` | supervised consumer |
| Selector adapter | `src/lnl_toolbox/treatments/selector_adapter.py` | `tests/test_treatments.py` | supervised consumer |
| WeightProvider/Binary RCN | `src/lnl_toolbox/treatments/weights.py` | `tests/test_importance_reweighting.py` | 未接训练 |
| Reliability/Statistic contracts | `src/lnl_toolbox/estimators/base.py` | `tests/test_estimators.py` | 未接训练 |
| DivideMix GMM | `src/lnl_toolbox/estimators/dividemix_gmm.py` | `tests/test_dividemix_gmm.py` | 未接训练 |
| Reliability adapter | `src/lnl_toolbox/estimators/selection_adapter.py` | `tests/test_reliability_selection_adapter.py` | 组件链测试 |
| transition snapshot | `src/lnl_toolbox/training/snapshots.py` | `tests/test_transition_estimators.py` | 可独立调用 |
| Anchor/Dual-T | `src/lnl_toolbox/noise/estimators.py` | `tests/test_transition_estimators.py`, `tests/test_pipeline.py` | plugin + staged training |
| Forward/Backward RiskCorrector | `src/lnl_toolbox/algorithms/transition_risk.py` | `tests/test_transition_risk.py`, `tests/test_pipeline.py` | plugin + staged training |
| TransitionArtifact | `src/lnl_toolbox/noise/transition.py` | `tests/test_transition_estimators.py` | save/load |
| update policies | `src/lnl_toolbox/algorithms/update_policy.py` | `tests/test_update_policy.py` | plugin + training |
| CDR update | `src/lnl_toolbox/algorithms/cdr.py` | `tests/test_cdr.py` | training + checkpoint |
| supervised consumer | `src/lnl_toolbox/algorithms/supervised.py` | `tests/test_torch_training.py` | active |
| experiment builder | `src/lnl_toolbox/training/experiment.py` | `tests/test_noisy_ce_baseline.py` | active |
| checkpoint | `src/lnl_toolbox/training/checkpoint.py` | `tests/test_cdr.py`, `tests/test_noisy_ce_baseline.py` | Algorithm-owned |
| builtin catalog | `src/lnl_toolbox/plugins/builtin/catalog.py` | `tests/test_plugins.py` | active |
| Stateful contract | `src/lnl_toolbox/core/component.py` | `tests/test_core.py` | 无通用 ownership |

`StatisticResult` 当前没有具体 `StatisticEstimator` producer 或 consumer。
Weight 和 reliability 链路有跨组件测试但没有合法 YAML/训练构造入口。
TransitionEstimator 与 Forward/Backward RiskCorrector 已由 staged pipeline 消费，但仍不代表论文完整复现。

## 17. 原文版本与证据索引

| # | 简称 | 版本 | 原文来源 | 本地状态 |
|---:|---|---|---|---|
| 01 | UPM | AAAI 2021 | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17221) | PDF |
| 02 | CAL | CVPR 2021 | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_A_Second-Order_Approach_to_Learning_With_Instance-Dependent_Label_Noise_CVPR_2021_paper.html) | PDF |
| 03 | PDL | NeurIPS 2020 | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/5607fe8879e4fd269e88387e8cb30b7e-Abstract.html) | PDF |
| 04 | JoCoR | CVPR 2020 | [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.html) | PDF |
| 05 | DSS | CVPR 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.html) | PDF |
| 06 | CDR | ICLR 2021 | [OpenReview PDF](https://openreview.net/pdf?id=Eql5b1_hTE4) | 在线补充 |
| 07 | CNLCU | ICLR 2022 | [OpenReview](https://openreview.net/forum?id=xENf4QUL4LW) | PDF |
| 08 | MentorNet | ICML 2018 | [PMLR](https://proceedings.mlr.press/v80/jiang18c.html) | PDF |
| 09 | Co-teaching | NeurIPS 2018 | [NeurIPS](https://papers.nips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html) | PDF |
| 10 | Loss Correction | CVPR 2017 | [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html) | PDF |
| 11 | APL | ICML 2020 | [PMLR](https://proceedings.mlr.press/v119/ma20c.html) | PDF |
| 12 | GCE | NeurIPS 2018 | [NeurIPS](https://papers.nips.cc/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html) | PDF |
| 13 | VolMinNet | ICML 2021 | [PMLR](https://proceedings.mlr.press/v139/li21l.html) | PDF |
| 14 | Natarajan | NeurIPS 2013 | [NeurIPS](https://papers.nips.cc/paper/2013/hash/3871bd64012152bfb53fdf04b401193f-Abstract.html) | PDF |
| 15 | T-Revision | NeurIPS 2019 | [NeurIPS](https://papers.nips.cc/paper/2019/hash/9308b0d6e5898366a4a986bc33f3d3e7-Abstract.html) | PDF |
| 16 | Dual-T | NeurIPS 2020 | [NeurIPS](https://papers.nips.cc/paper/2020/hash/512c5cad6c37edb98ae91c8a76c3a291-Abstract.html) | PDF |
| 17 | MC-LDCE | SDM 2022 / arXiv v1 | [arXiv](https://arxiv.org/abs/2203.10858) | 在线补充 |
| 18 | Importance Reweighting | TPAMI 2016 | [arXiv](https://arxiv.org/abs/1411.7718) | PDF |
| 19 | CWD | TPAMI 2022 | [作者 PDF](https://gcatnjust.github.io/ChenGong/paper/gong_tpami22.pdf) | PDF |
| 20 | PCSE | TPAMI 2024/2025 | [作者页面](https://randydkx.github.io/) | PDF |
| 21 | DLD | CVPR 2025 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Hou_Directional_Label_Diffusion_Model_for_Learning_from_Noisy_Labels_CVPR_2025_paper.html) | PDF |
| 22 | FINE | CVPR 2026 | [CVF proceedings](https://openaccess.thecvf.com/CVPR2026) | PDF |
| 23 | CA2C | ICCV 2025 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Sheng_CA2C_A_Prior-Knowledge-Free_Approach_for_Robust_Label_Noise_Learning_via_ICCV_2025_paper.html) | PDF |
| 24 | DivideMix | ICLR 2020 | [OpenReview](https://openreview.net/forum?id=qBbAV8Wg90K) | PDF |
| 25 | L2RW | ICML 2018 | [PMLR](https://proceedings.mlr.press/v80/ren18a.html) | PDF |
| 26 | LEND | Machine Learning 2022 | [arXiv](https://arxiv.org/abs/2206.13025) | PDF |

维护者在新增论文 method 时，应先更新本文对应条目的“当前实现、缺失、可以声明、
禁止声明”，再更新公开 plugin/config 文档。若论文原文、官方代码与本 taxonomy 冲突，
以论文原文和可验证实现为准，并记录具体公式、Algorithm 或章节证据。
