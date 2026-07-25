# LNL 论文实现映射指南

> 归档位置与维护规则见本目录的 `README.md`。

本文按 `papers/manifest.json` 的编号顺序，把每篇论文转换为可执行的 toolbox
实现路线。这里出现的规划文件、函数和类不代表已经实现；它们用于确定未来代码
应放在哪里、按什么顺序调用，以及需要复用哪些已有能力。

## 使用规则

1. 按论文编号顺序维护，当前总数为 26 篇。
2. 每个条目必须区分：
   - `已有`：仓库当前已经存在；
   - `扩展`：在现有文件中增加能力；
   - `规划`：未来可能新增的文件、函数或类。
3. 后续论文与已有条目重叠时，必须合并：
   - 同一职责只保留一个文件和一个公共函数；
   - 在原条目上追加“使用论文”和参数差异；
   - 不得为每篇论文复制 Dataset、posterior 收集、checkpoint 或 evaluator；
   - 只有论文特有的数学、状态或 Pipeline 才使用论文名文件。
4. 文件夹优先复用现有的 `data/`、`noise/`、`algorithms/`、`training/`、
   `evaluation/`、`plugins/`、`configs/` 和 `tests/`，非必要不增加新文件夹。
5. 所有 `state[N]` 必须按 `(dataset, split, global_index)` 访问，不能使用
   batch 位置代替样本身份。
6. clean label 只能进入 evaluator，不能进入训练 Dataset、Loss、Selector、
   NoiseModel 或 Algorithm。
7. 证据标签：
   - `[论文]`：论文正文、公式或算法明确给出；
   - `[代码]`：官方实现明确采用；
   - `[推断]`：为适配当前 toolbox 所作的工程设计；
   - `[差异]`：论文与官方代码存在差异；
   - `[待核实]`：尚未通过代码或实验确认。

### 重叠能力主索引

后续条目引用下列职责时，必须扩展这里指定的唯一实现，不得另建同义协议：

| 共享职责 | 唯一位置 / 接口 | 当前状态 | 使用论文 |
|---|---|---|---|
| noisy posterior 快照 | `noise/estimators.py::PosteriorSnapshot` | 已有 | UPM、CAL、PDL、Loss Correction、T-Revision、Dual-T、Importance Reweighting |
| posterior / feature 收集 | `training/snapshots.py::collect_posterior_snapshot()` | 已有 posterior 收集；feature 收集仍规划 | UPM、CAL、PDL、Loss Correction、T-Revision、Dual-T、MC-LDCE、CWD、PCSE、DLD、LEND |
| 全局转移矩阵 artifact | `noise/transition.py::TransitionArtifact` | 已有 | CAL、Loss Correction、VolMinNet、T-Revision、Dual-T、MC-LDCE、PCSE |
| 实例转移查询 | `noise/transition.py::InstanceTransitionProvider` | 规划 | UPM、PDL |
| 选样结果 | `selectors/base.py::SelectionResult` | 已有 hard-mask 协议 | JoCoR、DSS、CNLCU、Co-teaching、FINE、DivideMix、LEND |
| 小损失排序与保留率 | `selectors/basic.py::SmallLossSelector`、`selectors/schedules.py` | 已有 fixed、constant、linear 保留率 | JoCoR、CNLCU、Co-teaching |
| 按样本历史状态 | `selectors/history.py` | 规划 | DSS、CNLCU、LEND、CA2C、DivideMix |
| 连续样本权重 | `treatments/weights.py::WeightResult` / `WeightProvider` | 已有基础协议和 Binary RCN provider；其他方法规划 | MentorNet、T-Revision、Importance Reweighting、L2RW、CA2C、DivideMix |
| soft target 结果 | `core/result.py::SoftTargetResult` | 规划 | UPM、DLD、CA2C、DivideMix、LEND |
| 特征快照 | `training/snapshots.py::FeatureSnapshot` | 规划 | MC-LDCE、CWD、PCSE、DLD、LEND |
| 特征邻域图 | `data/neighbors.py::NeighborGraphArtifact` | 规划 | DLD、LEND |
| 全局/分类统计量 | `noise/statistics.py::StatisticArtifact` | 规划 | MC-LDCE、CWD、PCSE |
| 半监督 batch | `data/semi_supervised.py::SemiSupervisedBatch` | 规划 | DivideMix |
| 转移矩阵风险校正 | `algorithms/transition_risk.py::RiskCorrector` | 规划 | Loss Correction、Learning with Noisy Labels |
| 可训练全局转移模型 | `noise/transition.py::TrainableTransitionModel` | 规划 | VolMinNet、T-Revision |
| 双网络 peer exchange | `algorithms/coteaching.py::peer_exchange()` | 扩展现有 NumPy helper | JoCoR、CNLCU、Co-teaching |
| backward 与参数更新 | `algorithms/update_policy.py::ParameterUpdatePolicy` | 已有；Standard 与 CDR 首批实现 | CDR；未来单模型参数级更新方法 |

“唯一位置”是 guideline 的合并目标，不表示规划项已经存在。若同事分支已提供等价
公共接口，应合入同事接口并回改本表，而不是并存两套。

当前生产 Runner 已接入逐样本 Loss 与 batch Selector；WeightProvider 和
TransitionEstimator 仍是可独立调用的旁路组件，不得写成已经接入训练主链。

## 当前进度

- 当前任务：逐篇建立论文到 toolbox 的实现映射。
- 当前分支：`loss`。
- 基线提交：`3bcaa64`。
- 已完成条目：26。
- 总条目：26。
- 进度：`26 / 26 = 100%`。
- 当前成熟度含义：条目完成不等于算法已经实现。
- 已检查：论文原文、论文目录、分类文档、当前源码结构，以及 UPM、CAL、
  PDL、JoCoR、DSS、CDR、CNLCU、MentorNet、Co-teaching、Loss Correction、
  APL、VolMinNet、T-Revision、PCSE、DLD、FINE、CA2C、DivideMix、L2RW
  官方代码；GCE、Learning with Noisy Labels、Dual-T、MC-LDCE、
  Importance Reweighting、CWD 与 LEND 未发现论文作者发布的官方实现。
- 已修改：`papers/implement/paper-implementation-guideline.md`（继续维护）。
- 已新增：`papers/implement/paper-implementation-guideline.md`。
- 本地 checkpoint commit：无。
- 阻塞项：UPM 官方仓库代码存在明显缺失；CAL 官方代码基于旧版 PyTorch，
  且数据路径携带 clean label；PDL 官方代码依赖顺序位置定位样本，且与论文
  在训练阶段和矩阵拟合细节上存在差异；DSS 是 CVPR 2026 新方法；CDR 官方
  代码与论文更新式不完全一致；MentorNet 官方代码基于 TensorFlow 1.8，均不能
  直接复制为 toolbox 实现；Co-teaching 官方代码基于 PyTorch 0.3；
  Loss Correction 官方代码基于旧 Keras；VolMinNet 与 T-Revision 作者代码
  对矩阵有效性和数值稳定性的处理不足；L2RW 依赖可信 clean validation，
  与仓库当前“clean label 只进 evaluator”的规则冲突，实施前必须先获得团队批准
  并建立显式 trusted-supervision 边界；其余复杂方法均需按独立 Pipeline 移植，
  不能直接复制作者的单脚本实现。
- 精确下一步：由团队审阅 26 篇映射及共享接口索引，确定实现优先级；任何算法
  实施前再把对应条目的规划路径与同事分支现有接口对照，合并同义职责。
- history cleanup：不需要。
- push readiness：未准备；当前只是完整 guideline，尚未进行实现审阅。

---

## 01. UPM：Tackling Instance-Dependent Label Noise via a Universal Probabilistic Model

### 论文信息

- 编号 / 文件：`01_instance_dependent/01_upm_aaai2021.pdf`
- 会议与年份：AAAI 2021
- 作者：Qizhou Wang, Bo Han, Tongliang Liu, Gang Niu, Jian Yang, Chen Gong
- 官方页面：<https://ojs.aaai.org/index.php/AAAI/article/view/17221>
- 官方代码：<https://github.com/QizhouWang/instance-dependent-label-noise>
- 当前成熟度：L2
- 核对状态：已阅读论文；已检查官方 `pretrain.py`、`main.py` 和
  `cloth1m.py`；尚未运行
- Toolbox 归属：`InstanceNoiseModel + PosteriorRefiner + Pipeline`
- 不是：普通硬标签 Loss、普通 Selector、全局 `[C,C]` TransitionEstimator

### 1. 论文实际做了什么

[论文] UPM 假设每个样本属于 confusing 或 unconfusing：

```text
unconfusing：noisy label 一定等于 true label
confusing：noisy label 在给定 x 后与 true label 独立
```

每个训练样本维护两个量：

```text
ψ_i = P(noisy label_i | x_i)     # 预训练 noisy classifier 给出
η_i = P(sample i is confusing)   # 主训练期间学习
```

目标分类器输出：

```text
h_i = softmax(model(x_i))        # [B,C]
```

论文 Eq. 8 的 latent clean-label posterior 为：

```text
q_i ∝ h_i ⊙ [(1-η_i) one_hot(noisy_label_i) + η_i ψ_i 1]
```

`q_i` 归一化为 `[B,C]` 后，作为 soft target 更新分类器；随后按论文
Eq. 11–12 更新当前 batch 对应的 `η_i` 并投影到 `[0,1]`。

### 2. 完整调用顺序

```text
NoiseManifest + Dataset
        ↓
阶段 A：用 noisy label 预训练 noisy classifier
        ↓
对所有训练样本收集 noisy posterior [N,C]
        ↓
按 noisy target 取 ψ[N]
        ↓
初始化 η[N] = eta_init
        ↓
阶段 B：每个 batch
        ├── 读取 input / noisy target / global index
        ├── 根据 h、ψ、η 计算 q[B,C]
        ├── 用 q 更新目标分类器
        └── 到达更新日程后更新 η[global index]
        ↓
clean validation / test
        ↓
checkpoint：模型、优化器、scheduler、ψ、η、index、阶段
```

### 3. 按顺序映射到文件和函数

| 顺序 | 状态 | 目录 / 文件 | 函数或类 | 输入 → 输出 | 实现要求与依据 |
|---:|---|---|---|---|---|
| 1 | 已有 | `data/torch_cifar.py` | `TorchCifarDataset` | CIFAR + indices → `input/target/index` | 复用稳定 global index；`target` 是当前训练标签。 |
| 2 | 已有 | `data/noisy_dataset.py` | `NoisyTargetDataset` | clean Dataset + manifest mapping → noisy training Dataset | 只替换 target，不暴露 clean target。 |
| 3 | 规划，共享 | `training/snapshots.py` | `pretrain_noisy_classifier()` | noisy DataLoader + model config → noisy classifier checkpoint | [论文][代码] 先按 noisy label 训练普通分类器；后续 Anchor、Dual-T 等若需要相同 warm-up，应复用此函数。 |
| 4 | 规划，共享 | `training/snapshots.py` | `collect_posterior_snapshot()` | model + DataLoader → `PosteriorSnapshot[N,C]` | `inference_mode` 下按 global index 收集，不能依赖 shuffle 后的数组顺序。后续 TransitionEstimator 复用。 |
| 5 | 规划，UPM | `noise/upm.py` | `UPMNoiseState.from_snapshot()` | Snapshot + noisy targets → `psi[N]`、`eta[N]`、indices | `psi_i` 取 posterior 中 noisy target 对应概率；`eta` 初始化为 `eta_init`，论文示例为 0.01。 |
| 6 | 规划，UPM | `noise/upm.py` | `UPMNoiseState.lookup()` | global indices `[B]` → `psi[B]`、`eta[B]` | 所有逐样本状态通过 global index 访问。 |
| 7 | 规划，UPM | `algorithms/upm.py` | `estimate_clean_posterior()` | logits `[B,C]`、noisy target `[B]`、psi `[B]`、eta `[B]` → q `[B,C]` | 实现 Eq. 8；q 有限、非负、逐行和为 1。构造 q 时模型 posterior 必须 detach。 |
| 8 | 规划，UPM | `algorithms/upm.py` | `upm_soft_target_objective()` | logits `[B,C]` + detached q `[B,C]` → scalar | `-mean(sum(q * log_softmax(logits)))`；这是 UPM 私有 objective，不修改现有硬标签 Loss `[B]` 协议。 |
| 9 | 规划，UPM | `algorithms/upm.py` | `update_confusion_probabilities_()` | q、psi、eta、noisy target、global index → 更新 eta `[N]` | 按 Eq. 11 梯度上升、Eq. 12 clamp；使用 `no_grad`，不能让 eta 更新进入模型 autograd。 |
| 10 | 规划，UPM | `algorithms/upm.py` | `UPMAlgorithm.step()` | Batch + RunState → StepResult | 顺序固定为 posterior、模型更新、条件式 eta 更新；拥有 UPM 私有状态。 |
| 11 | 规划，UPM | `training/upm_pipeline.py` | `run_upm_experiment()` | resolved config → run directory | 编排预训练、posterior 收集、主训练、clean evaluation、checkpoint 和 resume；不能塞入普通 supervised runner 的 if/else。 |
| 12 | 扩展，共享 | `training/checkpoint.py` | `save_checkpoint()` / `load_checkpoint()` | Algorithm 私有状态 ↔ checkpoint | 未来扩展通用私有状态字段；保存 psi、eta、global indices、当前阶段和 noisy-classifier identity。后续有状态算法复用。 |
| 13 | 扩展，共享 | `evaluation/metrics.py` | `confusion_probability_summary()` | eta `[N]` → mean、quantiles、histogram | 训练诊断不需要 clean label；合成噪声下的识别指标放在 evaluator 单独计算。 |
| 14 | 扩展，共享 | `plugins/builtin/catalog.py` | `build_builtin_pipeline()` | `{name: upm, ...}` → UPM Pipeline | 注册为 `kind="pipeline"`；以后完整 JoCoR、DivideMix、T-Revision 等也复用这个插件类型。 |
| 15 | 规划，UPM | `configs/algorithm/upm.yaml` | — | 算法配置 | 保存 `eta_init`、`eta_lr`、eta 更新起始 epoch/间隔、posterior epsilon。 |
| 16 | 规划，UPM | `configs/experiment/cifar10_upm_smoke.yaml` | — | 实验配置 | 小样本两阶段 smoke；共享现有数据、模型、optimizer、noise manifest 配置。 |
| 17 | 规划，UPM | `tests/test_upm.py` | UPM 单元/集成测试 | fixture → assertions | 集中测试论文数学、逐样本状态、交替更新和恢复，避免把 UPM 测试分散到 Loss/Selector 文件。 |

### 4. 规划接口

以下接口名称是 guideline 约定，尚未创建：

```python
@dataclass
class UPMNoiseState:
    global_indices: Tensor       # int64[N]
    noisy_label_probability: Tensor  # psi, float[N]
    confusion_probability: Tensor    # eta, float[N]

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PosteriorSnapshot,
        eta_init: float = 0.01,
    ) -> "UPMNoiseState": ...

    def lookup(self, global_indices: Tensor) -> tuple[Tensor, Tensor]: ...

    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

```python
def estimate_clean_posterior(
    logits: Tensor,             # [B,C]
    noisy_targets: Tensor,      # [B]
    noisy_label_probability: Tensor,  # psi [B]
    confusion_probability: Tensor,    # eta [B]
    eps: float = 1e-8,
) -> Tensor:                    # detached q [B,C]
    ...
```

```python
def update_confusion_probabilities_(
    state: UPMNoiseState,
    global_indices: Tensor,     # [B]
    posterior: Tensor,          # detached q [B,C]
    noisy_targets: Tensor,      # [B]
    learning_rate: float,
    eps: float = 1e-4,
) -> None:
    ...
```

```python
def run_upm_experiment(
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    ...
```

### 5. Pipeline 伪代码

```text
function run_upm_experiment(config):
    train_loader, clean_val_loader, clean_test_loader =
        build_existing_noisy_cifar_loaders(config)

    if no reusable noisy-classifier artifact:
        noisy_model = pretrain_noisy_classifier(train_loader, config.pretrain)
        save noisy_model and its config/hash

    snapshot = collect_posterior_snapshot(noisy_model, train_loader)
    upm_state = UPMNoiseState.from_snapshot(snapshot, eta_init)

    target_model, optimizer, scheduler = initialize_main_training(config)
    algorithm = UPMAlgorithm(target_model, optimizer, upm_state, eta_schedule)

    for epoch:
        for batch in train_loader:
            logits = target_model(batch.input)

            psi, eta = upm_state.lookup(batch.index)
            q = estimate_clean_posterior(
                logits.detach(), batch.target, psi, eta
            )

            objective = upm_soft_target_objective(logits, q.detach())
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

            if eta_schedule.should_update(epoch):
                with no_grad:
                    update_confusion_probabilities_(
                        upm_state, batch.index, q, batch.target, eta_lr
                    )

        evaluate on clean validation
        save model/optimizer/scheduler/RunState/psi/eta/indices/stage

    load best checkpoint
    evaluate on clean test
```

### 6. 配置草案

```yaml
pipeline:
  name: upm
  eta_init: 0.01
  eta_lr: 0.7
  eta_update:
    start_epoch: 35
    interval: 5
  posterior_eps: 1.0e-8

pretrain:
  loss: {name: ce}
  reuse_artifact: null
```

[论文] CIFAR 实验训练 160 epochs，分类器初始学习率为 0.05、每 40 epochs
除以 10；`eta` 从第 35 epoch 起每 5 epochs 更新。CIFAR-10 使用
`eta_lr=0.7`。

[论文][代码] Clothing1M 使用 ResNet-50、batch size 32、15 epochs、
`eta_lr=0.05`，从第二个 epoch 开始更新 eta；模型学习率每 5 epochs
衰减。

这些数据集特定默认值应放在实验 YAML，而不是写死在 `UPMAlgorithm`。

### 7. Checkpoint 必需状态

```text
model
optimizer
scheduler
RunState / completed_epoch
pipeline_stage
psi[N]
eta[N]
global_indices[N]
dataset + split
noisy classifier config/hash
noise manifest identity
UPM hyperparameters
best validation metric
```

恢复时必须验证 psi、eta 与当前 dataset/split/global indices 完全对齐；不得
重新计算后悄悄覆盖旧状态。

### 8. 最小测试

1. `estimate_clean_posterior()` 与 Eq. 8 手算一致。
2. `eta=0` 时 q 只保留 noisy target。
3. `eta` 增大时模型 posterior 的影响增大。
4. q 有限、非负、逐行和为 1，并与模型图 detach。
5. eta 更新后严格位于 `[0,1]`。
6. shuffle 前后，相同 global index 取得相同 psi/eta。
7. eta schedule 在开始 epoch 前不更新，并按 interval 更新。
8. UPM soft-target objective 能产生有限梯度并更新模型。
9. checkpoint roundtrip 后 psi、eta、阶段、epoch 和 step 一致。
10. 训练 batch 不包含 clean target；合成噪声真值只由 evaluator 使用。
11. CPU 单步与 CUDA 小样本两阶段 smoke 均可完成。

### 9. 论文与官方代码核对

- `[论文][代码]` 先训练 noisy classifier，再计算每个样本 noisy label 对应的
  posterior 概率。
- `[论文][代码]` q 使用当前目标模型的 detached posterior 构造。
- `[论文][代码]` eta 是逐样本状态，并按 Dataset index 更新。
- `[论文][代码]` eta 更新后 clamp 到 `[0,1]`。
- `[论文]` CIFAR 的 eta 更新从第 35 epoch 开始、每 5 epochs 一次。
- `[代码]` 公开仓库主要提供 Clothing1M 路径，从第二个 epoch开始逐 batch
  更新 eta。
- `[代码]` Clothing1M 使用类别权重缓解不平衡；这是数据集策略，不是 UPM
  概率模型的通用部分。
- `[差异]` 官方仓库的 `main.py` 中 `args` 初始化缺失，且 `eta_hist` 初始化
  被注释但后续仍被使用，因此不能直接视为可运行的完整复现。
- `[推断]` Toolbox 应把 posterior 收集做成共享能力，因为 Anchor、Dual-T
  和其他估计器也可能使用相同的 warm-up 输出。
- `[推断]` UPM 应作为独立 Pipeline；当前普通监督 runner 不负责两阶段训练
  和逐样本 eta 状态。

### 10. 当前未实现

- `training/snapshots.py`
- `noise/upm.py`
- `algorithms/upm.py`
- `training/upm_pipeline.py`
- pipeline plugin kind
- UPM 配置和测试
- Clothing1M Dataset
- UPM checkpoint 私有状态
- 论文结果复现

因此当前 toolbox 不能宣称支持 UPM；本条目只是未来实现指南。

---

## 02. CAL：A Second-Order Approach to Learning with Instance-Dependent Label Noise

### 论文信息

- 编号 / 文件：`01_instance_dependent/02_cal_cvpr2021.pdf`
- 会议与年份：CVPR 2021
- 作者：Zhaowei Zhu, Tongliang Liu, Yang Liu
- 官方页面：<https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_A_Second-Order_Approach_to_Learning_With_Instance-Dependent_Label_Noise_CVPR_2021_paper.html>
- 官方代码：<https://github.com/UCSC-REAL/CAL>
- 当前成熟度：L2
- 核对状态：已阅读论文；已检查官方运行脚本、实验编排、Dataset、模型更新和
  loss 实现；尚未运行
- Toolbox 归属：`StatisticEstimator + RiskCorrector + Pipeline`
- 不是：普通硬标签 Loss、普通 Selector、只输出全局 `[C,C]` 的
  TransitionEstimator

### 1. 论文实际做了什么

[论文] CAL 研究实例依赖噪声。论文以 Bayes 最优标签 `Y*` 作为潜在 clean
类别，并定义：

```text
T_ij(x) = P(noisy label=j | Y*=i, X=x)
```

Peer Loss / CORES² 主要使用一阶统计。CAL 进一步估计噪声转移量与分类损失之间
的二阶协方差，使 IDN 风险转化为更容易处理的 class-dependent noise 风险。
多分类目标的核心形式为：

```text
CAL risk
  = Peer/CORES² risk
  - sum_i sum_j P(Y*=i)
      Cov(T_ij(X), loss(f(X), j) | Y*=i)
```

真实 `Y*` 和真实 `T_ij(x)` 不可见，因此论文先构造代理数据集 `D_hat`：

```text
CORES² warm-up
    → 计算每个样本的 adjusted loss
    → 低于 L_min：保留 noisy label
    → 高于 L_max：改成模型 argmax label
    → 位于中间：暂时丢弃
    → 得到 proxy Bayes label 与 retained mask
```

随后用代理标签构造逐样本指示量：

```text
T_hat_ij(x_n)
  = 1{proxy_label_n=i and noisy_label_n=j}
```

再按 proxy class 对该指示量和各类别 loss 做中心化，估计协方差校正项。

### 2. 完整调用顺序

```text
NoiseManifest + NoisyTargetDataset
        → 阶段 A：用 noisy label 训练 CORES² warm-up 模型
        → 使用共享 posterior collector 按 global index 收集预测
        → 计算 adjusted loss 与 predicted label
        → 构造 CALProxyArtifact
           ├── retained + 保留 noisy label
           ├── retained + 使用 predicted label
           └── dropped
        → 阶段 B：初始化 CAL 训练模型
        → 每个 batch
           ├── 读取 input / noisy target / global index
           ├── 按 global index 查询 proxy label 与 retained mask
           ├── 计算 noisy CE 与 confidence regularizer
           ├── 估计 centered transition-loss covariance
           └── 组合 CAL risk、反向传播并更新模型
        → clean validation / test
        → checkpoint：模型、优化器、scheduler、proxy artifact identity、
          阶段和协方差统计
```

训练 batch 不携带 clean target。代理标签是由 warm-up 模型和 noisy label
构造的训练状态，不是真实 clean label。

### 3. 与已有条目的重叠合并

| 共享职责 | 合并后的唯一位置 | CAL 如何复用 |
|---|---|---|
| noisy Dataset 与稳定 index | `data/noisy_dataset.py` / `NoisyTargetDataset` | 不创建 CAL Dataset；继续返回 `input/target/index` |
| noisy warm-up | `training/snapshots.py` / `pretrain_noisy_classifier()` | 与 UPM 共用训练入口，通过配置选择 CORES² objective |
| 全量预测收集 | `training/snapshots.py` / `collect_posterior_snapshot()` | 与 UPM、Anchor、未来 Dual-T 共用；CAL 从 snapshot 取得 argmax 和概率 |
| checkpoint 私有状态 | `training/checkpoint.py` / 通用 algorithm state | 扩展同一协议，不创建 CAL checkpoint 文件 |
| clean evaluation | `evaluation/metrics.py` 与现有 evaluator | 不创建论文专用 accuracy evaluator |
| pipeline 注册 | `plugins/builtin/catalog.py` / `build_builtin_pipeline()` | 与 UPM 共用 `kind="pipeline"` |

如果后续论文也需要 CORES² adjusted loss、代理标签或同类二阶统计，应在上述
公共函数增加使用论文和参数差异，不得复制实现。

### 4. 按顺序映射到文件和函数

| 顺序 | 状态 | 目录 / 文件 | 函数或类 | 输入 → 输出 | 实现要求与依据 |
|---:|---|---|---|---|---|
| 1 | 已有 | `data/noisy_dataset.py` | `NoisyTargetDataset` | Dataset + manifest → noisy Dataset | 训练 target 只允许 noisy label；保留稳定 global index。 |
| 2 | 规划，共享 | `training/snapshots.py` | `pretrain_noisy_classifier()` | loader + warm-up config → model artifact | 与 UPM 合并为一个函数；CAL 通过 objective 配置启用 CORES²。 |
| 3 | 规划，共享 | `training/snapshots.py` | `collect_posterior_snapshot()` | model + loader → `PosteriorSnapshot[N,C]` | 与 UPM 合并；按 global index 收集，不依赖 loader 顺序。 |
| 4 | 规划，CAL | `algorithms/cal.py` | `cores2_adjusted_losses()` | logits `[B,C]` + noisy targets `[B]` + noisy prior `[C]` → `[B]` | [论文][代码] 计算 sample sieve 使用的 confidence-regularized loss；返回逐样本值。 |
| 5 | 规划，CAL | `noise/cal.py` | `CALProxyArtifact` | indices + proxy targets + sample status → versioned artifact | 保存 retained/relabelled/dropped 状态、阈值、warm-up hash 和 mapping hash；不得保存 clean target。 |
| 6 | 规划，CAL | `noise/cal.py` | `build_cal_proxy_artifact()` | Snapshot + adjusted loss + noisy targets → artifact | 按 `L_min/L_max` 执行保留、重标和丢弃；所有结果绑定 global index。 |
| 7 | 规划，CAL | `noise/cal.py` | `CALProxyArtifact.lookup()` | global indices `[B]` → proxy targets、mask、status | batch shuffle 前后必须取得同一状态。 |
| 8 | 规划，CAL | `algorithms/cal.py` | `cal_transition_indicators()` | proxy target、noisy target、retained mask → class-pair indicators | 实现 `1{Y_hat*=i, noisy=j}`；优先稀疏计算，避免长期保存 `[C,C,N]`。 |
| 9 | 规划，CAL | `algorithms/cal.py` | `cal_covariance_correction()` | all-class losses `[B,C]` + proxy/noisy labels + class prior → scalar + statistics | 实现论文 Eq. 3 的二阶协方差校正；只使用 proxy label。 |
| 10 | 规划，CAL | `algorithms/cal.py` | `cal_objective()` | logits、noisy targets、proxy state、统计量 → scalar | 组合 noisy CE、confidence regularizer 与 covariance correction；它需要逐样本状态，因此不注册为普通 Loss。 |
| 11 | 规划，CAL | `algorithms/cal.py` | `CALAlgorithm.step()` | Batch + RunState → StepResult | 查询 artifact、计算 CAL risk、更新模型并输出分项指标。 |
| 12 | 规划，CAL | `training/cal_pipeline.py` | `run_cal_experiment()` | resolved config → run directory | 编排 warm-up、代理数据构造、CAL 重训练、clean evaluation、checkpoint 和 resume。 |
| 13 | 扩展，共享 | `training/checkpoint.py` | `save_checkpoint()` / `load_checkpoint()` | pipeline 私有状态 ↔ checkpoint | 与 UPM 合并同一扩展点；保存 proxy hash、阶段、统计量和 warm-up identity。 |
| 14 | 扩展，共享 | `evaluation/metrics.py` | `proxy_sample_summary()` | proxy status → counts/rates | 不用 clean label 的训练诊断；合成数据的 proxy correctness 只能由 evaluator 计算。 |
| 15 | 扩展，共享 | `plugins/builtin/catalog.py` | `build_builtin_pipeline()` | `{name: cal, ...}` → CAL Pipeline | 复用 UPM 首次规划的 pipeline plugin kind。 |
| 16 | 规划，CAL | `configs/algorithm/cal.yaml` | — | CAL 算法配置 | 保存 warm-up、sieve、confidence regularizer 和 covariance 参数。 |
| 17 | 规划，CAL | `configs/experiment/cifar10_cal_smoke.yaml` | — | 两阶段 smoke 配置 | 复用现有数据、模型、optimizer、manifest 和 artifact 目录协议。 |
| 18 | 规划，CAL | `tests/test_cal.py` | CAL 单元/集成测试 | fixtures → assertions | 集中测试代理标签、二阶校正、全局索引、梯度、checkpoint 和 smoke。 |

### 5. 规划接口

以下名称是 guideline 约定，尚未创建：

```python
@dataclass
class CALProxyArtifact:
    global_indices: Tensor     # int64[N]
    proxy_targets: Tensor      # int64[N]
    sample_status: Tensor      # keep / relabel / drop
    warmup_artifact_hash: str
    lower_threshold: float
    upper_threshold: float

    def lookup(
        self,
        global_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "CALProxyArtifact": ...
```

```python
def cores2_adjusted_losses(
    logits: Tensor,             # [B,C]
    noisy_targets: Tensor,      # [B]
    noisy_prior: Tensor,        # [C]
    confidence_weight: float,
    eps: float = 1e-5,
) -> Tensor:                    # [B]
    ...
```

```python
def cal_covariance_correction(
    all_class_losses: Tensor,   # [B,C]
    proxy_targets: Tensor,      # [B]
    noisy_targets: Tensor,      # [B]
    retained_mask: Tensor,      # bool[B]
    proxy_class_prior: Tensor,  # [C]
    reference_loss_means: Tensor,  # [C,C]
) -> tuple[Tensor, Tensor]:
    """Return scalar correction and detached [C,C] loss statistics."""
    ...
```

```python
def run_cal_experiment(
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    ...
```

### 6. Pipeline 伪代码

```text
function run_cal_experiment(config):
    train_loader, clean_val_loader, clean_test_loader =
        build_existing_noisy_cifar_loaders(config)

    if no reusable CAL proxy artifact:
        warmup_model = pretrain_noisy_classifier(
            train_loader,
            objective=CORES2(config.warmup)
        )
        snapshot = collect_posterior_snapshot(warmup_model, train_loader)

        adjusted_loss = compute CORES2 adjusted loss for every global index
        proxy_artifact = build_cal_proxy_artifact(
            snapshot,
            adjusted_loss,
            noisy targets,
            lower_threshold,
            upper_threshold
        )
        save artifact with model/config/mapping hashes

    model, optimizer, scheduler = initialize_CAL_training(config)
    algorithm = CALAlgorithm(model, optimizer, proxy_artifact, config.cal)

    for epoch:
        for batch in train_loader:
            proxy_target, retained_mask, status =
                proxy_artifact.lookup(batch.index)

            logits = model(batch.input)
            all_class_loss = loss against every possible class
            base = CE(logits, batch.target)
            confidence = confidence_regularizer(logits, noisy_prior)
            correction, batch_statistics = cal_covariance_correction(
                all_class_loss,
                proxy_target,
                batch.target,
                retained_mask,
                proxy_class_prior,
                reference_loss_means
            )

            objective = base - confidence_weight * confidence - correction
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

        update reference loss means using detached epoch statistics
        evaluate on clean validation
        save model/optimizer/scheduler/stage/proxy hash/statistics

    load best checkpoint
    evaluate on clean test
```

### 7. 配置草案

```yaml
pipeline:
  name: cal

warmup:
  epochs: 65
  objective:
    name: cores2
    confidence_weight: 2.0

sieve:
  lower_threshold: -8.0
  upper_threshold: -8.0

cal:
  epochs: 100
  confidence_weight: 1.0
  probability_eps: 1.0e-5
  reuse_proxy_artifact: null
```

[论文][代码] CIFAR-10 使用 ResNet-34、batch size 128、SGD momentum 0.9、
weight decay 0.0005；sample sieve 在 65 epochs 后构造代理数据，CAL 阶段训练
100 epochs。论文实验中 CIFAR-10 的 warm-up / CAL confidence 系数分别为
2 / 1，CIFAR-100 为 10 / 10。

[论文][代码] CIFAR 实验令 `L_min=L_max=-8`，因此代理构造通常成为“保留
noisy label”或“改成预测标签”的二分过程；实现仍保留两个阈值，不把该实验
设置写死在算法中。

### 8. Checkpoint 必需状态

```text
model
optimizer
scheduler
RunState / completed_epoch / global_step
pipeline_stage
warm-up model config/hash
CALProxyArtifact path/hash
dataset + split + global-index mapping hash
noise manifest identity
proxy class prior
reference_loss_means[C,C]
CAL hyperparameters
best validation metric
```

恢复阶段 B 时必须加载同一个 proxy artifact，且验证 dataset、manifest、
global indices 和 warm-up identity。不得重新构造代理标签后静默覆盖原状态。

### 9. 最小测试

1. `cores2_adjusted_losses()` 与手算结果一致。
2. 低于下阈值的样本保留 noisy label。
3. 高于上阈值的样本使用模型 argmax label。
4. 两阈值之间的样本被标记为 dropped。
5. `L_min=L_max` 时除边界等值外得到确定的二分状态。
6. 输入顺序改变后，相同 global index 的 proxy label 和 status 不变。
7. artifact 对重复 index、非法标签、非有限 score 和 hash 篡改明确失败。
8. transition indicator 与 `1{proxy=i, noisy=j}` 手算一致。
9. covariance 为零时 CAL objective 退化为配置的 Peer/CORES² risk。
10. 非零 covariance fixture 的校正项与 Eq. 3 手算一致。
11. 极端 logits 下 objective 和梯度有限。
12. dropped 样本不会进入 proxy class 统计。
13. 训练 batch 不包含 clean target；proxy correctness 只在 evaluator 中计算。
14. checkpoint roundtrip 后阶段、artifact hash、loss means、epoch 和 step 一致。
15. UPM 与 CAL 调用同一个 posterior collector，不存在重复实现。
16. CPU 单步与 CUDA 两阶段小样本 smoke 均可完成。

### 10. 论文与官方代码核对

- `[论文][代码]` 方法分为构造 `D_hat` 和训练 CAL 两个阶段。
- `[论文][代码]` 代理标签由 noisy label、sample sieve 和模型预测产生。
- `[论文][代码]` 二阶项使用代理类别、noisy 类别和对所有候选类别计算的
  loss 构造。
- `[代码]` 官方入口通过 `crossentropy` 与 `crossentropy_CAL` 手工切换两个
  阶段，并用 `sieve_65_*.pt` 在阶段间传递代理标签和权重。
- `[代码]` 官方实现构造 `[C,C,N]` 的 `T_mat` 指示张量，并在训练前按代理
  类别中心化；toolbox 可使用等价的稀疏/按 batch 计算避免大张量常驻。
- `[代码]` 官方实现逐 epoch 维护 `[C,C]` 的 `loss_mean_all`，下一 epoch
  用它中心化各类别 loss；该状态必须进入 checkpoint。
- `[差异]` 官方 CIFAR Dataset 同时返回 `true_label`，训练函数也接收该参数；
  真实矩阵诊断路径默认关闭，CAL 的可用估计来自 distilled/proxy label。
  Toolbox 不得把 clean label 放入训练 batch，即使只是为了调试。
- `[差异]` 官方代码面向 Python 3.6、PyTorch 1.4，且实验参数和阶段切换写在
  脚本中；toolbox 应使用 YAML、artifact identity 和可恢复 Pipeline。
- `[推断]` CAL 不是当前 `loss(logits, targets) -> [B]` 协议，因为它还依赖
  proxy label、retained mask、类别先验和跨 batch/epoch 统计。
- `[推断]` CAL 也不是普通全局 TransitionEstimator；它消费的是逐样本
  class-pair 指示量与 loss 的协方差，而不是只输出一个 `[C,C]` 矩阵。

### 11. 当前未实现

- `training/snapshots.py` 中的共享 warm-up 与 posterior 收集
- `noise/cal.py`
- `algorithms/cal.py`
- `training/cal_pipeline.py`
- pipeline plugin kind
- CAL 配置和测试
- CAL proxy artifact 与 checkpoint 私有状态
- Clothing1M 数据路径
- 论文结果复现

因此当前 toolbox 不能宣称支持 CAL；本条目只是未来实现指南。

---

## 03. PDL：Part-dependent Label Noise: Towards Instance-dependent Label Noise

### 论文信息

- 编号 / 文件：`01_instance_dependent/03_pdl_neurips2020.pdf`
- 会议与年份：NeurIPS 2020
- 作者：Xiaobo Xia, Tongliang Liu, Bo Han, Nannan Wang, Mingming Gong,
  Haifeng Liu, Gang Niu, Dacheng Tao, Masashi Sugiyama
- 官方页面：<https://proceedings.neurips.cc/paper/2020/hash/5607fe8879e4fd269e88387e8cb30b7e-Abstract.html>
- 官方代码：<https://github.com/xiaoboxia/Part-dependent-label-noise>
- 当前成熟度：L2
- 核对状态：已阅读论文；已检查官方 `main.py`、`models.py`、`tools.py` 和
  `data.py`；尚未运行
- Toolbox 归属：`InstanceTransitionEstimator + InstanceTransitionProvider + Pipeline`
- 不是：普通全局 `[C,C]` TransitionEstimator、普通 Loss、普通 Selector

### 1. 论文实际做了什么

[论文] 完全自由的实例依赖转移矩阵 `T(x)` 难以从 noisy data 中识别。PDL
引入一个中间假设：标注者的错误依赖实例中的语义部件，而不是任意依赖整个实例。

首先从 noisy 模型取得深层表示 `z(x)`，学习 `r` 个部件和每个样本的非负组合
系数：

```text
min sum_i ||z(x_i) - W h(x_i)||²
s.t. h(x_i) >= 0, ||h(x_i)||₁ = 1                    (Eq. 1)
```

每个部件对应一个类别转移矩阵 `P_j[C,C]`。实例转移矩阵由同一组系数组合：

```text
T(x) ≈ sum_j h_j(x) P_j                               (Eq. 2)
```

其中统一方向仍然是：

```text
T(x)[i,j] = P(noisy label=j | clean label=i, X=x)
p_noisy(x) = p_clean(x) @ T(x)
```

对 clean class `i` 的 anchor point `x`，有：

```text
T_i·(x) = P(noisy label=· | X=x)                      (Eq. 3)
```

因此每类至少选择 `r` 个 anchor candidates，用它们的 noisy posterior 和
`h(x)` 拟合所有部件矩阵：

```text
min sum_i sum_l ||T_i·(x_i^l) - sum_j h_j(x_i^l)P_j[i,:]||²
s.t. P_j >= 0, sum_k P_j[i,k] = 1                    (Eq. 4)
```

得到 `h[N,r]` 和 `P[r,C,C]` 后，可以按需生成完整的 `T(x)[B,C,C]`。论文再
将它交给已有 Forward 或 Importance Reweighting 方法；可选的 `ΔT` revision
来自 T-Revision，不是 PDL 特有公式。

### 2. 论文 Algorithm 2：实验用 IDN 生成器

[论文] PDL 还给出了合成 IDN benchmark 的生成方式。对每个 clean 样本：

```text
q_i ~ TruncatedNormal(mean=tau, std=0.1, range=[0,1])
score = x_i @ W_{clean_label_i}
score[clean_label_i] = -infinity
off_diagonal = q_i * softmax(score)
probability[clean_label_i] = 1 - q_i
noisy_label_i ~ Categorical(probability)
```

这只用于离线生成实验标签。clean label 可以进入 generator 来创建 manifest，
但生成结束后不能进入训练 batch。

当前 `generate_instance_dependent()` 使用的是 score-weighted ambiguity 方案，
不等于 Algorithm 2，因此不得静默改写它。未来应在同一个
`noise/generators.py` 中增加名称明确的 PDL benchmark 模式。

### 3. 完整调用顺序

```text
NoiseManifest + NoisyTargetDataset
        → 从 noisy training subset 划出 noisy validation subset
        → 阶段 A：noisy train + noisy validation 训练/选择 warm-up 模型
        → 使用共享 snapshot collector
           ├── PosteriorSnapshot[N,C]
           └── FeatureSnapshot[N,D]
        → 阶段 B：最小化 Eq. 1
           ├── feature parts W[D,r]
           └── coefficients h[N,r]
        → 每类选择至少 r 个 anchor candidates
        → 用 anchor posterior 得到 T_i·(x_anchor)
        → 阶段 C：最小化 Eq. 4，得到 P[r,C,C]
        → PartTransitionArtifact
           └── transitions_for(global_indices) → T(x)[B,C,C]
        → 阶段 D：共享 Forward 或 Reweight risk correction
        → 可选：未来共享 T-Revision 的 ΔT
        → clean validation / test
        → checkpoint：所有阶段、snapshot identity、parts、coefficients、
          part matrices、corrector 和 optimizer 状态
```

论文中的 noisy validation 只使用 noisy label 做模型选择。Toolbox 可以继续保留
独立 clean validation/test evaluator，但 clean target 不得进入上述阶段。

### 4. 与已有条目的重叠合并

| 共享职责 | 合并后的唯一位置 | PDL 如何复用 |
|---|---|---|
| noisy Dataset 与稳定 index | `data/noisy_dataset.py` / `NoisyTargetDataset` | 不创建 PDL Dataset；训练、noisy validation 均返回 `input/target/index` |
| 数据划分 | `data/torch_cifar.py` / `stratified_split()` | 对 noisy targets 划分 noisy validation；不使用 clean target 决定 PDL 内部划分 |
| warm-up | `training/snapshots.py` / `pretrain_noisy_classifier()` | 与 UPM、CAL 共用 |
| posterior 收集 | `training/snapshots.py` / `collect_posterior_snapshot()` | 与 UPM、CAL、Anchor 共用 |
| feature 收集 | `training/snapshots.py` / `collect_feature_snapshot()` | 与未来 LEND、PCSE 等 feature-based 方法共用 |
| anchor 候选 | `noise/estimators.py` / `select_anchor_candidates()` | 扩展现有 Anchor 的确定性选择逻辑，不复制 PDL anchor helper |
| transition 方向与验证 | `noise/transition.py` | 继续使用 `clean_to_noisy_row`；增加实例级 provider，不改变全局 artifact |
| Forward / Reweight 消费 | `algorithms/transition_risk.py` | PDL 只提供 `T(x)`；第 10、18 篇继续核对并合并风险公式 |
| T revision | 未来 T-Revision 共享组件 | PDL 不创建私有 `PDLRevision` |
| checkpoint / evaluator / pipeline registry | 已有公共扩展点 | 与 UPM、CAL 共用，不复制论文专用基础设施 |

原先仅负责 posterior 的虚拟收集文件统一扩展为
`training/snapshots.py`，因为它现在同时收集 posterior 与 features。

### 5. 按顺序映射到文件和函数

| 顺序 | 状态 | 目录 / 文件 | 函数或类 | 输入 → 输出 | 实现要求与依据 |
|---:|---|---|---|---|---|
| 1 | 已有 | `data/noisy_dataset.py` | `NoisyTargetDataset` | Dataset + manifest → noisy Dataset | 复用 stable global index；不暴露 clean target。 |
| 2 | 已有，复用 | `data/torch_cifar.py` | `stratified_split()` | noisy targets + size + seed → index splits | PDL 的内部 validation 按 noisy label 或稳定随机 index 划分。 |
| 3 | 扩展，实验 | `noise/generators.py` | `generate_pdl_idn()` | inputs + clean targets + tau + seed → `NoiseManifest` | 实现 Algorithm 2；与现有 score-weighted IDN 并存，不改变原函数语义。 |
| 4 | 规划，共享 | `training/snapshots.py` | `pretrain_noisy_classifier()` | noisy train/validation + config → warm-up artifact | 与 UPM、CAL 合并；noisy validation 用于模型选择。 |
| 5 | 规划，共享 | `training/snapshots.py` | `collect_posterior_snapshot()` | model + loader → `PosteriorSnapshot[N,C]` | 使用现有 snapshot contract；按 global index 收集。 |
| 6 | 规划，共享 | `training/snapshots.py` | `FeatureSnapshot` | features + indices + model identity → immutable snapshot | `[N,D]`、有限值、唯一 index、版本/hash；不得包含 clean target。 |
| 7 | 规划，共享 | `training/snapshots.py` | `collect_feature_snapshot()` | model + feature hook + loader → `FeatureSnapshot[N,D]` | inference mode 下收集指定层输出；与 posterior 的 index 集合严格对齐。 |
| 8 | 扩展，共享 | `noise/estimators.py` | `select_anchor_candidates()` | PosteriorSnapshot + candidates per class → indices `[C,K]` | 每类至少 `r` 个唯一候选；score 并列按最小 global index；现有 Anchor 可复用第一个候选。 |
| 9 | 扩展，共享 | `noise/transition.py` | `InstanceTransitionProvider` | global indices `[B]` → `T(x)[B,C,C]` | 与全局 `TransitionProvider` 分开，防止 `[C,C]` 和 `[B,C,C]` 混用。 |
| 10 | 规划，PDL | `noise/pdl.py` | `fit_part_representation()` | FeatureSnapshot + num parts → parts `[D,r]`、coefficients `[N,r]` | 实现 Eq. 1；每行 coefficient 非负且和为 1。 |
| 11 | 规划，PDL | `noise/pdl.py` | `fit_part_transition_matrices()` | anchor rows + anchor coefficients → `P[r,C,C]` | 实现 Eq. 4；每个 `P_j` 非负且逐行和为 1；不足 `r` 个候选或系数矩阵秩不足时失败。 |
| 12 | 规划，PDL | `noise/pdl.py` | `PartTransitionArtifact` | parts + coefficients + P + indices + hashes → compact artifact | 不长期物化 `[N,C,C]`；保存 snapshot、anchor、配置和 mapping hash。 |
| 13 | 规划，PDL | `noise/pdl.py` | `PartTransitionArtifact.transitions_for()` | global indices `[B]` → `[B,C,C]` | 以 `einsum` 实现 Eq. 2；输出有限、非负、逐行和为 1。 |
| 14 | 规划，共享 | `algorithms/transition_risk.py` | `forward_corrected_losses()` | logits、targets、`T(x)[B,C,C]` → `[B]` | 共享 transition consumer；第 10 篇进一步核对 Forward 原式。 |
| 15 | 规划，共享 | `algorithms/transition_risk.py` | `importance_reweighted_losses()` | logits、targets、`T(x)[B,C,C]` → `[B]` | 共享逐样本权重 consumer；第 18 篇进一步核对权重假设。 |
| 16 | 规划，PDL | `algorithms/pdl.py` | `PDLAlgorithm.step()` | Batch + instance provider + risk corrector → StepResult | 按 batch index 查询 `T(x)`，再调用共享 corrector 和 optimizer。 |
| 17 | 规划，PDL | `training/pdl_pipeline.py` | `run_pdl_experiment()` | resolved config → run directory | 编排 warm-up、snapshot、Eq. 1、anchor、Eq. 4、corrected training 和 resume。 |
| 18 | 扩展，共享 | `training/checkpoint.py` | 通用 pipeline/algorithm state | PDL state ↔ checkpoint | 保存当前阶段、artifact hash、factorization/optimizer 状态和 corrector identity。 |
| 19 | 扩展，共享 | `evaluation/metrics.py` | `transition_summary()` | provider + indices → diagonal/entropy/row statistics | 无需 clean label 的诊断；真 `T(x)` 对比仅限合成实验 evaluator。 |
| 20 | 扩展，共享 | `plugins/builtin/catalog.py` | `build_builtin_pipeline()` / estimator builder | configs → PDL components | 注册 `instance_transition_estimator/pdl` 与 `pipeline/pdl`，不伪装成全局 estimator。 |
| 21 | 规划，PDL | `configs/noise/pdl.yaml` | — | estimator config | num parts、factorization、anchor 与 matrix-fit 参数。 |
| 22 | 规划，PDL | `configs/experiment/cifar10_pdl_smoke.yaml` | — | 多阶段 smoke config | 复用现有 CIFAR、manifest、模型、optimizer 和 artifact 目录。 |
| 23 | 规划，PDL | `tests/test_pdl.py` | PDL 单元/集成测试 | fixtures → assertions | 集中测试 Eq. 1–4、provider、generator、校正调用、checkpoint 和 smoke。 |

### 6. 规划接口

```python
@dataclass(frozen=True)
class FeatureSnapshot:
    features: np.ndarray          # float[N,D]
    global_indices: np.ndarray    # int64[N]
    dataset: str
    split: str
    model_artifact_hash: str

    @property
    def snapshot_hash(self) -> str: ...
```

```python
class InstanceTransitionProvider(Protocol):
    @property
    def num_classes(self) -> int: ...

    def transitions_for(
        self,
        global_indices: Tensor,    # [B]
        *,
        device: Any = None,
        dtype: Any = None,
    ) -> Tensor:                   # [B,C,C]
        ...
```

```python
@dataclass(frozen=True)
class PartTransitionArtifact:
    feature_parts: np.ndarray       # [D,r]
    coefficients: np.ndarray        # [N,r]
    part_matrices: np.ndarray       # [r,C,C]
    global_indices: np.ndarray      # [N]
    anchor_global_indices: np.ndarray  # [C,K]
    source_feature_hash: str
    source_posterior_hash: str

    def transitions_for(
        self,
        global_indices: Tensor,
        *,
        device: Any = None,
        dtype: Any = None,
    ) -> Tensor: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "PartTransitionArtifact": ...
```

```python
def fit_part_representation(
    snapshot: FeatureSnapshot,
    num_parts: int,
    *,
    max_iterations: int,
    tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return feature_parts[D,r] and simplex coefficients[N,r]."""
    ...
```

```python
def fit_part_transition_matrices(
    anchor_transition_rows: Tensor,  # [C,K,C]
    anchor_coefficients: Tensor,     # [C,K,r]
    *,
    max_iterations: int,
    tolerance: float,
) -> Tensor:                         # [r,C,C]
    ...
```

### 7. Pipeline 伪代码

```text
function run_pdl_experiment(config):
    noisy_train, clean_val, clean_test =
        build_existing_noisy_cifar_loaders(config)

    pdl_train_indices, noisy_val_indices =
        stratified_split(noisy targets, noisy_validation_size, seed)
    noisy_train_loader = subset noisy_train by pdl_train_indices
    noisy_val_loader = subset noisy_train by noisy_val_indices

    warmup = pretrain_noisy_classifier(
        noisy_train_loader,
        noisy_validation=noisy_val_loader,
        config=config.warmup
    )

    union_loader = stable union(noisy_train_loader, noisy_val_loader)
    posterior = collect_posterior_snapshot(warmup, union_loader)
    features = collect_feature_snapshot(warmup, union_loader, feature_layer)
    assert exact dataset/split/global-index alignment

    feature_parts, coefficients =
        fit_part_representation(features, num_parts)

    anchors = select_anchor_candidates(
        posterior,
        candidates_per_class >= num_parts,
        strategy=config.anchor.strategy
    )
    anchor_rows = posterior probabilities at each class anchor
    anchor_coefficients = coefficients at anchor global indices

    part_matrices = fit_part_transition_matrices(
        anchor_rows,
        anchor_coefficients
    )
    artifact = PartTransitionArtifact(
        feature_parts,
        coefficients,
        part_matrices,
        global_indices,
        anchors,
        snapshot hashes
    )
    validate and save artifact

    model, optimizer, scheduler = initialize_corrected_training(config)
    corrector = build shared Forward or Reweight risk corrector

    for epoch:
        for batch in noisy_train_loader:
            logits = model(batch.input)
            transition = artifact.transitions_for(batch.index)
            per_sample_loss = corrector(logits, batch.target, transition)
            objective = per_sample_loss.mean()
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

        evaluate on clean validation
        save model/optimizer/scheduler/stage/artifact identity

    load best checkpoint
    evaluate on clean test
```

### 8. 配置草案

```yaml
pipeline:
  name: pdl

warmup:
  loss: {name: ce}
  noisy_validation_size: 5000
  feature_layer: penultimate

instance_transition:
  name: pdl
  num_parts: 20
  factorization:
    max_iterations: 100
    tolerance: 1.0e-5
  anchors:
    candidates_per_class: 20
    strategy: topk
  matrix_fit:
    max_iterations: 1500
    tolerance: 1.0e-5

risk_corrector:
  name: forward

revision:
  enabled: false
```

[论文] CIFAR-10 使用 ResNet-34、batch size 128、SGD momentum 0.9、
weight decay `1e-4`、初始学习率 `1e-2`，在第 40、80 epoch 除以 10，初始化
阶段共 100 epochs；不使用数据增强。训练集的 10% 作为 noisy validation。

[代码] 公开 CIFAR 脚本将 `num_parts=20`、factorization iterations 设为 10，
并将 warm-up / corrected training / revision 分别设为 5 / 50 / 50 epochs。
这些是代码仓库的运行设置，不应覆盖论文设置；两套参数都只能放在 YAML。

### 9. Checkpoint 必需状态

```text
model
optimizer
scheduler
RunState / completed_epoch / global_step
pipeline_stage
noise manifest identity
noisy train / noisy validation global indices
warm-up model config/hash
PosteriorSnapshot hash
FeatureSnapshot hash
feature_parts[D,r]
coefficients[N,r] + global indices
anchor global indices[C,K]
part_matrices[r,C,C]
PartTransitionArtifact hash
risk corrector config
optional shared T-Revision state
best validation metric
```

如果允许在 Eq. 1 或 Eq. 4 的长迭代过程中恢复，还必须保存对应优化器、迭代数和
随机状态。恢复 corrected-training 阶段时不得重新拟合 artifact 后静默替换。

### 10. 最小测试

1. Algorithm 2 generator 在相同 seed 下完全一致。
2. generator 的对角概率为 `1-q_i`，非对角和为 `q_i`。
3. generator clean label 只存在于 manifest 创建和 evaluator，不进入训练 batch。
4. `FeatureSnapshot` 对非法 shape、NaN、重复 index 和空 identity 失败。
5. posterior 与 feature snapshot 的 index 集合不一致时失败。
6. Eq. 1 fixture 中 coefficients 非负、逐行和为 1，并降低重构误差。
7. 输入顺序改变后，以 global index 查询的 coefficient 不变。
8. 每类 anchor 数量不足、anchor 重复或 coefficient design rank 不足时失败。
9. anchor score 并列时按最小 global index 确定顺序。
10. Eq. 4 的小型可识别 fixture 能恢复预设 part matrices。
11. 每个 part matrix 有限、非负、逐行和为 1。
12. Eq. 2 与手工加权结果一致。
13. `transitions_for()` 输出严格为 `[B,C,C]`，且每行和为 1。
14. compact artifact 与显式物化全部 `T(x)` 的结果一致。
15. artifact 保存/加载一致，矩阵、系数或 metadata 篡改后 hash 校验失败。
16. 全局 `[C,C]` provider 与实例 `[B,C,C]` provider 不可互换。
17. Forward/Reweight consumer 能接收 PDL provider 并返回 `[B]`。
18. DataLoader shuffle 前后，相同 global index 得到相同 `T(x)`。
19. checkpoint roundtrip 后阶段、indices、artifact hash、epoch 和 step 一致。
20. CPU 单步和 CUDA 小样本多阶段 smoke 可完成。

### 11. 论文与官方代码核对

- `[论文][代码]` 先训练 noisy classifier，再提取 deep representation 和 noisy
  posterior。
- `[论文][代码]` 使用非负、单位 `l1` 的逐样本 coefficients 组合部件。
- `[论文][代码]` 从高 noisy-posterior 样本构造 anchor candidates。
- `[论文][代码]` 用 anchor transition rows 和 coefficients 通过平方误差拟合
  part matrices。
- `[代码]` 官方实现用类似 NMF 的乘法更新处理 train + noisy validation
  representations，并把 coefficients 存为 `[N,r]` 数组。
- `[代码]` anchor 实现使用 97% 到 99% 的多个 percentile 阈值构造候选，
  并用 MSE 优化部件矩阵。
- `[代码]` corrected training 根据
  `idx = batch_index * batch_size + position` 查询 coefficient，因此强制
  `shuffle=False`。Toolbox 必须改为 global-index lookup，不能复制该假设。
- `[差异]` 官方代码分别为 train 和 noisy validation 拟合
  `basis_matrix_group`；论文 Eq. 2–4 描述的是一组共享 part matrices。
  Toolbox 默认忠于论文，使用一个共享 artifact，并把代码行为记录为可复现实验
  差异而非默认接口。
- `[差异]` 官方仓库主要展示 importance-reweighting 风格的 corrected loss
  和后续 revision；论文同时报告 PTD-F、PTD-R、PTD-F-V、PTD-R-V。
- `[差异]` 论文和代码的 CIFAR epoch 数不同；不得把作者脚本的覆盖值误写成
  论文默认值。
- `[代码]` 官方实现会截断小于 `1e-6` 的矩阵项并重新归一化。Toolbox 应在
  优化步骤中显式执行 simplex projection；持久化 artifact 校验失败时不得隐式
  修复或归一化。
- `[推断]` `NoiseManifest.per_sample_transition[N,C]` 只表示实际 clean class
  对应的一行，不能充当 PDL 的完整 `T(x)[N,C,C]`。
- `[推断]` 紧凑保存 `h[N,r] + P[r,C,C]` 比保存全部 `[N,C,C]` 更符合论文
  结构，也更节省空间。

### 12. 当前未实现

- `training/snapshots.py`
- `FeatureSnapshot` 与 feature collector
- 通用 `InstanceTransitionProvider`
- `select_anchor_candidates()`
- `noise/pdl.py`
- `algorithms/transition_risk.py`
- `algorithms/pdl.py`
- `training/pdl_pipeline.py`
- PDL benchmark generator mode
- PDL 配置、artifact、checkpoint 私有状态和测试
- shared T-Revision
- 论文结果复现

因此当前 toolbox 不能宣称支持 PDL；本条目只是未来实现指南。

---

## 04. JoCoR：Combating Noisy Labels by Agreement

### 论文信息

- 编号 / 文件：`02_sample_selection/04_jocor_cvpr2020.pdf`
- 会议与年份：CVPR 2020
- 作者：Hongxin Wei, Lei Feng, Xiangyu Chen, Bo An
- 论文页面：<https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.html>
- 官方代码：<https://github.com/hongxin001/JoCoR>
- 当前成熟度：L2
- 核对状态：已阅读论文；已检查官方 `algorithm/jocor.py` 和
  `algorithm/loss.py`；尚未运行
- Toolbox 归属：`双网络 Algorithm + 共享小损失 Selector + Pipeline`
- 不是：可直接注册为 `loss(logits, targets) -> [B]` 的普通单网络 Loss

### 1. 论文实际做了什么

[论文] JoCoR 同时初始化两个不同的分类网络。对样本 `i`，两个网络共同产生一个
逐样本联合分数：

```text
l_sup(i) = CE(logits_1[i], target[i]) + CE(logits_2[i], target[i])

l_con(i) = KL(p_1[i] || p_2[i]) + KL(p_2[i] || p_1[i])

l_joint(i) = (1 - lambda) * l_sup(i) + lambda * l_con(i)
```

其中 `p_1`、`p_2` 是两个网络的 softmax 输出。论文用 Jensen-Shannon divergence
说明 agreement 思想，实际公式和官方代码使用双向 KL。

[论文] 每个 batch 只按 `l_joint[B]` 选择共同的小损失集合：

```text
keep_count >= R(epoch) * batch_size
selected = lowest_joint_loss_samples(keep_count)
objective = mean(l_joint[selected])
```

两个网络使用同一个 `selected` 集合、同一个联合目标并同时更新。它不是
Co-teaching 的“网络 A 选样给网络 B、网络 B 选样给网络 A”。

[论文] remember rate 为：

```text
R(epoch) = 1 - min(epoch / Tk * noise_rate, noise_rate)
```

### 2. 完整调用顺序

```text
NoisyTargetDataset
  -> batch(input, noisy target, global index)
  -> model_1 + model_2
  -> base Loss 分别输出 loss_1[B]、loss_2[B]
  -> symmetric_kl_per_sample() 输出 agreement[B]
  -> jocor_joint_scores() 输出 joint[B]
  -> SmallLossSelector.select(joint, global_index, keep_rate)
  -> 同一个 SelectionResult 供两个网络使用
  -> mean(joint[selected])
  -> backward + 更新两个网络
  -> clean evaluator 仅计算模型质量和选样质量
```

### 3. 与已有条目的重叠合并

- `[已有]` `losses/torch_losses.py` 继续提供逐样本 CE 等基础 Loss；JoCoR 不复制
  CE。
- `[已有]` `algorithms/coteaching.py` 的 `remember_rate()` 和小损失排序逻辑，
  后续应迁入或包装为共享的 `RememberRateSchedule`、`SmallLossSelector`。
- `[规划/共享]` 第 7 篇 CNLCU 和第 9 篇 Co-teaching 共用双网络容器、remember
  rate、SelectionResult、稳定排序和 peer-update 工具。
- `[论文特有]` 只有 joint score、agreement 和“同集合联合更新”留在 JoCoR
  Algorithm。
- 不为 JoCoR 新建 Dataset、checkpoint 格式、evaluator 或另一套噪声生成器。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] src/lnl_toolbox/core/result.py`
   - `SelectionResult`
   - 保存 batch positions、global indices、scores、keep rate 和统计量。
2. `[规划/共享] src/lnl_toolbox/selectors/small_loss.py`
   - `RememberRateSchedule`
   - `SmallLossSelector.select(scores, global_indices, keep_rate)`
   - 并列分数按最小 global index 决定，保证输入重排后结果稳定。
3. `[规划] src/lnl_toolbox/algorithms/jocor.py`
   - `symmetric_kl_per_sample(logits_1, logits_2)`
   - `jocor_joint_scores(loss_1, loss_2, logits_1, logits_2, lambda_)`
   - `JoCoRAlgorithm.step(batch, context)`
4. `[规划] src/lnl_toolbox/training/jocor_pipeline.py`
   - 构造两个模型、optimizer、selector、loss、evaluator 和 checkpoint。
5. `[扩展/高冲突] src/lnl_toolbox/plugins/builtin/catalog.py`
   - 注册 `algorithm/jocor` 和共享 selector；由集成人统一修改。
6. `[规划] configs/algorithm/jocor.yaml`
   - 只保存算法参数，不写本机路径。
7. `[规划] tests/test_jocor.py`
   - 数学、双网络更新、选择协议和恢复测试。

### 5. 规划接口

```python
def symmetric_kl_per_sample(
    logits_1: Tensor,  # [B,C]
    logits_2: Tensor,  # [B,C]
) -> Tensor:           # [B]
    ...


class JoCoRAlgorithm:
    def step(self, batch: Mapping[str, Tensor], context: ExperimentContext) -> StepResult:
        # batch 只允许 input、target、index
        ...
```

协议要求：

- 基础 Loss 仍严格返回 `[B]`。
- `symmetric_kl_per_sample()` 必须返回 `[B]`，不能先聚合。
- Selector 只接收 joint score 和 global index，不接收 clean target。
- `SelectionResult` 是一次 step 的不可变结果；两个网络共享它。
- optimizer 可实现为一个含两组参数的 optimizer，或两个同步 step 的 optimizer；
  checkpoint 必须明确保存实际形式。

### 6. Pipeline 伪代码

```python
for batch in train_loader:
    logits_1 = model_1(batch["input"])
    logits_2 = model_2(batch["input"])

    loss_1 = loss_fn(logits_1, batch["target"])
    loss_2 = loss_fn(logits_2, batch["target"])
    agreement = symmetric_kl_per_sample(logits_1, logits_2)
    joint = (1 - lambda_) * (loss_1 + loss_2) + lambda_ * agreement

    keep_rate = remember_schedule(epoch)
    selection = selector.select(joint.detach(), batch["index"], keep_rate)
    objective = joint[selection.batch_positions].mean()

    zero_all_gradients()
    objective.backward()
    step_all_optimizers()
```

### 7. 配置草案

```yaml
algorithm:
  name: jocor
  lambda: 0.9
  remember:
    noise_rate: 0.4
    gradual_epochs: 10

loss:
  name: ce

models:
  - {name: preact_resnet18}
  - {name: preact_resnet18}
```

`lambda`、网络结构、训练轮数和学习率由实验配置给出；不得把论文某个数据集的
搜索结果写成全局默认值。

### 8. Checkpoint 必需状态

- 两个 model state；
- 一个或两个 optimizer state；
- scheduler state；
- epoch、global step、seed 和 RNG state；
- loss、`lambda`、remember schedule 配置；
- noise manifest identity；
- 当前 best metric；
- 双网络各自 evaluator 统计。

JoCoR 没有跨 batch 的 selector 私有状态；SelectionResult 不需要持久化。

### 9. 最小测试

1. 监督项等于两个逐样本 CE 之和。
2. 两个 logits 相同时双向 KL 为零。
3. 交换两个网络后 joint score 不变。
4. 所有输出严格为 `[B]`，极端 logits 的前向和梯度有限。
5. joint score 并列时按 global index 稳定选择。
6. 两个网络使用完全相同的 SelectionResult。
7. 一步训练后两个网络参数都发生变化。
8. 测试明确拒绝 Co-teaching 式交叉选择语义。
9. remember rate 与论文公式一致，边界在 `[1-noise_rate, 1]`。
10. checkpoint roundtrip 后两个模型、optimizer、epoch 和 step 一致。
11. clean label 不进入 Algorithm 或 Selector。
12. CPU 单步与 CUDA smoke 可完成。

### 10. 论文与官方代码核对

- `[论文][代码]` 联合分数由两份 CE 和双向 KL 构成。
- `[论文][代码]` 两个网络共用一个小联合损失集合并联合更新。
- `[论文][代码]` 不使用 disagreement 过滤，也不交叉交换两个集合。
- `[代码]` 官方实现把两个网络参数交给同一个 optimizer。
- `[代码]` 官方代码用 `argsort` 和
  `int((1-forget_rate)*batch_size)` 计算保留数。
- `[代码]` clean/noisy truth 只用于 `pure_ratio` 日志；toolbox 必须把它限制在
  evaluator。
- `[差异]` 官方代码支持
  `linspace(0, forget_rate ** exponent, gradual_epochs)`；论文 Algorithm 1 是
  线性增长到 `noise_rate`。toolbox 默认忠于论文，`exponent` 只能作为显式
  复现实验选项。

### 11. 当前未实现

- 共享 `SelectionResult` 和 `SmallLossSelector`
- `JoCoRAlgorithm`
- JoCoR pipeline、配置、checkpoint 适配和测试
- 论文结果复现

因此当前 toolbox 不能宣称支持 JoCoR。

---

## 05. DSS：Debiased Sample Selection for Learning with Noisy Labels

### 论文信息

- 编号 / 文件：`02_sample_selection/05_dss_cvpr2026.pdf`
- 会议与年份：CVPR 2026
- 作者：Weiran Pan, Wei Wei, Wenfeng Xie
- 论文页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.html>
- 官方代码：<https://github.com/Aliinton/DSS>
- 当前成熟度：L2
- 核对状态：已阅读论文；已检查官方 `train.py`、
  `algorithm/dynamic_sample_selection.py` 和 `model/SampleSelector.py`；尚未运行
- Toolbox 归属：`有状态 Selector + Marginal Adjuster + Candidate-Class Risk`
- 不是：单一新 Loss，也不是必须绑定双网络的完整论文 Pipeline

### 1. 论文实际做了什么

[论文] DSS 由三个可组合部分构成：

```text
BASE：预测类别等于 noisy target 才选中
MDA：用动态类别边际分布校正 posterior，再做 BASE
CCS：排除“持续上升”的候选真实类别，避免把它们作为负类压制
```

BASE 的样本集合为：

```text
C = {i | argmax_c p(c | x_i) == noisy_target_i}
```

MDA 维护类别边际的指数移动平均：

```text
marginal <- momentum * marginal
          + (1 - momentum) * mean_batch(p(c | x))

debiased(c | x) =
    [p(c | x) / marginal[c]]
    / sum_k [p(k | x) / marginal[k]]
```

然后用 `argmax(debiased) == noisy_target` 选样。

[论文] CCS 为每个 `(sample, class)` 保存置信度轨迹，对每个非 noisy-target 类别
做 Mann-Kendall 上升趋势检验。显著上升的类别进入候选真实类别集合 `I_i`，训练时
从 softmax 分母排除：

```text
masked_CE_i =
  -log exp(logit[i, noisy_target])
       / sum_{c not in I_i} exp(logit[i,c])
```

noisy target 永远不能被排除。CCS 不直接重标注样本。

### 2. 完整调用顺序

```text
batch(input, noisy target, global index)
  -> model posterior [B,C]
  -> MarginalDistributionAdjuster.update()
  -> debiased posterior [B,C]
  -> PredictionMatchSelector
  -> sample SelectionResult
  -> CandidateClassState 按 global index 更新轨迹
  -> candidate exclusion mask [B,C]
  -> candidate_masked_ce()[B]
  -> 只聚合 sample mask 中的逐样本风险
  -> optimizer
```

### 3. 与已有条目的重叠合并

- 复用第 4 篇的 `SelectionResult`，不另造 DSS 专属选样返回类型。
- MDA 是独立的 posterior 校准组件；后续需要类别边际校正的方法复用它。
- CCS 的 `[B,C]` 类别排除 mask 与 `[B]` 样本选择 mask 是两种不同协议，不能
  混成一个布尔数组。
- `candidate_masked_ce()` 是额外输入的风险函数，不注册成当前普通
  `loss(logits, targets)`；否则会隐藏必要的 candidate mask。
- 论文核心 DSS 是单网络；官方代码中 DSS+ 的双网络交叉选择和一致性学习属于
  可选 Pipeline，不进入基础 DSS 组件。
- 不复制 Dataset；全部跨 epoch 状态都按 global index 查询。

### 4. 按顺序映射到文件和函数

1. `[规划/共享] src/lnl_toolbox/selectors/marginal.py`
   - `MarginalDistributionAdjuster.update(probabilities)`
   - `MarginalDistributionAdjuster.adjust(probabilities)`
2. `[规划] src/lnl_toolbox/selectors/dss.py`
   - `PredictionMatchSelector.select(probabilities, targets, indices)`
   - `mann_kendall_upward_score(history)`
   - `CandidateClassState.update(indices, probabilities, targets)`
   - `DSSSelector`
3. `[规划/共享] src/lnl_toolbox/algorithms/masked_risk.py`
   - `candidate_masked_ce(logits, targets, excluded_classes) -> [B]`
4. `[规划] src/lnl_toolbox/algorithms/dss.py`
   - 组合 sample selection、candidate mask 和 optimizer step。
5. `[规划] src/lnl_toolbox/training/dss_pipeline.py`
   - warm-up、每 epoch 状态更新、评测、恢复。
6. `[扩展/高冲突] src/lnl_toolbox/plugins/builtin/catalog.py`
   - 注册 MDA、DSS selector 和 DSS pipeline；由集成人修改。
7. `[规划] configs/algorithm/dss.yaml`
8. `[规划] tests/test_dss.py`

为避免文件碎片化，实现时可把 `marginal.py` 合并进 `selectors/dss.py`；只有出现
第二个消费者后再拆出共享文件。

### 5. 规划接口

```python
@dataclass(frozen=True)
class CandidateClassMask:
    global_indices: Tensor  # [B]
    excluded: Tensor        # bool [B,C]


def candidate_masked_ce(
    logits: Tensor,                  # [B,C]
    targets: Tensor,                 # [B]
    candidate_mask: CandidateClassMask,
) -> Tensor:                         # [B]
    ...
```

严格约束：

- noisy target 对应位置必须始终为 `False`。
- 每行至少保留 noisy target，不能产生空分母。
- MDA 的 marginal 必须有限、严格为正并归一化。
- `CandidateClassState` 只接收 posterior、noisy target、global index 和 epoch。
- clean target 和 flip mask 只允许 evaluator 使用。

### 6. Pipeline 伪代码

```python
for epoch in range(max_epochs):
    for batch in train_loader:
        logits = model(batch["input"])
        probs = softmax(logits).detach()

        marginal.update(probs)
        adjusted = marginal.adjust(probs)
        selection = match_selector.select(
            adjusted, batch["target"], batch["index"]
        )
        candidates = candidate_state.lookup(batch["index"])
        per_sample = candidate_masked_ce(
            logits, batch["target"], candidates
        )
        objective = per_sample[selection.batch_positions].mean()
        update_model(objective)

    if epoch >= warmup_epochs:
        candidate_state.update_from_epoch_snapshot(...)
```

论文 Algorithm 1 的“何时更新 C 和 I”必须在实现中固定为明确的 epoch 边界，避免
同一 epoch 内一半样本使用旧状态、一半使用新状态。

### 7. 配置草案

```yaml
algorithm:
  name: dss
  warmup_epochs: 10
  marginal:
    momentum: 0.99
    target_prior: uniform
  candidate_classes:
    test: mann_kendall
    alpha: 0.10
```

`target_prior` 非均衡数据可显式指定；默认只表示论文默认均匀先验，不允许从 clean
target 偷看类别分布。

### 8. Checkpoint 必需状态

- model、optimizer、scheduler、epoch 和 global step；
- marginal distribution；
- 每个 global index 的历史 posterior 或与精确 Mann-Kendall 等价的完整状态；
- current prediction、sample selection mask；
- candidate exclusion mask 和显著性参数；
- warm-up 阶段；
- global-index identity / dataset fingerprint；
- noise manifest identity 和 RNG state。

官方代码只保存 model/optimizer 不足以精确恢复 DSS；toolbox 必须保存 selector
全部状态。

### 9. 最小测试

1. 均匀 marginal 下 MDA 不改变 posterior。
2. MDA 输出有限、非负、逐行和为 1。
3. BASE 只选 `argmax == noisy_target` 的样本。
4. 手工轨迹的 Mann-Kendall 分数与公式一致。
5. 持续上升类别被排除，非显著类别不被排除。
6. noisy target 永远不进入 excluded mask。
7. candidate masked CE 与手算值一致。
8. excluded 类别 logit 的梯度为零，目标类梯度保持正确方向。
9. sample mask 与 candidate mask 的 shape 混用时明确失败。
10. shuffle 前后相同 global index 的状态一致。
11. clean target 不进入 selector 和 risk。
12. checkpoint roundtrip 后 MDA、轨迹、mask、epoch 完全一致。
13. CPU 单步和 CUDA 小样本 smoke 可完成。

### 10. 论文与官方代码核对

- `[论文][代码]` BASE 使用预测类别与 noisy target 一致性选样。
- `[论文][代码]` MDA 使用预测边际 EMA 校正 posterior。
- `[论文][代码]` CCS 使用 Mann-Kendall 上升趋势并排除候选类别。
- `[代码]` 官方实现保存 `[N, epochs, C]` posterior、累计 score、当前预测和
  marginal。
- `[代码]` Mann-Kendall 统计比较当前置信度与所有历史 epoch，显著性阈值为
  `norm.ppf(1-alpha)`；noisy-target 类被强制排除在 candidate 集合之外。
- `[代码]` clean truth 虽被加载并保存在 selector 对象中，但不参与选择公式；
  toolbox 不应复制这条数据通路。
- `[差异]` 官方实现的 epoch 状态更新时间使新 mask 主要作用于下一 epoch；
  toolbox 必须把这一时序写入配置和 checkpoint。
- `[差异]` 官方 checkpoint 没有保存 selector 状态，不能作为 toolbox 恢复标准。

### 11. 当前未实现

- MDA、PredictionMatchSelector 和 CandidateClassState
- candidate-class masked risk
- DSS Algorithm / Pipeline、配置、checkpoint 和测试
- DSS+ 双网络与一致性扩展
- 论文结果复现

因此当前 toolbox 不能宣称支持 DSS。

---

## 06. CDR：Robust Early-Learning

### 论文信息

- 编号 / 实际文件：
  `02_sample_selection/06_robust_early_learning_hinderin.pdf`
- `manifest.json` 旧记录：`02_sample_selection/06_elr_iclr2021.pdf` 下载失败
- 会议与年份：ICLR 2021
- 作者：Xiaobo Xia, Tongliang Liu, Bo Han, Chen Gong, Nannan Wang,
  Zongyuan Ge, Yi Chang
- 官方页面：<https://openreview.net/forum?id=Eql5b1_hTE4>
- 官方代码：<https://github.com/xiaoboxia/CDR>
- 当前成熟度：组件 L3；完整 CDR Pipeline L2
- 核对状态：已阅读论文；已检查官方 `main.py`；论文模式组件已实现并完成数学/CPU/CUDA 测试
- Toolbox 归属：`ParameterUpdatePolicy + 单网络 Pipeline`
- 不是：Loss、样本 Selector 或 TransitionEstimator

### 1. 论文实际做了什么

[论文] CDR 不挑样本，而是在每个 backward 后给每个标量参数计算重要性：

```text
criticality_i = abs(gradient_i * parameter_i)
```

设噪声率为 `tau`，全模型标量参数总数为 `m`，选择最大的
`(1 - tau) * m` 个作为 critical parameters。

[论文] critical parameter 使用：

```text
w_c <- w_c - lr * [(1 - tau) * grad_loss + lambda * sign(w_c)]
```

non-critical parameter 只做：

```text
w_n <- w_n - lr * lambda * sign(w_n)
```

因此论文中的正则项是 `lambda * ||W||_1` 所对应的 `sign(W)`，不是普通 optimizer
的 L2 `weight_decay * W`。论文还使用 noisy validation 的最小分类错误做 early
stopping。

### 2. 完整调用顺序

```text
batch(input, noisy target, global index)
  -> model
  -> 任意逐样本 Loss[B]
  -> mean()
  -> backward 得到原始 loss gradient
  -> CDRUpdatePolicy 计算全局 abs(grad * param)
  -> 生成 critical/non-critical 参数 mask
  -> 对两类参数执行论文更新式
  -> optimizer state / scheduler
  -> noisy validation early stopping
```

### 3. 与已有条目的重叠合并

- 复用现有 Loss 协议；CDR 不复制 CE。
- CDR 与 Selector 完全正交：Selector 决定哪些样本形成 loss，CDR 决定 loss
  backward 后哪些参数吸收梯度。
- `[已有/共享] algorithms/update_policy.py` 定义所有单模型、单 objective
  参数级更新共用的 `ParameterUpdatePolicy`；CDR 数学独立保存在
  `algorithms/cdr.py`。
- 数据、噪声 manifest、evaluator、checkpoint 外壳继续复用通用 runner。
- noisy validation 不等于 clean validation，配置必须显式区分。

### 4. 按顺序映射到文件和函数

1. `[已有/共享] src/lnl_toolbox/algorithms/update_policy.py`
   - `ParameterUpdateInput` / `ParameterUpdateResult`
   - `ParameterUpdatePolicy`
   - `StandardUpdatePolicy`
2. `[已有] src/lnl_toolbox/algorithms/cdr.py`
   - `critical_parameter_masks(named_parameters, noise_rate)`
   - `CDRUpdatePolicy.update(request)`
3. `[已有/扩展] src/lnl_toolbox/algorithms/supervised.py`
   - 所有 scalar objective 统一委托 ParameterUpdatePolicy 完成 backward/update。
4. `[已有/扩展] src/lnl_toolbox/training/experiment.py`
   - 构造 policy、聚合 `update_*` 指标并校验 resume 配置。
5. `[规划] src/lnl_toolbox/training/cdr_pipeline.py`
   - 仅在通用 runner 无法表达论文的显式 L1 更新时使用。
6. `[已有/高冲突] src/lnl_toolbox/plugins/builtin/catalog.py`
   - 注册 `parameter_update_policy/standard` 与 `parameter_update_policy/cdr`。
7. `[已有] configs/algorithm/cdr.yaml`
8. `[已有] tests/test_update_policy.py`、`tests/test_cdr.py`

### 5. 规划接口

```python
class ParameterUpdatePolicy(Protocol):
    def update(
        self,
        request: ParameterUpdateInput,
    ) -> ParameterUpdateResult:
        """拥有 zero_grad、backward 和一次受控 optimizer update。"""


def critical_parameter_masks(
    named_parameters: Iterable[tuple[str, Parameter]],
    critical_fraction: float,
) -> Mapping[str, Tensor]:
    ...
```

严格约束：

- 排序范围必须显式说明是“全部可训练标量”还是仅权重矩阵。
- 默认忠于论文，纳入所有有梯度的可训练参数。
- 并列 criticality 按 `(parameter_name, flat_offset)` 稳定打破。
- 使用论文模式时 optimizer 的普通 weight decay 必须为 0，避免重复正则。
- policy 不读取 global index、clean target 或样本选择真值。

### 6. Pipeline 伪代码

```python
logits = model(batch["input"])
per_sample = loss_fn(logits, batch["target"])
objective = per_sample.mean()
policy.update(ParameterUpdateInput(
    objective=objective,
    model=model,
    optimizer=optimizer,
    run_state=state,
))
```

不能简单把 non-critical 的 `.grad` 设为零后调用带 L2 weight decay 的 SGD，并
宣称等价于论文 Eq. 5–6。

### 7. 配置草案

```yaml
parameter_update:
  name: cdr
  noise_rate: 0.4
  l1_decay: 0.001
  critical_scope: all_trainable

optimizer:
  name: sgd
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0
```

### 8. Checkpoint 必需状态

- model、optimizer/scheduler 和 momentum state；
- epoch、global step、current learning rate；
- `noise_rate`、critical scope、gradient scale 和 L1 decay；
- `[完整 Pipeline 尚缺]` noisy validation identity、best noisy-validation error、early-stop state；
- noise manifest identity 和 RNG state。

critical mask 每 step 重算，不需要持久化；若为调试保存，只能作为 artifact，
不能成为恢复前提。

### 9. 最小测试

1. 小模型中 `abs(grad*w)` 与手算一致。
2. critical 参数数量严格匹配约定的舍入规则。
3. 并列值按参数名和 flat offset 稳定处理。
4. critical 参数包含缩放后的 loss gradient 和 L1 项。
5. non-critical 参数不包含 loss gradient，只包含 L1 项。
6. `w=0` 时 `sign(w)=0`。
7. optimizer weight decay 非零时论文模式明确报错。
8. 1D bias / BatchNorm 是否纳入与 `critical_scope` 一致。
9. 与 Selector 组合时只改变 selected objective，不改变 policy 协议。
10. clean target 不进入训练或 early stopping。
11. checkpoint roundtrip 后 momentum、best error 和 step 一致。
12. CPU 与 CUDA 的 mask、dtype、device 和更新结果一致。

### 10. 论文与官方代码核对

- `[论文][代码]` criticality 都以 `abs(gradient * parameter)` 为基础。
- `[论文][代码]` critical fraction 与 `1-noise_rate` 相关。
- `[代码]` 官方实现只拼接 `param.dim() in [2,4]` 的权重，忽略 bias 和
  BatchNorm 一维参数。
- `[代码]` 官方实现把 critical loss gradient 乘 `1-noise_rate`，non-critical
  loss gradient 置零，然后调用带 `weight_decay` 的 SGD。
- `[差异]` 因此官方代码实际对参数使用 optimizer 的 L2 decay；论文 Eq. 5–6
  明确写的是 `lambda * sign(W)`。
- `[差异]` 官方代码中的 gradual `clip` 先被计算，随后又被常数
  `1-noise_rate` 覆盖，实际没有 gradual 变化。
- `[差异]` 官方阈值用 `>= threshold`，并列时 critical 参数数量可能超过目标；
  toolbox 应采用确定性的精确 top-k。
- `[推断]` toolbox 当前只实现论文模式；未来若增加
  `compatibility_mode: official_code`，必须显式复现作者代码的参数 scope 和
  L2 decay，不能与论文模式静默混合。

### 11. 当前实施状态

- `[已实现]` 通用 ParameterUpdatePolicy、Standard policy 和 plugin/config。
- `[已实现]` CDR Eq. (3)-(6)、全局精确 top-k、稳定并列规则和 L1 更新。
- `[已实现]` 与 Loss/Selector 的生产组合、policy checkpoint 身份和 resume 配置校验。
- `[已验证]` 手算、失败边界、CPU/CUDA、checkpoint 与 noisy CUDA smoke。
- `[未实现]` noisy-validation early stopping、官方代码 compatibility mode 和论文结果复现。

因此当前 toolbox 可宣称支持 **paper-mode CDR ParameterUpdatePolicy 组件**，
但不能宣称完整复现包含 noisy-validation early stopping 的 CDR Pipeline。

---

## 07. CNLCU：Sample Selection with Uncertainty of Losses

### 论文信息

- 编号 / 文件：`02_sample_selection/07_uncertainty_selection.pdf`
- 会议与年份：ICLR 2022
- 作者：Xiaobo Xia, Tongliang Liu, Bo Han, Mingming Gong, Jun Yu,
  Gang Niu, Masashi Sugiyama
- 官方页面：<https://openreview.net/forum?id=xENf4QUL4LW>
- 官方代码：<https://github.com/xiaoboxia/CNLCU>
- 当前成熟度：L2
- 核对状态：已阅读论文；已检查官方 `loss.py`、
  `main_ours_soft.py` 和 `main_ours_hard.py`；尚未运行
- Toolbox 归属：`有状态不确定性 Selector + 双网络交换 Algorithm`
- 不是：普通 Loss 或单次 batch 的无状态 small-loss 排序

### 1. 论文实际做了什么

[论文] 对每个样本保存固定时间窗口内的训练 loss：

```text
L_t(i) = {loss_1(i), ..., loss_t(i)}
```

CNLCU-S 使用 soft truncation：

```text
psi(x) = log(1 + x + x^2 / 2)
robust_mean_s(i) = mean_t psi(loss_t(i))
```

CNLCU-H 先用 KNN 删除 `t_o` 个异常 loss，再计算剩余 loss 的均值。

[论文] Selector 不直接按 robust mean 排序，而使用置信区间下界。soft 版本：

```text
score_s =
  robust_mean_s
  - sigma^2 * (t + sigma^2 * log(2t) / t^2) / (selected_count - sigma^2)
```

hard 版本：

```text
score_h =
  robust_mean_h
  - [2 * sqrt(2*tau_min) * L * (t + sqrt(2)*t_o)
     * sqrt(log(4t) / selected_count)]
    / [(t-t_o) * sqrt(t)]
```

被选择次数少的样本会得到更低的下界，从而获得再次尝试的机会。

[论文] 两个网络分别计算 score 和小分集合，然后像 Co-teaching 一样交叉更新：

```text
network_1 用 network_2 选择的样本更新
network_2 用 network_1 选择的样本更新
```

remember rate 与第 4、9 篇共用论文线性 schedule。

### 2. 完整调用顺序

```text
双网络 logits
  -> 基础 Loss 分别输出 [B]
  -> LossHistoryState 按 global index 更新固定窗口
  -> CNLCU-S 或 CNLCU-H robust mean
  -> 读取 selected_count[N]
  -> confidence_lower_bound_scores[B]
  -> 两个 SmallLossSelector 结果
  -> peer exchange
  -> 各自基础 loss 在对方集合上聚合
  -> 更新两网络
  -> 更新每个样本的 selected_count
```

### 3. 与已有条目的重叠合并

- 复用第 4 篇的 `SelectionResult`、`RememberRateSchedule` 和稳定 top-k。
- 复用第 9 篇 Co-teaching 的双网络 peer-exchange 外壳；CNLCU 只替换 score
  provider 和增加历史状态。
- `[规划/共享] LossHistoryState` 同时可服务后续使用历史 loss 的方法；不得在
  CNLCU Algorithm 内用 batch 顺序维护匿名数组。
- CNLCU-S/H 是同一 selector 的 `estimator: soft|hard` 配置，不建两套 Pipeline。
- clean/noisy truth 只用于 evaluator 的 selection precision，不进入 score。

### 4. 按顺序映射到文件和函数

1. `[规划/共享] src/lnl_toolbox/selectors/history.py`
   - `LossHistoryState.update(indices, losses, epoch)`
   - `LossHistoryState.lookup(indices)`
   - `SelectionCountState.increment(indices)`
2. `[规划] src/lnl_toolbox/selectors/cnlcu.py`
   - `soft_truncated_mean(loss_history)`
   - `hard_truncated_mean(loss_history, outlier_policy)`
   - `cnlcu_soft_scores(mean, count, sigma2, t)`
   - `cnlcu_hard_scores(mean, count, tau_min, loss_bound, outlier_count, t)`
   - `CNLCUSelector`
3. `[扩展/共享] src/lnl_toolbox/algorithms/coteaching.py`
   - 后续抽出通用 `peer_exchange()`；保持现有函数兼容。
4. `[规划] src/lnl_toolbox/algorithms/cnlcu.py`
   - 双网络 logits、历史更新、两次选择和交叉更新。
5. `[规划] src/lnl_toolbox/training/cnlcu_pipeline.py`
6. `[扩展/高冲突] src/lnl_toolbox/plugins/builtin/catalog.py`
7. `[规划] configs/algorithm/cnlcu.yaml`
8. `[规划] tests/test_cnlcu.py`

若同事的 Selector 分支已有 state store 或 selection contract，以上规划必须实现为
其扩展，不能并行创建第二套。

### 5. 规划接口

```python
class StatefulScoreProvider(Protocol):
    def observe(
        self,
        *,
        global_indices: Tensor,  # [B]
        losses: Tensor,          # [B]
        epoch: int,
    ) -> None:
        ...

    def scores(self, global_indices: Tensor) -> Tensor:  # [B]
        ...
```

严格约束：

- history 和 selected count 以 `(dataset, split, global_index)` 为键。
- 窗口重置只能发生在配置规定的 epoch 边界。
- `selected_count` 的初始值必须避免除零，并在 metadata 中记录。
- 非法分母、NaN、负 count 或不足的 hard-history 必须明确失败。
- score 用 `.detach()` 参与排序；基础 loss 保留梯度用于 peer update。

### 6. Pipeline 伪代码

```python
loss_1 = loss_fn(model_1(x), noisy_target)  # [B]
loss_2 = loss_fn(model_2(x), noisy_target)  # [B]

history_1.observe(index, loss_1.detach(), epoch)
history_2.observe(index, loss_2.detach(), epoch)

score_1 = history_1.scores(index)
score_2 = history_2.scores(index)
selected_1 = selector.select(score_1, index, keep_rate)
selected_2 = selector.select(score_2, index, keep_rate)

objective_1 = loss_1[selected_2.batch_positions].mean()
objective_2 = loss_2[selected_1.batch_positions].mean()
update_both(objective_1, objective_2)

history_1.increment_selected(selected_1.global_indices)
history_2.increment_selected(selected_2.global_indices)
```

### 7. 配置草案

```yaml
algorithm:
  name: cnlcu
  estimator: soft
  history_window: 10
  confidence:
    sigma2: 0.1
  remember:
    noise_rate: 0.4
    gradual_epochs: 10
```

hard 模式额外配置 `tau_min`、`loss_bound` 和明确的 outlier policy。`sigma2` /
`tau_min` 可由 noisy validation 选择，不能使用 clean validation 调参。

### 8. Checkpoint 必需状态

- 两个 model、optimizer、scheduler；
- 两份 loss history 及各自窗口位置；
- 两份 selected-count state；
- estimator 类型、置信界参数、loss bound 和 outlier policy；
- remember schedule 阶段；
- dataset/split/global-index identity；
- epoch、global step、RNG 和 noise manifest identity。

### 9. 最小测试

1. `psi(x)`、soft robust mean 与手算一致。
2. hard truncation 在固定 fixture 中删除预期 outlier。
3. soft/hard confidence score 与论文公式一致。
4. selected count 较小时 score 更低，其他量相同时更容易被选。
5. history window 在规定边界重置，窗口内顺序正确。
6. shuffle 前后相同 global index 的 history 不变。
7. 两个网络严格使用对方的 SelectionResult。
8. remember schedule 与论文 Algorithm 1 一致。
9. 不足历史、非法分母、NaN 和非法参数明确失败。
10. clean truth 不进入 history、score 或 selection。
11. checkpoint roundtrip 后历史、count、窗口和 step 完全一致。
12. CPU 单步与 CUDA 多 epoch smoke 可完成。

### 10. 论文与官方代码核对

- `[论文][代码]` 都提供 CNLCU-S 和 CNLCU-H 两种 robust mean。
- `[论文][代码]` 都使用置信下界鼓励较少被选的样本。
- `[论文][代码]` 都采用双网络交叉更新和线性 forget-rate schedule。
- `[代码]` soft 实现对 score 使用 `relu(mean - bound)`；论文公式未要求
  ReLU。toolbox 默认保留论文原值，代码兼容模式才启用截断。
- `[代码]` 官方实现按 epoch 构造/拼接全数据 history 和 selection count。
- `[代码]` 官方 soft/hard 主循环用 batch 序号与固定 batch size 回写状态；
  toolbox 必须改成 global-index lookup。
- `[代码]` `noise_or_not` 仅用于 pure ratio；toolbox 放入 evaluator。
- `[差异]` 官方实现的若干 `s=epoch+1`、`count+1`、窗口初始化和 hard 公式常数
  是工程离散化；必须通过显式 compatibility mode 复现，不能替换论文默认公式。

### 11. 当前未实现

- LossHistoryState / SelectionCountState
- CNLCU-S/H score provider
- 通用 peer-exchange 双网络外壳
- CNLCU Algorithm / Pipeline、配置、checkpoint 和测试
- 论文结果复现

因此当前 toolbox 不能宣称支持 CNLCU。

---

## 08. MentorNet：Learning Data-Driven Curriculum

### 论文信息

- 编号 / 文件：`02_sample_selection/08_mentornet_icml2018.pdf`
- 会议与年份：ICML 2018
- 作者：Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, Li Fei-Fei
- 论文页面：<https://proceedings.mlr.press/v80/jiang18c.html>
- 官方代码：<https://github.com/google/mentornet>
- 当前成熟度：L2
- 核对状态：已阅读论文；已检查官方
  `code/cifar_train_mentornet.py`、`code/utils.py`、
  `code/training_mentornet/models.py` 和
  `code/training_mentornet/train.py`；尚未运行
- Toolbox 归属：`WeightProvider + Mentor Model + Student Pipeline`
- 不是：普通 Loss；也不应只用二值 sample mask 表达

### 1. 论文实际做了什么

[论文] StudentNet 的训练目标对每个样本引入 `v_i in [0,1]`：

```text
F(w,v) =
  mean_i [v_i * loss_i(w)]
  + curriculum_regularizer G(v; lambda)
  + theta * ||w||_2^2
```

MentorNet `g_m(z_i; Theta)` 根据样本和训练状态特征输出动态权重。论文实验中的
特征包含：

```text
current loss
loss - moving percentile loss
label feature
training epoch percentage
```

[论文] 预定义线性 curriculum 的解析权重为：

```text
lambda2 == 0:
    v_i = 1(loss_i <= lambda1)
lambda2 != 0:
    v_i = clip(1 - (loss_i - lambda1) / lambda2, 0, 1)
```

数据驱动 MentorNet 用另一份小数据 `D'` 训练；监督目标 `v_i*` 表示该标签是否
正确。目标实验数据没有 clean truth 时，可从另一份受信数据训练 MentorNet 后迁移。

[论文] SPADE 在每个 mini-batch 在线计算 `v[B]`，固定 MentorNet 后用
`mean(v * loss)` 更新 StudentNet。测试时只保留 StudentNet。

### 2. 完整调用顺序

```text
阶段 A（可选，离线）：
trusted curriculum dataset D'
  -> Student feature snapshot
  -> curriculum target v*
  -> train MentorNet
  -> MentorArtifact

阶段 B（目标 noisy run）：
batch(input, noisy target, global index)
  -> StudentNet -> per-sample loss[B]
  -> moving-percentile state
  -> MentorFeatureBatch(loss, loss_diff, noisy label feature, epoch)
  -> WeightProvider -> WeightResult.weights[B]
  -> mean(weights.detach() * loss)
  -> update StudentNet
```

### 3. 与已有条目的重叠合并

- 复用现有逐样本 Loss；MentorNet 只产生权重 `[B]`。
- `WeightResult` 与 `SelectionResult` 分开：前者允许连续 `[0,1]` 权重，后者
  表示选择位置。需要二值 mask 的消费者可由 `weights > 0` 派生，但不能反向丢失
  权重信息。
- moving percentile 属于共享训练统计状态，可与后续 reweighting 方法复用；
  不与 CNLCU 的逐样本 loss history 混成同一状态。
- MentorNet PD（预定义 curriculum）和 DD（数据驱动 curriculum）使用同一
  WeightProvider 接口、不同 MentorArtifact。
- 不给目标训练 Dataset 暴露 clean target。DD 所需 `v*` 只能来自显式隔离的
  trusted curriculum dataset / 离线 artifact 创建任务。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] src/lnl_toolbox/core/result.py`
   - `WeightResult(weights[B], global_indices[B], metadata)`
2. `[规划] src/lnl_toolbox/algorithms/mentornet.py`
   - `MentorFeatureBatch`
   - `MovingPercentileState.update(losses, epoch)`
   - `PredefinedCurriculumWeightProvider`
   - `MentorNetWeightProvider`
   - `weighted_student_objective(losses, weights)`
3. `[规划] src/lnl_toolbox/models/mentornet.py`
   - `MentorNet`
   - loss/loss-difference encoder、label/epoch embedding 和 sigmoid head。
4. `[规划] src/lnl_toolbox/training/mentornet_pipeline.py`
   - 加载/冻结 MentorArtifact，执行 burn-in 和 Student 更新。
5. `[规划] src/lnl_toolbox/training/mentor_learning.py`
   - 只有实现 DD 离线教师训练时才新增；首版可只支持 PD，避免过早扩展。
6. `[扩展/高冲突] src/lnl_toolbox/plugins/builtin/catalog.py`
   - 注册 `weight_provider/mentornet_pd|mentornet_dd`。
7. `[规划] configs/algorithm/mentornet.yaml`
8. `[规划] tests/test_mentornet.py`

### 5. 规划接口

```python
@dataclass(frozen=True)
class WeightResult:
    weights: Tensor         # finite [B], 0 <= weight <= 1
    global_indices: Tensor  # [B]
    metadata: Mapping[str, Any]


class WeightProvider(Protocol):
    def weights(
        self,
        *,
        losses: Tensor,          # detached [B]
        noisy_targets: Tensor,   # [B]
        global_indices: Tensor,  # [B]
        epoch_fraction: Tensor,  # [B]
    ) -> WeightResult:
        ...
```

严格约束：

- MentorNet 看到的 `losses` 必须 detach，Student 更新时 weights 也 detach。
- `weighted_student_objective` 默认忠于论文/官方训练代码，计算
  `mean(weights * losses)`，不是除以 `sum(weights)`。
- 全零权重必须明确失败或由配置定义安全退化，不能产生静默零训练。
- stochastic dropout 使用独立 RNG stream，并进入 checkpoint。
- MentorArtifact 必须记录模型结构、feature schema、训练来源和 hash。

### 6. Pipeline 伪代码

```python
for batch in train_loader:
    logits = student(batch["input"])
    per_sample = loss_fn(logits, batch["target"])  # [B]

    stats = moving_percentile.update(per_sample.detach(), epoch)
    weight_result = mentor_provider.weights(
        losses=per_sample.detach(),
        noisy_targets=batch["target"],
        global_indices=batch["index"],
        epoch_fraction=epoch_fraction,
    )
    weights = apply_burn_in_and_dropout(
        weight_result.weights,
        epoch=epoch,
        rng=curriculum_rng,
    ).detach()

    objective = (weights * per_sample).mean()
    update_student(objective)
```

MentorNet 自身的训练和 StudentNet 的目标 noisy run 是两个阶段。首个可验收版本应
先实现论文的预定义 curriculum 和固定 MentorArtifact 消费，再实现 DD 训练器。

### 7. 配置草案

```yaml
algorithm:
  name: mentornet
  curriculum: predefined_linear
  lambda1: 1.0
  lambda2: 1.0
  burn_in:
    epochs: 18
    mode: all
  moving_percentile:
    percentile: 0.70
    decay: 0.95
  example_dropout:
    schedule: []
```

DD 模式必须额外提供 `mentor_artifact`，不能在目标训练 run 中临时读取 clean
target 生成 `v*`。

### 8. Checkpoint 必需状态

- StudentNet model、optimizer、scheduler；
- MentorNet 参数或 MentorArtifact identity/hash；
- MentorNet 是否冻结及 curriculum mode；
- moving-percentile 累计值；
- burn-in 阶段、epoch percentage 和 dropout schedule；
- curriculum RNG state；
- loss schema、label embedding schema；
- epoch、global step、noise manifest identity。

若实现 DD 教师训练，还需单独保存 Mentor optimizer、trusted dataset identity 和
teacher-training step；不能混入 Student checkpoint 的普通 model 字段。

### 9. 最小测试

1. 预定义 hard / linear curriculum 与论文 Eq. 7 一致。
2. 权重有限、shape 为 `[B]` 且位于 `[0,1]`。
3. `mean(weights*losses)` 与手算一致。
4. 测试明确区分 weighted mean 与除以权重和的 normalized mean。
5. moving percentile EMA 与手算一致。
6. burn-in 的 `all` / Bernoulli 行为由配置确定且 seed 可复现。
7. MentorNet 参数在 Student step 中不变化。
8. weights 路径不会把梯度传回 Student loss 特征。
9. 全零权重按约定失败或安全退化。
10. target noisy run 无法访问 clean target 或 `v*`。
11. MentorArtifact schema/hash 不匹配时失败。
12. checkpoint roundtrip 后 moving state、RNG、artifact identity 和 step 一致。
13. CPU 单步和 CUDA smoke 可完成。

### 10. 论文与官方代码核对

- `[论文][代码]` MentorNet 输入含 loss、loss 与移动统计的差、label 和 epoch
  percentage，输出 `[0,1]` 权重。
- `[论文][代码]` Student 使用 `mean(weight * loss)` 更新，而不是按权重和归一化。
- `[论文][代码]` MentorNet 在 Student 训练中作为固定 teacher，测试时不用。
- `[代码]` 官方 TensorFlow 实现维护 batch loss percentile 的 EMA。
- `[代码]` 官方 CIFAR runner 实际给 MentorNet 的 label feature 全部置零，未使用
  noisy label embedding；toolbox 应把 feature schema 记录在 artifact 中。
- `[代码]` 官方重实现的 burn-in 配置说明为全样本权重 1；论文正文描述首 20%
  epoch 使用 Bernoulli 随机 dropout。两者必须作为不同 mode。
- `[代码]` 官方 `probabilistic_sample(..., mode='random')` 按指定比例均匀丢样，
  与 MentorNet 输出权重无关，然后保留被选样本的连续权重。
- `[代码]` ResNet runner 把 weight decay loss 乘平均样本利用率。
- `[差异]` 官方代码调用 `tf.stop_gradient(v)` 却没有接收返回 tensor；按
  TensorFlow 语义这不会截断后续 `v` 的梯度路径。toolbox 应忠于 SPADE 中固定
  `v` 的更新语义，显式 detach。
- `[代码]` 官方仓库基于 Python 2.7 / TensorFlow 1.8，不能直接成为 PyTorch
  toolbox 依赖。

### 11. 当前未实现

- `WeightResult` / `WeightProvider`
- MentorNet PyTorch 模型和 MentorArtifact
- moving-percentile、burn-in 和 dropout state
- predefined / data-driven curriculum
- MentorNet Pipeline、配置、checkpoint 和测试
- 论文结果复现

因此当前 toolbox 不能宣称支持 MentorNet。

---

## 09. Co-teaching：Robust Training of Deep Neural Networks with Extremely Noisy Labels

### 论文信息

- 编号 / 文件：`02_sample_selection/09_coteaching_neurips2018.pdf`
- 会议与年份：NeurIPS 2018
- 作者：Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu,
  Ivor W. Tsang, Masashi Sugiyama
- 论文页面：<https://papers.nips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html>
- 官方代码：<https://github.com/bhanML/Co-teaching>
- 当前成熟度：L2
- 核对状态：已阅读论文；已检查官方 `loss.py`、`main.py`；尚未运行
- Toolbox 归属：`Selector + 双网络 Algorithm + Pipeline`
- 不是：一种新的 Loss；也不是把两个网络的 loss 直接相加

### 1. 论文实际做了什么

[论文] 同时初始化两个不同的网络。对每个 mini-batch，两个网络分别计算逐样本
loss，各自选择小损失样本，再把自己的选择交给对方更新：

```text
network A loss -> A 选出 small-loss set A
network B loss -> B 选出 small-loss set B

update A with set B
update B with set A
```

论文 Algorithm 1 的保留率为：

```text
R(T) = 1 - min((T / Tk) * τ, τ)
```

其中 `τ` 是预估噪声率，`Tk` 是逐步丢弃的 epoch 数。Co-teaching 的关键是
`peer exchange`；每个网络不能用自己选出的样本更新自己。

### 2. 完整调用顺序

```text
同一 noisy batch(input, target, global index)
        ├── model A -> per-sample loss A[B]
        └── model B -> per-sample loss B[B]
                    ↓
RememberRateSchedule(epoch) -> keep_rate
                    ↓
SmallLossSelector(A) / SmallLossSelector(B)
                    ↓
SelectionResult A / SelectionResult B
                    ↓
peer exchange
        ├── objective A = mean(loss A[positions selected by B])
        └── objective B = mean(loss B[positions selected by A])
                    ↓
分别 backward / optimizer step
                    ↓
clean evaluator + 双网络 checkpoint
```

### 3. 与已有条目的重叠合并

- 与第 4 篇 JoCoR、第 7 篇 CNLCU 共用 `SelectionResult`、
  `RememberRateSchedule`、稳定小损失排序和双网络 checkpoint 外壳。
- 现有 `algorithms/coteaching.py` 已有 NumPy `remember_rate()`、
  `_small_loss()` 和 `coteaching_exchange()`；它是兼容 helper，不再新建
  第二个同义 Co-teaching 数学文件。
- 排序相同 loss 时必须按 global index 打破并列，不能依赖 batch 顺序。
- clean/noisy 真值只供 evaluator 计算 pure ratio，不进入 selector 或 algorithm。
- Co-teaching 和 JoCoR 的区别必须保留：Co-teaching 是两套独立 small-loss set
  并交叉更新；JoCoR 是联合 loss 和共同选择集合。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] src/lnl_toolbox/core/result.py`
   - `SelectionResult(batch_positions, global_indices, scores, metadata)`
2. `[规划/共享] src/lnl_toolbox/selectors/small_loss.py`
   - `RememberRateSchedule`
   - `SmallLossSelector.select(losses, global_indices, keep_rate)`
3. `[扩展] src/lnl_toolbox/algorithms/coteaching.py`
   - 保留现有 `remember_rate()` / `coteaching_exchange()`
   - 增加 Torch `peer_exchange()` 和 `CoTeachingAlgorithm`
4. `[规划] src/lnl_toolbox/training/coteaching_pipeline.py`
   - 构造双模型、双 optimizer、scheduler、训练与恢复流程。
5. `[扩展/高冲突] src/lnl_toolbox/plugins/builtin/catalog.py`
   - 注册 `selector/small_loss`、`algorithm/coteaching` 和对应 pipeline。
6. `[规划] configs/algorithm/coteaching.yaml`
7. `[规划] tests/test_coteaching.py`

### 5. 规划接口

```python
def peer_exchange(
    *,
    losses_a: Tensor,             # [B], 保留梯度
    losses_b: Tensor,             # [B], 保留梯度
    selected_by_a: SelectionResult,
    selected_by_b: SelectionResult,
) -> tuple[Tensor, Tensor]:
    """返回 objective_a、objective_b；A 使用 B 的选择，B 使用 A 的选择。"""
    ...
```

严格约束：

- selector 只读取 detached loss 和 global index。
- `SelectionResult.batch_positions` 只在当前 batch 内消费；
  `global_indices` 用于审计、评测和状态恢复。
- 选择数量必须明确约定。论文要求集合大小至少为 `R(T)|B|`，toolbox 规范版使用
  `ceil`；作者代码兼容模式使用 `int()` 向下取整。
- 至少保留一个样本；非法 keep rate、空 batch、NaN loss 必须失败。

### 6. Pipeline 伪代码

```python
for batch in train_loader:
    logits_a = model_a(batch["input"])
    logits_b = model_b(batch["input"])
    losses_a = loss_fn(logits_a, batch["target"])  # [B]
    losses_b = loss_fn(logits_b, batch["target"])  # [B]

    keep_rate = remember_schedule(epoch)
    selected_a = selector.select(
        losses_a.detach(), batch["index"], keep_rate
    )
    selected_b = selector.select(
        losses_b.detach(), batch["index"], keep_rate
    )

    objective_a, objective_b = peer_exchange(
        losses_a=losses_a,
        losses_b=losses_b,
        selected_by_a=selected_a,
        selected_by_b=selected_b,
    )
    update_model_a(objective_a)
    update_model_b(objective_b)
```

### 7. 配置草案

```yaml
algorithm:
  name: coteaching
  remember:
    noise_rate: 0.4
    gradual_epochs: 10
    exponent: 1.0
    count_rounding: ceil
  selector: {name: small_loss}
```

`noise_rate` 是算法超参数，不得由 clean label 在线估计。若与 manifest 的实际噪声率
不同，resolved config 必须同时记录二者。

### 8. Checkpoint 必需状态

- 两个 model、optimizer、scheduler；
- 两个模型的初始化 seed / RNG stream；
- loss 配置、selector 配置、remember-rate schedule；
- epoch、global step、noise manifest identity；
- best metric 与选用哪个网络进行最终评测的规则。

### 9. 最小测试

1. A 严格使用 B 的选择，B 严格使用 A 的选择。
2. 相同 loss 并列时按 global index 稳定选择。
3. 论文保留率公式及边界正确。
4. 论文 `ceil` 规范模式与官方 `int` 兼容模式明确区分。
5. 两个网络参数都发生更新，且不会误用同一个 optimizer。
6. 任意注册 loss 的 `[B]` 输出均可接入。
7. clean truth 不进入 selector 或 algorithm。
8. checkpoint roundtrip 后两个网络、optimizer、schedule 和 step 一致。
9. CPU 单步、CUDA 两 epoch smoke 和 resume 可完成。

### 10. 论文与官方代码核对

- `[论文][代码]` 两个网络分别按小损失选样，再交叉更新。
- `[论文][代码]` 默认 forget rate 使用噪声率，并在前 `num_gradual` 个 epoch
  线性升高。
- `[代码]` 官方 `loss.py` 用 `np.argsort` 排序，使用
  `int(remember_rate * batch_size)`，即向下取整。
- `[代码]` 官方 pure ratio 读取 `noise_or_not`；toolbox 只能在 evaluator 中
  计算该指标。
- `[差异]` 作者代码的 schedule 前段终点是 `forget_rate**exponent`，论文主公式
  写作线性到 `τ`；兼容模式必须显式记录 exponent。
- `[代码]` 官方环境为 Python 2.7 / PyTorch 0.3，不能直接复用。

### 11. 当前未实现

- 公共 `SelectionResult` / `SmallLossSelector`
- Torch 双网络 Algorithm 与统一 peer-exchange
- 双模型 checkpoint、Pipeline、配置和正式测试
- 论文结果复现

当前已有的 NumPy helper 只证明基本选择交换逻辑存在，不能宣称完整支持 Co-teaching。

---

## 10. Loss Correction：Making Deep Neural Networks Robust to Label Noise

### 论文信息

- 编号 / 文件：`03_robust_loss/10_loss_correction_cvpr2017.pdf`
- 会议与年份：CVPR 2017
- 作者：Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon,
  Richard Nock, Lizhen Qu
- 官方页面：<https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html>
- 官方代码：<https://github.com/giorgiop/loss-correction>
- 当前成熟度：L2（Anchor estimator 已有，校正风险未实现）
- 核对状态：已阅读论文；已检查官方 `loss.py`、`models.py`；尚未运行
- Toolbox 归属：`TransitionEstimator + RiskCorrector + Pipeline`
- 不是：把 `T` 写死进普通 CE 类；也不是训练时生成噪声

### 1. 论文实际做了什么

论文采用：

```text
T[i,j] = P(noisy label=j | clean label=i)
p_noisy(row) = p_clean(row) @ T
```

与当前 toolbox 的 `clean_to_noisy_row` 方向一致。

[论文] Backward correction 先构造每个假设类别对应的 loss 向量：

```text
loss_vector = [loss(clean_class=0), ..., loss(clean_class=C-1)]
corrected_vector = inverse(T) @ loss_vector
result = corrected_vector[observed_noisy_target]
```

它给出无偏 clean risk，但需要 `T` 可逆，结果可以为负，病态矩阵会放大方差。

[论文] Forward correction 不求逆，而是在 loss 前把模型 clean posterior 映射为：

```text
p_noisy = p_clean @ T
result = -log(p_noisy[observed_noisy_target])  # CE 情形
```

对于 proper composite loss，它与 clean risk 具有相同 minimizer。

### 2. T 未知时的完整调用顺序

```text
noisy train data
  -> 普通 noisy-label warm-up
  -> collect PosteriorSnapshot[N,C]
  -> AnchorTransitionEstimator
  -> TransitionArtifact[C,C]
  -> 重新初始化或复用模型
  -> ForwardRiskCorrector / BackwardRiskCorrector
  -> noisy-label training
  -> clean validation / test
```

论文 Eq. 12–13 的 anchor 估计已经由
`noise/estimators.py::AnchorTransitionEstimator` 实现。

### 3. 与已有条目的重叠合并

- 不再实现第二套 posterior snapshot、anchor 搜索或矩阵 artifact。
- `TransitionArtifact` 是估计器与消费者之间的唯一边界；Forward、Backward、
  T-Revision 等只消费 provider，不拥有估计过程。
- Forward/Backward 与第 14 篇 Natarajan 都属于风险校正，共用
  `algorithms/transition_risk.py::RiskCorrector`。
- 普通 `Loss(logits, target)->[B]` 保持不变；RiskCorrector 在其外层组合
  transition 与 base loss。
- Backward correction 需要所有假设目标的 loss `[B,C]`；这一内部能力应由现有
  loss 模块扩展，不为每种校正复制 one-hot 循环。

### 4. 按顺序映射到文件和函数

1. `[已有] src/lnl_toolbox/noise/estimators.py`
   - `PosteriorSnapshot`
   - `AnchorTransitionEstimator.estimate()`
2. `[已有] src/lnl_toolbox/noise/transition.py`
   - `TransitionProvider`、`TransitionArtifact`、`KnownTransition`
3. `[扩展] src/lnl_toolbox/losses/torch_losses.py`
   - `loss_for_all_targets(loss, logits) -> Tensor[B,C]`
4. `[规划/共享] src/lnl_toolbox/algorithms/transition_risk.py`
   - `ForwardRiskCorrector`
   - `BackwardRiskCorrector`
5. `[规划/共享] src/lnl_toolbox/training/snapshots.py`
   - noisy warm-up 与 posterior 收集。
6. `[规划] src/lnl_toolbox/training/loss_correction_pipeline.py`
   - 只负责编排估计和校正阶段。
7. `[扩展/高冲突] src/lnl_toolbox/plugins/builtin/catalog.py`
   - 注册 `risk_corrector/forward|backward`。
8. `[规划] configs/algorithm/loss_correction.yaml`
9. `[规划] tests/test_transition_risk.py`

### 5. 规划接口

```python
class RiskCorrector(Protocol):
    def per_sample_risk(
        self,
        *,
        logits: Tensor,          # [B,C]
        noisy_targets: Tensor,   # [B]
        transition: TransitionProvider,
    ) -> Tensor:                 # finite [B]
        ...
```

```python
def loss_for_all_targets(
    base_loss: nn.Module,
    logits: Tensor,
) -> Tensor:                    # [B,C]
    ...
```

严格约束：

- Forward 计算必须遵循 `p_clean @ T`，并使用稳定 log-space。
- Backward 允许返回负值；不得偷偷 clamp 为零。
- Backward 在求解前检查方阵、可逆性和条件数；超限明确失败。
- provider 的 device/dtype 必须跟随 logits；不得在 step 中重新从 NPZ 读取。

### 6. Pipeline 伪代码

```python
if config.transition.known:
    transition = KnownTransition.from_manifest(manifest)
else:
    noisy_model = pretrain_noisy_classifier(train_loader)
    snapshot = collect_posterior_snapshot(noisy_model, train_loader)
    transition = AnchorTransitionEstimator().estimate(snapshot)

corrector = build_risk_corrector(config.risk_corrector)
for batch in train_loader:
    logits = model(batch["input"])
    per_sample = corrector.per_sample_risk(
        logits=logits,
        noisy_targets=batch["target"],
        transition=transition,
    )
    update_model(per_sample.mean())
```

### 7. 配置草案

```yaml
algorithm:
  name: loss_correction
  correction: forward
transition:
  source: anchor
  warmup_epochs: 20
loss: {name: ce}
```

Backward 模式还需显式配置允许的最大条件数。已知矩阵模式只从 manifest 或
TransitionArtifact 读取，不允许把矩阵散落写在 loss 配置中。

### 8. Checkpoint 必需状态

- 当前训练阶段（warm-up / corrected training）；
- warm-up 与主模型、各自 optimizer/scheduler；
- PosteriorSnapshot hash、TransitionArtifact hash 和矩阵方向；
- corrector 类型、base loss 配置、condition-number policy；
- epoch、global step、manifest identity 和 RNG。

### 9. 最小测试

1. Forward 手算满足 `p_noisy=p_clean@T`。
2. `T=I` 时 Forward/Backward 均退化为 base loss。
3. Backward 与枚举公式、梯度一致，并允许负风险。
4. 奇异或病态 `T` 按配置失败。
5. Anchor estimator 现有测试继续通过。
6. 任意输入重排不改变 artifact；global index tie-break 稳定。
7. clean target 不进入 Snapshot 或 RiskCorrector。
8. known/estimated transition 两条 pipeline 都可调用。
9. checkpoint 恢复后继续使用同一 artifact hash。
10. CPU 单步与 CUDA smoke 可完成。

### 10. 论文与官方代码核对

- `[论文][代码]` 官方 `robust('backward', P)` 使用 `inverse(P)` 左乘 one-hot
  noisy target，再与 `log(y_pred)` 组合。
- `[论文][代码]` Forward 使用 `y_pred @ P` 后与 noisy one-hot 计算 CE。
- `[论文]` Anchor 方法训练 noisy posterior 后按每类最大 posterior 取一行。
- `[代码]` 官方实现对 posterior 做 Keras epsilon clip；toolbox 应使用稳定
  log-space，而不是改变矩阵。
- `[代码]` 官方仓库是旧 Keras 实现，且未提供当前 toolbox 所需的 artifact、
  hash、global index 和 checkpoint 边界。

### 11. 当前未实现

- Forward / Backward RiskCorrector
- 通用 `[B,C]` all-target loss 内部协议
- warm-up → artifact → corrected training Pipeline
- 对应配置、checkpoint 和测试

Anchor estimator 已实现不等于整篇 Loss Correction 已实现。

---

## 11. Normalized Losses / APL：Normalized Loss Functions for Deep Learning with Noisy Labels

### 论文信息

- 编号 / 文件：`03_robust_loss/11_normalized_losses_icml2020.pdf`
- 会议与年份：ICML 2020
- 作者：Xingjun Ma, Hanxun Huang, Yisen Wang, Simone Romano,
  Sarah Erfani, James Bailey
- 官方页面：<https://proceedings.mlr.press/v119/ma20c.html>
- 官方代码：<https://github.com/HanxunH/Active-Passive-Losses>
- 当前成熟度：L3（P0 的 NCE、MAE、RCE、APL 已实现）
- 核对状态：已阅读论文；已检查官方 `loss.py`、`main.py`、`trainer.py`；
  当前仓库已有数学与集成测试
- Toolbox 归属：`Loss`
- 不是：Selector、NoiseModel 或 transition correction

### 1. 论文实际做了什么

[论文] 对任意多分类 loss，按所有可能目标的 loss 和归一化：

```text
L_normalized(logits, y)
  = L(logits, y) / sum_j L(logits, j)
```

论文证明在相应条件下，归一化 loss 对对称/非对称标签噪声具有鲁棒性。

论文进一步区分：

```text
Active loss：
  只有 observed target 的 elementary term 非零，例如 CE / NCE

Passive loss：
  至少一个非 target elementary term 非零，例如 MAE / RCE
```

APL 为：

```text
APL = alpha * active_loss + beta * passive_loss
alpha > 0, beta > 0
```

论文实验组合包括 NCE+MAE、NCE+RCE、NFL+MAE、NFL+RCE。

### 2. 当前 toolbox 已有调用顺序

```text
YAML loss config
  -> PluginCatalog
  -> NCE / MAE / RCE
  -> ActivePassiveLoss(active, passive, alpha, beta)
  -> per-sample loss[B]
  -> Algorithm.mean()
  -> backward
```

当前 P0 明确只支持：

```text
active = NCE
passive = MAE or RCE
```

这忠于论文允许的组合子集，但不能宣称覆盖论文全部 NFL 组合。

### 3. 与已有条目的重叠合并

- 继续使用 `losses/torch_losses.py`，不创建 `apl_loss.py`、`nce_loss.py`
  等重复文件。
- 与第 12 篇 GCE 共用普通 `Loss(logits, targets)->[B]` 协议。
- 论文通用归一化需要 all-target loss `[B,C]`；若实现，复用第 10 篇规划的
  `loss_for_all_targets()`，不另建 normalization 数据通道。
- APL 的子 loss 继续由 registry 递归构造。
- evaluator 只聚合实际配置的 objective；不同 loss 数值不能直接比较优劣。

### 4. 按顺序映射到文件和函数

1. `[已有] src/lnl_toolbox/losses/torch_losses.py`
   - `NormalizedCrossEntropyLoss`
   - `MeanAbsoluteErrorLoss`
   - `ReverseCrossEntropyLoss`
   - `ActivePassiveLoss`
2. `[已有] src/lnl_toolbox/plugins/builtin/catalog.py`
   - `nce`、`mae`、`rce`、`apl` 的构造与嵌套校验。
3. `[已有] src/lnl_toolbox/training/experiment.py`
   - 校验 `[B]` 并由 algorithm 求 mean。
4. `[扩展，可选] src/lnl_toolbox/losses/torch_losses.py`
   - `NormalizedFocalLoss`
   - 通用 `NormalizedLoss`，复用 `loss_for_all_targets()`。
5. `[扩展] tests/test_torch_losses.py`
   - 已有 P0 测试；将来只在实现 NFL 时补论文全组合。

### 5. 当前公共接口

```python
loss(
    logits: Tensor,       # [B,C]
    targets: Tensor,      # long [B]
) -> Tensor               # finite [B]
```

```python
ActivePassiveLoss(
    active=NormalizedCrossEntropyLoss(...),
    passive=MeanAbsoluteErrorLoss(...) | ReverseCrossEntropyLoss(...),
    alpha: float > 0,
    beta: float > 0,
)
```

当前公式：

```text
NCE = -log(p_y) / sum_j[-log(p_j)]
MAE = scale * (1 - p_y), default scale=2
RCE = -log_zero * (1 - p_y), default log_zero=-4
APL = alpha*NCE + beta*(MAE or RCE)
```

### 6. Pipeline 伪代码

```python
loss_fn = catalog.build("loss", config["loss"])

for batch in train_loader:
    logits = model(batch["input"])
    per_sample = validate_per_sample_loss(
        loss_fn(logits, batch["target"]),
        batch_size=len(batch["target"]),
    )
    update_model(per_sample.mean())
```

APL 不拥有 optimizer、selector、sample history 或 clean target。

### 7. 配置草案

```yaml
loss:
  name: apl
  alpha: 1.0
  beta: 1.0
  active: {name: nce, eps: 1.0e-8}
  passive: {name: rce, log_zero: -4.0}
```

论文不同数据集使用的 alpha/beta 属实验超参数；默认值不是论文统一最优值。

### 8. Checkpoint 必需状态

Loss 本身无可学习状态。checkpoint 只需保存 resolved loss config，包括：

- loss 名称和子 loss；
- alpha、beta；
- eps、MAE scale、RCE log_zero；
- 将来 NFL 的 gamma。

恢复时配置必须一致；不能只记录字符串 `apl`。

### 9. 最小测试

1. NCE 与手算公式一致，所有假设标签的 NCE 和约为 1。
2. MAE、RCE 与 `p_y` 公式一致。
3. APL 等于两项逐样本加权和。
4. alpha/beta 非正、active/passive 角色错误时失败。
5. 极端 logits 的前向、反向和梯度有限。
6. 输出严格为 `[B]`。
7. registry 可构造 NCE+MAE、NCE+RCE。
8. 六种 P0 loss 均可完成 CPU 单步；APL CUDA smoke 可完成。
9. 将来通用 normalization 必须验证分母和 all-target shape。

### 10. 论文与官方代码核对

- `[论文][代码]` NCE 使用 target CE 除以所有类别 CE 的和。
- `[论文][代码]` APL 把 robust active 与 robust passive loss 相加。
- `[代码]` 官方 `loss.py` 同时实现 NCE+RCE、NCE+MAE、NFL+RCE、
  NFL+MAE 等组合。
- `[代码]` 官方 MAE 实现使用 reduced 形式 `1-p_y`；原始分类 MAE 为
  `2(1-p_y)`。toolbox 用显式 `scale` 表达这一区别。
- `[代码]` 官方 RCE 把 one-hot 的零截断到 `1e-4`；toolbox 等价地显式使用
  `log_zero`，避免隐藏常数。
- `[差异]` 当前 toolbox 对 APL 角色做严格 P0 校验，只覆盖 NCE + MAE/RCE；
  论文还允许 NFL 组合。

### 11. 当前未实现

- Normalized Focal Loss
- 对任意 base loss 的通用归一化 wrapper
- NFL+MAE / NFL+RCE
- 论文全部数据集和超参数复现

P0 主线已实现并可训练；完整论文覆盖仍需上述扩展。

---

## 12. GCE：Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels

### 论文信息

- 编号 / 文件：`03_robust_loss/12_gce_neurips2018.pdf`
- 会议与年份：NeurIPS 2018
- 作者：Zhilu Zhang, Mert R. Sabuncu
- 论文页面：<https://papers.nips.cc/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html>
- 官方代码：论文及作者公开页面未给出可核实的作者官方仓库
- 当前成熟度：L3（标准 GCE 已实现；Truncated GCE 未实现）
- 核对状态：已阅读论文并核对当前数学实现；无作者代码可交叉检查
- Toolbox 归属：`Loss`；Truncated 版本还需要 `Selector + Stateful Pipeline`
- 不是：把 `q` 当作样本权重；也不是 APL 的 NCE

### 1. 论文实际做了什么

[论文] 标准 GCE（论文记为 `L_q`）：

```text
L_q(p, y) = (1 - p_y^q) / q
0 < q <= 1
```

边界关系：

```text
q -> 0：L_q -> -log(p_y)          # CE
q = 1：L_q = 1 - p_y              # 与分类 MAE 成比例
```

梯度相对 CE 多出 `p_y^q` 因子，因此降低低置信样本对更新的影响。`q` 越接近 0，
越像 CE；越接近 1，越鲁棒但也越可能欠拟合。

[论文] Truncated GCE 在阈值 `k` 处截断：

```text
L_trunc(p_y) =
    L_q(k),  if p_y <= k
    L_q(p_y), if p_y > k
```

论文 Algorithm 1 不是仅替换公式，而是 alternate convex search：周期性按全训练集
posterior 更新保留权重，再训练加权 GCE。

### 2. 完整调用顺序

标准 GCE 已有：

```text
logits + noisy target
  -> GeneralizedCrossEntropyLoss(q)
  -> per-sample loss[B]
  -> Algorithm.mean()
```

Truncated GCE 未来流程：

```text
阶段 0：全部样本 weight=1
  -> 训练标准 GCE
  -> 周期性收集全训练集 p_y[global_index]
  -> weight[index] = 1(p_y > k)
  -> 用固定 weight 继续训练
  -> 重复指定轮数
```

### 3. 与已有条目的重叠合并

- 标准 GCE 继续位于 `losses/torch_losses.py`，不新增另一份 GCE。
- 与第 11 篇 APL 共用逐样本 `[B]` 协议，但 GCE 不是 NCE。
- Truncated GCE 的阈值 mask 是持久选样状态，复用 `SelectionResult` 和
  global-index state；不能把 mask 藏进无状态 Loss。
- posterior 收集复用 `training/snapshots.py`，不创建只给 GCE 的 DataLoader。
- 当前 `MeanAbsoluteErrorLoss(scale=2)` 在 `q=1` 时是标准 GCE 的 2 倍；
  测试必须比较比例，不能错误要求数值完全相等。

### 4. 按顺序映射到文件和函数

1. `[已有] src/lnl_toolbox/losses/torch_losses.py`
   - `GeneralizedCrossEntropyLoss(q=0.7)`
2. `[已有] src/lnl_toolbox/plugins/builtin/catalog.py`
   - `loss/gce`
3. `[规划/共享] src/lnl_toolbox/core/result.py`
   - `SelectionResult`
4. `[规划] src/lnl_toolbox/algorithms/truncated_gce.py`
   - `TruncatedGCEState`
   - `update_probability_mask()`
   - `truncated_gce_objective()`
5. `[规划/共享] src/lnl_toolbox/training/snapshots.py`
   - 按 global index 收集当前 `p_y`。
6. `[规划] src/lnl_toolbox/training/truncated_gce_pipeline.py`
7. `[规划] configs/algorithm/truncated_gce.yaml`
8. `[扩展] tests/test_torch_losses.py`
   - 保留标准 GCE 数学测试。
9. `[规划] tests/test_truncated_gce.py`
   - 只在实现有状态截断版时新增。

### 5. 当前与规划接口

当前：

```python
GeneralizedCrossEntropyLoss(q: float = 0.7)

loss(
    logits: Tensor,       # [B,C]
    targets: Tensor,      # [B]
) -> Tensor               # [B]
```

未来截断状态：

```python
@dataclass
class TruncatedGCEState:
    global_indices: Tensor  # [N]
    keep_mask: Tensor       # bool[N]
    update_round: int

    def lookup(self, indices: Tensor) -> Tensor: ...
    def state_dict(self) -> dict: ...
```

### 6. 伪代码

标准 GCE：

```python
log_p_y = log_softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
per_sample = -expm1(q * log_p_y) / q
```

Truncated GCE：

```python
for round in range(num_rounds):
    train_with_fixed_global_mask(state.keep_mask)
    probabilities = collect_target_probabilities(model, train_loader)
    state.keep_mask = probabilities.by_global_index > k
    state.update_round += 1
```

### 7. 配置草案

```yaml
loss:
  name: gce
  q: 0.7
```

未来截断版：

```yaml
algorithm:
  name: truncated_gce
  q: 0.7
  k: 0.5
  update_interval: 1
  num_rounds: 3
```

标准 `gce` 和有状态 `truncated_gce` 必须使用不同 plugin kind/name。

### 8. Checkpoint 必需状态

标准 GCE 无训练状态，只保存 q。截断版额外保存：

- global index 与 keep mask；
- 当前 alternate round 和下一次更新 epoch；
- q、k 与 mask 生成模型 identity；
- model、optimizer、scheduler、RNG 和 manifest identity。

### 9. 最小测试

1. 普通与极低 `p_y` 下公式和梯度正确。
2. `q -> 0` 逼近 CE。
3. `q=1` 等于 `1-p_y`。
4. 与 MAE 默认 scale 的 2 倍关系明确。
5. 极端 logits 前向和反向有限。
6. 非法 q 明确失败。
7. registry 构造和 CPU/CUDA smoke 可完成。
8. 截断版在阈值下为常数且梯度为零。
9. 截断 mask 按 global index 持久化，shuffle 后不变。
10. resume 后 alternate round 和 mask 完全一致。

### 10. 论文与实现核对

- `[论文][已有]` 当前实现严格使用 `(1-p_y^q)/q`。
- `[已有]` 当前使用 `log_softmax + expm1`，避免人为 eps 截断并保持数值稳定。
- `[论文]` q=1 是 `1-p_y`，只与常用分类 MAE 成比例。
- `[论文]` Truncated GCE 包含周期性样本权重更新，不能只增加一个截断 loss 名称。
- `[待核实]` 未发现作者发布并由论文页面指向的官方实现，因此本条不把任何第三方
  GitHub 代码标为“官方代码”。

### 11. 当前未实现

- Truncated GCE
- alternate convex search 的全局 mask、Pipeline 和 checkpoint
- 论文完整实验复现

标准 GCE 已达到论文公式一致性；不能据此宣称支持 Truncated GCE。

---

## 13. VolMinNet：Provably End-to-end Label-noise Learning without Anchor Points

### 论文信息

- 编号 / 文件：`04_statistics/13_volminnet_icml2021.pdf`
- 会议与年份：ICML 2021
- 作者：Xuefeng Li, Tongliang Liu, Bo Han, Gang Niu, Masashi Sugiyama
- 官方页面：<https://proceedings.mlr.press/v139/li21l.html>
- 作者代码：<https://github.com/xuefeng-li1/Provably-end-to-end-label-noise-learning-without-anchor-points>
- 当前成熟度：L2
- 核对状态：已阅读论文；已检查作者 `main.py`、`models.py`；尚未运行
- Toolbox 归属：`Trainable NoiseModel + Objective + Pipeline`
- 不是：离线 `TransitionEstimator.estimate(snapshot)`；也不是普通 Loss

### 1. 论文实际做了什么

[论文] VolMinNet 不依赖 anchor point，而是同时学习 clean posterior 网络和
class-conditional transition matrix。论文使用列向量记法：

```text
T_paper[i,j] = P(noisy=i | clean=j)
p_noisy(column) = T_paper @ p_clean(column)
```

论文的 `T_paper` 列随机。当前 toolbox 采用行记法：

```text
T_toolbox[clean,noisy] = T_paper.T
p_noisy(row) = p_clean(row) @ T_toolbox
```

实现边界处必须显式转置，不能混用。

论文联合目标：

```text
mean CE(p_clean @ T_toolbox, noisy_target)
  + lambda * logdet(T_paper)
```

最小化 log determinant 等价于缩小 transition simplex 的体积。可识别性依赖
clean posterior “sufficiently scattered” 和矩阵的可识别性假设，而不是自动适用于
任意实例依赖噪声。

### 2. 完整调用顺序

```text
noisy batch(input, target, global index)
  -> classifier -> p_clean[B,C]
  -> TrainableTransitionModel -> T_toolbox[C,C]
  -> p_noisy = p_clean @ T_toolbox
  -> noisy CE[B]
  -> volume_regularizer = logdet(T)
  -> joint objective
  -> 同时更新 classifier 与 transition parameters
  -> epoch 边界导出 TransitionArtifact
  -> clean evaluation / checkpoint
```

### 3. 与已有条目的重叠合并

- 训练中的矩阵是有状态 `nn.Module`，不塞进无状态
  `TransitionEstimator` 协议。
- 对外导出时复用 `TransitionArtifact` 和
  `clean_to_noisy_row` 方向，供诊断或后续消费者使用。
- `TrainableTransitionModel` 与第 15 篇 T-Revision 共用“可训练全局矩阵”
  基础接口，但参数化和正则项不同。
- noisy CE 复用现有逐样本 CE 数学；模型输出先经过 T，不能把 clean posterior
  直接与 noisy target 比较。
- 不新建 Dataset、manifest、evaluator 或 checkpoint 格式。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] src/lnl_toolbox/noise/transition.py`
   - `TrainableTransitionModel` protocol/base contract
   - `export_transition_artifact()`
2. `[规划] src/lnl_toolbox/noise/volmin.py`
   - `VolMinTransitionModel`
   - `volume_regularizer()`
3. `[规划] src/lnl_toolbox/algorithms/volmin.py`
   - `VolMinAlgorithm.step()`
   - `volmin_objective()`
4. `[规划] src/lnl_toolbox/training/volmin_pipeline.py`
5. `[扩展/高冲突] src/lnl_toolbox/plugins/builtin/catalog.py`
   - 注册 `noise_model/volmin`、`algorithm/volmin`。
6. `[规划] configs/algorithm/volmin.yaml`
7. `[规划] tests/test_volmin.py`

### 5. 规划接口

```python
class TrainableTransitionModel(nn.Module):
    def matrix(self) -> Tensor:          # row-stochastic [C,C]
        ...

    def export_artifact(
        self,
        *,
        estimator: str,
        metadata: Mapping[str, Any],
    ) -> TransitionArtifact:
        ...
```

```python
def volmin_objective(
    *,
    clean_probabilities: Tensor,  # [B,C]
    noisy_targets: Tensor,        # [B]
    transition: Tensor,           # [C,C], row-stochastic
    lambda_volume: float,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    ...
```

严格约束：

- `T` 有限、非负、每行和为 1；对角占优策略必须显式。
- 使用 `torch.linalg.slogdet`；符号非正或结果非有限时明确失败。
- 不能对 `abs(det)` 静默取 log，这会掩盖非法方向/奇异矩阵。
- 导出的 artifact detach 到 CPU，并记录模型 step、parameterization 和父配置 hash。

### 6. Pipeline 伪代码

```python
model = build_classifier(config.model)
transition_model = VolMinTransitionModel(num_classes, init=config.init)

for batch in train_loader:
    clean_prob = softmax(model(batch["input"]), dim=1)
    transition = transition_model.matrix()
    noisy_prob = clean_prob @ transition

    ce = -log(noisy_prob.gather(1, batch["target"][:, None])).squeeze(1)
    sign, logabsdet = slogdet(transition)
    require(sign > 0 and finite(logabsdet))
    objective = ce.mean() + lambda_volume * logabsdet
    update_classifier_and_transition(objective)

    if artifact_schedule.at_epoch_end:
        export TransitionArtifact
```

### 7. 配置草案

```yaml
algorithm:
  name: volmin
  lambda_volume: 1.0e-4
noise_model:
  name: volmin
  parameterization: diagonal_dominant_row
  init: 2.0
loss: {name: ce}
```

该 loss 字段表示 noisy-posterior 上的 base objective；不是让 registry 普通 CE
跳过 transition。

### 8. Checkpoint 必需状态

- classifier、transition model 的参数；
- 两个 optimizer/scheduler 或参数组；
- lambda、parameterization、初始化和矩阵方向；
- 最近导出的 TransitionArtifact hash；
- epoch、global step、manifest identity 和 RNG。

### 9. 最小测试

1. 参数化始终产生非负、行和为 1 的 `[C,C]`。
2. toolbox 行方向与论文列方向转置后数值一致。
3. `p_noisy=p_clean@T` 与手算一致。
4. objective 同时给 classifier 与 transition 参数有限梯度。
5. identity transition 退化为普通 CE。
6. 非正 determinant、NaN 或奇异矩阵明确失败。
7. Artifact 导出/加载后方向、数值和 hash 一致。
8. checkpoint roundtrip 后两组参数与 optimizer state 一致。
9. clean truth 不进入 objective。
10. CPU 单步、CUDA smoke 与 resume 可完成。

### 10. 论文与作者代码核对

- `[论文][代码]` 作者代码联合优化 noisy CE 与 `lambda * logdet(T)`。
- `[代码]` `models.py::sig_t` 用 sigmoid 生成非对角元素，加 identity 后按一维
  归一化，形成对角占优矩阵。
- `[代码]` `main.py` 计算 `out = clean @ t`，与 toolbox 行方向一致；
  论文正文则用列方向描述。
- `[代码]` 作者使用 `t.slogdet().logabsdet`，没有检查 determinant sign；
  toolbox 必须额外校验。
- `[差异]` 作者仓库名称与论文一致，但 PMLR 页面未直接提供 Code 链接；
  本指南把它标为作者代码，不把工程细节提升为论文定理。

### 11. 当前未实现

- TrainableTransitionModel 基础协议
- VolMin transition 参数化和 volume objective
- 联合训练 Pipeline、artifact lineage、配置和测试
- 论文结果复现

当前离线 `TransitionArtifact` 不能替代 VolMinNet 的联合可训练矩阵。

---

## 14. Learning with Noisy Labels：Unbiased Risk and Label-dependent Costs

### 论文信息

- 编号 / 文件：`04_statistics/14_learning_with_noisy_labels_neurips2013.pdf`
- 会议与年份：NeurIPS 2013
- 作者：Nagarajan Natarajan, Inderjit S. Dhillon, Pradeep Ravikumar,
  Ambuj Tewari
- 论文页面：<https://papers.nips.cc/paper/2013/hash/3871bd64012152bfb53fdf04b401193f-Abstract.html>
- 官方代码：论文与作者页面未提供可核实的作者官方实现
- 当前成熟度：L1
- 核对状态：已阅读论文；无官方代码可交叉检查
- Toolbox 归属：`Binary Noise Rates + RiskCorrector`
- 不是：通用多分类 transition correction；论文理论对象是二分类 `y∈{-1,+1}`

### 1. 论文实际做了什么

论文假设 class-conditional binary noise：

```text
rho_plus  = P(noisy=-1 | clean=+1)
rho_minus = P(noisy=+1 | clean=-1)
rho_plus + rho_minus < 1
```

第一种方法为任意有界 base loss 构造无偏代理：

```text
corrected_loss(t, y)
  = ((1 - rho_{-y}) * loss(t, y)
     - rho_y * loss(t, -y))
    / (1 - rho_plus - rho_minus)
```

在 noisy label 上取期望等于 clean loss。该 corrected loss 可以为负；即使 base loss
非负，也不能裁剪。

第二种方法把问题转换为 label-dependent cost。论文 Theorem 9：

```text
alpha_star = (1 - rho_plus + rho_minus) / 2
A_rho      = (1 - rho_plus - rho_minus) / 2
```

对 margin loss：

```text
loss_alpha(t,y)
  = (1-alpha) * 1(y=+1) * loss(t)
    + alpha * 1(y=-1) * loss(-t)
```

可使用 weighted logistic regression / biased SVM 等凸优化方法。

### 2. 完整调用顺序

```text
KnownTransition[2,2] or explicit known rates
  -> BinaryNoiseRates(rho_plus, rho_minus)
  -> binary model margin[B]
  -> base loss for observed and opposite labels
  -> UnbiasedBinaryRiskCorrector
  -> corrected per-sample risk[B]
  -> mean + backward
```

或：

```text
BinaryNoiseRates
  -> alpha_star
  -> LabelDependentCostCorrector
  -> weighted binary surrogate[B]
  -> mean + backward
```

### 3. 与已有条目的重叠合并

- 与第 10 篇 Patrini Backward correction 共用
  `algorithms/transition_risk.py::RiskCorrector`，不单独创建一个普通 loss。
- `BinaryNoiseRates` 可从 toolbox 行方向 `KnownTransition[2,2]` 精确转换：
  `rho_plus=T[+1,-1]`、`rho_minus=T[-1,+1]`。
- 不修改现有多分类 Loss 输入协议；binary corrector 负责把 `[B,2]` logits 转成
  margin，或接收显式 `[B]` margin。
- 论文中的调参实验使用 cross-validation，不代表 toolbox 可以读取 clean validation
  选择噪声率；生产路径只能使用 known rates 或显式 noisy-validation 策略。
- 后续第 18 篇 importance reweighting 若需要样本权重，复用 `WeightResult`，
  不把本论文的 label-dependent class cost 混成按样本历史状态。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] src/lnl_toolbox/noise/transition.py`
   - `BinaryNoiseRates`
   - `BinaryNoiseRates.from_transition(provider)`
2. `[规划/共享] src/lnl_toolbox/algorithms/transition_risk.py`
   - `UnbiasedBinaryRiskCorrector`
   - `LabelDependentCostCorrector`
   - `binary_margin_from_logits()`
3. `[扩展/高冲突] src/lnl_toolbox/plugins/builtin/catalog.py`
   - 注册 `risk_corrector/natarajan_unbiased|label_dependent_cost`。
4. `[规划] configs/algorithm/natarajan_binary.yaml`
5. `[规划/共享] tests/test_transition_risk.py`
   - 与 Forward/Backward 放在同一风险校正测试文件。

### 5. 规划接口

```python
@dataclass(frozen=True)
class BinaryNoiseRates:
    rho_plus: float
    rho_minus: float

    def __post_init__(self) -> None:
        # finite, each in [0,1), sum < 1
        ...
```

```python
def unbiased_binary_risk(
    *,
    margin: Tensor,           # [B]
    noisy_targets: Tensor,    # {-1,+1} or validated 0/1 mapping
    rates: BinaryNoiseRates,
    base_margin_loss: Callable,
) -> Tensor:                  # [B], may be negative
    ...
```

严格约束：

- target 编码必须显式，不允许猜测 0/1 中哪个类是 `+1`。
- denominator 过小按阈值失败；不能用 eps 偷偷改变理论公式。
- corrected risk 允许为负，但必须有限。
- 该组件只支持 2 类；多分类调用明确拒绝。

### 6. 伪代码

```python
def unbiased_binary_risk(margin, noisy_y, rates, base_loss):
    rho_y = where(noisy_y == +1, rates.rho_plus, rates.rho_minus)
    rho_minus_y = where(noisy_y == +1, rates.rho_minus, rates.rho_plus)
    observed = base_loss(margin, noisy_y)
    opposite = base_loss(margin, -noisy_y)
    return (
        (1 - rho_minus_y) * observed - rho_y * opposite
    ) / (1 - rates.rho_plus - rates.rho_minus)
```

### 7. 配置草案

```yaml
algorithm:
  name: supervised
risk_corrector:
  name: natarajan_unbiased
  positive_class_index: 1
transition:
  source: known_manifest
loss:
  name: binary_logistic
```

第一版不应在 CIFAR-10 多分类配置中暴露该方法。

### 8. Checkpoint 必需状态

Corrector 无训练状态，但 resolved checkpoint 必须保存：

- rho_plus、rho_minus 及其来源 artifact hash；
- positive class mapping；
- base margin loss 配置；
- corrector 类型和 denominator policy。

模型、optimizer、scheduler 等继续由通用 runner 保存。

### 9. 最小测试

1. 枚举 clean/noisy 二分类分布验证无偏期望。
2. `rho_plus=rho_minus=0` 退化为 base loss。
3. corrected risk 可为负且不被 clamp。
4. 非法 rate、和大于等于 1、分母过小明确失败。
5. 0/1 与 ±1 映射在指定 positive class 后一致。
6. `BinaryNoiseRates.from_transition()` 方向正确。
7. label-dependent alpha 与论文公式一致。
8. 多分类输入明确拒绝。
9. CPU 单步和梯度有限。
10. checkpoint/resolved config 保留 rates 和映射。

### 10. 论文与实现核对

- `[论文]` 无偏代理适用于任意有界 loss，但优化难度取决于校正后函数性质。
- `[论文]` 校正风险可以失去非负性和凸性；论文另给出二阶导对称条件。
- `[论文]` label-dependent cost 路径在合适凸、classification-calibrated margin
  loss 下保持高效。
- `[论文]` 实验中噪声率可经交叉验证调节，这是实验选择，不是算法获得 clean truth
  的许可。
- `[待核实]` 未发现作者官方代码，因此本条只依据论文公式给出伪代码。

### 11. 当前未实现

- BinaryNoiseRates
- 两种 binary RiskCorrector
- binary margin / target mapping
- 配置和风险校正测试
- 论文结果复现

当前 toolbox 的多分类 CE/GCE/APL 不能视为本论文的实现。

---

## 15. T-Revision：Are Anchor Points Really Indispensable in Label-Noise Learning?

### 论文信息

- 编号 / 文件：`04_statistics/15_t_revision_neurips2019.pdf`
- 会议与年份：NeurIPS 2019
- 作者：Xiaobo Xia, Tongliang Liu, Nannan Wang, Bo Han, Chen Gong,
  Gang Niu, Masashi Sugiyama
- 论文页面：<https://papers.nips.cc/paper/2019/hash/9308b0d6e5898366a4a986bc33f3d3e7-Abstract.html>
- 官方代码：<https://github.com/xiaoboxia/T-Revision>
- 当前成熟度：L2
- 核对状态：已阅读论文；已检查官方 `main.py`、`loss.py` 和模型 revision 层；
  尚未运行
- Toolbox 归属：`Anchor Estimator + Trainable NoiseModel + WeightProvider + Pipeline`
- 不是：普通无状态 TransitionEstimator；也不是只把 Anchor 矩阵保存一次

### 1. 论文实际做了什么

论文与 toolbox 使用相同的行方向：

```text
T[i,j] = P(noisy=j | clean=i)
p_noisy(column) = T.T @ p_clean(column)
p_noisy(row) = p_clean(row) @ T
```

[论文] 先使用 high noisy-class posterior 的 pseudo-anchor 初始化 `T_hat`，再加入
可学习 slack：

```text
T_revised = T_hat + DeltaT
```

并与分类器共同训练。样本权重来自 importance ratio：

```text
weight_i
  = p_clean(noisy_target_i | x_i)
    / p_noisy(noisy_target_i | x_i)

p_noisy = p_clean @ T_revised
objective_i = weight_i * base_loss(logits_i, noisy_target_i)
```

论文流程不是一步：

```text
阶段 A：noisy posterior warm-up
阶段 B：pseudo-anchor 初始化 T_hat
阶段 C：固定 T_hat 的 reweight training
阶段 D：初始化 DeltaT=0，联合 revision training
```

### 2. 完整调用顺序

```text
noisy train data
  -> warm-up model
  -> PosteriorSnapshot
  -> AnchorTransitionEstimator
  -> initial TransitionArtifact(T_hat)
  -> fixed-T reweight stage
  -> RevisedTransitionModel(T_hat, DeltaT)
  -> revised importance weights[B]
  -> weighted loss + joint update
  -> epoch artifact snapshots
  -> noisy validation for stage/model choice
  -> clean test evaluation
```

### 3. 与已有条目的重叠合并

- pseudo-anchor 初始化直接复用现有
  `AnchorTransitionEstimator`，不复制 `estimate_transition()`。
- 初始和各 revision epoch 的矩阵都导出为 `TransitionArtifact`，metadata 记录
  parent artifact hash。
- 与第 13 篇 VolMinNet 共用 `TrainableTransitionModel` 外部协议，但使用不同
  parameterization、objective 和 Pipeline。
- importance ratio 输出复用 `WeightResult`；第 18 篇出现时继续扩展同一
  `WeightProvider`，不再新建样本权重协议。
- base loss 继续返回 `[B]`；T-Revision 只在外层乘 detached 或按论文指定梯度
  语义的 weight。
- clean validation/test 与 noisy validation 必须区分；算法选 epoch 的验证集不能
  静默读取 clean training truth。

### 4. 按顺序映射到文件和函数

1. `[已有] src/lnl_toolbox/noise/estimators.py`
   - `PosteriorSnapshot`
   - `AnchorTransitionEstimator`
2. `[已有] src/lnl_toolbox/noise/transition.py`
   - `TransitionArtifact`
3. `[扩展/共享] src/lnl_toolbox/noise/transition.py`
   - `TrainableTransitionModel` contract
4. `[规划] src/lnl_toolbox/noise/t_revision.py`
   - `RevisedTransitionModel`
   - `revised_matrix()`
   - `export_revision_artifact()`
5. `[扩展/共享] src/lnl_toolbox/core/result.py`
   - `WeightResult`
6. `[规划] src/lnl_toolbox/algorithms/t_revision.py`
   - `TransitionImportanceWeightProvider`
   - `TRevisionAlgorithm`
7. `[规划] src/lnl_toolbox/training/t_revision_pipeline.py`
   - 显式四阶段状态机与 resume。
8. `[扩展/高冲突] src/lnl_toolbox/plugins/builtin/catalog.py`
   - 注册 `noise_model/t_revision`、`weight_provider/transition_importance`、
     `pipeline/t_revision`。
9. `[规划] configs/algorithm/t_revision.yaml`
10. `[规划] tests/test_t_revision.py`

### 5. 规划接口

```python
class RevisedTransitionModel(TrainableTransitionModel):
    def __init__(
        self,
        initial: TransitionArtifact,
        parameterization: str = "row_simplex_slack",
    ) -> None: ...

    def matrix(self) -> Tensor: ...  # valid [C,C], clean_to_noisy_row
```

```python
class TransitionImportanceWeightProvider:
    def weights(
        self,
        *,
        clean_probabilities: Tensor,  # [B,C]
        noisy_targets: Tensor,        # [B]
        global_indices: Tensor,       # [B]
        transition: Tensor,           # [C,C]
    ) -> WeightResult:
        ...
```

严格约束：

- revised matrix 每次前向都必须有限、非负、行和为 1。
- 分母 `p_noisy_y` 必须严格正且有限；安全策略显式配置并记录。
- `WeightResult` metadata 记录 transition artifact/step identity。
- 是否 detach ratio 是算法语义，必须配置/测试；不能用旧 `Variable` 隐式决定。
- stage transition 只能发生在 epoch 边界并写入 checkpoint。

### 6. Pipeline 伪代码

```python
warmup_model = pretrain_noisy_classifier(train_loader)
snapshot = collect_posterior_snapshot(warmup_model, train_loader)
initial_T = AnchorTransitionEstimator().estimate(snapshot)

model = initialize_reweight_model(warmup_model)
train_fixed_transition_stage(model, initial_T)

noise_model = RevisedTransitionModel(initial_T)
for epoch in revision_stage:
    for batch in train_loader:
        logits = model(batch["input"])
        p_clean = softmax(logits, dim=1)
        transition = noise_model.matrix()
        result = weight_provider.weights(
            clean_probabilities=p_clean,
            noisy_targets=batch["target"],
            global_indices=batch["index"],
            transition=transition,
        )
        base = loss_fn(logits, batch["target"])  # [B]
        objective = mean(result.weights * base)
        update_model_and_revision(objective)

    export revision TransitionArtifact(parent_hash=initial_T.artifact_hash)
    evaluate on explicitly configured noisy validation
```

### 7. 配置草案

```yaml
pipeline:
  name: t_revision
  warmup_epochs: 20
  reweight_epochs: 20
  revision_epochs: 20
transition:
  estimator: {name: anchor}
  revision:
    parameterization: row_simplex_slack
weight_provider:
  name: transition_importance
  denominator_floor: 1.0e-8
  detach: false
loss: {name: ce}
```

若配置没有独立 noisy validation，必须采用预先规定的 epoch 数或明确策略，不能回退
使用 clean test 选模型。

### 8. Checkpoint 必需状态

- 当前阶段及阶段内 epoch/global step；
- warm-up、reweight/revision 模型和对应 optimizer/scheduler；
- initial TransitionArtifact hash；
- revision raw parameters、parameterization 和当前导出 artifact hash；
- weight-provider 分母策略与 detach 语义；
- noisy validation identity、manifest identity、RNG；
- best stage metric 和模型选择规则。

### 9. 最小测试

1. Anchor 初始化复用现有 estimator，artifact hash 稳定。
2. `p_noisy=p_clean@T` 方向与手算一致。
3. importance ratio 与论文公式一致。
4. identity T 时 weight 为 1，退化为 base loss。
5. revised matrix 始终有效；非法 slack 不被静默接受。
6. classifier 与 revision 参数均获得预期梯度。
7. denominator 接近零按显式策略处理且保持有限。
8. WeightResult global index 与 batch 对齐。
9. stage checkpoint/resume 后阶段、模型、slack、optimizer 和 artifact lineage 一致。
10. clean truth 不进入 estimator、weight provider 或训练阶段。
11. fixed-T 与 revision 两条路径分别 CPU 单步通过。
12. CUDA 多阶段 smoke 和 resume 可完成。

### 10. 论文与官方代码核对

- `[论文][代码]` 都先用高 noisy posterior 样本初始化 T，再进行 reweight 和
  slack revision。
- `[论文][代码]` importance ratio 为当前 clean posterior 的 observed-label
  概率除以经 T 映射后的 noisy posterior 概率。
- `[代码]` 官方 `loss.py` 的 v2 实现向量化了 ratio 和逐样本 CE。
- `[代码]` 官方 revision 直接计算 `T + correction`。
- `[差异]` 论文正文建议把负元素投影并行归一化，但脚注说明实验调 slack 时未把
  `T+DeltaT` 推回有效转移矩阵；官方代码也未严格保证有效性。toolbox 必须选择
  并记录安全的 row-simplex parameterization，不能复制无约束矩阵。
- `[差异]` 官方代码用旧 `Variable(beta, requires_grad=True)` 包装 ratio，
  梯度语义含混；toolbox 必须以显式 detach 配置和 autograd 测试定义行为。
- `[代码]` 官方训练脚本曾用真实 T 计算 estimate error；toolbox 只能在合成实验的
  evaluator 中使用该信息。

### 11. 当前未实现

- RevisedTransitionModel
- transition importance WeightProvider
- 四阶段 T-Revision Pipeline 与 noisy-validation 选择
- revision artifact lineage、checkpoint、配置和测试
- 论文结果复现

当前 Anchor estimator 是 T-Revision 的基础组件，不代表 T-Revision 已实现。

---

## 16. Dual-T：Reducing Estimation Error for Transition Matrix in Label-noise Learning

### 论文信息

- 编号 / 文件：`04_statistics/16_dual_t_neurips2020.pdf`
- 会议与年份：NeurIPS 2020
- 作者：Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Jiankang Deng,
  Gang Niu, Masashi Sugiyama
- 论文页面：<https://papers.nips.cc/paper/2020/hash/512c5cad6c37edb98ae91c8a76c3a291-Abstract.html>
- 官方代码：未发现论文作者发布的官方实现
- 当前成熟度：L3（论文原式实现并通过数学/协议测试；未发现官方代码）
- Toolbox 归属：`TransitionEstimator`
- 不是：RiskCorrector、Loss 或联合训练 NoiseModel

### 1. 论文实际做了什么

[论文] Dual-T 引入中间标签 `Y'`，把直接估计 clean→noisy 的困难矩阵分解为：

```text
T_club[i,l]  = P(intermediate=l | clean=i)
T_spade[l,j] = P(noisy=j | intermediate=l)
T[i,j]       = sum_l T_club[i,l] * T_spade[l,j]
```

论文用已训练模型的 `P(noisy|x)` 定义 `P(intermediate|x)`。`T_club` 仍用
anchor 方法估计；中间标签取 posterior 的 `argmax`，`T_spade` 用中间标签与
observed noisy label 的频数估计。按 toolbox 的行向量约定，最终为：

```text
T = T_club @ T_spade
p_noisy = p_clean @ T
```

### 2. 完整调用顺序

```text
noisy Dataset -> warm-up model -> PosteriorSnapshot
-> anchor estimate T_club
-> argmax intermediate labels
-> count intermediate-to-noisy T_spade
-> compose and validate T
-> TransitionArtifact(matrix=T, factors, anchors, source hash)
```

### 3. 与已有条目的重叠合并

- warm-up 和 posterior 收集复用 `training/snapshots.py`。
- `T_club` 复用现有 `AnchorTransitionEstimator`，不复制 anchor 代码。
- 最终结果复用 `TransitionArtifact`；两个因子写入 metadata，不另建 artifact。
- 后续 Forward/Backward、T-Revision 是消费者，不属于本 estimator。

### 4. 按顺序映射到文件和函数

1. `[已有] noise/estimators.py::PosteriorSnapshot`
2. `[已有] training/snapshots.py::collect_posterior_snapshot()`
3. `[已有] noise/estimators.py::AnchorTransitionEstimator.estimate()`
4. `[已有] noise/estimators.py::DualTransitionEstimator.estimate()`
5. `[已有] noise/transition.py::TransitionArtifact`
6. `[已有/高冲突] plugins/builtin/catalog.py` 注册 `transition_estimator/dual_t`
7. `[已有] tests/test_transition_estimators.py` 的 collector 与 Dual-T 数学/协议测试

### 5. 规划接口

```python
class DualTransitionEstimator:
    def estimate(self, snapshot: PosteriorSnapshot) -> TransitionArtifact:
        """Return clean-to-noisy row-stochastic T and both factors."""
```

并列 `argmax` 与 anchor 均按最小 global index 决胜；任何中间类别没有样本时明确
失败。不得静默加平滑或归一化，除非配置显式声明并写入 metadata。

### 6. Pipeline 伪代码

```python
snapshot = collect_posterior_snapshot(warmup_model, train_loader)
t_club = AnchorTransitionEstimator().estimate(snapshot)
intermediate = snapshot.noisy_probabilities.argmax(axis=1)
t_spade = count_conditional(
    source=intermediate,
    target=snapshot.noisy_targets,
    classes=snapshot.num_classes,
)
t = t_club.matrix @ t_spade
return TransitionArtifact(t, estimator="dual_t", metadata={
    "t_club": t_club.matrix, "t_spade": t_spade,
    "anchor_indices": t_club.metadata["anchor_indices"],
})
```

### 7. 配置草案

```yaml
transition:
  estimator: {name: dual_t}
```

### 8. Checkpoint 必需状态

Estimator 无训练状态；run artifact 保存 snapshot hash、两个因子、anchor global
indices、合成方向、最终 artifact hash 和 warm-up model identity。

### 9. 最小测试

1. 理想 posterior 与频数能恢复手算 T。
2. `T_club @ T_spade` 方向正确。
3. 输入重排不改变矩阵或 anchor global index。
4. 空中间类别、重复 index、非法概率明确失败。
5. 保存加载后 factors 与 hash 一致。
6. identity factor 退化到另一 factor。

### 10. 论文与代码核对

- `[论文]` Algorithm 1 明确给出 anchor、hard intermediate label 和频数估计。
- `[论文]` 正文采用列向量记号，矩阵书写次序与 toolbox 行向量表示不同；本条
  用条件概率下标推导并固定为 `T_club @ T_spade`。
- `[待核实]` 未发现作者官方代码，因此只依据论文公式和算法给出伪代码。

### 11. 当前实现状态与边界

- 已实现：共享 posterior collector、DualTransitionEstimator、registry、
  `T_club/T_spade/counts/anchors` artifact metadata 及数学/协议测试。
- 未实现：warm-up trainer、公共 runner/CLI 接入、Forward/Backward 等消费者。
- 当前可由代码显式调用 `collect_posterior_snapshot()` 后构造 `dual_t` estimator；
  尚不能仅靠实验 YAML 自动执行完整两阶段论文流程。

---

## 17. MC-LDCE：Multi-class Label Noise Learning via Loss Decomposition and Centroid Estimation

### 论文信息

- 编号 / 文件：`04_statistics/17_mc_ldce_sdm2022.pdf`
- 会议与年份：SDM 2022
- 作者：Yongliang Ding, Tao Zhou, Chuang Zhang, Yijing Luo, Juan Tang, Chen Gong
- 论文页面：<https://arxiv.org/abs/2203.10858>
- 官方代码：未发现论文作者发布的官方实现
- 当前成熟度：L1
- Toolbox 归属：`StatisticEstimator + global RiskCorrector + Pipeline`
- 不是：可替换 CE 的逐样本 Loss

### 1. 论文实际做了什么

[论文] 对线性分类器 `h(x;W)=W^T x` 的平方损失分解后，只有 clean centroid
依赖标签：

```text
mu_noisy = mean(x_i @ one_hot(noisy_i)^T)          # [D,C]
M = sum_i pi_i * sum_j T[i,j] * K[i->j]^T          # [C,C]
mu_clean_hat = mu_noisy @ pinv(M)
risk = 1 + mean(x^T W W^T x) - 2 trace(W^T mu_clean_hat)
```

clean class prior `pi` 由 `pi_noisy = pi_clean @ T` 求得。论文算法先用 VolMinNet
估计 T，再恢复 centroid 并优化分解后的全局风险。

### 2. 完整调用顺序

```text
noisy data -> feature snapshot
-> TransitionArtifact (paper uses VolMinNet)
-> solve clean class prior -> build M
-> estimate clean centroid -> StatisticArtifact
-> MC-LDCE global objective -> optimize classifier/model
```

### 3. 与已有条目的重叠合并

- T 复用 `TransitionArtifact`，不内嵌 VolMinNet。
- feature 收集与第 19、20、26 篇共用 `FeatureSnapshot`。
- centroid 与 per-class statistics 统一放入 `StatisticArtifact`。
- 风险是全局 batch/statistic objective，不能伪装为 `[B]` loss。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] training/snapshots.py::collect_feature_snapshot()`
2. `[规划/共享] noise/statistics.py::StatisticArtifact`
3. `[规划] noise/statistics.py::estimate_clean_priors()`
4. `[规划] noise/statistics.py::build_label_imputation_matrix()`
5. `[规划] noise/statistics.py::estimate_clean_centroid()`
6. `[规划] algorithms/statistic_risk.py::mc_ldce_objective()`
7. `[规划] training/statistic_pipeline.py::run_mc_ldce_experiment()`
8. `[规划] tests/test_statistic_methods.py`

### 5. 规划接口

```python
@dataclass(frozen=True)
class StatisticArtifact:
    kind: str
    values: Mapping[str, np.ndarray]
    global_indices_hash: str
    feature_snapshot_hash: str
    transition_artifact_hash: str | None
    metadata: Mapping[str, Any]

def estimate_clean_centroid(
    snapshot: FeatureSnapshot,
    transition: TransitionArtifact,
) -> StatisticArtifact: ...
```

伪逆容差、矩阵秩和 condition number 必须记录；不能用默认 `pinv` 掩盖不可识别。

### 6. Pipeline 伪代码

```python
features = collect_feature_snapshot(model, train_loader, layer)
t = load_transition_artifact(config.transition)
pi_clean = solve_simplex(features.noisy_class_prior, t.matrix)
m = build_label_imputation_matrix(pi_clean, t.matrix)
require_identifiable(m, rank_policy)
mu_clean = features.noisy_centroid @ pinv(m, rcond=config.rcond)
stats = StatisticArtifact(kind="clean_centroid", values={"mu": mu_clean})
train_with_global_objective(model, stats, mc_ldce_objective)
```

### 7. 配置草案

```yaml
pipeline: {name: mc_ldce}
transition: {artifact: path/to/transition.npz}
statistics: {feature_layer: classifier_input, rcond: 1.0e-8}
```

### 8. Checkpoint 必需状态

feature/model identity、T hash、M、rank/rcond、clean-prior estimate、centroid artifact
hash，以及模型/optimizer/scheduler 和当前 stage。

### 9. 最小测试

1. 无噪声 T 恢复 noisy centroid 本身。
2. 小型可逆例与手算一致。
3. 不可识别 M 按策略失败。
4. feature/index 错位与 T hash 不匹配失败。
5. objective 与显式平方损失分解一致。
6. checkpoint/resume 保持统计量身份。

### 10. 论文与代码核对

- `[论文]` Algorithm 1 明确先估计 T、先验和 centroid，再优化模型。
- `[论文]` 深层模型使用特征表示时，统计量绑定具体 feature layer。
- `[待核实]` 未发现作者官方代码；`pinv`、rank policy 与 artifact 是工程推断。

### 11. 当前未实现

- FeatureSnapshot、StatisticArtifact、centroid estimator 与 MC-LDCE objective/Pipeline

---

## 18. Importance Reweighting：Classification with Noisy Labels by Importance Reweighting

### 论文信息

- 编号 / 文件：`04_statistics/18_importance_reweighting_tpami2016.pdf`
- 期刊与年份：IEEE TPAMI 2016
- 作者：Tongliang Liu, Dacheng Tao
- 论文页面：<https://arxiv.org/abs/1411.7718>
- 官方代码：未发现论文作者发布的官方实现
- 当前成熟度：L1
- Toolbox 归属：`BinaryNoiseRateEstimator + WeightProvider`
- 不是：多分类通用 loss

### 1. 论文实际做了什么

[论文] 对二分类 asymmetric random classification noise：

```text
rho_y = P(noisy=-y | clean=y),  rho_plus + rho_minus < 1
p = P(noisy_label | x)
beta(x,noisy_y)
  = [p - rho_{-noisy_y}] / [(1-rho_plus-rho_minus) * p]
risk = mean(beta_i * base_loss(model(x_i), noisy_y_i))
```

论文还用 `min_x P(noisy=y|x)` 估计噪声率，并讨论概率分类器、核密度和 KLIEP
密度比估计。该结论依赖 mutual irreducibility/anchor 条件。

### 2. 完整调用顺序

```text
binary noisy data -> estimate noisy posterior
-> estimate/receive BinaryNoiseRates
-> per-batch posterior lookup -> importance weights
-> base loss[B] * WeightResult[B] -> mean -> update
```

### 3. 与已有条目的重叠合并

- 二元 rates 复用第 14 篇 `BinaryNoiseRates`。
- posterior 复用 `PosteriorSnapshot`。
- 输出复用 `WeightResult/WeightProvider`；不创建新的 weight 协议。
- 第一版只消费已给 rates/posterior；KLIEP 是可选 estimator，不塞进 provider。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] training/snapshots.py::collect_posterior_snapshot()`
2. `[规划/共享] algorithms/transition_risk.py::BinaryNoiseRates`
3. `[规划] noise/estimators.py::MinimumPosteriorNoiseRateEstimator`
4. `[扩展/共享] core/result.py::WeightResult`
5. `[规划] algorithms/reweighting.py::ImportanceWeightProvider`
6. `[规划] tests/test_reweighting.py`

### 5. 规划接口

```python
class ImportanceWeightProvider:
    def weights(
        self,
        noisy_label_probability: Tensor,  # [B]
        noisy_targets: Tensor,            # explicit {-1,+1}
        global_indices: Tensor,
        rates: BinaryNoiseRates,
    ) -> WeightResult: ...
```

`p=0` 时按论文定义 weight=0；其他无效分母必须失败。默认不 clamp 负 weight，
而应验证理论条件；若提供裁剪变体，必须另名并标明非论文原式。

### 6. Pipeline 伪代码

```python
p = snapshot.lookup(batch["index"]).gather(batch["target"])
numerator = p - opposite_rate(batch["target"], rates)
denominator = (1 - rates.sum) * p
weights = where(p == 0, 0, numerator / denominator)
objective = mean(weights * base_loss(logits, batch["target"]))
```

### 7. 配置草案

```yaml
weight_provider:
  name: binary_importance
  rates: {rho_plus: 0.2, rho_minus: 0.1}
```

### 8. Checkpoint 必需状态

rates 及来源、posterior snapshot hash、target 编码、weight 公式版本和模型训练状态。

### 9. 最小测试

1. 零噪声时 weight=1。
2. 手算 posterior/rates 与 weight 一致。
3. 加权 noisy risk 的枚举期望等于 clean risk。
4. `p=0`、非法 rates 与多分类输入行为明确。
5. global-index lookup 不受 shuffle 影响。
6. CPU 梯度有限。

### 10. 论文与代码核对

- `[论文]` 方法严格限定为二分类 class-conditional noise。
- `[论文]` rates 的最小 posterior 估计需要额外可识别性条件。
- `[待核实]` 未发现作者官方代码，故只保留论文原式与显式伪代码。

### 11. 当前未实现

- rates estimator、ImportanceWeightProvider、二分类训练适配与测试

---

## 19. CWD：Class-Wise Denoising for Robust Learning under Label Noise

### 论文信息

- 编号 / 文件：`04_statistics/19_cwd_tpami2022.pdf`
- 期刊与年份：IEEE TPAMI 2022
- 作者：Chen Gong, Yongliang Ding, Bo Han, Gang Niu, Jian Yang,
  Jane You, Dacheng Tao, Masashi Sugiyama
- 论文页面：<https://doi.org/10.1109/TPAMI.2022.3170950>
- 官方代码：未发现论文作者发布的官方实现
- 当前成熟度：L1
- Toolbox 归属：`StatisticEstimator + global RiskCorrector`

### 1. 论文实际做了什么

[论文] CWD 把一类 margin loss 的风险写成标签无关项和 clean centroid 项：

```text
risk = mean(g(h(x;w))) + Q * <w, mu_clean>
mu_clean = mean(y_i * x_i)
```

二分类时，论文分别假设 noisy positive 或 noisy negative 可靠，构造两个虚拟
auxiliary dataset 的 centroid，再合成：

```text
mu_clean_hat =
  [1/(1-2*pi_pos*eta_pos) + 1/(1-2*pi_neg*eta_neg) - 1] * mu_noisy
```

多分类扩展为每类一个虚拟 auxiliary set，并用 imputation/coefficient matrices
恢复 centroid。虚拟数据集只用于推导，不应在 toolbox 中实际复制样本。

### 2. 完整调用顺序

```text
noisy data -> FeatureSnapshot -> noisy centroid/class priors
-> noise rates or TransitionArtifact
-> class-wise coefficient system
-> clean StatisticArtifact
-> decomposed global risk -> optimize
```

### 3. 与已有条目的重叠合并

- FeatureSnapshot、StatisticArtifact、prior solver 与第 17/20 篇共用。
- 二分类 rates 复用第 14/18 篇。
- 不实际创建 virtual Dataset，也不把 CWD 放入普通 loss registry。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] training/snapshots.py::collect_feature_snapshot()`
2. `[规划/共享] noise/statistics.py::StatisticArtifact`
3. `[扩展/共享] noise/statistics.py::estimate_clean_priors()`
4. `[规划] noise/statistics.py::estimate_cwd_centroid()`
5. `[规划] algorithms/statistic_risk.py::cwd_objective()`
6. `[扩展] training/statistic_pipeline.py::run_cwd_experiment()`
7. `[扩展] tests/test_statistic_methods.py`

### 5. 规划接口

```python
def estimate_cwd_centroid(
    snapshot: FeatureSnapshot,
    noise: BinaryNoiseRates | TransitionArtifact,
    *,
    variant: Literal["binary", "multiclass"],
) -> StatisticArtifact: ...
```

### 6. Pipeline 伪代码

```python
features = collect_feature_snapshot(model, train_loader, layer)
stats = estimate_cwd_centroid(features, configured_noise_information)
for batch in train_loader:
    representation, margin = model.forward_with_features(batch["input"])
    objective = cwd_objective(representation, margin, stats)
    optimizer_step(objective)
```

### 7. 配置草案

```yaml
pipeline: {name: cwd}
statistics: {feature_layer: classifier_input, variant: multiclass}
transition: {artifact: path/to/transition.npz}
```

### 8. Checkpoint 必需状态

feature layer/model hash、noise information identity、class priors、coefficient matrices、
centroid artifact、loss decomposition variant 和训练状态。

### 9. 最小测试

1. 二分类公式与枚举 centroid 一致。
2. 无噪声时退化为原始风险。
3. 多分类 coefficient system 与手算一致。
4. 奇异分母/矩阵明确失败。
5. 不生成 virtual dataset。
6. global objective 梯度与直接实现一致。

### 10. 论文与代码核对

- `[论文]` 方法只适用于论文列出的可分解 loss/模型条件，不是任意 CE 替换。
- `[论文]` 合成实验使用已知 flip rates；toolbox 必须从显式 provider 获得。
- `[待核实]` 未发现作者官方代码。

### 11. 当前未实现

- CWD centroid estimator、global objective、Pipeline 与测试

---

## 20. PCSE：Estimating Per-Class Statistics for Label Noise Learning

### 论文信息

- 编号 / 文件：`04_statistics/20_pcse_tpami2024.pdf`
- 期刊与年份：IEEE TPAMI 2024（正式卷期 2025）
- 作者：Wenshui Luo, Shuo Chen, Tongliang Liu, Bo Han, Gang Niu,
  Masashi Sugiyama, Dacheng Tao, Chen Gong
- 官方代码：<https://github.com/randydkx/PCSE>
- 当前成熟度：L2；已阅读论文并检查官方 `PCSE.py` 结构，尚未运行
- Toolbox 归属：`StatisticEstimator + post-processing Pipeline`
- 不是：训练阶段的普通 Loss 或 Selector

### 1. 论文实际做了什么

[论文] PCSE 在一个预训练 noisy model 的若干 feature layer 上，按 noisy class
收集一阶、二阶矩：

```text
noisy mean_j   = E[phi(x) | noisy=j]
noisy second_j = E[phi(x) phi(x)^T | noisy=j]
```

利用 `pi_noisy = pi_clean @ T`、转移矩阵和 MC-LDCE 的 coefficient matrix M，
恢复 clean class 的 mean、second moment 与 covariance。随后建立 GDA posterior；
多个 feature layer 的 posterior 用非负且和为 1 的 ensemble weights 聚合，权重由
noisy validation 的 NLL 学习。最终以该生成式分类器替换原 classifier 做推理。

### 2. 完整调用顺序

```text
pretrained noisy model -> TransitionArtifact
-> selected-layer FeatureSnapshots
-> noisy per-class first/second moments
-> clean priors + coefficient matrix
-> clean per-class means/covariances
-> layer-wise GDA posterior
-> noisy-validation ensemble weights
-> post-processed classifier evaluation
```

### 3. 与已有条目的重叠合并

- T 复用 `TransitionArtifact`；估计方式由配置决定，论文实验用 VolMinNet。
- snapshot、prior solver、M 和 StatisticArtifact 与第 17/19 篇共用。
- PCSE 特有内容仅是 per-layer per-class statistics、GDA 和 ensemble Pipeline。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] training/snapshots.py::collect_feature_snapshot()`
2. `[扩展/共享] noise/statistics.py::estimate_clean_priors()`
3. `[扩展/共享] noise/statistics.py::build_label_imputation_matrix()`
4. `[扩展/共享] noise/statistics.py::StatisticArtifact`
5. `[规划] noise/statistics.py::estimate_per_class_moments()`
6. `[规划] models/generative.py::GaussianDiscriminantClassifier`
7. `[规划] training/pcse_pipeline.py::run_pcse_experiment()`
8. `[规划] tests/test_pcse.py`

### 5. 规划接口

```python
class GaussianDiscriminantClassifier:
    def fit(self, statistics: StatisticArtifact) -> None: ...
    def predict_proba(self, features: Tensor) -> Tensor: ...  # [B,C]

def estimate_per_class_moments(
    snapshots: Sequence[FeatureSnapshot],
    transition: TransitionArtifact,
) -> StatisticArtifact: ...
```

矩阵逆、协方差正定性和 shrinkage 必须是显式策略；不得无记录地改用伪逆或加
对角 epsilon。

### 6. Pipeline 伪代码

```python
base = load_pretrained_model(config.base_checkpoint)
t = load_transition_artifact(config.transition)
snapshots = [collect_feature_snapshot(base, train_loader, layer) for layer in layers]
stats = estimate_per_class_moments(snapshots, t)
classifiers = [GaussianDiscriminantClassifier(s) for s in stats.by_layer()]
weights = fit_simplex_ensemble(classifiers, noisy_validation_loader)
return PCSEPredictor(base, classifiers, weights)
```

### 7. 配置草案

```yaml
pipeline: {name: pcse}
base_checkpoint: path/to/best.pt
transition: {artifact: path/to/transition.npz}
statistics:
  feature_layers: [layer2, layer3, classifier_input]
  covariance: {policy: fail, shrinkage: 0.0}
```

### 8. Checkpoint 必需状态

base checkpoint hash、T hash、各 layer 名/shape/snapshot hash、priors、M、means、
second moments、covariances、GDA 参数、ensemble weights 和 noisy-validation identity。

### 9. 最小测试

1. 无噪声时恢复直接 per-class statistics。
2. 可逆小例一阶/二阶矩与手算一致。
3. covariance 对称且按配置检查正定。
4. layer 顺序与 feature dimension 错误失败。
5. ensemble weights 非负且和为 1。
6. artifact roundtrip 和推理可复现。
7. PCSE 不更新 base model。

### 10. 论文与官方代码核对

- `[论文][代码]` 都先训练 base model，再运行 PCSE 做 post-processing。
- `[代码]` 官方仓库以 `crossEntropy.py` 预训练，`PCSE.py` 负责统计估计和修饰。
- `[代码]` 官方 README 明确称 PCSE model-agnostic，并允许 CE、Co-teaching、
  JoCoR 等任意 base model。
- `[推断]` toolbox 将单脚本拆成共享 statistics artifact 与 PCSE Pipeline。

### 11. 当前未实现

- per-class statistics、GDA classifier、ensemble、PCSE Pipeline 与测试

---

## 21. DLD：Directional Label Diffusion Model for Learning from Noisy Labels

### 论文信息

- 编号 / 文件：`05_hybrid/21_dld_cvpr2025.pdf`
- 会议与年份：CVPR 2025
- 作者：Senyu Hou, Gaoxia Jiang, Jia Zhang, Shangrong Yang, Husheng Guo,
  Yaqing Guo, Wenjian Wang
- 官方代码：<https://github.com/SenyuHou/DLD>
- 当前成熟度：L2；已阅读论文并检查官方 `train_on_CIFAR.py` 及 `utils/` 分层，
  尚未运行
- Toolbox 归属：`FeatureGraph + LabelPrecorrection + Diffusion Pipeline`
- 不是：普通 LabelNoise generator、Loss 或 TransitionEstimator

### 1. 论文实际做了什么

[论文] 第一阶段用冻结/预训练 feature extractor 对 weak/strong 两种增强构建
KNN label distributions `p_w,p_s`，再用两视图 KL divergence 的 GMM 将样本分为
clean、noisy、hard：

```text
clean: y0=observed noisy one-hot,  yn=0
noisy: y0=argmax((p_w+p_s)/2),    yn=observed noisy one-hot
hard:  y0=(p_w+p_s)/2,            yn=normalize(abs(p_w-p_s))
```

第二阶段定义方向 `y_d=yn-y0`，前向加噪：

```text
y_t = y0 + alpha_bar_t * y_d + beta_bar_t * epsilon
```

分别学习 direction network 与 noise network：

```text
L_d   = ||y_d - y_theta(y_t,yn,x,t)||^2
L_eps = ||eps - eps_theta(y_t,yn,x,t)||^2
```

之后按论文的少步 reverse process 得到修正标签用于分类。

### 2. 完整调用顺序

```text
noisy Dataset -> weak/strong feature snapshots
-> two KNN graphs/distributions -> KL scores -> GMM partition
-> y0/yn pre-correction artifact
-> directional diffusion training
-> reverse label sampling -> soft/hard target
-> classifier training/evaluation
```

### 3. 与已有条目的重叠合并

- 特征收集复用 `FeatureSnapshot`，邻域结构复用 `NeighborGraphArtifact`。
- 分组结果复用 `SelectionResult`，soft label 复用 `SoftTargetResult`。
- 不把 DLD 的 diffusion objective 注册为普通 classification loss。
- 官方脚本直接 `update_label()`；toolbox 必须保留 manifest target，并以按
  global index 查询的 artifact 覆盖算法目标。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] training/snapshots.py::collect_feature_snapshot()`
2. `[规划/共享] data/neighbors.py::build_neighbor_graph()`
3. `[规划] noise/dld.py::build_view_label_distribution()`
4. `[规划] noise/dld.py::precorrect_labels()`
5. `[规划] models/directional_diffusion.py::DirectionalLabelDiffusion`
6. `[扩展/共享] core/result.py::SelectionResult/SoftTargetResult`
7. `[规划] training/dld_pipeline.py::run_dld_experiment()`
8. `[规划] tests/test_dld.py`

### 5. 规划接口

```python
@dataclass(frozen=True)
class DLDPrecorrectionArtifact:
    global_indices: np.ndarray
    partition: np.ndarray       # clean/noisy/hard
    y0: np.ndarray              # [N,C]
    yn: np.ndarray              # [N,C]
    weak_graph_hash: str
    strong_graph_hash: str
    artifact_hash: str
```

### 6. Pipeline 伪代码

```python
weak, strong = collect_two_view_features(extractor, train_dataset)
pw = build_view_label_distribution(build_neighbor_graph(weak), noisy_targets)
ps = build_view_label_distribution(build_neighbor_graph(strong), noisy_targets)
partition = fit_three_way_partition(kl_divergence(pw, ps))
pre = precorrect_labels(partition, pw, ps, noisy_targets, global_indices)
diffusion = train_direction_and_noise_models(pre, feature_conditioning)
targets = reverse_sample_labels(diffusion, pre, steps=config.reverse_steps)
train_classifier_with_soft_targets(targets)
```

### 7. 配置草案

```yaml
pipeline: {name: dld, warmup_epochs: 5}
features: {extractor: clip_vit, views: [weak, strong]}
neighbors: {k: 20}
diffusion: {steps: 5, direction_weight: 1.0, noise_weight: 1.0}
```

### 8. Checkpoint 必需状态

feature extractor identity、两个 feature/graph hash、GMM 参数和 partition、`y0/yn`、
两个 diffusion model/optimizer/EMA、schedule、reverse RNG、classifier 状态与 stage。

### 9. 最小测试

1. 两视图 KNN distribution 行和为 1 且按 global index 对齐。
2. 三类 partition 构造 `y0/yn` 与论文 Eq. 14–15 一致。
3. forward diffusion 与手算一致。
4. 两个 MSE objective 梯度只进入对应模型。
5. reverse 固定 RNG 可复现。
6. batch 中不泄漏 clean truth。
7. 多阶段 checkpoint/resume 一致。

### 10. 论文与官方代码核对

- `[论文][代码]` 都使用 weak/strong feature、pre-correction 与 directional
  diffusion 两阶段。
- `[代码]` 官方 `train_on_CIFAR.py::train()` 预计算两视图 embedding，并从
  `utils.pre_correction`、`utils.directional_diffusion_model` 组装流程。
- `[代码]` 官方支持 IDN/symmetric/asymmetric noise，但直接更新 dataset labels；
  toolbox 只通过 manifest 和 index artifact 注入。
- `[代码]` README 表示 diffusion checkpoints “available soon”；复现前需固定
  官方 commit 和预训练 extractor 版本。

### 11. 当前未实现

- 两视图图结构、pre-correction、diffusion 模型/Pipeline、配置和测试

---

## 22. FINE：Revisiting Learning with Noisy Labels: Active Forgetting and Noise Suppression

### 论文信息

- 编号 / 文件：`05_hybrid/22_active_forgetting_cvpr2026.pdf`
- 会议与年份：CVPR 2026
- 作者：Mengmeng Sheng, Zeren Sun, Tao Chen, Jinshan Pan, Yazhou Yao, Fumin Shen
- 官方代码：<https://github.com/NUST-Machine-Intelligence-Laboratory/FINE>
- 当前成熟度：L2；已阅读论文并检查官方 `SED_FINE.py` 结构，尚未运行
- Toolbox 归属：`Selector consumer + noisy-subset regularizer`
- 不是：Selector 本身

### 1. 论文实际做了什么

[论文] FINE 接在现有 baseline 的 clean/noisy partition 后。clean subset 继续使用
baseline objective；对 noisy subset 同时执行：

```text
active forgetting:
  L_MU = +(1/C) * sum_c one_hot(noisy)_c * log p_c
         # negative cross entropy，梯度方向使模型遗忘 observed label

noise suppression:
  choose complementary class != observed noisy class
  L_NL = -(1/C) * sum_c complementary_c * log(1-p_c)

total = L_base + beta * L_MU + gamma * L_NL
```

论文默认思想是先用 machine unlearning 遗忘已吸收的噪声，再用 negative learning
抑制继续拟合。

### 2. 完整调用顺序

```text
warm-up/baseline -> Selector returns clean/noisy partition
-> baseline objective on selected clean data
-> active-forgetting + negative-learning on rejected/noisy data
-> combined objective -> model update
```

### 3. 与已有条目的重叠合并

- FINE 必须消费同事 Selector 的 `SelectionResult`，不重写选样。
- base algorithm 仍拥有 optimizer/step；FINE 只提供附加 objective。
- complementary-label RNG 接入统一 checkpoint RNG，不另设 dataset。
- 若 Selector 只返回 selected indices，noisy subset 是当前 batch 补集。

### 4. 按顺序映射到文件和函数

1. `[规划/共享] core/result.py::SelectionResult`
2. `[规划] algorithms/fine.py::ActiveForgettingRegularizer`
3. `[规划] algorithms/fine.py::sample_complementary_targets()`
4. `[规划] algorithms/fine.py::negative_learning_objective()`
5. `[扩展] training/experiment.py` 通过 algorithm composition 调用，不写 FINE 分支
6. `[扩展/高冲突] plugins/builtin/catalog.py` 注册 `regularizer/fine`
7. `[规划] tests/test_fine.py`

### 5. 规划接口

```python
class FINERegularizer:
    def __call__(
        self,
        logits: Tensor,              # [B,C]
        noisy_targets: Tensor,       # [B]
        selection: SelectionResult,
        generator: torch.Generator,
    ) -> Tensor:                     # scalar additional objective
        ...
```

要求 `beta>0,gamma>0`。negative CE 在 `p_y→0` 时无下界，概率稳定化和最大幅值
策略必须显式配置并标注为工程安全策略。

### 6. Pipeline 伪代码

```python
losses = base_loss(logits, noisy_targets)       # [B]
selection = selector.select(losses, indices)
base = algorithm.aggregate(losses, selection)
fine = fine_regularizer(logits, noisy_targets, selection, rng)
objective = base + fine
optimizer_step(objective)
```

### 7. 配置草案

```yaml
algorithm: {name: supervised_with_selector}
regularizer:
  name: fine
  beta: 0.001
  gamma: 0.1
  probability_floor: 1.0e-8
```

### 8. Checkpoint 必需状态

Selector 私有状态由其自身保存；FINE 保存 beta/gamma、安全策略和 complementary
sampling generator state。其余由 base algorithm 保存。

### 9. 最小测试

1. clean/noisy mask 与 SelectionResult 对齐。
2. noisy subset 的 active-forgetting 梯度方向与 CE 相反。
3. complementary class 永不等于 observed target。
4. negative-learning 手算一致且极端 logits 有限。
5. 空 noisy subset 返回零附加项。
6. RNG checkpoint/resume 后 complementary sequence 一致。
7. 用两个不同 Selector 验证可插拔。

### 10. 论文与官方代码核对

- `[论文][代码]` FINE 都是附加在 sample-selection baseline 上的 plug-and-play
  regularizer。
- `[论文][代码]` 都使用 negative CE 做 forgetting、complementary negative
  learning 做 suppression。
- `[代码]` 官方仓库以 `SED_FINE.py` 等完整训练脚本复现；toolbox 不复制训练器，
  只抽取 regularizer contract。
- `[推断]` 稳定化策略需在 toolbox 中显式记录，不能改变论文符号后仍称原式。

### 11. 当前未实现

- FINERegularizer、complement RNG checkpoint、Selector 集成与测试

---

## 23. CA2C：A Prior-Knowledge-Free Approach for Robust Label Noise Learning via Asymmetric Co-learning and Co-training

### 论文信息

- 编号 / 文件：`05_hybrid/23_ca2c_iccv2025.pdf`
- 会议与年份：ICCV 2025
- 作者：Mengmeng Sheng, Zeren Sun, Tianfei Zhou, Xiangbo Shu, Jinshan Pan,
  Yazhou Yao
- 官方代码：<https://github.com/NUST-Machine-Intelligence-Laboratory/CA2C>
- 当前成熟度：L2；已阅读论文并检查官方 `main.py`、`loss.py`，尚未运行
- Toolbox 归属：`Dual-network asymmetric Pipeline`
- 不是：Co-teaching 的 small-loss peer exchange

### 1. 论文实际做了什么

[论文] CA2C 同时训练两种不同范式的模型：

```text
P-model: partial-label learning，消费 candidate label set
N-model: negative learning，消费 complementary label set
```

warm-up 后进行交叉引导：N-model 的 top-K 类生成 P-model candidate labels；
P-model 的最低 `C-K` 类生成 N-model complementary labels。N-model 生成的
candidate labels 按 global index 累积到 memory bank `M[N,C]`。P-model 同时
使用 `argmax(M)` 的 hard target 和 `M/row_sum(M)` 的 soft target，并以由 memory
置信度构造的权重消歧；N-model 使用 negative-learning objective。

### 2. 完整调用顺序

```text
initialize P/N models -> noisy-label warm-up
-> each batch cross-guidance predictions
-> candidate/complementary label results
-> update global memory bank by index
-> confidence/hard/soft targets
-> update P-model with partial-label objective
-> update N-model with negative-learning objective
```

### 3. 与已有条目的重叠合并

- 双网络生命周期可复用共同 model-pair/checkpoint helper，但不能复用
  `peer_exchange()` 的 small-loss 语义。
- memory 复用全局 index state 约束；soft target/weight 复用公共 result。
- 不建立新的 Dataset；candidate/complementary labels 是 algorithm artifact。

### 4. 按顺序映射到文件和函数

1. `[规划/共享] core/result.py::SoftTargetResult/WeightResult`
2. `[扩展/共享] selectors/history.py::IndexedState`
3. `[规划] algorithms/ca2c.py::CandidateMemory`
4. `[规划] algorithms/ca2c.py::cross_guidance()`
5. `[规划] algorithms/ca2c.py::partial_label_objective()`
6. `[规划] algorithms/ca2c.py::negative_label_objective()`
7. `[规划] training/ca2c_pipeline.py::run_ca2c_experiment()`
8. `[规划] tests/test_ca2c.py`

### 5. 规划接口

```python
@dataclass
class CandidateMemory:
    global_indices: Tensor    # [N]
    counts: Tensor            # [N,C]

    def update_(self, indices: Tensor, candidate_mask: Tensor) -> None: ...
    def targets(self, indices: Tensor) -> SoftTargetResult: ...
```

`K` 必须满足 `0<K<C`；memory lookup 不允许用 dataset position 猜 global index。

### 6. Pipeline 伪代码

```python
for batch in train_loader:
    p_logits, n_logits = p_model(x), n_model(x)
    candidates = topk(n_logits.detach(), k)
    complements = bottomk(p_logits.detach(), classes-k)
    memory.update_(batch["index"], candidates)
    targets = memory.targets(batch["index"])
    p_loss = partial_label_objective(p_logits, targets, lambda_hard_soft)
    n_loss = negative_label_objective(n_logits, complements)
    update_p(p_loss)
    update_n(n_loss)
```

### 7. 配置草案

```yaml
pipeline: {name: ca2c, warmup_epochs: 10}
ca2c: {candidate_k: 3, hard_weight: 0.5}
```

### 8. Checkpoint 必需状态

两模型/optimizer/scheduler、阶段、CandidateMemory counts/index/hash、K、hard/soft
权重、cross-guidance detach 语义、RNG 与 manifest identity。

### 9. 最小测试

1. top-K/bottom-(C-K) 集合互补且 shape 正确。
2. memory 按 global index 累积，shuffle 不改变结果。
3. hard/soft targets 与手算一致，soft 行和为 1。
4. P/N objective 梯度仅进入目标网络。
5. K 边界与空 memory 明确失败。
6. 双网络 checkpoint/resume 完整。
7. batch 不含 clean truth。

### 10. 论文与官方代码核对

- `[论文][代码]` 都采用 P/N 两种不对称学习范式和 cross-guidance。
- `[论文][代码]` 都以全局 memory 做 candidate label 置信度累积。
- `[代码]` 官方 `main.py` 将训练、memory 和更新写在单一入口；toolbox 按状态、
  objective 与 Pipeline 拆分。
- `[论文]` “prior-knowledge-free”指不依赖 noise rate/阈值，不表示没有 K 等算法参数。

### 11. 当前未实现

- CandidateMemory、两种 objective、cross-guidance、CA2C Pipeline 与测试

---

## 24. DivideMix：Learning with Noisy Labels as Semi-Supervised Learning

### 论文信息

- 编号 / 文件：`05_hybrid/24_dividemix_iclr2020.pdf`
- 会议与年份：ICLR 2020
- 作者：Junnan Li, Richard Socher, Steven C. H. Hoi
- 官方代码：<https://github.com/LiJunnan1992/DivideMix>
- 当前成熟度：L2；已阅读论文并核对官方 `Train_cifar.py`，尚未运行
- Toolbox 归属：`Loss-mixture Selector + Semi-supervised dual-network Pipeline`
- 不是：单个 Selector 或 Co-teaching 的 batch peer exchange

### 1. 论文实际做了什么

[论文] warm-up 后，每个网络在全训练集计算逐样本 CE，归一化并拟合二成分 GMM；
低均值成分的 posterior `w_i` 是 clean probability。网络 A 的分组训练网络 B，
反之亦然。`w_i>=tau` 进入 labeled set，其余只把图像作为 unlabeled set。

每个网络固定另一个网络后执行改进 MixMatch：

```text
labeled co-refinement:
  y_bar = w * observed_one_hot + (1-w) * mean_augment_prediction
  target_x = sharpen(y_bar, T)

unlabeled co-guessing:
  q_bar = mean(predictions of both networks over augmentations)
  target_u = sharpen(q_bar, T)

MixUp over labeled + unlabeled:
  lambda ~ Beta(alpha,alpha); lambda=max(lambda,1-lambda)
  L = Lx_soft_CE + ramp(lambda_u)*Lu_MSE + prior_regularization
```

asymmetric noise 的 warm-up 额外最大化 entropy，避免过早过置信。

### 2. 完整调用顺序

```text
two models -> warm-up
-> each epoch collect indexed CE histories
-> two GMM clean-probability results
-> cross divide labeled/unlabeled views
-> co-refine/co-guess soft targets
-> MixUp + semi-supervised objective
-> sequentially update each network -> ensemble evaluation
```

### 3. 与已有条目的重叠合并

- indexed loss history 复用 `selectors/history.py`。
- GMM 同时输出 `WeightResult(clean_probability)` 与阈值化 `SelectionResult`。
- soft target 复用 `SoftTargetResult`；半监督数据复用唯一
  `SemiSupervisedBatch`。
- 双网络可复用 checkpoint helper，但不调用 Co-teaching small-loss exchange。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] selectors/history.py::IndexedLossHistory`
2. `[规划] selectors/loss_mixture.py::GaussianMixtureSelector`
3. `[扩展/共享] core/result.py::SelectionResult/WeightResult/SoftTargetResult`
4. `[规划/共享] data/semi_supervised.py::SemiSupervisedBatch`
5. `[规划] algorithms/dividemix.py::co_refine/co_guess/sharpen/mixmatch_objective`
6. `[规划] training/dividemix_pipeline.py::run_dividemix_experiment()`
7. `[规划] tests/test_dividemix.py`

### 5. 规划接口

```python
class GaussianMixtureSelector:
    def fit_select(
        self, losses: Tensor, global_indices: Tensor
    ) -> tuple[WeightResult, SelectionResult]: ...

@dataclass(frozen=True)
class SemiSupervisedBatch:
    labeled_views: Tensor
    noisy_targets: Tensor
    clean_probabilities: Tensor
    unlabeled_views: Tensor
    labeled_indices: Tensor
    unlabeled_indices: Tensor
```

### 6. Pipeline 伪代码

```python
warmup(net1); warmup(net2)
for epoch in robust_epochs:
    prob2, split2 = gmm.fit_select(eval_losses(net1))
    prob1, split1 = gmm.fit_select(eval_losses(net2))
    train_one(net1, frozen_peer=net2, division=(prob1, split1))
    train_one(net2, frozen_peer=net1, division=(prob2, split2))
    evaluate_ensemble(net1, net2)
```

### 7. 配置草案

```yaml
pipeline: {name: dividemix, warmup_epochs: 10}
selector: {name: loss_gmm, threshold: 0.5, history_epochs: 1}
mixmatch: {augmentations: 2, temperature: 0.5, beta_alpha: 4.0}
unsupervised: {weight: 25.0, rampup_epochs: 16}
```

### 8. Checkpoint 必需状态

两模型/optimizer/scheduler、epoch/stage、每网 loss history、GMM 参数和 component
identity、clean probabilities/split index identity、MixUp RNG、ramp-up step 与
manifest identity。

### 9. 最小测试

1. GMM 选择较低均值 component，global index 稳定。
2. cross-divide 方向为 A→B、B→A。
3. co-refine/co-guess/sharpen 与手算一致且行和为 1。
4. labeled/unlabeled batch 不携带 clean truth。
5. MixUp lambda 与 soft target 对齐。
6. Lx/Lu/prior regularizer 与官方公式一致。
7. 两网络更新时 peer 冻结。
8. checkpoint/resume 后 GMM/history/RNG 连续。

### 10. 论文与官方代码核对

- `[论文][代码]` 都采用两网络 co-divide、GMM、co-refinement、co-guessing 和 MixMatch。
- `[代码]` 官方 `eval_train()` 按 global index 写入 50000 个 loss，归一化后拟合
  两成分 GMM；高噪声时平均最近 5 epochs。
- `[代码]` 官方 `train()` 明确冻结 peer、用两个 augmentation、Beta MixUp、
  soft CE、unlabeled MSE 和 uniform-prior penalty。
- `[推断]` toolbox 将官方单脚本拆为 selector、semi-supervised protocol 与 Pipeline。

### 11. 当前未实现

- loss-GMM Selector、SemiSupervisedBatch、MixMatch objective、双网络 Pipeline 与测试

---

## 25. L2RW：Learning to Reweight Examples for Robust Deep Learning

### 论文信息

- 编号 / 文件：`05_hybrid/25_l2rw_icml2018.pdf`
- 会议与年份：ICML 2018
- 作者：Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel Urtasun
- 官方代码：<https://github.com/uber-research/learning-to-reweight-examples>
- 当前成熟度：L2；已阅读论文并检查官方 TensorFlow 1.10 仓库，尚未运行
- Toolbox 归属：`Meta WeightProvider + bilevel Pipeline`
- 关键前提：需要少量 clean、balanced validation data

### 1. 论文实际做了什么

[论文] 每个 noisy training batch 给每个样本一个临时 `epsilon_i=0`，用它构造
一步虚拟更新：

```text
theta_hat(epsilon)
  = theta - alpha * grad_theta sum_i epsilon_i * train_loss_i(theta)
```

在 `theta_hat` 上计算 clean validation mini-batch loss，并对 epsilon 求
二阶 meta-gradient：

```text
u_i = - d validation_loss(theta_hat(epsilon)) / d epsilon_i | epsilon=0
w_i = max(u_i,0)
w = w / sum(w) if sum(w)>0 else all_zero
```

随后回到原参数，用 `w` 加权原 training batch loss并做真实 optimizer step。

### 2. 完整调用顺序

```text
noisy train batch + trusted clean validation batch
-> differentiable virtual one-step update
-> clean validation loss at virtual parameters
-> meta-gradient w.r.t. epsilon
-> nonnegative normalized WeightResult
-> real weighted train update
```

### 3. 与已有条目的重叠合并

- 输出复用 `WeightResult`，但 provider 依赖模型梯度，不能当纯函数 selector。
- 必须使用 functional/differentiable optimizer，不覆盖真实 optimizer state。
- 当前仓库规则要求 clean label 只能进入 evaluator，因此 L2RW **当前不可接入**。
  实施前必须由团队显式批准 `TrustedSupervisionProvider` 例外，并保证它不是
  noisy training data 的隐藏 clean truth。

### 4. 按顺序映射到文件和函数

1. `[阻塞/规划] data/trusted.py::TrustedValidationProvider`
2. `[扩展/共享] core/result.py::WeightResult`
3. `[规划] algorithms/l2rw.py::meta_reweight()`
4. `[规划] training/l2rw_pipeline.py::run_l2rw_experiment()`
5. `[规划] tests/test_l2rw.py`

在团队批准 policy 前，不得创建或注册上述 trusted-data 接口。

### 5. 规划接口

```python
def meta_reweight(
    model: Module,
    training_batch: Batch,
    trusted_batch: TrustedBatch,
    *,
    virtual_learning_rate: float,
) -> WeightResult: ...
```

`TrustedBatch` 必须记录数据来源、split、是否平衡和独立 fingerprint；普通 validation
loader 不可被自动提升为 trusted training supervision。

### 6. Pipeline 伪代码

```python
epsilon = zeros(batch_size, requires_grad=True)
train_losses = loss(model(train_x), noisy_y)       # [B]
virtual_params = differentiable_sgd(
    model.params, sum(epsilon * train_losses), alpha
)
meta_loss = mean(loss(functional_call(model, virtual_params, trusted_x), trusted_y))
grad_epsilon = grad(meta_loss, epsilon)
raw = relu(-grad_epsilon)
weights = raw / raw.sum() if raw.sum() > 0 else zeros_like(raw)
real_objective = sum(weights.detach() * train_losses)
optimizer_step(real_objective)
```

### 7. 配置草案

```yaml
pipeline: {name: l2rw}
trusted_validation:
  manifest: path/to/explicitly_audited_clean_subset.npz
meta: {virtual_learning_rate: 0.1}
```

该配置只有在 repository safety policy 修改并获批准后才可启用。

### 8. Checkpoint 必需状态

模型/optimizer/scheduler、trusted subset fingerprint/provenance、sampler state、
virtual-step rule、target balance、global step 和 RNG；epsilon/weights 为逐步临时量。

### 9. 最小测试

1. 一步小线性模型 meta-gradient 与有限差分一致。
2. weights 非负；和为 1 或全零。
3. 有助于 trusted loss 的样本获得更高 weight。
4. virtual step 不修改真实模型/optimizer。
5. 二阶梯度有限，AMP 行为显式。
6. 未配置显式 trusted source 时拒绝运行。
7. 普通 clean test/validation 不能误接入。

### 10. 论文与官方代码核对

- `[论文][代码]` 都使用 clean validation mini-batch、虚拟更新和二阶自动微分。
- `[论文]` Eq. 9 的退化分母使全零 raw weight 保持全零，不改为 uniform。
- `[代码]` 官方实现基于 TensorFlow 1.10/protobuf；不能复制到当前 PyTorch runner。
- `[差异/阻塞]` 论文方法的 clean supervision 前提与仓库当前安全规则冲突。

### 11. 当前未实现

- safety policy 例外、TrustedValidationProvider、meta-reweight、Pipeline 与测试

---

## 26. LEND：Towards Harnessing Feature Embedding for Robust Learning with Noisy Labels

### 论文信息

- 编号 / 文件：`05_hybrid/26_lend_mlj2022.pdf`
- 期刊与年份：Machine Learning 2022
- 作者：Chuang Zhang, Li Shen, Jian Yang, Chen Gong
- 论文页面：<https://arxiv.org/abs/2206.13025>
- 官方代码：未发现论文作者发布的官方实现
- 当前成熟度：L1
- Toolbox 归属：`FeatureGraph + indexed label state + Selector`
- 不是：直接用 diluted label 替换训练 target 的 label-correction Loss

### 1. 论文实际做了什么

[论文] 每个 mini-batch 从当前模型取得 feature `V[B,D]`，构造稀疏 KNN 相似矩阵：

```text
A[i,j] = max(v_i @ v_j, 0)^gamma,  if j in KNN(i), i!=j
W' = A.T @ A
W  = D^(-1/2) @ W' @ D^(-1/2)
```

用 observed noisy one-hot 初始化 `Z(0)`，迭代 T 次标签稀释：

```text
Z(t) = alpha * W @ Z(t-1) + (1-alpha) * Z(t-1)
```

再按 global sample identity 做 epoch momentum：

```text
Z_epoch = (1-beta) * Z_current + beta * Z_previous
```

最终并不拿 diluted label 直接监督；只有当 `argmax(Z_epoch_i)` 等于 observed noisy
label 时才选择该样本，仍以原 noisy label 计算 base loss。

### 2. 完整调用顺序

```text
batch input/noisy target/index -> current feature embedding
-> batch KNN graph/similarity W -> iterative label dilution
-> update indexed momentum Z[N,C]
-> select agreement with observed label
-> SelectionResult -> base loss on selected samples -> update model
```

### 3. 与已有条目的重叠合并

- KNN 图复用 `NeighborGraphArtifact` 的构建数学；LEND 的图是 batch-local，
  DLD 可是 dataset-level，scope 必须写入 artifact。
- `Z[N,C]` 复用 `selectors/history.py` 的 indexed state。
- 选样输出复用 `SelectionResult`，不创建 LEND 专用 selector result。
- soft diluted label 可用 `SoftTargetResult` 诊断，但训练不消费其 clean 替代标签。

### 4. 按顺序映射到文件和函数

1. `[扩展/共享] data/neighbors.py::build_neighbor_graph(scope="batch")`
2. `[扩展/共享] selectors/history.py::IndexedSoftLabelState`
3. `[规划] selectors/lend.py::dilute_labels()`
4. `[规划] selectors/lend.py::LENDSelector.select()`
5. `[扩展] training/experiment.py` 通过 Selector contract 调用
6. `[扩展/高冲突] plugins/builtin/catalog.py` 注册 `selector/lend`
7. `[规划] tests/test_lend.py`

### 5. 规划接口

```python
class LENDSelector:
    def select(
        self,
        *,
        features: Tensor,        # [B,D]
        noisy_targets: Tensor,   # [B]
        global_indices: Tensor,  # [B]
    ) -> SelectionResult:
        ...
```

内部 state 为 `Z[N,C]`；首次出现样本时以 observed one-hot 初始化。batch 不满 K
或 degree 为零时必须按显式策略失败/降阶，不能产生 NaN。

### 6. Pipeline 伪代码

```python
features, logits = model.forward_with_features(batch["input"])
graph = build_neighbor_graph(features.detach(), k, gamma, scope="batch")
z0 = one_hot(batch["target"])
z = repeat(lambda z: alpha * graph.W @ z + (1-alpha) * z, z0, steps)
state.update_momentum_(batch["index"], z, beta)
selected = state.lookup(batch["index"]).argmax(1) == batch["target"]
result = SelectionResult(mask=selected, global_indices=batch["index"])
objective = mean(base_loss(logits, batch["target"])[selected])
```

### 7. 配置草案

```yaml
selector:
  name: lend
  neighbors: 10
  gamma: 1.0
  diffusion_alpha: 0.99
  diffusion_steps: 10
  momentum: 0.9
```

### 8. Checkpoint 必需状态

`Z[N,C]`、global indices/fingerprint、alpha/beta/gamma/k/steps、graph scope、model
feature-layer identity、epoch/global step、optimizer/scheduler 和 manifest identity。

### 9. 最小测试

1. A/W 与论文公式手算一致，W 有限且对称。
2. Z 初始化、迭代和 momentum 与手算一致。
3. shuffle 后按 global index 读取同一历史。
4. selection 只比较 diluted argmax 与 observed noisy target。
5. 未选中样本不贡献梯度。
6. k 边界、零 degree、空 selected set 行为明确。
7. checkpoint/resume 后 Z 连续。
8. training path 不读取 clean truth。

### 10. 论文与代码核对

- `[论文]` Algorithm 1 明确在每个 mini-batch 建图、迭代 dilution、momentum
  更新并用 agreement 二值选样。
- `[论文]` Eq. 7 仍对 observed noisy label 计算 loss；diluted label 只用于选样。
- `[待核实]` 未发现作者官方代码，因此文件和函数为按论文映射的工程伪代码。
- `[推断]` 原算法的 batch-local KNN 对 batch composition 敏感；toolbox 必须保存
  sampler/RNG 并将 graph scope 写入配置，不能擅自改成全数据图仍声称同一算法。

### 11. 当前未实现

- batch neighbor graph、IndexedSoftLabelState、LENDSelector、配置和测试

至此，`papers/manifest.json` 中可读取的 26 篇论文均已完成实现映射。所有条目都应先
复用顶部“重叠能力主索引”的唯一接口；若同事分支已经实现同义组件，合并同事接口并
回改本指南，不并存两套文件、函数或协议。
