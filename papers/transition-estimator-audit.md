# Transition Estimator 论文审计

本文按 `papers/reading-standard.md` 记录 TransitionEstimator 的算法依据与实施状态。
研究结论以论文原文为准；作者代码只用于核对实现细节。

## 1. Anchor Transition（L3）

### 来源与成熟度

- 论文：Patrini et al., *Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach*, CVPR 2017。
- 本地 PDF：`papers/03_robust_loss/10_loss_correction_cvpr2017.pdf`。
- 原文页面：<https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html>。
- 作者代码：<https://github.com/giorgiop/loss-correction>，已核对 `models.py::NoiseEstimator`。
- 成熟度：Anchor estimator 为 L3；Forward/Backward 仅完成论文定位，本轮未实现。

### 问题、假设与公式

论文采用类别条件噪声：

```text
T[i,j] = P(noisy label=j | clean label=i)
p_noisy = p_clean @ T
```

先用含噪标签训练分类器，得到后验估计
`P(noisy label=j | x)`。在“每个类别存在几乎确定属于该类的样本”假设下，
论文式 (12)–(13) 对每个 clean 类别 `i` 选择：

```text
anchor_i = argmax_x P(noisy label=i | x)
T[i,j] = P(noisy label=j | anchor_i)
```

输入概率形状为 `[N,C]`，输出为 `[C,C]`。估计阶段只需要 noisy posterior、
noisy target 和稳定 global index；不得读取 clean target、flip mask 或真实 `T`。

### 训练协议与状态

完整论文流程是两阶段：先训练 noisy classifier 并估计 `T`，再由
Forward/Backward correction 消费该矩阵重新训练模型。本轮只实现离线估计器，
不接入 runner，不拥有 optimizer、scheduler 或 checkpoint 私有状态。

### 伪代码

```text
for clean_class i in 0..C-1:
    best_score = max posterior[:, i]
    candidates = samples whose posterior[:, i] == best_score
    anchor = candidate with smallest global_index  # [工程推断：确定性并列规则]
    T[i, :] = posterior[anchor, :]
return TransitionArtifact(T, anchor indices, snapshot hash)
```

### 论文与作者代码差异

- 论文公式是直接 `argmax`；本实现严格采用该路径。
- 作者代码另有可选的 97% 分位数离群过滤，并在实验脚本中通常启用；这不是
  本轮默认算法，未来若增加必须作为显式配置和独立变体记录。
- 作者代码可再次做行归一化并保留 `alpha` 与单位阵混合的入口；本接口已严格
  验证 posterior 行和为 1，因此不隐式归一化，且不实现未启用的 `alpha` 混合。
- 论文未规定精确并列的选择。为保证输入重排不影响产物，本实现选择最小
  global index；这是工程推断，不宣称为论文贡献。

### Toolbox 映射与限制

```text
PosteriorSnapshot → AnchorTransitionEstimator → TransitionArtifact
```

- Snapshot hash 绑定 dataset、split、概率、noisy target 和 global index。
- Artifact hash 绑定矩阵、算法、来源 hash、版本、方向和 metadata。
- 非法概率或矩阵直接失败，不裁剪、不重归一化。
- Artifact 是 Forward/Backward、Importance Weighting 等未来模块的输入，不是 Loss。
- 当前 Manifest 的 `per_sample_transition[N,C]` 不是 PDL 的 `T(x)[N,C,C]`。

## 2. Dual-T（论文原式实现）

### 来源、假设与公式

- 论文：Yao et al., *Dual T: Reducing Estimation Error for Transition Matrix
  in Label-noise Learning*, NeurIPS 2020。
- 本地 PDF：`papers/04_statistics/16_dual_t_neurips2020.pdf`。
- 论文页面：<https://papers.nips.cc/paper/2020/hash/512c5cad6c37edb98ae91c8a76c3a291-Abstract.html>。
- 官方代码：未发现作者发布的官方实现；实现依据为原文 Algorithm 1。

论文引入 hard intermediate label，把 clean→noisy 转移拆为：

```text
T_club[i,l]  = P(intermediate=l | clean=i)
T_spade[l,j] = P(noisy=j | intermediate=l)
T             = T_club @ T_spade
```

`T_club` 复用 Anchor estimator；intermediate label 是 noisy posterior 的
`argmax`；`T_spade` 由 intermediate 与 observed noisy target 的频数估计。
矩阵乘法按 Toolbox 行向量约定书写，仍满足 `p_noisy = p_clean @ T`。

### Toolbox 实现与工程边界

```text
model + noisy loader
  → collect_posterior_snapshot()
  → PosteriorSnapshot
  → DualTransitionEstimator
  → TransitionArtifact(matrix, factors, counts, anchors, source hash)
```

- Collector 在 inference mode 下收集并按 global index 排序，不改变输入 Dataset，
  不读取 clean target，结束后恢复模型原训练/eval 状态。
- Dual-T 不复制 anchor 逻辑；`T_club` 来自现有 Anchor artifact。
- intermediate `argmax` 的类别并列遵循 NumPy 最小类别 index；这是确定性工程规则。
- 任一 intermediate 类别没有样本时，频数条件概率不可定义，因此明确失败；
  不静默加平滑、不重归一化。
- `T_club`、`T_spade`、频数、anchor global indices 与父 artifact hash 均保存
  在最终 Artifact metadata，且受 artifact hash 保护。
- Estimator 无 optimizer/checkpoint 私有状态，当前不接入公共 runner；warm-up
  训练和 Forward/Backward 等消费者不属于本轮。

### 伪代码

```text
t_club_artifact = AnchorTransitionEstimator.estimate(snapshot)
intermediate = argmax(snapshot.noisy_probabilities, axis=1)
counts[l,j] = number of samples with intermediate=l and noisy_target=j
fail if any row count is zero
t_spade = counts / row_sum(counts)
t = t_club_artifact.matrix @ t_spade
return TransitionArtifact(t, estimator="dual_t", factor metadata)
```

### 论文一致性说明

- 原文 Algorithm 1 的 anchor、hard intermediate label、频数估计和因子合成均保留。
- 原文采用的矩阵记号与本项目行向量方向不同；本实现依据条件概率下标固定为
  `T_club @ T_spade`，没有改变概率语义。
- 因未找到官方代码，不能宣称完成作者实现逐行复现；当前成熟度是“论文原式
  实现并通过数学/协议测试”。

## 3. 后续算法边界（未实现）

| 方法 | 正确模块归属 | 本轮状态 |
|---|---|---|
| T-Revision | 有状态 NoiseModel；联合优化 `T + ΔT` | 仅定位论文，未实现 |
| VolMinNet | 独立联合训练 Pipeline | 未实现 |
| PDL | InstanceNoiseModel，输出 `T(x)[B,C,C]` | 未实现 |
| UPM | InstanceNoiseModel + PosteriorRefiner | 未实现 |
| CAL | StatisticEstimator / RiskCorrector | 未实现 |

## 4. 当前任务进度

- 当前任务：共享 posterior collector + Dual-T 离线 TransitionEstimator。
- 分支/基线：`loss` / `6be5677`（该提交已包含完整论文 guideline）。
- 清单：论文/指南核对、collector、Dual-T、factor metadata、registry、
  focused tests、完整回归、最终 diff/allowlist，共 8 项。
- 已完成：8 项；进度 `8 / 8 = 100%`。
- 已修改：`noise/estimators.py`、`noise/__init__.py`、`training/__init__.py`、
  plugin catalog、两份测试、两份公共文档及两份论文指南。
- 已新增：`training/snapshots.py`。
- focused tests：`test_transition_estimators.py` 15 项（含 CUDA collector）、
  `test_plugins.py` 6 项和 `test_noise.py` 16 项通过。
- 完整回归：临时映射 F 盘 CIFAR-10 后，`unittest` 85 项全部通过；junction
  已验证并清理，源数据未受影响。
- local checkpoint commits：无；本任务未获 commit 授权。
- blockers：无。
- 明确未做：warm-up trainer、runner/CLI/checkpoint 接入、消费者算法、
  Loss/Selector 修改。
- exact next step：由用户审阅本地 diff；如需提交，另行授权 commit。
- history cleanup：不需要。push readiness：代码和测试已就绪，但尚未获 commit
  或 push 授权。
- 协作提示：plugin catalog、plugin tests、data-flow 和 file-map 是高冲突文件；
  与 Selector 分支合并时只整合 Dual-T 对应的 import、注册项和文档行。

## 5. Taxonomy P1 本地集成状态

- 当前任务：合并 Loss、Selector、WeightProvider 与 TransitionEstimator 基础组件。
- 当前分支：`codex/integrate-taxonomy-p1`。
- 两个来源：`origin/loss@6be5677`、`origin/ce_baseline@3f11ad0`。
- merge-base：`cb9b847`。
- 清单：计划替换、集成分支、远程验证、本地 merge/冲突、四类联合检查、
  focused/完整回归、CUDA smoke/resume、最终保护检查，共 8 项。
- 已完成：全部 8 项；进度 `8 / 8 = 100%`。
- 已确认同事提交：
  - `f2e241f feat(selector): add batch selection and keep-rate schedules`
  - `3f11ad0 feat: add sample treatment and binary RCN reweighting`
- 本地 checkpoint：`15e3915 checkpoint: prepare taxonomy P1 integration`。
- 本地 merge 已完成内容整合；实际冲突集中在 plugin export/catalog、统一 runner、
  两份公共文档和两份联合测试，均按“保留 Selector 生产路径、并列接入
  TransitionEstimator 旁路”的原则解决。
- 受保护范围：同事的 `selectors/`、`treatments/`、监督 Algorithm、README、
  Selector/WeightProvider 配置和专属测试不得改写。
- 四类联合检查：Loss、Selector、Treatment/WeightProvider、TransitionEstimator
  focused tests 共 65 项通过；训练与 noisy resume 测试 25 项通过；完整
  `unittest` 127 项通过。
- CUDA smoke：noisy CE + AllSelector、fixed Small-Loss、linear Small-Loss
  均完成 2 epochs；fixed Small-Loss 从 epoch 2 恢复至 epoch 3。epoch/global
  step、selector ratio、manifest mapping hash、checkpoint 均连续，峰值显存
  不超过 162 MB。
- 最终检查：临时 CIFAR junction 与 smoke 产物均已清理，F 盘源数据完整；
  allowlist、受保护文件和最终 diff 检查通过。
- exact next step：由用户审阅当前本地 merge 结果；如接受，再单独授权创建最终
  merge commit。禁止 push。
- history cleanup / push readiness：均未就绪；禁止 push。
