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

## 2. 后续算法边界（未实现）

| 方法 | 正确模块归属 | 本轮状态 |
|---|---|---|
| T-Revision | 有状态 NoiseModel；联合优化 `T + ΔT` | 仅定位论文，未实现 |
| Dual-T | 第二个离线 TransitionEstimator | 未实现 |
| VolMinNet | 独立联合训练 Pipeline | 未实现 |
| PDL | InstanceNoiseModel，输出 `T(x)[B,C,C]` | 未实现 |
| UPM | InstanceNoiseModel + PosteriorRefiner | 未实现 |
| CAL | StatisticEstimator / RiskCorrector | 未实现 |

## 3. 实施进度

- 分支/基线：`loss` / `cb9b847`。
- 已完成：Snapshot 严格校验与哈希、Artifact v1 持久化与篡改检测、Anchor
  原式、registry/builder、CPU/CUDA Tensor 转换测试、跨模块协议更新。
- 定向验证：`test_transition_estimators` 8 项、`test_noise` 16 项、
  `test_plugins` 6 项，共 30 项通过。
- 完整回归：临时映射真实 CIFAR-10 后，`unittest` 76 项全部通过；映射已移除。
- CUDA noisy smoke：symmetric 0.4 + CE 完成 2 epochs / 8 steps，所有 loss
  有限，生成 resolved config、manifest、last/best checkpoint；峰值显存约
  160.6 MB，产物已清理。
- 遗留问题：配置中的相对 `output_root` 会在现有 noisy runner 内与绝对
  manifest 路径混用而失败；使用绝对 `--output-dir` 可运行。本轮按范围约束未
  修改 `training/`。
- 最终 allowlist：工作区变化仅包含计划批准的 7 个修改文件和 3 个必要新增
  文件；无数据、运行产物、junction、checkpoint 或缓存进入 Git 状态。
- 明确未做：runner/CLI/checkpoint 接入、Loss/Selector 修改、复杂 estimator。
