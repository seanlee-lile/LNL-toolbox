# DLD workflow

Directional Label Diffusion（DLD）在 Toolbox 中通过统一的
`doctor → list → validate → dry-run → run → resume` 流程执行。当前实现是
**paper-oriented、带前置条件的 workflow**，不是论文数值或作者代码的逐项复现。

## 前置条件与状态

正式 DLD 配置必须使用真实、可验证且与配置身份兼容的预训练特征提取器。当前支持的
formal source 由 recipe/readiness 检查明确列出，例如
`torchvision_resnet34_imagenet1k_v1` 或兼容的 `upm_main_best` 来源。Toolbox 不会自动
联网下载 torchvision 权重；缺少权重、身份不匹配或 schema 无效时，Validate、Dry-run 和
Paper 页面都会显示 prerequisite 未满足并阻止运行。

随机初始化后冻结的 TinyCNN/仓库内冻结模型仅用于 engineering/smoke 链路，以检查数据、
artifact、生命周期和恢复逻辑。它不是正式 pretrained extractor，也不能据此宣称 formal
或数值复现就绪。

## 当前 workflow

```text
noisy CIFAR train
-> frozen, validated feature extractor with weak/strong views
-> cosine-similarity weighted KNN distributions
-> KL(p_s || p_w) and deterministic two-component GMM partition
-> paper-oriented y0 and yn, with yd = yn - y0
-> independent direction/noise predictors and optimizers
-> deterministic five-step reverse sampling
-> noisy-validation checkpoint selection
-> clean-test reporting
```

在 cosine backend 中，近邻按余弦相似度选择，并以 `1 / (similarity + delta)` 形成行归一化
权重；无效分母会显式失败而不是静默裁剪。零初始推理向量是明确的 Toolbox engineering
选择。

## 命令

```powershell
lnl papers show dld
lnl validate --recipe cifar10-dld-smoke --check-data
lnl run --recipe cifar10-dld-smoke --dry-run --check-data
lnl run --recipe cifar10-dld-smoke --check-data
lnl resume <run-directory>
```

`--epochs N` 仅覆盖 `dld.diffusion.epochs`。运行会生成
`dld_precorrection.npz`、`last.pt`、`best.pt`、`metrics.jsonl` 和
`final_metrics.json`。恢复会验证 manifest、稳定样本映射、特征提取器身份、转换、
pre-correction artifact、schedule、推理策略和 fidelity identity。

## Claim boundary

smoke 成功只证明短预算 engineering 链路可用。paper-oriented/formal 配置也不自动表示
论文数值、完整数据协议、预训练表示设置或多 seed 统计已复现。任何数值复现结论都应以
catalog 的 `reproduction_status` 和独立实验记录为准。
