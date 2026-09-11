# Directional Label Diffusion: reproduction status

## Current status

DLD 已实现为带 artifact、pre-correction 和 checkpoint 合同的 **paper-oriented workflow**。
它可以通过 smoke 检查工程生命周期，但当前不宣称论文数值或作者代码的逐项复现。

正式/paper-oriented 配置的前置条件是真实、可验证且身份兼容的预训练特征提取器。当前
支持的 extractor/source 由配置和 readiness 检查明确声明，例如
`torchvision_resnet34_imagenet1k_v1` 与兼容的 `upm_main_best`。Toolbox 不会自动联网下载
torchvision 权重。随机初始化后冻结的 TinyCNN 或仓库内冻结模型只属于 engineering/smoke
路径，绝不能作为 formal pretrained representation。

## Workflow fidelity

`paper_oriented_v2_cosine_similarity` 使用平均 weak/strong `y0`、估计 `yn`、方向
`yn - y0`、包含 self-neighbors、`KL(p_s || p_w)`、平均 diffusion schedule 和五次确定性
反向步骤。cosine backend 选择最高余弦相似度近邻，并使用
`1 / (similarity + delta)` 后做行归一化；无效的非正分母会失败而不是裁剪。零初始推理
向量是明确的 Toolbox engineering 选择。

这些实现选择不等同于论文的完整表示学习、数据/噪声协议、训练预算或报告数值。

## Commands

```powershell
lnl papers show dld
lnl validate --recipe cifar10-dld-smoke --check-data
lnl run --recipe cifar10-dld-smoke --dry-run --check-data
lnl run --recipe cifar10-dld-smoke --check-data
lnl resume <run-directory>
```

开始 formal/paper-oriented 运行前，先在 Paper 页面或 `lnl validate` 中确认 extractor 的
readiness 为 READY。缺少权重、身份不兼容或 schema 无效时应修正配置/来源，而不是以
random frozen source 替代。

pre-correction artifact 记录稳定索引、noisy targets、邻居分布、partition evidence、
`y0`、`yn`、`yd`、condition features、manifest/mapping hashes、feature/transform identity、
KNN/GMM 设置和 fidelity。它会先写入 sibling temporary NPZ、重载验证，再原子替换。

## Claim boundary

workflow 或 smoke 成功不能作为论文 numerical reproduction 的证据。完整的论文噪声/数据
协议、正式预训练表示、全预算训练、多 seed 聚合和论文表格对比仍需独立记录与验证。
