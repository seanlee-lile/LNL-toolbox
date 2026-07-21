# 配置目录说明

这里的 YAML 都是 LNL 示例插件配置，不是核心框架 schema。

通用核心只要求最终向 `ExperimentContext.config` 提供一个只读 mapping；配置来源可以是 YAML、JSON、Hydra、命令行、数据库或调用方直接构造的字典。后续若采用 Hydra，也应通过 adapter 接入，不能让 core 直接依赖 Hydra。

训练实验通过顶层 `loss` mapping 选择逐样本 PyTorch objective，缺省时兼容为 CE：

```yaml
loss:
  name: gce
  q: 0.7
```

`configs/algorithm/` 保存可复用的组件片段；当前入口尚未实现 Hydra defaults 合并，因此完整实验 YAML 需要显式包含所选 loss 配置。

通用训练入口可选读取一个已经生成的 Noise Manifest：

```yaml
noise:
  manifest: data/noise/cifar10-symmetric-0.4-seed1.npz
```

缺少 `noise` 时保持干净标签训练；配置后仅训练集按 global index 使用 manifest 中的噪声标签，validation/test 仍使用干净标签。训练不会临时重采样噪声。`lnl-clean-train` 专用于 clean baseline，会拒绝 `noise` 配置；噪声实验使用 `lnl-train`。

