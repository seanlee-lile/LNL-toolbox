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

顶层 `selector` mapping 选择单 batch 的 hard sample selector。缺省时使用 `all`，与原有全样本均值训练一致：

```yaml
selector:
  name: small_loss
  keep_rate: 0.5
```

`small_loss` 的固定浮点配置保持兼容，也可显式写为 constant schedule：

```yaml
selector:
  name: small_loss
  keep_rate:
    name: constant
    value: 0.8
```

linear schedule 使用零基 epoch：epoch 0 为 `start`，epoch `warmup_epochs` 达到 `end`，之后固定为 `end`。

```yaml
selector:
  name: small_loss
  keep_rate:
    name: linear
    start: 1.0
    end: 0.6
    warmup_epochs: 10
```

所有 rate 必须位于 `(0, 1]`，`warmup_epochs` 必须为正整数。keep rate 不从 noise rate 推导。当前通用训练路径只提供无状态的 `all` 和 `small_loss`。Selector 只接收 detached 的逐样本 loss、stable global index 和当前零基 epoch；不读取 clean label，也不管理优化器或训练生命周期。Co-teaching 的双网络 peer exchange 仍属于独立 Algorithm/Pipeline，不使用该配置替代。

通用训练入口支持两种互斥噪声来源。运行开始时生成：

```yaml
noise:
  name: symmetric       # 或 pairflip
  rate: 0.4
  seed: 17
  manifest_filename: noise_manifest.npz
```

读取已经生成的 Noise Manifest：

```yaml
noise:
  manifest: data/noise/cifar10-symmetric-0.4-seed1.npz
  manifest_filename: noise_manifest.npz
```

`manifest` 不能与 `name/rate/seed` 混用。缺少 `noise` 时保持干净标签训练；配置后仅训练集按 global index 使用噪声标签，validation/test 仍使用干净标签。最终映射固定保存到运行目录，恢复时不重新生成、也不重新依赖外部源文件。`lnl-clean-train` 是 clean-only 包装，会拒绝 `noise`；噪声实验使用 `lnl-train`。

