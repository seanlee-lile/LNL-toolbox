# 配置目录说明

这里的 YAML 使用版本化公共合同。每个文件必须显式写出：

```yaml
schema_version: 1
kind: experiment  # 或 fragment / mentor_artifact
```

## 用户可见配方

活动 YAML 是内部可复现资产，不等于都需要出现在用户菜单中。
`src/lnl_toolbox/cli/data/recipe_catalog.json` 的 `public` 清单维护默认公开模板：

- `lnl list experiments` 只显示少量公开模板；
- `lnl list experiments --all` 显示全部内置实验配置；
- Web 新手模式只显示公开模板；
- Web“新建 YAML”默认按 26 篇论文各显示一份正式配置；“通用监督组合”模式继续使用少量公开模板；
- Web 的 Recipe 编辑页仍可访问全部内部配置。

论文正式配置模式只复制论文目录指定的完整 YAML，不把专用 runner 改写成通用监督配置，
也不允许在复制时混入 loss/selector 等跨生命周期覆盖。生成后可在 Recipe 编辑页审阅和修改。
界面同时显示 `paper_protocol`、`paper_oriented` 或 `engineering` 等 fidelity，避免把完整预算
误称为已完成论文数值复现。

Smoke 配置会缩小数据、模型或训练预算以检查链路；正式配置还定义模型、优化器、
scheduler、数据规模和论文生命周期，不能只通过修改 epoch 将 Smoke 当作正式复现。

`kind: experiment` 表示可直接交给 `lnl run` 的完整配置；`kind: fragment` 仅保存参数
片段，不能独立运行。历史版本原样保存在
`archive/configs-legacy-2026-08-21/`，并由 SHA-256 清单保护。

完整实验统一使用 `execution.runner`、`data.name`、`loader`、`model`、`optimizer` 和
`trainer`。论文特有阶段仍保留独立 mapping，不得为了外观统一而改变训练生命周期。
`noise.type` 以及顶层 `epochs/batch_size/learning_rate` 已废弃。

`data.name` 选择数据适配器；本机位置由 `lnl data register` 登记。完整 recipe 通常不写
`data.root`。当同一适配器只有一个本地登记时会自动解析；存在多个登记时必须用
`lnl run ... --data <alias>` 明确选择。

训练实验通过顶层 `loss` mapping 选择逐样本 PyTorch objective，缺省时兼容为 CE：

```yaml
loss:
  name: gce
  q: 0.7
```

`configs/algorithm/` 保存标记为 `kind: fragment` 的参考片段；当前入口未实现 Hydra
defaults 合并，因此完整实验 YAML 必须显式包含所需参数。

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

