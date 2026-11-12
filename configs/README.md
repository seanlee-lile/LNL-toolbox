# 配置目录说明

这里的 YAML 使用版本化公共合同。每个文件必须显式写出：

```yaml
schema_version: 1
kind: experiment  # 或 fragment / mentor_artifact
```

## 用户可见配方

### 默认配置与历史 best 配置

`configs/reproduction/*_default.yaml` 是原有 26 篇论文的默认入口。本轮另外提供
26 份 `*_cifar10.yaml`、22 份 `*_cifar100.yaml`，对应
`outputs/论文参数五方对照_默认栏补全.xlsx` 中的历史 best 列，不替换默认入口。
Binary Risk、Importance Reweighting、MC-LDCE、PDL 没有历史 C100 best，未创建该组。
C10 的 Binary Risk、Importance Reweighting、CWD 仍为飞机/汽车二分类，不是十分类。

这些历史变体可通过独立 ID 或文件路径调用，例如：

```powershell
lnl validate --recipe apl-cifar100 --data my-cifar100
lnl run --recipe apl-cifar100 --data my-cifar100
lnl run --config configs/reproduction/apl_cifar10.yaml --data my-cifar10
```

48 份配置均来自表中引用的 `artifact/full-test-best-20261106/<数据集>/<论文>/config.json`。
文件头记录源文件 SHA-256、历史代码版本、恢复状态和迁移字段；大矩阵保留完整值。
当前代码不会自动切换为历史代码，因此相同参数不保证重现历史分数。
C10 CNLCU 来源未记录独立代码版本号，文件头明确标注缺失；C10/C100 CWD 和
C10 Loss Correction 的原始输入文本只做了部分恢复，不能宣称逐字恢复。

迁移仅移除机器绝对数据目录、旧 Web/运行时记录和旧 `noise.type` 别名，
为每份配置设置独立相对输出目录；数据通过本机登记及 `--data` 指定。
CWD C10 的历史驱动 dropout=0.5 已写入 `model.feature_dropout`。
PCSE 的旧 `pretraining_stage.mode=train` 已省略：当前执行器始终执行同一 CE 预训练阶段，
其模型、预算、优化器和检查点选择不变。L2RW 的虚拟更新学习率保留。

下列资源只迁移了引用路径，没有复制、下载或生成。需要准备与历史来源相同内容的文件，
不能用同名的任意文件替代；缺失时 `validate` 应报错。

| 配置 | 必需资源位置（相对仓库） |
|---|---|
| CAL C10 | `data/cal/IDN_0.2_C10-subset.pt`（历史带噪/干净标签及顺序须与输入对齐） |
| CAL C100 | `data/cal/IDN_0.2_C100.pt` |
| DLD C100 | `data/dld/cifar100-idn-0.2.pt` |
| UPM C100 | `data/upm-cifar100-mlp/labels.pt` |
| CWD C100 | `data/cwd/cifar100-best-transition.npz` |
| LEND C10 | `data/lend/cifar10-best-noise-manifest.npz`（保留历史来源哈希约束） |
| MentorNet C10/C100 | `data/mentornet/cifar10-symmetric04-official-port-seed20260729/mentor_artifact.pt` |
| DLD C10/C100 | 原配置使用的 TorchVision ResNet-34 `IMAGENET1K_V1` 缓存权重 |

表中 DSS C10 的 `early_stopping` 总项写“未显式设置”，但四个子项和原始配置均有值；
历史 YAML 保留完整字典，不因总项空缺而删除。本文不将历史 best 称为论文唯一参数或新训练结果。

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

`manifest` 不能与 `name/rate/seed` 混用。缺少 `noise` 时保持干净标签训练；配置后仅训练集按 global index 使用噪声标签，validation/test 仍使用干净标签。最终映射固定保存到运行目录，恢复时不重新生成、也不重新依赖外部源文件。使用 `lnl run` 启动训练。

## Conditional recipes / prerequisites

有些 recipe 是 conditional：配置文件存在不表示当前机器已经能够运行。PCSE、DLD、CAL 和 MentorNet 等流程可能分别需要 checkpoint/source run、真实 pretrained feature extractor、对齐的 external clean/noisy label artifact 或已验证的 MentorArtifact。

来源可由 YAML 中的路径、run directory/environment、external label artifact 或 MentorArtifact 提供。readiness 至少会经过配置、schema、identity compatibility 和 consumer validation；仅 `PATH_EXISTS` 不等于 `READY`。不要把任意 `.pt` 文件、随机冻结模型或未对齐标签文件当作正式来源。

本地数据登记同样按 adapter source contract 进行：单文件 adapter 使用 `lnl data register ... --path <file>`，目录 adapter 使用 `lnl data register ... --root <directory>`。不要同时传递无关的 source 参数。

这里仅说明配置合同；Web 操作、DatasetProfile 和具体方法的 prerequisite 指引请见 [Toolbox 使用手册](../docs/toolbox-usage-manual.md)。

