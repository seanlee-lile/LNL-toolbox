# LNL Toolbox

一个面向研究实验的通用、可扩展运行框架，LNL 是首个插件集合，而不是核心层的硬编码前提。

## 从这里开始

- [使用说明](docs/usage-guide.md)
- [开发指南](docs/development-guide.md)
- [逐文件职责](docs/file-map.md)
- [项目管理指南](docs/project-management-guide.md)
- [LNL 长期架构与实验路线](toolbox-architecture.md)

## 通用核心

`src/lnl_toolbox/core/` 只固定最小协议：

- `Batch`：不透明 payload，可承载图像、文本、图、数组或任务自定义对象；
- `ExperimentContext`：配置、工作目录、随机种子和外部服务；
- `Algorithm`：run/cycle/step 生命周期，不规定 cycle 必须是 epoch；
- `RunState`：框架级进度和指标，算法私有状态仍由算法自己保存；
- `StepResult`：统一封装输出、指标、artifact 和元数据；
- `Evaluator`：消费结果，不预设 accuracy、F1 或噪声识别指标；
- `ArtifactSink / CheckpointStore`：把本地文件、对象存储或实验平台隔离为 adapter；
- `PluginCatalog`：按 kind/name 注册并按 capability 发现组件。

核心不依赖 PyTorch，也不假设分类标签、噪声率、单网络或双网络。

## 可选内置插件

`src/lnl_toolbox/plugins/builtin/` 注册少量参考插件，用来验证扩展点：

- symmetric、pairflip、instance-dependent 噪声示例；
- 可训练的逐样本 CE、GCE、NCE、MAE、RCE 和 APL；
- Co-teaching 的交叉选样函数。

这些实现不是运行框架的必选依赖。旧导入路径暂时保留，方便前期实验代码迁移。

## 快速验证

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

训练、clean baseline、数据检查和噪声生成命令均支持无参数交互向导。例如：

```powershell
lnl-train
lnl-clean-train
lnl-inspect-data
lnl-make-noise
```

只要提供任意命令行参数，程序就继续使用原有的非交互 argparse 模式，适合脚本和批量实验。

可选：生成 LNL 噪声清单（输入为一维 `.npy` 标签数组）：

```powershell
$env:PYTHONPATH = "src"
python -m lnl_toolbox.cli.make_noise labels.npy artifacts/noise/demo.npz --kind symmetric --rate 0.4 --classes 10 --seed 1
```

要使用固定噪声训练，在实验 YAML 中加入 `noise.manifest`，再运行 `lnl-train --config <yaml>`。程序会先校验数据 fingerprint、标签范围和转移概率；train 使用 noisy target，validation/test 继续使用干净标签。`lnl-clean-train` 会拒绝噪声配置。

论文原文位于 `papers/`，逐篇中文摘要、代码状态和伪代码见 `docs/paper-summaries.md`；详细架构取舍见 `toolbox-architecture.md`。

## 结构参考

- Cleanlab：模型无关的数据诊断 API；
- Co-teaching / JoCoR：双模型与选样逻辑应由插件算法管理；
- DivideMix：warm-up、全数据统计和阶段切换不应写入通用 Runner；
- Active-Passive-Losses：loss 配置与训练器解耦。
