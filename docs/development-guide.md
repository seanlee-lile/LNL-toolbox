# LNL Toolbox 开发指南

## 新论文方法接入 checklist

新增算法属于开发行为，不存在运行时 `paper register`。先确定下面的扩展类型；
不要把一份可运行 YAML、一个 plugin 或 smoke 成功等同于论文方法已完整接入。
本节不使用 Scratch 作为正式产品扩展方案。

| 类型 | 必需登记点 | 条件性登记点 | 最低测试 |
| --- | --- | --- | --- |
| CONFIG_ONLY | 实验 YAML、recipe catalog、paper catalog | 发行包配置清单；可选 Web 中文 presentation | 配置/runner/requirements、discovery、packaging |
| COMPONENT_EXTENSION | 上述清单、组件实现、builtin component factory、明确的 requirements | 参数校验、组件专用 guidance | 组件行为与失败输入、requirements、原有方法不回归 |
| DEDICATED_RUNNER | 上述清单、专用 runner、RunnerSpec/requirements | method 路由集合、配置 validator、新顶层字段的 schema、预算 planner | admission、数据角色、checkpoint/恢复、最终结果 |
| PREREQUISITE_WORKFLOW | 对应方法接入、SourceDescriptor、consumer validator 及注册 | source adapter、producer 专用入口；不是每个方法都要 producer | 缺失/无效/READY、schema/identity、执行前重新验证 |

### 共同登记顺序

1. 实现范围：先写清论文机制、当前支持边界、数据要求与未完成项。
2. YAML：使用已支持的配置字段，选择真实存在的 runner 和组件。
   自定义 YAML 可通过 `lnl run --config ...` 使用，但不会自动进入内置 catalog。
3. Recipe：在 `src/lnl_toolbox/cli/data/recipe_catalog.json` 的 `recipes` 登记路径；
   按真实需要设置 `public`/`conditional`。recipe ID 由 YAML 文件名派生，必须避免冲突。
4. Paper：在 `src/lnl_toolbox/paper_catalog.json` 登记论文 metadata 和实际 recipe ID；
   paper ID 不等于 `method` 路由名。复用 supervised 时不要直接填写一个未知 method ID。
5. Packaging：发行配置应列入 `pyproject.toml` 的 setuptools data-files；
   源码 checkout 能读取配置，不证明安装包已包含它。
6. Requirements：在 runner provider 声明实际数据/方法合同，不根据论文标题猜测。
   新 loss 必须显式审计 requirements；不能因为 factory 可构造就默认其具有 CE 的合同。
7. Web：论文列表来自 backend catalog，`web/paper_presentation.json` 是可选展示覆盖；
   没有覆盖时应回退原 metadata。新增专用控件不是登记论文的必要步骤。
8. Tests/docs：补对应测试与复现说明；保持 implemented、engineering/smoke、
   paper-oriented、conditional 与 numerical reproduction verified 的区别。

### 专用 runner 与 prerequisite

新 runner 在 `training/runners.py` 显式登记，声明 requirements、resume 支持和预算路径。
需要新增 `method` 路由、配置 validator 或顶层 schema 字段时，逐项列入任务 allowlist，
不能认为注册一个 RunnerSpec 会自动完成这些工作。CLI 应继续复用通用 run/validate，
不按论文名字新增分支。

外部输入复用 `SourceDescriptor`、readiness 和 consumer 校验；路径存在不等于 READY。
schema、身份、role、manifest 和 mapping 的要求由真实 consumer 决定。
producer 的 prepare/train 与 Student execution 分开；不要隐式启动 producer。

### 验收与禁止事项

- 先做静态登记一致性检查，再做配置/组件测试和轻量 fixture preflight。
- 未知组件、缺失输入、无效 artifact 必须有可检测错误；不得伪造 requirements。
- 检查正式 execution admission、completed resume no-op、unfinished resume 重新准入。
- 有恢复能力时验证 RNG/optimizer/scheduler/方法私有状态；不支持时明确声明。
- 不为登记论文修改现有算法公式、训练循环、checkpoint/resume 语义或 dataset 行为。
- 不修改现有论文参数，不以 smoke 证明数值复现，不将数据/权重/生成运行提交到 Git。
- 新增 registry/catalog/Web 元数据属于共享文件修改，先协调并批准精确 allowlist。

开发入口见后续章节；用户命令合同以 [CLI 使用手册](toolbox-usage-manual.md) 为准。

登记材料准备完成后，在已安装项目的环境运行：

```powershell
python scripts/check_paper_registration.py
python -m unittest tests.test_paper_registration -v
```

checker 只检查内置清单、YAML、runner/requirements 与正式配置打包引用，不执行
preflight、数据加载或 artifact consumer。PASS 不等于数据就绪、算法正确或数值复现。
`clean` 的 requirements 按其真实委托关系检查 supervised provider，不生成第二份声明。

### 生成接入草稿（不会注册论文）

```powershell
python scripts/new_paper.py new-paper-demo --extension-type COMPONENT_EXTENSION
python scripts/check_paper_registration.py --draft artifacts/paper-drafts/new-paper-demo
```

默认只在 `artifacts/paper-drafts/<paper-id>/` 生成 paper metadata、fragment YAML、
recipe/package 清单片段和 reproduction/checklist 草稿；拒绝覆盖已有目录。
fragment 不是可训练实验，不生成虚假 loss、requirements 或 prerequisite。
checker 对未完成 TODO 返回 1，这是草稿尚未完成，不代表生成失败。

可用 `--output-dir` 指定独立草稿根目录，或明确使用
`--base-config configs/experiment/cifar10_clean_smoke.yaml` 复制现有实验。
复制不会缩短预算或改变训练值，也不会实现新论文；草稿标为 engineering/experiment、
unavailable、not_run，必须由开发者审阅。生成器不写正式 catalog、源码或打包配置。
完成实现和测试后再人工整合片段并运行仓库一致性检查；生成成功不等于注册成功。

## 1. 开发环境

推荐为本项目创建独立 Conda 环境，避免与 Cleanlab、Ultralytics 等其他项目的依赖混在一起。

```powershell
conda create -n lnl-toolbox python=3.11 -y
conda activate lnl-toolbox
```

当前已验证的环境为：

- Python 3.11.15
- PyTorch 2.5.1
- torchvision 0.20.1
- CUDA Runtime 12.1
- NumPy 2.0.1
- PyYAML 6.0.3
- Pillow 11.1.0

## 2. 安装依赖

### NVIDIA GPU（CUDA 12.1）

先安装对应 CUDA 版本的 PyTorch：

```powershell
pip install torch==2.5.1 torchvision==0.20.1 `
  --index-url https://download.pytorch.org/whl/cu121
```

再安装其余固定版本依赖：

```powershell
pip install -r requirements.txt
```

最后以 editable 模式安装项目：

```powershell
pip install -e . --no-deps
```

### 仅使用 CPU

```powershell
pip install torch==2.5.1 torchvision==0.20.1 `
  --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
```

`requirements.txt` 固定版本用于复现实验；`pyproject.toml` 中的版本范围用于描述工具包兼容性。
两条安装路径都包含需要 RandAugment 的公开方法依赖；修改依赖时必须同步维护二者。

## 3. 检查环境

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

预期 GPU 环境输出 PyTorch 版本、`True` 和显卡名称。如果 CUDA 不可用，训练配置中的 `device: auto` 会回退到 CPU。

## 4. 准备数据

数据文件不提交到 Git。请使用以下目录：

```text
data/
├── cifar10/
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── test_batch
│   └── batches.meta
└── cifar100/
    ├── train
    ├── test
    └── meta
```

`data/` 已加入 `.gitignore`。

下载 CIFAR 官方 Python 版本后，只复制解压目录中的数据文件，不要提交压缩包或数据集。
统一 CLI 不会在训练时自动下载数据；普通用户使用 `lnl run <source> --dry-run`
执行与正式训练一致的数据 preflight。`lnl validate --check-data` 主要用于配置开发、
CI 和高级排错。

## 5. 运行测试

项目当前使用 Python 标准库 `unittest`，不要求安装 pytest：

```powershell
python -m unittest discover -s tests -v
```

提交代码前应保证完整测试全部通过。

## 6. 运行训练

人工启动实验时，可以无参数进入交互向导：

```powershell
lnl-train
```

向导先选择现有 YAML，再覆盖常用参数并展示最终配置；确认默认为否。只要命令中出现任意参数，就完全使用 argparse 非交互模式，不会在批处理过程中等待输入。

快速 GPU/CPU smoke test：

```powershell
lnl-train --config configs/experiment/cifar10_smoke.yaml
```

等价的模块运行方式：

```powershell
lnl run `
  --config configs/experiment/cifar10_smoke.yaml
```

完整 CIFAR-10 配置：

```powershell
lnl-train --config configs/experiment/cifar10_clean.yaml
```

每次运行会在 `artifacts/runs/` 下生成：

- `resolved_config.yaml`：最终采用的配置；
- `environment.json`：Python、PyTorch、CUDA、GPU 和 seed；
- `metrics.jsonl`：逐 epoch 指标；
- `last.pt`：模型、优化器、RunState 和算法状态；
- `final_metrics.json`：最终测试指标。

## 7. 恢复训练

`--epochs` 表示恢复后的总 epoch 目标，而不是额外训练轮数。例如，从已经完成 2 个 epoch 的 checkpoint 继续到第 3 个 epoch：

```powershell
lnl-train `
  --config configs/experiment/cifar10_smoke.yaml `
  --resume artifacts/runs/cifar10_smoke/last.pt `
  --epochs 3
```

恢复运行会继续使用 checkpoint 所在的运行目录，并在原有 `metrics.jsonl` 后追加指标。

## 8. 无需激活 Conda 的运行方式

自动化工具可以在激活环境后直接调用解释器：

```powershell
python -m unittest discover -s tests -v
```

不要把本机 Python、数据或临时运行目录的绝对路径写进项目配置、源码或用户文档。

## 9. 开发约定

- 通用协议放在 `src/lnl_toolbox/core/`，不要在核心层写死 PyTorch、CIFAR 或标签噪声假设。
- 数据适配器放在 `data/` Python 模块中，原始数据保存在仓库根目录的 `data/`。
- 模型、loss、算法和 evaluator 保持可替换，论文方法优先实现为插件。
- Dataset 必须返回稳定的全局 `index`，方便后续噪声清单和样本选择算法对应样本。
- 新功能应补充 `unittest`，并保证已有测试不回归。
- 实验结果、checkpoint 和数据文件不得提交到 Git。

## 10. 依赖维护

只有源码直接使用、或运行训练必需的包才应加入 `requirements.txt`。更新关键依赖后，需要重新运行：

```powershell
python -m unittest discover -s tests -v
lnl-train --config configs/experiment/cifar10_smoke.yaml
```

不要直接使用完整 Conda 环境的 `pip freeze` 覆盖 requirements；那会把其他项目的 Cleanlab、OpenCV、Pandas、Ultralytics 等无关依赖一起带入。

## 11. 统一实验服务与质量门禁

用户入口统一调用 `ExperimentService`；runner 只负责训练生命周期，服务层负责共享
preflight、标准运行元数据和 `final_metrics.json` Result Contract。新增 runner 时应在
`training/runners.py` 注册自身的预算路径与 `RunPlan`，CLI 不得按论文名称分支。

常用命令：

```powershell
lnl run cifar10-symmetric-ce-smoke --set trainer.epochs=2 --dry-run
lnl run configs/experiment/cifar10_symmetric_ce_smoke.yaml
lnl sweep cifar10-symmetric-ce-smoke --seeds 1 2 3
lnl sweep cifar10-clean-smoke `
  --matrix 'loader.batch_size=[256,512]' `
  --matrix 'optimizer.lr=[0.01,0.001]'
lnl sweep sweep-spec.yaml --dry-run
lnl sweep status artifacts/sweeps/example
lnl compare artifacts/sweeps/example
lnl report artifacts/sweeps/example
```

所有活动 YAML 使用 `schema_version: 1`。完整 recipe 写 `kind: experiment`，参数片段写
`kind: fragment`。不要在公开 recipe 中写本机 CIFAR/MNIST root；先执行
`lnl data register`，只有在同一 adapter 登记多份数据时才通过 `--data <alias>` 选择。
配置读取会拒绝未知顶层字段和旧式 `noise.type`、顶层 `epochs/batch_size/learning_rate`。
原始迁移前配置位于 `archive/configs-legacy-2026-08-21/`，仅供恢复和比对。

`--dry-run` 默认验证数据与外部 artifact，只跳过真正训练和 checkpoint 写入；仅当
数据尚未准备时才使用 `--no-check-data`。Sweep matrix 必须调用
`core.config_overrides` 的 dotted-path override，不得自行修改嵌套配置。每个任务由
seed、resolved override 和 config hash 共同标识。

CLI matrix 值必须是 JSON 数组并保持原始数值、布尔或字符串类型。未传 `--seeds`
时只使用配置自身的 seed；显式 seeds 会与所有参数维度继续做笛卡尔积。Web 的
“参数 Sweep”可从内置 recipe 或项目 YAML 生成路径菜单，并在执行前显示完整运行计划。
参数路径、分组、用途和修改权限均来自参数元数据 Registry；服务端拒绝未知、锁定或
类型不兼容的矩阵值。项目 YAML 必须先保存并通过配置校验，不能把未保存文本直接交给
训练或 Sweep。

Web 的“运行管理”直接读取 `metrics.jsonl`、`final_metrics.json` 和 Sweep manifest，
可查看最终指标、部分训练进度及多运行逐 epoch 叠加曲线。路径按钮通过本机 Windows
对话框选择文件或目录；该能力只允许 localhost 请求，非 Windows 环境继续手工输入。

论文正式配置的 Web 参数面板以 `web/lnl_parameter_metadata_registry_revised.yaml` 为权限和
解释来源。基础参数与论文参数默认展开，高级参数默认折叠，锁定参数只读展示。修改
论文参数必须确认复现影响，保存后的 `meta.web_parameter_record` 会记录原值、新值、
formal recipe 和 `modified_from_paper`；该字段专供 Web 展示，不能占用训练参数抽样使用
的顶层 `parameter_record`。无论参数控件还是完整 YAML 文本，锁定字段都由服务端
再次比较并拒绝修改。普通 WebUI 不得提供绕过该校验的 runner、variant、protocol、
fidelity、组件 wiring、模型选择或数据/噪声协议修改入口。

Web 页面维护唯一的当前配置来源（内置 recipe 或项目 YAML 路径）。论文页、YAML
编辑器、Sweep 和运行管理之间只传递这个来源与输出目录；内置 recipe 的编辑动作必须
先另存副本，项目 YAML 不得反向注册成内置 recipe。

比较层把 `group_by` 视为允许变化的研究维度，把 `require_equal` 视为同组公平比较
必须一致的条件。Noise Manifest 只在同一 seed 和可比条件下跨方法核对；不同 seed
不要求共享 manifest。`lnl report` 必须直接消费 `lnl compare` 使用的同一比较结果，
不得另写 aggregation 或 fairness 逻辑。

CI 在 Python 3.10 和 3.12 上执行 Ruff、完整 unittest、CLI 测试和 coverage；发布 job
分别从 wheel 与 sdist 安装，并验证 `lnl --help`、公开 recipe discovery 和 VolMinNet
smoke 配置预检。
# 注册第三方数据源（2026-08-18）

新增数据集时实现 `DatasetAdapter`：`validate(DataSpec)` 负责路径和布局检查，`load(DataSpec, split, seed=...)` 返回 `RawDatasetSplit`。随后在 `create_dataset_registry()` 的数据源注册函数中登记名称和别名。适配器不得包含论文方法、训练阶段或 optimizer 逻辑。

必须保证：global index 在 shuffle/subset/view 后稳定；observed 与 clean label 分开保存；真实噪声训练 split 不向 batch 暴露 clean label；缺失数据只输出本地准备说明，不下载、不回退。需要持久化预处理或 split 状态的适配器可提供 `identity_artifacts()`，由统一服务写入并验证 run-local JSON。

runner 只允许调用 `prepare_experiment_data()`，不得直接导入 CIFAR reader、`TorchCifarDataset` 或自行构造 `DataLoader`。新增/修改 runner 后运行 `tests/test_data_service.py` 的静态门禁。

## 机器本地数据登记与可用性证据（2026-08-20）

`data/local_catalog.py` 只保存本机路径和验证证据，不保存数据本身。新增适配器登记项时，
必须维持 `registered -> layout_validated -> training_verified` 的严格语义；文件签名改变后，
既有训练证据必须转为 `verification_stale`。不得因为注册成功或一次 `validate()` 成功就宣称
数据可训练。

CLI、Web、doctor、dry-run、run 和 sweep 不得直接读取 `DatasetRegistry`、
`LocalDatasetCatalog` 或自行判断文件布局；它们必须调用 `DataService`。新增数据管理能力时，
readiness 检查必须真实加载 train/test，不能用 `Path.exists()` 代替 adapter 校验。Web
focused tests 必须与 `tests/` 一起进入 CI。

每个新增文件格式至少需要：官方结构的临时 fixture、通过 `ExperimentService` 的实际
1 epoch、`data_manifest.json`、epoch 指标，以及训练 batch 不泄漏 clean target 的既有门禁。
本机真实数据若可用，还应再通过 `lnl data verify`。外部数据格式依据应来自数据集发布方
或论文课题组官方仓库；未经上述实验，不在文档中标记为训练可用。

## 声明论文方法的数据要求（2026-08-22）

新增论文或新增 runner 配置时，必须在中央 runner registry 提供 `MethodRequirements`。
声明应反映当前可执行链路，而不是论文理论上可能支持的数据：包括 modality、类别数、
manifest、clean/trusted supervision、validation target、噪声率先验和外部来源。多字段
先验或“矩阵/产物任选其一”使用 `ConfigInputRequirement`，不得在兼容性层计算论文公式、
改写配置或补默认值。共享 runner 必须按组件配置识别方法，不得按论文标题分支。

每个正式 recipe 都应有非空 requirements；测试还必须覆盖 observed-only native noise 的
失败边界，并冻结已有 provider 的结果。算法、adapter 和 experiment runner 的能力扩展应
作为独立任务审批，不能借修改兼容性元数据静默放宽。

### 兼容性 UI 的输入边界

新增 adapter 时，优先提供可选的 `semantic_hints`，只补充原始 train/test 无法观测的
语义事实。不要把模态、观测标签、稳定索引或原生噪声来源交给用户自由填写。新增
`required_user_inputs` 时，必须同时提供 Web/CLI 的渲染或自动解析路径；若是 runner 元数据
缺失，应返回开发者元数据错误，不应显示为用户输入框。方法噪声率先验必须由具体 recipe
声明真实配置路径，预训练要求必须验证实际文件/目录或已存在运行产物。
