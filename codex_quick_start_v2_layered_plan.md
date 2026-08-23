# Codex 执行计划：重做 Quick Start，使其真正体现 LNL-toolbox 能力

> 目标分支：`codex/cli`
>
> 目标用户：第一次打开 WebUI、只想尽快验证工具箱是否实用的科研工作者。
>
> 本计划假设执行 Agent 的自主判断能力有限，因此请严格按顺序执行，不要自行扩大范围，不要为了“顺手优化”重构其他模块。

---

# 0. 本任务只解决什么

当前 Quick Start 的选择逻辑有四个根本问题：

1. **数据集入口错误**  
   当前 Quick Start 主要从“已经登记且 ready 的数据集”中选择。  
   新手真正需要的是：**给一个路径 → 系统自动识别 → 自动登记 → 自动 inspect → 继续实验**。

2. **噪声入口错误**  
   当前 Quick Start 从“已有 reproduction recipe 中出现过的 `noise` 字符串”反推噪声选项。  
   这导致 `pdl`、`external_torch` 等内部标识直接暴露。  
   正确逻辑应该是：**从工具箱真实的 noise generator 能力生成新手可读的噪声选项**。

3. **方法入口错误**  
   当前 Quick Start 从“过滤后的已有 recipe”中抽取 method，所以没有现成 YAML 的算法几乎等于不存在。  
   正确逻辑应该是：**从论文/方法实现能力出发，检查当前 dataset + noise 下方法是否能运行**。

4. **“论文复现配置”和“工具箱适配配置”混淆**  
   某个方法支持 CIFAR-10，不代表仓库一定已有该方法在 CIFAR-10 + 当前噪声下的正式 reproduction YAML。  
   正确做法必须区分：
   - **论文复现配置**：仓库已有、正式登记的 reproduction recipe。
   - **Toolbox 适配配置**：基于现有方法实现和模板，为当前 dataset/noise 生成的可运行配置。

   **Toolbox 适配配置绝对不能标成论文复现。**

---

# 1. 本任务明确不做什么

以下内容不要修改：

- 不删除现有“新手教程”。
- 不删除现有“新建 YAML”。
- 不删除现有“本地数据集”。
- 不删除现有 Dataset-first compatibility 详细页面。
- 不删除参数 Sweep。
- 不删除运行管理。
- 不删除论文方法。
- 不删除自由输入。
- 不把现有模块藏到“高级模式”。
- 不修改第二个 `global_indices` / split-local sample identity bug。
- 不重写训练核心。
- 不批量修改论文 reproduction YAML。
- 不批量修改 `paper_catalog.json`。
- 不重新实现一套 compatibility system。
- 不重新实现 job polling / console / stop job。
- 不创建第二套 `/api/run`。
- 不因为 Quick Start 需求重构整个 `web/index.html`。
- 不把新的 Quick Start 逻辑继续堆进 `web/index.html`。

---

# 2. 最终用户流程

完成后，第一次打开 WebUI 的科研工作者应该能这样使用：

```text
打开 WebUI
↓
Quick Start

1. 数据集
选择一个目录：
F:\datasets\cifar10

[自动识别并继续]

系统：
✓ 识别为 CIFAR-10
✓ 已自动登记
✓ 已完成实际加载检查
✓ 50,000 train / 10,000 test
✓ 原始训练标签干净

↓

2. 标签噪声

● 保持干净
○ 添加人工噪声

如果选择人工噪声：

噪声类型：
[对称噪声]

噪声率：
[20%]

随机种子：
[1]

↓

3. 方法

可直接运行
- Co-teaching
- DivideMix
- VolMinNet
- ...

补充输入后可运行
- JoCoR
  需要：噪声率先验
- ...

当前实现不适用
- FINE
  原因：当前实现要求 100 类
- ...

↓

4. 配置来源

如果仓库有完全匹配的正式 reproduction：
✓ 论文复现配置

如果没有完全匹配：
✓ Toolbox 适配配置
  基于该方法已有模板，为当前数据集和噪声条件生成
  不宣称是论文原始复现

↓

[预演] [开始训练]
```

对于 Clothing1M / Animal10N 等官方天然噪声数据：

```text
1. 数据集
✓ 自动识别

2. 标签噪声
✓ 数据集本身包含真实世界标签噪声
  本次 Quick Start 直接使用原始 noisy labels

3. 方法
显示当前数据能力下所有已实现论文方法的状态
```

---

# 3. 总体代码分层

本任务必须按下面的层次拆分。

## 3.1 `web/index.html`

职责：

- WebUI 总壳。
- 左侧模块导航。
- 现有其他模块。
- 极少量 Quick Start 集成代码。

**禁止继续承担 Quick Start 的业务逻辑。**

---

## 3.2 `web/assets/quick_start.js`

职责：

- Quick Start 的前端状态。
- Quick Start 的 DOM 渲染。
- 调用 Quick Start API。
- 展示 dataset / noise / method / plan。
- 调用现有 `/api/run`。
- 与现有日志/job system 对接。

不负责：

- dataset 自动识别规则。
- compatibility 判断。
- noise backend 配置规则。
- method 能不能运行。
- YAML 科研语义。

---

## 3.3 `web/assets/quick_start.css`

职责：

- 仅保存 Quick Start 样式。

不修改全站基础主题。

---

## 3.4 `web/quick_start_api.py`

职责：

- 把 HTTP JSON 转给 `QuickStartService`。
- 把 service 返回结果转成 JSON payload。
- 做最基本输入检查。

**不写业务逻辑。**

---

## 3.5 `src/lnl_toolbox/quickstart/`

新建一个独立 package，作为 Quick Start 后端编排层。

目录：

```text
src/lnl_toolbox/quickstart/
├── __init__.py
├── models.py
├── templates.py
└── service.py
```

职责：

- 数据路径识别后的流程编排。
- noise selection 标准化。
- method capability 枚举。
- method config 适配。
- 区分 reproduction / toolbox-adapted。
- 生成最终 Quick Start plan。

---

## 3.6 `src/lnl_toolbox/data/probe.py`

职责：

- 只负责根据路径判断“这个路径最像哪个已注册 adapter”。
- 只做识别，不做 register。
- 只做识别，不做 inspect。
- 不修改 catalog。

---

## 3.7 `src/lnl_toolbox/noise/quickstart_catalog.py`

职责：

- 定义新手界面允许看到的 noise capability。
- 将内部 noise key 映射为人类可理解的名字。
- 明确哪些内部 noise source 不能出现在新手菜单。

---

# 4. 执行顺序总览

严格按以下顺序执行：

```text
阶段 1：数据集自动识别
阶段 2：新手噪声目录
阶段 3：Quick Start 后端模型
阶段 4：方法模板适配
阶段 5：Quick Start Service
阶段 6：Quick Start API
阶段 7：静态资源路由
阶段 8：Quick Start JS/CSS
阶段 9：瘦身 index.html
阶段 10：测试和 README
阶段 11：完整验收
```

**每一个阶段的 targeted tests 未通过，不要进入下一阶段。**

---

# 5. 阶段 1：建立数据集路径自动识别

## 5.1 新建文件

```text
src/lnl_toolbox/data/probe.py
```

---

## 5.2 文件职责

该文件只负责：

```text
用户给出路径
↓
判断这最像哪个现有 dataset adapter
↓
给出候选项
```

不要在这里：

- register。
- inspect。
- 写 catalog。
- 修改 dataset contract。
- 读取整套训练数据。
- 跑训练。

---

## 5.3 在 `probe.py` 新建数据结构

建议：

```python
@dataclass(frozen=True)
class ProbeCandidate:
    adapter: str
    confidence: str
    reason: str
    data: dict[str, str]
```

其中：

```text
adapter
```

必须是 `create_dataset_registry()` 中真实存在的 adapter key。

```text
confidence
```

只允许：

```text
high
medium
low
```

```text
data
```

直接采用未来 `DataService.register(name, adapter, data)` 所需要的结构。

---

再定义：

```python
@dataclass(frozen=True)
class DatasetProbeResult:
    path: str
    status: str
    candidates: tuple[ProbeCandidate, ...]
    existing_alias: str | None = None
```

`status`：

```text
detected
ambiguous
unsupported
already_registered
```

---

## 5.4 实现公共函数

```python
def probe_dataset_path(
    path: str | Path,
    *,
    data_service: DataService | None = None,
) -> DatasetProbeResult:
    ...
```

以及：

```python
def suggest_dataset_alias(
    adapter: str,
    path: str | Path,
    existing_names: Iterable[str],
) -> str:
    ...
```

---

## 5.5 自动识别规则

### CIFAR-10

识别典型结构：

```text
cifar-10-batches-py/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```

用户既可能选择：

```text
...\cifar10
```

也可能直接选择：

```text
...\cifar10\cifar-10-batches-py
```

probe 必须兼容这两种输入。

候选 adapter：

```text
cifar10
```

---

### CIFAR-100

识别：

```text
cifar-100-python/
├── train
├── test
└── meta
```

候选：

```text
cifar100
```

---

### CIFAR-N

检查 CIFAR 基础数据结构后，再检查对应人工标签文件。

至少识别现有 adapter 使用的官方名称：

```text
CIFAR-10_human.pt
CIFAR-100_human.pt
```

如果存在，优先返回：

```text
cifar10n
```

或：

```text
cifar100n
```

`ProbeCandidate.data` 中必须把：

```text
root
noise_path
```

都填好。

如果同一目录既可作为 CIFAR-10 clean，又可作为 CIFAR-10N：

- CIFAR-N 为高优先候选。
- clean CIFAR 可作为第二候选。
- 如果代码无法确定用户意图，返回 `ambiguous`，不要偷偷选。

---

### MNIST / FashionMNIST

识别官方 IDX / IDX.GZ 结构。

考虑现有 adapter 可接受的目录形态，例如：

```text
root/raw/
```

或：

```text
root/MNIST/raw/
```

或：

```text
root/FashionMNIST/raw/
```

只根据真实文件签名识别。

不要仅根据目录名中包含 `mnist` 来判断。

---

### Clothing1M

使用现有 `real_noise.py` 所要求的文件作为强签名。

至少检查类似：

```text
noisy_train_key_list.txt
clean_val_key_list.txt
clean_test_key_list.txt
noisy_label_kv.txt
clean_label_kv.txt
```

满足核心签名时：

```text
adapter = clothing1m
confidence = high
```

---

### Animal10N

使用当前 adapter 已支持的数据结构识别：

- official binary layout；或
- train/training + test/testing image directory layout。

不要仅根据路径名称 `animal10n` 判断。

---

### UCI binary

如果用户选择的是单文件：

- 可以返回 UCI candidate；
- 但 confidence 不要轻易设为 high；
- 如果现有 loader 能做轻量 schema 检查，可用于增强 confidence；
- 不要为了 probe 跑训练。

---

## 5.6 已登记路径优先

如果路径已经存在于 local dataset catalog：

返回：

```text
status = already_registered
existing_alias = <existing alias>
```

不要重复注册第二份。

---

## 5.7 自动 alias

例如：

```text
cifar10-local
cifar100-local
clothing1m-local
```

如果冲突：

```text
cifar10-local-2
cifar10-local-3
```

用户 Quick Start 不需要自己填 alias。

---

# 6. 阶段 1 测试

## 6.1 新建文件

```text
tests/test_data_probe.py
```

---

## 6.2 必须覆盖

### Test A

临时目录构造 CIFAR-10 签名。

要求：

```text
status = detected
adapter = cifar10
confidence = high
```

### Test B

CIFAR-100。

### Test C

CIFAR-10 + `CIFAR-10_human.pt`。

要求能识别 CIFAR-N candidate。

### Test D

Clothing1M 文件签名。

### Test E

Animal10N 文件结构。

### Test F

路径已登记。

要求：

```text
already_registered
existing_alias != None
```

### Test G

无法识别目录。

要求：

```text
unsupported
```

不能随便猜 adapter。

### Test H

多个候选合理存在。

要求：

```text
ambiguous
```

不能擅自选择。

---

## 6.3 阶段完成命令

```bash
python -m unittest tests.test_data_probe -v
```

不通过就停止。

---

# 7. 阶段 2：建立真正的新手噪声能力目录

## 7.1 新建文件

```text
src/lnl_toolbox/noise/quickstart_catalog.py
```

---

## 7.2 文件职责

当前 Web Quick Start 的错误是：

```text
从已有 recipe.meta.noise
→ 直接生成下拉菜单
```

本文件要替代这种逻辑。

它负责定义：

> 工具箱本身有哪些适合在 Quick Start 展示的人工噪声能力。

---

## 7.3 新建数据结构

建议：

```python
@dataclass(frozen=True)
class QuickStartNoiseSpec:
    key: str
    label: str
    description: str
    backend_name: str | None
    visible: bool
    requires_rate: bool
    requires_seed: bool
    category: str
```

`category` 可使用：

```text
clean
synthetic
native
source_only
```

---

## 7.4 至少定义以下用户可见能力

根据仓库当前真实 generator：

```text
clean
symmetric
pairflip
class_conditional
instance_dependent
pdl
```

显示名称不要直接使用内部 key：

```text
clean
→ 保持干净标签

symmetric
→ 对称噪声

pairflip
→ Pair-flip 噪声

class_conditional
→ 类条件噪声

instance_dependent
→ 实例依赖噪声

pdl
→ PDL 实例依赖噪声
```

每项提供一行简短说明。

---

## 7.5 明确禁止出现在新手噪声菜单

例如当前 recipe 中可能出现：

```text
external_torch
official_uniform_flip
binary_asymmetric_rcn
```

如果它们是：

- 外部标签文件来源；
- 官方固定标签源；
- 特定实验协议；

则不能伪装成普通“噪声类型”。

可以登记为：

```text
visible = False
category = source_only
```

但 Quick Start UI 不显示。

---

## 7.6 提供公共函数

```python
def quick_start_noise_specs() -> tuple[QuickStartNoiseSpec, ...]:
    ...
```

```python
def visible_synthetic_noise_specs() -> tuple[QuickStartNoiseSpec, ...]:
    ...
```

```python
def quick_start_noise_spec(key: str) -> QuickStartNoiseSpec:
    ...
```

以及：

```python
def build_noise_config(
    key: str,
    *,
    rate: float | None,
    seed: int | None,
) -> dict[str, object] | None:
    ...
```

---

## 7.7 关键约束

`build_noise_config()` 必须复用仓库现有训练 config contract。

不要自己发明：

```yaml
quick_noise:
```

这种新格式。

它最终必须生成当前 `DataService` / noise manifest 流程已经能理解的结构。

---

# 8. 阶段 2 测试

## 8.1 新建文件

```text
tests/test_quickstart_noise_catalog.py
```

---

## 8.2 必须覆盖

- `symmetric` 显示中文可读名字。
- `external_torch` 不在 beginner-visible 列表。
- `clean` 不生成 synthetic noise config。
- `symmetric + rate=0.2` 生成合法配置。
- rate < 0 或 > 1 时失败。
- 需要 rate 的 noise 没有 rate 时失败。

---

## 8.3 阶段完成命令

```bash
python -m unittest tests.test_quickstart_noise_catalog -v
```

不通过就停止。

---

# 9. 阶段 3：建立 Quick Start 后端 package

## 9.1 新建目录

```text
src/lnl_toolbox/quickstart/
```

---

## 9.2 新建文件

```text
src/lnl_toolbox/quickstart/__init__.py
```

功能：

只导出稳定公共接口。

建议最终导出：

```python
QuickStartService
QuickStartPlan
QuickStartMethodOption
```

不要把内部 helper 全部 export。

---

## 9.3 新建文件

```text
src/lnl_toolbox/quickstart/models.py
```

---

## 9.4 `models.py` 功能

这里仅存放 DTO / dataclass。

不要访问磁盘。

不要加载 YAML。

不要调用 DataService。

---

## 9.5 建议数据结构

### `QuickStartDatasetSummary`

字段至少：

```text
alias
adapter
path
display_name
num_classes
train_size
validation_size
test_size
noise_status
noise_origin
clean_train_labels
```

---

### `QuickStartNoiseSelection`

字段至少：

```text
kind
key
rate
seed
```

`kind`：

```text
clean
synthetic
native
```

---

### `QuickStartMethodOption`

字段至少：

```text
paper_id
acronym
title
summary
venue
year
status
reasons
required_user_inputs
```

`status` 只允许：

```text
ready
needs_input
unsupported
metadata_error
```

---

### `QuickStartPlan`

字段至少：

```text
plan_id
dataset_alias
paper_id
method
noise
config_kind
recipe_id
generated_config_path
status
required_user_inputs
command
dry_run_command
summary
```

`config_kind`：

```text
paper_reproduction
toolbox_adapted
```

---

# 10. 阶段 4：实现方法模板和配置适配

## 10.1 新建文件

```text
src/lnl_toolbox/quickstart/templates.py
```

---

## 10.2 文件职责

解决核心问题：

> “方法实现支持当前数据集，但仓库未必正好存在当前 dataset + noise 的 reproduction YAML。”

本文件负责：

1. 找一篇论文已有的正式 config，作为**方法模板来源**；
2. 基于当前 dataset + noise 生成候选配置；
3. 判断当前条件是否正好匹配已有 reproduction；
4. 保证适配配置不会被错误叫做 reproduction。

---

# 11. 在 `templates.py` 实现“方法模板来源”

## 11.1 不新增第二份论文目录

用户显示名必须继续来自当前：

```text
paper_catalog.json
```

不要再新建：

```text
quickstart_methods.json
```

---

## 11.2 新建函数

建议：

```python
def method_template_for_paper(
    paper: Mapping[str, object],
) -> MethodTemplate:
    ...
```

`MethodTemplate` 可以是本文件内部 dataclass。

模板优先级：

1. paper catalog 中明确的 default reproduction recipe；
2. 若 default 不存在，取第一份合法 reproduction config；
3. 若一篇已实现论文完全没有任何 config，则返回 metadata error，不要凭空造 config。

---

# 12. 实现 exact reproduction 查找

新增：

```python
def find_exact_reproduction(
    paper: Mapping[str, object],
    *,
    dataset_adapter: str,
    noise_selection: QuickStartNoiseSelection,
) -> str | None:
    ...
```

目的：

如果 paper catalog 中已有一份正式 reproduction config：

```text
dataset = 当前 dataset 类型
noise = 当前选择
```

完全匹配，则直接返回其 `recipe_id`。

此时：

```text
config_kind = paper_reproduction
```

---

# 13. 实现 toolbox-adapted config

新增：

```python
def adapt_method_template(
    base_config: Mapping[str, object],
    *,
    dataset_alias: str,
    dataset_profile: Mapping[str, object],
    noise_selection: QuickStartNoiseSelection,
    data_service: DataService,
    method_inputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    ...
```

---

## 13.1 规则：数据集

复用现有 DataService / ExperimentService 的 local dataset apply 机制。

不要手写：

```python
config["data"]["root"] = ...
```

如果仓库已有公共方法能把 `--data alias` 应用到 config，使用该方法。

目标是得到：

```text
当前 local dataset
```

的实际 config。

---

## 13.2 规则：clean

如果用户选择：

```text
保持干净
```

候选配置不得残留 base recipe 的：

```text
external_torch
symmetric
pdl
...
```

人工噪声来源。

要把 top-level noise 状态重置为当前训练框架认可的 clean/no-noise 表达。

---

## 13.3 规则：synthetic

如果选择：

```text
symmetric 0.2
```

则使用 `quickstart_catalog.build_noise_config()`。

不要从 base recipe 保留另一个噪声源。

---

## 13.4 规则：native

如果数据集 profile 表明：

```text
noise_origin = native
```

Quick Start 默认使用 dataset observed labels。

不要叠加 synthetic noise。

不要保留 base recipe 中原本针对 CIFAR 的 synthetic/external noise source。

---

## 13.5 规则：方法 noise-rate prior

如果 runner requirements 声明：

```text
requires_method_noise_prior = True
```

且用户选择的 synthetic noise rate 已知，比如：

```text
0.2
```

则 Quick Start 可以将：

```text
method noise-rate prior
```

默认设为：

```text
0.2
```

因为这是人工生成噪声，工具箱知道实际设置的 rate。

但是必须保留用户修改入口。

---

对于天然噪声：

```text
noise rate unknown
```

时，不允许默认填 0。

如果方法需要 prior：

```text
status = needs_input
required_user_inputs includes noise_rate_prior
```

---

# 14. 科研语义：绝不把适配配置冒充复现

只要发生以下任何一项：

- dataset 与原 reproduction 不同；
- noise 类型不同；
- noise rate 不同；
- noise source 从 external 改成 generated；
- config 被 Quick Start 派生；

则：

```text
config_kind = toolbox_adapted
```

UI 必须显示：

```text
Toolbox 适配配置
```

并说明：

```text
使用该论文/方法的现有实现和基础训练模板，
为当前数据集与噪声条件生成。
不等同于论文原始复现实验。
```

---

# 15. 阶段 5：实现 QuickStartService

## 15.1 新建文件

```text
src/lnl_toolbox/quickstart/service.py
```

---

## 15.2 文件职责

这是整个 Quick Start 的业务编排层。

Web 不应该绕过它。

---

# 16. `QuickStartService.probe()`

实现：

```python
def probe(self, path: str) -> DatasetProbeResult:
    ...
```

直接调用：

```python
probe_dataset_path()
```

---

# 17. `QuickStartService.register_and_inspect()`

实现：

```python
def register_and_inspect(
    self,
    path: str,
    *,
    selected_adapter: str | None = None,
) -> QuickStartDatasetSummary | DatasetProbeResult:
    ...
```

行为：

### 情况 A：路径已经登记

直接复用 existing alias。

如果已经 ready：

直接返回 summary。

如果 registered 但未 inspect：

自动 inspect。

---

### 情况 B：唯一 high-confidence candidate

自动：

```text
suggest alias
↓
DataService.register(...)
↓
DataService.inspect(...)
↓
summary
```

用户不需要输入 alias。

用户不需要选择 adapter。

---

### 情况 C：ambiguous

返回：

```text
status = ambiguous
candidates = [...]
```

Web 再让用户选择一次。

不要擅自 register。

---

### 情况 D：unsupported

返回：

```text
无法自动识别
```

Web 给用户：

```text
打开“本地数据集”手动登记
```

不要让 Quick Start 无限增加特殊表单。

---

# 18. `QuickStartService.noise_options()`

实现：

```python
def noise_options(
    self,
    dataset_alias: str,
) -> dict[str, object]:
    ...
```

规则：

---

## 18.1 clean dataset

返回：

```text
dataset_state = clean

options:
- clean
- visible synthetic noise specs
```

---

## 18.2 native-noise dataset

返回：

```text
dataset_state = native

options:
- native
```

Quick Start 不主动允许 synthetic overlay。

需要叠加噪声的高级用户去“新建 YAML”。

---

## 18.3 unknown dataset

返回：

```text
dataset_state = unknown
requires_confirmation = True
```

Web 只问：

```text
标签干净
本身有噪声
不知道
```

确认使用现有 dataset declaration API / DataService declaration 机制。

---

# 19. `QuickStartService.method_options()`

这是最关键的一步。

实现：

```python
def method_options(
    self,
    dataset_alias: str,
    noise_selection: QuickStartNoiseSelection,
) -> tuple[QuickStartMethodOption, ...]:
    ...
```

---

## 19.1 方法来源必须是论文 catalog

不要再从：

```text
compatible reproduction recipes
```

中抽 method。

应该：

```text
遍历已实现 paper catalog
```

每篇论文都尝试建立 candidate config。

---

## 19.2 用户显示信息

必须来自 paper catalog：

```text
acronym
title
summary
venue
year
```

禁止用：

```text
runner name
raw method id
noise source id
```

作为主显示名。

---

## 19.3 对每篇论文构造 candidate config

流程：

```text
paper
↓
找到 method template
↓
应用当前 local dataset
↓
应用当前 noise selection
↓
补上可以自动推导的 method input
↓
得到 concrete candidate config
```

---

## 19.4 用现有 compatibility 判断

不要在 QuickStartService 重新写：

```text
if CIFAR10 and CoTeaching...
```

必须把 concrete candidate config 交给现有：

```python
ExperimentService.list_config_compatibility(...)
```

或等价的 concrete config compatibility 公共逻辑。

---

## 19.5 状态映射

现有 compatibility：

```text
compatible
```

→

```text
ready
```

---

```text
compatible_with_requirements
```

→

```text
needs_input
```

---

```text
incompatible
```

→

```text
unsupported
```

---

method requirements 缺失：

```text
metadata_error
```

---

# 20. 方法不能因为“没有 exact YAML”而消失

这是强制验收要求。

例如当前 dataset 是 CIFAR-10：

如果某方法：

- modality 支持 image；
- class count 支持 10；
- 当前 label/noise capability 满足；
- runner requirements 满足；

即使没有：

```text
这个方法 + CIFAR10 + symmetric20
```

的 exact reproduction YAML，

它仍然应该出现在：

```text
可直接运行
```

或：

```text
补充输入后可运行
```

中。

---

# 21. 真正不兼容的方法仍然保留显示

例如当前代码 FINE requirements 明确：

```text
exact_classes = 100
```

那么 CIFAR-10 上应该显示：

```text
FINE
当前实现不适用
原因：当前实现要求 100 个类别
```

而不是：

- 消失；
- 被错误标绿。

---

# 22. `QuickStartService.build_plan()`

实现：

```python
def build_plan(
    self,
    *,
    dataset_alias: str,
    noise_selection: QuickStartNoiseSelection,
    paper_id: str,
    user_inputs: Mapping[str, object] | None = None,
) -> QuickStartPlan:
    ...
```

---

## 22.1 先找 exact reproduction

若存在完全匹配的 official reproduction：

返回：

```text
config_kind = paper_reproduction
recipe_id = ...
```

运行命令：

```bash
lnl run --recipe <recipe_id> --data <dataset_alias> --check-data
```

---

## 22.2 没有 exact reproduction

生成：

```text
toolbox_adapted
```

配置。

---

## 22.3 生成配置存储路径

建议：

```text
artifacts/web-quick-start/configs/
```

例如：

```text
artifacts/web-quick-start/configs/
  20260823-131500-cifar10-coteaching-symmetric020.yaml
```

不要写回：

```text
configs/papers/
```

不要污染正式 recipe。

---

## 22.4 同时保存 provenance sidecar

例如：

```text
artifacts/web-quick-start/configs/
  20260823-131500-cifar10-coteaching-symmetric020.json
```

内容至少：

```json
{
  "kind": "toolbox_adapted",
  "dataset_alias": "...",
  "paper_id": "...",
  "base_recipe": "...",
  "noise": {...},
  "generated_at": "..."
}
```

这样未来用户知道这不是官方 reproduction。

---

## 22.5 生成前必须验证

candidate config 必须经过：

1. config validation；
2. `ExperimentService.preflight(..., check_data=True)`。

只有通过后：

```text
status = ready
```

否则返回：

```text
needs_input
unsupported
```

或真实错误。

---

## 22.6 toolbox-adapted command

使用：

```bash
lnl run --config <generated_yaml> --check-data
```

预演：

```bash
lnl run --config <generated_yaml> --dry-run --check-data
```

不要伪造 recipe id。

---

# 23. 阶段 5 单元测试

## 23.1 新建文件

```text
tests/test_quickstart_service.py
```

---

## 23.2 必须覆盖

### Dataset

- 新路径能 probe → register → inspect。
- 已登记路径不会重复 register。
- ambiguous 不会自动 register。
- unsupported 不会伪造 ready。

### Noise

- clean CIFAR 可以选 synthetic noise。
- synthetic noise 选项与是否存在 reproduction YAML 无关。
- native dataset Quick Start 不提供额外 synthetic overlay。
- `external_torch` 不作为普通 noise option。

### Methods

- 方法列表来自 paper/method capability，而不是剩余 reproduction YAML。
- 没有 exact CIFAR-10 recipe 的 image method，在结构兼容时仍能显示。
- FINE 在 CIFAR-10 上保持 unsupported。
- incompatible method 仍可显示原因。
- 主显示字段是 acronym/title/summary。

### Prior

- Co-teaching / DivideMix 等方法若要求 method noise prior：
  - synthetic rate 已知时可默认等于该 rate；
  - native unknown rate 时必须 `needs_input`。

### Plan

- exact match → `paper_reproduction`。
- nonexact match → `toolbox_adapted`。
- toolbox-adapted 不写入正式 configs。
- toolbox-adapted command 使用 `--config`。
- candidate preflight 不通过时不能返回 ready。

---

## 23.3 阶段完成命令

```bash
python -m unittest tests.test_quickstart_service -v
```

不通过就停止。

---

# 24. 阶段 6：增加独立 Quick Start Web API 层

## 24.1 新建文件

```text
web/quick_start_api.py
```

---

## 24.2 文件职责

只做：

```text
JSON 输入
↓
QuickStartService
↓
JSON 输出
```

不要把 dataset probe / method adaptation 逻辑放这里。

---

## 24.3 建议函数

```python
def probe_payload(body: Mapping[str, object]) -> dict[str, object]:
    ...
```

```python
def register_payload(body: Mapping[str, object]) -> dict[str, object]:
    ...
```

```python
def noise_options_payload(alias: str) -> dict[str, object]:
    ...
```

```python
def method_options_payload(body: Mapping[str, object]) -> dict[str, object]:
    ...
```

```python
def plan_payload(body: Mapping[str, object]) -> dict[str, object]:
    ...
```

---

# 25. 修改 `web/command_console.py`

这是已有文件，只允许做“接线”。

---

## 25.1 新增 API 路由

建议：

```text
POST /api/quick-start/probe
POST /api/quick-start/register
GET  /api/quick-start/noises?dataset=<alias>
POST /api/quick-start/methods
POST /api/quick-start/plan
```

---

## 25.2 不新增 run endpoint

禁止：

```text
/api/quick-start/run
```

最后仍然使用现有：

```text
/api/run
```

---

## 25.3 `command_console.py` 不能复制 service 逻辑

路由 handler 只做：

```python
body = parse...
value = quick_start_api.xxx_payload(...)
send_json(value)
```

---

# 26. 阶段 6 API 测试

## 26.1 新建文件

```text
web/test_quick_start_api.py
```

---

## 26.2 测试

使用 mock service，测试：

- probe request shape。
- register request shape。
- noise response。
- methods response。
- plan response。
- malformed request 返回 4xx。
- service exception 进入现有错误处理。

不要在这个文件重复测试 method compatibility 算法。

---

# 27. 阶段 7：给 Quick Start 独立静态资源

## 27.1 新建目录

```text
web/assets/
```

如果已有则复用。

---

## 27.2 新建文件

```text
web/assets/quick_start.js
```

---

## 27.3 新建文件

```text
web/assets/quick_start.css
```

---

# 28. 修改 `web/command_console.py` 支持静态文件

当前 server 主要服务：

```text
/
 /recipe
```

新增最小白名单：

```text
/assets/quick_start.js
/assets/quick_start.css
```

直接复用现有 `_serve_file()`。

不要实现通用任意文件 server。

建议：

```python
STATIC_ASSETS = {
    "/assets/quick_start.js": WEB_ROOT / "assets" / "quick_start.js",
    "/assets/quick_start.css": WEB_ROOT / "assets" / "quick_start.css",
}
```

只允许这些已知路径。

---

# 29. `quick_start.js` 的职责

Quick Start 所有新代码集中在该文件。

不要继续堆进 `index.html`。

---

## 29.1 自己维护 Quick Start 局部 state

例如：

```javascript
const quickStartState = {
  path: "",
  probe: null,
  dataset: null,
  noise: null,
  methods: [],
  selectedPaperId: "",
  plan: null,
  loading: false,
  error: ""
};
```

不要再往全局 `state` 添加十几个：

```text
quickDataset
quickRegisterAlias
quickAdapter
quickRoot
quickNoiseMode
...
```

---

## 29.2 对外只暴露一个 controller

例如：

```javascript
window.quickStartController = {
  mount,
  render,
  onModuleEnter
};
```

不要向 window 暴露十几个 helper。

---

# 30. Quick Start UI 第一步：路径

页面首屏：

```text
1. 数据集

选择数据集目录或文件
[ F:\datasets\cifar10 ] [浏览]

[自动识别并继续]

也可以：
[使用已登记数据集]
```

注意主次：

- “路径”是新手主入口。
- “已登记数据集”是方便老用户的次入口。

---

# 31. 路径选择按钮

复用现有 `/api/picker`。

不要自己实现 OS 文件浏览器。

目录 adapter 使用 folder picker。

UCI 等单文件允许 file picker。

---

# 32. 自动识别结果

如果 detected：

```text
✓ 检测到 CIFAR-10

[继续]
```

继续时调用 register API。

也可以 probe 后直接自动 register + inspect。

推荐：

```text
自动识别并继续
```

一次完成：

```text
probe
→ unique
→ register
→ inspect
```

用户不需要理解三步。

---

# 33. ambiguous UI

如果：

```text
ambiguous
```

只显示候选：

```text
检测到多个可能的数据格式：

○ CIFAR-10
○ CIFAR-10N

[选择并继续]
```

不要显示 adapter 内部 Python class。

---

# 34. unsupported UI

显示：

```text
暂时无法自动识别这个数据路径。

[打开“本地数据集”手动登记]
```

切换到现有：

```text
data
```

module。

---

# 35. Quick Start UI 第二步：噪声

## 35.1 clean dataset

显示：

```text
2. 标签噪声

✓ 原始训练标签是干净的

● 保持干净
○ 添加人工噪声
```

选择“添加人工噪声”后：

```text
噪声类型
[对称噪声 ▼]

说明：
每个样本以给定概率随机翻转到其他类别。

噪声率
[20%]

随机种子
[1]
```

---

## 35.2 native dataset

显示：

```text
✓ 该数据集本身包含真实世界标签噪声

Quick Start 将直接使用数据集原始 noisy labels。
```

不要出现：

```text
external_torch
```

不要默认允许再叠加 symmetric。

---

## 35.3 unknown dataset

只问：

```text
系统无法判断训练标签是否含噪声。

○ 标签干净
○ 本身有噪声
○ 不知道
```

如果仍选择“不知道”，允许继续查看，但 method list 会根据现有 compatibility 变成 needs_input / unknown。

不要把 Dataset-first 的全部字段复制过来。

---

# 36. Quick Start UI 第三步：方法

不要用一个只有：

```text
pdl
external_torch
...
```

的裸下拉。

---

## 36.1 推荐使用卡片 / 可读列表

分三组：

```text
可直接运行
补充输入后可运行
当前实现不适用
```

---

## 36.2 每个方法卡片显示

```text
Co-teaching
Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels
NeurIPS 2018

两个网络分别选择 small-loss 样本，并交叉提供给对方更新。

状态：可直接运行
```

---

## 36.3 内部 id

可以放在折叠的：

```text
实现详情
```

里面。

不要作为 primary label。

---

## 36.4 所有已实现论文尽量显示

不要因为：

```text
没有 exact reproduction YAML
```

就不显示。

如果方法确实不适用：

```text
FINE
当前实现不适用
原因：当前实现要求 100 个类别
```

这样反而体现 toolbox 能力边界。

---

# 37. 方法选择后的配置来源

选中方法后调用：

```text
POST /api/quick-start/plan
```

---

## 37.1 exact reproduction

显示绿色标识：

```text
论文复现配置
```

并显示：

```text
recipe id
```

但 recipe id 是次要信息。

---

## 37.2 toolbox-adapted

显示：

```text
Toolbox 适配配置
```

并明确：

```text
该配置基于此方法已有训练模板，
针对当前数据集和噪声条件自动生成。
它不是论文原始 reproduction 配置。
```

---

# 38. 缺少简单 method input

例如：

```text
noise_rate_prior
```

直接在 Quick Start 显示：

```text
该方法还需要：

算法使用的噪声率先验
[0.20]
```

synthetic rate 已知时默认填相同值。

---

# 39. 缺少复杂资源

例如：

- checkpoint；
- manifest；
- 特殊 external labels；

Quick Start 暂不重做资源管理。

显示：

```text
该方法还需要额外实验资源。

[打开详细配置]
```

跳现有：

```text
YAML editor
```

或 Dataset-first 详细页。

---

# 40. Quick Start 运行按钮

只保留：

```text
[预演]
[开始训练]
```

如果 plan `status != ready`：

禁用。

---

# 41. 继续使用现有运行系统

`quick_start.js` 得到 backend 返回的 command 后：

继续调用现有：

```text
/api/run
```

并复用：

- Preview。
- job polling。
- stop job。
- log console。

不要复制这些代码。

---

# 42. `quick_start.css`

仅负责 Quick Start。

建议类名统一前缀：

```text
.qs-
```

例如：

```text
.qs-root
.qs-step
.qs-step-title
.qs-dataset-card
.qs-noise-card
.qs-method-group
.qs-method-card
.qs-method-status
.qs-plan-card
```

避免污染现有全局 class。

---

# 43. 阶段 8：瘦身 `web/index.html`

这是本任务的重要要求。

当前 `index.html` 已经很大。

Quick Start 新功能不能继续全部写进去。

---

# 44. `web/index.html` 只新增两行资源引用

在 `<head>`：

```html
<link rel="stylesheet" href="/assets/quick_start.css">
```

在主脚本执行前：

```html
<script src="/assets/quick_start.js"></script>
```

---

# 45. 从 `index.html` 删除旧 Quick Start CSS

删除当前内联的：

```text
.quick-start-step
...
```

相关 Quick Start 专用样式。

搬到：

```text
web/assets/quick_start.css
```

不要顺手移动其他模块 CSS。

---

# 46. 从 `index.html` 删除旧 Quick Start JS

删除当前旧实现的 Quick Start helper，包括但不限于：

```text
quickStartRecipeMeta
quickStartCompatibilityRecipes
quickStartRecipeEligible
quickStartNoiseOptions
quickStartFilteredRecipes
quickStartSelectedRecipe
quickStartProfile
quickStartRegistrationHtml
quickRegisterAndInspect
quickStartCommand
```

以及旧 Quick Start 大段 rendering。

---

# 47. 删除旧的 Quick Start 全局 state 字段

例如当前类似：

```text
quickDataset
quickRegisterOpen
quickRegisterAlias
quickAdapter
quickRoot
quickPath
quickLabels
quickNoiseMode
quickNoiseKey
quickMethod
quickRecipe
quickMethodNoisePrior
quickError
```

这些迁移到：

```text
quick_start.js
```

的局部 state。

---

# 48. `index.html` 只保留最小桥接

例如：

```javascript
if (state.module === "quickstart") {
  window.quickStartController.mount(parameterPanel, {
    runCommand,
    switchModule,
    pickPath,
    setPreview
  });
  return;
}
```

实际接口名称可根据现有代码调整。

但原则：

> `index.html` 不再知道 Quick Start 如何 probe、如何列 noise、如何列 method、如何生成 plan。

---

# 49. `index.html` 体积验收要求

完成后：

- Quick Start 专用业务代码在 `index.html` 中不超过约 **40 行**。
- `index.html` 总行数不能比当前约 2253 行更大。
- 理想情况应该明显减少至少约 100 行。

不要为了满足行数而把代码压成一行。

要求是**职责拆分**，不是 minify。

---

# 50. 不要在本任务重构其他模块

不要趁机把：

- YAML builder；
- Sweep；
- beginner tutorial；
- data module；

一起拆 JS 文件。

那是另一个任务。

本次只把新增/已有 Quick Start 抽出去。

---

# 51. 阶段 9：Web 资源测试

## 51.1 新建文件

```text
web/test_quick_start_assets.py
```

---

## 51.2 测试

- `/assets/quick_start.js` 返回成功。
- `/assets/quick_start.css` 返回成功。
- 不允许任意 `../` 路径。
- `index.html` 包含两个 asset 引用。
- `index.html` 不再包含旧 `quickStartNoiseOptions`。
- `index.html` 不再包含旧 `quickStartCompatibilityRecipes`。
- Quick Start 仍然是左侧第一个 module。
- beginner / yaml / data / sweep / papers / freeform 等旧模块仍存在。

---

# 52. 修改现有 `web/test_command_console.py`

当前已有一个 Quick Start 测试是基于：

```text
index 中存在 quickRegisterAndInspect
index 中存在 quickStartNoiseOptions
index 中存在 quickStartCommand
```

这种旧架构 marker。

必须删除/改写这部分。

新的测试不要要求 Quick Start 逻辑继续内联。

改为检查：

```text
quickstart 是第一入口
quick_start.js 已加载
quick_start.css 已加载
/api/run 仍存在
旧模块仍存在
```

---

# 53. 阶段 10：README

## 53.1 修改文件

```text
web/README.md
```

---

## 53.2 新增一节

标题建议：

```text
Quick Start
```

说明：

```text
路径
→ 自动识别
→ 自动登记与 inspect
→ 选择标签噪声
→ 选择论文方法
→ 论文复现配置 / Toolbox 适配配置
→ 预演 / 训练
```

---

## 53.3 必须解释两个配置概念

### 论文复现配置

仓库正式登记、条件完全匹配的 reproduction recipe。

### Toolbox 适配配置

基于已有方法实现和模板，为当前 dataset/noise 自动派生。

**不等同于论文 reproduction。**

---

# 54. 建议不修改的文件

除非测试证明存在阻塞 bug，否则本任务不要改：

```text
src/lnl_toolbox/training/service.py
src/lnl_toolbox/training/runners.py
src/lnl_toolbox/training/data_service.py
src/lnl_toolbox/data/contracts.py
src/lnl_toolbox/data/profile.py
src/lnl_toolbox/paper_catalog.json
configs/
```

原因：

现有 compatibility / runner requirements / paper metadata 已经是后端事实来源。

Quick Start 的问题主要是：

```text
选择逻辑和编排层错误
```

不是重新发明 training contract。

---

# 55. 如果发现方法 metadata 不完整怎么办

不要在 Quick Start 里猜。

例如某个 runner 没有 requirements metadata：

返回：

```text
metadata_error
```

UI：

```text
该方法的兼容性元数据尚未完整登记
```

记录为独立维护问题。

不要为了让列表变绿，在 Quick Start 写：

```javascript
if (method === "...") compatible
```

---

# 56. 如果某篇论文没有任何基础 YAML 怎么办

不要生成凭空配置。

返回：

```text
metadata_error
```

说明：

```text
该方法已有实现，但缺少可用于 Quick Start 的基础配置模板。
```

这能真实暴露 toolbox 缺口。

不要把它直接隐藏。

---

# 57. 如果 dataset probe 识别不了怎么办

不要继续给 Quick Start 增加所有 adapter 的手工表单。

直接：

```text
无法自动识别。

[打开“本地数据集”]
```

现有详细页面负责复杂登记。

Quick Start 保持简单。

---

# 58. 不得出现的反模式

执行过程中，如果准备写出以下逻辑，请停止。

---

## 58.1 禁止

```javascript
const methods = compatibleRecipes.map(...)
```

方法不能再从 recipe list 反推。

---

## 58.2 禁止

```javascript
const noiseOptions = recipes.map(recipe => recipe.meta.noise)
```

noise 不能再从 recipe string 反推。

---

## 58.3 禁止

```javascript
if (dataset === "cifar10") ...
```

Web 前端不硬编码 dataset 兼容性。

---

## 58.4 禁止

```javascript
if (method === "coteaching") ...
```

Web 前端不硬编码 method compatibility。

---

## 58.5 禁止

```text
external_torch
```

作为普通新手 noise type。

---

## 58.6 禁止

因为没有 exact recipe 就直接把方法从列表删除。

---

## 58.7 禁止

把 toolbox-adapted config 显示为：

```text
论文复现配置
```

---

## 58.8 禁止

继续给 `index.html` 新增数百行 Quick Start 代码。

---

# 59. 推荐代码依赖方向

必须保持：

```text
web/index.html
        ↓
web/assets/quick_start.js
        ↓ HTTP
web/quick_start_api.py
        ↓
src/lnl_toolbox/quickstart/service.py
        ↓
├── data/probe.py
├── noise/quickstart_catalog.py
├── quickstart/templates.py
├── existing DataService
├── existing ExperimentService
└── existing paper catalog
```

不要反向依赖。

`src/lnl_toolbox` 不允许 import `web/*`。

---

# 60. 最终测试命令

先运行新增 targeted tests：

```bash
python -m unittest tests.test_data_probe -v
python -m unittest tests.test_quickstart_noise_catalog -v
python -m unittest tests.test_quickstart_service -v
python -m unittest web.test_quick_start_api -v
python -m unittest web.test_quick_start_assets -v
```

然后运行现有 Web tests：

```bash
python -m unittest discover -s web -p "test_*.py" -v
```

再运行与 data / compatibility 直接相关测试：

```bash
python -m unittest tests.test_data_service -v
python -m unittest tests.test_data_adapters -v
python -m unittest tests.test_compatibility -v
```

如果仓库当前统一使用 pytest，也执行对应 targeted pytest。

不要为了让无关历史测试通过而大改其他模块。

---

# 61. 手工验收 Case A：全新 CIFAR-10 用户

前置条件：

- catalog 中没有这个路径。
- 用户没有先进入“本地数据集”。

操作：

```text
打开 WebUI
→ Quick Start
→ 选择 CIFAR-10 文件夹
→ 自动识别并继续
```

预期：

- 不要求输入 alias。
- 不要求先懂 adapter。
- 自动识别 CIFAR-10。
- 自动 register。
- 自动 inspect。
- 显示 train/test/classes。
- 显示原始标签干净。

---

# 62. 手工验收 Case B：CIFAR-10 人工噪声

操作：

```text
CIFAR-10
→ 添加人工噪声
→ 对称噪声
→ 20%
```

预期：

- 这个选项存在，即使当前仓库没有某个“CIFAR10 + Sym20 + 某方法”的 exact reproduction YAML。
- `external_torch` 不出现在 noise type。
- `pdl` 不以裸字符串形式出现。
- 用户看到人类可读说明。

---

# 63. 手工验收 Case C：CIFAR-10 方法展示

选：

```text
CIFAR-10
Symmetric 20%
```

预期：

方法列表不是只剩下几个有 exact YAML 的论文。

必须：

- 遍历已实现论文；
- 按真实 compatibility 分类；
- 结构上能支持 CIFAR-10 的方法显示 ready / needs_input；
- FINE 由于当前实现 100-class restriction 显示 unsupported；
- unsupported 仍然可展开看原因。

---

# 64. 手工验收 Case D：方法名称

页面不能出现以下作为主要选项：

```text
instance_transition
multi_model
external_torch
```

应该显示：

```text
PDL
JoCoR
Co-teaching
DivideMix
...
```

并有标题/摘要。

---

# 65. 手工验收 Case E：天然噪声

选择 Clothing1M / Animal10N 路径。

预期：

- 自动识别。
- 自动注册 + inspect。
- 显示“数据集本身包含真实世界标签噪声”。
- Quick Start 不再问用户是否添加普通 synthetic noise。
- 方法列表基于数据能力和 method requirements，而不是已有 Clothing1M YAML 数量。

---

# 66. 手工验收 Case F：没有 exact reproduction

选择一个：

```text
dataset + noise + method
```

组合，方法兼容但仓库无完全匹配 reproduction。

预期：

生成：

```text
Toolbox 适配配置
```

而不是方法消失。

页面必须明确说明：

```text
不是论文原始 reproduction
```

最终 `preflight` 通过后才能 Run。

---

# 67. 手工验收 Case G：exact reproduction

如果仓库正好有当前条件下正式 reproduction：

预期：

```text
论文复现配置
```

并直接使用正式 recipe。

不要没必要再生成 adapted YAML。

---

# 68. 手工验收 Case H：旧功能不受影响

依次打开：

```text
新手教程
新建 YAML
本地数据集
参数 Sweep
运行管理
论文方法
自由输入
```

预期：

全部仍然存在。

现有 Dataset-first compatibility 详细流程仍存在。

---

# 69. 最终验收标准

只有下面全部满足，任务才算完成。

## 架构

- [ ] 新建 `src/lnl_toolbox/data/probe.py`，只负责路径识别。
- [ ] 新建 `src/lnl_toolbox/noise/quickstart_catalog.py`，只负责新手噪声能力目录。
- [ ] 新建 `src/lnl_toolbox/quickstart/__init__.py`。
- [ ] 新建 `src/lnl_toolbox/quickstart/models.py`。
- [ ] 新建 `src/lnl_toolbox/quickstart/templates.py`。
- [ ] 新建 `src/lnl_toolbox/quickstart/service.py`。
- [ ] 新建 `web/quick_start_api.py`。
- [ ] 新建 `web/assets/quick_start.js`。
- [ ] 新建 `web/assets/quick_start.css`。
- [ ] Quick Start 业务逻辑不继续堆进 `web/index.html`。
- [ ] `index.html` 中 Quick Start 专用业务桥接不超过约 40 行。
- [ ] `index.html` 总行数不高于修改前约 2253 行，并应明显减少。
- [ ] 没有为了行数把 JS 压成不可读的一行。

## 数据集

- [ ] 新手只给数据路径即可开始。
- [ ] CIFAR-10 可自动识别。
- [ ] CIFAR-100 可自动识别。
- [ ] CIFAR-N 能识别官方 noise-label 文件。
- [ ] Clothing1M 能基于真实文件签名识别。
- [ ] Animal10N 能基于真实结构识别。
- [ ] 已登记路径自动复用，不重复注册。
- [ ] 唯一高可信候选自动生成 alias。
- [ ] 自动完成 register + inspect。
- [ ] ambiguous 时让用户选择，不擅自猜。
- [ ] unsupported 时引导到现有“本地数据集”。

## 噪声

- [ ] CIFAR-10 等 clean dataset 明确显示“原始标签干净”。
- [ ] clean dataset 可以选择“保持干净”。
- [ ] clean dataset 可以直接选择工具箱支持的人工噪声。
- [ ] synthetic noise 选项来自 noise capability catalog，不来自已有 YAML。
- [ ] `external_torch` 不作为新手普通噪声类型出现。
- [ ] `pdl` 不以无法理解的裸内部字符串出现。
- [ ] noise rate 能正常传入实际 candidate config。
- [ ] synthetic noise rate 已知时，可用于需要 noise-rate prior 的方法默认值。
- [ ] native-noise dataset Quick Start 默认直接使用 observed noisy labels。
- [ ] native-noise dataset 不主动提供二次 synthetic pollution。

## 方法

- [ ] 方法列表来自已实现论文/方法 catalog，而不是 filtered reproduction recipes。
- [ ] 所有已实现论文尽可能显示。
- [ ] 方法使用 acronym/title/summary/venue/year 作为用户界面信息。
- [ ] runner/internal method id 不作为主要标签。
- [ ] 没有 exact YAML 的兼容方法仍然可以显示。
- [ ] 方法分成“可直接运行 / 补充输入后可运行 / 当前实现不适用”。
- [ ] 真正 incompatible 方法仍显示真实原因。
- [ ] FINE 在 CIFAR-10 上不会被错误标为 compatible。
- [ ] compatibility 仍然由现有 ExperimentService / MethodRequirements 决定。
- [ ] Web JS 不 hardcode dataset-method compatibility。

## 配置

- [ ] exact 条件匹配正式 reproduction 时使用 `paper_reproduction`。
- [ ] 没有 exact reproduction 但方法兼容时生成 `toolbox_adapted`。
- [ ] `toolbox_adapted` 明确标注“不等同于论文原始复现”。
- [ ] generated config 不写入正式 `configs/papers/`。
- [ ] generated config 存在独立 artifact 目录。
- [ ] generated config 有 provenance sidecar。
- [ ] generated config 必须通过 validation/preflight 后才允许 Run。
- [ ] reproduction command 使用 `--recipe`。
- [ ] adapted command 使用 `--config`。
- [ ] 不伪造 recipe id。

## Web

- [ ] Quick Start 仍是默认第一入口。
- [ ] 原有所有模块继续显示在 Quick Start 下面。
- [ ] Quick Start JS/CSS 使用独立静态资源。
- [ ] `command_console.py` 只增加薄路由，不堆业务逻辑。
- [ ] 不新增第二套 run endpoint。
- [ ] 继续复用现有 `/api/run`。
- [ ] 继续复用现有 Preview。
- [ ] 继续复用现有 job polling。
- [ ] 继续复用现有日志。
- [ ] 继续复用现有 stop job。
- [ ] Dataset-first 详细页面没有被删除或隐藏。
- [ ] YAML editor 没有被删除或重写。

## 回归

- [ ] `tests.test_data_probe` 全通过。
- [ ] `tests.test_quickstart_noise_catalog` 全通过。
- [ ] `tests.test_quickstart_service` 全通过。
- [ ] `web.test_quick_start_api` 全通过。
- [ ] `web.test_quick_start_assets` 全通过。
- [ ] 现有 Web tests 全通过。
- [ ] data service / adapters / compatibility 相关测试全通过。
- [ ] 本任务没有修改 global index bug。
- [ ] 本任务没有大规模修改训练核心。
- [ ] 本任务没有为了 Quick Start 批量改论文 reproduction YAML。

---

# 70. Codex 最后提交前必须自检

提交前请自己回答下面 8 个问题；任何一个回答为“否”，都不要结束任务：

1. 一个从未登记 CIFAR-10 的用户，是否只给一个目录就能继续？
2. CIFAR-10 是否可以直接添加工具箱已有的人工噪声，而不是只能选择已有 YAML 中出现过的 noise？
3. 页面是否彻底避免把 `external_torch` 当成普通新手噪声类型？
4. 方法列表是否不再由 reproduction recipe 数量决定？
5. 没有 exact YAML 但实现本身兼容的方法，是否仍然能看到并运行 Toolbox 适配配置？
6. Toolbox 适配配置是否明确没有冒充论文 reproduction？
7. Quick Start 的核心 JS 是否已经从 `index.html` 拆出去？
8. 原有详细功能是否全部还在 Quick Start 下面？

全部回答“是”后，再提交。
