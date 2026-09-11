# EXECUTION PLAN — LNL Scratch V2：真正的公式积木编辑器

目标仓库：`seanlee-lile/LNL-toolbox`  
目标分支：`codex/cli`  
主要目录：`src/lnl_toolbox/scratch/`

---

## 0. 任务定义

当前 Scratch 的 `Recipe → Block → Context → Executor` 思路可以保留，但当前 Web 交互必须重做。

本任务的目标不是“美化卡片”，而是实现真正像 Scratch 的编辑过程：

1. 页面默认不是空白，而是最简单的监督训练骨架。
2. 用户永远知道下一步可以做什么。
3. 不能使用的积木直接灰掉。
4. 可用积木可以点击，也可以真正拖到合法位置。
5. Epoch Loop / Batch Loop 必须是清楚可见的容器。
6. 已经放进去的积木可以从循环外拖进循环内，也可以反向移动。
7. 数据首先处理：路径、数据集特征、标签情况、是否加噪、有哪些噪声方式。
8. 训练只暴露两个循环：Epoch Loop、Batch Loop。
9. Batch 内不是“某篇论文一个大黑盒”，而是论文中的数学步骤。
10. 论文中的核心公式必须各自成为可拖积木。
11. 点击论文模板后，直接看到该论文如何由这些公式拼起来。
12. 用户不需要手改 YAML，也不需要手写 `model_a / logits_b / selected_a` 这类内部 slot。

本轮先把 **GCE** 和 **Co-teaching** 做到最终标准。没有通过人工搭建验收以前，不继续批量改剩余论文。

---

# 1. 先确认当前问题，禁止误判

先读以下文件：

```text
src/lnl_toolbox/scratch/registry.py
src/lnl_toolbox/scratch/validation.py
src/lnl_toolbox/scratch/web/index.html
src/lnl_toolbox/scratch/web/scratch.js
src/lnl_toolbox/scratch/web/scratch.css
src/lnl_toolbox/scratch/web/server.py
src/lnl_toolbox/scratch/recipes/papers/gce.yaml
src/lnl_toolbox/scratch/recipes/papers/coteaching.yaml
```

确认当前问题：

- `scratch.js` 初始 recipe 是 `steps: []`。
- palette 所有 block 默认 `draggable=true`。
- palette 所有 block 默认 `onclick => addStep(item.id)`。
- `addStep()` 把 step 直接 push 到最外层 `state.recipe.steps`。
- 现有 step 拖动只使用 `move-index`，所以只能在同一个数组内排序。
- Loop 内虽然有 nested `drawSteps()`，但已有 step 不能跨父级移动。
- inspector 主要把参数统一变成普通 input。
- “打开”使用 `prompt()` 输入 recipe 文件名。
- 当前 GCE recipe 仍是一个 `gce_loss` 黑盒。
- 当前 Co-teaching 的核心样本交换仍被 `peer_exchange` 大块遮住。

如果实际代码已经发生变化，以当前分支代码为准，但必须逐条验证上述行为。

---

# 2. 架构硬约束

## 2.1 保留简单解释器

继续保持：

```text
Recipe = 有序步骤
Block = Python 操作/公式
Context = 字符串 key 的共享工作台
Executor = 从上到下递归执行
```

不要增加：

```text
Compiler
IR
DAG
Graph engine
typed-port system
workflow framework
```

## 2.2 旧算法代码只作为“算法数据库”

允许检索：

```text
src/lnl_toolbox/algorithms/
src/lnl_toolbox/training/
src/lnl_toolbox/losses/
src/lnl_toolbox/selectors/
src/lnl_toolbox/estimators/
src/lnl_toolbox/treatments/
configs/
paper_catalog.json
```

用途：

- 核对公式
- 核对训练顺序
- 核对默认参数
- 核对 tensor shape
- 核对数值稳定处理

禁止新 Scratch 论文公式 block 直接调用旧 runner/旧算法完成工作。

## 2.3 数据模块允许复用

用户明确允许数据第一步联动已有模块。

Scratch Web 可以复用现有 DataService / local catalog / inspect 逻辑获取数据事实。

不要复制另一套数据集注册系统。

---

# 3. 本轮文件范围

主要修改：

```text
src/lnl_toolbox/scratch/registry.py
src/lnl_toolbox/scratch/validation.py

src/lnl_toolbox/scratch/blocks/runtime.py
src/lnl_toolbox/scratch/blocks/data.py
src/lnl_toolbox/scratch/blocks/models.py
src/lnl_toolbox/scratch/blocks/control.py
src/lnl_toolbox/scratch/blocks/forward.py
src/lnl_toolbox/scratch/blocks/losses.py
src/lnl_toolbox/scratch/blocks/selection.py
src/lnl_toolbox/scratch/blocks/optimization.py
src/lnl_toolbox/scratch/blocks/evaluation.py
src/lnl_toolbox/scratch/blocks/paper_specific/coteaching.py

src/lnl_toolbox/scratch/recipes/examples/default_supervised.yaml
src/lnl_toolbox/scratch/recipes/papers/gce.yaml
src/lnl_toolbox/scratch/recipes/papers/coteaching.yaml

src/lnl_toolbox/scratch/web/index.html
src/lnl_toolbox/scratch/web/scratch.js
src/lnl_toolbox/scratch/web/scratch.css
src/lnl_toolbox/scratch/web/server.py

src/lnl_toolbox/scratch/README.md
src/lnl_toolbox/scratch/REFERENCE_MAP.md
```

建议新增：

```text
src/lnl_toolbox/scratch/web/data_bridge.py
src/lnl_toolbox/scratch/tests/test_block_metadata.py
src/lnl_toolbox/scratch/tests/test_formula_templates.py
```

如果 `data_bridge.py` 只有很少代码，可以直接并入 `server.py`。

---

# 4. STEP 1 — 升级 BlockDefinition metadata

修改 `scratch/registry.py`。

现有字段保留：

```python
id
name
category
description
kind
params
requires
provides
execute
```

新增，并全部给默认值：

```python
placement: tuple[str, ...] = ("any",)
stage: str = "train"
ui_group: str = ""
formula: str | None = None
formula_ref: str | None = None
paper: str | None = None
beginner_visible: bool = True
```

同步修改：

```python
BlockDefinition.describe()
block(...)
```

`/api/blocks` 必须返回这些字段。

### placement 仅允许

```text
top
epoch
batch
any
```

语义：

- `top`：所有训练循环外。
- `epoch`：Epoch Loop 内、Batch Loop 外。
- `batch`：Batch Loop 内。
- `any`：少量 utility；不要滥用。

### stage 仅用于 UI

```text
data
setup
train
evaluate
```

### ui_group 面向用户

至少：

```text
① 数据准备
② 初始化
③ 训练结构
④ 前向与概率
⑤ 损失公式
⑥ 样本选择
⑦ 标签与矩阵
⑧ 反向传播与更新
⑨ 评估
⑩ 论文专用
```

不要把源码目录名暴露给新手。

### 公式 metadata

任何对应论文数学公式的 block 必须有：

```python
formula="..."
formula_ref="..."
paper="..."
```

如果论文没有 equation number，不要编造，写：

```text
Algorithm 1 / method definition / small-loss selection step
```

### STEP 1 验收

- 旧 block 在不填新字段时仍可注册。
- `/api/blocks` 能看到新字段。
- 现有 executor 不受影响。

---

# 5. STEP 2 — 给现有 block 标 placement

逐个实际检查 block id 后填写。

推荐规则：

```text
set_seed                    top
select_device               top
load_*                      top
create_loader               top
create_model                top
create_optimizer            top
epoch_loop                  top
batch_loop                  epoch
get_batch                   batch
zero_grad                   batch
forward                     batch
softmax                     batch
loss / formula              batch
selection                   batch
backward                    batch
optimizer_step              batch
scheduler_step              按真实语义标 epoch/batch
最终 evaluate               top
batch metric                batch
```

如果有 `repeat_n`、generic condition 等不希望新手看到：

```python
beginner_visible=False
```

普通积木栏只暴露两个循环：

```text
Epoch Loop
Batch Loop
```

### STEP 2 验收

新增 `test_block_metadata.py`，检查：

- placement 合法；
- stage 合法；
- beginner-visible formula block 若是论文公式必须有 formula；
- block id 唯一。

---

# 6. STEP 3 — 默认网页必须加载基础模板

新增：

```text
src/lnl_toolbox/scratch/recipes/examples/default_supervised.yaml
```

骨架至少：

```yaml
schema_version: 1
name: new_algorithm
description: Scratch default supervised skeleton

steps:
  - block: select_dataset
    params:
      dataset: ""
      save_as: train_dataset

  - block: configure_noise
    params:
      method: none

  - block: create_loader
    params:
      dataset: train_dataset
      batch_size: 128
      shuffle: true
      save_as: train_loader

  - block: create_model
    params:
      model: resnet18
      save_as: model

  - block: create_optimizer
    params:
      optimizer: sgd
      model: model
      lr: 0.1
      save_as: optimizer

  - block: epoch_loop
    params:
      epochs: 200
    steps:
      - block: batch_loop
        params:
          loader: train_loader
        steps:
          - block: get_batch

  - block: evaluate_accuracy
```

如果实际 block 参数名字不同，按真实实现调整。

注意：

- Batch Loop 内只预置 `Get Batch`。
- 不要预置 CE/GCE。
- Get Batch 后显示明显空位：“把训练公式拖到这里”。

后端 `server.py` 新增：

```text
GET /api/default-recipe
```

返回这个文件。

`scratch.js` 初始化顺序改为：

```text
GET /api/blocks
GET /api/default-recipe
hydrateUiIds()
draw()
```

“新建”按钮也必须重新加载这个默认模板，不能再创建 `steps: []`。

### STEP 3 验收

每次：

```text
刷新页面
点击新建算法
```

中央都至少显示：

```text
数据
初始化
Epoch Loop
Batch Loop
评估
```

不再出现空白编辑器。

---

# 7. STEP 4 — 数据必须成为第一步

修改 `blocks/data.py`，增加/统一：

```text
select_dataset
configure_noise
```

## select_dataset

至少这些参数：

```text
source_mode
dataset
path
save_as
```

推荐 metadata：

```python
placement=("top",)
stage="data"
ui_group="① 数据准备"
```

参数类型：

```text
source_mode → enum
dataset     → dataset
path        → path
save_as     → slot/output name
```

`source_mode` 可有：

```text
registered
builtin
custom_path
```

默认优先 `registered`。

---

# 8. STEP 5 — Scratch Web 联动现有数据服务

在 `server.py` 或 `data_bridge.py` 增加：

```text
GET /api/datasets
GET /api/dataset/<alias>
```

实现前先搜索：

```text
web/command_console.py
src/lnl_toolbox/data/
```

复用主 Web 已有 DataService / local catalog / inspect 路径。

不要自己判断：

```text
目录里有没有 data_batch_1
文件名像不像 CIFAR
```

返回真实可获得信息，例如：

```json
{
  "alias": "cifar10-local",
  "path": "F:/datasets/cifar10",
  "adapter": "cifar10",
  "train_samples": 50000,
  "test_samples": 10000,
  "num_classes": 10,
  "input_shape": [3, 32, 32],
  "has_clean_target": true,
  "has_noisy_target": false,
  "has_sample_index": true,
  "layout_validated": true,
  "training_verified": true
}
```

不存在/未知字段返回 `null`，前端显示“未知”，禁止猜。

---

# 9. STEP 6 — 数据 UI 必须展示事实

点击 `select_dataset` 后，右侧除了控件，还显示：

```text
数据集信息

名称
路径
adapter
Train 样本数
Test 样本数
类别数
输入 shape
Clean target
Noisy target
Sample index
Inspect 状态
Training verify 状态
```

`type=dataset` 不能是文本框。

必须：

```text
GET /api/datasets
→ <select>
```

如果没有数据：

```text
暂无已登记数据
前往“本地数据集”管理
```

如果选择 `custom_path` 才显示路径输入。

---

# 10. STEP 7 — 噪声必须是数据步骤的一部分

`configure_noise`：

```text
placement=top
stage=data
```

参数：

```text
method
rate
seed
必要时 transition matrix
```

噪声方法只使用仓库实际已有的实现。

至少检查仓库是否有：

```text
none
symmetric
asymmetric
instance-dependent
natural/existing noisy label
```

如果实际名称不同用真实名称。

### 解锁规则

没有选择有效数据集：

```text
configure_noise 灰
```

提示：

```text
请先选择并成功识别数据集。
```

数据有 clean labels：

允许人工造噪：

```text
symmetric
asymmetric
instance-dependent
```

天然 noisy dataset：

允许：

```text
使用已有 noisy labels
```

没有 clean truth 时：

不能提供依赖 clean truth 的造噪方式。

右侧显示：

```text
输入：clean labels
操作：Symmetric noise rate=0.2
输出：noisy labels
测试集：不加噪
```

---

# 11. STEP 8 — Loop 只能有两个

普通用户只看到：

```text
Epoch Loop
Batch Loop
```

`epoch_loop`：

```text
placement=top
```

默认主训练只能有一个。已有时 palette 中灰掉。

`batch_loop`：

```text
placement=epoch
```

默认主 Batch Loop 只能有一个。已有时灰掉。

复杂多 loader 未来再扩展，本轮不要提前抽象。

---

# 12. STEP 9 — Loop UI 必须是明显容器

当前 Loop 不能再像普通卡片。

目标：

```text
┌─ Epoch Loop · 200 epochs ─────────────────┐
│                                           │
│  + 添加 Epoch 级步骤                      │
│                                           │
│  ┌─ Batch Loop · train_loader ─────────┐  │
│  │                                    │  │
│  │ [Get Batch]                        │  │
│  │                                    │  │
│  │ + 把 Batch 内公式拖到这里          │  │
│  │                                    │  │
│  └────────────────────────────────────┘  │
│                                           │
└───────────────────────────────────────────┘
```

每两个 block 中间都有 drop zone。

drop zone 文案：

```text
root  → + 添加循环外步骤
epoch → + 添加 Epoch 级步骤
batch → + 添加 Batch 公式/操作
```

---

# 13. STEP 10 — 废弃 `move-index` 拖动模型

这是当前最重要 bug。

现有 step 必须可以跨层移动。

每个前端 step 加临时：

```javascript
_uiId
```

Recipe 加载后递归：

```javascript
ensureUiIds(steps)
```

拖 palette 新 block：

```javascript
event.dataTransfer.setData(
  "application/x-lnl-new-block",
  block.id
)
```

拖已有 step：

```javascript
event.dataTransfer.setData(
  "application/x-lnl-step",
  step._uiId
)
```

禁止继续只依赖：

```text
move-index
```

---

# 14. STEP 11 — 实现这些 helper

`scratch.js` 中明确实现：

```javascript
findStepById(uiId)
findParentArrayAndIndex(uiId)
getChildrenArray(parentId)
removeStepById(uiId)
insertStep(parentId, index, step)
getPlacementContext(parentId)
ensureUiIds(steps)
stripUiFields(recipe)
```

约定：

```text
parentId="__root__"
```

表示根 `recipe.steps`。

Loop 的 parentId 使用 loop 自己 `_uiId`。

---

# 15. STEP 12 — `_uiId` 禁止保存

调用：

```text
validate
save
run
```

前：

```javascript
stripUiFields()
```

深拷贝并删除：

```text
_uiId
任何 _ui*
```

后端 YAML 不出现 UI 临时字段。

---

# 16. STEP 13 — 真正支持跨 parent 拖动

必须支持：

```text
root → epoch
root → batch
epoch → root
epoch → batch
batch → epoch
batch → root
同一层重新排序
```

但只有 placement 合法才允许 drop。

例：

```text
Forward → Batch
```

合法。

```text
Select Dataset → Batch
```

非法。

非法时：

- drop zone 不高亮为合法；
- cursor not-allowed；
- drop 无效果；
- 有原因说明。

---

# 17. STEP 14 — 拖动时高亮所有合法位置

dragstart 时：

```text
合法 drop zone → .drop-valid
非法 drop zone → .drop-invalid
```

dragend 清除。

不要让用户猜该拖哪里。

---

# 18. STEP 15 — 点击积木不再 append root

删除当前：

```javascript
node.onclick = () => addStep(item.id)
```

以及：

```javascript
state.recipe.steps.push(...)
```

这种无上下文行为。

新增：

```javascript
state.activeInsertionTarget = {
  parentId: "...",
  index: 3,
  context: "batch"
}
```

用户点击任何 drop zone：

该 zone 成为 active target。

视觉高亮并显示：

```text
当前插入位置：Batch Loop，第 4 步
```

palette block 只有在对 active target 合法时才可点击。

点击后：

```text
插入 active target
```

绝不能无条件进入 root。

---

# 19. STEP 16 — 默认 active target

打开默认模板后，自动选中：

```text
Batch Loop → Get Batch 后面的 drop zone
```

页面“下一步”显示：

```text
下一步：添加 Forward。
```

但如果数据/model 尚未满足，Forward 仍灰，并提示应先完成前置步骤。

---

# 20. STEP 17 — 可点/不可点状态

不能硬编码每篇论文：

```javascript
if (block.id === "gce") ...
```

主要使用：

```text
placement
requires
provides
```

外加少量结构规则：

```text
Epoch Loop 单例
Batch Loop 单例
```

实现：

```javascript
availableKeysBefore(target)
canInsert(block, target)
availabilityReason(block, target)
```

### availableKeysBefore

简单按顺序扫描：

1. 继承父级已有 provides；
2. 扫描目标位置之前的 block；
3. 每个 block 的 `provides` 加入 Set；
4. 进入 Batch 继承外层 model/optimizer/loader；
5. 不做复杂 condition 路径证明。

---

# 21. STEP 18 — Palette 灰度规则

可用：

```text
正常颜色
draggable=true
aria-disabled=false
cursor=grab
```

不可用：

```text
灰色
draggable=false
aria-disabled=true
cursor=not-allowed
```

tooltip/说明必须给原因。

例：

```text
Forward
当前不可用：请先创建模型，并选择 Batch Loop 内的位置。
```

```text
GCE q-loss
当前不可用：前面还没有 target_probability。
```

```text
Backward
当前不可用：前面还没有 scalar loss。
```

任何 disabled block 点击都不能插入。

---

# 22. STEP 19 — 每次编辑都重新计算

以下操作后必须重新 `renderPalette()` / availability：

```text
添加
删除
复制
移动
跨 Loop 移动
修改参数
修改 slot
选择数据集
修改噪声
打开模板
新建算法
```

---

# 23. STEP 20 — 参数编辑器改为真实控件

新增：

```javascript
renderParamControl(name, schema, step)
```

映射：

```text
int          → number
float        → number
bool         → checkbox/switch
enum         → select
path         → path/text 控件
dataset      → /api/datasets 下拉
slot         → 上游可用 Context key 下拉
model        → select
optimizer    → select
noise_method → select
string       → text
```

---

# 24. STEP 21 — slot 禁止新手手写

读取已有 slot 的参数：

```text
model
input
logits
loss
indices
probabilities
labels
optimizer
```

必须根据当前步骤前的 available keys 做 select。

例如：

```text
Forward
模型：[model_a ▼]
输入：[images ▼]
输出名称：[logits_a]
```

只有 `save_as` 这类“创建新 slot 名”的字段可以输入文字，但必须有合理默认值。

---

# 25. STEP 22 — Inspector 固定四部分

点击 block 后显示：

```text
1. 名称和说明
2. 公式
3. 输入 / 输出
4. 参数
```

公式 block 示例：

```text
GCE q-loss

公式
L_q(f(x), y) = (1 - p_y^q) / q

输入
target_probability

输出
loss_per_sample

参数
q = 0.7

论文
Generalized Cross Entropy ...
```

formula 不允许只塞在 description。

---

# 26. STEP 23 — GCE 必须拆到公式级

当前大块：

```text
gce_loss
```

可保留兼容，但：

```python
beginner_visible=False
```

新建至少：

```text
forward
softmax_probability
gather_target_probability
gce_q_formula
mean_loss
backward
optimizer_step
```

---

# 27. STEP 24 — `softmax_probability`

文件：

```text
blocks/forward.py
```

metadata：

```text
placement=batch
stage=train
ui_group=④ 前向与概率
requires=logits
provides=probabilities
```

执行逻辑：

```python
ctx[save_as] = torch.softmax(ctx[logits], dim=1)
```

参数：

```text
logits
save_as
```

---

# 28. STEP 25 — `gather_target_probability`

文件：

```text
blocks/losses.py
```

数学意义：

```text
p_y = f_y(x)
```

metadata：

```text
placement=batch
ui_group=⑤ 损失公式
requires=probabilities, labels
provides=target_probability
formula="p_y = f_y(x)"
paper=GCE paper
```

执行：

按 labels gather 每个样本真实类别概率。

---

# 29. STEP 26 — `gce_q_formula`

文件：

```text
blocks/losses.py
```

公式：

```text
L_q(f(x), y) = (1 - p_y^q) / q
```

参数：

```text
input slot
q
save_as
```

输出：

```text
loss_per_sample
```

不要在这里 mean。

数值稳定的：

```text
clamp
epsilon
```

留在 block 内，不拆成用户积木。

---

# 30. STEP 27 — 重写 GCE recipe

`recipes/papers/gce.yaml` 不得再用一个 `gce_loss`。

Batch 内至少：

```text
Get Batch
Zero Grad
Forward
Softmax Probability
Gather target-class probability p_y
GCE q Formula
Mean Loss
Backward
Optimizer Step
```

### GCE 模板人工可读目标

```text
[Forward]
[Softmax]
[p_y = f_y(x)]
[L_q = (1 - p_y^q) / q]
[Mean]
[Backward]
[Optimizer Step]
```

---

# 31. STEP 28 — 先人工手搭 GCE

不要直接加载 GCE 模板。

必须：

```text
新建算法
→ 选择数据
→ 查看数据特征
→ 选择噪声
→ 确认模型
→ 确认优化器
→ 进入 Batch 插入位
→ Forward
→ Softmax
→ p_y
→ GCE q formula
→ Mean
→ Backward
→ Optimizer Step
→ 检查
→ smoke run
```

禁止：

```text
手改 YAML
手写 slot
```

如果不能完成，停止，不准继续 Co-teaching。

---

# 32. STEP 29 — 论文模板入口重做

当前 `prompt()` 必须删除。

header 调整为：

```text
新建算法
论文模板
我的 Recipe
保存
检查
运行
```

新增：

```text
GET /api/templates
```

至少返回：

```json
[
  {
    "id": "gce",
    "name": "GCE",
    "path": "papers/gce.yaml",
    "status": "formula-ready"
  },
  {
    "id": "coteaching",
    "name": "Co-teaching",
    "path": "papers/coteaching.yaml",
    "status": "formula-ready"
  }
]
```

其他旧 recipe 可返回：

```text
status=legacy-scratch
```

---

# 33. STEP 30 — 论文模板 UI

点击“论文模板”显示真实列表/modal/侧栏：

```text
搜索
论文名
状态
简介
打开
```

禁止 prompt 文件名。

GCE：

```text
公式积木已展开
```

Co-teaching：

```text
公式积木已展开
```

其他：

```text
旧版积木模板，尚未逐公式展开
```

不要误导。

---

# 34. STEP 31 — Co-teaching 拆公式和算法步骤

开始前实际检索：

```text
旧 Co-teaching 算法实现
旧 training lifecycle
当前 scratch peer_exchange
论文
```

禁止凭记忆写。

循环外：

```text
Dataset
Noise
DataLoader
Model A
Model B
Optimizer A
Optimizer B
```

Epoch 内、Batch 外：

```text
Remember/Forget Rate Schedule
```

Batch 内：

```text
Get Batch
Zero Grad A
Zero Grad B
Forward A
Forward B
Per-sample CE A
Per-sample CE B
Small-loss Selection A
Small-loss Selection B
Use B selected indices to compute A loss
Use A selected indices to compute B loss
Backward A
Optimizer Step A
Backward B
Optimizer Step B
```

---

# 35. STEP 32 — 不允许 `peer_exchange` 继续遮住选择过程

如果当前 `peer_exchange` 同时做：

```text
排序
small-loss
selected_a
selected_b
```

它必须退出 beginner 模式：

```python
beginner_visible=False
```

拆为：

```text
remember_rate_formula
small_loss_indices
select_by_indices / cross_select_loss
```

---

# 36. STEP 33 — `remember_rate_formula`

文件：

```text
blocks/paper_specific/coteaching.py
```

必须从论文/旧实现核对真实 schedule。

metadata：

```text
paper
formula
formula_ref
placement=epoch（若真实逻辑如此）
```

输出：

```text
remember_rate
```

如果真实实现是 forget rate，再清晰区分：

```text
forget_rate
remember_rate = 1 - forget_rate
```

不要混名。

---

# 37. STEP 34 — `small_loss_indices`

优先放公共：

```text
blocks/selection.py
```

输入：

```text
loss_per_sample
remember_rate
```

输出：

```text
selected_indices
```

公式/说明：

```text
选择逐样本损失最小的前 R(T) 比例。
```

---

# 38. STEP 35 — Co-teaching 交叉选择必须直接看得懂

可以继续用 `select_by_indices`，但 UI 卡片显示：

```text
[A ← B]
使用 B 选择的样本计算 A 的更新损失
```

另一路：

```text
[B ← A]
使用 A 选择的样本计算 B 的更新损失
```

不需要 DAG。

仍然纵向。

---

# 39. STEP 36 — 重写 Co-teaching recipe

目标视觉：

```text
① 数据准备
[Dataset]
[Noise]

② 初始化
[Model A]
[Model B]
[Optimizer A]
[Optimizer B]

③ 训练
┌ Epoch Loop ───────────────────────────┐
│ [Remember-rate Formula]              │
│                                      │
│ ┌ Batch Loop ──────────────────────┐ │
│ │ [Get Batch]                     │ │
│ │ [Zero Grad A]                   │ │
│ │ [Zero Grad B]                   │ │
│ │ [A Forward]                     │ │
│ │ [B Forward]                     │ │
│ │ [A Per-sample CE]               │ │
│ │ [B Per-sample CE]               │ │
│ │ [A Small-loss Set]              │ │
│ │ [B Small-loss Set]              │ │
│ │ [A ← B Cross-selected Loss]     │ │
│ │ [B ← A Cross-selected Loss]     │ │
│ │ [Backward A]                    │ │
│ │ [Step A]                        │ │
│ │ [Backward B]                    │ │
│ │ [Step B]                        │ │
│ └─────────────────────────────────┘ │
└──────────────────────────────────────┘
```

---

# 40. STEP 37 — 卡片显示公式/摘要

新增：

```javascript
renderStepSummary(step, info)
```

普通卡片至少显示：

```text
名称
关键公式（如有）
关键参数
输出 slot
```

例如：

```text
GCE q-loss
Lq = (1 - p_y^q)/q
q=0.7
→ loss_per_sample
```

Co-teaching：

```text
[A ← B] Cross-selected Loss
loss_a_per_sample + selected_b
→ loss_a
```

---

# 41. STEP 38 — 页面顶部显示“下一步建议”

增加：

```text
下一步：选择数据集
下一步：决定是否添加标签噪声
下一步：进入 Batch Loop 添加 Forward
下一步：添加概率/损失公式
```

实现简单函数：

```javascript
getNextSuggestion()
```

只根据 recipe 是否具备：

```text
dataset
noise
model
optimizer
epoch_loop
batch_loop
forward
scalar loss
backward
optimizer step
```

不要调用 AI，不要控制执行。

---

# 42. STEP 39 — 检查/运行按钮也要有状态

Recipe 修改后：

```text
state.validated=false
```

运行按钮立即灰。

只有：

```text
后端 validate 成功
数据有效
```

才：

```text
Run enabled
```

如果再次移动/改参数：

```text
Run disabled
```

直到重新检查。

---

# 43. STEP 40 — 后端 validation 增加 placement

修改：

```text
scratch/validation.py
```

递归检查上下文：

```python
validate_steps(steps, context="top")
```

遇到 `epoch_loop`：

```text
children context=epoch
```

遇到 `batch_loop`：

```text
children context=batch
```

拒绝：

```text
batch_loop at top
forward at top
dataset inside batch
model creation inside batch
batch-only formula outside batch
```

前端灰度不是安全边界，后端必须重复校验。

---

# 44. STEP 41 — requires/provides 后端继续检查

按顺序维护 available keys。

不要实现复杂 graph。

错误必须说明：

```text
第几个 step
哪个 block
缺哪个 key
应该先添加什么类型的积木
```

---

# 45. STEP 42 — 错误定位到具体卡片

后端已有 path/block_id 时继续使用。

前端：

```text
错误 step 红边
自动展开父 Loop
scrollIntoView
Inspector 显示错误
```

例如：

```text
GCE q-loss 无法运行
缺少 target_probability

建议：
先添加“读取真实类别概率 p_y”。
```

---

# 46. STEP 43 — CSS 必须实现这些状态

至少新增：

```text
.palette-block.disabled
.drop-target
.drop-target.active
.drop-target.drop-valid
.drop-target.drop-invalid

.loop-block
.loop-body
.epoch-loop
.batch-loop

.formula-block
.formula-text
.step-summary

.error-step
```

要求：

- disabled 明显灰；
- loop 明显包围 children；
- formula 和控制块能区分；
- active insertion target 一眼可见；
- drag 时合法/非法位置明显。

不要追求动画。

---

# 47. STEP 44 — README 改状态声明

当前不能继续把 26 篇 synthetic recipe 描述成已经完成最终 Scratch。

README 必须写：

```text
Formula-ready:
- GCE
- Co-teaching

Legacy Scratch recipes:
- 其他现有 paper recipes
- 已能按语义执行/验证，但尚未逐论文公式展开
```

---

# 48. STEP 45 — REFERENCE_MAP 更新

GCE：

```text
旧实现路径
论文来源
公式 1 → block
公式 2 → block
...
```

Co-teaching 同样。

若论文和旧实现冲突：

```text
记录差异
回查论文
论文为数学标准
不要静默猜
```

---

# 49. STEP 46 — 自动测试只加关键的

新增：

```text
test_block_metadata.py
test_formula_templates.py
```

## test_block_metadata.py

检查：

```text
block id unique
placement valid
stage valid
formula metadata complete
beginner-visible block metadata 可用于 UI
```

## test_formula_templates.py

GCE 顺序至少：

```text
epoch_loop
batch_loop
forward
softmax_probability
gather_target_probability
gce_q_formula
mean_loss
backward
optimizer_step
```

Co-teaching 至少：

```text
2 models
2 optimizers
epoch loop
remember rate
batch loop
2 forwards
2 per-sample CE
2 small-loss selections
2 cross selections
2 backwards
2 optimizer steps
```

不要建立 26 × datasets × noise 的测试爆炸。

---

# 50. STEP 47 — 必须真人/浏览器流程测试

不能只跑 pytest。

必须实际启动：

```bash
python -m lnl_toolbox.scratch.web.server
```

并尝试页面操作。

---

# 51. 人工验收 A — 从默认模板搭 GCE

必须从：

```text
新建算法
```

开始，不加载论文模板。

按：

```text
1. 页面自动有骨架。
2. 选择真实可用数据集。
3. 查看路径/shape/classes/train/test/label/index。
4. 选择不加噪或一个真实噪声。
5. 确认模型。
6. 确认优化器。
7. 选择 Batch Loop 内 Get Batch 后的 drop zone。
8. 添加 Forward。
9. 添加 Softmax。
10. 添加 p_y。
11. 添加 GCE q formula。
12. q=0.7。
13. 添加 Mean。
14. 添加 Backward。
15. 添加 Optimizer Step。
16. 点击检查。
17. 检查通过。
18. Run 解锁。
19. smoke run 成功。
```

全过程：

```text
不得手改 YAML
不得手写内部 slot
```

失败则整个任务未完成。

---

# 52. 人工验收 B — 按钮灰度

必须现场验证：

### root active 时

```text
Forward 灰
```

原因：

```text
Forward 只能放在 Batch Loop 内。
```

### Batch active 但没有 model 时

```text
Forward 灰
```

原因：

```text
请先创建模型。
```

### Forward 后、Softmax 前

```text
gce_q_formula 灰
```

原因：

```text
缺少 target_probability。
```

### p_y 生成后

```text
gce_q_formula 解锁
```

### loss_per_sample 生成后

```text
Mean Loss 解锁
```

### scalar loss 生成后

```text
Backward 解锁
```

---

# 53. 人工验收 C — 真正跨层拖动

必须证明已有 step 可以跨 parent。

至少测试：

```text
Batch 中某 step → 同一 Batch 另一位置
旧 recipe 中 root 的 batch-only step → Batch 内
Epoch 中合法 step → 另一个 Epoch 位置
```

非法：

```text
Dataset → Batch
```

必须拒绝。

---

# 54. 人工验收 D — GCE 论文模板

点击：

```text
论文模板 → GCE
```

不得 prompt。

中央直接出现：

```text
Forward
Softmax
p_y
GCE q formula
Mean
Backward
Step
```

点击每个公式：

右侧有：

```text
公式
论文
公式说明
输入
输出
参数
```

---

# 55. 人工验收 E — Co-teaching 模板

点击：

```text
论文模板 → Co-teaching
```

一眼能看出：

```text
Model A / B
Optimizer A / B
Epoch 内 remember-rate
Batch 内 A/B forward
A/B per-sample CE
A/B small-loss
A←B
B←A
A/B backward/update
```

不能只显示一个 `Peer Exchange` 黑盒。

---

# 56. 人工验收 F — 循环内外

Co-teaching：

```text
模型创建：循环外
优化器创建：循环外
remember-rate（若论文如此）：Epoch 内、Batch 外
样本级公式：Batch 内
Backward：Batch 内
Optimizer Step：Batch 内
```

必须和论文/旧实现真实生命周期一致。

---

# 57. 后续论文的固定迁移流程

GCE + Co-teaching 通过后，剩余论文逐篇：

```text
1. 打开论文。
2. 列算法步骤。
3. 列论文公式。
4. 搜旧实现。
5. 给每个公式找到旧实现位置。
6. 判断：
   - 复用公共公式 block
   - 新增公共公式 block
   - 新增论文专用公式 block
7. 给 block 填 formula/paper/formula_ref。
8. 写 recipe。
9. UI 打开模板。
10. 人工检查循环内外。
11. 人工拖动至少一个公式重新组装。
12. smoke。
```

不要把整篇算法包装成一个 block。

---

# 58. 公式积木的粒度标准

应该拆：

```text
论文单独定义的 objective
概率公式
风险公式
transition matrix 公式
weighting 公式
selection 公式
label correction 公式
consistency 公式
论文中明确的一步矩阵变换
```

不拆：

```text
torch.log
tensor.clamp
reshape
索引语句
to(device)
epsilon
数值稳定实现细节
```

判断：

> 论文作者是否会把这件事写成一个独立数学步骤？

如果会，优先做积木。

---

# 59. 最终页面参考

第一次打开：

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ LNL Scratch  新建算法  论文模板  我的Recipe  保存  检查  运行(灰)      │
├───────────────┬────────────────────────────────────┬────────────────────┤
│ 下一步        │ 新算法                             │ 当前积木           │
│ 选择数据集    │                                    │                    │
│               │ ① 数据准备                         │ 数据集             │
│ ① 数据准备    │ [选择数据集：尚未选择]             │ [cifar10 ▼]        │
│ 数据集        │ [噪声：不添加]                     │                    │
│ 噪声(灰)      │                                    │ 数据集信息         │
│               │ ② 初始化                           │ Path               │
│ ② 初始化      │ [Model: ResNet18]                 │ Train/Test         │
│ 模型          │ [Optimizer: SGD]                  │ Classes            │
│ 优化器        │                                    │ Shape              │
│               │ ③ 训练                             │ clean/noisy/index  │
│ ③ 训练结构    │ ┌ Epoch Loop ×200 ──────────────┐ │                    │
│ Epoch灰       │ │ ┌ Batch Loop ───────────────┐ │ │                    │
│ Batch灰       │ │ │ [Get Batch]               │ │ │                    │
│               │ │ │ + 把公式拖到这里          │ │ │                    │
│ ④ 前向与概率  │ │ └───────────────────────────┘ │ │                    │
│ Forward       │ └────────────────────────────────┘ │                    │
│ Softmax灰     │                                    │                    │
│               │ ④ 评估                             │                    │
│ ⑤ 损失公式    │ [Evaluate Accuracy]                │                    │
└───────────────┴────────────────────────────────────┴────────────────────┘
```

---

# 60. 明确禁止“假完成”

任何一个存在，都不能宣布完成：

```text
1. 默认页面仍然 steps=[]。
2. 点击 palette block 仍然 append root。
3. 已有 step 仍只能同层拖。
4. Loop 内外视觉不清楚。
5. 所有 palette block 永远可点。
6. 禁用 block 没有原因。
7. slot 仍要求手写。
8. 数据只显示 path 输入框。
9. 噪声不根据数据状态联动。
10. 普通用户还能看到一堆 generic loop。
11. 打开模板还用 prompt。
12. GCE 仍只有 gce_loss。
13. Co-teaching 仍只有 peer_exchange 遮住主要逻辑。
14. formula 没有独立 metadata。
15. 错误不能定位具体积木。
16. 只跑测试，没有人工搭 GCE。
17. README 仍宣称 26 篇都是 formula-ready。
```

---

# 61. Agent 必须严格按这个执行顺序

```text
01 读当前 scratch 关键文件
02 升级 BlockDefinition metadata
03 给核心 block 标 placement/stage
04 建 default_supervised.yaml
05 首页/新建自动加载默认模板
06 重写拖动 identity：_uiId
07 支持跨 parent remove/insert
08 做可见 drop zones
09 activeInsertionTarget
10 删除 onclick append-root
11 placement 灰度
12 requires/provides 灰度
13 参数控件类型化
14 slot 改自动 select
15 接已有数据 catalog/inspect
16 数据特征面板
17 噪声联动
18 GCE 公式 blocks
19 重写 GCE recipe
20 人工从默认模板搭 GCE
21 失败就继续修，不准往下
22 模板浏览器
23 Co-teaching 公式/选择拆分
24 重写 Co-teaching recipe
25 人工检查 Co-teaching
26 后端 placement validation
27 错误定位
28 CSS 状态
29 两个关键测试
30 README/REFERENCE_MAP
31 最终再人工完整验收
```

---

# 62. Agent 提交前必须回答这些问题

最终总结逐条回答：

```text
1. 默认页面加载什么？
2. Forward 在 root 时是否灰？为什么？
3. Forward 在 Batch 中何时解锁？
4. 已存在 block 能否跨 Loop 拖？
5. 点击 palette block 现在插在哪里？
6. 数据路径在哪里显示？
7. 数据 shape/classes/train/test 在哪里显示？
8. Clean/noisy label 状态在哪里显示？
9. 噪声选项从什么真实实现得到？
10. 普通用户能看到几个循环？答案应为 2。
11. GCE 被拆成哪些公式 blocks？
12. Co-teaching 被拆成哪些公式/算法 blocks？
13. 论文模板如何打开？是否还 prompt？
14. 哪两篇是 formula-ready？
15. 是否真正用鼠标从默认模板搭过 GCE？
16. 是否需要修改 YAML？答案必须“不需要”。
17. 是否需要手写 slot？答案必须“不需要”。
```

第 16/17 不是“不需要”，任务未完成。

---

# 63. 最终验收清单

## 默认体验
- [ ] 页面默认有骨架。
- [ ] 新建算法也加载骨架。
- [ ] 有下一步提示。
- [ ] Batch 内有明显拖放空位。

## 数据
- [ ] 可选已有数据集。
- [ ] 显示真实路径。
- [ ] 显示 train/test 数量。
- [ ] 显示 classes。
- [ ] 显示 shape。
- [ ] 显示 clean/noisy target。
- [ ] 显示 sample index。
- [ ] 噪声选项受数据能力约束。

## 拖拽
- [ ] palette 可拖。
- [ ] 已有 step 可跨 parent。
- [ ] root/epoch/batch drop zone 可见。
- [ ] 合法 drop 高亮。
- [ ] 非法 drop 被拒绝。
- [ ] 不再只用 move-index。

## 点击
- [ ] 不再 append root。
- [ ] active insertion target 明确。
- [ ] 不可用积木灰。
- [ ] 灰度有原因。

## 循环
- [ ] 普通用户只见 Epoch / Batch。
- [ ] Epoch 在 top。
- [ ] Batch 在 Epoch 内。
- [ ] batch 公式只能在 Batch。
- [ ] epoch 公式能在 Epoch 内 Batch 外。
- [ ] setup 在循环外。

## 参数
- [ ] enum 为 select。
- [ ] dataset 为 select。
- [ ] slot 输入为 select。
- [ ] 不手写内部 slot。

## 公式
- [ ] 公式 metadata 独立。
- [ ] 公式在 Inspector 可见。
- [ ] GCE 已拆公式。
- [ ] Co-teaching 已拆选择/交换逻辑。
- [ ] 公式 block 可拖。

## 模板
- [ ] 模板是列表，不是 prompt。
- [ ] GCE formula-ready。
- [ ] Co-teaching formula-ready。
- [ ] 其他未完成论文明确标 legacy/pending。
- [ ] 点击模板直接看到完整组装。

## 检查/运行
- [ ] 修改后 Run 灰。
- [ ] validate 通过后 Run 解锁。
- [ ] 错误高亮具体 block。
- [ ] 父 Loop 自动展开。

## 人工验证
- [ ] 完全靠鼠标从默认模板搭出 GCE。
- [ ] 不编辑 YAML。
- [ ] 不手写 slot。
- [ ] GCE validate 通过。
- [ ] GCE smoke 成功。
- [ ] Co-teaching 模板循环层级正确。
- [ ] Co-teaching 核心数学过程全部可见。

---

# 64. 唯一设计准则

如果 Agent 对某个实现拿不准，只问一个问题：

> 一个不了解仓库内部 Python 结构的新手，现在能不能像 Scratch 一样，通过“看哪些积木亮着 → 拖到明显的位置 → 看公式 → 继续下一步”，搭出一篇已有论文？

如果答案是否：

先修交互，不要继续增加功能。
