# SCRATCH_PAPER_MIGRATION_LOOP_PLAN.md

目标：用一套固定短流程，循环把 26 篇论文迁移成 **协议正确 + 公式可拖 + 可运行** 的 Scratch 模板。

> 核心策略：**一个计划 + 一个论文状态表 + 一个当前下标。**
> Agent 不需要记住 26 篇，只需要每次处理 `current_index` 指向的那一篇。

---

## 1. 为什么采用这个方案

这个 Agent 的问题不是不会写代码，而是会在目标不够具体时选择最省事的替代方案。

因此不要再给它 2000～4000 行的逐论文长计划。

只保留一套固定循环：

```text
读当前论文
→ 找正式配置
→ 核对 Scratch 缺口
→ 先补缺口
→ 再拆公式
→ 改正式 Recipe
→ 静态核对
→ 用运行时限额试跑
→ UI 检查
→ 更新状态
→ index + 1
```

上下文压缩后，只要重新读取：

```text
SCRATCH_PAPER_MIGRATION_LOOP_PLAN.md
SCRATCH_PAPER_MIGRATION_STATE.yaml
```

即可继续。

---

# 2. 必须新建状态文件

仓库根目录新建：

```text
SCRATCH_PAPER_MIGRATION_STATE.yaml
```

初始内容：

```yaml
current_index: 0

papers:
  - index: 0
    id: gce
    status: REPAIR_FIRST

  - index: 1
    id: coteaching
    status: REAUDIT

  - index: 2
    id: apl
    status: PENDING

  - index: 3
    id: binary_risk
    status: PENDING

  - index: 4
    id: loss_correction
    status: PENDING

  - index: 5
    id: jocor
    status: PENDING

  - index: 6
    id: importance_reweighting
    status: PENDING

  - index: 7
    id: cdr
    status: PENDING

  - index: 8
    id: dual_t
    status: PENDING

  - index: 9
    id: pdl
    status: PENDING

  - index: 10
    id: volminnet
    status: PENDING

  - index: 11
    id: t_revision
    status: PENDING

  - index: 12
    id: cwd
    status: PENDING

  - index: 13
    id: pcse
    status: PENDING

  - index: 14
    id: dss
    status: PENDING

  - index: 15
    id: cnlcu
    status: PENDING

  - index: 16
    id: mentornet
    status: PENDING

  - index: 17
    id: fine
    status: PENDING

  - index: 18
    id: lend
    status: PENDING

  - index: 19
    id: l2rw
    status: PENDING

  - index: 20
    id: cal
    status: PENDING

  - index: 21
    id: mc_ldce
    status: PENDING

  - index: 22
    id: upm
    status: PENDING

  - index: 23
    id: dividemix
    status: PENDING

  - index: 24
    id: dld
    status: PENDING

  - index: 25
    id: ca2c
    status: PENDING
```

如果 `paper_catalog.json` 中真实 id 与上面不同：

**先用 catalog 的真实 id 修正状态表，再开始。**

不要凭记忆改算法名。

---

# 3. 每篇论文允许的状态

只允许：

```text
PENDING
IN_PROGRESS
BLOCKED
READY
```

GCE 初始的 `REPAIR_FIRST` 和 Co-teaching 的 `REAUDIT` 只用于第一次启动。

开始处理时都改为：

```text
IN_PROGRESS
```

完成全部验收：

```text
READY
```

遇到当前无法解决的问题：

```text
BLOCKED
```

`BLOCKED` 时：

- 记录原因；
- 不允许 index + 1；
- 先解决阻塞。

---

# 4. 状态表每篇再维护这些字段

开始某篇时补齐：

```yaml
protocol_source:
paper_source:
legacy_sources: []
missing_capabilities: []
new_public_blocks: []
new_paper_blocks: []
protocol_check: PENDING
formula_check: PENDING
runtime_check: PENDING
ui_check: PENDING
notes: ""
```

不要建立另一堆长进度文档。

状态表就是进度真源。

---

# 5. 每次 Agent 启动时只做这四件事

```text
1. 读取本 PLAN。
2. 读取 SCRATCH_PAPER_MIGRATION_STATE.yaml。
3. 找 current_index 对应论文。
4. 只处理这一篇。
```

禁止提前实现下一篇。

---

# 6. 固定循环：每篇论文只执行 STEP A ～ STEP J

---

## STEP A — 找正式来源

读取：

```text
src/lnl_toolbox/paper_catalog.json
```

找到当前论文：

```text
paper source
reproduction / formal config
implementation paths
lifecycle / fidelity
```

然后读取：

```text
正式 config
旧实现
论文 Method / Algorithm / Equations
```

### 禁止

不要先读 Scratch smoke 然后照 smoke 重写。

### 完成后写状态表

```yaml
protocol_source: ...
paper_source: ...
legacy_sources:
  - ...
```

---

## STEP B — 提取正式协议

从正式 config 提取这些字段：

```text
dataset
split
preprocessing
augmentation
noise
noise manifest / target semantics
model
optimizer
scheduler
batch size
epochs
training stages
validation
best-model selection
test
external artifact（如果有）
```

只检查“正式实验实际上需要什么”。

不写长说明。

---

## STEP C — 检查 Scratch 缺什么

逐项对照当前 Scratch。

缺失能力写入：

```yaml
missing_capabilities:
  - resnet34
  - multistep_scheduler
  - ...
```

### 关键规则

只要正式协议需要，而 Scratch 没有：

**先补能力。**

禁止改正式协议来绕过缺失。

### 禁止替代

```text
正式 CIFAR → synthetic
正式 ResNet → MLP
正式 scheduler → 删除
正式 manifest → batch 内随机噪声
正式多阶段 → 单阶段
正式 120 epochs → Recipe 改 1 epoch
```

---

## STEP D — 先补基础能力

按 `missing_capabilities` 一项一项补。

能做公共 block 就做公共 block。

例如：

```text
模型
scheduler
noise manifest
batch contract
EMA
GMM
transition operation
history state
MixUp
```

每补一个能力：

```text
先单测
再从 missing_capabilities 删除
```

只有：

```yaml
missing_capabilities: []
```

才允许进入 STEP E。

---

## STEP E — 拆论文公式

现在才开始公式拆解。

规则：

```text
论文中独立公式/数学步骤
→ 一个可拖 block
```

不要拆：

```text
clamp
reshape
torch.log 内部
device transfer
```

### Block 必须满足

```text
输入 slot 可选
输出可 save_as
不绑定 paper 名
不绑定 recipe 名
不固定 ctx key
有 formula / formula_ref / paper metadata
```

### 已有公共公式

必须复用，不重复造。

---

## STEP F — 重写正式 Paper Recipe

修改：

```text
src/lnl_toolbox/scratch/recipes/papers/<paper>.yaml
```

Recipe 必须使用：

```text
正式 dataset
正式 preprocessing
正式 noise
正式 model
正式 optimizer
正式 scheduler
正式 epochs
正式 lifecycle
拆开的公式 blocks
```

### 非常重要

**Recipe 本身永远保持正式配置。**

不要为了测试速度修改 Recipe。

---

## STEP G — 静态协议核对

比较：

```text
正式 config
vs
Scratch paper recipe
```

至少检查：

```text
dataset
preprocessing
noise
model
optimizer
scheduler
epochs
batch size
关键 lifecycle
```

任何不一致：

```text
protocol_check: FAIL
```

回 STEP C/D/F 修。

一致后：

```yaml
protocol_check: PASS
```

---

## STEP H — 公式核对

逐公式检查：

```text
论文公式
旧实现
Scratch block
```

三者必须一致。

如果旧实现和论文冲突：

```text
论文数学为标准
记录 notes
```

全部通过：

```yaml
formula_check: PASS
```

---

## STEP I — 受限运行

运行的必须是：

```text
正式 Paper Recipe
```

但通过运行时限制缩短：

```text
max_epochs = 1
max_batches = 1
```

如方法需要状态更新：

可用：

```text
max_epochs = 2
max_batches = 1~2
```

只要足以验证 lifecycle。

### 禁止

不要创建另一个缩水 paper recipe。

不要把正式 Recipe 改成：

```text
synthetic
MLP
1 epoch
```

通过后：

```yaml
runtime_check: PASS
```

---

## STEP J — WebUI 检查并结束当前论文

打开：

```text
论文模板 → 当前论文
```

检查：

```text
正式数据/模型/训练配置可见
循环层级正确
核心公式逐块可见
至少删除一个公式 block
再从 palette 拖回
Validate 成功
```

通过后：

```yaml
ui_check: PASS
status: READY
```

然后：

```text
current_index += 1
```

保存状态表。

重新读取 PLAN + STATE，再处理下一篇。

---

# 7. 唯一完成条件

当前论文只有同时满足：

```yaml
missing_capabilities: []
protocol_check: PASS
formula_check: PASS
runtime_check: PASS
ui_check: PASS
status: READY
```

才能：

```text
current_index + 1
```

少一个都不允许跳。

---

# 8. GCE 作为第一次循环的特殊要求

`current_index = 0` 时，GCE 必须先修正当前已发现的问题。

正式 config 以当前仓库：

```text
configs/experiment/gce_cifar10_noise02_reproduction.yaml
```

为准。

已知至少检查：

```text
CIFAR-10
validation split
gce2018 preprocessing
symmetric noise 0.2
noise manifest
validation target semantics
ResNet-34
base_width
SGD 完整参数
MultiStepLR
milestones
gamma
120 epochs
batch size
validation
best model
final test
Batch contract
GCE q=0.7
```

### GCE 禁止

```text
synthetic
MLP
ResNet18 替代 ResNet34
ToTensor-only 替代 gce2018
batch 内随机噪声替代 manifest
删除 MultiStepLR
Recipe 改 1 epoch
```

GCE 成为第一个：

```text
protocol + formula + runtime + UI 全 PASS
```

的黄金模板后，再继续 index 1。

---

# 9. Co-teaching 必须重新审计

虽然 Co-teaching 之前做过公式拆解：

当前仍必须重新执行 A-J。

重点确认：

```text
正式数据协议
正式模型
forget/remember schedule
两个模型
两个 optimizer
scheduler
epochs
peer small-loss exchange
validation/test lifecycle
```

不能因为以前 `formula-ready` 就直接 READY。

---

# 10. 运行时限额是唯一允许的“缩短训练”方法

需要一个统一机制：

```text
max_epochs
max_batches
skip_final_test（仅结构验证时）
```

这些属于：

```text
Executor runtime
```

不是 Recipe。

UI 应能显示：

```text
正式配置：120 epochs
本次结构验证：只执行 1 epoch / 1 batch
```

---

# 11. `recipes/papers/` 和测试夹具严格分开

```text
scratch/recipes/papers/
```

只放正式论文模板。

```text
scratch/tests/fixtures/
```

才能放：

```text
synthetic
fake CIFAR
tiny model
1 epoch toy config
```

测试夹具不能出现在论文模板列表。

---

# 12. 公共零件规则

拆公式时优先形成公共零件。

典型公共积木：

```text
softmax
gather probability
per-sample CE
weighted sum
sample weighting
small-loss selection
KL / symmetric KL
transition multiply
matrix inverse / pseudoinverse
row normalize
logdet
EMA
history update
GMM
MixUp
sharpen
mask/reduction
centroid
distance
```

`paper_specific` 只放真正论文专用公式。

即使 paper-specific：

仍必须可以被其他 recipe 拖走使用。

---

# 13. 防止 giant block

如果一个 beginner-visible block：

```text
输入 logits/features/labels
内部连续执行 3 个以上论文公式
直接输出最终 loss
```

默认认为拆得不够。

继续拆。

旧 giant block 可以保留：

```text
beginner_visible=False
```

但正式 paper recipe 不得继续使用。

---

# 14. Agent 每完成一篇只输出这个短报告

```text
PAPER: <id>

Protocol source:
<path>

Missing capabilities:
[]

New public blocks:
- ...

New paper blocks:
- ...

Protocol: PASS
Formula: PASS
Runtime-limited run: PASS
UI: PASS

Status: READY
Next index: N
```

不要输出长篇总结。

---

# 15. BLOCKED 的处理

如果当前论文遇到无法确认的问题：

例如：

```text
找不到正式 config
论文与旧实现冲突
外部 pretrained artifact 缺失
当前仓库根本没实现某正式阶段
```

状态：

```yaml
status: BLOCKED
notes: "具体原因"
```

不要：

```text
降低标准
换 smoke
跳下一篇
```

先解决当前阻塞。

---

# 16. Context compaction 后恢复流程

Agent 只执行：

```text
1. 重新读 SCRATCH_PAPER_MIGRATION_LOOP_PLAN.md
2. 重新读 SCRATCH_PAPER_MIGRATION_STATE.yaml
3. 找 current_index
4. 查看当前论文 status 和 missing_capabilities
5. 从 A-J 中第一个未完成阶段继续
```

不需要恢复之前的聊天历史。

---

# 17. 最终验收

当：

```yaml
current_index: 26
```

并且 26 篇全部：

```text
status=READY
```

才算论文迁移结束。

最终模板页必须有：

```text
26 / 26
协议正确
公式可拖
受限运行可执行
```

---

# 18. 唯一总原则

每处理一篇只问两件事：

```text
1. 如果取消 runtime limit，它是不是会按正式实验协议运行？
2. 论文核心数学是不是已经拆成可以拿去别处使用的积木？
```

两个答案都必须是：

```text
YES
```

否则不要 index + 1。
