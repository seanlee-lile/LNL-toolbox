SCRATCH_PAPER_MIGRATION_LOOP_PLAN.md

目标：把论文算法转换成 Scratch 中可拖、可复用、可组合、可运行的积木与 Paper Recipe。

本计划是开发计划，不是审计计划。

1. 核心原则

对 SCRATCH_PAPER_MIGRATION_STATE.yaml 中 current_index 指向的论文：

读取正式配置 + 旧实现 + 论文
↓
检查 Scratch 缺什么
↓
缺什么就实现什么
↓
已有积木能表达就复用
↓
拆核心公式
↓
重写 Paper Recipe
↓
验证
↓
通过 → READY
↓
current_index + 1

发现缺失不是结束条件，而是开发任务。

例如发现 Scratch 缺：

模型
scheduler
数据处理
noise manifest
batch contract
history state
EMA
GMM
多模型生命周期
某个公式 block

必须先实现，再继续验证。

不得因为“当前 Scratch 还没有”就直接标 BLOCKED。

2. 本任务的真实目标

目标是：

把仓库中已经存在的论文算法
↓
转换成 Scratch 可表达的结构
↓
论文核心步骤变成可拖积木
↓
Paper Recipe 能完整表达原有算法流程

不是要求：

完整跑完论文训练
复现论文最终 accuracy
下载所有论文实验资产
真正执行 120/200 epoch

3. 正式 Recipe 与测试必须分开

Paper Recipe 必须保留正式配置所表达的：

dataset
preprocessing
noise
model
optimizer
scheduler
epochs
lifecycle
算法公式

为了快速验证，可以使用：

max_epochs
max_batches
fixture
mock external artifact

但这些只能用于测试。

禁止为了测试方便缩水正式 Recipe。

禁止：

CIFAR → synthetic
ResNet → MLP
120 epochs → Recipe 改成 1
删除 scheduler
删除 noise manifest
删除正式训练阶段

4. 外部资源不等于 BLOCKED

如果论文需要：

checkpoint
pretrained model
artifact
manifest
特殊数据文件

Scratch 应该提供相应输入能力，例如：

Load Checkpoint
Load Artifact
Load Manifest

结构验证时允许用兼容 fixture 测试接口和后续调用链。

本机缺少真实 checkpoint / artifact / dataset，不得因此直接 BLOCKED。

5. BLOCKED 只允许一种情况

只有当：

无法从论文
+
正式配置
+
仓库旧实现

确定算法本身应该如何实现，

或者存在真实数学/实现冲突且无法判断当前项目应采用哪种 fidelity，

才允许：

status: BLOCKED

以下都不是 BLOCKED：

Scratch 缺 block
Scratch 缺模型
Scratch 缺 scheduler
Scratch recipe 还是 smoke
Scratch 缺 lifecycle
本机没有 checkpoint
本机没有完整数据
完整训练太慢

这些都属于：

实现 / fixture / runtime limit

6. 每篇论文固定执行流程

每篇只执行下面 6 步。

STEP 1 — 读取来源

读取当前论文：

paper_catalog.json
正式 / reproduction config
旧正式实现
论文 Method / Algorithm / Equations

先确认：

正式算法到底做什么
正式 Recipe 需要哪些组件

不要以 Scratch 当前 smoke 为标准。

STEP 2 — 检查并补 Scratch 能力

对照正式流程检查：

数据
预处理
噪声
模型
优化器
scheduler
batch 数据结构
训练阶段
状态
评估
外部输入

Scratch 缺什么：

直接实现。

优先实现成公共能力。

实现后继续当前论文，不要停止。

STEP 3 — 复用已有积木，再拆新公式

新增任何 block 前：

先搜索：

src/lnl_toolbox/scratch/blocks/
registry

如果已有数学语义相同的 block：

必须复用。

如果现有 block 只缺：

参数
slot
save_as
通用选项

优先扩展已有公共 block。

只有数学/算法语义确实不同才新增。

7. 什么必须拆成积木

论文中的独立算法步骤/公式，例如：

loss
risk
sample weight
selection score
transition formula
matrix operation
pseudo label
history update
EMA
GMM clean probability
meta weight
label diffusion
centroid
agreement loss

应成为独立可拖 block。

不要把一整篇论文核心数学藏在：

<Paper> Objective
<Paper> Train Step
<Paper> Entire Loss

这种 giant block 中。

旧 giant block 可以暂时保留兼容：

beginner_visible=False

正式 Paper Recipe 不再使用。

8. 新积木必须可复用

新 block 不能绑定：

paper name
recipe name
固定 Context key
固定输出名
固定 batch size
固定 class 数

应该通过：

输入 slot
参数
save_as

连接。

paper metadata 只表示来源，不限制使用范围。

9. STEP 4 — 重写正式 Paper Recipe

修改：

src/lnl_toolbox/scratch/recipes/papers/<paper>.yaml

Recipe 应表达：

正式数据/输入
正式模型
正式 optimizer/scheduler
正式 lifecycle
拆开的论文公式
评估流程

如果论文有：

warmup
stage1/stage2
双模型
EMA
meta
transition estimation
GMM

都应在 Scratch Recipe 中看得出来。

10. STEP 5 — 验证

每篇只做三类验证。

A. Algorithm / Formula

核对：

论文
↔
旧实现
↔
Scratch block

核心公式必须一致。

如果论文与旧实现存在已知 fidelity 差异：

按当前项目正式配置选择的 fidelity 实现，并记录。

不要偷偷任选一个。

B. Recipe Expressiveness

确认 Scratch Recipe 能表达正式流程：

数据
模型
噪声
optimizer
scheduler
epochs
阶段
算法步骤

这里检查的是：

能否表达

不是要求完整复现实验。

C. Structural Runtime

能用真实资源时：

正式 Recipe + runtime limit

例如：

max_epochs=1
max_batches=1

有外部资源缺失时：

使用 contract-compatible fixture

只验证：

调用链
shape
状态更新
梯度
slot

不得因为测试 fixture 而修改正式 Paper Recipe。

11. UI 验证

每篇只需要确认：

Paper Recipe 能被 WebUI 正常加载
核心 blocks 能在 palette/recipe 中显示
Validate 能正常工作

不要求 26 篇每篇都人工“删除再拖回”。

出现新的 block 类型、拖放规则或 UI 能力时，再做代表性的真实拖放检查。

12. READY 条件

当前论文满足：

算法步骤已正确拆解
+
Scratch 能表达正式流程
+
结构运行验证通过
+
Recipe 可在 WebUI 中加载

即可：

status: READY

然后：

current_index += 1

13. 连续执行规则

从当前 current_index 开始：

连续处理所有剩余论文
直到 index = 26

不要每篇停下来等待用户确认。

如果某篇真的符合 BLOCKED 条件：

记录原因
继续下一 index

所以：

BLOCKED ≠ 停止整批任务

14. State 文件只维护进度

使用：

SCRATCH_PAPER_MIGRATION_STATE.yaml

至少维护：

current_index: 15

papers:
  - index: 15
    id: cnlcu
    status: IN_PROGRESS
    notes: ""

完成：

status: READY

真正无法确定算法：

status: BLOCKED
notes: "无法确定的具体问题"

不要因为普通 missing capability 写 BLOCKED。

15. Context compaction / 新对话恢复

每开始一个新对话或发生 context compaction：

重新读取：

SCRATCH_PAPER_MIGRATION_LOOP_PLAN.md
SCRATCH_PAPER_MIGRATION_STATE.yaml

然后从：

current_index

继续。

不要依赖之前聊天中的隐含规则。

16. GCE 事故后的永久规则

之前出现过：

正式 GCE
↓
为了 smoke
↓
synthetic + MLP

这种情况以后禁止。

正确做法是：

正式 Recipe 保持正式
↓
Scratch 缺能力就补能力
↓
测试用 runtime limit

17. 唯一判断标准

每篇论文完成前只问两个问题：

1

Scratch 是否已经能够表达仓库中这篇论文的真实算法流程？

2

这篇论文的核心算法步骤是否已经变成可复用积木？

两个答案都是 YES：

READY

否则：

继续修改

不要把“发现问题”当成工作结束。