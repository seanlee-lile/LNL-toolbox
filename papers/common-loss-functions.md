# 常见分类损失函数：公式、直觉与代码实现

本文用于阅读 Learning with Noisy Labels（LNL）论文时快速查找常见损失函数。重点不是记忆论文中的长公式，而是理解它们在单标签分类中真正计算了什么。

## 1. 统一符号

对于一个 (K) 分类样本 (x)：

- (z_k)：网络对第 (k) 类输出的 logit；
- (p_k=p(k\mid x))：softmax 后第 (k) 类的预测概率；
- (q_k=q(k\mid x))：数据标签的 one-hot 编码；
- (y)：数据中观察到的标签，在 LNL 中可能是 noisy label；
- (y^*)：未知的真实标签；
- (p_y)：模型分配给观察标签 (y) 的概率。

其中：

\[
p_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}},
\qquad
\sum_{k=1}^{K}p_k=1
\]

单标签分类中的 (q) 是 one-hot 向量：

\[
q_y=1,\qquad q_{k\ne y}=0
\]

因此，只要公式中存在 (q_k)，通常都可以利用 one-hot 性质大幅化简。

例如三分类中：

\[
p=[0.7,0.2,0.1],\qquad q=[1,0,0],\qquad y=1
\]

下面均使用自然对数。

---

## 2. Cross Entropy（CE）

### 原始公式

\[
L_{\mathrm{CE}}
=-\sum_{k=1}^{K}q_k\log p_k
\]

因为只有 (q_y=1)，所以：

\[
\boxed{L_{\mathrm{CE}}=-\log p_y}
\]

示例：

\[
L_{\mathrm{CE}}=-\log0.7\approx0.357
\]

### 直觉

CE 强迫模型提高观察标签 (y) 的预测概率。当 (p_y\to0) 时，损失趋近无穷大，因此模型会持续关注自己不认同的标签。

在干净数据中，这是 CE 学习能力强的原因；在 noisy label 中，错误标签往往具有很小的 (p_y)，却会产生很大的损失和梯度，最终诱导网络记忆错误标签。

### 代码

```python
per_sample_loss = F.cross_entropy(
    logits,
    targets,
    reduction="none",
)
```

手工形式：

```python
log_p = F.log_softmax(logits, dim=1)
per_sample_loss = -log_p.gather(1, targets[:, None]).squeeze(1)
```

### LNL 特点

- 优点：收敛快，学习能力强；
- 缺点：损失无上界，容易让高损失噪声样本支配训练；
- 不需要噪声率、转移矩阵或干净验证集。

---

## 3. Mean Absolute Error（MAE）

### 原始公式

\[
L_{\mathrm{MAE}}
=\sum_{k=1}^{K}|p_k-q_k|
\]

展开标签类别与非标签类别：

\[
L_{\mathrm{MAE}}
=|p_y-1|+\sum_{k\ne y}|p_k-0|
\]

因为 (p_k\in[0,1])，并且 (sum_{k\ne y}p_k=1-p_y)：

\[
L_{\mathrm{MAE}}
=(1-p_y)+(1-p_y)
\]

所以：

\[
\boxed{L_{\mathrm{MAE}}=2(1-p_y)}
\]

示例：

\[
L_{\mathrm{MAE}}=2(1-0.7)=0.6
\]

### 直觉

MAE 直接计算预测概率向量与 one-hot 标签向量之间的 (L_1) 距离。它的取值有界：

\[
0\le L_{\mathrm{MAE}}\le2
\]

因此，错误标签不会像在 CE 中那样产生无限大的损失。

但是：

\[
\frac{\partial L_{\mathrm{MAE}}}{\partial z_y}
=-2p_y(1-p_y)
\]

当模型给观察标签的概率 (p_y\approx0) 时，梯度也接近零。这可以抑制错误标签，却也会忽略真正困难但标签正确的样本，因而容易欠拟合。

### 代码

```python
p = F.softmax(logits, dim=1)
p_y = p.gather(1, targets[:, None]).squeeze(1)
per_sample_loss = 2.0 * (1.0 - p_y)
```

### LNL 特点

- 优点：损失有界，在对称标签噪声等假设下具有理论鲁棒性；
- 缺点：困难样本梯度很小，优化慢，容易欠拟合；
- 不需要噪声率或转移矩阵。

注意：有些实现省略系数 2，使用 (1-p_y)。这不会改变最优方向，但会改变梯度尺度以及与其他损失组合时的相对权重。

---

## 4. Reverse Cross Entropy（RCE）

### 原始公式

普通 CE 是：

\[
-\sum_k q_k\log p_k
\]

RCE 交换 (p) 与 (q) 的位置：

\[
L_{\mathrm{RCE}}
=-\sum_{k=1}^{K}p_k\log q_k
\]

one-hot 标签包含 (q_{k\ne y}=0)，但 (log0) 不存在。因此论文将非标签位置截断为一个负常数：

\[
\log q_y=0,
\qquad
\log q_{k\ne y}=A,
\qquad A<0
\]

例如 (A=-4)。于是：

\[
L_{\mathrm{RCE}}
=-p_y\log1-\sum_{k\ne y}p_kA
\]

\[
=-A\sum_{k\ne y}p_k
\]

所以：

\[
\boxed{L_{\mathrm{RCE}}=-A(1-p_y)}
\]

当 (A=-4) 时：

\[
L_{\mathrm{RCE}}=4(1-p_y)
\]

示例：

\[
L_{\mathrm{RCE}}=4(1-0.7)=1.2
\]

### 直觉

RCE 显式惩罚模型分配给非标签类别的概率。在严格 one-hot 标签设定中，RCE 与 MAE 都正比于 (1-p_y)，本质上只相差一个常数尺度。

### 代码

推荐直接使用化简形式，避免计算 (log0)：

```python
p = F.softmax(logits, dim=1)
p_y = p.gather(1, targets[:, None]).squeeze(1)

A = -4.0
per_sample_loss = -A * (1.0 - p_y)
```

### LNL 特点

- 优点：有界，具有类似 MAE 的抗噪性质；
- 缺点：单独使用也容易欠拟合；
- 超参数 (A) 主要控制损失尺度；
- 不需要噪声率或转移矩阵。

---

## 5. Focal Loss（FL）

### 原始公式

\[
L_{\mathrm{FL}}
=-\sum_{k=1}^{K}q_k(1-p_k)^\gamma\log p_k,
\qquad \gamma\ge0
\]

利用 one-hot 标签化简：

\[
\boxed{
L_{\mathrm{FL}}
=-(1-p_y)^\gamma\log p_y
}
\]

也就是：

\[
L_{\mathrm{FL}}
=(1-p_y)^\gamma L_{\mathrm{CE}}
\]

当 (gamma=0) 时：

\[
L_{\mathrm{FL}}=L_{\mathrm{CE}}
\]

示例取 (gamma=2)：

\[
L_{\mathrm{FL}}
=(1-0.7)^2(-\log0.7)
\approx0.032
\]

### 直觉

Focal Loss 降低容易样本的权重，将训练重心放到困难样本上：

- (p_y\) 很大：权重 ((1-p_y)^\gamma) 很小；
- (p_y\) 很小：权重接近 1。

这对目标检测中的类别不平衡很有效。但在 LNL 中，困难样本包含大量 noisy samples，FL 可能把更多注意力放到错误标签上，因此它本身不具有本文要求的噪声鲁棒性。

### 代码

```python
log_p = F.log_softmax(logits, dim=1)
log_p_y = log_p.gather(1, targets[:, None]).squeeze(1)
p_y = log_p_y.exp()

per_sample_loss = -(1.0 - p_y).pow(gamma) * log_p_y
```

### LNL 特点

- 优点：聚焦困难样本；
- 缺点：可能同时聚焦错误标签，且仍然继承 CE 在 (p_y\to0) 时的无界性；
- (gamma) 越大，对容易样本的抑制越强。

---

## 6. Generalized Cross Entropy（GCE）

### 公式

\[
\boxed{
L_{\mathrm{GCE}}
=\frac{1-p_y^\rho}{\rho},
\qquad \rho\in(0,1]
}
\]

当 (ho\to0) 时：

\[
\frac{1-p_y^\rho}{\rho}
\longrightarrow-\log p_y
\]

因此 GCE 接近 CE。

当 (ho=1) 时：

\[
L_{\mathrm{GCE}}=1-p_y
\]

它相当于 MAE 的一半。

### 直觉

GCE 用 (ho) 在 CE 的学习能力和 MAE 的鲁棒性之间折中：

- (ho\) 接近 0：更像 CE，学习快，但更容易拟合噪声；
- (ho\) 接近 1：更像 MAE，更抗噪，但更容易欠拟合。

### 代码

```python
p = F.softmax(logits, dim=1)
p_y = p.gather(1, targets[:, None]).squeeze(1)
per_sample_loss = (1.0 - p_y.pow(rho)) / rho
```

实际实现应保证 (ho>0)，并对概率进行数值保护。

---

## 7. Symmetric Cross Entropy（SCE）

SCE 是 CE 与 RCE 的线性组合：

\[
\boxed{
L_{\mathrm{SCE}}
=\alpha L_{\mathrm{CE}}
+\beta L_{\mathrm{RCE}}
}
\]

### 直觉

- CE 提供较强的学习信号；
- RCE 提供有界、相对抗噪的信号；
- (alpha,eta) 控制二者的平衡。

SCE 的名称来自 CE 与 Reverse CE 的组合，并不意味着整个 SCE 在所有噪声条件下都具有严格鲁棒性。本文指出，其中只有 RCE 项满足相应的理论鲁棒条件。

### 代码

```python
ce = -log_p_y
rce = -A * (1.0 - p_y)
per_sample_loss = alpha * ce + beta * rce
```

### LNL 特点

- 优点：比单独使用 MAE/RCE 更容易训练；
- 缺点：(alpha,eta,A) 的尺度互相影响，跨数据集通常需要重新调节；
- 不要求噪声率或转移矩阵，但超参数选择可能受验证集质量影响。

---

## 8. 通用 Normalized Loss

论文《Normalized Loss Functions for Deep Learning with Noisy Labels》提出：

\[
\boxed{
L_{\mathrm{norm}}(x,y)
=\frac{L(x,y)}{\sum_{j=1}^{K}L(x,j)}
}
\]

分母的意思不是 batch 内求和，而是：

> 对同一个样本 (x)，依次假设它的标签为 (1,2,\ldots,K)，分别计算损失，再把这些损失相加。

因此：

\[
\sum_{y=1}^{K}L_{\mathrm{norm}}(x,y)=1
\]

“所有可能标签的损失之和为常数”是论文证明噪声鲁棒性的核心条件。

这不是对训练集、batch 或样本权重做归一化。

---

## 9. Normalized Cross Entropy（NCE）

把每个类别 (j) 都假设为当前标签时：

\[
L_{\mathrm{CE}}(x,j)=-\log p_j
\]

所以：

\[
\boxed{
L_{\mathrm{NCE}}
=\frac{-\log p_y}
{\sum_{j=1}^{K}-\log p_j}
}
\]

示例：

\[
L_{\mathrm{NCE}}
=\frac{-\log0.7}
{-\log0.7-\log0.2-\log0.1}
\approx0.084
\]

### 代码

```python
log_p = F.log_softmax(logits, dim=1)

# 同一个样本分别以每一类为标签时的 CE
all_ce = -log_p                       # [batch_size, num_classes]
target_ce = all_ce.gather(
    1,
    targets[:, None],
).squeeze(1)

per_sample_loss = target_ce / all_ce.sum(dim=1).clamp_min(eps)
```

### 重要局限

NCE 的损失可以因为两种原因下降：

1. 分子 (-\log p_y) 下降，即 (p_y) 上升；
2. 分母中其他类别的损失增大。

第二条意味着：即使 (p_y) 没有明显提高，模型也可能通过把某个非标签类别的概率压得极小而降低 NCE。这是论文解释 NCE 容易欠拟合的重要原因。

---

## 10. Normalized MAE（NMAE）

因为：

\[
L_{\mathrm{MAE}}(x,j)=2(1-p_j)
\]

分母为：

\[
\sum_{j=1}^{K}2(1-p_j)
=2\left(K-\sum_jp_j\right)
=2(K-1)
\]

所以：

\[
\boxed{
L_{\mathrm{NMAE}}
=\frac{1-p_y}{K-1}
=\frac{1}{2(K-1)}L_{\mathrm{MAE}}
}
\]

NMAE 只是 MAE 的常数缩放，不改变单独优化时的最优方向。

---

## 11. Normalized RCE（NRCE）

因为：

\[
L_{\mathrm{RCE}}(x,j)=-A(1-p_j)
\]

所以：

\[
\boxed{
L_{\mathrm{NRCE}}
=\frac{1-p_y}{K-1}
}
\]

在本文的 one-hot 标签设定下：

\[
\boxed{L_{\mathrm{NRCE}}=L_{\mathrm{NMAE}}}
\]

两者的原始定义和解释不同，但归一化后的数学形式相同。

---

## 12. Normalized Focal Loss（NFL）

对每个可能标签 (j)：

\[
L_{\mathrm{FL}}(x,j)
=-(1-p_j)^\gamma\log p_j
\]

因此：

\[
\boxed{
L_{\mathrm{NFL}}
=\frac{-(1-p_y)^\gamma\log p_y}
{\sum_{j=1}^{K}-(1-p_j)^\gamma\log p_j}
}
\]

### 代码

```python
log_p = F.log_softmax(logits, dim=1)
p = log_p.exp()

all_focal = -(1.0 - p).pow(gamma) * log_p
target_focal = all_focal.gather(
    1,
    targets[:, None],
).squeeze(1)

per_sample_loss = target_focal / all_focal.sum(dim=1).clamp_min(eps)
```

---

## 13. Normalized GCE（NGCE）

GCE 对假设标签 (j) 的损失为：

\[
L_{\mathrm{GCE}}(x,j)=\frac{1-p_j^\rho}{\rho}
\]

归一化后，分子和分母中的 (ho) 抵消：

\[
\boxed{
L_{\mathrm{NGCE}}
=\frac{1-p_y^\rho}
{K-\sum_{j=1}^{K}p_j^\rho}
}
\]

代码：

```python
p = F.softmax(logits, dim=1)
p_power = p.pow(rho)
p_y_power = p_power.gather(1, targets[:, None]).squeeze(1)

numerator = 1.0 - p_y_power
denominator = logits.shape[1] - p_power.sum(dim=1)
per_sample_loss = numerator / denominator.clamp_min(eps)
```

---

## 14. Active、Passive 与 APL

本文按照损失公式中是否显式包含非标签类别项，将损失分为两类。

### Active loss

只显式构造观察标签类别 (y) 对应的损失项：

- CE；
- NCE；
- FL；
- NFL；
- GCE；
- NGCE。

### Passive loss

除了提高 (p_y)，还在公式中显式惩罚至少一个非标签类别：

- MAE；
- NMAE；
- RCE；
- NRCE。

注意：Active loss 并不表示反向传播只更新第 (y) 个 logit。softmax 会耦合所有 logits，因此 CE 也会更新所有类别。Active/Passive 是作者对损失公式结构的分类。

### Active Passive Loss（APL）

\[
\boxed{
L_{\mathrm{APL}}
=\alpha L_{\mathrm{Active}}
+\beta L_{\mathrm{Passive}}
}
\]

论文要求其中的 Active 和 Passive 项都具有相应的噪声鲁棒性。非鲁棒损失需要先做归一化，因此典型组合为：

- NCE + MAE；
- NCE + RCE；
- NFL + MAE；
- NFL + RCE；
- NGCE + MAE；
- NGCE + RCE。

它的目标是：用 Active 项增强学习能力，用 Passive 项从另一个方向约束类别概率，从而缓解单个鲁棒损失的欠拟合。

---

## 15. 快速比较

| Loss | 单标签形式 | 是否有界 | 主要特点 |
|---|---|---:|---|
| CE | (-\log p_y) | 否 | 学习强，但容易记忆噪声 |
| MAE | (2(1-p_y)) | 是 | 抗噪，但容易欠拟合 |
| RCE | (-A(1-p_y)) | 是 | one-hot 下近似缩放 MAE |
| FL | (-(1-p_y)^\gamma\log p_y) | 否 | 聚焦困难样本，也可能聚焦噪声 |
| GCE | ((1-p_y^\rho)/\rho) | 是 | 在 CE 与 MAE 之间折中 |
| SCE | (alpha CE+\beta RCE) | 整体否 | 组合学习能力与抗噪性 |
| NCE | (CE_y/\sum_jCE_j) | 是 | 满足常数和，但可能欠拟合 |
| NMAE | ((1-p_y)/(K-1)) | 是 | MAE 的常数缩放 |
| NRCE | ((1-p_y)/(K-1)) | 是 | one-hot 下等于 NMAE |
| NFL | (FL_y/\sum_jFL_j) | 是 | 归一化 FL |
| NGCE | ((1-p_y^\rho)/(K-\sum_jp_j^\rho)) | 是 | 归一化 GCE |

---

## 16. LNL-toolbox 实现注意事项

1. 所有损失模块应先返回 `reduction="none"` 的逐样本损失，再由训练器决定 `mean`、加权或样本选择。
2. 优先使用 `log_softmax`，不要先 `softmax` 再直接 `log`，避免概率下溢。
3. NCE、NFL、NGCE 的分母应使用 `clamp_min(eps)` 做数值保护。
4. RCE 推荐使用 (-A(1-p_y)) 的化简形式，避免显式计算 `log(0)`。
5. MAE 是否包含系数 2、RCE 的 (A)、组合损失的 (alpha,eta) 都会影响梯度尺度，实验配置必须保存这些参数。
6. 上述 one-hot 化简不适用于任意 soft label。若输入 mixup、label smoothing 或伪标签分布，需要回到原始向量公式重新推导。
7. 归一化鲁棒性的结论是总体风险和特定噪声假设下的理论性质，不等于在任意 instance-dependent 或 open-set 噪声下都有效。
8. 这些损失通常不需要已知噪声率、转移矩阵或干净训练集；但 (ho,gamma,alpha,eta,A) 的选择可能依赖验证实验。

建议至少编写以下单元测试：

- CE 手工形式与 `F.cross_entropy` 相等；
- MAE 等于 (2(1-p_y))；
- (A=-4) 时 RCE 等于 (4(1-p_y))；
- (gamma=0) 时 FL 等于 CE；
- (ho\to0) 时 GCE 数值接近 CE；
- (ho=1) 时 GCE 等于 (1-p_y)；
- 对固定样本，归一化损失在所有假设标签上的和约等于 1；
- NMAE 与 NRCE 在 one-hot 设置下相等；
- 极端 logits 下损失与梯度均不出现 `NaN` 或 `Inf`。
