# 标签噪声学习的概率模型

标签噪声学习（Learning with Noisy Labels, LNL）本质上是一个带隐变量的概率学习问题。训练数据提供的标签可能错误，而真正希望预测的真实标签并不可见。

## 1. 基本随机变量

用以下随机变量描述标签噪声问题：

- $X$：观测到的输入，例如图像；
- $Y$：不可见的真实标签；
- $\widetilde{Y}$：数据集中观测到的、可能错误的标签；
- $Z$：标签是否干净的潜在指示变量。

训练时能够观察到的数据为：

$$
\mathcal{D}_{\mathrm{noisy}}
=
\{(x_i,\widetilde{y}_i)\}_{i=1}^{N}.
$$

真正希望学习的是干净标签的后验分布：

$$
P(Y\mid X).
$$

一个最基本的生成过程可以表示为：

$$
X \longrightarrow Y \longrightarrow \widetilde{Y}.
$$

对于实例依赖噪声，$X$ 还会直接影响标签被破坏的方式，因此更准确地写成：

$$
P(\widetilde{Y}\mid Y,X).
$$

## 2. 噪声转移模型

定义实例级标签转移概率：

$$
T_{ij}(x)
=
P(\widetilde{Y}=j\mid Y=i,X=x).
$$

它表示：对于真实类别为 $i$ 的样本 $x$，其标签被观测成类别 $j$ 的概率。

根据全概率公式，观测标签的后验分布为：

$$
P(\widetilde{Y}=j\mid X=x)
=
\sum_{i=1}^{K}
P(\widetilde{Y}=j\mid Y=i,X=x)
P(Y=i\mid X=x).
$$

写成矩阵形式：

$$
\widetilde{\boldsymbol{p}}(x)
=
T(x)^{\mathsf T}\boldsymbol{p}(x),
$$

其中：

$$
\boldsymbol{p}(x)
=
\begin{bmatrix}
P(Y=1\mid x)\\
\vdots\\
P(Y=K\mid x)
\end{bmatrix}
$$

是真实标签后验，而：

$$
\widetilde{\boldsymbol{p}}(x)
=
\begin{bmatrix}
P(\widetilde{Y}=1\mid x)\\
\vdots\\
P(\widetilde{Y}=K\mid x)
\end{bmatrix}
$$

是 noisy label 的后验。

因此，LNL 最核心的概率关系是：

$$
\boxed{
P(\widetilde{Y}\mid X)
=
T(X)^{\mathsf T}P(Y\mid X)
}.
$$

## 3. 不同噪声类型

不同标签噪声类型，本质上是对 $T(x)$ 作出了不同假设。

### 3.1 对称噪声

假设每个标签以相同比例翻转到其余类别：

$$
P(\widetilde{Y}=j\mid Y=i)
=
\begin{cases}
1-\rho, & j=i,\\[4pt]
\dfrac{\rho}{K-1}, & j\ne i,
\end{cases}
$$

其中 $\rho$ 是请求的噪声率，$K$ 是类别数。

### 3.2 类别条件噪声

类别条件噪声（Class-Conditional Noise, CCN）假设标签错误依赖真实类别，但不依赖具体样本：

$$
P(\widetilde{Y}\mid Y,X)
=
P(\widetilde{Y}\mid Y).
$$

因此转移矩阵为固定矩阵：

$$
T_{ij}
=
P(\widetilde{Y}=j\mid Y=i).
$$

例如，真实的猫更容易被误标成狗，卡车更容易被误标成汽车。

### 3.3 实例依赖噪声

实例依赖噪声（Instance-Dependent Noise, IDN）允许噪声机制依赖具体输入：

$$
P(\widetilde{Y}\mid Y,X)
\ne
P(\widetilde{Y}\mid Y).
$$

此时：

$$
T_{ij}(x)
=
P(\widetilde{Y}=j\mid Y=i,X=x).
$$

同属一个真实类别的两个样本，也可能具有完全不同的标签出错概率。例如，清晰且典型的猫图像不容易被误标，遮挡严重或同时出现狗的猫图像则更容易被误标。

## 4. 不可辨识性

训练数据只能直接提供关于 $P(\widetilde{Y}\mid X)$ 的信息，但等式右侧同时包含两个未知对象：

$$
P(\widetilde{Y}\mid X)
=
T(X)^{\mathsf T}P(Y\mid X).
$$

未知对象分别是：

- 真实标签后验 $P(Y\mid X)$；
- 噪声机制 $T(X)$。

同一个观测标签分布，可能由多组不同的真实后验和噪声机制共同产生。因此，只依靠 noisy dataset，通常不能无条件地同时恢复 $T(X)$ 和 $P(Y\mid X)$。这就是标签噪声学习的不可辨识性问题。

论文通常需要增加额外假设，例如：

- 转移矩阵与实例无关；
- 数据中存在 anchor points；
- 神经网络先学习简单模式，随后才记忆噪声；
- 干净样本通常具有较小损失；
- 真实类别的预测置信度具有持续上升趋势；
- 噪声机制具有低秩、局部或 part-dependent 结构；
- 二阶统计量能够提供一阶边际分布之外的约束。

阅读一篇 LNL 论文时，首先应当询问：

> 这篇论文为了让不可辨识问题变得可解，增加了什么假设？

## 5. 干净指示变量

引入潜在变量：

$$
Z_i\in\{0,1\},
$$

其中：

$$
Z_i
=
\begin{cases}
1, & \widetilde{y}_i=y_i,\\
0, & \widetilde{y}_i\ne y_i.
\end{cases}
$$

它表示第 $i$ 个样本的观测标签是否干净。相应的联合概率可以分解为：

$$
P(\widetilde{Y},Y,Z\mid X)
=
P(Y\mid X)
P(Z\mid X,Y)
P(\widetilde{Y}\mid X,Y,Z).
$$

当 $Z=1$ 时，通常有：

$$
\widetilde{Y}=Y.
$$

当 $Z=0$ 时，观测标签由噪声分布生成：

$$
\widetilde{Y}
\sim
P_{\mathrm{noise}}(\widetilde{Y}\mid X,Y).
$$

从这个角度看，大量 sample-selection 方法都在近似估计：

$$
P(Z_i=1\mid x_i,\widetilde{y}_i,\text{training history}).
$$

即：第 $i$ 个样本为干净样本的后验概率。

## 6. Sample selection 的概率解释

### 6.1 Small-loss

Small-loss 方法隐含地假设：

$$
P(Z_i=1\mid \ell_i\text{ 较小})
>
P(Z_i=1\mid \ell_i\text{ 较大}).
$$

即损失越小，样本越可能干净。这一关系并不是概率学上的必然结论，而是建立在神经网络训练动态的经验假设上。

### 6.2 混合模型

DivideMix 等方法把逐样本损失看成由干净分量和噪声分量组成的混合分布：

$$
P(\ell)
=
\pi_{\mathrm{clean}}P(\ell\mid Z=1)
+
\pi_{\mathrm{noisy}}P(\ell\mid Z=0).
$$

通过拟合双峰高斯混合模型，可以估计：

$$
P(Z_i=1\mid \ell_i).
$$

因此，这类方法可以输出连续的 clean probability，而不只是二值选择结果。

### 6.3 DSS 的 BASE

DSS 的 BASE 使用预测与 noisy label 是否一致来构造硬选择结果：

$$
\widehat{Z}_i
=
\mathbb{I}
\left[
\arg\max_y P_\theta(y\mid x_i)
=
\widetilde{y}_i
\right].
$$

但预测与 noisy label 一致并不严格等价于标签真实，因为模型可能已经记忆了错误标签。这会产生自我确认偏差。

## 7. DSS 中 MDA 的概率意义

根据贝叶斯公式：

$$
P(Y=y\mid X=x)
\propto
P(X=x\mid Y=y)P(Y=y).
$$

模型预测既受到类条件信息 $P(X\mid Y)$ 的影响，也受到动态类别边际 $P(Y)$ 的影响。容易学习的类别可能具有更高的预测边际，从而在 sample selection 中被过度选择。

MDA 使用指数滑动平均估计动态预测边际：

$$
p'_t(y)
=
\lambda p'_{t-1}(y)
+
(1-\lambda)
\frac{1}{|B_t|}
\sum_{x\in B_t}p_\theta(y\mid x).
$$

然后调整模型预测：

$$
\widehat{p}(y\mid x)
=
\frac{p_\theta(y\mid x)/p'(y)}
{\displaystyle\sum_{c=1}^{K}p_\theta(c\mid x)/p'(c)}.
$$

MDA 不是独立的 clean-sample 判据，而是对 selector 使用的预测分数进行类别级去偏。BASE 随后根据调整后的概率选择样本：

$$
\widehat{Z}_i
=
\mathbb{I}
\left[
\arg\max_y\widehat{p}(y\mid x_i)
=
\widetilde{y}_i
\right].
$$

## 8. DSS 中 CCS 的概率意义

CCS 观察某个样本对类别 $c$ 的预测置信度序列：

$$
\left\{
p_{\theta_t}(Y=c\mid x_i)
\right\}_{t=1}^{T}.
$$

如果该置信度随训练过程持续显著上升，CCS 就把类别 $c$ 视为可能的真实类别。可以将其理解为：

$$
P\!\left(
Y_i=c
\mid
\text{confidence history}
\right)
$$

可能较高。

但置信度上升并不能保证该类别一定正确，因此 CCS 不直接重标注，而是暂时将候选类别集合 $I_i$ 从交叉熵的分母中排除：

$$
\ell_{\mathrm{CCS}}
=
-\log
\frac{
\exp f_\theta(x_i)_{\widetilde{y}_i}
}{
\displaystyle
\sum_{c\in\mathcal{Y}\setminus I_i}
\exp f_\theta(x_i)_c
}.
$$

如果真实类别 $y_i$ 被识别为候选类别，则普通交叉熵和 CCS 对其 logit 的梯度分别为：

$$
\frac{\partial\ell_{\mathrm{CE}}}
{\partial f_\theta(x_i)_{y_i}}
=
p_\theta(y_i\mid x_i)>0,
$$

以及：

$$
\frac{\partial\ell_{\mathrm{CCS}}}
{\partial f_\theta(x_i)_{y_i}}
=
0.
$$

因此，CCS 不会奖励候选类别，但会停止使用错误 noisy label 压低该类别。

## 9. UPM 的隐变量模型

UPM 为每个样本引入潜在混淆程度 $\eta_i$，用来描述该样本的观测标签有多大程度值得信任。其核心推断对象是：

$$
P(Y_i\mid x_i,\widetilde{y}_i,\eta_i).
$$

在一般形式下，真实标签的后验满足：

$$
q_i(y)
=
P(Y_i=y\mid x_i,\widetilde{y}_i)
\propto
P_\theta(Y_i=y\mid x_i)
P(\widetilde{y}_i\mid Y_i=y,x_i).
$$

算法可以使用类似 EM 的交替过程：

1. E-step：根据当前模型和噪声机制估计真实标签后验 $q_i(y)$；
2. M-step：使用 $q_i(y)$ 更新模型参数，并更新样本级混淆参数 $\eta_i$。

UPM 展示了 LNL 作为隐变量推断问题的一种直接建模方式。

## 10. CAL 与二阶统计量

仅观察一阶分布 $P(\widetilde{Y}\mid X)$ 时，真实后验和实例依赖噪声机制可能相互混淆。CAL 引入二阶统计信息，例如变量之间的协方差或联合变化关系，以提供一阶边际分布之外的约束。

直观来说：

- 一阶统计描述某个变量平均出现多少；
- 二阶统计描述两个变量如何共同变化。

二阶统计量的意义不是单纯增加模型复杂度，而是尝试增加可观测约束，缓解 IDN 中的不可辨识问题。

## 11. PDL 的结构化转移模型

PDL 不允许 $T(x)$ 对每个样本完全自由变化，而是利用局部组成或视觉 part 为实例依赖噪声加入结构假设。其思想可以抽象为：

$$
T(x)
=
\sum_{r=1}^{R}h_r(x)T^{(r)},
$$

其中：

- $T^{(r)}$：第 $r$ 个基础噪声转移模式；
- $h_r(x)$：样本 $x$ 对该模式的权重；
- $T(x)$：最终的实例级转移矩阵。

结构化表示比为每个样本自由学习一个完整的 $K\times K$ 转移矩阵更容易约束，也更可能具有可辨识性。

## 12. LNL 方法的统一概率分类

### 12.1 估计噪声过程

目标是估计：

$$
P(\widetilde{Y}\mid Y,X).
$$

代表方法包括 transition matrix、loss correction、UPM、CAL 和 PDL。

### 12.2 估计样本是否干净

目标是估计：

$$
P(Z=1\mid X,\widetilde{Y},\text{history}).
$$

代表方法包括 small-loss、Co-teaching、JoCoR、DivideMix 和 DSS。

### 12.3 估计真实标签

目标是估计：

$$
P(Y\mid X,\widetilde{Y}).
$$

代表路线包括 label correction、pseudo-labeling 和 EM。CCS 则使用训练动态识别可能的真实类别，但采取比直接重标注更保守的决策。

### 12.4 构造噪声鲁棒风险

有些方法不显式估计 $T$、$Z$ 或 $Y$，而是直接构造噪声鲁棒目标：

$$
R(\theta)
=
\mathbb{E}_{X,\widetilde{Y}}
\left[
\ell_{\mathrm{robust}}
(f_\theta(X),\widetilde{Y})
\right].
$$

GCE、normalized loss 等方法属于这一方向。

## 13. 对 LNL-toolbox 的启示

Toolbox 不应假设所有算法都只输出“干净/噪声”二值判断。不同算法可能产生：

```python
SampleInference(
    clean_probability=clean_probability,       # P(Z=1 | ...)
    sample_weight=sample_weight,               # 逐样本训练权重
    corrected_distribution=corrected_prob,     # P(Y | X, noisy label)
    noise_transition=transition,               # P(noisy Y | Y, X)
    selected_mask=selected_mask,               # 硬选择结果
    candidate_classes=candidate_classes,       # 可能的真实类别
)
```

这些对象不一定需要由同一个算法全部提供，但统一接口应当允许表达它们。

标签噪声学习最核心的认识是：

$$
\boxed{
\text{LNL 是在只观察到 }(X,\widetilde{Y})\text{ 的条件下，}
\text{推断不可见的 }Y、Z\text{ 或噪声机制。}
}
$$

## 14. 阅读论文时的四个问题

阅读每篇 LNL 论文时，可以固定回答以下问题：

1. 论文定义了哪些随机变量？
2. 哪些变量可以观察，哪些是隐变量？
3. 论文对 $P(\widetilde{Y}\mid Y,X)$ 作了什么假设？
4. 方法最终估计的是 $T(X)$、$P(Z=1)$、$P(Y\mid X)$，还是只构造了鲁棒风险？

回答完这四个问题，一篇论文的概率模型通常就已经基本清晰。
