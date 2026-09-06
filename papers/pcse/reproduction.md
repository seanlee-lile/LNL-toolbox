# PCSE real-CIFAR workflow

`cifar10-pcse-reproduction` 是真实数据上的 conditional engineering workflow。它执行当前
PCSE transition、statistic recovery、多层 GDA、noisy-validation ensemble fitting 和
clean-test inference；它不是论文数值或多 seed 结果已经验证的声明。

## Source contract

PCSE 不再只接受 UPM。当前只支持三类角色明确、可严格恢复的 classifier source：

- UPM 主模型：`upm_main_best`；
- 监督 CE 最佳模型：`supervised_best`；
- Co-teaching peer A 最佳模型：`coteaching_peer_a_best`。

JoCoR 等没有单一明确 peer role 的来源目前不支持。Toolbox 不会从任意 checkpoint 猜测
来源角色。source 必须通过 checkpoint schema、方法/角色、模型结构、类别数、dataset
fingerprint、noise manifest/mapping 和 digest 的 identity-compatible 检查。

`LNL_PCSE_SOURCE_RUN` 是提供来源 run directory 的一种方式；正式 recipe 还必须在 YAML
中声明与该来源相符的 identity 字段。来源换成新的 CE、UPM 或 Co-teaching run 后，通常需要
复制 YAML、审计并更新全部 source identity，再执行 Validate。不可通过“只填一个 checkpoint
路径”绕过这些检查。

```powershell
lnl list experiments --profile reproduction --include-conditional
$env:LNL_PCSE_SOURCE_RUN = (Resolve-Path <completed-source-run-directory>)
lnl validate --recipe cifar10-pcse-reproduction --check-data
lnl run --recipe cifar10-pcse-reproduction --dry-run --check-data
lnl run --recipe cifar10-pcse-reproduction --check-data
lnl resume <pcse-run-directory>
```

缺少环境变量、文件、hash、role、architecture 或 manifest provenance 时，Validate 与
Dry-run 会在训练前失败。这是有意的安全合同，不是要求用户随意替换 checkpoint 的提示。

## Data and validation boundaries

PCSE 使用稳定的训练样本索引和对应的 training noise manifest 进行训练侧统计。由 train
split 派生的 validation 可以使用同一 manifest scope；原生/native validation 不属于 train
manifest，保持其自身 observed targets，不会伪造或查询训练 manifest 的 corruption rate。
clean test labels 仅用于最终评价。

source model 只用于初始化 method-local PCSE model；随后 PCSE 按自身既有 lifecycle 更新本地
模型与 transition，不会修改 source checkpoint。transition parameterization 和 covariance
ridge 是显式工程选择，不能写成论文原始超参数。

## Artifacts and claim boundary

检查 `pretrained_best.pt`、`volmin_final.pt`、`transition_artifact.npz`、
`pcse_statistics.npz`、`pcse_gda.npz`、`pcse_ensemble.npz`、`last.pt` 和
`final_metrics.json`。Resume 会验证 method config、source identity、transition/feature
provenance 与 artifact hashes。

该 profile 表示可审计的真实数据 engineering workflow。paper-exact protocol、论文数值表格
和多 seed 聚合仍需独立实验与报告；recipe 存在、Dry-run 或 smoke 成功都不足以证明它们。
