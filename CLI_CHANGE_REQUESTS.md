# CLI 修改记录

本文件用于记录 CLI 使用过程中发现的问题、期望行为和验收结果。

## 当前环境

- 工作目录：`C:\Users\lenovo\.codex\worktrees\440a\LNL-toolbox`
- 开发分支：`codex/cli`
- 推荐环境：`conda activate pytorch`

## 待处理要求

请在后续操作中逐项补充：

1. 执行的命令。
2. 实际看到的结果或报错。
3. 期望的操作体验。
4. 修改后的验收命令。

## 变更记录

### 1. `list experiments` 默认输出难以阅读

- 执行命令：`lnl list experiments --profile smoke`
- 实际问题：输出使用无边框的管道符表格；recipe 较长时，PowerShell 自动换行会破坏列关系，字段含义也不直观。
- 期望体验：默认输出适合人类逐项阅读，清楚展示数据集、噪声、方法、执行器、规模、训练轮数和下一条命令。
- 处理方案：默认改为带编号和中文标签的分块列表；新增 `--format tsv`，保留适合脚本处理的稳定格式。
- 验收命令：

  ```powershell
  lnl list experiments --profile smoke
  lnl list experiments --profile smoke --format tsv
  ```

### 2. 其他目录命令存在相同的长行表格问题

- 涉及命令：`lnl list components`、`lnl papers list`、`lnl papers show <论文ID>`。
- 实际问题：组件和论文列表仍使用无边框管道符表格；论文详情把一份配置的全部字段压在一行，终端换行后难以辨认。
- 处理方案：组件按类型分组；论文按篇编号展示；论文配置按字段逐行展示；两个列表均支持 `--format tsv`。
- 细节处理：标签来源与模型选择说明改为中文；阶段式 runner 不再显示含义不明的 `-`，而是列出各阶段实际配置或说明由执行器决定。
- 验收命令：

  ```powershell
  lnl list components --kind loss
  lnl list components --kind loss --format tsv
  lnl papers list
  lnl papers list --format tsv
  lnl papers show dual-t
  ```

### 3. 组件列表缺少“如何组合”的操作入口

- 实际问题：`list components` 能看到注册项，但用户无法判断哪些槽位兼容，也不能安全生成组合配置。
- 处理方案：新增 `lnl compose list/check/create`。按 runner 展示组合边界；从 supervised recipe 生成新 YAML；写入前执行组件实例化和跨字段校验；默认拒绝覆盖文件。
- 安全边界：专用 runner 只显示完整 recipe；首版不允许把其论文生命周期拆成通用组件。
- 验收命令：

  ```powershell
  lnl compose list --runner supervised
  lnl compose list --runner dual_t
  lnl compose create --base cifar10-symmetric-ce-smoke --loss gce `
    --selector small_loss --keep-rate 0.6 `
    --output configs/experiment/my_gce_small_loss.yaml
  lnl compose check --config configs/experiment/my_gce_small_loss.yaml
  ```
