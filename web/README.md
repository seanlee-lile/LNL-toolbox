# LNL Toolbox Web Console

这是一个独立的本地网页入口，位于 web/，不修改现有 CLI、训练 runner 或数据模块。

## 启动

在仓库根目录运行：

    lnl web

然后打开 http://127.0.0.1:8765。
根路径是原有主控制台；Recipe/YAML 编辑器保留为独立子页面：
http://127.0.0.1:8765/recipe。`lnl web` 默认打开主页；`lnl web --no-open` 只启动服务。

直接开发后端时仍可运行：

    python web/command_console.py

如果环境中已经安装了 lnl，网页优先调用 lnl ...；否则回退到：

    python -m lnl_toolbox.cli.main ...

网页中的每个按钮都对应一个固定的参数列表。后端使用 shell=False，不接受浏览器传入的任意命令字符串。训练 1 epoch 是专门的短训练示例，输出目录为 artifacts/web-smoke。

## 快速验证

    python -m unittest discover -s web -p "test_*.py" -v

建议首次使用顺序：查看帮助 → 检查环境 → 查看 Smoke 配方 → 验证 Clean 配置 → 预演一次训练 → 训练 1 epoch。

“本地数据集”模块可生成 `lnl data register/inspect/verify/remove` 和
`lnl run ... --data <alias>`。页面显示机器本地 catalog 的当前状态，但不会读取或上传
数据内容。`registered`、`layout_validated` 与 `training_verified` 是不同状态；只有后端
真实完成 1 epoch 后才会显示 `training_verified`。Web 页面不会自动下载数据集。

数据页面的 list/status/path/register/inspect/remove 请求由后端直接交给公共 `DataService`，
不会在 Web 内复制 adapter 文件名或浅层路径判断。`verify` 仍作为后台任务运行，但其数据
检查、train/test 加载和证据记录同样由 `DataService` 完成。页面中的 `ready` 表示 adapter
已成功加载 train/test；训练证据单独显示。

数据登记页面按三个阶段工作：登记路径、实际加载 train/test、训练 1 epoch 验证。页面只显示当前操作需要的字段；登记后的“已登记，待检查”不代表可训练。删除登记需要二次确认，而且只删除 catalog 记录，不删除原始数据。后端权限或数据错误会作为页面错误返回，不应让页面停留在“启动中”。

数据集与训练模板是两个独立选择。`lnl data verify <alias>` 会根据登记数据自动建立一轮
验证配置，不需要 CIFAR、MNIST 等数据集专属 recipe；`--recipe` 仅保留为高级兼容选项。
新手模式和“使用已登记数据训练”会生成
`lnl run --recipe <template> --data <alias>`。这里的 recipe 是训练模板，`--data` 才决定
实际数据集；覆盖论文 recipe 的数据后只表示方法迁移测试，不再表示论文原始配置复现。
