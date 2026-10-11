# CI 失败修复与断言审计（2026-10-11）

## 范围和证据

对照用户提供的 [CI 运行 38041099927](https://github.com/seanlee-lile/LNL-toolbox/actions/runs/38041099927) 材料、当前源码及本地失败结果处理。GitHub API 确认该运行的 Python 3.12 单元测试 job 失败；原始日志下载需要仓库权限，本次未取得完整原始日志，不能把材料中的日志描述说成本次独立下载结果。

开始时工作区已有主会话修改。本次保留这些修改，只补充 CI 修复；没有修改正式实验 YAML、论文训练公式或外部来源校验规则，没有 push，没有删除测试或新增 skip。

## 材料中的 18 项

| 编号 | 原断言/场景 | 核实后的原因与处理 |
| --- | --- | --- |
| 1 | DivideMix 主训练必须 300 轮 | [论文 §4.1](https://arxiv.org/html/2002.07394) 的 300 轮包含预热。保留 CIFAR-10 的 10 轮预热、290 轮主训练，断言总和为 300；修正文档中的 310 轮。 |
| 2 | LEND 错误必须含 `feature-aware` | 非法模型已正确被拒绝。改查 `ValueError` 和 `model.name`，不固定旧措辞。 |
| 3 | PCSE 无来源环境变量必须失败 | 当前默认是内部预训练。分别验证内部模式通过、外部模式缺少来源仍失败。 |
| 4 | PCSE 默认必须 `external_checkpoint` | 测试过期。检查当前 `train`、交叉熵预训练和无外部来源；不改默认配置。 |
| 5 | Catalog 中 PCSE 缺环境变量必须失败 | 同 #3；内部模式不需要外部路径。 |
| 6 | YAML 数量必须 95 | 新合法配置使数量变为 97。去掉历史总数，保留版本、规范格式、唯一 recipe ID、有效配置路径及必备配方检查。 |
| 7 | 修改嵌套 split seed 必须阻止恢复 | 嵌套 seed 被统一实验 seed 覆盖，没有产生实际身份变化。改测顶层 seed、划分比例和噪声率变化均阻止恢复，并检查失败后 manifest/checkpoint 字节不变。 |
| 8 | L2RW 的 data seed 必须 0 | 过期的独立 seed。保留当前实验 seed 1234，要求 data/noise/trusted validation 全部与它一致。 |
| 9 | CA2C：CIFAR-10 + clean | 原 fixture 实际只有 2 类，K=2 非法。新增真实十类别 fixture 验证 ready/plan；K 保持 2。 |
| 10 | CA2C：Fashion-MNIST + clean | 同 #9；使用十类别灰度图 fixture 验证。 |
| 11 | CA2C：CIFAR-10 + symmetric | 同 #9；合成噪声率 0.2。 |
| 12 | CA2C：Fashion-MNIST + symmetric | 同 #10；合成噪声率 0.2。 |
| 13 | CA2C：CIFAR-10 + pairflip | 同 #9；不降低 K，不放宽类别边界。 |
| 14 | CA2C：Fashion-MNIST + pairflip | 同 #10；不降低 K，不放宽类别边界。 |
| 15 | CA2C：CIFAR-10 + PDL | 同 #9；验证可生成可执行计划。 |
| 16 | CA2C：Fashion-MNIST + PDL | 同 #10；验证可生成可执行计划。 |
| 17 | 改类别数后 matrix 必须不存在 | 当前矩阵已按新类别数和噪声协议重新生成，并非残留旧矩阵。检查 100×100、行和 1、对角 0.8、非对角 0.2/99；另测 pairflip 重建及 PDL 不提供已知矩阵。 |
| 18 | DivideMix 必须要求噪声率先验 | DivideMix 从损失 GMM 得到干净概率，不要求用户提供标量噪声率。保留 CoTeaching/CNLCU 的先验要求，修正 DivideMix 断言。 |

#9–16 的八种十类别组合与两类别非法输入均已单独运行通过。没有把方法数学约束改成“任何数据都能运行”。

## 额外发现

- **实际缺陷：** Quick Start 给 CoTeaching/CNLCU 设置新噪声率先验时，保留率终值仍是原模板值。现在仅在生成的运行配置中同步为 `1 - prior`。测试覆盖两种方法、干净输入和噪声率 0.3；正式模板不变。
- **实际缺陷：** JoCoR 的新先验字段在归一化阶段直接转 float，非法文本提前报错、丢失字段诊断。补充类型/范围错误的字段化提示，并避免无条件求值旧保留率。非法输入测试通过。
- **过期测试：** DLD 外部来源声明已支持来源描述器，测试仍只认旧 UPM role。改查预训练特征提取器类型、验证器、ImageNet/UPM 支持项及配置路径。
- **过期 Web 测试：** 隐藏的 `evaluation.selection_split` 不应被要求继续显示；无已知总样本数不能伪造原论文比例；干净输入上的 UPM 不是缺条件；兼容性入口不应绑定一整段旧 JS。补充实际执行 JS 的噪声信息复用测试，以及缺先验 JoCoR 的 HTTP 不可运行边界。

## 断言审计统计

静态扫描覆盖 `tests/`、`src/lnl_toolbox/scratch/tests/`、`web/`：初次扫描共 91 个测试文件、1,457 个测试方法、5,078 个断言调用；检出 89 个“len 与固定整数比较”的候选。修改后复扫为 91 个文件、1,458 个测试方法、5,082 个断言调用。扫描不等于逐条人工审查所有断言。

以下计数**只针对材料中逐项核实的 18 个失败子用例**，分类互斥，不能外推到全项目：

| 分类 | 核实数量 | 处理 | 说明 |
| --- | --- | --- | --- |
| A 核心正确性 | 0 | — | 材料内没有确认需要放宽的数学/身份约束。 |
| B 行为契约 | 8 | 保留 ready/plan 契约，修正 fixture 并补足八场景 | CA2C #9–16；非法两类别输入继续拒绝。 |
| C 论文复现 | 1 | 修正 1 | DivideMix 总预算断言，仍严格检查 300。 |
| D 过度断言 | 2 | 重构 2 | LEND 文案、历史 YAML 总数。 |
| E 已过期测试 | 7 | 更新 7 | PCSE 三项、嵌套 seed、L2RW seed、矩阵删除假设、DivideMix 先验。 |

额外审查了目录覆盖、错误诊断、种子归一化、外部来源声明和 Web/Scratch 相关测试。修改目录历史总数断言，但保留唯一性、必备配置、覆盖实际目录等约束。固定 fixture 大小、SHA-256 长度、论文网络层数、概率矩阵维度、预检不加载数据、并发去重调用次数等有真实契约的断言未机械重写。其余全部断言没有逐条人工分类，也没有完成 26 篇论文公式的重新审计。

## 三个典型重构

1. `len(paths) == 95` → 目录非空、逐个检查版本和规范结构、recipe ID 唯一、路径有效、必备 GCE 配方存在。新增合法配置不会触发历史计数失败。
2. 错误必须含 `feature-aware` → 非法模型必须抛 `ValueError`，诊断必须指向 `model.name`。仍保护拒绝行为，不固定修辞。
3. 修改 `data.split.seed` 后必须失败 → 修改真正生效的实验 seed、划分比例或噪声率后均拒绝恢复，且已有产物字节不变。保留并加强恢复身份保护。

## 本地验证

使用独立临时目录保存 coverage、包和安装验证环境，不覆盖原有 coverage 或修改日常 Python 环境。

| 检查 | 结果 |
| --- | --- |
| 初次针对性回归 | 58 项，11 个失败子用例；用于定位旧断言及当前工作区变化。 |
| 首次完整单元测试 | 1,060 项，4 个失败子用例；含 JoCoR 两个输入诊断问题，以及运行开始前已载入的旧 resume/DLD 测试。 |
| 修复后完整单元测试 | 1,060 项通过；之后扩充的 CA2C 八场景与拒绝边界另行复测通过。 |
| Scratch 完整测试 | 244 项通过。 |
| Web 首轮完整测试 | 153 项，4 项失败；已定位并修正上述旧假设。 |
| Web 针对性回归 | 5 项通过，包含 HTTP 与 Node 执行验证。 |
| Web 最终完整回归 | 154 项通过，零失败；包含新增的 Node 执行测试。 |
| Ruff、git diff --check | 通过。 |
| wheel / sdist 构建 | 通过。 |
| wheel / sdist 独立安装 | 两者的 `lnl --help`、smoke 配方列表、VolMinNet 配置校验均通过。此处仅是 CI 包安装检查，不是 Web 训练验收。 |
| Python 3.10 | 408 个文件语法编译通过；运行 import 失败，因为该环境未安装 PyTorch。没有给用户环境安装依赖。 |
| Python 3.12 / Ubuntu | 未在本机验证；GitHub 重新运行后才能确认。 |
| coverage report | 三套数据合并成功，报告总覆盖率 75%，退出码 0。CI 未设置 fail-under 门槛。没有取得修复前同范围覆盖率基线，不能宣称覆盖率提升或下降。 |

对应命令：

```text
python -m coverage run -m unittest discover -s tests -q
python -m coverage run -m unittest discover -s src/lnl_toolbox/scratch/tests -q
python -m coverage run -m unittest discover -s web -p test_*.py -v
python -m ruff check src tests web
git diff --check
python -m build --no-isolation --outdir <临时目录>/dist
```

三套 coverage 分别写临时数据文件，合并后执行 `coverage report`，完整套件合计 1,458 项通过。主验证环境为 Windows / Python 3.11.15；没有声称远程 CI 已变绿。两条因本地 pip 缓存/检查而滞留的早期安装进程已在核对命令行后停止；随后无缓存独立安装检查全部通过，未停止用户 Web 服务。

## 本次修改文件

- 生产代码：`src/lnl_toolbox/quickstart/templates.py`、`src/lnl_toolbox/core/config_schema.py`，只补充上述保留率同步与输入诊断。
- 单元测试：`tests/test_cli.py`、`test_config.py`、`test_importance_reweighting.py`、`test_l2rw.py`、`test_quickstart_service.py`、`test_registry.py`。
- Scratch 测试：`src/lnl_toolbox/scratch/tests/test_catalog_recipes.py`、`test_web_templates.py`。
- Web 测试：`web/test_command_console.py`、`test_quick_start_assets.py`、`test_quick_start_integration.py`、`test_quick_start_routes.py`。
- 文档：本文、`docs/development-guide.md` 的 CI 断言维护规范、`papers/dividemix/reproduction.md` 的预算说明。

这些文件中部分本来就有主会话修改；整份 git diff 不等于本次新增改动。长期测试规则已写入开发指南。
