# scripts/flylark_*.py — 飞书平台代码节点脚本

本目录下所有 `flylark_*.py` 均为 **飞书（Feishu/Lark）多维表格「代码节点」脚本**，
设计为直接粘贴到飞书代码节点执行，**与项目核心无依赖**（仅用标准库 `json`/`math`/`re`/`collections`，
不 `import` 任何 `agent`/`models`/`backtest`/`simulation`/`data` 模块）。

它们既不是测试、也不是命令行工具，**不会被 `pytest`、`daily_runner` 或任何核心链路引用**；
修改核心代码时无需（也不应）改动这些文件。相关平台能力边界见记忆
`flylark-platform-capabilities`（代码节点 = 完整 Linux 沙箱 + 文件跨运行持久 + 外网接腾讯行情）。

## 分类

### 一次性探针（诊断飞书平台能力，跑一次摸清边界即可，无需重复运行）

| 文件 | 用途 |
|------|------|
| `flylark_platform_probe.py` | 平台代码节点能力自诊断：一次性摸清沙箱边界（标准库/pandas/requests/外网/出参机制） |
| `flylark_persist_probe.py` | 沙箱文件持久性测试：决定进化记忆用「文件」还是「多维表格」 |
| `flylark_crossnode_probe.py` | 跨节点文件共享测试：决定进化闭环能否拆多个代码节点 |

### 进化闭环实装（飞书「自我进化」看板的核心节点，可复用）

| 文件 | 用途 |
|------|------|
| `flylark_evolution_nodes.py` | 进化闭环两个核心代码节点（可复制，拆节点方案） |
| `flylark_growth_dashboard.py` | 进化成长看板：把「自我进化」做成可亲见的成长轨迹 |
| `flylark_stress_test.py` | 历史极端压力测试：真实历史 K 线回测议会信号在极端行情下的存活 |
| `flylark_numeric_guard.py` | Code-as-Reasoning 数值护栏：防 LLM 编造数字 |
| `flylark_shadow_weight.py` | 影子权重 + 反事实验证：防进化过拟合 |
| `flylark_macro_context.py` | 宏观数据接入层：平台内置宏观工具 → 大师可消化的宏观上下文 |
| `flylark_independence_entropy.py` | 观点汇总增强：加入【有效独立观点熵】，替代节点 7 |

## 维护约定

- 新增飞书节点脚本时，沿用 `flylark_` 前缀并在本文档登记分类。
- 若某脚本已被飞书侧固化、不再需要本地副本，直接删除本文件并同步本 README。
