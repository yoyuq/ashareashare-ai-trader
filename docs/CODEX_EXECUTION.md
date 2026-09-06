# CODEX 工程任务执行记录（2026-09-06）

实际任务书是 `docs/CODEX_TASKS.md`；用户给出的 `docs/CODEX/_TASKS.md` 不存在。
工作分支：`codex/complete-engineering-tasks-20260906`。未 commit/push，保留维护者已有改动。

**交付状态：任务 1、2 的工程实现、真实探测及可选纯函数测试已交付；离线全量回归通过。全库 Ruff 仍有 752 条既有问题，因此不能宣称任务书“所有验收全绿”已完成。**

## 需求与落实位置

| 任务 / 要求 | 实现与证据 |
|---|---|
| 数据源按行情、基本面、资金流、事件、舆情、宏观分类 | [实测矩阵](data_source_inventory.md)；`scripts/probe_source_inventory.py` 的 `SOURCES` 和 `exported_catalog()` |
| 实测历史起点、至少 10 只股票覆盖、PIT 字段、失败摘要 | [机器证据](data_source_inventory_evidence.json)；`fetch_dc()`、`summarize_frame()`、`probe_sample()`；宏观不具有股票维度，单独披露实际返回期数 |
| 代理失败后直连重试一次、可重跑 | `transport()`、`probe_sample()`、`run_source()`；无数据替代分支，未完成样本单独标记 |
| 同底层去重 | 矩阵“同底层重复与保留依据”；证据中的 `comparisons` 保留代表封装的实测比较 |
| 整数手、单笔/单日限额、报价偏离、停牌/ST/涨跌停、费用及资金 | `scripts/sandbox_order_sheet.py` 的 `Limits`、`fees()`、`validate_orders()`；卖出使用真实可卖数量，买入不预支未成交卖单收入 |
| 默认预览，显式确认才写文件 | `parser()`、`main()` 的 `--dry-run` / `--confirm`；[输入契约及使用步骤](order_sheet_usage.md) |
| 实时报价的真实获取路径 | `scripts/capture_order_quotes.py` 获取腾讯原始报价、行情时间和上下限价，保留响应及哈希，合并真实交易状态快照；缺状态或陈旧报价拒绝 |
| 字节幂等、生成日期与输入快照哈希 | 输入原始字节与参数 SHA-256、`render_sheet()`、`_write_once()`；真实调用时间只进入审计 |
| 每次生成审计、重复确认及冲突防护 | `_append_audit()`、`_audit_lock()`、`_check_daily_plan()`；先写预留，后原子创建；同账本同日不同快照拒绝确认 |
| 人工操作单为终点 | 不接券商、不下单；文件说明人工复核步骤 |
| 每条防呆规则、幂等性离线单测 | `tests/unit/test_sandbox_order_sheet.py`；测试固定输入仅限单元测试，不是运行数据 fallback |
| 可选任务 3：纯函数单测 | `tests/unit/test_pure_function_edges.py`：24 个清洗、市场结构、拥挤度边界用例 |
| 当前 FastAPI 下原有路由回归测试 | `tests/unit/test_v5_api_p16.py` 改为验证 OpenAPI 与真实 ASGI 请求，仍检查无尾斜杠路径和重定向行为；未修改服务端 auth 或模型导出 |

## 验收环境与基线

使用项目 `.venv\Scripts\python.exe`（Python 3.12.10），设置 `PYTHON_DOTENV_DISABLED=1`，避免测试收集阶段自动加载配置。Windows 默认临时目录存在 `pytest-current` 权限错误，因此验收使用项目 `.pytest_cache` 下的独立新临时目录；没有减少测试范围。

修改前全量回归有一个失败：FastAPI 0.139.2 将嵌套路由保留为包装器，旧测试直接遍历顶层 `.path` 无法看到 `/api/v1/decisions`。已修测试兼容性，并用实际 HTTP 行为增强验证。

修改前 Ruff 有 **755 条问题，涉及 161 个文件**。其中包括冻结研究脚本的 `F821`、Python 3.11 目标与 3.12 f-string 语法不一致、未使用导入、单行多语句等。本次没有放宽 Ruff 规则、修改排除列表或批量改写冻结脚本。

## 最终验证结果

| 检查 | 结果 |
|---|---|
| `pytest tests/ -m "not network"`（项目虚拟环境，独立临时目录） | **732 passed, 1 skipped, 8 deselected**；23.95 秒 |
| 本次 7 个 Python 文件的 Ruff E/F/W 检查 | **通过** |
| `ruff check . --select=E,F,W --ignore=E501` | **未通过：752 条**；相比修改前减少 3 条，本次改动未增加 lint 问题 |
| `git diff --check` | 通过 |
| 矩阵与独立 JSON 一致性 | 通过；JSON 再生 Markdown 内容一致，逐项核对样本、日期、响应哈希及失败后直连重试证据 |
| 真实数据探测 | 25 类：21 类股票接口各 10 只 + 4 个宏观序列；214 个主矩阵样本中 165 个非空响应、49 个如实失败、0 个未完成；430 个公开符号索引另标是否实测 |
| 操作单独立审查 | 已修复并复核交易时段、卖出费用现金缺口、慢计算后重新校验时钟、异常审计互斥及写出后审计失败提示 |

跳过的 1 项是 Windows 无权限创建真实符号链接；同一安全边界另有 mock 路径解析测试通过。8 项联网测试按要求排除。操作单专用测试为 83 passed、1 skipped；探测脚本 16 项与纯函数 24 项通过。

全量回归复现命令（在项目根目录运行，临时目录仅用于测试）：

```powershell
$env:PYTHON_DOTENV_DISABLED='1'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe -m pytest tests/ -m "not network" -q --basetemp=.pytest_cache/codex-final-verified-20260906
.venv\Scripts\python.exe -m ruff check . --select=E,F,W --ignore=E501
```

最后一次手工演示未传 `--dry-run`，核对默认模式：

```powershell
.venv\Scripts\python.exe scripts/sandbox_order_sheet.py --audit-log .pytest_cache/codex-final-demo/audit.jsonl --output .pytest_cache/codex-final-demo/order.md
```

实际输出：`拒绝生成操作单: 缺少真实输入 --events, --account, --quotes；零模拟，拒绝生成`。退出码 **2**；`order.md` 不存在；审计记录 `mode=dry-run`、`status=rejected`、`validation.passed=false`，含时间和已知输入/参数哈希。没有为了演示填充账户或行情。

## 尚未扩大的范围

- 公开接口枚举与实际联网验证分开：未测的导出符号不能据此认为可用。历史起点是本轮固定样本的可返回最早值，不是全市场最早值；公告字段不证明历史版本完整。
- 同底层封装按安装包源码映射，并对代表组实际比较；未对每个封装作长期稳定性压测。
- 未抽取 `scripts/_common.py`：现有 parquet 片段的索引、去重键、类型压缩和输出契约不同，缺少适合本次机械搬迁的公共边界。
- 未提供真实账户与当日完整行情/交易状态快照；不编造输入以制作可执行操作单。成功预览/确认路径由离线单测覆盖，手工演示核对缺输入时的默认 dry-run 拒绝行为。
- 全库 Ruff 门槛仍未完成：既有问题跨 161 个文件，部分位于必须保持行为的研究脚本。已提供可复核的基线数量和本次范围通过结果，没有以屏蔽规则冒充全绿。
