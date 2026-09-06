# AGENTS.md — Agent 协作说明 (Codex / 其他 AI 编码代理通用)

> 本文件是给 AI 编码代理的工作须知。**完整项目细节以 CLAUDE.md 为准，先读它。**
> 快速命令速查也在 CLAUDE.md「Essential Commands」一节。

## 必须遵守的红线 (违反 = 工作无效)

1. **不读不写 `reports/`** — 策略研究产物（预注册判据/回测结果/操作单）是私有研究内容，与代码优化无关，不要打开、引用或复述其中内容。
2. **不读不传 `.env` / `.mcp.json`** — 含 API 密钥，永远不要把内容写进任何输出或提交。
3. **零模拟纪律** — 数据/回测/行情相关代码一律真实数据、缺数据报错不兜底；**禁止**给数据获取代码加"失败时用模拟/随机/占位数据"的 fallback。
4. **不改动以下行为**:
   - `simulation/daily_runner.py` 的每日三阶段行为与交易语义 (Windows 任务计划在实跑)
   - `api/server.py` 顶层的 auth 全局变量与中间件 (`tests/unit/test_v3_regressions.py` monkeypatch 它们)
   - `api/server.py` 对 `api/schemas.py` 模型的 re-export (`test_v5_api_p16.py` 依赖)
   - `web/dashboard.py` 的入口/派发结构与 `web/theme.py` CSS
5. **研究脚本** (`scripts/run_*` / `forward_*` / `build_*`) 的逻辑语义不要动——它们对应已冻结的实验判据；重构需保持行为逐字一致。

## 验收硬门槛 (每次修改后必须全绿)

```bash
pytest tests/ -m "not network"        # 离线全量回归
ruff check . --select=E,F,W --ignore=E501
```

## 项目速览

- A股 AI 量化分析 + 策略研究平台: LangGraph 多智能体分析 → 事件驱动回测 → 纸面交易 → 前瞻 OOS 验证 → 自主学习闭环
- 架构分层、模块地图、数据源降级链、主题约束见 CLAUDE.md 与 docs/ARCHITECTURE.md
- 股票代码格式 `{market}.{code}` (如 `sh.600519`)
- Streamlit 主题约束: 表格必须用 `render_dataframe()` 而非 `st.dataframe` (GlideDataEditor 与暗色主题冲突)

## 提交规范

- 直接提交到分支即可; commit message 用中文一行式 (项目惯例, 见 `git log`)
- 不做 git history rewrite
- `reports/`、`simulation_data/`、`replay_data/`、`.env`、`.mcp.json` 已在 `.gitignore`, 不要改 gitignore 让它们入库
