# A股 AI Trader — 代码架构

> 平台层代码架构速览。策略研究方法论与运行记录为私有研究产物, 不在本仓库公开范围。

```
┌─────────────────────────────────────────────────────────────────┐
│ 展示/服务层                                                      │
│   web/ (Streamlit Dashboard)  ·  api/ (FastAPI REST)             │
├─────────────────────────────────────────────────────────────────┤
│ 智能体层                                                         │
│   agent/ (LangGraph 多子代理工作流 + chat_agent + 学习闭环)        │
│   models/ (模型路由, 单模型 DeepSeek V4-Flash + 成本监控)          │
├─────────────────────────────────────────────────────────────────┤
│ 分析/回测层                                                      │
│   analysis/ (技术指标/市场体制/风控/胜率)                          │
│   backtest/ (事件驱动引擎: T+1/涨跌停/整手/费率 + 过拟合检测)       │
├─────────────────────────────────────────────────────────────────┤
│ 交易模拟层                                                       │
│   simulation/ (paper_trader T+1 · daily_runner · portfolio)      │
├─────────────────────────────────────────────────────────────────┤
│ 数据层                                                          │
│   data/ (DataRouter 多源降级链 + 缓存)                            │
│   knowledge/ (三层知识库: YAML规则 / 参考md / ChromaDB)           │
└─────────────────────────────────────────────────────────────────┘
```

## web/ — Streamlit Dashboard (v6.0 模块化)

```
web/
├── dashboard.py      入口: set_page_config → 主题 → sidebar → tab 派发 (~180行)
├── theme.py          GitHub-dark CSS 主题 (render_dataframe 自定义表格, 避开
│                     Streamlit GlideDataEditor 与暗色主题的冲突)
├── api_client.py     Dashboard↔API 共享 HTTP 层 (统一 X-API-Key / timeout / JSON)
├── data.py           缓存加载器 (@st.cache_data) + 全市场抓取 + 组件单例
├── realtime.py       ⚡实时行情 tab (@st.fragment 10s 自动刷新)
├── tabs/             9 个 tab 模块, 各暴露 render():
│   overview / market_table / opportunity / technical / backtest_tab /
│   signals / portfolio_tab / risk / knowledge
└── chat.py, viz_components.py, progress_tracker.py, render.py
```

要点: tab 懒加载派发 (`if/elif` + 模块内 import); `@st.fragment` 的
sidebar 持仓速览与实时 fragment 均在模块顶层注册 (Streamlit 语义要求);
缓存装饰器随函数整段迁移, cache identity 不变。

## api/ — FastAPI REST (v6.0 模块化)

```
api/
├── server.py         app + CORS + 认证/限频中间件 + /health + 路由注册 (~240行)
├── schemas.py        全部 Pydantic 请求/响应模型 (server.py re-export 保持兼容)
├── deps.py           组件单例工厂 (get_router/get_knowledge/get_analyzer/...)
├── routers/          领域路由 (path 与单文件时代逐字一致):
│   market.py         /stock/{symbol} /market/regime /realtime/market /cost/summary
│   analysis.py       /analyze POST+GET /backtest /strategies /system/benchmark
│   chat.py           /chat + /chat/history GET/DELETE
│   portfolio.py      /portfolio/summary /portfolio/mtm /trades/stats /risk/status
│   decisions.py      /decisions 4 条 (决策日志)
│   competition.py    /competition/* 5 条 (架构/知识库/提示词/Benchmark/工作流可视化)
│   bot.py            /bot/wecom + /bot/push/daily + /bot/push/alert
└── mcp_server.py     MCP 服务 (独立入口)
```

安全设计: X-API-Key 认证 (多 key 轮换 + 常量时间比较, fail-closed — 未配置
API_KEY 时除白名单外全部 403), 60s 滑动窗口限频 (429), CORS 默认关闭
(显式配置 `CORS_ORIGINS` 才放开), 敏感端点 (持仓/推送) 不在白名单。

## agent/ — LangGraph 多子代理

`Data Prep → Technical → Market Scanner → Strategy Match → Backtest →
Bull/Bear Debate → Critic Audit → Judge → Synthesis → Decision Log`。
子代理继承 `BaseAgent` 返回 `AgentResult`; code-as-reasoning 模式
(LLM 定计划 → Python 算数 → LLM 叙述, `NumericSafetyChecker` 防数字幻觉)。
`agent/learning/` 为自主学习闭环 (研究→查重→真实回测测试→向量库→滚动重测)。

## data/ — 多源降级数据路由

AKShare → 腾讯 (实时报价, 免代理) → 东财 → Baostock (历史 K 线, 免代理)
自动降级 + 冷却; 缓存: 实时 3s / 日线 1h / 股票列表 24h。
`simulation_data/full_market_cache.json` 为全市场快照 (空行情护栏 MIN_QUOTES)。

## 回测与模拟

- `backtest/engine.py` + `broker.py`: 事件驱动, 真实 A 股约束 (T+1、
  10%/20%/30% 涨跌停、停牌、整手、佣金/印花税/过户费); `overfitting.py`
  6 种过拟合检测 (time-split / PBO / Deflated SR / walk-forward 等)。
- `simulation/`: paper_trader (T+1 结算 + validator 硬约束 + 拒单日志),
  `daily_runner` 每日三阶段 (分析→交易→快照)。
