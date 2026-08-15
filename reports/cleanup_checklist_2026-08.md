# 清理与修复清单（严格执行版）

> 生成：2026-08-13 ｜ 依据：`reports/project_health_review_2026-08.md`
> 执行原则：每项「改 → 跑测试 → 确认」，不偷工减料。勾选状态随进度更新。

## P0 —— 高危（先做，必须全部完成）

- [x] **P0-1** 修 `market_diagnostic` 节点必失效：`route()` 现已支持 `temperature`/`max_tokens` 透传（`models/router.py`），测试 22 passed。顺带落实 P1-6 的 timeout/max_retries、月预算软提醒、预算状态落盘。
- [x] **P0-2** 让 `strategy_backtest.py` 走事件驱动 `EventDrivenBacktestEngine`/`AShareBroker`（或明确标注简化口径）——已核实事件引擎并非死代码（strategy_executor/api/dashboard/benchmark 均在用），故采用方案 (b)「明确标注简化口径」：`strategy_backtest` 返回 `methodology` 字段标注信号级口径与幸存者偏差，避免误读。
- [x] **P0-3** 修正回测聚合统计：Sharpe 改日频（去掉 `√n_trades`），`total_return`/`max_dd` 改用个股中位数（组合口径），并重标 `_grade` 的 sharpe 阈值。
- [x] **P0-4** 回撤断路器真实生效：触发 -8%/-15% 时对存量持仓执行减仓/清仓，而非只 `halt_buys`。
- [x] **P0-5** DecisionValidator 补传实盘上下文：`daily_runner` 传 `market_data` + `portfolio`，让 8 项硬约束全部生效。
- [x] **P0-6** 数据层静默失败分离：空数据/`{"error":...}` 占位表与「确实无数据」区分，失败计入降级，不缓存占位表。
- [x] **P0-7** 熔断按 `(source, symbol)` 记失败 + 降级窗口过期重置计数。
- [x] **P0-8** API 安全：收敛 CORS（默认关闭跨域）、密钥轮换（多 key 逗号分隔）、IP 滑动窗口限频、关 `API_ALLOW_INSECURE_NO_AUTH`、敏感白名单已移出（仅留公开行情/文档）。
- [x] **P0-9** git 治理：拆分支 `chore/p0-safety-hardening`、按 6 主题提交，隔离核心改动与实验探针。

## P1 —— 重要（接着做）

- [x] **P1-1** 缓存 TTL 分层落地（日K 1h / 分钟 5min），实时行情 3s、股票列表 5min 补缓存（`data/router.py`）。
- [x] **P1-2** 成交量口径统一：Tencent 日 K volume ×100，全源断言单位一致（`data/providers/base.py` `standardize()`）。
- [x] **P1-3** 缓存加 LRU 上限/淘汰 + 负缓存（30s）+ 并发合并单飞（`_route_with_fallback`/`_fetch`/`_cache_set`）。
- [x] **P1-4** 数据校验加强：量/额负值清洗、日期单调排序去重、零价→NaN→ffill（`data/processors/cleaning.py`）。
- [x] **P1-5** Baostock `login()` 返回码校验（`data/providers/baostock_provider.py`）。
- [x] **P1-6** 成本模型对齐 DeepSeek 真实峰谷定价，价格集中 `model_config.yaml` 单源；统一所有 LLM 调用走 router（含自进化模块）；`monthly_budget` 生效 + 预算落盘。
- [x] **P1-7** LangGraph 节点失败显式告警/分级降级；统一 model_trace（tokens/latency/cost）；校验 checkpoint 表结构。
- [x] **P1-8** 消除双注册表漂移：一致性测试或删硬编码；更新 YAML workflow 拓扑。
- [x] **P1-9** NumericSafetyChecker 校验失败改写/重生成而非仅告警；按量级绝对容差；多参数豁免。
- [x] **P1-10** 简化回测涨跌停板块感知（复用 `broker._limit_pct_for`）。
- [x] **P1-11** 6 种过拟合检测真实传参/统一入口（`overfitting.py` vs `overfitting_check.py`）。
- [x] **P1-12** 费率单一来源（`broker.py` vs `optimized_strategies.py:67`）。
- [x] **P1-13** 组合反事实提高阈值 + 同期基准对照 + 限制回流频次。
- [x] **P1-14** Dashboard 拆分模块 + HTML 转义（防 XSS）+ 股票搜索改索引。
- [x] **P1-15** 实盘/缓存一致性：统一写缓存 + 显著展示数据时间戳与陈旧标记。
- [x] **P1-16** API 端点统一尾斜杠 + `response_model` + Pydantic 校验器。
- [x] **P1-17** 知识库三层联动（YAML→向量库重建流程）+ 检索质量离线评测 + 规则 YAML schema。

## P2 —— 优化（收尾）

- [x] **P2-1** 兜底价过期处理 + 刷新失败显式报错。
- [x] **P2-2** `full_market_cache` 用 A 股交易日 + 补北交所 bj 前缀。
- [x] **P2-3** BaseAgent `success` 语义修正 + `model_used` 填充 + 并发竞态修复。
- [x] **P2-4** 清理提示词孤儿文件 + `ChatAgent.SYSTEM_PROMPT` 单源化。
- [x] **P2-5** 清理死代码（`TASK_ROUTING`/`DEEPSEEK_PRO_MODEL`/单值 `ModelTier`/`force_tier`）。
- [x] **P2-6** CI：mypy 逐步放开、network 标记统一到 conftest。
- [x] **P2-7** 仓库卫生：清理 log 产物、`flylark_*.py` 归类或标注。
