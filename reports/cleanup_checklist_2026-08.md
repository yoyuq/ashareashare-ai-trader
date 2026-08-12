# 清理与修复清单（严格执行版）

> 生成：2026-08-13 ｜ 依据：`reports/project_health_review_2026-08.md`
> 执行原则：每项「改 → 跑测试 → 确认」，不偷工减料。勾选状态随进度更新。

## P0 —— 高危（先做，必须全部完成）

- [x] **P0-1** 修 `market_diagnostic` 节点必失效：`route()` 现已支持 `temperature`/`max_tokens` 透传（`models/router.py`），测试 22 passed。顺带落实 P1-6 的 timeout/max_retries、月预算软提醒、预算状态落盘。
- [ ] **P0-2** 让 `strategy_backtest.py` 走事件驱动 `EventDrivenBacktestEngine`/`AShareBroker`（或明确标注简化口径）——选择接入事件引擎。
- [ ] **P0-3** 修正回测聚合统计：Sharpe 改等权组合净值曲线 + 日频口径（去掉 `√n_trades`），`total_return` 改组合收益，`max_dd` 改净值回撤。
- [x] **P0-4** 回撤断路器真实生效：触发 -8%/-15% 时对存量持仓执行减仓/清仓，而非只 `halt_buys`。
- [x] **P0-5** DecisionValidator 补传实盘上下文：`daily_runner` 传 `market_data` + `portfolio`，让 8 项硬约束全部生效。
- [x] **P0-6** 数据层静默失败分离：空数据/`{"error":...}` 占位表与「确实无数据」区分，失败计入降级，不缓存占位表。
- [x] **P0-7** 熔断按 `(source, symbol)` 记失败 + 降级窗口过期重置计数。
- [ ] **P0-8** API 安全：收敛 CORS、密钥轮换机制、IP/速率限制、关 `API_ALLOW_INSECURE_NO_AUTH`、移出敏感白名单。
- [ ] **P0-9** git 治理：拆分支、按主题提交，隔离核心改动与实验探针。

## P1 —— 重要（接着做）

- [ ] **P1-1** 缓存 TTL 分层落地（删死代码 Redis 层或让分层真正生效），实时行情/股票列表补缓存。
- [ ] **P1-2** 成交量口径统一：Tencent 日 K volume ×100，全源断言单位一致。
- [ ] **P1-3** 缓存加 LRU 上限/淘汰 + 负缓存 + 并发合并（防击穿）。
- [ ] **P1-4** 数据校验加强：价格>0、量≥0、日期单调、OHLC 关系。
- [ ] **P1-5** Baostock `login()` 返回码校验。
- [ ] **P1-6** 成本模型对齐 DeepSeek 真实峰谷定价，价格集中 `model_config.yaml` 单源；统一所有 LLM 调用走 router（含自进化模块）；`monthly_budget` 生效 + 预算落盘。
- [ ] **P1-7** LangGraph 节点失败显式告警/分级降级；统一 model_trace（tokens/latency/cost）；校验 checkpoint 表结构。
- [ ] **P1-8** 消除双注册表漂移：一致性测试或删硬编码；更新 YAML workflow 拓扑。
- [ ] **P1-9** NumericSafetyChecker 校验失败改写/重生成而非仅告警；按量级绝对容差；多参数豁免。
- [ ] **P1-10** 简化回测涨跌停板块感知（复用 `broker._limit_pct_for`）。
- [ ] **P1-11** 6 种过拟合检测真实传参/统一入口（`overfitting.py` vs `overfitting_check.py`）。
- [ ] **P1-12** 费率单一来源（`broker.py` vs `optimized_strategies.py:67`）。
- [ ] **P1-13** 组合反事实提高阈值 + 同期基准对照 + 限制回流频次。
- [ ] **P1-14** Dashboard 拆分模块 + HTML 转义（防 XSS）+ 股票搜索改索引。
- [ ] **P1-15** 实盘/缓存一致性：统一写缓存 + 显著展示数据时间戳与陈旧标记。
- [ ] **P1-16** API 端点统一尾斜杠 + `response_model` + Pydantic 校验器。
- [ ] **P1-17** 知识库三层联动（YAML→向量库重建流程）+ 检索质量离线评测 + 规则 YAML schema。

## P2 —— 优化（收尾）

- [ ] **P2-1** 兜底价过期处理 + 刷新失败显式报错。
- [ ] **P2-2** `full_market_cache` 用 A 股交易日 + 补北交所 bj 前缀。
- [ ] **P2-3** BaseAgent `success` 语义修正 + `model_used` 填充 + 并发竞态修复。
- [ ] **P2-4** 清理提示词孤儿文件 + `ChatAgent.SYSTEM_PROMPT` 单源化。
- [ ] **P2-5** 清理死代码（`TASK_ROUTING`/`DEEPSEEK_PRO_MODEL`/单值 `ModelTier`/`force_tier`）。
- [ ] **P2-6** CI：mypy 逐步放开、network 标记统一到 conftest。
- [ ] **P2-7** 仓库卫生：清理 log 产物、`flylark_*.py` 归类或标注。
