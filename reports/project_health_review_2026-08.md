# 项目全盘体检报告

> 日期：2026-08-13 ｜ 方法：4 个并行代码体检 agent（数据层 / Agent 编排 / 回测风控 / Web·知识库·进化·测试）+ 联网检索（DeepSeek 定价、A股数据接口、LangGraph、Streamlit）
> 关键发现已由人工二次核实（见 P0 标注 ✅）。

---

## 0. 总体评价

项目整体架构清晰、诚实度高：数据多源降级、Code-as-Reasoning 数值安全、进化系统 A/B 验证（预注册判据 + 配对差分 + 双模型复验）都做得比同类项目扎实。**但存在几处"精心实现却未真正接入"的断层**——事件驱动回测引擎没进策略排名、LLM 风控监督节点必然失效、回撤断路器不卖存量——这些是"看起来有、实际没生效"的高危问题，优先级最高。

按严重程度分三档：

- **P0（高危）**：影响实盘正确性/资金安全/审计能力，建议立即处理。
- **P1（重要）**：影响可靠性、统计正确性、可维护性。
- **P2（优化）**：代码卫生、健壮性、长期演进。

---

## 1. 外部情报（联网，直接可动作）

### 1.1 DeepSeek 涨价在即，成本模型需立即复核 ⚠️
- 当前阵容即 `deepseek-v4-flash`（快/便宜）与 `deepseek-v4-pro`（强），均 1M 上下文、384K 输出。
- **峰谷定价已于 2026-07 中旬生效**：工作日 9:00–12:00、14:00–18:00（北京时间）价格翻倍。Flash 峰值：输入命中缓存 ¥0.04/M、miss ¥2/M、输出 ¥4/M；谷值 ¥0.02/¥1/¥2。
- **2026-08-06 又预告一轮"较大涨幅"全面涨价**，幅度未公布。
- **缓存命中价 ≈ miss 价的 1/50**——对 agent 复用稳定 system prompt 的场景收益巨大，值得确认项目是否在吃这个折扣。
- 旧别名 `deepseek-chat`/`deepseek-reasoner` 已于 **2026-07-24 停用**。

**对照项目现状**：`models/router.py:49-51` 写死 input ¥1/M、output ¥2/M、缓存 ¥0.02/M，峰谷窗口 9-12/14-18——**谷值是对的，但没体现"峰值翻倍"和"即将到来的再涨价"**；且预算追踪是进程内存态、重启清零。建议把价格集中到 `config/model_config.yaml` 单源并加注释。

### 1.2 A股数据接口生态
- AKShare、Baostock 2026 年**均未停更**，仍是最主流免费组合；但部分 AKShare 接口已废弃：`stock_individual_info_em` 不可达、`stock_profit_sheet_ths` 已废弃（改用 `stock_financial_benefit_ths`）。
- **北向资金**日频/盘中买卖明细自 2024-08-19 起官方停发——若任何模块仍依赖北向实时数据需确认。
- 多来源交叉验证 + 异常降级是 2026 共识，项目方向正确。

### 1.3 LangGraph 生产最佳实践
- 生产环境用**持久化 checkpointer**（SQLite/Postgres）而非 InMemorySaver，每次运行用唯一 `thread_id`。
- 状态里持久化**结构化字段**（attempts/risk_level/decision_record），而非依赖模型 prose。
- 显式**重试/升级/fail-closed** 策略；对**控制流本身做测试**（而非只测最终答案）。

### 1.4 Streamlit 1.60（2026-07-21）
- `st.dataframe` 修复了 pandas 3 `ArrowStringArray` 崩溃、排序保留选中行、新增 `client.disableDataExport` 关闭 CSV 导出。项目里"自定义 render_dataframe 规避暗色主题冲突"的做法在官方 release notes 里没有对应条目，1.60 有表格修复，**值得评估是否已可回归原生 `st.dataframe`**。

---

## 2. P0 —— 高危（建议立即处理）

### P0-1 ✅ LLM 风控监督节点必然失效
`agent/orchestration/workflow.py:485` 向 `route()` 传 `temperature=0.3`，但 `models/router.py:163-169` 的 `route()` 签名**没有 `temperature` 参数** → 每次调用必抛 `TypeError`，被 `workflow.py:606-609` 的 except 静默吞掉，永远走 `default_diag` 保守值。**LLM 风控监督（risk_level / 仓位系数）实际从未生效。**
> 修复：删掉 `temperature=` 或给 `route()` 加 `**kwargs`/`temperature`；并对节点失败加告警打点而非静默。

### P0-2 ✅ 事件驱动回测引擎没接入策略评级链路
`backtest/broker.py`+`engine.py` 完整实现了 T+1/涨跌停/停牌/手数/费率/滑点，但 `backtest/strategy_backtest.py:25-27` 实际调用的是 `recommender.py`/`optimized_strategies.py`/`strategies_v3.py` 的**简化回测器**；`AShareBroker` 只被 tests/dashboard/api 引用，从未进入策略排名链路。
> 修复：让 `strategy_backtest.py` 走 `EventDrivenBacktestEngine`，或明确标注报告来自简化口径。

### P0-3 ✅ 聚合统计错误（Sharpe/收益/回撤）
`backtest/strategy_backtest.py:202` `sharpe = mean/std * np.sqrt(n)`，其中 `n` 是**跨股票拼接的交易笔数**（871 笔 → √871≈29.5 倍放大，导致 dual_ma 显示 Sharpe 3.95）；`:201` `total_return_pct` 是各票盈亏简单求和（2168% 无经济意义）；`:205` max_dd 把逐笔收益当净值曲线（-256% 无意义）。
> 修复：改为等权组合净值曲线 + 日频 Sharpe（`_group_sharpe` 里已有正确的 `np.sqrt(252)` 写法可复用），并标注口径。

### P0-4 回撤断路器名不副实
`analysis/risk_controls.py:94-98` 标称 -8% 减仓 50% / -15% 清仓，但 `simulation/daily_runner.py:1692-1717` 仅置 `halt_buys` 并把 `risk_mult` 用于**新买单**，从不强制卖出存量。
> 修复：断路器触发时对现有持仓执行真实减仓动作。

### P0-5 DecisionValidator 8 项硬约束实盘只跑 4 项
`simulation/daily_runner.py:1830-1831` 传入 `market_data={}` 且无 `portfolio`，导致 `validator.py:165-231` 的 limit_unbuyable / lot_too_small / concentration / t1_pending 全部跳过，只剩参数校验。
> 修复：补传实时行情与持仓快照。

### P0-6 数据层静默失败（空数据当成功）
贯穿多处的危险模式——被墙/失效与"确实无数据"无法区分，可能带病下单：
- `EastMoneyProvider.get_stock_list` 捕获异常直接 `return pd.DataFrame()`（`eastmoney_provider.py:274-275`），不计失败。
- `CacheLayer.get_or_fetch` 只判 `not df.empty` 就缓存（`data/cache.py:255-258`），AKShare 的 `{"error":[...]}` 占位表被当成功缓存。
- `TencentFinanceProvider._validate_response` 覆写为仅 `not df.empty`（`tencent_provider.py:218-219`）。
- `DataRouter.get_realtime_quote` 全源失败返回 `{"error":[...]}` 的"正常"DataFrame（`data/router.py:143`）。

### P0-7 回撤断路器之外的"熔断误伤"
`data/router.py:219-228` 失败计数**按源不按标的**，某只退市/新股/北交所标的连续失败 3 次会把整个健康源降级 5 分钟；`_on_success` 才清零、降级窗口过期也不重置 → 脆弱源实际永久降级。
> 修复：按 (source, symbol) 记失败，降级过期时重置计数。

### P0-8 API 认证 + CORS 暴露面
`api/server.py:46-53` CORS 默认 `*`，X-API-Key 静态共享、无轮换、无客户端隔离；白名单含公开行情 `/api/v1/realtime/market`（`:63`），CORS 通配下任意站点可读，并可携带泄露的 key 调用其余端点。另有 `API_ALLOW_INSECURE_NO_AUTH=true` 且 `.env` 明文密钥。
> 修复：生产强制收敛 `CORS_ORIGINS`、密钥轮换、加 IP/速率限制、关 `API_ALLOW_INSECURE_NO_AUTH`。

### P0-9 未提交改动混入核心交易路径
`git status` 显示 `backtest/broker.py`、`backtest/engine.py`、`simulation/daily_runner.py` 等核心成交/引擎逻辑已改未提交，同时新增 `adversarial.py`、一批 `flylark_*.py` 与 `reports/evolution_*.md` 混在 main 分支，无法审计/回滚。
> 修复：拆分支、按主题提交（核心修复 vs 实验脚本）。

---

## 3. P1 —— 重要

**数据层**
- 缓存 TTL 分层是"假"的：`data/cache.py` 的 Redis `CacheLayer` 全项目仅导出/测试引用，是死代码；真实缓存是 `DataRouter` 内扁平内存缓存，**所有 K 线统一 5 分钟 TTL**（`router.py:49-50`），实时行情/股票列表无缓存。要么删 Redis 层，要么把 TTL 分层真正落地。
- 成交量口径跨源不一致（100 倍）：`standardize()` 只对 AKShare/EastMoney volume ×100，Tencent 日 K 未换算（腾讯返回"手"），双源切换会引入 100 倍偏差（`tencent_provider.py:161`）。
- 缓存无上限/无淘汰/无负缓存（`router.py:49,182`），全市场 5884 只 × 多复权组合持续累积 DataFrame。
- 数据校验过弱：仅查列存在 + ≥5 行，不校验价格非负/量非负/日期单调。
- Baostock `bs.login()` 返回值未校验（`baostock_provider.py:50-52`）。

**Agent 编排**
- 预算/成本模型与实际 DeepSeek 定价脱节；自进化模块 `agent/evolution/prompt_optimizer.py:273` 直接 `client.chat.completions.create` **绕过 ModelRouter**，不记账/不扣预算/不峰谷；`monthly_budget` 从未 enforce；预算进程内存态重启清零；`ModelConfig.request_timeout=90` 未传给 AsyncOpenAI。
- LangGraph 节点 try/except 吞异常只写 `state["errors"]`，不中断、不告警；checkpoint 用 SqliteSaver 但表名可能不符 schema。
- 双注册表漂移：`agent/sub_agents/__init__.py` 硬编码 registry 与 `config/agents.yaml` 需手工同步，YAML 的 workflow 拓扑已过时。
- `NumericSafetyChecker` 仅 advisory（`workflow.py:712-715` 只 `logger.warning`），校验失败不阻断/改写；final_report 不跑该 checker。

**回测/风控**
- 简化回测器涨跌停非板块感知：硬编码 ±10%，创业板/科创 20%、北交所 30% 未区分。
- 6 种过拟合检测未全跑：Walk-Forward 恒 0、参数敏感性实为跨策略变异系数、`split_time_series` 是未调用的静态方法；且 `overfitting.py` 与 `overfitting_check.py` 两套口径不一。
- 幸存者偏差已诚实标注（`strategy_backtest.py:232-234` `survivorship_bias: HIGH`）但未消除。
- 组合反事实近同义反复：`portfolio_counterfactual.py:145` "移除最差票"在下跌日几乎必然 verified（阈值仅 0.05%），`drag_experiences` 又把反复标记票降权写入 memory——属后视偏差回流。
- 费率两套模型：`broker.py` 万3/0.05%卖/0.00001 与 `optimized_strategies.py:67` 固定 `COST_PER_RT=0.0031` 并存。

**Web/API/知识库**
- Dashboard 单文件 2052 行；自定义表格字符串值未 `html.escape`（`dashboard.py:443` `str(val)`），有 XSS 风险；5100+ 股票 `st.selectbox` 下拉 + `str.contains` 全量扫每次交互。
- 实盘与缓存无一致性约定：`get_full_market` ttl=300 只当 fallback、从不写缓存；闭市可能展示过期数据且无显著时间戳提示。
- API 端点尾斜杠不统一、多数裸 dict 无 `response_model`、`BacktestRequest` 日期裸 str 未校验。
- 知识库三层"各自为政"：改 YAML 不联动进向量库；检索用自研 md5 加权 n-gram 哈希向量，无标准 embedding、无 recall@k 基准；`trading_rules.yaml` 强调"禁止 LLM 定义口径"却无 schema 锁定。

---

## 4. P2 —— 优化

- 硬编码兜底价静态日期陈旧（`scripts/shared.py:62-64`），过期仍返回仅打 warning。
- `full_market_cache` 一致性：`refresh_market_cache.py:89` 用本机 `date.today()` 而非 A 股交易日；`dashboard.py:223` 前缀只认 sh/sz、丢北交所 bj。
- BaseAgent `success` 不可信（agent 捕获异常仍 `success=True`）、`model_used` 恒空、`_start_context` 复用存在并发竞态。
- 提示词孤儿/漂移：`chat_assistant_v1.1.txt` 残留、bull/bear/judge 等已删类对应 txt 未清；`ChatAgent.SYSTEM_PROMPT` 硬编码于 `chat_agent.py:511` 与 `chat_assistant.txt` 双源。
- 死代码：`TASK_ROUTING` 全映射 FLASH、`DEEPSEEK_PRO_MODEL`、单值 `ModelTier`、`force_tier`。
- CI 可收紧：mypy 仅配置未执行、network 标记只标了 RAG 一条。
- 仓库卫生：`glm_2020_err.log`(539KB)、`scripts/flylark_*.py` 一次性探针与核心无依赖，建议移入独立目录或标注用途。

---

## 5. 值得保留 / 做得好的

- 进化系统 A/B 验证严谨：预注册判据、配对差分噪声标定、5 次重跑、CSI300 稳健性、双模型（GLM）复验。真实价值收敛为"仅在牛转崩/空头段降险（deepseek +2.0pp 显著，GLM 未确认），牛市/震荡无稳定增益"——**诚实可信，建议保留此结论**。
- 三端摩擦一致（回放/回测/纸盘费率统一）。
- `CodeExecutor` 沙箱（AST 审计 + 白名单）较严。
- API 认证已 fail-closed + `hmac.compare_digest` 常量时间比较。
- 数据多源降级 + 5 分钟冷却的方向正确。

---

## 6. 建议的下一步顺序

1. **先止血（P0，半天内）**：修 `market_diagnostic` temperature bug（一行）→ 断路器真实减仓 → DecisionValidator 补传实盘上下文。
2. **回测口径治理（P0-2/3）**：让策略评级走事件引擎，修正 Sharpe/回撤聚合为等权组合净值口径。
3. **数据层静默失败 + 熔断误伤（P0-6/7）**：空数据与真实失败分离、按标的记失败。
4. **成本/密钥安全（P0-8 + 1.1）**：复核 DeepSeek 峰谷与涨价、统一 LLM 调用走 router、收敛 CORS、密钥轮换。
5. **git 治理（P0-9）**：拆分支提交，隔离核心改动与实验探针。
6. 之后按 P1 清单逐项推进（缓存分层、量口径、知识库检索评测、Dashboard 拆分等）。
