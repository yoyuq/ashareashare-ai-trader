# ashare-ai-trader 全面审查报告

- **审查日期**：2026-07-29
- **版本**：v2.14（Beta）
- **规模**：115 个 Python 文件，约 3.6 万行
- **审查方式**：5 个并行专项审查（回测金融逻辑 / 安全 / 数据指标 / Agent-LLM / 架构测试）+ 静态分析（ruff + 编译）交叉核实。所有 Critical/High 问题均经源码逐行验证，多处已实弹复现（标注 ✅）。

---

## ═══ v3.0 复查 (2026-08-01) — 本轮修复对照 ═══

本报告的 P0/P1 问题已在 v3.0 大规模修复。以下为逐项对照现状：

### 已修复并验证 (196 项测试保护)
- [x] **P0 沙箱逃逸 RCE**: code_executor 移除裸 `__import__` + 受限导入白名单 + dunder 内省拦截
- [x] **P0 认证 fail-open**: 改 fail-closed + `hmac.compare_digest` + 仅接受 X-API-Key 头 + CORS 经 `CORS_ORIGINS` 收敛
- [x] **P0 PBO/MC/DSR**: 真 CSCV-PBO + 符号随机化 MC + Lo(2002) DSR + K 多重检验；`variant_returns` 接线（跨策略矩阵）
- [x] **P0 回测同日收盘成交**: T+1 开盘成交 + 开盘封板判定（`_is_sealed_at_open`）；显式价格绕过已堵
- [x] **P0 数据层静默错误**: 复权语义统一 / 假K线删除 / 空响应熔断 / 股票列表 schema / 北交所代码 / 成交量单位
- [x] **P0 6 个运行时崩溃点**: 全部修复
- [x] **P1 交易规则适配**: 主板 ST ±5%→±10%（2026-07-06 新规）/ 过户费两市 / 盘后固定价格会话 / 沪深基金收盘竞价
- [x] **P1 模拟盘 T+1/涨跌停**: 已接线（sealed flags + 板块感知阈值 10/20/30%）
- [x] **P1 幸存者偏差**: 回测报告显式标注（当前快照不含退市股）
- [x] **P1 模型**: 全量 `deepseek-v4-flash`（删 PRO/Ollama 层）+ 思考模式适配（工具路径关闭思考防 400）
- [x] **P1 AI 防护**: NumericSafetyChecker 真正生效（numpy+Series 修复）/ synthesis 交易参数代码注入并兜底覆盖 / ChromaDB 真实向量检索（哈希嵌入余弦）
- [x] **P1 集成测试**: 断言修复 + tmp 隔离（不再污染真实账户）

### v3.0 新增
- 数据清洗 `clean_ohlcv` 三层接入（standardize + 分析层 + 券商停牌防护）
- API 敏感端点（portfolio/mtm、bot/*）强制鉴权（此前未鉴权可读持仓）
- 预算硬切断 `BUDGET_HARD_CUT`（耗尽抛 BudgetExhaustedError）
- CI 硬化（189 非网络测试 + 覆盖率 + F821/F823 硬门槛）
- 测试从 146 → **196 项**（对抗回归 + 行为测试）

### 遗留（诚实标注）
- 真实 PBO 参数扫描变体矩阵（当前用跨策略矩阵近似，需参数扫描基础设施）
- `daily_runner`/`providers`/`knowledge` 覆盖率偏低（12-33%）
- `notify/` vs `notifications/` 双通知系统未收敛
- `paper_account.py` / `parallel_scanner.py` / bull/bear/judge 模块类层为死代码（未删）
- `config/agents.yaml` 已移除不存在的类引用；README 已同步 v3.0

---



## 一、总体结论

> **核心计算层的骨架是中上的，但大量"卖点功能"是装饰性实现，且存在一批"看似在工作、实则产出错误数据"的系统性 bug。文档严重领先实现。**

### 值得肯定（已验证）

- 分层干净：`data → analysis → backtest` 依赖方向正确，**无循环依赖**
- 凯利公式本体正确，且有 25% 上限 + 分数凯利 + regime 乘数三重保护（`analysis/winrate.py:333-369`）
- 费率口径（佣金万3 最低¥5、印花税 0.05% 仅卖出、过户费 0.001% 仅上交所、整手取整、买入资金校验）模块间基本一致
- `.env` 已正确 gitignore 且从未入 git 历史；SQL 全参数化占位符、YAML 全 `safe_load`、全项目无 pickle/marshal 反序列化 sink
- 组合持久化用了原子写（临时文件 + rename），`to_dict/from_dict` 往返完整
- 可选依赖普遍做了 guarded import 优雅降级（chromadb / redis / xtquant / pandas-ta / numba）
- 净值逐日 mark-to-market、最大回撤 `expanding().max()`、Sortino 用下行偏差 —— 基础绩效实现正确

### 四个支柱性问题

1. **安全**：代码执行"沙箱"可被平凡逃逸 → 未授权远程代码执行（已实弹 PoC）
2. **可信度**：过拟合防控三层（PBO/MC/DSR）统计实现是错的，倾向于给任何策略背书"无过拟合"
3. **正确性**：A股交易规则（T+1/涨跌停）在真实链路上未生效；数据层有多处静默产出错误数据
4. **工程化**：README 唯一安装命令因依赖 pin 不可解析而**装不上**，Docker 构建同样失败

### 成熟度判断

核心计算层（analysis/backtest/risk）质量中上；编排与工程化（config/deps/CI/docker/test-coverage）处于"看起来完整、跑起来心虚"的 Beta 早期状态。

---

## 二、P0 — 必须立即处理的致命问题

### 安全：未授权远程代码执行（RCE）链路【✅ 已实弹验证】

- **位置**：`agent/tools/code_executor.py:87`（`SAFE_BUILTINS` 暴露 `"__import__": __import__`）、`:156-189`（AST 黑名单审计）
- **概述**：所谓"受控沙箱"只是同进程内 `exec(code, safe_globals, safe_locals)`，前置 AST 黑名单只拦幼稚写法（`import os`、字面量 `exec(...)`）。
- **实弹 PoC**（本仓库 `.venv` 中执行成功）：
  ```python
  pwn = __builtins__['__import__']('subprocess').check_output('echo PWNED', shell=True).decode()
  # → {'pwn': 'PWNED\r\n'}  ✅
  ```
- **叠加放大**：`api/server.py:54` 认证 **fail-open**（未设 `API_KEY` 环境变量 → 全部 20+ 端点零认证）+ CORS `allow_origins=["*"]` + 绑 `0.0.0.0`
- **完整攻击链**：`POST /api/v1/analyze`（默认无认证）→ `_technical_analysis_node` → LLM 生成 Python → `exec()` → 服务器进程内任意命令执行（可窃取 `.env` 中 DeepSeek/QMT/Telegram 密钥）
- **修复**：
  1. 立即从 `SAFE_BUILTINS` 删除 `__import__`（numpy/pandas 已预注入，代码无需 import）
  2. 根本上改为"LLM 只输出结构化指标参数、可信 Python 执行"，或真子进程沙箱（nsjail/firejail/RestrictedPython + 禁网 + 资源限额）
  3. 认证改 fail-closed；CORS 收敛到明确域名白名单

### 过拟合防控体系是错的（给过拟合策略背书）

- `backtest/overfitting.py:165-177` **PBO 恒等于 0**：用"随机打乱收益率再算 Sharpe"冒充 Bailey (2014) CSCV-PBO，但打乱不改变均值/标准差 → Sharpe 不变 → `pbo ≡ 0`，任何过拟合策略都报"无过拟合"。真实 PBO 需 CSCV（切块组合配对 train/test，比较多样体"样本内最优者样本外是否仍最优"）。
- `:317-327` **Monte Carlo p 值恒≈0.5**：`np.random.choice(returns, replace=True)` 自助法重采样收益本身，零假设错误（应破坏"信号→收益"配对，如随机入场点位/符号翻转），纯噪声策略也不报警。
- `:183-224` **DSR 的 p 值未真正缩减**：算了 `expected_max`（BLPR 期望最大 SR）得到 `deflated_sr`，但返回的 p 值完全没用它、也没用试验次数 K，而是对原始 SR 做普通正态检验 → 多重检验校正形同虚设。Sharpe 标准误也错（`1/√n` 应为 Lo (2002) 的 `√((1+0.5·SR²)/n)`）。

### A股交易规则在真实链路上未生效

- `backtest/broker.py:382-409` 涨跌停判定读 `bar.get("pct_change", 0)`，但主力源 Baostock 输出列名是 `pctChg`，且 `data/providers/base.py` 的 `col_map` 未映射 → **真实数据上涨跌停模块永不触发**，回测胜率系统性虚高。
- `analysis/recommender.py:116-494` 9 个内置策略全部"**同日收盘决策 + 同日收盘成交**"（第 i 日用 `closes.iloc[i]` 出信号并以同一根 close 成交）；`_backtest_limit_up:288-290` 还在**封板涨停价买入**（现实中封板买不进）→ 胜率/盈亏比严重高估，且这些数字直接喂给推荐与凯利仓位。
- `simulation/paper_trader.py:311-422`（真正被 `daily_runner` 和 `broker/live.py:PaperBroker` 使用的引擎）**完全无 T+1、无涨跌停封板检测、无滑点**；实现了规则的 `simulation/paper_account.py:PaperAccount` 反而未接入 runner。

### 数据层静默产出错误数据

- `analysis/factor_factory.py:263-266` 因子"前瞻收益"实为**后视收益**：`close.iloc[-1]/close.iloc[-(1+forward_period)]-1` 是过去 N 日收益，与因子同 bar。✅ 实证 `returns_5d` 因子与该"前瞻"收益相关系数 = **1.0000**（数学恒等）→ 所有动量类因子 IC 被机械抬高，因子筛选完全失效（目标泄漏）。
- `data/providers/fundamentals.py:134` "最新财报"实为**最老财报**：`ak.stock_financial_abstract_ths` 返回前按报告期**升序**排序（✅ 已核对 akshare 源码），`latest = df.iloc[0]` 取到最早一期；且 `_yoy`（同比）实为相邻两期的环比。→ 基本面评分整体错误。
- `data/router.py:107` 内存缓存 key **不含复权方式**：`f"{method}:{symbol}:{start}:{end}"`，含 adjust 的 `cache_key()` 静态方法（:267）从未被调用 → qfq/hfq/raw 数据互相污染，下游全部指标建立在错误价格序列上，且无日志。

### "代码即推理消除幻觉"在关键节点未落实

- `agent/orchestration/workflow.py:291-329` 市场扫描节点只给 LLM 两个字符串（`regime` + 标的数量），但 prompt（`market_scanner.txt:16-23`）要求输出个股代码+异动强度+关键数据（如放量3倍突破60日线）→ **LLM 凭空捏造**，`scan_results` 随后注入 synthesis 成为"事实依据"。
- `knowledge/prompts/system/synthesis.txt:30-32` 要求交易建议表含 `入场价|止损价|止盈价|仓位%`，但注入上下文无任何价格/仓位计算（ATR 止损等数据存在于 state 却从未传入）→ 日报里的价格是幻觉。同文件 :40 又写"所有数字必须由代码计算"，指令自相矛盾。
- `knowledge/prompts/system/technical_analyst.txt:13-44` 索要 ADX/KDJ/OBV/换手率/支撑阻力位，但 `workflow.py:388-393` 只注入 11 个字段，一半维度无数据 → 逼 LLM 编数。

### 已核实的运行时崩溃点（静态分析 + 手工验证）

| 文件:行 | 问题 | 后果 |
|---|---|---|
| `analysis/indicators.py:694` | `np` 局部变量提前引用（:701 局部 `import numpy as np` 使其全函数变局部） | `search_similar_patterns` 形态搜索必崩 UnboundLocalError |
| `broker/live.py:389-390` | `xtquant` 模块名未绑定（只有 `from xtquant import xtdata, xttrader`） | QMT 实盘下单必然 NameError → 永远 REJECTED |
| `knowledge/strategies/strategy_executor.py:141,195` | `logger` 从未导入/定义 | 异常处理路径自身崩溃，掩盖原始错误 |
| `analysis/factor_factory.py:78` | `high_low_ratio` lambda 参数是 `h,l,n` 却用未定义的 `c` | 该算子调用即 NameError |
| `simulation/daily_runner.py:560` | `regime_info` 未定义（在 try/except 内被吞） | 日终通知永久静默失败 |
| `backtest/broker.py:112-115,467` | `Account.total_return` 引用不存在的 `total_return_pct` | `get_performance()` 抛 AttributeError |

### 安装即坏

- `pyproject.toml:29` `pandas-ta>=0.3.14` 在 PyPI **不存在正式版**（只发布过 `0.3.14b0/b1` 永久 beta；PEP 440 下 `0.3.14b0 < 0.3.14`，匹配空集）→ README:23 安装命令 `pip install -e ".[dev,backtest]"` 与 `docker/Dockerfile:12` 构建全部失败。修复：`>=0.3.14b0`。
- 同类隐患：`pyproject.toml:72` `mplfinance>=0.3.10`（实为 `0.12.10b0`，且根本没被 import）。

---

## 三、各维度 High 级问题

### 回测与金融逻辑

- **H1** DSR p 值未缩减、Sharpe 标准误公式错（应 Lo 2002）
- **H2** `analysis/winrate.py:406` 推荐"建议股数"恒按 ¥10/股硬编码：`shares = int(position_amount/10/100)*100`。茅台 entry≈1800、仓位¥20,000 → 输出 200 股（价值¥36万），严重超买
- **H3** `analysis/recommender.py:547` 逐笔交易收益用 √252 年化（间隔不定，无意义）；`max_dd` 用 `cumsum(trades_net)` 百分点直接累加当净值曲线
- **H4** `backtest/broker.py:226-238` 加仓不更新 `can_sell_date` → 新增股份当日即可卖出（T+1 漏洞）
- **H5** `simulation/paper_account.py:111-113` 卖出费率把印花税按 0.1% 计（2023-08 起应为 0.05%），与其余模块不一致
- **M** multi_factor 回测 `vol_score` 被覆盖且加了两次（低波因子丢失）；"Walk-Forward"实为 OOS 季度稳定性无逐窗再优化；冲击成本用单日 |ret| 近似日波动率（系统性低估）；ST ±5% 涨跌停未覆盖；市价单以收盘价成交（时序乐观）

### 安全

- **High** 认证 fail-open + CORS `*` + 绑 `0.0.0.0`（见 P0）
- **Medium** Prompt 注入 → 工具/代码执行链路无隔离（外部数据直接拼入 LLM 上下文，LLM 可触发代码执行）
- **Medium** API Key 可经 query string 传递（落日志/Referer）+ `!=` 非常量时间比较（时序侧信道）
- **Medium** 依赖全不锁版本（无 lockfile/hash）；`langchain-deepseek>=1.0.0` 并非公认 PyPI 包名（typosquat 风险）；`pyfolio` 已停止维护
- **Low** 会话文件名由用户输入手工净化（建议白名单 + resolve 校验）
- **已验证安全**：密钥无硬编码、无 pickle/非安全 yaml、SQL 全参数化、无 SSRF、MCP Server 只读白名单正确、QMT 实盘未被 API/Agent 引用

### 数据层与指标

- **H1** 复权方式各源语义不一致：Baostock 永远 qfq 且忽略请求；AKShare `None→qfq`；EastMoney `None→不复权`。降级时价格序列在除权日跳空 → 假信号
- **H2** 双源交叉验证**从未生效**：各源 date dtype 不一（str/date/Timestamp）→ 交集为 0 恒跳过；默认配置下 Tencent 返回无列空 DataFrame → `set_index("date")` 直接 KeyError 崩溃
- **H3** ADX/+DI/-DI 在 DatetimeIndex 输入下全 NaN（`np.where` 返回 ndarray 重建为 RangeIndex，与 atr 索引对齐失败）→ 下游评分静默偏低
- **H4** 增量计算窗口 120 日历天 ≈ 80 交易日 < ma_250/ema_250 → ✅ 实证 ema_250 全量=68.93 vs 增量=58.84（差 10 点），ma_250 增量恒 NaN
- **H5** `base.py:169` 校验硬性要求 ≥5 行 → 1~4 天合法请求被判失败并触发健康源降级 5 分钟
- **H6** `data/processors/pit.py` PIT 处理器实现完整但**从未接入**生产管线（仅测试引用）→ 历史回测用"未来"财报与"今天才存在"的成分股，前视偏差
- **M** RSI/ATR/ADX 用 `ewm(span=n)`（α=2/(n+1)）而非 Wilder α=1/n（✅ RSI(14) 与标准平均差 4.45）；EastMoney 缓存 TTL 用 `.seconds` 跨天失效；EastMoney 未实现单数 `get_realtime_quote`（死代码）；`register()` 优先级排序 key 是常量（无效）；涨跌停 ±9.5% 一刀切（ST/创业板/科创板/北交所未区分，用复权价对 shift(1) 除权日必误判）；`scanner.py:423` macd_hist 自己比自己恒 False；Tencent 停牌股产出 -100% 假跌幅；Baostock 北交所代码错配到 sz

### Agent / LLM 层

- **H1** DeepSeek 旧模型名 `deepseek-chat`/`deepseek-reasoner` 已于 **2026-07-24 弃用**（今天 07-29）→ 若端点已下线，全部付费调用 404，静默降级为模板
- **H2** DeepSeek 调用**无超时**（SDK 默认 600s）；`ModelConfig.request_timeout=60` 是死字段 → 端点 hang 时流水线阻塞 10 分钟
- **H3** 沙箱逃逸（见 P0）
- **H4** `NumericSafetyChecker` 校验的是一份**随即丢弃**的内部 narrative，用户可见的正式报告从不校验；校验失败时锁定数字反而完全不注入
- **H5** Critic 审计用硬编码输入（`filter_ipo_days` 缺失、单体制占比 1.0）→ 所有股票恒定触发相同缺陷，robustness 恒 ≈73/100，走形式还烧 PRO 档 token
- **H6** `/api/v1/cost/summary` 每次 new 一个 `CostMonitor()` → 恒返回 0；`record_call` 在生产代码零调用方，成本告警/切断从未生效
- **H7** 预算超限强制 LOCAL，但 Ollama 离线时 chain 清空回填 `[FLASH]` → 继续付费调用，日预算形同虚设
- **H8** LLM 输出解析失败静默落默认值（默认 HOLD/conviction 0.3 被当真实决策持久化，污染反思注入与胜率统计）；空响应 `content=""` 无检测
- **M** 高峰降级窗口（9-12,14-18）覆盖整个交易时段，盘中最该保证质量时用最弱模型；成本虚构"高峰×2"定价；月预算从不执行；"断点续跑"完全不可用（缺 `langgraph-checkpoint-sqlite` + 每次新 thread_id + 同步 Saver 阻塞事件循环）；反思注入主路径恒为空；SignalTracker 在 LLM 路径因 `debate` NameError 被静默跳过；ChromaDB"向量检索"实为关键词匹配 + 硬编码相似度 0.5；`/api/v1/analyze` `router=None` 静默退化为模板，`suggestions` 字段从未被赋值恒为空；`agents.yaml` 引用不存在的类（bull/bear/judge 类层是死代码）

### 架构 / 代码质量 / 测试

- **C2** 三个最关键模块（`data/router` 降级/交叉验证、`models/router` LLM 兜底链、`recommender.RecommendationEngine`）**零行为测试**，CI 全绿 = 虚假安全感
- **C3** 8 个网络测试断言为零（`try/except` 包住真实 LLM 调用只 print，pytest 忽略返回值）+ 恒真断言（`assert True`、`assert t2 is None or t2 is not None`）；注册了 `network`/`slow` 标记却零测试使用，每次全量测试照常烧钱；`test_deepseek_api.py:14` 把 key 片段打进日志
- **H1** `requests` 被 4 个核心模块顶层硬 import，但 pyproject 与 requirements **均未声明**（仅靠 akshare 传递性带入）
- **H2** `config/settings.yaml`/`model_config.yaml` **从未被任何代码加载**（grep 证实仅在注释出现）；真实配置是 ~20 处散落 getenv 且默认值已互相冲突（Postgres 密码代码默认 `"ashare"` ≠ .env）；`symbols.yaml` 有 6 个重复加载器
- **H3** `notifications/`（标 DEPRECATED）仍接在主分析管线 `run_daily_analysis.py`，与 `notify/` 双跑
- **H4** CI 形同摆设：mypy `if: false`、lint `continue-on-error: true`、integration/e2e/deepseek/competition 测试全不在 CI、无覆盖率门槛；Docker 把 pytest/black/mypy/backtrader/vectorbt 装进生产镜像、editable 安装先于 `COPY . .`
- **H5** TA-Lib"双引擎"是虚构：全仓零 `import talib`，`use_talib` 参数被写从不读；scikit-learn 宣称但从未 import
- **H6** `web/dashboard.py` 1450 行上帝模块 + 12 处裸 `except: pass`（初始化失败被吞 → 界面看似正常实则无数据）；streamlit 仅在 `[all]` extra，按 README 装 `[dev,backtest]` 后 `streamlit run` 直接 ImportError
- **H7** 凭据/实盘风险：`.env` 工作区有疑似真实 key；`broker/live.py:300` 硬编码 Windows QMT 路径；`submit_order` 直连真实账户无 dry-run/金额上限守卫
- **M** 依赖清单冗余漂移（torch/transformers/celery/asyncpg 等 ~18 个从未 import，`pip install -r` 平白拉 2-3GB）；4 个扫描器职责重叠（`ParallelScanner` 是死代码）+ composite_scorer/factor_factory/factor_evaluator 孤儿簇；大量 `except Exception: pass/continue` 静默吞噬；业务常量（风控阈值/权重/模型名/URL）硬编码；4 个上帝模块；scripts 166 处 print 与 loguru 混用；README 数字与代码漂移（宣称 90 单测，实为 190）

---

## 四、系统性模式（最该警惕的共性问题）

1. **"装饰性实现"泛滥**：PBO、Monte Carlo、DSR、双源交叉验证、Critic 对抗审计、向量检索（实为关键词匹配+硬编码相似度0.5）、断点续跑、成本监控、PIT 处理器、TA-Lib 双引擎 —— **功能都存在，实现都不生效**。这比"没有功能"更危险，因为它制造虚假可信度。

2. **"看似在工作、实则是错的"数据 bug**：排序方向反、缓存 key 漏字段、索引对齐丢失、前瞻/后视收益混淆 —— 这类 bug 不崩溃、静默产出错误结果，是量化项目的隐形杀手。

3. **静默失败**：大量 `except Exception: pass/continue`，解析失败落默认值，通知失败 debug 日志带过。故障无法定位，错误数据无法察觉。

4. **文档领先实现**：README 的 8 数据源、TA-Lib 双引擎、90 单测、断点续跑等多处与代码不符。

5. **配置断层**：配置文件没人读，代码里全是硬编码（风控阈值、模型名、权重、URL 散落各处）。

---

## 五、修复路线图

### P0 — 本周（止血）
1. 删 `code_executor.py` 的 `__import__`，认证改 fail-closed，CORS 收敛 ← *阻断 RCE*
2. 修 `pandas-ta>=0.3.14b0` ← *解锁安装/Docker*
3. 验证并更新 DeepSeek 模型名 ← *否则所有 LLM 调用可能已 404*
4. 修 6 个运行时崩溃点（见 P0 表）← *都是几行的修复*
5. 修 fundamentals 排序方向、缓存 key 加 adjust、factor_factory 前瞻收益对齐

### P1 — 本迭代（可信度）
6. 回测改"T 日信号 / T+1 开盘成交 + 涨跌停禁买禁卖"，修 `pctChg` 字段映射
7. 重写或移除 PBO/MC/DSR（要么实现真正的 CSCV-PBO，要么别宣称）
8. 把 T+1/涨跌停从 paper_account 统一接入真实 paper_trader
9. 建中央配置加载器，接线或删除死配置文件
10. 给网络测试打 `network` 标记、删恒真断言；CI 纳入 integration

### P2 — 下一迭代（收敛）
11. 让 LLM 只做叙事：Python 算好异动/价格/仓位注入 prompt
12. 补 data router / models router / recommender 行为测试
13. 收敛双通知系统、4 个扫描器、死代码簇
14. 拆 god-module（dashboard 1450行 / workflow 1351行 / server 1233行）

---

*本报告由 5 个专项审查 agent 并行生成，经静态分析交叉核实。各维度完整发现（含 Medium/Low 全部条目）见随附审查记录。*

---

## 六、修复记录（2026-07-29 当日完成）

### P0 — 已修复（13 项）
| # | 问题 | 修复 | 验证 |
|---|---|---|---|
| 1 | 沙箱逃逸 → RCE | 删除裸 `__import__`，改为仅放行 numpy/pandas 的受限导入；AST 审计新增 dunder 内省拦截 + 危险属性链黑名单 | 6 条逃逸路径实弹复测全部拦截，白名单 import 正常 |
| 2 | 认证 fail-open | 改 fail-closed（未设 `API_KEY` 拒绝服务，`API_ALLOW_INSECURE_NO_AUTH` 开发开关）；`hmac.compare_digest` 常量时间比较；仅接受 `X-API-Key` 头；CORS 经 `CORS_ORIGINS` 可配 | 编译通过 + 文档同步 |
| 3 | `pandas-ta>=0.3.14` 不可解析 | 改 `>=0.3.14b0`（另修 mplfinance） | 依赖可解析 |
| 4 | 6 个运行时崩溃点 | indicators `np` 局部引用、broker `total_return_pct` 缺失、live `xtconstant` 导入、strategy_executor `logger` 导入、daily_runner `regime_info`、factor_factory lambda | ruff F821/F823 清零 |
| 5 | 财报取最老一期 + 增速实为环比 | 报告期降序取最新；新增 `_yoy_growth` 同比（一年前同期，缺失退回环比） | 逻辑审查 |
| 6 | 缓存 key 不含复权方式 | 内联 key 加入 `frequency:adjust` | 逻辑审查 |
| 7 | 因子前瞻收益目标泄漏 | 因子时点前移 `forward_period` 根与未来收益对齐 | 数值验证：IC 1.000000 → −0.049 |
| 8 | `server.py` 不加载 `.env` | 加 `load_dotenv()` 于环境变量读取前 | 编译通过 |

### P1 — 已修复（6 项）
| # | 问题 | 修复 | 验证 |
|---|---|---|---|
| 9 | PBO 恒为 0（打乱收益≠PBO） | 重写为真正的 CSCV-PBO（切块组合划分，IS 最优变体 OOS 是否低于中位数）；无变体矩阵时返回 NaN 不参与判定 | 随机矩阵 PBO=0.65，含真信号 0.06 |
| 10 | MC p 值恒≈0.5（bootstrap 零假设错） | 改为符号随机化零分布（保留波动结构、期望收益为 0） | 噪声 p=0.53、强 alpha p=0.000、负收益 p=0.998 |
| 11 | DSR p 值未缩减 | p 值基于 `(SR−SR₀)/se`，se 用 Lo 2002 偏度/峰度修正公式 | K=1 p=0.046 显著；K=1000 p=1.0 被多重检验抹平 |
| 12 | 回测同日收盘成交 + 涨停板买封板价 | 新增 `_exec_prices` 执行层：T 日信号 / T+1 开盘成交，涨停封板禁买、跌停封板禁卖；全部 10 个策略回测函数重构；顺带修复 multi_factor 因子覆盖/重复加权、low_volatility 索引错位 | 合成数据验证时序与封板拦截 |
| 13 | 模拟盘无 T+1/涨跌停 | `execute_buy` 涨停封板拒买；`execute_sell` 当日买入拒卖（T+1）+ 跌停封板拒卖；测试同步更新 | 5 项行为验证通过 |
| 14 | 涨跌停判定字段不匹配 | `standardize()` col_map 增加 `pctChg→pct_change`、`preclose→pre_close` | 单位核对一致 |

### 工程化 — 已修复
- 联网测试（deepseek/e2e/competition/rag）标记 `@pytest.mark.network`，pyproject 默认 `-m "not network"` 排除 → 全量套件从**挂起/烧钱**变为 **11 秒、162 通过**
- 删除恒真断言（`assert True` 等 4 处），替换为有意义断言；移除测试中的 API key 片段打印
- 陈旧测试断言修正（策略数 9→≥9、PBO NaN 兼容）
- 清理 5 处 F811 重复导入；`.env.example` / README 补充 API 认证文档

### 仍未处理（建议后续迭代）
- DeepSeek 模型名 `deepseek-chat/reasoner` 已过弃用期（需用实际 key 验证后更新）
- 回测引擎 `backtest/engine.py` 市价单以收盘价成交（与 recommender 的修复同口径，需同步改造）
- 中央配置加载器（settings.yaml 接线）、双通知系统收敛、4 扫描器收敛
- Critic 审计硬编码输入、NumericSafetyChecker 校验对象错位、"断点续跑"死功能
- 各数据源复权语义统一、PIT 处理器接入回测管线、RSI/ATR 改 Wilder 平滑

