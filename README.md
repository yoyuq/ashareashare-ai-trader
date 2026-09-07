# 🏦 A股智能分析Agent (AShare AI Trader) v6.1

> AI驱动的A股量化分析与策略研究平台 — 多源数据 → 多智能体分析 → 回测验证 → 纸面交易 → 前瞻 OOS 验证 → 自主学习闭环

[![Tests](https://github.com/yoyuq/ashareashare-ai-trader/actions/workflows/ci.yml/badge.svg)](https://github.com/yoyuq/ashareashare-ai-trader/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-6.1-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## 快速开始

```bash
# 1. 克隆 + 虚拟环境
git clone https://github.com/yoyuq/ashareashare-ai-trader.git
cd ashare-ai-trader
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. 安装
pip install -e ".[dev,backtest]"

# 3. 配置 (只需 DeepSeek API Key)
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-xxx

# 4. 快速验证
python scripts/run_daily_analysis.py --no-llm --symbols sh.600519,sz.300750

# 5. 启动 Dashboard
streamlit run web/dashboard.py
```

## 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    分析管线 (LangGraph 有向图)                      │
│  数据准备 → 技术分析 → 市场扫描 → 策略匹配 → 回测验证                │
│      ↓                                                           │
│  多视角辩论 (3视角×多空 = 6路并行 → Judge 定向修订最弱视角)           │
│      ↓                                                           │
│  Critic 5维审计 → 综合研判(代码算交易参数)                          │
│      ↓                                                           │
│  Evaluator 质量评估 ──不达标──→ 回炉修订(带批评重写, ≤2轮)          │
│      ↓ 达标                                                      │
│  DecisionValidator 硬约束校验 (涨跌停/T+1/仓位/整手/集中度)          │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                    执行层 / 验证层                                  │
│  Paper Trading (T+1/涨跌停) | 8层风控 | Daily Runner               │
│  kill-test: LLM vs 规则 vs 随机 三路前向收益对比 (alpha 验证)        │
│  ChromaDB RAG: 语义检索 + K线形态相似 → 注入辩论上下文               │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                 策略研究闭环 (v5.x-v6.x, 私有研究产物)               │
│  因子/事件/筛层 A/B 实验 (预注册判据, 先写 gate 后跑回测)             │
│      ↓                                                           │
│  前瞻 OOS 验证: 回测 → 冻结入场 → 纸面跟踪 → 判定日逐字执行          │
│  拥挤度/注意力监控 (测量层只读) · 全程零模拟 (缺数据报错不兜底)        │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                 自主学习闭环 (agent/learning/)                     │
│  联网研究 → 语义查重 → 忠实测试 → 向量库记录 → 滚动重测              │
└──────────────────────────────────────────────────────────────────┘
```

### 核心特性

| 特性 | 说明 |
|------|------|
| **8 数据源** | Baostock / Tencent / EastMoney / AKShare / Tushare / 另类数据 / 基本面 / 财务 |
| **130+ 技术指标** | pandas-ta 引擎, 10大类 (趋势/动量/波动/量/形态/周期/统计/自定义/因子) |
| **6 市场状态** | strong_bull / weak_bull / range_bound / weak_bear / strong_bear / crisis |
| **统一模型** | DeepSeek V4-Flash (全量, ¥1-2/M) — 单模型路由 + 预算/成本监控 |
| **多智能体协作** | DeerFlow 模式 — 多视角发散 / Evaluator 反思回炉 / DecisionValidator 执行前校验 |
| **RAG 检索** | ChromaDB 双通道 (bge-m3 语义 + K线形态相似), 余弦相似度, 中文降级 |
| **自主学习闭环** | 联网研究 (Bocha) → bge-m3 语义查重 → 忠实模板测试 → 知识库回写 → 滚动重测 |
| **前瞻 OOS 验证** | 回测通过 → 预注册冻结 → 纸面平行 bet → 判定日逐字执行 (零模拟) |
| **6 层过拟合防控** | 时间分割 → PBO → Deflated SR → Walk-Forward → 参数敏感性 → Monte Carlo |
| **8 层风控** | 回撤熔断 / ATR止损 / 移动止盈 / 市场仓位 / 防踩踏 / 持仓天数 / 跌停检测 / 相关性 |
| **代码即推理** | Python 做计算, LLM 只叙事 — NumericSafetyChecker 消除数字幻觉 |
| **Chat Agent** | 自然语言问答 + Function Calling (含联网搜策略/判策略/学习落库 4 工具) |

---

## 项目结构

```
ashare-ai-trader/
├── agent/              # AI Agent 层
│   ├── chat_agent.py       # 对话式Agent (自然语言→工具调用, 含学习4工具)
│   ├── orchestration/      # LangGraph 工作流编排 (11节点 + 反思循环)
│   ├── sub_agents/         # 子Agent (Technical/Synthesis/Critic/Evaluator/Validator...)
│   ├── learning/           # 自主学习闭环 (researcher/tester/curiosity/kb_writer)
│   └── tools/              # Agent 工具 (Code Executor / NumericSafetyChecker)
├── analysis/           # 分析引擎 (指标/市场状态/扫描器/风控/胜率/北向)
├── data/               # 数据层 (多源路由降级 + PIT清洗 + 缓存)
├── models/             # LLM 模型层 (单模型路由 + 成本监控)
├── backtest/           # 回测引擎 (事件驱动 + A股券商模拟 + 6层过拟合防控)
├── simulation/         # 模拟交易 (paper_trader / portfolio / daily_runner)
├── killtest/           # 三路信号验证 (v3 历史验证工具, 仅测试引用)
├── knowledge/          # 知识库 (YAML规则/参考md/ChromaDB向量库)
├── notify/             # 通知系统
├── api/                # REST API (v6.0 模块化)
│   ├── server.py           # app + 认证/限频中间件 + 路由注册
│   ├── schemas.py / deps.py
│   └── routers/            # market/analysis/chat/portfolio/decisions/competition/bot
├── web/                # Dashboard (v6.0 模块化)
│   ├── dashboard.py        # 入口 (~180行)
│   ├── theme.py / api_client.py / data.py / realtime.py
│   └── tabs/               # 9 个 tab 模块
├── scripts/            # 自动化与研究脚本 (见下表)
├── tests/              # 测试 (离线 + 网络)
└── config/             # YAML 配置
```

### scripts/ 速查

| 类别 | 代表脚本 |
|------|---------|
| 每日流程 | `run_daily_analysis.py` (手动 CLI) · `python -m simulation.daily_runner` (调度入口) |
| 回测对比 | `backtest_compare.py` / `historical_replay.py` / `compare_replay_ab.py` |
| 数据建库 | `fetch_live_panel.py` (实时面板增量) / `fetch_lhb_history.py` (龙虎榜) / `build_gdhs_history.py` (股东户数) / `build_disclosure_timing.py` (披露时间表) / `fetch_restricted_release.py` (解禁) / `fetch_dividends.py` / `fetch_delisted_daily.py` (退市股并入) |
| 前瞻验证 | `forward_register_*.py` (bet 注册) / `forward_track.py` (纸面跟踪快照) |
| 监控测量 | `crowding_watch.py` (拥挤度/换手分位监控) / `attention_collector.py` (人气榜采集) |
| 知识库 | `rebuild_knowledge_index.py` / `eval_retrieval.py` (检索质量基准) |
| 学习闭环 | `learn_external.py` (--auto/--revalidate/--report) |
| kill-test | `killtest.py` (三路信号 CLI) |
| 部署运维 | `fix_scheduled_tasks.ps1` / `downgrade_daily_live_weekly.py` (任务计划) |

> 策略研究的预注册文档、A/B 结果与操作单为私有研究产物 (reports/ 不入库)。

---

## 使用方式

### 1. 每日一键分析

```bash
python scripts/run_daily_analysis.py            # 完整 LLM 流程
python scripts/run_daily_analysis.py --no-llm   # 快速规则引擎
python -m simulation.daily_runner               # 完整交易日流程 (分析→交易→总结)
python -m simulation.daily_runner --dry-run     # 仅分析
```

### 2. Dashboard / API

```bash
streamlit run web/dashboard.py                  # http://localhost:8501
uvicorn api.server:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" -H "X-API-Key: your-key" \
  -d '{"message": "茅台怎么样？"}'
```

> **API 认证**：`API_KEY` 是自定义的接口密码 (非 AI 模型密钥), 经 `X-API-Key` 请求头传递。安全默认 fail-closed：未设置时除 `/health` 外全部 403。

### 3. 策略回测对比

```bash
python scripts/backtest_compare.py --years 3 --symbols 10
```

### 4. 前瞻 OOS 验证 (回测 → 纸面跟踪)

```bash
python scripts/forward_register_cold_lowvol.py  # 注册平行 bet (一次性)
python scripts/forward_track.py                 # 每交易日快照跟踪 → registry.json
python scripts/crowding_watch.py                # 拥挤度/换手分位监控 (测量层)
```

### 5. 自主学习闭环

```bash
python scripts/learn_external.py --topics 价值投资策略 --dry-run  # 研究+测试 (不联网)
python scripts/learn_external.py --auto 3                        # 好奇队列自动轮转
python scripts/learn_external.py --revalidate                    # 滚动重测 (out-of-sample)
python scripts/learn_external.py --report                        # 学习元报告
```

### 6. RAG 检索

```bash
python -c "from knowledge.manager import KnowledgeManager; km=KnowledgeManager(); print(km.rag_query('震荡市适合什么均线策略'))"
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 数据 | AKShare / Baostock / 东方财富 / 腾讯行情 / Tushare |
| 数据库 | PostgreSQL + TimescaleDB / Redis / ChromaDB |
| 向量 | SiliconFlow BAAI/bge-m3 (1024维语义) |
| 分析 | pandas-ta / scikit-learn / numpy / scipy |
| LLM | DeepSeek V4-Flash (统一) |
| 搜索 | Bocha 联网搜索 (学习闭环) |
| 编排 | LangGraph |
| 回测 | 自研事件驱动引擎 |
| Web | FastAPI + Streamlit |
| 部署 | Windows 任务计划 |

---

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | 否 | API 地址 (默认 https://api.deepseek.com/v1) |
| `SILICONFLOW_API_KEY` | 否 | 硅基流动 Key (bge-m3 语义向量, 学习闭环) |
| `SEARCH_PROVIDER` / `SEARCH_API_KEY` | 否 | Bocha 联网搜索 (学习闭环真联网) |
| `API_KEY` | 否 | REST API 认证密码 (支持逗号分隔多 key 轮换; 未设置时默认拒绝访问) |
| `API_ALLOW_INSECURE_NO_AUTH` | 否 | 未设 API_KEY 时允许无认证访问 (仅本地开发) |
| `CORS_ORIGINS` | 否 | CORS 来源白名单 (默认关闭跨域) |
| `RATE_LIMIT_PER_MINUTE` / `RATE_LIMIT_BURST` | 否 | 滑动窗口限频 |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | 否 | 数据库连接 |
| `REDIS_HOST/PORT/DB` | 否 | Redis 连接 |

---

## 项目状态 (v6.1)

- [x] **Phase 1-5**: 数据层 / 分析引擎 / 回测引擎 / 知识库+RAG / AI Agent / 代码即推理 / 多智能体 / kill-test / 模拟交易 (v3.x)
- [x] **自主学习闭环** (v5.10-v5.11): 联网研究 + bge-m3 语义查重 + 忠实测试 + 判据预注册 + 滚动重测 + Chat Agent 4 学习工具
- [x] **策略研究管线** (v5.12): A/B 实验框架 + 预注册纪律 (gate 先写死) + 前瞻 OOS 验证闭环 + 退市股幸存者偏差修正
- [x] **监控测量层**: 拥挤度/换手分位/注意力监控 (只读预警, 不做选券)
- [x] **v6.0**: web/api 模块化拆分 (行为不变, 路由逐字一致)
- [x] **v6.1**: 死码清理 + daily_runner 交易日守卫 + 数据零模拟纪律加固
- [ ] **进行中**: 前瞻 OOS 判定日 (60td 逐字执行) · 长期信号跟踪积累 · 生产化监控

## 诚实声明

- **工程质量**: 离线测试 + CI 硬门槛 + 幻觉治理 (NumericSafetyChecker) + 6 层过拟合检测; 全程零模拟纪律 (行情/回测/压力测试一律真实数据, 缺数据报错不兜底)。
- **投资可行性**: 回测 alpha 与实盘可实现收益之间存在系统性差距 (成本/执行/幸存者偏差/风格轮动)。本项目的回应是**预注册 + 前瞻 OOS 验证**: 策略判据在回测前写死, 真伪由判定日的 out-of-sample 证据裁决, 不做事后调参。
- **已知边界**: 免费数据源 (Baostock/Tencent) 长期可靠性未验证; 分钟/L2 级历史数据不可得, 高频方向不在可测范围; 回放与实盘存在固有执行差异 (T+1 开盘 vs 实时)。

---

## License

MIT — 个人研究用途。**⚠️ 风险提示：历史数据不代表未来表现，本项目仅供研究和学习，不构成投资建议。**
