# 🏦 A股智能分析Agent (AShare AI Trader) v3.1

> AI驱动的A股量化分析助手 — 市场扫描 → 多维分析 → 策略回测 → 多视角辩论 → 综合研判 → 质量评估 → 决策验证 → 模拟交易

[![Tests](https://github.com/hjl/ashare-ai-trader/actions/workflows/ci.yml/badge.svg)](https://github.com/hjl/ashare-ai-trader/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-3.1-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## 快速开始

```bash
# 1. 克隆 + 虚拟环境
git clone https://github.com/hjl/ashare-ai-trader.git
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

### Docker 一键部署

```bash
docker compose up -d          # 启动全部服务(API+DB+Redis)
curl http://localhost:8000/health
```

---

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
```

### 核心特性

| 特性 | 说明 |
|------|------|
| **8 数据源** | Baostock / Tencent / EastMoney / AKShare / Tushare / 另类数据 / 基本面 / 财务 |
| **130+ 技术指标** | pandas-ta 引擎, 10大类 (趋势/动量/波动/量/形态/周期/统计/自定义/因子) |
| **6 市场状态** | strong_bull / weak_bull / range_bound / weak_bear / strong_bear / crisis |
| **9 策略回测** | 双均线 / MACD / 布林 / RSI / 动量突破 / 涨停板 / 低波动 / 海龟 / 多因子 |
| **统一模型** | DeepSeek V4-Flash (全量, ¥1-2/M) — v3.0 移除 Ollama/Pro 分层 |
| **多智能体协作** | v3.1 DeerFlow 模式 — 多视角发散 / Evaluator 反思回炉 / DecisionValidator 执行前校验 |
| **RAG 检索** | ChromaDB 双通道 (语义 + K线形态相似), 余弦相似度, 中文降级 |
| **kill-test 验证** | 三路信号 (LLM/规则/随机) N日前向收益对比 + Welch t 显著性 |
| **6 层过拟合防控** | 时间分割 → PBO → Deflated SR → Walk-Forward → 参数敏感性 → Monte Carlo |
| **8 层风控** | 回撤熔断 / ATR止损 / 移动止盈 / 市场仓位 / 防踩踏 / 持仓天数 / 跌停检测 / 相关性 |
| **代码即推理** | Python 做计算, LLM 只叙事 — 消除 AI 幻觉 |
| **Chat Agent** | 自然语言问答 + Function Calling 工具调用 + 多轮会话持久化 |

---

## 项目结构

```
ashare-ai-trader/
├── agent/              # AI Agent 层
│   ├── chat_agent.py       # 对话式Agent (自然语言→工具调用)
│   ├── orchestration/      # LangGraph 工作流编排 (11节点 + 反思循环)
│   ├── sub_agents/         # 子Agent (Technical/Synthesis/Critic/Evaluator/Validator...)
│   └── tools/              # Agent 工具 (Code Executor / NumericSafetyChecker)
├── analysis/           # 分析引擎
│   ├── indicators.py       # 130+ 技术指标
│   ├── regime.py           # 6 状态市场识别
│   ├── scanner.py          # 全市场扫描器 (5000+股)
│   ├── recommender.py      # 交易推荐引擎 + 9 策略回测
│   ├── winrate.py          # 胜率分析 + 凯利仓位 + 信号追踪
│   ├── multiframe.py       # 多时间框架 + 因子IC衰减
│   ├── sector_rotation.py  # 板块轮动分析
│   ├── northbound.py       # 北向资金追踪
│   ├── risk_controls.py    # 8层风控 + 信号分级
│   └── incremental.py      # 增量计算引擎
├── data/               # 数据层
│   ├── router.py           # 多源路由器 (降级+缓存+交叉验证)
│   ├── providers/          # 8 个数据源
│   ├── processors/         # PIT (Point-in-Time) + clean_ohlcv 清洗
│   ├── storage/            # PostgreSQL/TimescaleDB models
│   └── cache.py            # Redis 缓存层
├── models/             # LLM 模型层
│   ├── router.py           # 单模型路由 (V4-Flash + 预算/成本)
│   └── cost_monitor.py     # API 成本追踪
├── backtest/           # 回测引擎
│   ├── engine.py           # 事件驱动回测引擎
│   ├── broker.py           # A股券商模拟 (T+1/涨跌停/费率/停牌守护)
│   ├── overfitting.py      # 6 层过拟合防控 (CSCV-PBO/DSR)
│   └── impact.py           # Almgren-Chriss 冲击成本
├── simulation/         # 模拟交易
│   ├── paper_trader.py     # Paper Trading 引擎
│   ├── portfolio.py        # 持仓状态管理 (JSON持久化)
│   └── daily_runner.py     # 每日完整流程
├── killtest/           # 🆕 信号验证 (v3.1)
│   ├── arms.py             # 三路信号: Rule / Random / LLM
│   ├── tracker.py          # 信号持久化 + N日前向收益结算
│   ├── report.py           # 对比 + Welch t 显著性
│   └── runner.py           # 编排 (历史回溯 / 基线)
├── knowledge/          # 知识库
│   ├── manager.py          # 知识库管理器 (Prompt注入/ChromaDB向量检索)
│   ├── prompts/            # Agent System Prompts
│   ├── rules/              # 交易规则/指标手册 YAML
│   ├── strategies/         # 策略注册表
│   ├── reference/          # 参考文档 (术语表/市场周期)
│   └── vector_store/       # ChromaDB 向量库 (cosine)
├── notify/             # 🆕 通知系统 (v3.1, 收敛自 notifications/)
│   ├── __init__.py         # NotificationManager (新 API)
│   └── service.py          # NotificationService/DailySummary (旧通道服务)
├── scripts/            # 自动化脚本
│   ├── run_daily_analysis.py  # 每日分析流水线
│   ├── morning_buy.py         # 早盘买入 (消费 workflow trade_params)
│   ├── evening_sell.py        # 盘后卖出检查
│   ├── evening_summary.py     # 盘后总结 (MTM+止盈止损)
│   ├── killtest.py            # kill-test CLI (三路信号对比)
│   ├── scheduler.py           # 定时调度器
│   ├── backtest_compare.py    # 策略对比回测
│   └── shared.py              # 共享工具 (ATR/凯利/价格)
├── api/                # REST API
│   └── server.py             # FastAPI 服务 (20+ 端点)
├── web/                # Dashboard
│   └── dashboard.py          # Streamlit 面板
├── tests/              # 测试 (260 离线通过 + 28 网络)
│   ├── unit/                 # 单元测试
│   ├── integration/          # 集成测试
│   └── test_smoke.py         # 冒烟测试
├── reports/            # 自动生成的每日报告
├── simulation_data/    # 模拟交易状态 + validation_journal.jsonl
├── killtest_data/      # kill-test 信号/结果持久化
├── config/             # YAML 配置
│   ├── settings.yaml        # 全局配置
│   ├── agents.yaml          # Agent 注册表 (7 个)
│   └── symbols.yaml         # 股票池 (60+精选+行业ETF)
└── docker/             # Docker 部署
    └── docker-compose.yml
```

---

## 使用方式

### 1. 每日一键分析

```bash
# 完整流程 (LLM模式)
python scripts/run_daily_analysis.py

# 快速规则引擎 (几秒完成)
python scripts/run_daily_analysis.py --no-llm

# 指定标的
python scripts/run_daily_analysis.py --symbols sh.600519,sz.300750,sz.002415
```

### 2. 模拟交易

```bash
# 完整交易日流程 (分析→买入→总结)
python -m simulation.daily_runner

# 早盘买入
python scripts/morning_buy.py

# 盘后卖出检查
python scripts/evening_sell.py

# 盘后总结
python scripts/evening_summary.py --summarize
```

### 3. Chat Agent (自然语言交互)

```bash
# 启动 API
uvicorn api.server:app --host 0.0.0.0 --port 8000

# 对话
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"message": "茅台怎么样？"}'
```

> **关于 API 认证**：`API_KEY` 不是 AI 模型密钥，而是**你自己定义**的一串密码，用于保护本服务的接口，调用时经 `X-API-Key` 请求头传递（如上例）。
> 安全默认（fail-closed）：未设置 `API_KEY` 时，除 `/health` 外所有请求返回 403；本地开发可在 `.env` 设置 `API_ALLOW_INSECURE_NO_AUTH=true` 放开无认证访问。

### 4. 策略回测对比

```bash
python scripts/backtest_compare.py --years 3 --symbols 10
```

### 5. 定时调度

```bash
python scripts/run_scheduler.py  # 后台常驻, 自动执行盘前/盘中/盘后任务
```

### 6. kill-test — 三路信号对比 (验证信号是否有 alpha)

```bash
# 规则臂 vs 随机臂, 回溯 500 日
python scripts/killtest.py --history 500 --arms rule,random

# 指定股票池 + N日后回看
python scripts/killtest.py --universe sh.600519,sz.000858,sz.300750 --lookahead 10

# 含 LLM 臂 (读取每日分析产物, 随日跑积累)
python scripts/killtest.py --arms rule,random,llm --seed 7
```

> **口径**：方向化 N 日前向收益 (BUY=做多, SELL=做空, 正=方向正确)。p<0.05 表示显著优于随机/全市场基线。LLM 臂需要每天真实跑分析积累信号，rule/random 可回溯历史立即出结果。

### 7. RAG 检索

```bash
# 语义检索参考文档
python -c "from knowledge.manager import KnowledgeManager; km=KnowledgeManager(); print(km.rag_query('震荡市适合什么均线策略'))"

# K线形态相似检索 (注入辩论上下文)
python -c "from knowledge.manager import KnowledgeManager; km=KnowledgeManager(); print(km.search_similar_klines('十字星 放量', top_k=3))"
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 数据 | AKShare / Baostock / 东方财富 / 腾讯行情 / Tushare |
| 数据库 | PostgreSQL + TimescaleDB / Redis / ChromaDB |
| 分析 | pandas-ta / scikit-learn / numpy / scipy |
| LLM | DeepSeek V4-Flash (统一) |
| 编排 | LangGraph |
| 回测 | 自研事件驱动引擎 |
| Web | FastAPI + Streamlit |
| 部署 | Docker Compose |

---

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | 否 | API 地址 (默认 https://api.deepseek.com/v1) |
| `DEEPSEEK_FLASH_MODEL` | 否 | 模型名 (默认 deepseek-v4-flash; v3.0 起全量统一 flash) |
| `API_KEY` | 否 | REST API 认证密码 (自定义, 经 `X-API-Key` 请求头传递; 未设置时默认拒绝访问) |
| `API_ALLOW_INSECURE_NO_AUTH` | 否 | 未设 API_KEY 时允许无认证访问 (仅本地开发, 默认 false) |
| `CORS_ORIGINS` | 否 | CORS 来源白名单 (逗号分隔, 默认 `*`) |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | 否 | 数据库连接 |
| `REDIS_HOST/PORT/DB` | 否 | Redis 连接 |

---

## 项目状态 (v3.1 — Beta)

- [x] **Phase 1**: 数据基础层 — 8 数据源 + 多源路由 + 降级 + 缓存 + 停牌清洗
- [x] **Phase 2**: 分析引擎层 — 130+ 指标 + 6 市场状态 + 9 策略 + 板块轮动 + 北向资金
- [x] **Phase 3**: 回测引擎层 — 事件驱动 + A股券商模拟 + 6层过拟合防控 (CSCV-PBO/DSR) + 冲击成本
- [x] **Phase 3.5**: 知识库 + RAG — System Prompts + 规则 YAML + ChromaDB 双通道余弦检索 (语义 + K线形态)
- [x] **Phase 4**: AI Agent 层 — LangGraph 编排 + 多视角辩论 + Chat Agent + 统一 V4-Flash 模型
- [x] **Phase 4.5**: 代码即推理 — Python 计算 + LLM 叙事 + NumericSafetyChecker
- [x] **Phase 4.6** 🆕: 多智能体协作 (v3.1) — DeerFlow 模式 (Brainstormer 发散 / Evaluator 反思回炉 / DecisionValidator 执行前校验)
- [x] **Phase 4.7** 🆕: kill-test 信号验证 — 三路对比 (LLM/规则/随机) + Welch t 显著性
- [x] **Phase 5**: 模拟交易 — Paper Trading + 8层风控 + 每日自动化 + 定时调度
- [ ] **Phase 6**: 生产化 — 长期信号跟踪 (kill-test 数据积累) + 性能基准 + 监控面板 (进行中)

## 诚实声明

- **工程质量 ~90%**：260 离线测试 + CI 硬门槛 + 幻觉治理 + 过拟合检测，多智能体协作与 RAG 已真正生效
- **投资可行性未证实**：kill-test 框架已就位，但信号 alpha 需要 1-3 个月日跑积累三路对比数据才能下结论。规则臂在当前小样本 (4股×120日) 上 p=0.553，**未显著优于随机**——这正是框架存在的意义。
- **已知遗留**：LLM 臂需日跑积累历史信号；mootdx (akshare 传递依赖) 与 httpx>=0.27 冲突不可调和 (已注释)；免费数据源 (Baostock/Tencent) 长期可靠性未验证。

---

## License

MIT — 个人研究用途。**⚠️ 风险提示：历史数据不代表未来表现，本项目仅供研究和学习，不构成投资建议。**
