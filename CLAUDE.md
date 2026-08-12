# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

```bash
# Install (dev mode)
pip install -e ".[dev,backtest]"

# Run tests
pytest tests/unit/ -v                          # Unit tests only
pytest tests/test_smoke.py -v                  # Smoke tests
pytest tests/ -m "not network"                 # All tests except network-dependent
pytest tests/ -m network                       # Network tests only (needs API keys)

# Lint
ruff check . --select=E,F,W --ignore=E501

# Type check (disabled in CI, run manually)
mypy agent/ analysis/ data/ models/ --ignore-missing-imports

# Start services
streamlit run web/dashboard.py                 # Dashboard at :8501
uvicorn api.server:app --port 8000              # REST API at :8000

# Daily operations
python scripts/run_daily_analysis.py --no-llm   # Quick rule-engine analysis
python scripts/run_daily_analysis.py            # Full LLM analysis
python -m simulation.daily_runner               # Full day: analyze → trade → summarize
python -m simulation.daily_runner --dry-run     # Analyze only, skip trades
python scripts/backtest_compare.py --years 3     # Strategy backtest comparison
```

## Architecture Overview

### Stock Symbol Convention

Stocks use the format `{market}.{code}`: `sh.600519` (Shanghai), `sz.000858` (Shenzhen), `bj.8xxxxx` (Beijing). Dashboard selectors display the short code (`600519`) and resolve to the full symbol internally. The `scripts/shared.py::NAME_MAP` provides code-to-name lookups.

### Data Pipeline

```
AKShare (live) → Tencent (realtime) → EastMoney → Baostock (historical K-line)
                    ↓ (automatic fallback on failure)
              DataRouter → cache → provider downgrade (5-min cooldown after 3 failures)
```

- **Baostock** is the most reliable source — no proxy needed, works outside China, provides daily K-line with PE/PB/volume
- **Tencent** (`qt.gtimg.cn`) provides **real-time quotes** (price, change%, volume, turnover, PE, PB) without proxy — used for dashboard live data
- **EastMoney/AKShare** require a China proxy (`HTTP_PROXY=http://127.0.0.1:7897`) and are flaky outside trading hours
- DataRoute caches real-time quotes for 3s, daily K-line for 1h, stock lists for 24h (Redis) or 5min (in-memory fallback)
- `simulation_data/full_market_cache.json` is the full-market snapshot consumed by the dashboard

### AI Model Routing (3-Tier Funnel)

```
100% → DeepSeek V4-Flash (¥1-2/M tokens) — 全量统一模型 (v3.0, 2026-08)
```

`models/router.py` routes every task to `deepseek-v4-flash` (single-model, v3.0; Ollama local + V4-Pro tiers removed). Budget tracking default ¥1/day; DeepSeek peak-valley pricing (9-12h/14-18h ×2) reflected in cost model. API keys: `DEEPSEEK_API_KEY` in `.env`. The dashboard AI analysis button uses `deepseek-v4-flash` directly via `openai.OpenAI()` client with `base_url=https://api.deepseek.com/v1`.

### Analysis Pipeline (LangGraph)

```
Data Prep → Technical Analysis → Market Scanner → Strategy Match → Backtest
    → Bull/Bear Debate → Critic Audit → Judge → Synthesis → Decision Log (SQLite)
```

Each node is a sub-agent in `agent/sub_agents/`. The workflow runs via `AnalysisWorkflow` in `agent/orchestration/workflow.py`, sharing state through `MarketAnalysisState` (TypedDict in `agent/orchestration/state.py`). System prompts live in `knowledge/prompts/system/`.

**Sub-agent conventions**: All agents extend `BaseAgent` (`agent/sub_agents/base.py`) and return `AgentResult(name, content, confidence, metadata)`. The `agent/sub_agents/__init__.py` module provides both a hardcoded registry and a configurable `ConfigurableAgentRegistry` factory.

**Code-as-Reasoning**: `agent/tools/code_executor.py` implements a pattern where LLM creates an analysis plan → Python computes all numeric values → LLM narrates from locked numbers. `NumericSafetyChecker` validates no hallucinated figures appear in reports.

### Simulation & Paper Trading

- `simulation/daily_runner.py` — orchestrates daily flow: Phase 1 (scan+analyze markets), Phase 2 (execute buy/sell), Phase 3 (snapshot+summarize)
- `simulation/paper_trader.py` — paper trading engine with T+1 settlement, price limits, A-shares fees (commission ¥3/10k, stamp duty ¥5/10k sell-only, transfer fee ¥0.1/10k = 0.00001)
- `simulation/portfolio.py` — JSON-persisted portfolio at `simulation_data/portfolio.json`, contains `daily_snapshots` array used for equity curve and drawdown charts
- `agent/sub_agents/validator.py` — DecisionValidator: pre-execution hard-constraint checks (price limits, T+1, position caps, lot size, industry concentration) + rejection journal at `simulation_data/validation_journal.jsonl`

### Dashboard Architecture (Streamlit)

`web/dashboard.py` is a single-file multi-tab Streamlit app with a GitHub-dark CSS theme (`#0d1117` base). Key design decisions:
- **All `st.dataframe` calls replaced** with a custom `render_dataframe()` HTML table function — Streamlit 1.60's GlideDataEditor conflicts with the dark theme CSS, making table text invisible
- Stock selectors use `load_full_market_stocks()` (5100+ stocks from cache) with Streamlit's native search-as-you-type
- Live data comes from Tencent (`qt.gtimg.cn`), cached to `simulation_data/full_market_cache.json`
- API backend at `api/server.py` has 20+ endpoints under `/api/v1/` with `X-API-Key` auth

### Backtest Engine

Event-driven engine in `backtest/engine.py` with a realistic A-shares broker (`backtest/broker.py`) that models T+1, price limits (10%/20%/30%), trade suspensions, lot-size constraints, and fees. Overfitting detection in `backtest/overfitting.py` uses 6 methods: time-split, PBO, Deflated SR, walk-forward, parameter sensitivity, Monte Carlo. Results stored in `reports/strategy_backtest.json`.

### Knowledge Base

Three-layer system managed by `knowledge/manager.py`:
1. **Structured YAML rules** (`knowledge/rules/`, `knowledge/strategies/registry.yaml`) — trading rules, indicator guide, strategy registry
2. **Reference markdown** (`knowledge/reference/`) — glossary (~45 terms), market cycle history (7 bull/bear cycles), fundamental analysis thresholds
3. **ChromaDB vector store** — K-line pattern similarity search, RAG semantic retrieval

### Dashboard Dark Theme CSS

The CSS at the top of `dashboard.py` uses `!important` broadly. When debugging UI issues:
- DataFrames: use `render_dataframe()` not `st.dataframe` (GlideDataEditor incompatibility)
- The `[data-testid="stStatusWidget"]` spinner customization hides the default running-man SVG and replaces it with a CSS `::before` pseudo-element spinning circle
- All metric cards, tabs, expanders, alerts are styled with `#161b22` background, `#30363d` borders, `#c9d1d9` text

### Proxy Configuration

China stock APIs (EastMoney, AKShare) may need a proxy for non-China IPs. The proxy at `HTTP_PROXY=http://127.0.0.1:7897` is used by curl and Python requests. When proxy is down, fall back to:
- Tencent API for real-time quotes (no proxy needed, `qt.gtimg.cn`)
- Baostock for historical data (no proxy needed)

### Key Module Map

| Task | Entry Point |
|------|------------|
| Full daily workflow | `simulation/daily_runner.py` |
| Single-stock deep analysis | `agent/orchestration/workflow.py::AnalysisWorkflow` |
| Dashboard | `web/dashboard.py` (Streamlit) |
| REST API | `api/server.py` (FastAPI) |
| Strategy backtest | `backtest/engine.py` or `scripts/backtest_compare.py` |
| Market regime detection | `analysis/regime.py` |
| Technical indicators | `analysis/indicators.py::TechnicalAnalyzer` |
| AI conversation | `agent/chat_agent.py::ChatAssistant` |
| Data access | `data/router.py::DataRouter` |
| Model routing | `models/router.py::ModelRouter` |
| Risk evaluation | `analysis/risk_controls.py::PortfolioRiskManager` |
| Knowledge/strategies | `knowledge/manager.py::KnowledgeManager` |
| Live real-time data fetch | `data/providers/tencent_provider.py` (or curl `qt.gtimg.cn`) |
