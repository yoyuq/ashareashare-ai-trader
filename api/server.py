"""
FastAPI REST API — A股智能分析Agent

端点:
  GET  /health              — 健康检查
  GET  /api/v1/stock/{symbol} — 单股行情
  POST /api/v1/analyze      — 发起分析
  GET  /api/v1/analyze/{task_id} — 查询分析结果
  POST /api/v1/backtest     — 发起回测
  GET  /api/v1/strategies   — 策略列表
  GET  /api/v1/market/regime — 当前市场状态
  GET  /api/v1/cost/summary  — 模型成本
"""

import asyncio
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
from analysis.regime import get_regime_parameters

# ═══════════════════════════════════════════════════════════════
# 应用初始化
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="A股智能分析Agent API",
    version="2.14.0",
    description="AI驱动的A股量化分析助手 — REST API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔐 v2.14: API Key 认证中间件 (可选 — 未配置 API_KEY 时自动放行)
_API_KEY = os.getenv("API_KEY", "")
_AUTH_WHITELIST = {"/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Simple API Key authentication. Skips auth if API_KEY is not set."""
    if not _API_KEY or request.url.path in _AUTH_WHITELIST:
        return await call_next(request)

    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if api_key != _API_KEY:
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized", "detail": "Invalid or missing API key"},
        )
    return await call_next(request)

# 组件引用 (延迟初始化)
_router = None
_knowledge = None
_analyzer = None
_detector = None
_chat_agent = None
_tasks: Dict[str, Dict] = {}
_sessions: Dict[str, List[Dict]] = {}  # v2.8: 会话存储


def get_router():
    global _router
    if _router is None:
        from data.router import get_data_router
        _router = get_data_router()
    return _router


def get_knowledge():
    global _knowledge
    if _knowledge is None:
        from knowledge.manager import KnowledgeManager
        _knowledge = KnowledgeManager()
    return _knowledge


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from analysis.indicators import TechnicalAnalyzer
        _analyzer = TechnicalAnalyzer()
    return _analyzer


def get_detector():
    global _detector
    if _detector is None:
        from analysis.regime import MarketRegimeDetector
        _detector = MarketRegimeDetector()
    return _detector


# ═══════════════════════════════════════════════════════════════
# Pydantic 模型
# ═══════════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=10,
                                description="待分析标的列表,如 ['sh.600000', 'sz.000001']")
    task_type: str = Field(default="stock_analysis",
                           description="分析类型: daily_scan / stock_analysis / strategy_backtest")
    use_debate: bool = Field(default=True, description="是否启用多空辩论")


class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="标的代码")
    strategy_id: str = Field(..., description="策略ID")
    start_date: str = Field(default="2023-01-01")
    end_date: str = Field(default="2024-12-31")
    initial_capital: float = Field(default=100000, ge=10000)


class AnalyzeResponse(BaseModel):
    task_id: str
    status: str = "accepted"
    eta_seconds: int = 30


class TaskResult(BaseModel):
    task_id: str
    status: str  # pending / running / completed / failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """系统健康检查"""
    status = {
        "status": "healthy",
        "version": "2.14.0",
        "timestamp": datetime.now().isoformat(),
    }

    # 检查数据源
    try:
        router = get_router()
        ds_status = router.status
        status["data_sources"] = ds_status
    except Exception as e:
        status["data_sources"] = {"error": str(e)}

    # 检查Ollama
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        status["ollama"] = {
            "available": True,
            "models": [m.model for m in models.models] if hasattr(models, 'models') else [],
        }
    except Exception:
        status["ollama"] = {"available": False}

    return status


@app.get("/api/v1/stock/{symbol}")
async def get_stock_info(symbol: str):
    """获取单只股票的行情与技术指标"""
    router = get_router()
    analyzer = get_analyzer()

    from data.providers.base import DataFrequency, DataRequest

    try:
        req = DataRequest.recent(symbol, days=365)
        result = await router.get_daily_kline(req)

        if result.data.empty:
            raise HTTPException(404, f"未找到{symbol}的数据")

        # 计算指标
        indicator_result = analyzer.compute_all(result.data, symbol=symbol)
        last_row = indicator_result.to_dataframe().iloc[-1]

        # 关键指标
        key_metrics = {
            "symbol": symbol,
            "date": str(result.data["date"].iloc[-1]),
            "close": float(last_row.get("close", 0)),
            "ma_5": float(last_row.get("ma_5", 0)),
            "ma_20": float(last_row.get("ma_20", 0)),
            "ma_60": float(last_row.get("ma_60", 0)),
            "rsi_14": float(last_row.get("rsi_14", 0)),
            "macd_dif": float(last_row.get("macd_dif", 0)),
            "macd_hist": float(last_row.get("macd_hist", 0)),
            "bias_ma20": float(last_row.get("bias_ma20", 0)),
            "atr_14": float(last_row.get("atr_14", 0)),
            "trend_score": float(last_row.get("trend_score", 0)),
            "composite_score": float(last_row.get("composite_score", 0)),
            "vol_ratio": float(last_row.get("vol_ratio_5", 0)),
        }

        # 形态检测
        patterns = {
            k: int(v.iloc[-1]) if hasattr(v, 'iloc') else 0
            for k, v in indicator_result.patterns.items()
        }
        active_patterns = [k for k, v in patterns.items() if v > 0]
        key_metrics["active_patterns"] = active_patterns[:5]

        return key_metrics

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"查询失败: {e}")


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def start_analysis(req: AnalyzeRequest):
    """启动一次分析任务(异步)"""
    task_id = str(uuid.uuid4())[:8]

    _tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "request": req.model_dump(),
    }

    # 后台执行分析
    asyncio.create_task(_run_analysis(task_id, req))

    return AnalyzeResponse(task_id=task_id)


async def _run_analysis(task_id: str, req: AnalyzeRequest):
    """后台执行分析"""
    try:
        _tasks[task_id]["status"] = "running"

        from agent.orchestration.workflow import AnalysisWorkflow
        from knowledge.manager import KnowledgeManager

        km = KnowledgeManager()
        wf = AnalysisWorkflow(router=None, knowledge_manager=km)

        if req.task_type == "daily_scan":
            state = await wf.run_daily_scan(symbols=req.symbols)
        else:
            state = await wf.run_stock_analysis(symbol=req.symbols[0])

        _tasks[task_id]["result"] = {
            "report": state.get("final_report", ""),
            "regime": state.get("market_regime"),
            "suggestions": state.get("trade_suggestions", []),
            "debate": state.get("debate_result", {}),
        }
        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["completed_at"] = datetime.now().isoformat()
        logger.error(f"分析任务失败 {task_id}: {e}")


@app.get("/api/v1/analyze/{task_id}", response_model=TaskResult)
async def get_analysis_result(task_id: str):
    """查询分析任务结果"""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(404, f"任务不存在: {task_id}")
    return TaskResult(**task)


@app.post("/api/v1/backtest")
async def run_backtest(req: BacktestRequest):
    """运行回测"""
    from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
    from backtest.broker import AShareBroker
    from data.providers.base import DataRequest, DataFrequency

    router = get_router()

    # 拉取数据
    data_req = DataRequest(
        symbol=req.symbol,
        start_date=datetime.strptime(req.start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(req.end_date, "%Y-%m-%d").date(),
        frequency=DataFrequency.DAILY,
    )
    result = await router.get_daily_kline(data_req)

    if result.data.empty:
        raise HTTPException(404, f"未找到{req.symbol}的回测数据")

    # 回测
    config = BacktestConfig(
        initial_capital=req.initial_capital,
        start_date=datetime.strptime(req.start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(req.end_date, "%Y-%m-%d").date(),
    )

    engine = EventDrivenBacktestEngine(config)
    engine.load_data(req.symbol, result.data)

    # 使用真实的策略回测函数 (v2.8 修复)
    from analysis.recommender import STRATEGY_BACKTESTERS

    bt_func = STRATEGY_BACKTESTERS.get(req.strategy_id)
    if bt_func is None:
        raise HTTPException(400, f"未知策略: {req.strategy_id}, 可用: {list(STRATEGY_BACKTESTERS.keys())}")

    bt = bt_func(result.data)

    # 同时运行事件驱动回测获取完整指标
    def _event_strategy(today, bars, broker):
        sym = req.symbol
        if sym not in bars:
            return
        bar = bars[sym]
        close_val = bar["close"]
        pos = broker.account.positions.get(sym)

        if not hasattr(_event_strategy, "_last_action"):
            _event_strategy._last_action = {}
        last = _event_strategy._last_action.get(sym)

        # 使用策略的信号逻辑：基于回测函数的结果驱动
        if not hasattr(_event_strategy, "_signals"):
            _event_strategy._signals = bt.get("signals", 0)
            _event_strategy._signal_idx = 0

        # 简化版：根据信号数交替买卖
        if last != "buy" and _event_strategy._signal_idx < _event_strategy._signals:
            qty = int(broker.account.cash * 0.3 / close_val / 100) * 100
            if qty >= 100:
                broker.buy(sym, qty, price=close_val)
                _event_strategy._last_action[sym] = "buy"
                _event_strategy._signal_idx += 1
        elif last == "buy" and pos and pos.quantity > 0:
            broker.sell(sym, pos.quantity, price=close_val)
            _event_strategy._last_action[sym] = "sell"

    _event_strategy._last_action = {}
    _event_strategy._signals = bt.get("signals", 0)
    _event_strategy._signal_idx = 0
    backtest_result = engine.run(_event_strategy, progress_bar=False)

    strategy_name = (get_knowledge().get_strategy(req.strategy_id) or {}).get("name", req.strategy_id) if get_knowledge() else req.strategy_id

    return {
        "symbol": req.symbol,
        "strategy": req.strategy_id,
        "strategy_name": strategy_name,
        "period": f"{req.start_date} → {req.end_date}",
        # 事件驱动回测结果
        "total_return_pct": backtest_result.total_return,
        "annual_return_pct": backtest_result.annual_return,
        "sharpe": backtest_result.sharpe_ratio,
        "sortino": backtest_result.sortino_ratio,
        "max_drawdown_pct": backtest_result.max_drawdown,
        "total_trades": backtest_result.total_trades,
        "win_rate": backtest_result.win_rate,
        "var_95": backtest_result.var_95,
        "cvar_95": backtest_result.cvar_95,
        # 策略真实回测统计
        "strategy_backtest": {
            "signals": bt.get("signals", 0),
            "win_rate": bt.get("win_rate", 0),
            "profit_factor": bt.get("profit_factor", 1),
            "expected_value": bt.get("expected_value", 0),
            "sharpe": bt.get("sharpe", 0),
            "max_dd": bt.get("max_dd", 0),
        },
    }


@app.get("/api/v1/strategies")
async def list_strategies():
    """获取所有可用策略"""
    km = get_knowledge()
    strategies = km.list_strategies()
    return {
        "total": len(strategies),
        "strategies": [
            {
                "id": s["id"],
                "name": s["name"],
                "category": s["category"],
                "description": s.get("description", ""),
                "regimes": s.get("market_regimes", []),
                "capacity_limit": s.get("capacity_limit", 0),
            }
            for s in strategies
        ],
    }


@app.get("/api/v1/market/regime")
async def market_regime():
    """获取当前市场状态"""
    router = get_router()
    detector = get_detector()

    from data.providers.base import DataFrequency, DataRequest

    try:
        req = DataRequest.recent("sh.000300", days=365)
        result = await router.get_daily_kline(req)

        if result.data.empty:
            return {"regime": "unknown", "error": "无法获取沪深300数据"}

        regime = detector.detect(result.data)
        params = get_regime_parameters(regime.regime)

        return {
            "regime": regime.regime.value,
            "confidence": regime.confidence,
            "details": regime.details,
            "suggested_params": params,
        }
    except Exception as e:
        raise HTTPException(500, f"市场状态检测失败: {e}")


def get_chat_agent():
    """ChatAgent 懒加载 (v2.9: 使用 ModelRouter)"""
    global _chat_agent
    if _chat_agent is None:
        try:
            from models.router import ModelRouter
            model_router = ModelRouter()
        except Exception:
            model_router = None
        from agent.chat_agent import ChatAgent
        _chat_agent = ChatAgent(
            router=get_router(),
            knowledge=get_knowledge(),
            analyzer=get_analyzer(),
            model_router=model_router,
        )
        logger.info("ChatAgent 初始化完成 (ModelRouter 三层漏斗)")
    return _chat_agent


@app.get("/api/v1/cost/summary")
async def cost_summary():
    """模型API成本汇总"""
    try:
        from models.cost_monitor import CostMonitor
        monitor = CostMonitor()
        return monitor.daily_report()
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# 🆕 v2.8: Chat 端点
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    session_id: str = Field(default="default", description="会话ID, 用于多轮对话")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tool_calls: List[str] = Field(default_factory=list)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    🆕 v2.8 对话式AI端点

    支持多轮对话: 传入相同的 session_id 即可保持上下文。
    新会话: 不传 session_id 或传新值。

    示例:
      curl -X POST http://localhost:8000/api/v1/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "茅台怎么样？", "session_id": "user1"}'
    """
    try:
        agent = get_chat_agent()
        reply = await agent.chat(
            user_message=req.message,
            session_id=req.session_id,
        )

        # 获取工具调用记录 (最后一条assistant消息的tool_calls)
        tool_calls = []
        conv = agent.conversations.get(req.session_id, [])
        for msg in reversed(conv):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = [tc.get("function", {}).get("name", "unknown")
                             for tc in msg.get("tool_calls", [])]
                break

        return ChatResponse(
            reply=reply,
            session_id=req.session_id,
            tool_calls=tool_calls,
        )
    except Exception as e:
        logger.error(f"Chat失败: {e}")
        raise HTTPException(500, f"对话处理失败: {e}")


@app.get("/api/v1/chat/history")
async def chat_history(session_id: str = "default"):
    """获取会话历史"""
    agent = get_chat_agent()
    conv = agent.conversations.get(session_id, [])
    # 只返回用户和assistant消息, 不返回system/tool
    history = [
        {"role": m["role"], "content": m.get("content", "")[:500]}
        for m in conv
        if m["role"] in ("user", "assistant") and m.get("content")
    ]
    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": history[-20:],  # 最近20条
    }


@app.delete("/api/v1/chat/history")
async def clear_chat_history(session_id: str = "default"):
    """清除会话历史"""
    agent = get_chat_agent()
    if session_id in agent.conversations:
        # 保留system prompt
        system_msgs = [m for m in agent.conversations[session_id]
                       if m["role"] == "system"]
        agent.conversations[session_id] = system_msgs
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# ═══════════════════════════════════════════════════════════════
# 🆕 v2.14: 风控 & 组合 & 绩效 API
# ═══════════════════════════════════════════════════════════════

@app.get("/api/v1/risk/status")
async def risk_status():
    """获取当前组合风控状态 (回撤/波动/断路器/8层风控)"""
    try:
        from simulation.portfolio import PortfolioManager
        from analysis.risk_controls import PortfolioRiskManager
        import pandas as pd

        manager = PortfolioManager()
        state = manager.state
        risk_mgr = PortfolioRiskManager(initial_capital=state.initial_capital)

        # 构建日收益率
        daily_rets = pd.Series(dtype=float)
        if len(state.daily_snapshots) >= 5:
            daily_rets = pd.Series([
                s.daily_return_pct / 100 for s in state.daily_snapshots[-60:]
            ])

        risk_state = risk_mgr.update(state.total_value, daily_rets) if len(daily_rets) > 0 else None

        # 8层风控逐层检查
        layers = {}
        # L1
        layers["L1_circuit_breaker"] = {
            "active": risk_state.circuit_breaker_active if risk_state else False,
            "drawdown_pct": risk_state.drawdown_pct if risk_state else 0,
        }
        # L2-L4
        layers["L2_atr_stop"] = {"status": "active"}
        layers["L3_trailing_stop"] = {"status": "active"}
        layers["L4_position_limits"] = {"max_positions": 8, "single_pct": 0.20}

        # L5: 防踩踏
        if state.positions:
            losers = sum(1 for p in state.positions.values() if p.unrealized_pnl_pct < -3)
            layers["L5_stampede"] = {
                "triggered": losers / len(state.positions) >= 0.5 if state.positions else False,
                "losers": losers,
                "total": len(state.positions),
            }
        else:
            layers["L5_stampede"] = {"triggered": False, "note": "no positions"}

        # L6: 持仓天数
        max_days = 0
        for p in state.positions.values():
            if p.buy_date:
                try:
                    days = (date.today() - date.fromisoformat(p.buy_date)).days
                    max_days = max(max_days, days)
                except (ValueError, TypeError):
                    pass
        layers["L6_holding_days"] = {"max_days": max_days, "warning": max_days >= 20}

        # L7-L8
        layers["L7_limit_down"] = {"status": "monitoring"}
        layers["L8_correlation"] = {"status": "monitoring"}

        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio": {
                "total_value": round(state.total_value, 2),
                "cash": round(state.cash, 2),
                "position_value": round(state.position_value, 2),
                "total_return_pct": round(state.total_return_pct, 2),
                "positions_count": len(state.positions),
                "trade_count": state.trade_count,
                "win_rate": round(state.win_rate * 100, 1),
            },
            "risk": {
                "drawdown_pct": risk_state.drawdown_pct if risk_state else 0,
                "volatility_pct": risk_state.current_volatility if risk_state else 0,
                "risk_multiplier": risk_state.risk_multiplier if risk_state else 1.0,
                "circuit_breaker_active": risk_state.circuit_breaker_active if risk_state else False,
                "crisis_mode": risk_state.crisis_mode if risk_state else False,
                "suggested_exposure_pct": risk_state.suggested_exposure_pct if risk_state else 0.3,
                "warnings": risk_state.warning_messages if risk_state else [],
            },
            "layers": layers,
        }
    except Exception as e:
        raise HTTPException(500, f"风控状态查询失败: {e}")


@app.get("/api/v1/portfolio/summary")
async def portfolio_summary():
    """获取模拟交易组合摘要 (含持仓/交易/绩效)"""
    try:
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine

        manager = PortfolioManager()
        engine = PaperTradingEngine(manager)
        return engine.get_summary()
    except Exception as e:
        raise HTTPException(500, f"组合摘要查询失败: {e}")


@app.get("/api/v1/trades/stats")
async def trade_statistics():
    """获取交易绩效统计 (WinRateAnalyzer — 胜率/盈亏比/分场景/信号质量)"""
    try:
        from simulation.portfolio import PortfolioManager
        from analysis.winrate import WinRateAnalyzer
        import pandas as pd

        manager = PortfolioManager()
        state = manager.state

        if not state.trade_history:
            return {"total_trades": 0, "note": "无交易记录"}

        # 转换交易记录
        trades_df = pd.DataFrame([
            {
                "date": t.date, "symbol": t.symbol, "side": t.side,
                "price": t.price, "pnl": t.pnl or 0,
                "pnl_pct": t.pnl_pct or 0, "holding_days": 1,
            }
            for t in state.trade_history
            if t.side == "sell" and t.pnl is not None
        ])

        if trades_df.empty:
            return {"total_trades": 0, "note": "无已平仓交易"}

        analyzer = WinRateAnalyzer()
        report = analyzer.analyze(trades_df, pd.DataFrame())

        return {
            "total_trades": report.total_signals,
            "win_count": report.total_win,
            "win_rate": round(report.overall_win_rate * 100, 1),
            "profit_factor": report.profit_factor,
            "expected_value_pct": report.expected_value,
            "avg_holding_days": round(report.avg_holding_days, 1),
            "max_consecutive_loss": report.max_consecutive_loss,
            "signal_efficiency": report.signal_efficiency,
            "scenario_breakdown": {
                k: {
                    "win_rate": round(v["win_rate"] * 100, 1),
                    "trades": v["total"],
                    "profit_factor": v["profit_factor"],
                }
                for k, v in report.scenario_breakdown.items()
                if v["total"] > 0
            },
        }
    except Exception as e:
        raise HTTPException(500, f"交易统计查询失败: {e}")


@app.get("/api/v1/system/benchmark")
async def system_benchmark():
    """快速系统性能基准 (不含数据拉取)"""
    import time, numpy as np, pandas as pd

    rng = np.random.default_rng(42)
    close = 50 + np.cumsum(rng.normal(0.02, 1.5, 500))
    df = pd.DataFrame({
        "open": close - 0.3, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": rng.integers(100000, 5000000, 500).astype(float),
    })

    results = {}

    # 指标计算
    from analysis.indicators import TechnicalAnalyzer
    analyzer = TechnicalAnalyzer()
    t0 = time.perf_counter()
    analyzer.compute_all(df)
    results["indicators_500bars_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # 策略回测
    from analysis.recommender import STRATEGY_BACKTESTERS
    strat_times = {}
    for sid, bt in STRATEGY_BACKTESTERS.items():
        t0 = time.perf_counter()
        bt(df)
        strat_times[sid] = round((time.perf_counter() - t0) * 1000, 2)
    results["strategies_ms"] = strat_times

    # 信号分级
    from analysis.risk_controls import SignalGrader
    t0 = time.perf_counter()
    for _ in range(1000):
        SignalGrader.grade(65, 0.55, "range_bound")
    results["signal_grader_1000x_us"] = round((time.perf_counter() - t0) * 1000, 1)

    return {"status": "ok", "results": results}


# ═══════════════════════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
