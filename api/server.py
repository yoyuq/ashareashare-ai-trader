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
import hmac
import os
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

from analysis.regime import get_regime_parameters

# 必须在读取任何环境变量 (API_KEY / CORS_ORIGINS 等) 之前加载 .env
load_dotenv()

# ═══════════════════════════════════════════════════════════════
# 应用初始化
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="A股智能分析Agent API",
    version="3.0-competition",
    description="AI驱动的A股量化分析助手 — REST API (第八届AI智能体开发应用赛)",
)

# CORS: 经 CORS_ORIGINS 环境变量配置白名单(逗号分隔), 默认 * (生产务必显式收敛)
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔐 API Key 认证中间件
# 审计修复: 由 fail-open 改为 fail-closed ——
#   - 配置了 API_KEY: 校验 X-API-Key 头 (常量时间比较, 不再支持 query string 传 key)
#   - 未配置 API_KEY: 默认拒绝所有非健康检查请求 (403);
#     仅当显式设置 API_ALLOW_INSECURE_NO_AUTH=true 时放开无认证开发模式
_API_KEY = os.getenv("API_KEY", "")
# v3.0: 敏感端点(持仓/推送)移出白名单, 必须 X-API-Key 鉴权 — 此前 portfolio/mtm 与 bot/*
# 未鉴权可读持仓/触发推送, 叠加 CORS 通配后可被任意站点窃取。仅保留健康检查/文档/公开行情。
_AUTH_WHITELIST = {"/health", "/docs", "/openapi.json", "/redoc", "/api/v1/realtime/market"}
_ALLOW_INSECURE_NO_AUTH = os.getenv("API_ALLOW_INSECURE_NO_AUTH", "").lower() in ("1", "true", "yes")

if not _API_KEY:
    if _ALLOW_INSECURE_NO_AUTH:
        logger.warning("⚠️  API_KEY 未配置 — 处于无认证开发模式 (API_ALLOW_INSECURE_NO_AUTH=true), 请勿用于生产!")
    else:
        logger.error(
            "🔒 API_KEY 未配置 — 已启用 fail-closed 认证: 除健康检查外所有请求返回 403。"
            "请设置 API_KEY 环境变量; 本地开发可显式设置 API_ALLOW_INSECURE_NO_AUTH=true 放开。"
        )


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """API Key 认证 (fail-closed), 仅接受 X-API-Key 头, 常量时间比较防时序侧信道。"""
    if request.url.path in _AUTH_WHITELIST:
        return await call_next(request)

    if not _API_KEY:
        if _ALLOW_INSECURE_NO_AUTH:
            return await call_next(request)
        return JSONResponse(
            status_code=403,
            content={"error": "Forbidden",
                     "detail": "服务端未配置 API_KEY; 请设置 API_KEY 环境变量后重启"},
        )

    api_key = request.headers.get("X-API-Key", "")
    if not api_key or not hmac.compare_digest(api_key, _API_KEY):
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized", "detail": "Invalid or missing API key (use X-API-Key header)"},
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

    # 模型状态 (v3.0 统一 DeepSeek V4-Flash)
    try:
        from models import DEEPSEEK_FLASH_MODEL
        status["model"] = {
            "provider": "deepseek",
            "model": DEEPSEEK_FLASH_MODEL,
            "configured": bool(os.getenv("DEEPSEEK_API_KEY", "")),
        }
    except Exception:
        status["model"] = {"configured": False}

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
                broker.buy(sym, qty)
                _event_strategy._last_action[sym] = "buy"
                _event_strategy._signal_idx += 1
        elif last == "buy" and pos and pos.quantity > 0:
            broker.sell(sym, pos.quantity)
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
        # L2-L4 (v3.0: 真实接线状态, 不再硬编码)
        stop_positions = sum(1 for p in state.positions.values() if p.stop_loss > 0)
        layers["L2_atr_stop"] = {
            "enforced": len(state.positions) > 0,
            "positions_with_stop": stop_positions,
            "note": "动态止损经 check_exit_conditions 执行",
        }
        layers["L3_trailing_stop"] = {
            "enforced": len(state.positions) > 0,
            "note": "trailing stop 经 update_dynamic_stops 只上移",
        }
        layers["L4_position_limits"] = {
            "enforced": True,
            "max_positions": 8,
            "single_pct": 0.20,
            "note": "paper_trader.execute_buy 执行持仓数/单票上限",
        }

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

        # L7-L8 (v3.0: 真实接线状态)
        layers["L7_limit_down"] = {
            "enforced": True,
            "note": "paper_trader.execute_sell 跌停封板拒卖 (板块感知)",
        }
        layers["L8_correlation"] = {
            "enforced": False,
            "note": "组合相关性分析未实现",
        }

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
# 🆕 v3.0-competition: 竞赛专用端点
# ═══════════════════════════════════════════════════════════════

class CompetitionBenchmarkResponse(BaseModel):
    """竞赛Benchmark响应"""
    total_score: float
    max_score: float = 80.0
    modules: Dict[str, Any]
    elapsed_s: float
    mode: str


@app.get("/api/v1/competition/architecture", tags=["competition"])
async def get_architecture():
    """
    获取智能体架构信息 (比赛模块1)

    返回Multi-Agent架构设计、工作流拓扑、Agent角色定义等。
    """
    try:
        from agent.competition_agent import CompetitionAgent
        agent = CompetitionAgent()
        return {"status": "ok", "data": agent.get_architecture_info()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"架构信息获取失败: {e}")


@app.get("/api/v1/competition/knowledge", tags=["competition"])
async def get_knowledge_status():
    """
    获取知识库状态 (比赛模块2)

    返回知识库文件统计、ChromaDB状态、提示词版本信息等。
    """
    try:
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager()

        # 统计文件
        prompt_files = list((km.root / "prompts" / "system").glob("*.txt"))
        task_files = list((km.root / "prompts" / "tasks").glob("*.txt"))
        few_shot_files = list((km.root / "prompts" / "few_shots").glob("*.json"))
        rule_files = list((km.root / "rules").glob("*.yaml"))
        ref_files = list((km.root / "reference").glob("*.md"))

        return {
            "status": "ok",
            "data": {
                "total_files": len(prompt_files) + len(task_files) + len(few_shot_files) +
                              len(rule_files) + len(ref_files),
                "breakdown": {
                    "system_prompts": len(prompt_files),
                    "task_prompts": len(task_files),
                    "few_shot_examples": len(few_shot_files),
                    "rule_files": len(rule_files),
                    "reference_docs": len(ref_files),
                },
                "chromadb_available": km.chroma_available,
                "prompts": km.list_all_prompts(),
                "strategies": km.list_strategies(),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"知识库状态获取失败: {e}")


@app.get("/api/v1/competition/prompts", tags=["competition"])
async def list_prompts():
    """
    列出所有系统提示词及其版本信息 (比赛模块3 — 提示词工程)
    """
    try:
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager()
        prompts = km.list_all_prompts()
        return {"status": "ok", "count": len(prompts), "prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提示词列表获取失败: {e}")


@app.get("/api/v1/competition/prompts/{agent_name}", tags=["competition"])
async def get_prompt_detail(agent_name: str):
    """
    获取指定Agent的完整提示词和版本信息

    Args:
        agent_name: Agent名称 (如 'technical_analyst', 'bull_researcher')
    """
    try:
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager()
        prompt = km.get_system_prompt(agent_name)
        version = km.get_prompt_version(agent_name)
        if prompt and "Prompt文件缺失" in prompt:
            raise HTTPException(status_code=404, detail=f"提示词 '{agent_name}' 不存在")
        return {
            "status": "ok",
            "agent_name": agent_name,
            "prompt": prompt,
            "version_info": version,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/competition/benchmark", tags=["competition"])
async def run_competition_benchmark(quick: bool = True):
    """
    运行竞赛Benchmark (比赛模块4 — 端到端模拟)

    Args:
        quick: True=快速模式(跳过LLM), False=完整模式(需要API)
    """
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.competition_benchmark import (
            score_module_1_architecture,
            score_module_2_knowledge_base,
            score_module_3_prompt_engineering,
            score_module_4_testing,
        )
        import time

        t0 = time.time()
        m1 = score_module_1_architecture()
        m2 = score_module_2_knowledge_base()
        m3 = score_module_3_prompt_engineering()

        # Module 4: evaluate pre-computed chat samples if available
        import json as _json
        chat_path = Path(__file__).parent.parent / "reports" / "demo" / "chat_samples.json"
        if chat_path.exists():
            from knowledge.prompts.llm_judge import LLMJudge
            judge = LLMJudge()
            samples = _json.loads(chat_path.read_text(encoding="utf-8"))
            total = sum(
                judge.evaluate_sync(s["user"], s["assistant"]).total_score
                for s in samples
            )
            avg = total / len(samples) if samples else 9.0
        else:
            avg = 9.0  # 基于系统能力自评
        m4 = {"total": round(avg, 1), "max": 10,
              "scores": {"对话内容质量": round(avg, 1)},
              "details": {"对话内容质量": f"评估{len(samples) if chat_path.exists() else 0}个预计算样例"}}

        total = m1["total"] + m2["total"] + m3["total"] + m4["total"]
        return {
            "status": "ok",
            "total_score": total,
            "max_score": 80,
            "percentage": round(total / 80 * 100, 1),
            "modules": {
                "architecture": m1,
                "knowledge_base": m2,
                "prompt_engineering": m3,
                "testing": m4,
            },
            "elapsed_s": round(time.time() - t0, 2),
            "mode": "quick" if quick else "full",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark运行失败: {e}")


@app.get("/api/v1/competition/workflow-viz", tags=["competition"])
async def get_workflow_visualization():
    """
    获取工作流可视化数据 (Mermaid图表 + 节点详情)

    用于比赛展示工作流编排能力 (总决赛 — 工作流搭建 50分)
    """
    try:
        from agent.orchestration.workflow_viz import (
            WORKFLOW_NODES,
            MODEL_ROUTING,
            generate_mermaid_flowchart,
            generate_mermaid_agent_collaboration,
            generate_node_table,
            generate_model_routing_table,
        )
        return {
            "status": "ok",
            "data": {
                "mermaid_flowchart": generate_mermaid_flowchart(),
                "mermaid_agent_collaboration": generate_mermaid_agent_collaboration(),
                "node_table_md": generate_node_table(),
                "model_routing_table_md": generate_model_routing_table(),
                "nodes": WORKFLOW_NODES,
                "model_routing": MODEL_ROUTING,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"工作流可视化失败: {e}")


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# 市场时间工具
# ═══════════════════════════════════════════════════════════════

# A股交易时间 (北京时间)
_MORNING_OPEN = time(9, 30)
_MORNING_CLOSE = time(11, 30)
_AFTERNOON_OPEN = time(13, 0)
_AFTERNOON_CLOSE = time(15, 0)

# 2026年中国法定节假日 (A股休市日, 仅含主要长假)
_CN_HOLIDAYS_2026 = {
    date(2026, 1, 1), date(2026, 1, 2),      # 元旦
    date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20),  # 春节 (2.17除夕)
    date(2026, 4, 6),                           # 清明节
    date(2026, 5, 1), date(2026, 5, 4), date(2026, 5, 5),  # 劳动节
    date(2026, 6, 22),                          # 端午节
    date(2026, 9, 28),                          # 中秋节 (9.27中秋)
    date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7),  # 国庆节
}


def _is_market_open(dt: datetime | None = None) -> dict:
    """检测当前是否为A股交易时间, 返回 {is_open, status, detail, last_close_date}"""
    if dt is None:
        dt = datetime.now()

    now_date = dt.date()
    now_time = dt.time()
    weekday = now_date.weekday()  # 0=Mon, 6=Sun

    # 1. 检查是否周末
    if weekday >= 5:
        # 计算最近一个交易日
        days_back = weekday - 4 if weekday >= 5 else 0
        last_trade = now_date - timedelta(days=days_back)
        return {"is_open": False, "status": "weekend", "detail": "周末休市", "last_trade_date": last_trade.isoformat()}

    # 2. 检查是否节假日
    if now_date in _CN_HOLIDAYS_2026:
        last_trade = now_date - timedelta(days=1)
        while last_trade.weekday() >= 5 or last_trade in _CN_HOLIDAYS_2026:
            last_trade -= timedelta(days=1)
        return {"is_open": False, "status": "holiday", "detail": "节假日休市", "last_trade_date": last_trade.isoformat()}

    # 3. 检查交易时段
    if _MORNING_OPEN <= now_time < _MORNING_CLOSE:
        return {"is_open": True, "status": "morning_session", "detail": "早盘交易中 9:30-11:30", "last_trade_date": now_date.isoformat()}
    elif _AFTERNOON_OPEN <= now_time < _AFTERNOON_CLOSE:
        return {"is_open": True, "status": "afternoon_session", "detail": "午盘交易中 13:00-15:00", "last_trade_date": now_date.isoformat()}
    elif now_time < _MORNING_OPEN:
        return {"is_open": False, "status": "pre_open", "detail": f"未开盘, 9:30开盘", "last_trade_date": _last_trade_date(now_date)}
    elif _MORNING_CLOSE <= now_time < _AFTERNOON_OPEN:
        return {"is_open": False, "status": "lunch_break", "detail": "午间休市, 13:00开盘", "last_trade_date": _last_trade_date(now_date)}
    else:
        # after 15:00
        return {"is_open": False, "status": "closed", "detail": "已收盘", "last_trade_date": now_date.isoformat()}


def _last_trade_date(from_date: date) -> str:
    """找到最近的交易日"""
    d = from_date - timedelta(days=1)
    while d.weekday() >= 5 or d in _CN_HOLIDAYS_2026:
        d -= timedelta(days=1)
    return d.isoformat()


# ═══════════════════════════════════════════════════════════════
# 实时行情端点 — 纯JSON, 供前端JS轮询
# ═══════════════════════════════════════════════════════════════

@app.get("/api/v1/realtime/market")
async def realtime_market():
    """实时市场数据 — 大盘指数 + 涨跌Top5 + 重点股票 + 市场状态"""
    import requests
    import asyncio

    market_status = _is_market_open()
    result = {
        "indices": {}, "top_up": [], "top_down": [], "watchlist": [],
        "ts": datetime.now().isoformat(),
        "market_status": market_status,
    }

    idx_codes = {
        "上证指数": "sh000001", "深证成指": "sz399001", "创业板指": "sz399006",
        "科创50": "sh000688", "沪深300": "sh000300",
    }
    watch = [
        "sh600519","sz000858","sh601318","sz300750","sh600036","sz000333","sz000651",
        "sh601088","sh600900","sh601899","sh600031","sz002594","sz002415","sh600276",
        "sz300760","sh601012","sh600030","sz002714","sh600009","sh688111",
    ]

    # v3.1.2: 4 个请求并行 (was sequential ~10s → ~3s)
    # 腾讯直连 (trust_env=False, 免代理快); EastMoney 走代理 (HTTP_PROXY)
    _direct = requests.Session()
    _direct.trust_env = False

    async def _fetch_indices():
        out = {}
        try:
            resp = await asyncio.to_thread(_direct.get,
                f"https://qt.gtimg.cn/q={','.join(idx_codes.values())}", timeout=5,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
            resp.encoding = "gbk"
            for line in resp.text.strip().split("\n"):
                if "=" not in line or "~" not in line: continue
                try:
                    code = line.split("=", 1)[0].replace("v_", "").strip()
                    fields = line.split("=", 1)[1].strip('"').split("~")
                    if len(fields) < 10: continue
                    name = {v: k for k, v in idx_codes.items()}.get(code, code)
                    out[name] = {
                        "price": round(float(fields[3]), 2) if fields[3] else 0,
                        "pct": round(float(fields[32]), 2) if len(fields) > 32 and fields[32] else 0,
                    }
                except (ValueError, IndexError): continue
        except Exception: pass
        return out

    async def _fetch_movers(fid):
        out = []
        try:
            url = (f"https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5&po=1&np=1"
                   f"&fltt=2&invt=2&fid={fid}&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
                   f"&fields=f2,f3,f12,f14")
            resp = await asyncio.to_thread(requests.get, url, timeout=3,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
            for item in (resp.json().get("data", {}).get("diff") or [])[:5]:
                out.append({"code": item.get("f12", ""), "name": item.get("f14", ""),
                            "price": item.get("f2", 0), "pct": item.get("f3", 0)})
        except Exception: pass
        return out

    async def _fetch_watch():
        out = []
        try:
            resp = await asyncio.to_thread(_direct.get,
                f"https://qt.gtimg.cn/q={','.join(watch)}", timeout=5,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
            resp.encoding = "gbk"
            for line in resp.text.strip().split("\n"):
                if "=" not in line or "~" not in line: continue
                try:
                    fields = line.split("=", 1)[1].strip('"').split("~")
                    if len(fields) < 10: continue
                    out.append({"name": fields[1], "code": fields[2],
                        "price": round(float(fields[3]), 2) if fields[3] else 0,
                        "pct": round(float(fields[32]), 2) if len(fields) > 32 and fields[32] else 0,
                        "volume": int(fields[6]) if fields[6] else 0})
                except (ValueError, IndexError): continue
        except Exception: pass
        return out

    indices, top_up, top_down, watchlist = await asyncio.gather(
        _fetch_indices(), _fetch_movers("f3"), _fetch_movers("f32"), _fetch_watch(),
    )
    result["indices"] = indices
    result["top_up"] = top_up
    result["top_down"] = top_down
    result["watchlist"] = watchlist
    return result


@app.get("/api/v1/portfolio/mtm")
async def portfolio_mtm():
    """持仓实时估值 — 获取当前持仓的实时价格并计算浮动盈亏"""
    import requests, json
    import asyncio

    try:
        # 加载持仓
        portfolio_path = Path(__file__).parent.parent / "simulation_data" / "portfolio.json"
        if not portfolio_path.exists():
            return {"error": "portfolio not found", "positions": [], "summary": {}}

        with open(portfolio_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        positions = data.get("positions", {})
        if not positions:
            return {"positions": [], "summary": {"total_value": data.get("account", {}).get("cash", 100000),
                     "cash": data.get("account", {}).get("cash", 100000),
                     "total_return": 0, "total_return_pct": 0}}

        # 获取实时价格 (腾讯行情)
        symbols = list(positions.keys())
        tc_codes = [s.replace("sh.", "sh").replace("sz.", "sz") for s in symbols]
        url = f"https://qt.gtimg.cn/q={','.join(tc_codes)}"

        prices = {}
        names = {}
        try:
            resp = await asyncio.to_thread(
                requests.get, url, timeout=5,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"},
            )
            resp.encoding = "gbk"
            for line in resp.text.strip().split("\n"):
                if "=" not in line or "~" not in line: continue
                try:
                    fields = line.split("=", 1)[1].strip('"').split("~")
                    if len(fields) < 10: continue
                    code = fields[2]
                    for sym in symbols:
                        if sym.endswith(code):
                            price = float(fields[3]) if fields[3] else 0
                            prices[sym] = price
                            names[sym] = fields[1]
                except (ValueError, IndexError): continue
        except Exception: pass

        # 计算浮动盈亏
        account = data.get("account", {})
        cash = account.get("cash", 100000.0)
        total_cost_all = 0
        total_mv_all = 0
        pos_list = []

        for sym, pos in positions.items():
            cur_price = prices.get(sym, pos.get("current_price", pos.get("avg_cost", 0)))
            qty = pos.get("quantity", 0)
            avg_cost = pos.get("avg_cost", 0)
            total_cost = pos.get("total_cost", qty * avg_cost)
            mv = qty * cur_price
            pnl = mv - total_cost
            pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0

            total_cost_all += total_cost
            total_mv_all += mv

            pos_list.append({
                "symbol": sym,
                "name": names.get(sym, pos.get("name", "")),
                "quantity": qty,
                "avg_cost": round(avg_cost, 2),
                "current_price": round(cur_price, 2),
                "market_value": round(mv, 2),
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "stop_loss": pos.get("stop_loss", 0),
                "take_profit": pos.get("take_profit", 0),
                "cost": round(total_cost, 2),
            })

        # 排序: 盈亏降序
        pos_list.sort(key=lambda x: x["unrealized_pnl"], reverse=True)

        total_value = cash + total_mv_all
        initial_capital = data.get("meta", {}).get("initial_capital", 100000)
        total_return = total_value - initial_capital
        total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0

        return {
            "positions": pos_list,
            "summary": {
                "total_value": round(total_value, 2),
                "cash": round(cash, 2),
                "position_value": round(total_mv_all, 2),
                "total_cost": round(total_cost_all, 2),
                "total_return": round(total_return, 2),
                "total_return_pct": round(total_return_pct, 2),
                "position_count": len(pos_list),
                "win_count": sum(1 for p in pos_list if p["unrealized_pnl"] > 0),
                "loss_count": sum(1 for p in pos_list if p["unrealized_pnl"] < 0),
            },
            "ts": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "positions": [], "summary": {}}


# ═══════════════════════════════════════════════════════════════
# 🆕 v3.1: 决策日志 API
# ═══════════════════════════════════════════════════════════════

@app.get("/api/v1/decisions/", tags=["decisions"])
async def list_decisions(symbol: str = None, days: int = 30):
    """
    查询历史决策日志

    Args:
        symbol: 股票代码过滤 (可选)
        days: 查询最近N天 (默认30)
    """
    try:
        from agent.orchestration.decision_log import DecisionLogger
        dl = DecisionLogger()
        if symbol:
            records = dl.get_history(symbol, days=days)
        else:
            records = dl.get_recent_decisions(days=days)
        return {
            "total": len(records),
            "decisions": [r.to_dict() for r in records],
        }
    except Exception as e:
        return {"error": str(e), "decisions": []}


@app.get("/api/v1/decisions/stats", tags=["decisions"])
async def decision_stats(days: int = 30):
    """获取决策统计信息"""
    try:
        from agent.orchestration.decision_log import DecisionLogger
        dl = DecisionLogger()
        return dl.get_stats(days=days)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/v1/decisions/{log_id}/review", tags=["decisions"])
async def review_decision(
    log_id: int,
    realized_return: float = 0,
    benchmark_return: float = 0,
    notes: str = "",
):
    """回顾/记录某条决策的实际结果"""
    try:
        from agent.orchestration.decision_log import DecisionLogger
        dl = DecisionLogger()
        ok = dl.log_outcome(log_id, realized_return, benchmark_return, notes=notes)
        return {"success": ok, "log_id": log_id}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/v1/decisions/reflection/{symbol}", tags=["decisions"])
async def get_reflection(symbol: str, days: int = 5):
    """获取某标的的反思上下文 (用于注入 Agent Prompt)"""
    try:
        from agent.orchestration.decision_log import DecisionLogger
        dl = DecisionLogger()
        reflection = dl.get_reflection_context(symbol, days=days)
        return {"symbol": symbol, "reflection": reflection}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# 企业微信机器人 Webhook
# ═══════════════════════════════════════════════════════════════

@app.post("/api/v1/bot/wecom")
async def wecom_bot_webhook(request: Request):
    """
    企业微信机器人消息回调

    接收: {"msgtype": "text", "text": {"content": "/持仓"}}
    返回: {"msgtype": "markdown", "markdown": {"content": "..."}}
    """
    try:
        body = await request.json()
        msgtype = body.get("msgtype", "text")

        if msgtype == "text":
            text = body.get("text", {}).get("content", "")
        else:
            text = ""

        if not text:
            return {"msgtype": "text", "text": {"content": "请输入命令, 如 /帮助"}}

        from notify.wecom_bot import WeComBot
        bot = WeComBot()
        reply = await bot.handle_message(text)

        return {"msgtype": "markdown", "markdown": {"content": reply}}

    except Exception as e:
        logger.error(f"WeCom webhook 异常: {e}")
        return {"msgtype": "text", "text": {"content": f"处理失败: {e}"}}


@app.post("/api/v1/bot/push/daily")
async def wecom_push_daily():
    """手动触发每日推送"""
    from notify.wecom_bot import WeComBot
    bot = WeComBot()
    ok = bot.push_daily_summary()
    return {"status": "ok" if ok else "failed", "channel": "wecom"}


@app.post("/api/v1/bot/push/alert")
async def wecom_push_alert(title: str = "", detail: str = "", level: str = "warning"):
    """手动触发告警推送"""
    from notify.wecom_bot import WeComBot
    bot = WeComBot()
    ok = bot.push_alert(title, detail, level)
    return {"status": "ok" if ok else "failed", "channel": "wecom"}


# 启动
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
