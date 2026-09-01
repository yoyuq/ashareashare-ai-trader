"""APIRouter: 分析任务/回测/策略列表/系统基准 (v6.0 拆分自 server.py)"""
import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.deps import get_knowledge, get_router, get_tasks
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BacktestRequest,
    BacktestResponse,
    StrategiesResponse,
    TaskResult,
)

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def start_analysis(req: AnalyzeRequest):
    """启动一次分析任务(异步)"""
    task_id = str(uuid.uuid4())[:8]
    tasks = get_tasks()

    tasks[task_id] = {
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
    tasks = get_tasks()
    try:
        tasks[task_id]["status"] = "running"

        from agent.orchestration.workflow import AnalysisWorkflow
        from knowledge.manager import KnowledgeManager

        km = KnowledgeManager()
        wf = AnalysisWorkflow(router=None, knowledge_manager=km)

        if req.task_type == "daily_scan":
            state = await wf.run_daily_scan(symbols=req.symbols)
        else:
            state = await wf.run_stock_analysis(symbol=req.symbols[0])

        tasks[task_id]["result"] = {
            "report": state.get("final_report", ""),
            "regime": state.get("market_regime"),
            "suggestions": state.get("trade_suggestions", []),
            "debate": state.get("debate_result", {}),
        }
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        logger.error(f"分析任务失败 {task_id}: {e}")


@router.get("/analyze/{task_id}", response_model=TaskResult)
async def get_analysis_result(task_id: str):
    """查询分析任务结果"""
    task = get_tasks().get(task_id)
    if task is None:
        raise HTTPException(404, f"任务不存在: {task_id}")
    return TaskResult(**task)


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest):
    """运行回测"""
    from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
    from data.providers.base import DataFrequency, DataRequest

    data_router = get_router()

    # 拉取数据
    data_req = DataRequest(
        symbol=req.symbol,
        start_date=datetime.strptime(req.start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(req.end_date, "%Y-%m-%d").date(),
        frequency=DataFrequency.DAILY,
    )
    result = await data_router.get_daily_kline(data_req)

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


@router.get("/strategies", response_model=StrategiesResponse)
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


@router.get("/system/benchmark")
async def system_benchmark():
    """快速系统性能基准 (不含数据拉取)"""
    import time
    import numpy as np
    import pandas as pd

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
