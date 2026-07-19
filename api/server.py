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
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# ═══════════════════════════════════════════════════════════════
# 应用初始化
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="A股智能分析Agent API",
    version="2.1.0",
    description="AI驱动的A股量化分析助手 — REST API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 组件引用 (延迟初始化)
_router = None
_knowledge = None
_analyzer = None
_detector = None
_tasks: Dict[str, Dict] = {}


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
        "version": "2.1.0",
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
        # 拉取K线
        req = DataRequest(
            symbol=symbol,
            start_date=date.today().replace(year=date.today().year - 1),
            end_date=date.today(),
            frequency=DataFrequency.DAILY,
        )
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

    # 简单MACD策略
    def macd_strategy(today, bars, broker):
        sym = req.symbol
        if sym not in bars:
            return
        bar = bars[sym]
        close_val = bar["close"]
        pos = broker.account.positions.get(sym)

        # 简化信号: 持仓5天
        if not hasattr(macd_strategy, "_entry"):
            macd_strategy._entry = None
        if macd_strategy._entry is None:
            qty = int(broker.account.cash * 0.3 / close_val / 100) * 100
            if qty >= 100:
                broker.buy(sym, qty, price=close_val)
                macd_strategy._entry = today
        elif (today - macd_strategy._entry).days >= 5 and pos and pos.quantity > 0:
            broker.sell(sym, pos.quantity, price=close_val)
            macd_strategy._entry = None

    macd_strategy._entry = None
    backtest_result = engine.run(macd_strategy, progress_bar=False)

    return {
        "symbol": req.symbol,
        "strategy": req.strategy_id,
        "period": f"{req.start_date} → {req.end_date}",
        "total_return_pct": backtest_result.total_return,
        "annual_return_pct": backtest_result.annual_return,
        "sharpe": backtest_result.sharpe_ratio,
        "sortino": backtest_result.sortino_ratio,
        "max_drawdown_pct": backtest_result.max_drawdown,
        "total_trades": backtest_result.total_trades,
        "win_rate": backtest_result.win_rate,
        "var_95": backtest_result.var_95,
        "cvar_95": backtest_result.cvar_95,
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
        # 使用沪深300指数判断
        req = DataRequest(
            symbol="sh.000300",
            start_date=date.today().replace(year=date.today().year - 1),
            end_date=date.today(),
            frequency=DataFrequency.DAILY,
        )
        result = await router.get_daily_kline(req)

        if result.data.empty:
            return {"regime": "unknown", "error": "无法获取沪深300数据"}

        regime = detector.detect(result.data)
        params = __import__("analysis.regime", fromlist=["get_regime_parameters"]).get_regime_parameters(regime.regime)

        return {
            "regime": regime.regime.value,
            "confidence": regime.confidence,
            "details": regime.details,
            "suggested_params": params,
        }
    except Exception as e:
        raise HTTPException(500, f"市场状态检测失败: {e}")


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
# 启动
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
