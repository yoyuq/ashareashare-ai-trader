"""APIRouter: 决策日志 (v6.0 拆分自 server.py)"""
from fastapi import APIRouter

from api.schemas import DecisionsResponse

router = APIRouter(prefix="/api/v1", tags=["decisions"])


@router.get("/decisions", response_model=DecisionsResponse, tags=["decisions"])
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
        return {"total": 0, "error": str(e), "decisions": []}


@router.get("/decisions/stats", tags=["decisions"])
async def decision_stats(days: int = 30):
    """获取决策统计信息"""
    try:
        from agent.orchestration.decision_log import DecisionLogger
        dl = DecisionLogger()
        return dl.get_stats(days=days)
    except Exception as e:
        return {"error": str(e)}


@router.post("/decisions/{log_id}/review", tags=["decisions"])
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


@router.get("/decisions/reflection/{symbol}", tags=["decisions"])
async def get_reflection(symbol: str, days: int = 5):
    """获取某标的的反思上下文 (用于注入 Agent Prompt)"""
    try:
        from agent.orchestration.decision_log import DecisionLogger
        dl = DecisionLogger()
        reflection = dl.get_reflection_context(symbol, days=days)
        return {"symbol": symbol, "reflection": reflection}
    except Exception as e:
        return {"error": str(e)}
