"""API Pydantic 模型 (v6.0 拆分自 server.py)

所有请求/响应模型集中于此; server.py re-export 以保持 `server.XXX` 测试兼容。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    @field_validator("start_date", "end_date")
    @classmethod
    def _validate_date_format(cls, v: str) -> str:
        """日期必须为严格 YYYY-MM-DD (零填充) 格式 (v5.6 P1-16: 此前裸 str 未校验)。

        先用正则锁定零填充格式, 再用 strptime 校验日历有效性 (如 2023-02-30 拒绝),
        避免 strptime 对 "2023-1-1" 这类非严格格式的宽松接受。
        """
        import re
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError(f"日期格式须为 YYYY-MM-DD (零填充), 收到: {v!r}")
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"无效日期: {v!r}")
        return v

    @model_validator(mode="after")
    def _validate_date_order(self) -> "BacktestRequest":
        if datetime.strptime(self.start_date, "%Y-%m-%d") > datetime.strptime(self.end_date, "%Y-%m-%d"):
            raise ValueError("start_date 不得晚于 end_date")
        return self


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


# ── v5.6 P1-16: 响应模型 (此前多数端点返回裸 dict, 无 OpenAPI schema) ──
# 带 extra="allow" 的模型保留端点可能追加的额外字段, 避免反序列化时静默丢字段。

class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    version: str
    timestamp: str


class StockInfoResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    symbol: str
    date: Optional[str] = None
    close: Optional[float] = None
    active_patterns: List[str] = Field(default_factory=list)


class BacktestResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    symbol: str
    strategy: str
    strategy_name: str = ""
    period: str = ""
    total_return_pct: Optional[float] = None
    annual_return_pct: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    total_trades: Optional[int] = None
    win_rate: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    strategy_backtest: Dict[str, Any] = Field(default_factory=dict)


class StrategiesResponse(BaseModel):
    total: int
    strategies: List[Dict[str, Any]]


class RegimeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    regime: str
    confidence: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    suggested_params: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    session_id: str = Field(default="default", description="会话ID, 用于多轮对话")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tool_calls: List[str] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    session_id: str
    message_count: int
    history: List[Dict[str, Any]]


class DecisionsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    total: int
    decisions: List[Dict[str, Any]]
