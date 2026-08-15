"""v5.6 P1-16 API 端点: 尾斜杠统一 + response_model + BacktestRequest 日期校验

覆盖:
  - BacktestRequest 日期须 YYYY-MM-DD, 且 start_date <= end_date (此前裸 str 未校验)
  - 响应模型 (HealthResponse/BacktestResponse/...) 可构造并校验样例
  - /api/v1/decisions 端点无尾斜杠 (与其余端点一致)
"""
import pytest
from pydantic import ValidationError

from api import server


# ═══════════════════════════════════════════════════════════════
# BacktestRequest 日期校验
# ═══════════════════════════════════════════════════════════════

def test_backtest_request_valid_dates():
    r = server.BacktestRequest(
        symbol="sh.600519", strategy_id="ma_cross",
        start_date="2023-01-01", end_date="2024-12-31",
    )
    assert r.start_date == "2023-01-01"
    assert r.end_date == "2024-12-31"


@pytest.mark.parametrize("bad", [
    "2023/01/01",   # 斜杠分隔
    "20230101",     # 无分隔
    "notadate",     # 非日期
    "2023-1-1",     # 未补零
    "2023-13-40",   # 非法月/日
])
def test_backtest_request_rejects_bad_date_format(bad):
    with pytest.raises(ValidationError):
        server.BacktestRequest(symbol="s", strategy_id="t", start_date=bad)


def test_backtest_request_rejects_start_after_end():
    with pytest.raises(ValidationError):
        server.BacktestRequest(
            symbol="s", strategy_id="t",
            start_date="2024-01-01", end_date="2023-01-01",
        )


def test_backtest_request_allows_equal_dates():
    r = server.BacktestRequest(symbol="s", strategy_id="t",
                               start_date="2024-01-01", end_date="2024-01-01")
    assert r.end_date == "2024-01-01"


# ═══════════════════════════════════════════════════════════════
# 响应模型可构造
# ═══════════════════════════════════════════════════════════════

def test_response_models_validate_sample_payloads():
    server.HealthResponse(status="healthy", version="x", timestamp="t")
    server.StockInfoResponse(symbol="sh.600519", close=1800.0, active_patterns=["golden_cross"])
    server.BacktestResponse(symbol="sh.600519", strategy="ma",
                            total_return_pct=1.5, strategy_backtest={"signals": 3})
    server.StrategiesResponse(total=1, strategies=[{"id": "a", "name": "n", "category": "c"}])
    server.RegimeResponse(regime="bull", confidence=0.8)
    server.ChatHistoryResponse(session_id="s", message_count=1,
                               history=[{"role": "user", "content": "hi"}])
    server.DecisionsResponse(total=0, decisions=[])


# ═══════════════════════════════════════════════════════════════
# 尾斜杠统一
# ═══════════════════════════════════════════════════════════════

def test_decisions_endpoint_no_trailing_slash():
    """所有端点统一无尾斜杠: /api/v1/decisions 而非 /api/v1/decisions/"""
    paths = {getattr(r, "path", None) for r in server.app.routes}
    assert "/api/v1/decisions" in paths
    assert "/api/v1/decisions/" not in paths
