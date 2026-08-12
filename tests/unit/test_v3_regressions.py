"""v3.0 测试驱动修复的回归测试 — 覆盖对抗测试发现的 4 个 bug

1. 过户费沪深两市都收 (paper_trader + broker 延迟卖出路径)
2. API 敏感端点 (/portfolio/mtm, /bot/*) 必须 X-API-Key 鉴权
3. get_stock_list 跳过错误占位 DataFrame
4. 工具调用路径关闭 DeepSeek 思考模式 + reasoning_content 空值回退
"""

import asyncio
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# 1. 过户费两市都收 (2015 起)
# ═══════════════════════════════════════════════════════════════

def test_paper_trader_transfer_fee_both_exchanges():
    """paper_trader 买入/卖出过户费: sh/sz/bj 都应收取 (¥5000 → ¥0.05)"""
    from simulation.paper_trader import _calc_buy_fees, _calc_sell_fees

    for sym in ["sh.600519", "sz.000858", "bj.830799"]:
        buy_fees = _calc_buy_fees(5000, sym)
        sell_fees = _calc_sell_fees(5000, sym)
        assert abs(buy_fees[2] - 0.05) < 1e-9, f"{sym} 买入过户费应为 0.05, 实际 {buy_fees[2]}"
        assert abs(sell_fees[2] - 0.05) < 1e-9, f"{sym} 卖出过户费应为 0.05, 实际 {sell_fees[2]}"


def test_broker_deferred_sell_transfer_fee_both_exchanges():
    """broker 延迟卖出 (T+1 开盘成交) 对深市也收过户费"""
    from backtest.broker import AShareBroker, Position, PositionSide

    broker = AShareBroker()
    broker.deferred_execution = True
    broker._cur_date = date(2026, 8, 1)
    broker._st_symbols = set()
    broker.account.cash = 100000.0
    broker.account.positions["sz.000858"] = Position(
        symbol="sz.000858", side=PositionSide.LONG,
        buy_date="2026-07-31", can_sell_date=date(2026, 8, 1),
        quantity=100, avg_cost=50.0, total_cost=5000.0,
    )
    broker._cur_prices = {"sz.000858": {
        "open": 50.0, "high": 51.0, "low": 49.5, "close": 50.5,
        "pre_close": 50.0, "pct_change": 1.0,
    }}

    broker.sell("sz.000858", 100)          # deferred → 挂单
    filled = broker.execute_pending_orders()
    assert len(filled) == 1
    # 过户费 = 100股 × 开盘价50 × 0.00001 = 0.05 (两市都收)
    assert filled[0].transfer_fee == pytest.approx(0.05, rel=0.05), \
        f"深市延迟卖出也应收过户费, 实际 {filled[0].transfer_fee}"


# ═══════════════════════════════════════════════════════════════
# 2. API 敏感端点鉴权
# ═══════════════════════════════════════════════════════════════

def test_api_sensitive_endpoints_require_auth(monkeypatch, tmp_path):
    """/portfolio/mtm 与 /bot/* 必须鉴权; 仅白名单公开端点免鉴权"""
    from api import server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "_API_KEYS", {"testkey123"})
    monkeypatch.setattr(
        server, "_AUTH_WHITELIST",
        {"/health", "/docs", "/openapi.json", "/redoc", "/api/v1/realtime/market"},
    )
    monkeypatch.setenv("PORTFOLIO_PATH", str(tmp_path / "portfolio.json"))

    client = TestClient(server.app)

    # 公开端点 (白名单) 免鉴权
    assert client.get("/health").status_code == 200

    # 敏感端点无 key → 401
    assert client.get("/api/v1/portfolio/mtm").status_code == 401
    # query string 传 key 不生效
    assert client.get("/api/v1/portfolio/mtm?api_key=testkey123").status_code == 401
    # 正确 X-API-Key 头 → 200
    r = client.get("/api/v1/portfolio/mtm", headers={"X-API-Key": "testkey123"})
    assert r.status_code == 200


def test_api_multi_key_rotation(monkeypatch, tmp_path):
    """P0-8: 多 key 轮换 — 任一有效 key 均可通过 (滚动轮换过渡期)"""
    from api import server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "_API_KEYS", {"oldkey", "newkey"})
    monkeypatch.setattr(server, "_AUTH_WHITELIST", {"/health"})
    monkeypatch.setattr(server, "_rate_hits", {})
    monkeypatch.setenv("PORTFOLIO_PATH", str(tmp_path / "portfolio.json"))

    client = TestClient(server.app)
    assert client.get("/api/v1/portfolio/mtm", headers={"X-API-Key": "oldkey"}).status_code == 200
    assert client.get("/api/v1/portfolio/mtm", headers={"X-API-Key": "newkey"}).status_code == 200
    assert client.get("/api/v1/portfolio/mtm", headers={"X-API-Key": "bad"}).status_code == 401


def test_api_rate_limit(monkeypatch, tmp_path):
    """P0-8: 认证通过后按 IP 限频 — 超过 burst 返回 429"""
    from api import server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "_API_KEYS", {"k"})
    monkeypatch.setattr(server, "_AUTH_WHITELIST", {"/health"})
    monkeypatch.setattr(server, "_RATE_LIMIT_BURST", 2)
    monkeypatch.setattr(server, "_rate_hits", {})
    monkeypatch.setenv("PORTFOLIO_PATH", str(tmp_path / "portfolio.json"))

    client = TestClient(server.app)
    headers = {"X-API-Key": "k"}
    codes = [client.get("/api/v1/portfolio/mtm", headers=headers).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]


# ═══════════════════════════════════════════════════════════════
# 3. get_stock_list 跳过错误占位表
# ═══════════════════════════════════════════════════════════════

class _FakeProvider:
    """最小 DataProvider mock (仅 get_stock_list 有意义)"""
    def __init__(self, df, source):
        self.source = source
        self._df = df
        self.name = source.value
        self._healthy = True
    def is_healthy(self):
        return self._healthy
    async def get_stock_list(self):
        return self._df
    async def get_daily_kline(self, request):
        return None
    async def get_minute_kline(self, request):
        return None
    async def health_check(self):
        return True


def test_get_stock_list_skips_error_placeholder():
    """Tencent 返回 {"error":[...]} 非空表 → 应跳过并回退到下一源"""
    from data.router import DataRouter
    from data.providers.base import DataSource

    router = DataRouter(cross_validation=False)
    err_df = pd.DataFrame({"error": ["Tencent不支持股票列表"]})
    ok_df = pd.DataFrame({"code": ["600519", "000858"], "name": ["贵州茅台", "五粮液"]})
    router.register(_FakeProvider(err_df, DataSource.TENCENT), priority=1)
    router.register(_FakeProvider(ok_df, DataSource.AKSHARE), priority=2)

    result = asyncio.run(router.get_stock_list())
    assert "error" not in result.columns, "错误占位表不应被当作有效结果返回"
    assert set(result["symbol"]) == {"sh.600519", "sz.000858"}
    assert result["status"].tolist() == ["1", "1"]


def test_get_stock_list_all_error_raises():
    """所有源都返回错误表 → RuntimeError (不返回垃圾表)"""
    from data.router import DataRouter
    from data.providers.base import DataSource

    router = DataRouter(cross_validation=False)
    err_df = pd.DataFrame({"error": ["x"]})
    router.register(_FakeProvider(err_df, DataSource.TENCENT), priority=1)
    with pytest.raises(RuntimeError):
        asyncio.run(router.get_stock_list())


def test_router_per_symbol_degradation_isolates_delisted_stock():
    """P0-7: 单只退市/新股标的连续失败只降级该标的, 不误伤整个源; 过期窗口重置计数"""
    from datetime import timedelta
    from data.router import DataRouter
    from data.providers.base import DataSource, DataFrequency, DataRequest, DataResult
    from timeutil import now_cn_naive

    class _SymbolProvider:
        source = DataSource.AKSHARE
        name = "akshare"

        def __init__(self):
            self._healthy = True

        def is_healthy(self):
            return self._healthy

        async def get_daily_kline(self, request):
            if request.symbol == "sz.999999":
                raise RuntimeError("退市标的无数据")
            return DataResult(
                symbol=request.symbol, source=self.source,
                frequency=DataFrequency.DAILY,
                data=pd.DataFrame({
                    "date": pd.date_range("2026-01-01", periods=10),
                    "open": [1.0] * 10, "high": [1.0] * 10,
                    "low": [1.0] * 10, "close": [1.0] * 10,
                }),
            )

        def _validate_response(self, df, request):
            return not df.empty

        async def get_minute_kline(self, request):
            return DataResult(symbol=request.symbol, source=self.source,
                              frequency=DataFrequency.MIN1, data=pd.DataFrame())

        async def get_stock_list(self):
            return pd.DataFrame({"error": ["x"]})

        async def get_realtime_quote(self, symbols):
            return pd.DataFrame({"error": ["x"]})

        async def health_check(self):
            return True

    router = DataRouter(cross_validation=False)
    prov = _SymbolProvider()
    router.register(prov, priority=0)

    def _req(sym):
        return DataRequest(symbol=sym, start_date=date(2026, 1, 1),
                           end_date=date(2026, 1, 10),
                           frequency=DataFrequency.DAILY, adjust="qfq")

    async def _fetch(sym):
        try:
            return await router.get_daily_kline(_req(sym))
        except RuntimeError:
            return None

    # 退市标的连续失败 3 次 → 仅该标的降级
    for _ in range(3):
        assert asyncio.run(_fetch("sz.999999")) is None
    assert DataSource.AKSHARE not in router._active_sources("sz.999999"), \
        "退市标的连续失败后, 该标度应从活动源排除"
    # 健康标的未被误伤
    assert DataSource.AKSHARE in router._active_sources("sh.600519")
    res = asyncio.run(_fetch("sh.600519"))
    assert res is not None and len(res.data) == 10

    # 降级窗口过期 → 计数重置, 下次失败从 1 重新累计 (不再立即永久降级)
    router._downgraded_until[(DataSource.AKSHARE, "sz.999999")] = \
        now_cn_naive() - timedelta(minutes=1)
    asyncio.run(_fetch("sz.999999"))
    assert router._failure_counts.get((DataSource.AKSHARE, "sz.999999"), 0) == 1
    assert DataSource.AKSHARE in router._active_sources("sz.999999"), \
        "过期窗口重置后, 单次失败不应立即再降级"


# ═══════════════════════════════════════════════════════════════
# 4. 工具调用关闭思考 + reasoning_content 回退
# ═══════════════════════════════════════════════════════════════

def _mock_router(content, reasoning="", tool_calls=None, cache_hit=0):
    from models.router import ModelRouter
    router = ModelRouter()
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(
                content=content, reasoning_content=reasoning, tool_calls=tool_calls,
            )
            usage = SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, prompt_cache_hit_tokens=cache_hit,
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    router._deepseek_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    return router, captured


def test_tool_call_disables_thinking():
    """有 tools 时 extra_body 应关闭思考模式 (规避 reasoning_content 400)"""
    router, captured = _mock_router(content="ok")
    asyncio.run(router._call_model([{"role": "user", "content": "hi"}],
                                   tools=[{"type": "function", "function": {"name": "f"}}]))
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_no_tools_keeps_thinking_default():
    """无 tools 时不关闭思考 (保留推理质量)"""
    router, captured = _mock_router(content="ok")
    asyncio.run(router._call_model([{"role": "user", "content": "hi"}]))
    assert "extra_body" not in captured


def test_reasoning_content_fallback_when_content_empty():
    """思考模式 content 为空时回退 reasoning_content (max_tokens 不足场景)"""
    router, _ = _mock_router(content="", reasoning="这是思考链内容...")
    content, _ = asyncio.run(router._call_model([{"role": "user", "content": "q"}]))
    assert content == "这是思考链内容..."
