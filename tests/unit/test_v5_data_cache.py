"""v5.6 P1-1/P1-3 数据层缓存测试

覆盖:
  P1-1 实时行情 3s 缓存 / 股票列表 5min 缓存
  P1-3 LRU 上限淘汰 + 负缓存 + 并发合并 (单飞)
"""

import asyncio
from datetime import date

import pandas as pd
import pytest

from data.router import DataRouter
from data.providers.base import DataFrequency, DataRequest, DataResult, DataSource


class _CountingProvider:
    """统计调用次数的 mock provider (get_daily_kline 默认返回 10 行有效数据)"""

    def __init__(self, source=DataSource.AKSHARE):
        self.source = source
        self.name = source.value
        self._healthy = True
        self.kline_calls = 0
        self.quote_calls = 0
        self.stock_list_calls = 0

    def is_healthy(self):
        return self._healthy

    def _validate_response(self, df, request):
        return not df.empty

    async def get_daily_kline(self, request):
        self.kline_calls += 1
        return DataResult(
            symbol=request.symbol, source=self.source, frequency=DataFrequency.DAILY,
            data=pd.DataFrame({
                "date": pd.date_range("2026-01-01", periods=10),
                "open": [1.0] * 10, "high": [1.0] * 10,
                "low": [1.0] * 10, "close": [1.0] * 10,
            }),
        )

    async def get_minute_kline(self, request):
        return DataResult(
            symbol=request.symbol, source=self.source,
            frequency=DataFrequency.MIN5, data=pd.DataFrame(),
        )

    async def get_stock_list(self):
        self.stock_list_calls += 1
        return pd.DataFrame({"code": ["600519", "000858"], "name": ["茅台", "五粮液"]})

    async def get_realtime_quote(self, symbols):
        self.quote_calls += 1
        return pd.DataFrame({"symbol": symbols, "price": [10.0] * len(symbols)})

    async def health_check(self):
        return True


def _req(sym="sh.600519"):
    return DataRequest(symbol=sym, start_date=date(2026, 1, 1),
                       end_date=date(2026, 1, 10),
                       frequency=DataFrequency.DAILY, adjust="qfq")


# P1-1: 实时行情 3s 缓存
def test_realtime_quote_cached_within_ttl():
    router = DataRouter(cross_validation=False)
    prov = _CountingProvider()
    router.register(prov, priority=0)
    df1 = asyncio.run(router.get_realtime_quote(["sh.600519"]))
    df2 = asyncio.run(router.get_realtime_quote(["sh.600519"]))
    assert prov.quote_calls == 1, "TTL 内应命中缓存, 不重复请求"
    assert len(df1) == 1 and len(df2) == 1


# P1-1: 股票列表 5min 缓存
def test_stock_list_cached_within_ttl():
    router = DataRouter(cross_validation=False)
    prov = _CountingProvider()
    router.register(prov, priority=0)
    out1 = asyncio.run(router.get_stock_list())
    out2 = asyncio.run(router.get_stock_list())
    assert prov.stock_list_calls == 1, "TTL 内应命中缓存, 不重复请求"
    assert set(out1["symbol"]) == {"sh.600519", "sz.000858"}
    assert set(out2["symbol"]) == {"sh.600519", "sz.000858"}


# P1-3: LRU 上限淘汰
def test_cache_lru_eviction():
    router = DataRouter(cross_validation=False)
    router._cache_max_entries = 2  # 缩小上限便于测试
    router._cache_set("k1", "v1")
    router._cache_set("k2", "v2")
    router._cache_set("k3", "v3")  # 超限 → 淘汰最旧 k1
    assert "k1" not in router._cache
    assert "k2" in router._cache and "k3" in router._cache


# P1-3: 负缓存 — 全源失败 30s 内不重复击穿 provider
def test_negative_cache_short_circuits():
    class _FailProvider(_CountingProvider):
        async def get_daily_kline(self, request):
            self.kline_calls += 1
            raise RuntimeError("无数据")

    router = DataRouter(cross_validation=False)
    prov = _FailProvider()
    router.register(prov, priority=0)
    with pytest.raises(RuntimeError):
        asyncio.run(router.get_daily_kline(_req()))
    with pytest.raises(RuntimeError):
        asyncio.run(router.get_daily_kline(_req()))
    assert prov.kline_calls == 1, "负缓存命中后不应再次访问 provider"


# P1-3: 并发合并 (单飞) — 相同 key 并发请求只触发一次 provider 调用
def test_concurrent_merge_single_flight():
    class _SlowProvider(_CountingProvider):
        async def get_daily_kline(self, request):
            self.kline_calls += 1
            await asyncio.sleep(0.05)
            return DataResult(
                symbol=request.symbol, source=self.source, frequency=DataFrequency.DAILY,
                data=pd.DataFrame({
                    "date": pd.date_range("2026-01-01", periods=10),
                    "open": [1.0] * 10, "high": [1.0] * 10,
                    "low": [1.0] * 10, "close": [1.0] * 10,
                }),
            )

    router = DataRouter(cross_validation=False)
    prov = _SlowProvider()
    router.register(prov, priority=0)

    async def main():
        return await asyncio.gather(
            router.get_daily_kline(_req()),
            router.get_daily_kline(_req()),
        )

    r1, r2 = asyncio.run(main())
    assert prov.kline_calls == 1, "并发相同请求应合并为一次 provider 调用"
    assert r1 is not None and r2 is not None
    assert len(r1.data) == 10 and len(r2.data) == 10
