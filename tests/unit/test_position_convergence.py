"""目标仓位收敛 + 防御减仓共用弱股判定 `_below_ma20` 的单元测试.

覆盖 v3.5 新增的"只卖自身走弱股(现价<自身MA20), 保留强势股"决策规则:
- 走弱股 (现价<MA20) → 返回 (现价, MA20)
- 强势股 (现价>=MA20) → None (不卖)
- 数据不足 (<25 行无法算 MA20) → None (保守不卖)
- 数据异常/缺失 → None (保守不卖)
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


class _FakeDataRouter:
    """伪造 data_router: 按 symbol 返回预设的每日K线 DataFrame."""

    def __init__(self, frames: dict):
        self._frames = frames

    async def get_daily_kline(self, req):
        df = self._frames.get(req.symbol)
        return SimpleNamespace(data=df)


def _make_kline(closes):
    """构造 60 根K线, 每天收盘价循环取 closes, 最后一天为 closes[-1]."""
    n = 60
    closes = list(closes)
    # 用单调递增尾部保证最后一天是 MA20/现价判定依据
    arr = np.linspace(10.0, 20.0, n - 1)  # 前 59 天
    arr = np.concatenate([arr, [closes[-1]]])
    return pd.DataFrame({"date": pd.date_range("2026-06-01", periods=n, freq="D"),
                         "close": arr})


def _run(sym, router):
    from simulation.daily_runner import _below_ma20
    return asyncio.run(_below_ma20(sym, router))


def test_weak_stock_below_ma20_returns_tuple():
    """现价 < 自身MA20 → 返回 (现价, MA20), 触发卖出."""
    # 前段缓慢上涨后最后一天大幅下挫 → 现价跌破MA20
    n = 60
    closes = np.linspace(10.0, 20.0, n - 1)
    closes = np.concatenate([closes, [15.0]])  # 最后一天跌到 15.0, MA20 约 18.x
    df = pd.DataFrame({"date": pd.date_range("2026-06-01", periods=n, freq="D"),
                       "close": closes})
    router = _FakeDataRouter({"sh.600000": df})
    res = _run("sh.600000", router)
    assert res is not None
    px, ma20 = res
    assert px == pytest.approx(15.0)
    assert px < ma20


def test_strong_stock_above_ma20_returns_none():
    """现价 >= 自身MA20 (强势) → None, 不卖出."""
    n = 60
    closes = np.linspace(10.0, 20.0, n - 1)
    closes = np.concatenate([closes, [20.5]])  # 最后一天新高 20.5 > MA20
    df = pd.DataFrame({"date": pd.date_range("2026-06-01", periods=n, freq="D"),
                       "close": closes})
    router = _FakeDataRouter({"sh.600000": df})
    assert _run("sh.600000", router) is None


def test_insufficient_data_returns_none():
    """不足 25 根K线无法算 MA20 → None (保守不卖)."""
    df = pd.DataFrame({"date": pd.date_range("2026-06-01", periods=10, freq="D"),
                       "close": np.linspace(10.0, 5.0, 10)})
    router = _FakeDataRouter({"sh.600000": df})
    assert _run("sh.600000", router) is None


def test_data_exception_returns_none():
    """数据拉取异常 → None (保守不卖, 不抛冒泡)."""
    class _BrokenRouter:
        async def get_daily_kline(self, req):
            raise ConnectionError("boom")
    assert _run("sh.600000", _BrokenRouter()) is None


def test_missing_close_column_returns_none():
    """缺 close 列 → None."""
    df = pd.DataFrame({"date": pd.date_range("2026-06-01", periods=30, freq="D"),
                       "open": np.linspace(10.0, 20.0, 30)})
    router = _FakeDataRouter({"sh.600000": df})
    assert _run("sh.600000", router) is None