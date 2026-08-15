"""组合级事件驱动回测 (P0-2) 单元测试 — 信号抽取 + 引擎接入。

不依赖网络: 用合成 OHLCV + 合成 bt_func 验证适配器, 另用真实 _backtest_ma_trend
验证信号键已由信号回测器返回 (守护 P0-2 的信号抽取改造)。
"""
import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig
from backtest.strategy_portfolio_bt import (
    _extract_signal_dates,
    run_portfolio_event_backtest,
)


def _make_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = 50 + np.cumsum(rng.normal(0.02, 1.0, n))
    close = np.maximum(close, 2.0)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": rng.integers(100000, 5000000, n).astype(float),
    })


def test_extract_signal_dates_maps_indices():
    df = _make_df()

    def bt(df, symbol=""):
        return {"entry_sig": [3, 10], "exit_sig": [7], "signals": 2, "trades": []}

    e, x = _extract_signal_dates(bt, df, "sh.600000")
    dts = pd.to_datetime(df["date"]).dt.date.tolist()
    assert dts[3] in e and dts[10] in e
    assert dts[7] in x


def test_portfolio_event_backtest_produces_trades():
    df = _make_df()

    def bt(df, symbol=""):
        # 每 20 根 bar 发一个 entry, 8 根后 exit (交错) → 引擎应产生卖出成交
        n = len(df)
        entry = list(range(0, n - 1, 20))
        exit_ = [i + 8 for i in entry if i + 8 < n]
        return {"entry_sig": entry, "exit_sig": exit_, "signals": len(entry), "trades": []}

    config = BacktestConfig(
        initial_capital=100000.0,
        start_date=pd.Timestamp("2023-01-01").date(),
        end_date=pd.Timestamp("2023-06-01").date(),
        max_positions=3,
    )
    res = run_portfolio_event_backtest(bt, {"sh.600000": df}, config)
    assert res is not None
    assert res.total_trades >= 1
    assert len(res.equity_curve) > 0


def test_real_backtester_returns_signal_keys():
    """守护 P0-2: 信号回测器必须返回 entry_sig/exit_sig (否则组合级回测抽不到信号)。"""
    from analysis.recommender import _backtest_ma_trend
    df = _make_df(n=120)
    out = _backtest_ma_trend(df, symbol="sh.600000")
    assert "entry_sig" in out
    assert "exit_sig" in out
    assert isinstance(out["entry_sig"], list)
    assert isinstance(out["exit_sig"], list)
