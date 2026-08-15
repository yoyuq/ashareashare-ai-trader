"""analysis.benchmark — 基准纪律层单元测试 (阶段1 #107)。

验证: 等权价格净值 / 分红加回 (全收益) / 红利贡献 / 缺分红数据退回价格收益。
"""

import numpy as np
import pandas as pd
import pytest

from analysis.benchmark import (
    dividend_contribution_pct,
    matched_universe_curve,
    matched_universe_total_return_curve,
    total_return_daily_factor,
)


def _frame(closes_by_symbol: dict, dates=None) -> pd.DataFrame:
    """构造 replay 日线长表: {symbol: [close...]} → DataFrame。"""
    dates = dates or pd.date_range("2026-01-01", periods=len(next(iter(closes_by_symbol.values()))), freq="B")
    rows = []
    for sym, closes in closes_by_symbol.items():
        for d, c in zip(dates, closes):
            rows.append({"date": d, "symbol": sym, "close": float(c)})
    return pd.DataFrame(rows)


def _divs(records: list[tuple]) -> pd.DataFrame:
    """records: [(code, date, cash)] → 分红 DataFrame。"""
    return pd.DataFrame(records, columns=["code", "dividOperateDate", "dividCashPsBeforeTax"])


def test_equalweight_price_curve():
    """两只等权股票价格净值 = 各自归一后的均值。"""
    df = _frame({"a": [100.0, 110.0, 121.0], "b": [50.0, 55.0, 49.5]})
    eq = matched_universe_curve(df, ["a", "b"])
    assert eq.iloc[0] == pytest.approx(1.0, abs=1e-9)
    # a: 1.0, 1.1, 1.21; b: 1.0, 1.1, 0.99 → 均值 1.0, 1.1, 1.10
    assert eq.iloc[1] == pytest.approx(1.1, abs=1e-9)
    assert eq.iloc[2] == pytest.approx(1.10, abs=1e-9)


def test_total_return_adds_back_dividend():
    """不复权价格除权日下跳 D, 全收益应把它加回 (净值走平), 价格收益则下跌。"""
    # 除权日: close 从 100 跳到 99 (每股红利 1.0), 全收益应平。
    df = _frame({"a": [100.0, 99.0, 99.0]})
    divs = _divs([("a", pd.Timestamp("2026-01-02"), 1.0)])
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    price = matched_universe_curve(df, ["a"])
    tr = matched_universe_total_return_curve(df, ["a"], divs)
    assert price.iloc[1] == pytest.approx(0.99, abs=1e-9)   # 价格收益 -1%
    assert tr.iloc[1] == pytest.approx(1.0, abs=1e-9)       # 全收益 0% (红利加回)
    # 红利贡献 = 全收益 - 价格收益 ≈ +1pp
    contrib = dividend_contribution_pct(price, tr)
    assert contrib == pytest.approx(1.0, abs=0.01)


def test_total_return_aggregates_multiple_dividends_same_day():
    """同日多条红利合并后再加回。"""
    df = _frame({"a": [100.0, 98.0]})
    divs = _divs([("a", pd.Timestamp("2026-01-02"), 1.0),
                  ("a", pd.Timestamp("2026-01-02"), 1.0)])
    tr = matched_universe_total_return_curve(df, ["a"], divs)
    # (98 + 2)/100 = 1.0 → 全收益走平
    assert tr.iloc[1] == pytest.approx(1.0, abs=1e-9)


def test_total_return_missing_dividend_data_falls_back():
    """分红数据缺失 → 退回价格收益 (不伪造), 两曲线相等。"""
    df = _frame({"a": [100.0, 99.0]})
    price = matched_universe_curve(df, ["a"])
    tr_none = matched_universe_total_return_curve(df, ["a"], None)
    tr_bad = matched_universe_total_return_curve(df, ["a"], pd.DataFrame({"x": [1]}))
    assert tr_none.iloc[1] == pytest.approx(price.iloc[1], abs=1e-9)
    assert tr_bad.iloc[1] == pytest.approx(price.iloc[1], abs=1e-9)


def test_empty_universe_returns_empty():
    df = _frame({"a": [100.0, 99.0]})
    assert matched_universe_curve(df, []).empty
    assert matched_universe_total_return_curve(df, [], None).empty


def test_dividend_contribution_nan_on_short_series():
    p = pd.Series([1.0], index=pd.date_range("2026-01-01", periods=1, freq="B"))
    assert np.isnan(dividend_contribution_pct(p, p))


def test_total_return_daily_factor_adds_back_dividend():
    """日因子矩阵除权日 = (close+div)/prev_close; 无分红退回价格因子。"""
    df = _frame({"a": [100.0, 99.0, 99.0], "b": [50.0, 55.0, 55.0]})
    divs = _divs([("a", pd.Timestamp("2026-01-02"), 1.0)])
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    f = total_return_daily_factor(df, ["a", "b"], divs)
    assert f.index.equals(dates)
    # a: 除权日因子 = (99 + 1)/100 = 1.0 (红利加回)
    assert f.loc[pd.Timestamp("2026-01-02"), "a"] == pytest.approx(1.0, abs=1e-9)
    # b: 无分红, 次日因子 = 55/50 = 1.1
    assert f.loc[pd.Timestamp("2026-01-02"), "b"] == pytest.approx(1.1, abs=1e-9)
    # 无分红数据 → 退回价格因子
    fp = total_return_daily_factor(df, ["a", "b"], None)
    assert fp.loc[pd.Timestamp("2026-01-02"), "a"] == pytest.approx(0.99, abs=1e-9)
