"""
基准纪律层 (阶段1 #107): 统一「全收益 + 匹配 universe」。

问题 (roadmap §2.3「基准不严谨」):
  1. 大量「跑赢」用上证综指 (sh.000001), 而策略从 top-800 流动性 universe 选股, 天然小盘 tilt,
     对上证 (大盘市值加权) 自带便宜 → 基准 universe 不匹配。
  2. replay 日线是不复权 (adjustflag=3), 除权除息日价格下跳但没把现金红利加回; 上证综指也是
     价格指数(不含分红) → 两边都用「价格收益」而非「全收益」, 系统性低估真实市场回报约 2~3%/年,
     「跑赢」因此虚高。

本模块统一两条纪律:
  - **匹配 universe 基准**: 策略所属 universe 的等权净值 (而非错配的指数), 回答「选股是否真跑赢
    自己可投的池子」。
  - **全收益基准**: 把真实分红加回 (不复权价格 + 每股税前现金红利 → 总收益净值), 回答「是否真
    跑赢市场总回报 (含分红)」, 而非价格指数。

全程真实数据: 净值来自 replay 日线 parquet (不复权), 分红来自 Baostock query_dividend_data
(scripts/fetch_dividends.py → replay_data/dividends.parquet)。缺分红数据不伪造, 退回价格收益并显式标注。

用法:
    from analysis.benchmark import matched_universe_curve, matched_universe_total_return_curve
    eq_price = matched_universe_curve(df, uni)                 # 等权价格净值
    eq_tr    = matched_universe_total_return_curve(df, uni, div_df)  # 等权全收益净值
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# 分红 parquet 列名约定 (scripts/fetch_dividends.py 输出)
_DIV_CODE = "code"
_DIV_DATE = "dividOperateDate"
_DIV_CASH = "dividCashPsBeforeTax"


def matched_universe_curve(df: pd.DataFrame, symbols: List[str]) -> pd.Series:
    """
    策略所属 universe 的等权价格净值曲线 (首日归一, 停牌 ffill)。

    这是「选股是否真跑赢自己可投池子」的匹配基准。与各 A/B 脚本里重复的 `_curve`/`_equalweight_curve`
    同口径, 收敛到一处避免漂移。

    Args:
        df: replay 日线 (columns 含 date/symbol/close)。
        symbols: universe 符号列表 (须在 df.symbol 中)。

    Returns:
        等权净值 Series (索引 date, 首值=1.0); universe 无有效数据返回空 Series。
    """
    sub = df[df.symbol.isin(symbols)][["date", "symbol", "close"]].dropna(subset=["close"])
    if sub.empty:
        return pd.Series(dtype=float)
    piv = sub.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    piv = piv.reindex(columns=[s for s in symbols if s in piv.columns])
    if piv.empty:
        return pd.Series(dtype=float)
    norm = piv / piv.iloc[0]
    norm = norm.ffill().dropna(axis=1, how="all")
    if norm.empty:
        return pd.Series(dtype=float)
    return norm.mean(axis=1)


def total_return_daily_factor(
    df: pd.DataFrame, symbols: List[str], dividends: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    每日**全收益**因子矩阵 (行=date, 列=symbol): 因子_t = (close_t + 红利_t) / close_{t-1}。

    无分红数据 (dividends=None 或缺列) → 退回价格因子 close_t / close_{t-1} (即 pct_change + 1)。

    用途: 让「策略篮子」与「基准」共用同一口径, 避免「策略用价格收益、基准用全收益」的不对称比较
    (策略持有的票同样会除权除息, 其价格收益同样被系统性低估 ~2~3%/年)。
    """
    sub = df[df.symbol.isin(symbols)][["date", "symbol", "close"]].dropna(subset=["close"])
    if sub.empty:
        return pd.DataFrame()
    piv = sub.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    cols = [s for s in symbols if s in piv.columns]
    piv = piv.reindex(columns=cols).ffill()
    factor = piv / piv.shift(1)
    factor.iloc[0] = 1.0
    if dividends is not None and _DIV_DATE in dividends.columns and _DIV_CASH in dividends.columns:
        d = dividends[dividends[_DIV_CODE].isin(cols)]
        if not d.empty:
            agg = d.groupby([_DIV_CODE, _DIV_DATE])[_DIV_CASH].sum().reset_index()
            dpiv = agg.pivot_table(index=_DIV_DATE, columns=_DIV_CODE, values=_DIV_CASH, aggfunc="sum")
            dpiv = dpiv.reindex(index=piv.index, columns=cols).fillna(0.0)
            factor = (piv + dpiv) / piv.shift(1)
            factor.iloc[0] = 1.0
    return factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def _total_return_price(close: pd.Series, dividends: pd.Series) -> pd.Series:
    """
    不复权收盘价 + 每股税前现金红利 → 单股全收益净值 (首值=1.0)。

    除权除息日 t: 总收益因子 = (close_t + 红利_t) / close_{t-1}; 其余日 = close_t / close_{t-1}。
    """
    p = close.astype(float)
    d = dividends.reindex(p.index).fillna(0.0).astype(float)
    p_prev = p.shift(1)
    daily = (p + d) / p_prev
    daily.iloc[0] = 1.0  # 首日无收益
    daily = daily.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return daily.cumprod()


def matched_universe_total_return_curve(
    df: pd.DataFrame, symbols: List[str], dividends: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    匹配 universe 的等权**全收益**净值 (不复权价格 + 真实分红加回)。

    分红数据缺失 (dividends=None 或含分红列) 时退回价格净值, 不伪造 (调用方据 is_total_return 标注)。
    """
    if dividends is None or _DIV_DATE not in dividends.columns or _DIV_CASH not in dividends.columns:
        # 无分红数据 → 退回价格收益 (调用方应显式标注非全收益)
        return matched_universe_curve(df, symbols)

    sub = df[df.symbol.isin(symbols)][["date", "symbol", "close"]].dropna(subset=["close"])
    if sub.empty:
        return pd.Series(dtype=float)
    piv = sub.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    cols = [s for s in symbols if s in piv.columns]
    piv = piv.reindex(columns=cols)

    curves = {}
    for sym in cols:
        close = piv[sym].ffill()  # 停牌沿用前收
        div = dividends[dividends[_DIV_CODE] == sym]
        if div.empty:
            curves[sym] = (close / close.iloc[0]).ffill()
        else:
            dser = div.groupby(_DIV_DATE)[_DIV_CASH].sum()  # 同一日多条红利合并
            curves[sym] = _total_return_price(close, dser)

    if not curves:
        return pd.Series(dtype=float)
    mat = pd.DataFrame(curves).ffill().dropna(axis=1, how="all")
    if mat.empty:
        return pd.Series(dtype=float)
    return mat.mean(axis=1)


def dividend_contribution_pct(
    price_curve: pd.Series, total_return_curve: pd.Series
) -> float:
    """窗口内红利对总收益的贡献 (pp): total_return_pct − price_return_pct。"""
    p = price_curve.dropna()
    t = total_return_curve.dropna()
    if len(p) < 2 or len(t) < 2:
        return float("nan")
    # 按共同日期对齐后比较累计收益
    idx = p.index.intersection(t.index)
    if len(idx) < 2:
        return float("nan")
    p_ret = float(p.loc[idx].iloc[-1] / p.loc[idx].iloc[0] - 1.0) * 100.0
    t_ret = float(t.loc[idx].iloc[-1] / t.loc[idx].iloc[0] - 1.0) * 100.0
    return t_ret - p_ret
