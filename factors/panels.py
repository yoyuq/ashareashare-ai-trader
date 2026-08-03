"""因子面板计算 (v3.1.2) — 向量化跨时间轴计算因子, 供 IC 评估使用

原 `factors/evaluate.py` 用 "每交易日 × 每只股票" 双重循环调单股因子函数,
5107 只 × 60 日 = 30 万次切片, 慢到不可用。这里改为对每只股票一次性
**向量化**算出因子在整个时间轴上的序列, 评估端再按日取截面做 Spearman。
单股因子 (FACTORS) 保留用于实时单股场景; 面板口径与之一致。

所有因子函数: `fn(sym_df: pd.DataFrame, date_idx) -> pd.Series` —
输入单股完整日K (date index, 含 close/volume/amount/turn/peTTM/pbMRQ/open/high/low,
可选 mkt_close), 返回该股逐日因子值 Series。
"""

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

PANEL_FACTORS: Dict[str, Callable] = {}


def _register(name):
    def deco(fn):
        PANEL_FACTORS[name] = fn
        return fn
    return deco


def _tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


# ── baseline ──

@_register("vp_divergence")
def vp_divergence(df: pd.DataFrame, didx) -> pd.Series:
    """量价背离 (与 sources.vp_divergence 口径一致)"""
    c = df["close"]
    v = df["volume"]
    ret5 = c.pct_change(5)
    vol5 = v.rolling(5).mean() / v.shift(5).rolling(5).mean()
    base = ret5.clip(-0.2, 0.2) * 5
    # 价涨+量缩/价跌+量增 → 翻转符号
    pos = pd.Series(np.where(vol5 > 1, base, -base), index=df.index)
    # 价跌+量缩 → 温和正值 (0.3 系数, 与单股版一致)
    adj = (ret5 < 0) & (vol5 <= 1)
    return pos.where(~adj, pos * 0.3)


@_register("turnover_anom")
def turnover_anom(df: pd.DataFrame, didx) -> pd.Series:
    if "turn" not in df.columns:
        return pd.Series(0.0, index=df.index)
    t = pd.to_numeric(df["turn"], errors="coerce").fillna(0.0)
    base = t.shift(1).rolling(20).mean()
    return np.log(t / base + 1e-6)


@_register("rsi_extreme")
def rsi_extreme(df: pd.DataFrame, didx) -> pd.Series:
    c = df["close"]
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.rolling(14).mean()
    al = loss.rolling(14).mean()
    rs = ag / al.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)
    out = pd.Series(0.0, index=df.index)
    out[rsi > 70] = -(rsi[rsi > 70] - 70) / 30
    out[rsi < 30] = (30 - rsi[rsi < 30]) / 30
    return out


@_register("momentum_5")
def momentum_5(df: pd.DataFrame, didx) -> pd.Series:
    return df["close"].pct_change(5)


@_register("momentum_20")
def momentum_20(df: pd.DataFrame, didx) -> pd.Series:
    return df["close"].pct_change(20)


@_register("vol_spike")
def vol_spike(df: pd.DataFrame, didx) -> pd.Series:
    tr = _tr(df["high"], df["low"], df["close"])
    atr5 = tr.rolling(5).mean()
    base = tr.shift(5).rolling(15).mean()
    return np.log(atr5 / base + 1e-6)


@_register("bias_ma20")
def bias_ma20(df: pd.DataFrame, didx) -> pd.Series:
    c = df["close"]
    return c / c.rolling(20).mean() - 1


@_register("volume_ratio")
def volume_ratio(df: pd.DataFrame, didx) -> pd.Series:
    v = df["volume"]
    return v / v.shift(1).rolling(5).mean() - 1


# ── wave1 ──

@_register("ep")
def ep(df: pd.DataFrame, didx) -> pd.Series:
    if "peTTM" not in df.columns:
        return pd.Series(0.0, index=df.index)
    pe = pd.to_numeric(df["peTTM"], errors="coerce")
    clipped = np.clip(pe, 0.1, 200)
    out = 1.0 / clipped
    out[pe <= 0] = 0.0
    return out


@_register("bp")
def bp(df: pd.DataFrame, didx) -> pd.Series:
    if "pbMRQ" not in df.columns:
        return pd.Series(0.0, index=df.index)
    pb = pd.to_numeric(df["pbMRQ"], errors="coerce")
    clipped = np.clip(pb, 0.05, 50)
    out = 1.0 / clipped
    out[pb <= 0] = 0.0
    return out


def _pctile_20(series: pd.Series) -> pd.Series:
    """当日值在自身前20日中的分位数 (0=自身最便宜/低)"""
    return series.rolling(21, min_periods=11).apply(
        lambda w: (w[:-1] < w[-1]).mean(), raw=True
    )


@_register("pe_pct_20d")
def pe_pct_20d(df: pd.DataFrame, didx) -> pd.Series:
    if "peTTM" not in df.columns:
        return pd.Series(0.5, index=df.index)
    pe = pd.to_numeric(df["peTTM"], errors="coerce")
    out = _pctile_20(pe)
    return out.fillna(0.5)


@_register("pb_pct_20d")
def pb_pct_20d(df: pd.DataFrame, didx) -> pd.Series:
    if "pbMRQ" not in df.columns:
        return pd.Series(0.5, index=df.index)
    pb = pd.to_numeric(df["pbMRQ"], errors="coerce")
    out = _pctile_20(pb)
    return out.fillna(0.5)


@_register("amihud_illiq")
def amihud_illiq(df: pd.DataFrame, didx) -> pd.Series:
    if "amount" not in df.columns:
        return pd.Series(0.0, index=df.index)
    ill = df["close"].pct_change().abs() / df["amount"].replace(0, np.nan)
    return ill.rolling(5, min_periods=3).mean().fillna(0.0)


@_register("amount_ratio_5d")
def amount_ratio_5d(df: pd.DataFrame, didx) -> pd.Series:
    if "amount" not in df.columns:
        return pd.Series(0.0, index=df.index)
    a = pd.to_numeric(df["amount"], errors="coerce")
    return a / a.shift(1).rolling(5).mean() - 1


@_register("turn_pct_20d")
def turn_pct_20d(df: pd.DataFrame, didx) -> pd.Series:
    if "turn" not in df.columns:
        return pd.Series(0.5, index=df.index)
    t = pd.to_numeric(df["turn"], errors="coerce").fillna(0.0)
    return _pctile_20(t).fillna(0.5)


@_register("sharpe_20")
def sharpe_20(df: pd.DataFrame, didx) -> pd.Series:
    rets = df["close"].pct_change()
    std = rets.rolling(20, min_periods=10).std(ddof=0)
    r5 = df["close"].pct_change(5)
    return r5 / (std * np.sqrt(5) + 1e-9)


@_register("reversal_1d")
def reversal_1d(df: pd.DataFrame, didx) -> pd.Series:
    return -df["close"].pct_change()


@_register("vol_ratio_20")
def vol_ratio_20(df: pd.DataFrame, didx) -> pd.Series:
    tr = _tr(df["high"], df["low"], df["close"])
    a5 = tr.rolling(5).mean()
    a20 = tr.rolling(20).mean()
    return a5 / a20 - 1


@_register("gap_strength")
def gap_strength(df: pd.DataFrame, didx) -> pd.Series:
    return df["open"] / df["close"].shift(1) - 1


@_register("rel_strength_5d")
def rel_strength_5d(df: pd.DataFrame, didx) -> pd.Series:
    if "mkt_close" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df["close"].pct_change(5) - df["mkt_close"].pct_change(5)


@_register("rel_strength_20d")
def rel_strength_20d(df: pd.DataFrame, didx) -> pd.Series:
    if "mkt_close" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df["close"].pct_change(20) - df["mkt_close"].pct_change(20)


def panel_factor_list(group: str = None) -> list:
    from factors.sources import BASELINE_FACTORS, WAVE1_FACTORS
    if group == "baseline":
        return BASELINE_FACTORS
    if group == "wave1":
        return WAVE1_FACTORS
    return list(PANEL_FACTORS.keys())


def compute_factor_panels(
    data: Dict[str, pd.DataFrame], factor_names: List[str]
) -> Dict[str, pd.DataFrame]:
    """计算多只股票的多因子面板。

    Args:
        data: {symbol: 单股日K (date index, 含各列)}
        factor_names: 因子名列表

    Returns:
        {factor: DataFrame(index=date, columns=symbol)}
    """
    panels: Dict[str, Dict[str, pd.Series]] = {f: {} for f in factor_names}
    for sym, sdf in data.items():
        for f in factor_names:
            try:
                panels[f][sym] = PANEL_FACTORS[f](sdf, sdf.index)
            except Exception:
                panels[f][sym] = pd.Series(0.0, index=sdf.index)
    out = {}
    for f, cols in panels.items():
        out[f] = pd.DataFrame(cols).sort_index()
    return out


def compute_forward_returns(
    data: Dict[str, pd.DataFrame], lookahead: int
) -> pd.DataFrame:
    """每只股票的 N 日前向收益面板 (index=date, columns=symbol)。"""
    fwd = {}
    for sym, sdf in data.items():
        c = sdf["close"]
        fwd[sym] = c.shift(-lookahead) / c - 1
    return pd.DataFrame(fwd).sort_index()
