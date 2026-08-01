"""
v3 去参数化策略 — 修复 v2 过拟合问题

原则:
  1. 用标准指标默认值, 不做手动阈值调优
  2. 简单入场条件 (越简单越难过拟合)
  3. 统一出场逻辑: ATR宽止损 + 移动止盈 (复用动量v2验证通过的模式)
  4. 不加RSI/量确认过滤 (这些都是过拟合源)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List


def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    rs = gain.ewm(span=period, adjust=False).mean() / (loss.ewm(span=period, adjust=False).mean() + 1e-10)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = pd.Series(df["high"]), pd.Series(df["low"]), pd.Series(df["close"])
    tr = pd.concat([(h-l).abs(), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _exec(df: pd.DataFrame):
    opens = pd.Series(df["open"].values)
    closes = pd.Series(df["close"].values)
    highs = pd.Series(df["high"].values)
    ep = opens.shift(-1).copy(); ep.iloc[-1] = closes.iloc[-1]
    xp = opens.shift(-1).copy(); xp.iloc[-1] = closes.iloc[-1]
    lu = closes >= highs.shift(1) * 1.099
    ld = closes <= (highs.shift(1) * 0.9 * 0.99)
    ep[lu] = np.nan; xp[ld] = np.nan
    return ep, xp


def _stats(trades: List[float]) -> Dict[str, Any]:
    COST = 0.0031
    if not trades:
        return {"signals": 0, "win_rate": 0, "profit_factor": 1,
                "expected_value": 0, "sharpe": 0, "max_dd": 0,
                "avg_win": 0, "avg_loss": 0, "cost_adjusted": True}
    net = [t - COST * 100 for t in trades]
    wins = [t for t in net if t > 0]
    losses = [abs(t) for t in net if t <= 0]
    total = len(net)
    wr = len(wins) / total if total > 0 else 0
    pf = (sum(wins) / (sum(losses) + 1e-10)) if losses else 1.0
    ev = wr * (np.mean(wins) if wins else 0) - (1-wr) * (np.mean(losses) if losses else 1)
    rets = pd.Series([t/100 for t in net])
    sr = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(252))
    cumsum = np.cumsum(net)
    dd = abs(float(np.min(cumsum - np.maximum.accumulate(cumsum))))
    return {
        "signals": total, "win_rate": wr, "profit_factor": pf,
        "expected_value": ev, "sharpe": sr, "max_dd": dd,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "cost_adjusted": True,
    }


# ═══════════════════════════════════════════════════════════════
# 统一的稳健出场逻辑 (复用动量v2验证通过的模式)
# ═══════════════════════════════════════════════════════════════

def _check_exit(closes, i, entry_price, stop_price, peak_price, atr_mult=2.0):
    """返回 (should_exit, reason)"""
    cur_pnl = (closes.iloc[i] / entry_price - 1) * 100
    peak = max(peak_price, closes.iloc[i])

    hit_stop = closes.iloc[i] <= stop_price
    trail_stop = ((peak / entry_price - 1) * 100 > 8 and
                  (closes.iloc[i] / peak - 1) * 100 < -5)

    return hit_stop or trail_stop or cur_pnl < -10, cur_pnl, peak


def _update_stop(entry_price, stop_price, cur_pnl):
    """移动止损上移"""
    if cur_pnl > 4:
        return max(stop_price, entry_price * 1.02)
    return stop_price


# ═══════════════════════════════════════════════════════════════
# v3 策略
# ═══════════════════════════════════════════════════════════════


def bollinger_reversal_v3(df: pd.DataFrame) -> Dict[str, Any]:
    """布林带回归 v3: 标准BB(20,2), 下轨买入→中轨卖出, 无任何过滤器"""
    closes = pd.Series(df["close"].values)
    n = len(closes)
    if n < 30:
        return {"signals": 0}

    ma20 = closes.rolling(20).mean()
    std20 = closes.rolling(20).std()
    lower = ma20 - 2 * std20
    atr = _atr(df, 14)
    ep, xp = _exec(df)

    trades = []
    in_pos = False
    entry_price = 0.0
    stop_price = 0.0
    peak_price = 0.0

    for i in range(25, n):
        if not in_pos:
            if closes.iloc[i] <= lower.iloc[i]:
                px = ep[i]
                if not np.isnan(px) and atr.iloc[i] > 0:
                    in_pos = True
                    entry_price = px
                    stop_price = px - 2.5 * atr.iloc[i]
                    peak_price = px
        else:
            exit_sig, cur_pnl, peak_price = _check_exit(closes, i, entry_price, stop_price, peak_price)
            back_ma = closes.iloc[i] >= ma20.iloc[i]  # 回归中轨
            if exit_sig or back_ma:
                px = xp[i]
                if not np.isnan(px):
                    trades.append((px / entry_price - 1) * 100)
                    in_pos = False
                    peak_price = 0.0
            else:
                stop_price = _update_stop(entry_price, stop_price, cur_pnl)

    if in_pos and n > 0:
        trades.append((closes.iloc[-1] / entry_price - 1) * 100)
    return _stats(trades)


def rsi_reversal_v3(df: pd.DataFrame) -> Dict[str, Any]:
    """RSI回归 v3: 标准RSI<30买入, RSI>55卖出, 无确认过滤器"""
    closes = pd.Series(df["close"].values)
    n = len(closes)
    if n < 20:
        return {"signals": 0}

    rsi = _rsi(closes, 14)
    atr = _atr(df, 14)
    ep, xp = _exec(df)

    trades = []
    in_pos = False
    entry_price = 0.0
    stop_price = 0.0
    peak_price = 0.0

    for i in range(20, n):
        if not in_pos:
            if rsi.iloc[i] < 30:
                px = ep[i]
                if not np.isnan(px) and atr.iloc[i] > 0:
                    in_pos = True
                    entry_price = px
                    stop_price = px - 2 * atr.iloc[i]
                    peak_price = px
        else:
            exit_sig, cur_pnl, peak_price = _check_exit(closes, i, entry_price, stop_price, peak_price)
            rsi_out = rsi.iloc[i] > 55
            if exit_sig or rsi_out:
                px = xp[i]
                if not np.isnan(px):
                    trades.append((px / entry_price - 1) * 100)
                    in_pos = False
                    peak_price = 0.0
            else:
                stop_price = _update_stop(entry_price, stop_price, cur_pnl)

    if in_pos and n > 0:
        trades.append((closes.iloc[-1] / entry_price - 1) * 100)
    return _stats(trades)


def macd_trend_v3(df: pd.DataFrame) -> Dict[str, Any]:
    """MACD趋势 v3: 标准MACD(12,26,9)金叉/死叉, 无过滤器"""
    closes = pd.Series(df["close"].values)
    n = len(closes)
    if n < 40:
        return {"signals": 0}

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    histogram = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    atr = _atr(df, 14)
    ep, xp = _exec(df)

    trades = []
    in_pos = False
    entry_price = 0.0
    stop_price = 0.0
    peak_price = 0.0

    for i in range(30, n):
        if not in_pos:
            golden = histogram.iloc[i] > 0 and histogram.iloc[i-1] <= 0
            if golden:
                px = ep[i]
                if not np.isnan(px) and atr.iloc[i] > 0:
                    in_pos = True
                    entry_price = px
                    stop_price = px - 3 * atr.iloc[i]
                    peak_price = px
        else:
            exit_sig, cur_pnl, peak_price = _check_exit(closes, i, entry_price, stop_price, peak_price)
            dead = histogram.iloc[i] < 0 and histogram.iloc[i-1] >= 0
            if exit_sig or dead:
                px = xp[i]
                if not np.isnan(px):
                    trades.append((px / entry_price - 1) * 100)
                    in_pos = False
                    peak_price = 0.0
            else:
                stop_price = _update_stop(entry_price, stop_price, cur_pnl)

    if in_pos and n > 0:
        trades.append((closes.iloc[-1] / entry_price - 1) * 100)
    return _stats(trades)


def multifactor_v3(df: pd.DataFrame) -> Dict[str, Any]:
    """多因子融合 v3: 等权5因子, 20日评估, 无调优阈值"""
    closes = pd.Series(df["close"].values)
    volumes = pd.Series(df["volume"].values)
    highs = pd.Series(df["high"].values)
    n = len(closes)
    if n < 80:
        return {"signals": 0}

    rsi = _rsi(closes, 14)
    atr = _atr(df, 14)
    ep, xp = _exec(df)

    trades = []
    in_pos = False
    entry_price = 0.0
    stop_price = 0.0
    peak_price = 0.0
    hold_days = 0

    for i in range(80, n):
        if i % 20 != 0:
            if in_pos:
                hold_days += 1
            continue

        mom = np.clip((closes.iloc[i] / closes.iloc[max(0,i-10)] - 1) * 100, -15, 15) / 15
        r = rsi.iloc[i]
        rsi_s = (50 - r) / 25 if not np.isnan(r) else 0  # <50偏多
        vol_r = volumes.iloc[i] / (volumes.iloc[i-20:i].mean() + 1e-10)
        vol_s = np.clip(vol_r - 1, -0.5, 1)
        ma20 = closes.iloc[i-20:i].mean()
        trend_s = np.clip((closes.iloc[i] / (ma20 + 1e-10) - 1) * 5, -2, 2) / 2
        peak_60 = highs.iloc[max(0,i-60):i].max()
        qual_s = np.clip((closes.iloc[i] / (peak_60 + 1e-10) - 1) * 5, -2, 2) / 2

        composite = 0.2 * (mom + rsi_s + vol_s + trend_s + qual_s)

        if not in_pos and composite > 0.15:
            px = ep[i]
            if not np.isnan(px) and atr.iloc[i] > 0:
                in_pos = True
                entry_price = px
                stop_price = px - 3 * atr.iloc[i]
                peak_price = px
                hold_days = 0
        elif in_pos:
            exit_sig, cur_pnl, peak_price = _check_exit(closes, i, entry_price, stop_price, peak_price)
            if exit_sig or composite < -0.15 or hold_days >= 20:
                px = xp[i]
                if not np.isnan(px):
                    trades.append((px / entry_price - 1) * 100)
                    in_pos = False
                    peak_price = 0.0
            else:
                stop_price = _update_stop(entry_price, stop_price, cur_pnl)

    if in_pos and n > 0:
        trades.append((closes.iloc[-1] / entry_price - 1) * 100)
    return _stats(trades)


def pv_tension_v3(df: pd.DataFrame) -> Dict[str, Any]:
    """价量张力反转 v3: 虚涨→等回调→抄底, 去参数化"""
    closes = pd.Series(df["close"].values)
    volumes = pd.Series(df["volume"].values)
    n = len(closes)
    if n < 30:
        return {"signals": 0}

    rsi = _rsi(closes, 14)
    atr = _atr(df, 14)
    vol_ma20 = volumes.rolling(20).mean()
    ma20 = closes.rolling(20).mean()
    ep, xp = _exec(df)

    trades = []
    in_pos = False
    entry_price = 0.0
    stop_price = 0.0
    peak_price = 0.0
    watch = False

    for i in range(30, n):
        if not in_pos and not watch:
            chg_5 = (closes.iloc[i] / closes.iloc[max(0,i-5)] - 1) * 100
            vol_dry = volumes.iloc[i] < vol_ma20.iloc[i] * 0.8
            # 虚涨: 连涨但缩量
            if chg_5 > 3 and vol_dry:
                watch = True
        elif watch:
            # 等回调到位: RSI回落 + 回到均线下方
            if rsi.iloc[i] < 50 and closes.iloc[i] < ma20.iloc[i]:
                px = ep[i]
                if not np.isnan(px) and atr.iloc[i] > 0:
                    in_pos = True
                    entry_price = px
                    stop_price = px - 2 * atr.iloc[i]
                    peak_price = px
                    watch = False
            elif rsi.iloc[i] > 70:  # 不回调继续涨,放弃
                watch = False
        elif in_pos:
            exit_sig, cur_pnl, peak_price = _check_exit(closes, i, entry_price, stop_price, peak_price)
            back_ma = closes.iloc[i] >= ma20.iloc[i] and cur_pnl > 0
            if exit_sig or back_ma:
                px = xp[i]
                if not np.isnan(px):
                    trades.append((px / entry_price - 1) * 100)
                    in_pos = False
                    peak_price = 0.0
            else:
                stop_price = _update_stop(entry_price, stop_price, cur_pnl)

    if in_pos and n > 0:
        trades.append((closes.iloc[-1] / entry_price - 1) * 100)
    return _stats(trades)


# ═══════════════════════════════════════════════════════════════
# v3 注册表 (替换过拟合的v2版本)
# ═══════════════════════════════════════════════════════════════

V3_BACKTESTERS = {
    "bollinger_reversal_v3": bollinger_reversal_v3,
    "rsi_reversal_v3": rsi_reversal_v3,
    "macd_trend_v3": macd_trend_v3,
    "multi_factor_v3": multifactor_v3,
    "pv_tension_v3": pv_tension_v3,
}

V3_NAMES = {
    "bollinger_reversal_v3": "布林回归v3",
    "rsi_reversal_v3": "RSI回归v3",
    "macd_trend_v3": "MACD趋势v3",
    "multi_factor_v3": "多因子v3",
    "pv_tension_v3": "价量张力v3",
}
