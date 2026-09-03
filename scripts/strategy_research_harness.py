"""通用策略研究回测引擎 (零模拟, 真实日线, 向量化).

三条路径:
  1. 截面排名组合 (ranking): 每个再平衡日按 score 取 top_k 等权, 周/月再平衡.
  2. 信号持仓 (signal position): entry(T)=T收盘买, exit(T)=T收盘卖.
  3. 事件固定持有 (fixed hold): T收盘买, T+hold_days 收盘/开盘卖.

基准: 上证 + 匹配universe等权. 成本 31bp 全周转 ([[transaction-costs-unified]]).

约定: 策略函数 fn(df) -> pd.Series 与 df 同 index (NaN=不入选, 越大越优先 / bool).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

COST_BPS = 31.0
MAIN_BOARD = ("sh.60", "sz.00")

WINDOWS = {
    "2015牛转股灾": "replay_data/daily_2015-01-01_2015-12-31.parquet",
    "2016熔断震荡": "replay_data/daily_2016-01-01_2016-12-31.parquet",
    "2017漂亮50":   "replay_data/daily_2017-01-01_2017-12-31.parquet",
    "2018熊":       "replay_data/daily_2018-01-01_2018-12-31.parquet",
    "2019牛":       "replay_data/daily_2019-01-01_2019-12-31.parquet",
    "2020牛转崩":   "replay_data/daily_2020-06-01_2021-02-28.parquet",
    "2021白马转小盘": "replay_data/daily_2021-01-01_2021-12-31.parquet",
    "2022熊":       "replay_data/daily_2022-01-01_2022-12-31.parquet",
    "2023震荡":     "replay_data/daily_2023-01-01_2023-12-31.parquet",
    "2024震荡":     "replay_data/daily_2024-01-01_2024-12-31.parquet",
    "2025-26现期":  "replay_data/daily_2025-10-08_2026-07-31.parquet",
}

REGIME = {
    "2019牛": "纯牛", "2021白马转小盘": "纯牛", "2025-26现期": "纯牛",
    "2015牛转股灾": "熊/崩", "2018熊": "熊/崩", "2020牛转崩": "熊/崩", "2022熊": "熊/崩",
    "2016熔断震荡": "震荡", "2017漂亮50": "震荡", "2023震荡": "震荡", "2024震荡": "震荡",
}


def load_base_window(fp: str, boards: Sequence[str] = MAIN_BOARD) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / fp)
    df["date"] = pd.to_datetime(df["date"])
    df["isST"] = df["isST"].astype(str)
    df = df[df["is_trade"] == 1]
    df = df[df["isST"] == "0"]
    df = df[df["symbol"].str.startswith(tuple(boards))]
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    return df


def load_index() -> pd.DataFrame:
    idx = pd.read_parquet(ROOT / "replay_data" / "index_series.parquet")
    idx = idx[idx["symbol"] == "sh.000001"][["date", "close", "pctChg"]].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["ret"] = idx["close"].pct_change()
    return idx


def load_fundamentals() -> pd.DataFrame:
    fp = ROOT / "replay_data" / "fundamentals.parquet"
    if not fp.exists():
        return pd.DataFrame(columns=["symbol", "statDate", "pubDate", "roeAvg", "YOYNI", "MBRevenue", "netProfit"])
    fund = pd.read_parquet(fp)
    for c in ("pubDate", "statDate"):
        if c in fund.columns:
            fund[c] = pd.to_datetime(fund[c])
    return fund


def pt_fundamental_col(df: pd.DataFrame, fund: pd.DataFrame, col: str) -> pd.Series:
    """point-in-time 基本面列: 对每个 (symbol, date) 取 pubDate<=date 的最新一期 col. 防前视."""
    if fund.empty or col not in fund.columns:
        return pd.Series(np.nan, index=df.index)
    f = fund[["symbol", "pubDate", col]].dropna(subset=[col]).sort_values("pubDate")
    f["symbol"] = f["symbol"].astype(str)
    key = df[["date", "symbol"]].sort_values("date").reset_index()
    key["symbol"] = key["symbol"].astype(str)
    merged = pd.merge_asof(key, f, left_on="date", right_on="pubDate", by="symbol", direction="backward")
    merged = merged.set_index("index").sort_index()
    return merged[col].reindex(df.index)


def pick_rebalance_dates(dates: Sequence[pd.Timestamp], freq: str) -> List[pd.Timestamp]:
    dates = sorted(pd.Series(dates).dropna().unique())
    if freq == "daily":
        return list(dates)
    if freq == "weekly":
        s = pd.Series(dates)
        return s.groupby([s.dt.isocalendar().year, s.dt.isocalendar().week]).apply(lambda x: x.iloc[-1]).tolist()
    if freq == "monthly":
        s = pd.Series(dates)
        return s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()
    raise ValueError(f"unknown freq {freq}")


def _held_to_daily(df: pd.DataFrame) -> pd.Series:
    """held(close of d) → 每日组合收益 (用 held_prev 赚 ret[d], 等权)."""
    df = df.copy()
    df["held_prev"] = df.groupby("symbol")["held"].shift(1, fill_value=False)
    sub = df[df["held_prev"]]
    return sub.groupby("date")["ret"].mean().sort_index()


def _turnover_cost_daily(df: pd.DataFrame) -> pd.Series:
    """由 held 状态变化算每日换手成本 (round-trip 31bp 拆到换手率)."""
    df = df.copy()
    held_prev = df.groupby("symbol")["held"].shift(1, fill_value=False)
    entered = df["held"] & (~held_prev.astype(bool))
    exited = held_prev.astype(bool) & (~df["held"])
    n_pos = df.groupby("date")["held"].sum()
    n_ent = df[entered].groupby("date")["held"].count()
    n_ext = df[exited].groupby("date")["held"].count()
    turnover = (n_ent.add(n_ext, fill_value=0)) / (2.0 * n_pos.clip(lower=1))
    return turnover * COST_BPS / 10000.0


# --------------------------------------------------------------------------- #
def backtest_ranking_portfolio(df, score, top_k, freq="monthly", cost_bps=COST_BPS):
    df = df.copy()
    df["score"] = score.values
    dates = sorted(df["date"].unique())
    rebal = pick_rebalance_dates(dates, freq)
    sel = {}
    for T in rebal:
        day = df[df["date"] == T].dropna(subset=["score"])
        sel[T] = set(day.nlargest(top_k, "score")["symbol"]) if not day.empty else set()
    rl = sorted(rebal)
    date_gov = {}
    for i, T in enumerate(rl):
        T_next = rl[i + 1] if i + 1 < len(rl) else pd.Timestamp.max
        for d in dates:
            if T <= d < T_next:
                date_gov[d] = T
    df["gov"] = df["date"].map(date_gov)
    df["held"] = False
    for T in rl:
        top = sel[T]
        if not top:
            continue
        idx_T = df.index[df["gov"] == T]
        isin = df.loc[idx_T, "symbol"].isin(top)
        df.loc[idx_T[isin.values], "held"] = True
    daily_gross = _held_to_daily(df)
    cost = pd.Series(0.0, index=daily_gross.index)
    for i in range(1, len(rl)):
        prev, cur = sel[rl[i - 1]], sel[rl[i]]
        if not prev and not cur:
            continue
        replaced = len(cur - prev) if (prev and cur) else len(cur)
        if rl[i] in cost.index and top_k:
            cost[rl[i]] = cost_bps / 10000.0 * (replaced / top_k)
    daily_net = daily_gross - cost.reindex(daily_gross.index, fill_value=0.0)
    info = {"n_rebalance": len(rl), "turnover_cost_bps_equiv": round(float(cost.sum()) * 10000, 2)}
    return daily_net, info


def backtest_signal_position(df, entry, exit_, cost_bps=COST_BPS):
    df = df.copy()
    df["entry"] = entry.astype(bool)
    df["exit"] = exit_.astype(bool)

    def state(g):
        held = 0
        out = np.zeros(len(g), dtype=bool)
        e = g["entry"].values
        x = g["exit"].values
        for i in range(len(g)):
            if e[i] and not held:
                held = 1
            elif x[i] and held:
                held = 0
            out[i] = bool(held)
        return pd.Series(out, index=g.index)

    df["held"] = df.groupby("symbol", group_keys=False).apply(state)
    daily_gross = _held_to_daily(df)
    daily_cost = _turnover_cost_daily(df)
    daily_net = daily_gross.sub(daily_cost.reindex(daily_gross.index, fill_value=0.0), fill_value=0.0)
    info = {"n_pos_days": int(df["held"].sum()), "avg_held": round(float(df.groupby("date")["held"].sum().mean()), 2)}
    return daily_net, info


def backtest_fixed_hold(df, signal, hold_days=1, exit_price="close", cost_bps=COST_BPS):
    df = df.copy()
    df["sig"] = signal.astype(bool)
    col = "open" if exit_price == "open" else "close"
    df["exit"] = df.groupby("symbol")[col].shift(-hold_days)
    picks = df[df["sig"] & df["exit"].notna() & (df["exit"] > 0)].copy()
    picks["net_ret"] = picks["exit"] / picks["close"] - 1.0 - cost_bps / 10000.0
    return picks, {"n_trades": int(len(picks))}


# --------------------------------------------------------------------------- #
def compute_metrics(daily_ret):
    r = daily_ret.dropna()
    if len(r) == 0:
        return {"total": np.nan, "ann": np.nan, "sharpe": np.nan, "maxdd": np.nan, "n_days": 0}
    total = float((1.0 + r).prod() - 1.0)
    n = len(r)
    ann = float((1.0 + total) ** (252.0 / n) - 1.0) if n > 0 else np.nan
    std = float(r.std())
    sharpe = float(r.mean() / std * np.sqrt(252.0)) if std > 1e-12 else np.nan
    eq = (1.0 + r).cumprod()
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {"total": round(total, 4), "ann": round(ann, 4), "sharpe": round(sharpe, 3),
            "maxdd": round(maxdd, 4), "n_days": n}


def window_universe_benchmark(df, freq="monthly"):
    """匹配universe等权 (同基础过滤全池等权, 与策略同再平衡口径+同成本, apples-to-apples)."""
    n = df["symbol"].nunique()
    score = pd.Series(1.0, index=df.index)
    daily_net, _ = backtest_ranking_portfolio(df, score, top_k=n, freq=freq)
    return daily_net


def _idx_total(idx, t0, t1):
    seg = idx[(idx["date"] >= t0) & (idx["date"] <= t1)]
    r = seg["ret"].dropna()
    return float((1.0 + r).prod() - 1.0) if len(r) else np.nan


def _benchmarks(daily_ret, df, idx, window):
    t0, t1 = daily_ret.index.min(), daily_ret.index.max()
    uni = window_universe_benchmark(df)
    uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
    uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
    idx_total = _idx_total(idx, t0, t1)
    m = compute_metrics(daily_ret)
    m.update({
        "universe_total": round(uni_total, 4),
        "index_total": round(idx_total, 4),
        "vs_universe": round(m["total"] - uni_total, 4) if not np.isnan(m["total"]) else np.nan,
        "vs_index": round(m["total"] - idx_total, 4) if not np.isnan(m["total"]) else np.nan,
        "regime": REGIME.get(window, ""),
    })
    return m


def run_window_ranking(df, idx, score_fn, top_k, freq, window):
    score = score_fn(df)
    daily_ret, info = backtest_ranking_portfolio(df, score, top_k, freq)
    m = _benchmarks(daily_ret, df, idx, window)
    m.update(info)
    return m


def run_window_signal(df, idx, entry_fn, exit_fn, window):
    entry = entry_fn(df)
    exit_ = exit_fn(df)
    daily_ret, info = backtest_signal_position(df, entry, exit_)
    m = _benchmarks(daily_ret, df, idx, window)
    m.update(info)
    return m


def run_window_fixed_hold(df, idx, signal_fn, hold_days, exit_price, window):
    picks, info = backtest_fixed_hold(df, signal_fn(df), hold_days, exit_price)
    if picks.empty:
        return {"n_trades": 0, "regime": REGIME.get(window, "")}
    avg = float(picks["net_ret"].mean())
    med = float(picks["net_ret"].median())
    win = float((picks["net_ret"] > 0).mean())
    t0, t1 = df["date"].min(), df["date"].max()
    idx_total = _idx_total(idx, t0, t1)
    return {"n_trades": int(len(picks)), "avg_net_ret": round(avg, 4), "median_net_ret": round(med, 4),
            "win_rate": round(win, 4), "index_total": round(idx_total, 4), "regime": REGIME.get(window, "")}


def dump_json(payload, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
