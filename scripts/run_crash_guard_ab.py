"""崩溃断路器 A/B (预注册 §十): 冷落低波精选 top5 + 组合级回撤断路器。

问题: 冷落低波精选主判据② 唯一漏点 = 2019 纯牛 maxdd −30.2% (个股集中踩雷, 非市场崩盘)。
加一层回撤断路器能否「牛市不崩」而不破坏「熊/震荡跑赢」?

两个机制 (都预注册):
  V1 指数级: 每再平衡日, 上证距其 250 日滚动高点回撤 ≤ −20% → 本月空仓; 否则持 top5。
  V2 自净值级: 每再平衡日, 组合 NAV 距滚动峰值回撤 ≤ −20% → 本月空仓; 否则持 top5。

判据 (预注册 §十, 跑完不调): 断路器「有价值」⇔ 两条同时成立:
  (i) 纯牛 3 窗(2019/2020/2024) 全收益>0 且 maxdd ≥ −30%;
  (ii) 非纯牛 8 窗(2015/2016/2017/2018/2021/2022/2023/2025-26) ≥7/8 跑赢 universe。
  只压回撤不保收益 = 失败 (空仓当然不回撤, 同义反复不算赢)。

用法: python scripts/run_crash_guard_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import strategy_research_harness as H  # noqa: E402

# §六 客观 regime 重标注 (跑之前写死)
OBJECTIVE = {
    "2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
    "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
    "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
    "2024震荡": "纯牛", "2025-26现期": "震荡",
}
THRESHOLD = -0.20  # 回撤阈值 (预注册)
LOOKBACK = 250     # 指数回撤 lookback 交易日 (预注册)


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# 与 run_survivorship_test.py 一致的冷落低波 score
def _factors(df):
    out = df[["date", "symbol", "turn", "amount", "close"]].copy()
    s = df.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    out["std20"] = s.reindex(df.index)
    fmkt = df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8
    out["log_mkt"] = np.log1p(fmkt)
    return out


def _cold_lowvol(df):
    f = _factors(df)
    r_turn = f["turn"].groupby(df["date"]).rank(pct=True)
    r_std = f["std20"].groupby(df["date"]).rank(pct=True)
    r_mkt = f["log_mkt"].groupby(df["date"]).rank(pct=True)
    return -(r_turn + r_std + r_mkt)


def _index_drawdown(idx: pd.DataFrame, T, lookback: int = LOOKBACK) -> float:
    hist = idx[idx["date"] <= T]
    if hist.empty:
        return 0.0
    win = hist["close"].iloc[-lookback:]
    return float(win.iloc[-1] / win.max() - 1.0)


def _held_to_daily(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["held_prev"] = df.groupby("symbol")["held"].shift(1, fill_value=False)
    sub = df[df["held_prev"]]
    return sub.groupby("date")["ret"].mean().sort_index()


def _daily_from_sel(df, sel, rebal, top_k, cost_bps=H.COST_BPS):
    """由 sel (rebalance 日 -> 持仓 symbol 集, 空=现金) 生成每日净收益."""
    df = df.copy()
    dates = sorted(df["date"].unique())
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
        top = sel.get(T, set())
        if not top:
            continue
        idx_T = df.index[df["gov"] == T]
        isin = df.loc[idx_T, "symbol"].isin(top)
        df.loc[idx_T[isin.values], "held"] = True
    daily_gross = _held_to_daily(df)
    cost = pd.Series(0.0, index=daily_gross.index)
    for i in range(1, len(rl)):
        prev, cur = sel.get(rl[i - 1], set()), sel.get(rl[i], set())
        changed = max(len(cur - prev), len(prev - cur))
        if rl[i] in cost.index and top_k:
            cost[rl[i]] = cost_bps / 10000.0 * (changed / top_k)
    return daily_gross - cost.reindex(daily_gross.index, fill_value=0.0)


def backtest_index_breaker(df, score, top_k, freq, idx, threshold=THRESHOLD):
    df = df.copy()
    df["score"] = score.values
    dates = sorted(df["date"].unique())
    rebal = H.pick_rebalance_dates(dates, freq)
    sel = {}
    for T in rebal:
        if _index_drawdown(idx, T) <= threshold:
            sel[T] = set()
        else:
            day = df[df["date"] == T].dropna(subset=["score"])
            sel[T] = set(day.nlargest(top_k, "score")["symbol"]) if not day.empty else set()
    daily_net = _daily_from_sel(df, sel, rebal, top_k)
    n_cash = sum(1 for T in sel if not sel[T])
    return daily_net, {"n_cash_months": n_cash, "n_rebalance": len(rebal)}


def _mean_ret(df, symbols, d):
    if not symbols:
        return 0.0
    day = df[(df["date"] == d) & (df["symbol"].isin(symbols))]
    r = day["ret"].dropna()
    return float(r.mean()) if len(r) else 0.0


def backtest_selfnav_breaker(df, score, top_k, freq, threshold=THRESHOLD):
    df = df.copy()
    df["score"] = score.values
    dates = sorted(df["date"].unique())
    rebal = H.pick_rebalance_dates(dates, freq)
    rl = sorted(rebal)
    top_at = {}
    for T in rl:
        day = df[df["date"] == T].dropna(subset=["score"])
        top_at[T] = set(day.nlargest(top_k, "score")["symbol"]) if not day.empty else set()
    sel = {}
    nav = 1.0
    peak = 1.0
    for i, T in enumerate(rl):
        dd = nav / peak - 1.0 if peak > 0 else 0.0
        sel[T] = set() if dd <= threshold else top_at[T]
        T_next = rl[i + 1] if i + 1 < len(rl) else None
        if T_next is None:
            period = [d for d in dates if d > T]
        else:
            period = [d for d in dates if T < d <= T_next]
        for d in period:
            r = _mean_ret(df, sel[T], d)
            nav *= (1.0 + r)
            peak = max(peak, nav)
    daily_net = _daily_from_sel(df, sel, rl, top_k)
    n_cash = sum(1 for T in sel if not sel[T])
    return daily_net, {"n_cash_months": n_cash, "n_rebalance": len(rl)}


def _metrics(daily_net, df, idx):
    t0, t1 = daily_net.index.min(), daily_net.index.max()
    uni = H.window_universe_benchmark(df)
    uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
    uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
    idx_total = H._idx_total(idx, t0, t1)
    m = H.compute_metrics(daily_net)
    m.update({
        "universe_total": round(uni_total, 4),
        "index_total": round(idx_total, 4),
        "vs_universe": round(m["total"] - uni_total, 4) if not np.isnan(m["total"]) else np.nan,
        "vs_index": round(m["total"] - idx_total, 4) if not np.isnan(m["total"]) else np.nan,
    })
    return m


def _apply_criterion(res, label):
    niu = [w for w in H.WINDOWS if OBJECTIVE[w] == "纯牛"]
    nonniu = [w for w in H.WINDOWS if OBJECTIVE[w] != "纯牛"]
    i_pass = all(res[w]["total"] > 0 and res[w]["maxdd"] >= -0.30 for w in niu)
    i_details = "; ".join(f"{w} total={res[w]['total']*100:+.1f}% maxdd={res[w]['maxdd']*100:.1f}%"
                          for w in niu)
    ii_win = sum(1 for w in nonniu if res[w]["vs_universe"] >= 0)
    ii_pass = ii_win >= 7
    return i_pass, ii_pass, ii_win, i_details


def main() -> int:
    _force_utf8()
    idx = H.load_index()
    results = {"base": {}, "V1_index": {}, "V2_selfnav": {}}
    cash_info = {"V1_index": {}, "V2_selfnav": {}}

    print("=" * 104)
    print("崩溃断路器 A/B — 冷落低波精选 top5 (月再平衡, 31bp; 阈值 −20%)")
    print("=" * 104)
    hdr = f"{'窗口':<12} {'regime':<5} | {'base总':>8} {'base回撤':>8} {'base vsU':>8} | " \
          f"{'V1总':>8} {'V1回撤':>8} {'V1 vsU':>8} | {'V2总':>8} {'V2回撤':>8} {'V2 vsU':>8}"
    print(hdr)
    print("-" * 104)

    for window in H.WINDOWS:
        fp = H.WINDOWS[window]
        df = H.load_base_window(fp)
        score = _cold_lowvol(df)

        m_base = H.run_window_ranking(df, idx, _cold_lowvol, 5, "monthly", window)
        d1, i1 = backtest_index_breaker(df, score, 5, "monthly", idx)
        m1 = _metrics(d1, df, idx)
        m1.update(i1)
        d2, i2 = backtest_selfnav_breaker(df, score, 5, "monthly")
        m2 = _metrics(d2, df, idx)
        m2.update(i2)

        results["base"][window] = m_base
        results["V1_index"][window] = m1
        results["V2_selfnav"][window] = m2
        cash_info["V1_index"][window] = i1["n_cash_months"]
        cash_info["V2_selfnav"][window] = i2["n_cash_months"]

        def pct(x, nd=1):
            return f"{x*100:+.{nd}f}%" if not (isinstance(x, float) and np.isnan(x)) else "  nan "

        print(f"{window:<12} {OBJECTIVE[window]:<5} | {pct(m_base['total']):>8} "
              f"{pct(m_base['maxdd']):>8} {pct(m_base['vs_universe']):>8} | "
              f"{pct(m1['total']):>8} {pct(m1['maxdd']):>8} {pct(m1['vs_universe']):>8} | "
              f"{pct(m2['total']):>8} {pct(m2['maxdd']):>8} {pct(m2['vs_universe']):>8}")

    print("\n" + "=" * 104)
    print("预注册判据 (§十): 断路器「有价值」⇔ 主判据②修复 且 主判据①不破 (两条同时)")
    print("=" * 104)
    for label, res in [("base", results["base"]), ("V1指数级", results["V1_index"]),
                       ("V2自净值级", results["V2_selfnav"])]:
        i_pass, ii_pass, ii_win, i_details = _apply_criterion(res, label)
        verdict = "有价值" if (i_pass and ii_pass) else "失败"
        print(f"\n[{label}]")
        print(f"  (i) 主判据② 纯牛3窗: {'PASS' if i_pass else 'FAIL'}   ({i_details})")
        print(f"  (ii) 主判据① 非纯牛8窗跑赢 universe: {ii_win}/8 → {'PASS (≥7/8)' if ii_pass else 'FAIL'}")
        print(f"  → 结论: {verdict}")

    print("\n空仓月数诊断 (不 gate):")
    for window in H.WINDOWS:
        print(f"  {window:<12} V1指数 {cash_info['V1_index'][window]:>2} 月空仓 | "
              f"V2自净值 {cash_info['V2_selfnav'][window]:>2} 月空仓")

    H.dump_json({"threshold": THRESHOLD, "lookback": LOOKBACK, "objective": OBJECTIVE,
                 "base": results["base"], "V1_index": results["V1_index"],
                 "V2_selfnav": results["V2_selfnav"], "cash_months": cash_info},
                ROOT / "reports" / "crash_guard_ab_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
