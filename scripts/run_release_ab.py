"""限售解禁 个股风险回避 A/B — 冷落低波精选 (预注册 §十六).

规则 (写死): 再平衡日 T, 从候选池剔除 解禁时间∈[T+1, T+30] 且 占解禁前流通市值比例 ≥0.10 的票;
其余票按 冷落低波 score 取 top5. 2015 窗无解禁库 ⇒ 回避 no-op (R=基线, 如实披露).

判据 (§十六): ① 非纯牛8窗 R vs 匹配universe ≥0 ≥7/8 (不割超额); ② 纯牛3窗 R 全正且 dd≥−30.
副指标: R vs 基线 Sharpe/maxdd 配对差分, 剔除票数/受影响月.
用法: python scripts/run_release_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import strategy_research_harness as H  # noqa: E402

K = 30        # 未来 30 日历日
THETA = 0.10  # 解禁占流通市值 ≥10% 视为实质压力
COST_BPS = 31.0

OBJECTIVE = {
    "2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
    "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
    "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
    "2024震荡": "纯牛", "2025-26现期": "震荡",
}
PURE_BULL = [w for w, r in OBJECTIVE.items() if r == "纯牛"]
NON_PURE = [w for w, r in OBJECTIVE.items() if r != "纯牛"]


def _force_utf8():
    for n in ("stdout", "stderr"):
        s = getattr(sys, n, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _load_release() -> pd.DataFrame:
    fp = ROOT / "replay_data" / "release.parquet"
    if not fp.exists():
        return pd.DataFrame(columns=["_code", "_date", "pressure"])
    rel = pd.read_parquet(fp)
    if rel.empty:
        return rel
    col = next((c for c in rel.columns if "占解禁前流通市值比例" in c), None)
    rel["pressure"] = rel[col].astype(float) if col else 0.0
    rel["_code"] = rel["_code"].astype(str).str.zfill(6)
    rel["_date"] = pd.to_datetime(rel["_date"]).dt.date
    return rel[["_code", "_date", "pressure"]].copy()


def _exclude_on(ref, rel) -> set:
    """在再平衡日 ref 剔除 {解禁 ∈(ref,ref+30] 且 pressure≥θ} 的 6 位 code."""
    if rel.empty:
        return set()
    lo = ref.date() + pd.Timedelta(days=1)
    hi = ref.date() + pd.Timedelta(days=K)
    m = (rel["_date"] > lo) & (rel["_date"] <= hi) & (rel["pressure"] >= THETA)
    return set(rel.loc[m, "_code"])


def backtest_ranking_with_release_avoid(df, score, top_k, rel, freq="monthly", cost_bps=COST_BPS):
    df = df.copy()
    df["score"] = score.values
    df["_code6"] = df["symbol"].str.split(".").str[-1].str.zfill(6)
    dates = sorted(df["date"].unique())
    rebal = H.pick_rebalance_dates(dates, freq)
    sel, excluded = {}, {}
    for T in rebal:
        day = df[df["date"] == T].dropna(subset=["score"])
        if day.empty:
            sel[T], excluded[T] = set(), set()
            continue
        exc = _exclude_on(T, rel)
        excluded[T] = exc
        cand = day[~day["_code6"].isin(exc)]
        sel[T] = set(cand.nlargest(top_k, "score")["symbol"]) if not cand.empty else set()
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
    daily_gross = H._held_to_daily(df)
    cost = pd.Series(0.0, index=daily_gross.index)
    for i in range(1, len(rl)):
        prev, cur = sel[rl[i - 1]], sel[rl[i]]
        if not prev and not cur:
            continue
        replaced = len(cur - prev) if (prev and cur) else len(cur)
        if rl[i] in cost.index and top_k:
            cost[rl[i]] = cost_bps / 10000.0 * (replaced / top_k)
    daily_net = daily_gross - cost.reindex(daily_gross.index, fill_value=0.0)
    n_excluded_months = sum(1 for T in rl if excluded.get(T))
    total_excluded = sum(len(v) for v in excluded.values())
    info = {"n_rebalance": len(rl), "n_affected_months": n_excluded_months,
            "n_excluded_picks": total_excluded,
            "turnover_cost_bps_equiv": round(float(cost.sum()) * 10000, 2)}
    return daily_net, info


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


def _run_window(df, idx, rel, window, avoid):
    score = _cold_lowvol(df)
    if avoid:
        daily_net, info = backtest_ranking_with_release_avoid(df, score, 5, rel)
    else:
        daily_net, info = H.backtest_ranking_portfolio(df, score, 5, "monthly", cost_bps=COST_BPS)
        info = {"n_affected_months": 0, "n_excluded_picks": 0, **info}
    m = H.compute_metrics(daily_net)
    t0, t1 = daily_net.index.min(), daily_net.index.max()
    uni = H.window_universe_benchmark(df)
    uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
    uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
    idx_total = H._idx_total(idx, t0, t1)
    m.update({"universe_total": round(uni_total, 4), "index_total": round(idx_total, 4),
              "vs_universe": round(m["total"] - uni_total, 4) if not np.isnan(m["total"]) else np.nan,
              "vs_index": round(m["total"] - idx_total, 4) if not np.isnan(m["total"]) else np.nan,
              "regime": OBJECTIVE[window]})
    m.update(info)
    return m


def main() -> int:
    _force_utf8()
    idx = H.load_index()
    rel = _load_release()
    print(f"回避规则: 解禁∈(T,T+{K}d] 且 占流通≥{THETA} 剔除候选 (预注册§十六, K/θ 写死)")
    print(f"解禁库: {len(rel)} 条, 覆盖 {rel['_date'].min() if len(rel) else '-'} ~ {rel['_date'].max() if len(rel) else '-'}")
    print("WARN: 2015 窗无解禁库, 回避 no-op (R=基线)")
    print("=" * 118)
    res = {"baseline": {}, "release_avoid": {}}
    print(f"{'窗口':<12}{'regime':<5} | {'B0总/vsU/sr/dd':>30} | {'R总/vsU/sr/dd':>30} | {'受影响月/剔除票':>12}")
    print("-" * 118)
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        b = _run_window(df, idx, rel, window, avoid=False)
        r = _run_window(df, idx, rel, window, avoid=True)
        res["baseline"][window], res["release_avoid"][window] = b, r
        pb = lambda w: f"{w['total']*100:+5.1f}/{w['vs_universe']*100:+5.1f}/{w['sharpe']:+4.2f}/{w['maxdd']*100:5.1f}"
        print(f"{window:<12}{OBJECTIVE[window]:<5} | {pb(b):>30} | {pb(r):>30} | {r['n_affected_months']}月/{r['n_excluded_picks']}票")

    print("\n" + "=" * 118)
    print("预注册判据 (§十六) — 回避臂 R")
    for label, d in res.items():
        g1 = sum(1 for w in NON_PURE if d[w]["vs_universe"] is not None and d[w]["vs_universe"] >= 0)
        g2 = all(not np.isnan(d[w]["total"]) and d[w]["total"] > 0 and d[w]["maxdd"] >= -0.30 for w in PURE_BULL)
        g2d = "; ".join(f"{w} tot={d[w]['total']*100:+.1f}% dd={d[w]['maxdd']*100:.1f}% sr={d[w]['sharpe']}" for w in PURE_BULL)
        print(f"[{label}] ① 非纯牛8窗 vsU≥0: {g1}/8  ② 纯牛3窗全正&dd≥−30: {'PASS' if g2 else 'FAIL'}  ({g2d})")
    print("\n副指标 (诊断): R vs B0 逐窗 Sharpe/maxdd 配对差分")
    for w in H.WINDOWS:
        b, r = res["baseline"][w], res["release_avoid"][w]
        if r["n_affected_months"]:
            print(f"  {w:<12} Δsr={r['sharpe']-b['sharpe']:+.3f} Δdd={r['maxdd']-b['maxdd']:+.2f}pp  (受影响 {r['n_affected_months']}月/{r['n_excluded_picks']}票)")

    H.dump_json(res, ROOT / "reports" / "release_ab_result.json")
    print("\n结果已写 reports/release_ab_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())