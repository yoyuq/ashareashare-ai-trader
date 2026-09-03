"""北向+两融 牛市不崩守卫 A/B — 冷落低波精选 (预注册 §十五).

信号 (外生, point-in-time, 全库 2015-2026):
  北向 risk-off: 当日成交净买额 21 交易日滚动求和的最近期 < 0
  两融 risk-off: SSE 融资融券余额 < rolling60_max × 0.95 (自60日高点回撤>5%)
  触发 = OR. 再平衡日触发 ⇒ 该月 sel=空 (持现金), 下次非触发恢复 top5.
  判据 (写死): ① 非纯牛8窗 guard vs 匹配universe ≥0 窗口 ≥7/8 (不能割熊市超额)
              ② 纯牛3窗 guard 全 总收益>0 且 各窗 maxdd ≥ −30% (修好 2019/2020 深回撤)

用法: python scripts/run_guard_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import strategy_research_harness as H  # noqa: E402

IND_north = 21       # 北向净买 21 交易日滚动
MARGIN_WIN = 60      # 两融 60 交易日高点
MARGIN_DROP = 0.95   # 回撤 >5% 触发
COST_BPS = 31.0

# §六 客观 regime 重标注 (与 cost_reality/crash_guard 一致)
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


def _load_guard_series() -> pd.Series:
    """→ Series[risk_off] indexed by trading date, computed on full 2015-2026 libs (point-in-time ≤T)."""
    hsgt = pd.read_parquet(ROOT / "replay_data" / "hsgt.parquet")
    margin = pd.read_parquet(ROOT / "replay_data" / "margin_sse.parquet")
    hsgt["date"] = pd.to_datetime(hsgt["date"]); margin["date"] = pd.to_datetime(margin["date"])
    hsgt = hsgt.sort_values("date").set_index("date")["net_buy"]
    margin = margin.sort_values("date").set_index("date")["margin"]

    # 北向: 21交易日滚动和 <0 (实际交易序列)
    north = hsgt.rolling(IND_north, min_periods=IND_north).sum() < 0
    # 两融: 值 < 60日滚动max ×0.95
    mg = margin.rolling(MARGIN_WIN, min_periods=MARGIN_WIN).max() * MARGIN_DROP
    mgo = margin < mg
    ro = (north.reindex(hsgt.index).fillna(False)) | (mgo.reindex(margin.index).fillna(False))
    # 合并到统一日历 (两列→ OR), 取 'date' 索引
    idx = hsgt.index.union(margin.index).sort_values()
    ro = ro.reindex(idx).fillna(False)
    return ro


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


def backtest_ranking_with_guard(df, score, top_k, ro, freq="monthly", cost_bps=COST_BPS):
    """同 harness backtest_ranking_portfolio, 但 risk_off 再平衡日 sel=空 (持现金)."""
    df = df.copy()
    df["score"] = score.values
    dates = sorted(df["date"].unique())
    rebal = H.pick_rebalance_dates(dates, freq)
    sel = {}
    for T in rebal:
        day = df[df["date"] == T].dropna(subset=["score"])
        # guard: risk_off(T 当日 info) ⇒ 现金 (缺失/NaN 一律按非触发)
        risk_off = bool(ro.at[T]) if T in ro.index else False
        if risk_off:
            sel[T] = set()
        else:
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
    info = {"n_rebalance": len(rl), "n_cash_rebal": sum(1 for T in rl if not sel[T]),
            "turnover_cost_bps_equiv": round(float(cost.sum()) * 10000, 2)}
    return daily_net, info


def _run_window(df, idx, ro, window, with_guard):
    score = _cold_lowvol(df)
    if with_guard:
        daily_net, info = backtest_ranking_with_guard(df, score, 5, ro)
    else:
        daily_net, info = H.backtest_ranking_portfolio(df, score, 5, "monthly", cost_bps=COST_BPS)
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
    ro = _load_guard_series()
    print(f"守卫信号: 北向{IND_north}日净买<0  OR  两融<{int(MARGIN_WIN)}日高点×{MARGIN_DROP} (预注册§十五)")
    print(f"全史 risk_off 交易日数: {int(ro.sum())} / {len(ro)}  ({ro.sum()/len(ro)*100:.1f}%)")
    print("=" * 118)
    res = {"baseline": {}, "guard": {}}
    print(f"{'窗口':<12}{'regime':<5} | {'B0总/vsU':>22} | {'G总/vsU':>22} | {'G现金再平衡':>10}")
    print("-" * 118)
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        b = _run_window(df, idx, ro, window, with_guard=False)
        g = _run_window(df, idx, ro, window, with_guard=True)
        res["baseline"][window], res["guard"][window] = b, g
        def pc(x): return f"{x*100:+.1f}%" if not (isinstance(x, float) and np.isnan(x)) else "  nan "
        print(f"{window:<12}{OBJECTIVE[window]:<5} | {b['total']*100:+6.1f}%/{b['vs_universe']*100:+6.1f}%"
              f"{'':>2} | {g['total']*100:+6.1f}%/{g['vs_universe']*100:+6.1f}%"
              f"{'':>2} | {g.get('n_cash_rebal','-')}")

    print("\n" + "=" * 118)
    print("预注册判据 (§十五) — 守卫臂 G vs 基线 B0")
    for label, d in res.items():
        niu_win = sum(1 for w in NON_PURE if d[w]["vs_universe"] is not None and d[w]["vs_universe"] >= 0)
        niu = f"{niu_win}/8"
        pb = all(not np.isnan(d[w]["total"]) and d[w]["total"] > 0 and d[w]["maxdd"] >= -0.30 for w in PURE_BULL)
        pb_det = "; ".join(f"{w} tot={d[w]['total']*100:+.1f}% dd={d[w]['maxdd']*100:.1f}%" for w in PURE_BULL)
        print(f"[{label}] 主判据① 非纯牛8窗 vsU≥0: {niu}  主判据② 纯牛3窗全正&dd≥−30%: {'PASS' if pb else 'FAIL'}  ({pb_det})")

    # 裁决
    g = res["guard"]
    g1 = sum(1 for w in NON_PURE if g[w]["vs_universe"] is not None and g[w]["vs_universe"] >= 0) >= 7
    g2 = all(not np.isnan(g[w]["total"]) and g[w]["total"] > 0 and g[w]["maxdd"] >= -0.30 for w in PURE_BULL)
    print("-" * 118)
    if g1 and g2:
        print("裁决: 主判据① AND ② 全 PASS ⇒ 外生守卫有效, 可部署")
    else:
        print("裁决: FAIL — 外生北向/两融守卫救不了 2019 深回撤 (或割了熊市超额); "
              "2019 深回撤第3次确认为固有代价, 维持方向A接受回撤" if not g2 else "裁决: ① FAIL 守卫割了熊市超额")
    H.dump_json(res, ROOT / "reports" / "guard_ab_result.json")
    print("\n结果已写 reports/guard_ab_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())