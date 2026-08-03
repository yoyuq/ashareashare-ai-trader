"""40天回放 A/B 对比 — 基线 vs v3.2 因子变体

基线报告: replay_data/report_nothink_fixed_40d.json (原版 prompt)
v32 报告: replay_data/replay_report.json (v3.2 因子注入, 当前最新 run)

对比维度:
  1. 交易结果: 总收益 / 最大回撤 / 交易数 / 胜率
  2. 选股估值暴露: 买入标的的 中位PB / 中位PE20d分位 (v3.2 是否更偏价值)
  3. 期末持仓

用法: python scripts/compare_replay_ab.py [--baseline replay_data/report_nothink_fixed_40d.json]
                                         [--v32 replay_data/replay_report.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def metrics(report: dict) -> dict:
    eq = report.get("equity_curve", [])
    totals = [_f(e["total"]) for e in eq]
    trades = report.get("trades", [])
    # 胜率: 有 pnl 的卖出
    closed = [t for t in trades if t.get("side") == "sell" and t.get("pnl_pct") is not None]
    wins = sum(1 for t in closed if _f(t.get("pnl_pct")) > 0)
    win_rate = wins / len(closed) if closed else None
    # 最大回撤
    max_dd = 0.0
    peak = -1e18
    for t in totals:
        peak = max(peak, t)
        if peak > 0:
            max_dd = min(max_dd, t / peak - 1)
    return {
        "final_equity": totals[-1] if totals else 0,
        "total_return": _f(report.get("total_return_pct", 0)),
        "max_drawdown": max_dd,
        "n_trades": report.get("num_trades", len(trades)),
        "n_sells": len(closed),
        "win_rate": win_rate,
        "n_positions": len(report.get("final_positions", [])),
    }


def buys(report: dict) -> list:
    return [t for t in report.get("trades", []) if t.get("side") == "buy"]


def pick_valuation(report: dict, replay: str) -> dict:
    """买入标的在买入日的 PB / PE20d分位 (从回放缓存按 symbol 直查)."""
    df = pd.read_parquet(replay)
    df["date"] = pd.to_datetime(df["date"])
    pb_vals, pe_pct_vals, ep_vals = [], [], []
    n_found = 0
    for t in buys(report):
        sym = t.get("symbol")
        d = pd.Timestamp(t.get("date"))
        g = df[(df["symbol"] == sym) & (df["date"] <= d)]
        if g.empty:
            continue
        last = g.sort_values("date").iloc[-1]
        pb = last.get("pbMRQ")
        pe = last.get("peTTM")
        if pd.notna(pb) and pb > 0:
            pb_vals.append(float(pb))
            ep_vals.append(1.0 / float(np.clip(pb, 0.05, 50)))
        if pd.notna(pe) and pe > 0:
            hist = g.sort_values("date")["peTTM"].dropna()
            hist = hist[hist > 0]
            if len(hist) >= 5:
                pe_pct_vals.append(float((hist.iloc[:-1] < float(pe)).mean()))
        n_found += 1
    return {
        "n_buys_checked": n_found,
        "median_pb": float(np.median(pb_vals)) if pb_vals else None,
        "median_ep": float(np.median(ep_vals)) if ep_vals else None,
        "median_pe20d_pct": float(np.median(pe_pct_vals)) if pe_pct_vals else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="replay_data/report_nothink_fixed_40d.json")
    ap.add_argument("--v32", default="replay_data/replay_report.json")
    ap.add_argument("--replay", default="replay_data/daily_2026-02-21_2026-07-31.parquet")
    args = ap.parse_args()

    base = json.load(open(args.baseline, encoding="utf-8"))
    v32 = json.load(open(args.v32, encoding="utf-8"))

    mb = metrics(base)
    mv = metrics(v32)
    print(f"窗口: 基线 {base.get('window')} vs v32 {v32.get('window')}\n")
    print(f"{'指标':<14}{'基线':>12}{'v3.2':>12}{'Δ':>10}")
    print("-" * 50)
    rows = [
        ("总收益%", mb["total_return"], mv["total_return"], "pp"),
        ("最大回撤%", mb["max_drawdown"] * 100, mv["max_drawdown"] * 100, "pp"),
        ("交易数", mb["n_trades"], mv["n_trades"], ""),
        ("胜率%", (mb["win_rate"] or 0) * 100, (mv["win_rate"] or 0) * 100, "pp"),
        ("期末持仓", mb["n_positions"], mv["n_positions"], ""),
    ]
    for name, a, b, unit in rows:
        delta = b - a if unit == "pp" else b - a
        print(f"{name:<14}{a:>11.2f}{b:>11.2f}{delta:>+9.2f}")

    # 选股估值暴露
    print("\n买入标的估值暴露 (买入日):")
    print(f"{'指标':<22}{'基线':>12}{'v3.2':>12}")
    print("-" * 48)
    vb = pick_valuation(base, args.replay)
    vv = pick_valuation(v32, args.replay)
    for k, zh in [("n_buys_checked", "买入检查数"), ("median_pb", "中位PB"),
                  ("median_ep", "中位EP(1/PB)"), ("median_pe20d_pct", "中位PE20d分位")]:
        a = vb.get(k)
        b = vv.get(k)
        print(f"{zh:<22}{a if a is not None else '—':>12}{b if b is not None else '—':>12}")

    print("\n期末持仓:")
    print(f"  基线: {base.get('final_positions', [])}")
    print(f"  v3.2: {v32.get('final_positions', [])}")


if __name__ == "__main__":
    main()
