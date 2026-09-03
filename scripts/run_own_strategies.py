"""自编策略回测 — 预注册判据 (见 reports/strategy_research_preregistration.md §五).

三档时长:
  长线「冷落低波精选」 rank 月再平衡 top5 = 低换手+低波+小市值 复合
  中线「冷落反转」     rank 月再平衡 top5 = 低换手 + 20日超跌
  超短线「冷落隔夜」   fixed T收盘买最低换手20只, T+1开盘卖

判据(预注册, 跑完不调):
  长/中线: 非纯牛8窗≥7/8跑赢universe; 纯牛3窗全收益>0且maxdd≥-30%
  超短线: 单笔均净>0 且 ≥7/11窗口均净>0
用法: python scripts/run_own_strategies.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import strategy_research_harness as H


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# --------------------------------------------------------------------------- #
# 因子
# --------------------------------------------------------------------------- #
def _factors(df):
    """返回 turn(日换手) / std20(20日波动) / log_mkt(流通市值对数, 亿)."""
    out = df[["date", "symbol", "turn", "amount", "close"]].copy()
    s = df.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    out["std20"] = s.reindex(df.index)
    fmkt = df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8  # 亿
    out["log_mkt"] = np.log1p(fmkt)
    return out


def _cold_lowvol(df):
    f = _factors(df)
    r_turn = f["turn"].groupby(df["date"]).rank(pct=True)
    r_std = f["std20"].groupby(df["date"]).rank(pct=True)
    r_mkt = f["log_mkt"].groupby(df["date"]).rank(pct=True)
    return -(r_turn + r_std + r_mkt)  # 越大越优先(低换手+低波+小市值)


def _cold_reversal(df):
    f = _factors(df)
    r_turn = f["turn"].groupby(df["date"]).rank(pct=True)
    rev = df.groupby("symbol")["close"].pct_change(20)
    r_rev = (-rev).groupby(df["date"]).rank(pct=True)  # 20日跌幅越大 rank 越高
    return -(r_turn) + r_rev  # 低换手 + 超跌


def _cold_bottom20(df):
    r = df.groupby("date")["turn"].rank(method="first", ascending=True)
    return r <= 20


# --------------------------------------------------------------------------- #
OWN = [
    {"name": "冷落低波精选", "category": "长线", "kind": "rank", "fn": _cold_lowvol, "top_k": 5, "freq": "monthly"},
    {"name": "冷落反转", "category": "中线", "kind": "rank", "fn": _cold_reversal, "top_k": 5, "freq": "monthly"},
    {"name": "冷落隔夜", "category": "超短线", "kind": "fixed", "fn": _cold_bottom20, "hold": 1, "exit": "open"},
]


def main():
    _force_utf8()
    idx = H.load_index()
    results = {}
    t0 = time.time()
    for window, fp in H.WINDOWS.items():
        df = H.load_base_window(fp)
        if df.empty:
            continue
        for st in OWN:
            try:
                if st["kind"] == "rank":
                    m = H.run_window_ranking(df, idx, st["fn"], st["top_k"], st["freq"], window)
                else:
                    m = H.run_window_fixed_hold(df, idx, st["fn"], st["hold"], st["exit"], window)
            except Exception as e:
                m = {"error": str(e), "regime": H.REGIME.get(window, "")}
            results.setdefault(st["name"], {})[window] = m
        print(f"  [{window}] 完成 {time.time()-t0:.1f}s", flush=True)

    H.dump_json({"detail": results}, ROOT / "reports" / "own_strategy_result.json")

    # ── 判据评估 (预注册) ──
    print("\n" + "=" * 90)
    print("自编策略判据评估 (预注册口径)")
    print("=" * 90)
    for st in OWN:
        ws = results[st["name"]]
        if st["kind"] == "fixed":
            act = {w: m for w, m in ws.items() if "avg_net_ret" in m}
            pos = [w for w, m in act.items() if m["avg_net_ret"] > 0]
            mean_ret = float(np.mean([m["avg_net_ret"] for m in act.values()])) if act else np.nan
            passed = (mean_ret > 0) and (len(pos) >= 7)
            print(f"\n[{st['name']}] ({st['category']})")
            print(f"  单笔均净={mean_ret*100:+.3f}%  正窗口={len(pos)}/{len(act)}  判据={passed and 'PASS' or 'FAIL'}")
        else:
            act = {w: m for w, m in ws.items() if "vs_universe" in m and m["vs_universe"] == m["vs_universe"]}
            nonniu = {w: m for w, m in act.items() if m["regime"] != "纯牛"}
            niu = {w: m for w, m in act.items() if m["regime"] == "纯牛"}
            c1_win = sum(1 for m in nonniu.values() if m["vs_universe"] >= 0)
            c1 = c1_win >= 7 and len(nonniu) >= 8
            c2 = all(m["total"] > 0 and m["maxdd"] >= -0.30 for m in niu.values()) if niu else False
            mean_vu = float(np.mean([m["vs_universe"] for m in act.values()]))
            print(f"\n[{st['name']}] ({st['category']})")
            print(f"  均vs_universe={mean_vu*100:+.2f}pp  非纯牛赢={c1_win}/{len(nonniu)}  纯牛全收益>0&maxdd≥-30%={c2}")
            print(f"  → 主判据①={c1 and 'PASS' or 'FAIL'}  主判据②={c2 and 'PASS' or 'FAIL'}")
            # 逐窗口
            for w in H.WINDOWS:
                m = ws.get(w, {})
                if "vs_universe" not in m or m["vs_universe"] != m["vs_universe"]:
                    continue
                print(f"    {w:<14} regime={m['regime']:<4} total={m['total']*100:+7.1f}% vs_univ={m['vs_universe']*100:+7.2f}pp sharpe={m['sharpe']:+5.2f} maxdd={m['maxdd']*100:+6.1f}%")
    print("\n结果已写 reports/own_strategy_result.json")


if __name__ == "__main__":
    raise SystemExit(main())
