"""客观 regime 重标注 + 预注册判据重判 (见 preregistration.md §六).

只重标注, 不重跑回测: 策略结果已在 reports/own_strategy_result.json。
客观规则(预注册): 纯牛=上证收益>+10%; 熊/崩=上证收益<-10% 或 maxdd<=-25%; 震荡=其余。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent

# 预注册客观标签 (§六表)
OBJECTIVE = {
    "2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
    "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
    "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
    "2024震荡": "纯牛", "2025-26现期": "震荡",
}
WINDOWS = list(OBJECTIVE.keys())


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main():
    _force_utf8()
    d = json.load(open(ROOT / "reports" / "own_strategy_result.json", encoding="utf-8"))
    det = d["detail"]

    print("=" * 80)
    print("客观 regime 重标注后的预注册判据重判")
    print("=" * 80)
    for name, kind in [("冷落低波精选", "rank"), ("冷落反转", "rank"), ("冷落隔夜", "fixed")]:
        ws = det[name]
        if kind == "fixed":
            act = {w: m for w, m in ws.items() if "avg_net_ret" in m}
            pos = [w for w, m in act.items() if m["avg_net_ret"] > 0]
            mean_ret = float(np.mean([m["avg_net_ret"] for m in act.values()])) if act else np.nan
            passed = (mean_ret > 0) and (len(pos) >= 7)
            print(f"\n[{name}] 超短线")
            print(f"  单笔均净={mean_ret*100:+.3f}%  正窗口={len(pos)}/{len(act)}  → {'PASS' if passed else 'FAIL'}")
            continue
        # rank: 主判据①/②
        act = {w: m for w, m in ws.items() if "vs_universe" in m and m["vs_universe"] == m["vs_universe"]}
        nonniu = {w: m for w, m in act.items() if OBJECTIVE[w] != "纯牛"}
        niu = {w: m for w, m in act.items() if OBJECTIVE[w] == "纯牛"}
        c1_win = sum(1 for m in nonniu.values() if m["vs_universe"] >= 0)
        c1 = c1_win >= 7 and len(nonniu) >= 8
        c2 = all(m["total"] > 0 and m["maxdd"] >= -0.30 for m in niu.values()) if len(niu) >= 3 else False
        print(f"\n[{name}]")
        print(f"  非纯牛赢 {c1_win}/{len(nonniu)} → 主判据① {'PASS' if c1 else 'FAIL'}")
        niu_rows = "\n".join(
            f"    {w:<16}{OBJECTIVE[w]:<4} total={m['total']*100:+7.1f}%  maxdd={m['maxdd']*100:+6.1f}%  {'OK' if (m['total']>0 and m['maxdd']>=-0.30) else 'X'}"
            for w, m in sorted(niu.items())
        )
        print(f"  纯牛窗 (全收益>0 且 maxdd≥-30%):\n{niu_rows}")
        print(f"  → 主判据② {'PASS' if c2 else 'FAIL'}")
        # 逐窗口 vs_univ (带客观标签)
        print("  非纯牛逐窗 vs_univ:")
        for w in WINDOWS:
            if OBJECTIVE[w] == "纯牛":
                continue
            m = ws.get(w, {})
            if "vs_universe" in m and m["vs_universe"] == m["vs_universe"]:
                print(f"    {w:<16}{OBJECTIVE[w]:<4} {m['vs_universe']*100:+7.2f}pp  total={m['total']*100:+6.1f}%")

    out = {"objective_regime": OBJECTIVE, "detail": det}
    json.dump(out, open(ROOT / "reports" / "regime_reclassified_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n结果已写 reports/regime_reclassified_result.json")


if __name__ == "__main__":
    raise SystemExit(main())
