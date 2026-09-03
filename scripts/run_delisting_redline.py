"""退市红线票剔除 screen — 预注册 reports/agent_loop/prereg_delisting_redline.md.

Arm R: 换仓日剔除 (MBRevenue<3亿 且 netProfit<0) point-in-time 票, vs 在持 buffer 臂 A。
输出: reports/agent_loop/delisting_redline_result.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import strategy_research_harness as H  # noqa: E402
from run_rotation_vs_hold import _score, run_arm  # noqa: E402


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main() -> int:
    _force_utf8()
    fund = H.load_fundamentals()
    out = {}
    print(f"{'window':<12} {'arm':<4} {'total':>8} {'vsU':>8} {'maxdd':>8} {'repl':>5} {'excl':>6}")
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        r = H.pt_fundamental_col(df, fund, "MBRevenue")
        np_ = H.pt_fundamental_col(df, fund, "netProfit")
        mask = (r < 3e8) & (np_ < 0)
        mask = mask.fillna(False)
        s2 = _score(df).copy()
        mi = pd.MultiIndex.from_arrays([df["date"].values, df["symbol"].values])
        sub = df[mask.values]
        drop = pd.MultiIndex.from_arrays([sub["date"].values, sub["symbol"].values])
        s2[mi.isin(drop)] = np.nan
        uni = H.window_universe_benchmark(df)
        out[window] = {}
        dr_a, info_a = run_arm(df, _score(df), "buffer")
        dr_r, info_r = run_arm(df, s2, "buffer")
        for name, dr, info, n_excl in (("A", dr_a, info_a, 0), ("R", dr_r, info_r, int(mask.sum()))):
            m = H.compute_metrics(dr)
            t0, t1 = dr.index.min(), dr.index.max()
            uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
            uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
            vsu = round(m["total"] - uni_total, 4)
            out[window][name] = {"total": round(m["total"], 4), "vs_universe": vsu,
                                 "maxdd": round(m["maxdd"], 4), "n_replaced": info["n_replaced"],
                                 "excl": n_excl}
            print(f"{window:<12} {name:<4} {m['total']*100:>+7.1f}% {vsu*100:>+7.1f}% "
                  f"{m['maxdd']*100:>7.1f}% {info['n_replaced']:>5} {n_excl:>6}")

    w10 = [w for w in H.WINDOWS if w != "2015牛转股灾"]
    diffs = [out[w]["R"]["vs_universe"] - out[w]["A"]["vs_universe"] for w in w10]
    wins = sum(1 for x in diffs if x >= 0)
    avg_diff = float(np.mean(diffs))
    dd_a = float(np.mean([out[w]["A"]["maxdd"] for w in w10]))
    dd_r = float(np.mean([out[w]["R"]["maxdd"] for w in w10]))
    gate = bool(avg_diff >= 0.002 and wins >= 6 and dd_r >= dd_a)
    summary = {"avg_diff_vs_A": round(avg_diff, 4), "wins": f"{wins}/{len(w10)}",
               "avg_dd_A": round(dd_a, 4), "avg_dd_R": round(dd_r, 4),
               "excl_per_window": {w: out[w]["R"]["excl"] for w in H.WINDOWS},
               "gate": "PASS" if gate else "FAIL",
               "rule": "avg差≥+0.2pp 且 ≥6/10 且 dd不劣"}
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "delisting_redline_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary,
                              "caveat": "历史窗大部分在旧退市制度下, FAIL 对新规时代外推性弱, "
                                        "尾部风险转前瞻监控"},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
