"""左尾截断双 screen (面值/亏损) — 预注册 reports/agent_loop/prereg_tail_screen.md.

A = buffer 定型; P = 剔除 close<2元; Q = 剔除 peTTM<0。各自独立 vs A, score-NaN 注入。
输出: reports/agent_loop/tail_screen_result.json
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
from run_rotation_vs_hold import _score, month_ends, run_arm  # noqa: E402


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def run_screen(df: pd.DataFrame, score: pd.Series, mask: pd.Series) -> tuple[pd.Series, dict, int]:
    d = df.copy()
    s2 = score.copy()
    mi = pd.MultiIndex.from_arrays([d["date"].values, d["symbol"].values])
    sub = d[mask]
    drop = pd.MultiIndex.from_arrays([sub["date"].values, sub["symbol"].values])
    s2[mi.isin(drop)] = np.nan
    dr, info = run_arm(d, s2, "buffer")
    return dr, info, int(mask.sum())


def main() -> int:
    _force_utf8()
    out = {}
    print(f"{'window':<12} {'arm':<6} {'total':>8} {'vsU':>8} {'maxdd':>8} {'repl':>5} {'excl':>6}")
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        score = _score(df)
        uni = H.window_universe_benchmark(df)
        out[window] = {}
        dr_a, info_a = run_arm(df, score, "buffer")
        arms = {"A": (None, dr_a, info_a, 0)}
        m_px = df["close"] < 2.00
        m_pe = df["peTTM"] < 0
        arms["P"] = ("px<2", *run_screen(df, score, m_px))
        arms["Q"] = ("pe<0", *run_screen(df, score, m_pe))
        for name, (_, dr, info, n_excl) in arms.items():
            m = H.compute_metrics(dr)
            t0, t1 = dr.index.min(), dr.index.max()
            uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
            uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
            vsu = round(m["total"] - uni_total, 4)
            out[window][name] = {"total": round(m["total"], 4), "vs_universe": vsu,
                                 "maxdd": round(m["maxdd"], 4), "n_replaced": info["n_replaced"],
                                 "excl": n_excl}
            print(f"{window:<12} {name:<6} {m['total']*100:>+7.1f}% {vsu*100:>+7.1f}% "
                  f"{m['maxdd']*100:>7.1f}% {info['n_replaced']:>5} {n_excl:>6}")

    w10 = [w for w in H.WINDOWS if w != "2015牛转股灾"]
    summary = {}
    for arm in ("P", "Q"):
        diffs = [out[w][arm]["vs_universe"] - out[w]["A"]["vs_universe"] for w in w10]
        wins = sum(1 for x in diffs if x >= 0)
        avg_diff = float(np.mean(diffs))
        dd_a = float(np.mean([out[w]["A"]["maxdd"] for w in w10]))
        dd_b = float(np.mean([out[w][arm]["maxdd"] for w in w10]))
        gate = bool(avg_diff >= 0.002 and wins >= 6 and dd_b >= dd_a)
        repl = {w: (out[w]["A"]["n_replaced"], out[w][arm]["n_replaced"]) for w in H.WINDOWS}
        summary[arm] = {"avg_vsU_diff": round(avg_diff, 4), "wins": f"{wins}/{len(w10)}",
                        "avg_dd_A": round(dd_a, 4), "avg_dd_arm": round(dd_b, 4),
                        "repl_A_vs_arm": repl,
                        "gate": "PASS" if gate else "FAIL",
                        "rule": "avg差≥+0.2pp 且 ≥6/10 且 dd不劣"}
        print(f"\n[{arm}] {json.dumps({k: v for k, v in summary[arm].items() if k != 'repl_A_vs_arm'}, ensure_ascii=False)}")
    fp = ROOT / "reports" / "agent_loop" / "tail_screen_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
