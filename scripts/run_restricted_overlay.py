"""冷落低波 buffer + 解禁规避叠层 — 预注册 reports/agent_loop/prereg_restricted_overlay.md.

A = buffer 定型臂; B = 同 A 但月末换仓时剔除未来30交易日大额解禁票 (score 置 NaN)。
输出: reports/agent_loop/restricted_overlay_result.json
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
from run_rotation_vs_hold import COST, TOP_K, _score, month_ends, run_arm  # noqa: E402
from run_restricted_screen import load_events  # noqa: E402

AHEAD_TD = 30
RATIO_MIN = 0.05


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def blocked_by_rebalance(df: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """{月末日期T: 被禁票集合} — T 未来30交易日内有大额解禁的主板票。"""
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    dates = pd.DatetimeIndex(close.index)
    big = ev[ev["ratio"] >= RATIO_MIN]
    out = {}
    for T in month_ends(list(close.index)):
        if T not in set(close.index):
            continue
        ti = dates.get_loc(T)
        horizon = dates[ti: min(len(dates), ti + AHEAD_TD + 1)]
        ev_in = big[big["free_date"].isin(horizon)]
        out[T] = set(ev_in["sym"]) & set(close.columns)
    return out


def run_arm_overlay(df: pd.DataFrame, score: pd.Series, blocked: dict) -> tuple[pd.Series, dict]:
    """score 在换仓日 T 对被禁票置 NaN 后走定型 buffer 臂 (复用 run_arm)。"""
    d = df.copy()
    s2 = score.copy()
    # 按行置 NaN: 行日期是换仓日且票被禁
    drop = [(T, c) for T, blk in blocked.items() for c in blk]
    if drop:
        dd = pd.MultiIndex.from_tuples(drop)
        mi = pd.MultiIndex.from_arrays([d["date"].values, d["symbol"].values])
        s2[mi.isin(dd)] = np.nan
    return run_arm(d, s2, "buffer")


def main() -> int:
    _force_utf8()
    ev = load_events()
    print(f"events: {len(ev)} rows")
    idx = H.load_index()
    out = {}
    print(f"{'window':<12} {'arm':<8} {'total':>8} {'vsU':>8} {'maxdd':>8} {'repl':>5}")
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        score = _score(df)
        uni = H.window_universe_benchmark(df)
        ev_w = ev[(ev["free_date"] >= df["date"].min()) & (ev["free_date"] <= df["date"].max())]
        blocked = blocked_by_rebalance(df, ev_w) if len(ev_w) else {}
        out[window] = {}
        arms = {"A_buffer": run_arm(df, score, "buffer")}
        if blocked:
            arms["B_overlay"] = run_arm_overlay(df, score, blocked)
        else:
            arms["B_overlay"] = arms["A_buffer"]  # 无解禁数据窗: 等效退化 (2015窗, 披露)
        for name, (dr, info) in arms.items():
            m = H.compute_metrics(dr)
            t0, t1 = dr.index.min(), dr.index.max()
            uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
            uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
            vsu = round(m["total"] - uni_total, 4)
            out[window][name] = {"total": round(m["total"], 4), "vs_universe": vsu,
                                 "maxdd": round(m["maxdd"], 4), "n_replaced": info["n_replaced"]}
            print(f"{window:<12} {name:<10} {m['total']*100:>+7.1f}% {vsu*100:>+7.1f}% "
                  f"{m['maxdd']*100:>7.1f}% {info['n_replaced']:>5}")

    w10 = [w for w in H.WINDOWS if w != "2015牛转股灾"]
    diffs = [out[w]["B_overlay"]["vs_universe"] - out[w]["A_buffer"]["vs_universe"] for w in w10]
    wins = sum(1 for x in diffs if x >= 0)
    avg_diff = float(np.mean(diffs))
    avg_dd_b = float(np.mean([out[w]["B_overlay"]["maxdd"] for w in w10]))
    avg_dd_a = float(np.mean([out[w]["A_buffer"]["maxdd"] for w in w10]))
    # maxdd 为负值口径: 「不劣」= B 的 dd 不更深, 即 avg_dd_b >= avg_dd_a
    # (首跑时误写为 <= 判 FAIL — 符号约定笔误, 数据不变, 修正后重判; 见 iter 纪要披露)
    gate = bool(avg_diff >= 0.003 and wins >= 6 and avg_dd_b >= avg_dd_a)
    # 披露: 被剔票后续等权篮子收益已在 iter14 blocked 篮子验证, 此处仅记成本代价
    repl_b = sum(out[w]["B_overlay"]["n_replaced"] for w in H.WINDOWS)
    repl_a = sum(out[w]["A_buffer"]["n_replaced"] for w in H.WINDOWS)
    summary = {"avg_vsU_diff": round(avg_diff, 4), "B_ge_A_wins": f"{wins}/{len(w10)}",
               "avg_maxdd_B": round(avg_dd_b, 4), "avg_maxdd_A": round(avg_dd_a, 4),
               "replacements_B": repl_b, "replacements_A": repl_a,
               "gate": "PASS" if gate else "FAIL",
               "rule": "avg差≥+0.3pp 且 B≥A≥6/10 且 dd不劣"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "restricted_overlay_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
