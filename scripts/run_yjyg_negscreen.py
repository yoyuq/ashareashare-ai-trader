"""利空预告规避 screen — 预注册 reports/agent_loop/prereg_yjyg_negscreen.md.

A = buffer 定型臂; B = 同 A 但月末换仓时剔除近30自然日内发利空预告票 (score 置 NaN)。
利空 = 预告类型 ∈ {预减, 略减, 首亏, 续亏, 增亏}; 65bp。
输出: reports/agent_loop/yjyg_negscreen_result.json
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

FRESH_DAYS = 30
NEG_TYPES = {"预减", "略减", "首亏", "续亏", "增亏"}
SHARD_DIR = ROOT / "replay_data" / "yjyg"


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def load_bad_events() -> pd.DataFrame:
    frames = [pd.read_parquet(fp) for fp in sorted(SHARD_DIR.glob("*.parquet"))
              if fp.stat().st_size > 0]
    if not frames:
        raise RuntimeError("yjyg 分片库为空")
    ev = pd.concat(frames, ignore_index=True)
    ev["code"] = ev["股票代码"].astype(str).str.zfill(6)
    ev = ev[ev["code"].str.startswith(("60", "00"))]
    ev["ann_date"] = pd.to_datetime(ev["公告日期"], errors="coerce")
    ev = ev.dropna(subset=["ann_date"])
    ev["sym"] = ev["code"].map(lambda c: ("sh." if c.startswith("6") else "sz.") + c)
    bad = ev[ev["预告类型"].isin(NEG_TYPES)]
    print(f"bad events: {len(bad)} / {len(ev)} 行 (类型: {sorted(NEG_TYPES)})")
    return bad[["sym", "ann_date"]]


def blocked_by_rebalance(df: pd.DataFrame, bad: pd.DataFrame) -> dict:
    """{月末T: 近30自然日内有利空预告的主板票集合} + 每T剔除数。"""
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    out = {}
    for T in month_ends(list(close.index)):
        if T not in set(close.index):
            continue
        ev_in = bad[(bad["ann_date"] > T - pd.Timedelta(days=FRESH_DAYS))
                    & (bad["ann_date"] <= T)]
        out[T] = set(ev_in["sym"]) & set(close.columns)
    return out


def run_arm_screen(df: pd.DataFrame, score: pd.Series, blocked: dict) -> tuple[pd.Series, dict, int]:
    d = df.copy()
    s2 = score.copy()
    n_excl = sum(len(v) for v in blocked.values())
    drop = [(T, c) for T, blk in blocked.items() for c in blk]
    if drop:
        dd = pd.MultiIndex.from_tuples(drop)
        mi = pd.MultiIndex.from_arrays([d["date"].values, d["symbol"].values])
        s2[mi.isin(dd)] = np.nan
    dr, info = run_arm(d, s2, "buffer")
    return dr, info, n_excl


def main() -> int:
    _force_utf8()
    bad = load_bad_events()
    out = {}
    print(f"{'window':<12} {'arm':<8} {'total':>8} {'vsU':>8} {'maxdd':>8} {'repl':>5} {'excl':>5}")
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        score = _score(df)
        uni = H.window_universe_benchmark(df)
        ev_w = bad[(bad["ann_date"] >= df["date"].min())
                   & (bad["ann_date"] <= df["date"].max())]
        blocked = blocked_by_rebalance(df, ev_w) if len(ev_w) else {}
        dr_a, info_a = run_arm(df, score, "buffer")
        dr_b, info_b, n_excl = run_arm_screen(df, score, blocked)
        out[window] = {}
        for name, (dr, info) in {"A_buffer": (dr_a, info_a), "B_negscreen": (dr_b, info_b)}.items():
            m = H.compute_metrics(dr)
            t0, t1 = dr.index.min(), dr.index.max()
            uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
            uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
            vsu = round(m["total"] - uni_total, 4)
            out[window][name] = {"total": round(m["total"], 4), "vs_universe": vsu,
                                 "maxdd": round(m["maxdd"], 4), "n_replaced": info["n_replaced"]}
            print(f"{window:<12} {name:<10} {m['total']*100:>+7.1f}% {vsu*100:>+7.1f}% "
                  f"{m['maxdd']*100:>7.1f}% {info['n_replaced']:>5} "
                  f"{n_excl if name == 'B_negscreen' else 0:>5}")

    w10 = [w for w in H.WINDOWS if w != "2015牛转股灾"]
    diffs = [out[w]["B_negscreen"]["vs_universe"] - out[w]["A_buffer"]["vs_universe"] for w in w10]
    wins = sum(1 for x in diffs if x >= 0)
    avg_diff = float(np.mean(diffs))
    avg_dd_b = float(np.mean([out[w]["B_negscreen"]["maxdd"] for w in w10]))
    avg_dd_a = float(np.mean([out[w]["A_buffer"]["maxdd"] for w in w10]))
    # maxdd 负值口径: 「不劣」= avg_dd_B >= avg_dd_A
    gate = bool(avg_diff >= 0.002 and wins >= 6 and avg_dd_b >= avg_dd_a)
    total_excl = sum(out[w].get("_excl", 0) for w in H.WINDOWS)
    summary = {"avg_vsU_diff": round(avg_diff, 4), "B_ge_A_wins": f"{wins}/{len(w10)}",
               "avg_maxdd_B": round(avg_dd_b, 4), "avg_maxdd_A": round(avg_dd_a, 4),
               "gate": "PASS" if gate else "FAIL",
               "rule": "avg差≥+0.2pp 且 B≥A≥6/10 且 dd不劣"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "yjyg_negscreen_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
