"""换仓频率 A/B/C — 预注册 reports/agent_loop/prereg_rebalance_frequency.md.

A = 月频 (定型); B = 双周频 (半月末); C = 季频。
选券/成本/等权逐字一致, 唯一自由度 = 换仓频率。
Gate: max(B,C) vs A: avg差 ≥ +0.2pp 且 ≥6/11 且 dd差 ≥ −1pp。

输出: reports/agent_loop/frequency_result.json
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
from fetch_inst_holdings import _score  # noqa: E402
from run_rotation_vs_hold import month_ends, TOP_K, KEEP_ZONE, COST  # noqa: E402

OBJECTIVE = {"2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
             "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
             "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
             "2024震荡": "纯牛", "2025-26现期": "震荡"}


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def period_ends(dates: list, freq: str) -> list:
    """频率换仓日: 每期最后一个交易日。M=月末, 2W=半月末(15日/月末), Q=季末月末日。"""
    if freq == "M":
        return month_ends(dates)
    if freq == "2W":
        groups: dict = {}
        for d in dates:
            groups.setdefault((d.year, d.month, 0 if d.day <= 15 else 1), []).append(d)
    elif freq == "Q":
        groups = {}
        for d in dates:
            groups.setdefault((d.year, (d.month - 1) // 3), []).append(d)
    else:
        raise ValueError(f"未知频率: {freq}")
    return sorted(max(v) for v in groups.values())


def buffer_selections(df: pd.DataFrame, score: pd.Series, rl: list) -> dict:
    """定型 buffer 选券 (rl 由频率决定), 返回 {T: set}。"""
    d = df.copy()
    d["score"] = score.values
    day_score = {T: g.dropna(subset=["score"]).set_index("symbol")["score"]
                 for T, g in d.groupby("date")}
    held: list = []
    sel: dict = {}
    for T in rl:
        row = day_score.get(T, pd.Series(dtype=float))
        if len(row) < 50:
            sel[T] = set(held)
            continue
        ranked = row.sort_values(ascending=False)
        top5 = ranked.head(TOP_K).index.tolist()
        keepzone = ranked.head(KEEP_ZONE).index.tolist()
        new = [c for c in held if c in keepzone]
        for c in top5:
            if len(new) >= TOP_K:
                break
            if c not in new:
                new.append(c)
        sel[T] = set(new)
        held = new
    return sel


def run_arm(ret_cc: pd.DataFrame, rl: list, sel: dict) -> pd.Series:
    """定型 A 臂拼装 (月频基准): (T, T_next] 新票 close-to-close, 成本首日。"""
    dates = ret_cc.index
    pos = {d: i for i, d in enumerate(dates)}
    daily = []
    for i, T in enumerate(rl):
        T_next = rl[i + 1] if i + 1 < len(rl) else dates[-1]
        if T == dates[-1]:
            break
        t_idx, tn_idx = pos[T], pos[T_next]
        if t_idx + 1 > tn_idx:
            continue
        seg_dates = dates[t_idx + 1: tn_idx + 1]
        if len(seg_dates) == 0:
            continue
        sel_new = sorted(sel[T])
        seg = ret_cc.loc[seg_dates, sel_new].mean(axis=1)
        if i > 0 and rl[i - 1] in sel:
            repl = len(sel[T] - sel[rl[i - 1]]) / TOP_K
            if repl:
                seg = seg.copy()
                seg.iloc[0] -= repl * COST
        daily.append(seg)
    if not daily:
        return pd.Series(dtype=float)
    s = pd.concat(daily)
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()


def main() -> int:
    _force_utf8()
    out: dict = {}
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        score = _score(df)
        dates = sorted(df["date"].unique())
        ret_cc = df.pivot_table(index="date", columns="symbol", values="close").sort_index().pct_change(fill_method=None)
        out[window] = {}
        for arm, freq in (("A", "M"), ("B", "2W"), ("C", "Q")):
            rl = [T for T in period_ends(dates, freq) if T in set(dates)]
            sel = buffer_selections(df, score, rl)
            dr = run_arm(ret_cc, rl, sel)
            if dr.empty:
                raise RuntimeError(f"{arm} 臂空收益: {window}")
            m = H.compute_metrics(dr)
            out[window][arm] = {"total": round(m["total"], 4), "maxdd": round(m["maxdd"], 4),
                                "n_rebal": len(rl)}
        out[window]["B_minus_A"] = round(out[window]["B"]["total"] - out[window]["A"]["total"], 4)
        out[window]["C_minus_A"] = round(out[window]["C"]["total"] - out[window]["A"]["total"], 4)
        print(f"{window:<12} A={out[window]['A']['total']*100:>+8.1f}% (n={out[window]['A']['n_rebal']}) "
              f"B={out[window]['B']['total']*100:>+8.1f}% ({out[window]['B_minus_A']*100:>+6.2f}, n={out[window]['B']['n_rebal']}) "
              f"C={out[window]['C']['total']*100:>+8.1f}% ({out[window]['C_minus_A']*100:>+6.2f}, n={out[window]['C']['n_rebal']}) "
              f"ddA={out[window]['A']['maxdd']*100:>6.1f}% ddB={out[window]['B']['maxdd']*100:>6.1f}% "
              f"ddC={out[window]['C']['maxdd']*100:>6.1f}%")

    diffB = [out[w]["B_minus_A"] for w in H.WINDOWS]
    diffC = [out[w]["C_minus_A"] for w in H.WINDOWS]
    ddA = np.mean([out[w]["A"]["maxdd"] for w in H.WINDOWS])
    results = {}
    for name, diffs in (("B", diffB), ("C", diffC)):
        wins = sum(1 for x in diffs if x >= 0)
        avg = float(np.mean(diffs))
        results[name] = {"wins": f"{wins}/11", "avg_diff_pp": round(avg * 100, 2)}
    better = "B" if results["B"]["avg_diff_pp"] >= results["C"]["avg_diff_pp"] else "C"
    dd_best = np.mean([out[w][better]["maxdd"] for w in H.WINDOWS])
    dd_ok = bool((dd_best - ddA) >= -0.01)
    gate_pass = bool(int(results[better]["wins"].split("/")[0]) >= 6
                     and results[better]["avg_diff_pp"] >= 0.2 and dd_ok)
    summary = {
        "gate": {"better_arm": better, **results[better],
                 "dd_best_avg": round(float(dd_best), 4), "dd_A_avg": round(float(ddA), 4),
                 "dd_ok": dd_ok,
                 "required": "avg>=+0.2pp & wins>=6/11 & dd diff>=-1pp",
                 "PASS": gate_pass},
        "both": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "frequency_result.json"
    fp.write_text(json.dumps({"prereg": "prereg_rebalance_frequency.md",
                              "windows": _jsonable(out), "summary": summary},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


if __name__ == "__main__":
    raise SystemExit(main())
