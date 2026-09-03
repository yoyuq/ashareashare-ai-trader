"""换仓执行时机 A/B/C — 预注册 reports/agent_loop/prereg_execution_timing.md.

A = 月末收盘成交 (定型); B = 次日开盘成交; C = 次日收盘成交。
选券与持仓规则逐字一致 (buffer keep-zone=10, top5), 仅成交时点不同。
Gate: max(B,C) vs A: avg差 ≥ +0.2pp 且 ≥6/11 且 dd差 ≥ −1pp。

输出: reports/agent_loop/execution_timing_result.json
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


def buffer_selections(df: pd.DataFrame, score: pd.Series) -> tuple[list, dict]:
    """定型 buffer 选券序列: 返回 (rl, {T: sorted_list}) — 三臂共用, 只算一次。"""
    d = df.copy()
    d["score"] = score.values
    dates = sorted(d["date"].unique())
    rl = [T for T in month_ends(dates) if T in set(d["date"])]
    day_score = {T: g.dropna(subset=["score"]).set_index("symbol")["score"] for T, g in d.groupby("date")}
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
    return rl, sel


def run_timing_arm(close: pd.DataFrame, open_: pd.DataFrame, ret_cc: pd.DataFrame,
                   rl: list, sel: dict, arm: str) -> pd.Series:
    """按成交时点拼装日收益。

    close/open_: pivot (date × symbol); ret_cc: close-to-close。
    A: (T, T_next] 新票 close-to-close, 成本首日。
    B: (T, T1] 旧票 close→open; (T1, T_next] 新票 open→close; 成本新段首日。
       (T1 = T 后第一个交易日; T1 bar 的 open 基准)
    C: (T, T1] 旧票 close-to-close; (T1, T_next] 新票 close-to-close; 成本新段首日。
    """
    dates = close.index
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
        T1 = seg_dates[0]  # T 后第一个交易日
        sel_new = sorted(sel[T])
        sel_old = sorted(sel[rl[i - 1]]) if i > 0 and rl[i - 1] in sel else None
        prev_T = rl[i - 1] if i > 0 else None

        if arm == "A":
            seg = ret_cc.loc[seg_dates, sel_new].mean(axis=1)
            if prev_T is not None:
                repl = len(sel[T] - sel[prev_T]) / TOP_K
                if repl:
                    seg = seg.copy()
                    seg.iloc[0] -= repl * COST
            daily.append(seg)
        else:
            # 旧票段 (T, T1] — 单日, 等权均值直接成标量 (与 A 的 mean(axis=1) 同口径)
            if sel_old:
                if arm == "B":
                    old_r = (open_.loc[T1, sel_old] / close.loc[T, sel_old] - 1.0)
                else:
                    old_r = ret_cc.loc[T1, sel_old]
                daily.append(pd.Series([float(old_r.mean())], index=[T1]))
            # 新票段 (T1, T_next]
            new_dates = seg_dates[1:]
            if len(new_dates) == 0:
                continue
            if arm == "B":
                # open[T1] → close[t] 基准: adj[T1]=open, 其余 close
                adj = close.loc[[T1] + list(new_dates), sel_new].copy()
                adj.loc[T1] = open_.loc[T1, sel_new]
                seg = adj.pct_change(fill_method=None).iloc[1:].mean(axis=1)
            else:  # C
                seg = ret_cc.loc[new_dates, sel_new].mean(axis=1)
            if prev_T is not None:
                repl = len(sel[T] - sel[prev_T]) / TOP_K
                if repl:
                    seg = seg.copy()
                    seg.iloc[0] -= repl * COST
            daily.append(seg)
    if not daily:
        return pd.Series(dtype=float)
    s = pd.concat(daily)
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    return s


def main() -> int:
    _force_utf8()
    idx = H.load_index()
    out: dict = {}
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        score = _score(df)
        rl, sel = buffer_selections(df, score)
        close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        open_ = df.pivot_table(index="date", columns="symbol", values="open").sort_index()
        ret_cc = close.pct_change(fill_method=None)
        out[window] = {}
        for arm in ("A", "B", "C"):
            dr = run_timing_arm(close, open_, ret_cc, rl, sel, arm)
            if dr.empty:
                raise RuntimeError(f"{arm} 臂空收益: {window}")
            m = H.compute_metrics(dr)
            out[window][arm] = {"total": round(m["total"], 4), "maxdd": round(m["maxdd"], 4)}
        out[window]["B_minus_A"] = round(out[window]["B"]["total"] - out[window]["A"]["total"], 4)
        out[window]["C_minus_A"] = round(out[window]["C"]["total"] - out[window]["A"]["total"], 4)
        print(f"{window:<12} A={out[window]['A']['total']*100:>+8.1f}% "
              f"B={out[window]['B']['total']*100:>+8.1f}% ({out[window]['B_minus_A']*100:>+6.2f}) "
              f"C={out[window]['C']['total']*100:>+8.1f}% ({out[window]['C_minus_A']*100:>+6.2f}) "
              f"ddA={out[window]['A']['maxdd']*100:>6.1f}% ddB={out[window]['B']['maxdd']*100:>6.1f}% "
              f"ddC={out[window]['C']['maxdd']*100:>6.1f}%")

    # Gate
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
    fp = ROOT / "reports" / "agent_loop" / "execution_timing_result.json"
    fp.write_text(json.dumps({"prereg": "prereg_execution_timing.md",
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
