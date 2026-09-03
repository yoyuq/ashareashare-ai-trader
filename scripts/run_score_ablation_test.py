"""score 因子消融 A/B/C/D — 预注册 reports/agent_loop/prereg_score_ablation.md.

A = 三因子 (定型); B = turn+std20 (去市值); C = turn+log_mkt (去低波); D = std20+log_mkt (去换手)。
其余逐字一致 (全主板池内排名, buffer keep-zone=10, top5, 月末收盘, 65bp)。
Gate: max(B,C,D) vs A: avg差 ≥ +0.2pp 且 ≥6/11 且 dd差 ≥ −1pp。

输出: reports/agent_loop/score_ablation_result.json
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
from run_rotation_vs_hold import TOP_K, KEEP_ZONE, COST  # noqa: E402

OBJECTIVE = {"2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
             "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
             "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
             "2024震荡": "纯牛", "2025-26现期": "震荡"}

ARMS = {"A": ("turn", "std", "mkt"), "B": ("turn", "std"),
        "C": ("turn", "mkt"), "D": ("std", "mkt")}


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def score_factors(df: pd.DataFrame) -> pd.DataFrame:
    """因子值: turn / std20 (20日收盘滚动) / log_mkt (流通市值代理, 点内)。"""
    out = df[["date", "symbol", "turn", "amount", "close"]].copy()
    s = df.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    out["std"] = s.reindex(df.index)
    fmkt = df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8
    out["mkt"] = np.log1p(fmkt)
    return out


def score_variant(df: pd.DataFrame, factors: tuple[str, ...]) -> pd.Series:
    f = score_factors(df)
    parts = [f[name].groupby(df["date"]).rank(pct=True) for name in factors]
    return -sum(parts)  # 等权和 (与定型 _score 同口径, 因子数不影响排序)


def buffer_selections(df: pd.DataFrame, score: pd.Series, rl: list) -> dict:
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
        dates = sorted(df["date"].unique())
        s = pd.Series(pd.DatetimeIndex(dates))
        rl = s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()
        ret_cc = df.pivot_table(index="date", columns="symbol", values="close").sort_index().pct_change(fill_method=None)
        out[window] = {}
        for arm, factors in ARMS.items():
            score = score_variant(df, factors)
            sel = buffer_selections(df, score, rl)
            dr = run_arm(ret_cc, rl, sel)
            if dr.empty:
                raise RuntimeError(f"{arm} 臂空收益: {window}")
            m = H.compute_metrics(dr)
            out[window][arm] = {"total": round(m["total"], 4), "maxdd": round(m["maxdd"], 4),
                                "sel": sel}
        for arm in ("B", "C", "D"):
            ovs = [len(out[window]["A"]["sel"].get(T, set()) & out[window][arm]["sel"].get(T, set())) / 5
                   for T in rl if out[window]["A"]["sel"].get(T)]
            out[window][f"{arm}_overlap"] = round(float(np.mean(ovs)), 3) if ovs else None
        for arm in ("B", "C", "D"):
            out[window][f"{arm}_minus_A"] = round(out[window][arm]["total"] - out[window]["A"]["total"], 4)
        print(f"{window:<12} A={out[window]['A']['total']*100:>+8.1f}% "
              + " ".join(f"{arm}={out[window][arm]['total']*100:>+8.1f}% ({out[window][f'{arm}_minus_A']*100:>+7.2f}, ovl={out[window][f'{arm}_overlap']})"
                         for arm in ("B", "C", "D"))
              + f" ddA={out[window]['A']['maxdd']*100:>6.1f}%")
        for arm in ("A", "B", "C", "D"):
            del out[window][arm]["sel"]

    ddA = np.mean([out[w]["A"]["maxdd"] for w in H.WINDOWS])
    results = {}
    for arm in ("B", "C", "D"):
        diffs = [out[w][f"{arm}_minus_A"] for w in H.WINDOWS]
        wins = sum(1 for x in diffs if x >= 0)
        results[arm] = {"wins": f"{wins}/11", "avg_diff_pp": round(float(np.mean(diffs)) * 100, 2)}
    better = max(results, key=lambda a: results[a]["avg_diff_pp"])
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
        "all_arms": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "score_ablation_result.json"
    fp.write_text(json.dumps({"prereg": "prereg_score_ablation.md",
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
