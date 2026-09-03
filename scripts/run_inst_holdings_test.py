"""机构拥挤剔除 × 冷落 buffer 臂 A/B — 预注册 reports/agent_loop/prereg_inst_cold_intersection.md.

A = 定型 buffer 臂原样; C = 同 A, 但每月末候选池 (score 前30) 中
剔除有公募覆盖且 inst_ratio ≥ 池内中位数的票 (null/无覆盖不剔除)。

Gate (预注册写死): C−A 配对差分, 非纯牛且数据可用窗: ≥ceil(2/3·N) 窗 C≥A
且平均差 ≥ +0.5pp/窗。FAIL → 记墙不调参。

输出: reports/agent_loop/inst_cold_result.json
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import strategy_research_harness as H  # noqa: E402
from fetch_inst_holdings import WINDOWS_FROM_2016, TOPN_POOL, _score  # noqa: E402
from run_rotation_vs_hold import month_ends, run_arm, TOP_K, KEEP_ZONE  # noqa: E402

COST = 65.0 / 1e4
PARQUET = ROOT / "replay_data" / "inst_holdings.parquet"
LAG_DAYS = 60  # 预注册: 报告期季度末 + 60 自然日后可用

OBJECTIVE = {"2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
             "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
             "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
             "2024震荡": "纯牛", "2025-26现期": "震荡"}

_QUARTER_END = {"1": (3, 31), "2": (6, 30), "3": (9, 30), "4": (12, 31)}


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def period_end(period_id: str) -> date:
    y, q = period_id.split("Q")
    m, d = _QUARTER_END[q]
    return date(int(y), m, d)


def load_inst_holdings() -> pd.DataFrame:
    if not PARQUET.exists():
        raise RuntimeError(f"缺少 {PARQUET} — 先跑 scripts/fetch_inst_holdings.py (零模拟, 不兜底)")
    df = pd.read_parquet(PARQUET)
    df = df.dropna(subset=["value"])
    df = df[df["value"] > 0]  # null/0 = 无覆盖, 不剔除
    df["period_end"] = df["period"].astype(str).map(period_end)
    return df


def run_arm_c(df: pd.DataFrame, score: pd.Series, ratios: pd.DataFrame) -> tuple[pd.Series, dict, dict]:
    """buffer 臂 + 候选池拥挤剔除。复用 run_arm 的收益/成本逻辑, 仅在选择处注入过滤。"""
    d = df.copy()
    d["score"] = score.values
    dates = sorted(d["date"].unique())
    rebal = month_ends(dates)
    rl = [T for T in rebal if T in set(d["date"])]
    close = d.pivot_table(index="date", columns="symbol", values="close").sort_index()
    ret = close.pct_change(fill_method=None)
    day_score = {T: g.dropna(subset=["score"]).set_index("symbol")["score"] for T, g in d.groupby("date")}
    ratio_lookup = {(T, s): r for T, s, r in
                    zip(ratios["date"], ratios["symbol"], ratios["ratio"])}

    filtered_out_total = 0
    n_covered_total = 0
    held: list = []
    sel: dict = {}
    n_repl = 0
    for i, T in enumerate(rl):
        row = day_score.get(T, pd.Series(dtype=float))
        if len(row) < 50:
            sel[T] = set(held) if held else set()
            continue
        ranked = row.sort_values(ascending=False)
        pool = ranked.head(TOPN_POOL)
        cov = [(s, ratio_lookup[(T, s)]) for s in pool.index if (T, s) in ratio_lookup]
        n_covered_total += len(cov)
        cut: set[str] = set()
        if cov:
            med = float(np.median([r for _, r in cov]))
            cut = {s for s, r in cov if r >= med}
        filtered_out_total += len(cut)
        eff = ranked[~ranked.index.isin(cut)]  # 剔除拥挤票后按原排名续位
        top5 = eff.head(TOP_K).index.tolist()
        keepzone = eff.head(KEEP_ZONE).index.tolist()
        new = [c for c in held if c in keepzone]
        for c in top5:
            if len(new) >= TOP_K:
                break
            if c not in new:
                new.append(c)
        sel[T] = set(new)
        n_repl += len(set(new) - set(held)) if i > 0 else TOP_K
        held = new

    daily = []
    for i, T in enumerate(rl):
        T_next = rl[i + 1] if i + 1 < len(rl) else close.index[-1]
        idx = ret.index[(ret.index > T) & (ret.index <= T_next)]
        if idx.empty or not sel[T]:
            continue
        port = ret.loc[idx, sorted(sel[T])].mean(axis=1)
        if i > 0:
            replaced = len(sel[T] - sel[rl[i - 1]]) / TOP_K
            if replaced:
                port = port.copy()
                port.iloc[0] -= replaced * COST
        daily.append(port)
    info = {"n_rebalance": len(rl), "n_replaced": n_repl,
            "n_filtered_out": filtered_out_total, "n_covered": n_covered_total}
    if not daily:
        return pd.Series(dtype=float), info, {"holdings": {}}
    daily_ret = pd.concat(daily).dropna()
    return daily_ret, info, {"holdings": {str(T): sorted(s) for T, s in sel.items()}}


def main() -> int:
    _force_utf8()
    inst = load_inst_holdings()
    print(f"inst holdings rows: {len(inst)}, symbols: {inst['symbol'].nunique()}")
    idx = H.load_index()
    out: dict = {}
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        score = _score(df)
        no_data = window.startswith("2015")
        r_win = pd.DataFrame(columns=["date", "symbol", "ratio"])
        if not no_data:
            r_win = build_window_ratios(df, score, window, inst)
        out[window] = {}
        # A 臂 = 定型 buffer 原样
        dr_a, info_a = run_arm(df, score, "buffer")
        if dr_a.empty:
            raise RuntimeError(f"A 臂空收益: {window}")
        m_a = H.compute_metrics(dr_a)
        # C 臂
        if no_data:
            dr_c, info_c, hold_c = dr_a, dict(info_a), {"holdings": {}}
        else:
            dr_c, info_c, hold_c = run_arm_c(df, score, r_win)
            if dr_c.empty:
                raise RuntimeError(f"C 臂空收益: {window}")
        m_c = H.compute_metrics(dr_c)
        t0, t1 = dr_a.index.min(), dr_a.index.max()
        uni = H.window_universe_benchmark(df)
        uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
        uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
        idx_total = H._idx_total(idx, t0, t1)
        diff = m_c["total"] - m_a["total"]
        # 两臂重合度 (月末持仓)
        overlap = None
        if not no_data:
            ha = _buffer_holdings(df, score)
            hc = hold_c["holdings"]
            ovs = [len(set(ha.get(T, [])) & set(hc.get(T, []))) / TOP_K
                   for T in hc if T in ha]
            overlap = round(float(np.mean(ovs)), 3) if ovs else None
        out[window] = {
            "no_data": no_data,
            "A": {"total": round(m_a["total"], 4), "vs_universe": round(m_a["total"] - uni_total, 4),
                  "vs_index": round(m_a["total"] - idx_total, 4), "maxdd": round(m_a["maxdd"], 4),
                  **info_a},
            "C": {"total": round(m_c["total"], 4), "vs_universe": round(m_c["total"] - uni_total, 4),
                  "vs_index": round(m_c["total"] - idx_total, 4), "maxdd": round(m_c["maxdd"], 4),
                  **info_c},
            "C_minus_A": round(diff, 4),
            "arm_overlap_avg": overlap,
        }
        print(f"{window:<12} A={m_a['total']*100:>+7.1f}% C={m_c['total']*100:>+7.1f}% "
              f"diff={diff*100:>+6.2f}pp dd_a={m_a['maxdd']*100:>6.1f}% dd_c={m_c['maxdd']*100:>6.1f}% "
              f"overlap={overlap}")

    # ── Gate (预注册写死) ──
    avail = [w for w in H.WINDOWS if not out[w]["no_data"] and OBJECTIVE[w] != "纯牛"]
    diffs = [out[w]["C_minus_A"] for w in avail]
    wins = sum(1 for x in diffs if x >= 0)
    need = math.ceil(2 / 3 * len(avail))
    avg_diff = float(np.mean(diffs)) if diffs else float("nan")
    gate_pass = bool(wins >= need and avg_diff >= 0.005)
    niu = [w for w in H.WINDOWS if OBJECTIVE[w] == "纯牛"]
    summary = {
        "gate": {
            "available_nonniu_windows": avail,
            "wins_C_ge_A": f"{wins}/{len(avail)}",
            "required": f">= {need} and avg_diff >= +0.5pp",
            "avg_diff_pp": round(avg_diff * 100, 2),
            "PASS": gate_pass,
        },
        "niu_windows_disclosure": {w: out[w]["C_minus_A"] for w in niu},
        "cost_coverage": {w: {"covered": out[w]["C"].get("n_covered"),
                              "filtered": out[w]["C"].get("n_filtered_out")}
                          for w in H.WINDOWS if not out[w]["no_data"]},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "inst_cold_result.json"
    fp.write_text(json.dumps({"prereg": "prereg_inst_cold_intersection.md",
                              "windows": _jsonable(out), "summary": summary},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


def build_window_ratios(df: pd.DataFrame, score: pd.Series, window: str,
                        inst: pd.DataFrame) -> pd.DataFrame:
    """单窗 inst_ratio 构建 (见 build_inst_ratio 逻辑, 拆窗以免重复加载面板)。"""
    inst_by_sym: dict[str, pd.DataFrame] = {
        s: g[["period_end", "value"]].sort_values("period_end")
        for s, g in inst.groupby("symbol")
    }
    d = df.copy()
    d["score"] = score.values
    d["fmkt"] = d["amount"] * 100.0 / d["turn"].replace(0, np.nan) / 1e8
    dates = sorted(d["date"].unique())
    s = pd.Series(pd.DatetimeIndex(dates))
    mends = [T for T in s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()
             if T in set(d["date"])]
    out = []
    for T in mends:
        g = d[d["date"] == T].dropna(subset=["score"])
        if len(g) < 50:
            continue
        pool = g.sort_values("score", ascending=False).head(TOPN_POOL)
        t_date = T.date() if hasattr(T, "date") else T
        for _, row in pool.iterrows():
            h = inst_by_sym.get(row["symbol"])
            if h is None:
                continue
            usable = h[h["period_end"] <= t_date - timedelta(days=LAG_DAYS)]
            if usable.empty:
                continue
            v = float(usable.iloc[-1]["value"])
            fm = row["fmkt"]
            if not np.isfinite(fm) or fm <= 0:
                continue
            out.append({"date": T, "symbol": row["symbol"],
                        "ratio": v / (fm * 1e8), "period": str(usable.iloc[-1]["period_end"])})
    return pd.DataFrame(out, columns=["date", "symbol", "ratio", "period"])


def _buffer_holdings(df: pd.DataFrame, score: pd.Series) -> dict:
    """复刻 A 臂 buffer 持仓 (用于重合度披露)。"""
    d = df.copy()
    d["score"] = score.values
    dates = sorted(d["date"].unique())
    rl = [T for T in month_ends(dates) if T in set(d["date"])]
    day_score = {T: g.dropna(subset=["score"]).set_index("symbol")["score"] for T, g in d.groupby("date")}
    held: list = []
    sel: dict = {}
    for i, T in enumerate(rl):
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
        sel[T] = sorted(new)
        held = new
    return {str(T): v for T, v in sel.items()}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (datetime, pd.Timestamp, date)):
        return str(o)
    return o


if __name__ == "__main__":
    raise SystemExit(main())
