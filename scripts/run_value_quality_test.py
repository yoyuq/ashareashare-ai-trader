"""价值质量复合选股 V — 预注册 reports/agent_loop/prereg_value_quality.md (iter30).

A = 冷落低波定型 buffer 臂 (对照); V = 季频价值质量复合 (三重排雷 + PE/PB 最便宜 top10)。
Gate: V vs 匹配 universe, 非纯牛 6 窗 ≥4 赢且 avg ≥+0.5pp/窗;
副判据: 纯牛窗 avg dd ≥ −40% 且 avg 收益 ≥ −10% (价值陷阱硬约束)。

输出: reports/agent_loop/value_quality_result.json
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
from fetch_inst_holdings import _score as cold_score  # noqa: E402

OBJECTIVE = {"2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
             "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
             "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
             "2024震荡": "纯牛", "2025-26现期": "震荡"}

V_TOP_K = 10
FUND_FP = ROOT / "replay_data" / "fundamentals.parquet"


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def quarter_ends(dates: list) -> list:
    """季末月 (3/6/9/12) 最后交易日。"""
    s = pd.Series(pd.DatetimeIndex(dates))
    q = s[s.dt.month.isin([3, 6, 9, 12])]
    return q.groupby([q.dt.year, q.dt.month]).apply(lambda x: x.iloc[-1]).tolist()


def latest_fundamentals(fund: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    """每票截至 T 的最新报告期 (pubDate ≤ T), 杜绝未来函数。"""
    f = fund[fund["pubDate"] <= T].sort_values("statDate").groupby("symbol").tail(1)
    return f.set_index("symbol")


def value_selections(df: pd.DataFrame, fund: pd.DataFrame, rl: list) -> dict:
    """V 臂选券: 三重排雷 (netProfit/roe/YOYNI>0) + PE/PB 等和最低 top10, 季频。"""
    day = {T: g.set_index("symbol") for T, g in df.groupby("date")}
    sel: dict = {}
    for T in rl:
        g = day.get(T)
        if g is None:
            continue
        f = latest_fundamentals(fund, T)
        m = g.join(f[["roeAvg", "netProfit", "YOYNI"]], how="inner")
        m = m[(m["netProfit"] > 0) & (m["roeAvg"] > 0) & (m["YOYNI"] > 0)]
        m = m[(m["peTTM"] > 0) & (m["pbMRQ"] > 0)]
        if len(m) < 50:
            sel[T] = set()
            continue
        r = m["peTTM"].rank(pct=True) + m["pbMRQ"].rank(pct=True)
        sel[T] = set(r.nsmallest(V_TOP_K).index)
    return sel


def run_arm(ret_cc: pd.DataFrame, rl: list, sel: dict, top_k: int) -> pd.Series:
    """通用拼装: (T, T_next] 新票 close-to-close, 成本首日。"""
    dates = ret_cc.index
    pos = {d: i for i, d in enumerate(dates)}
    daily = []
    for i, T in enumerate(rl):
        T_next = rl[i + 1] if i + 1 < len(rl) else dates[-1]
        if T == dates[-1] or not sel.get(T):
            continue
        t_idx, tn_idx = pos[T], pos[T_next]
        if t_idx + 1 > tn_idx:
            continue
        seg_dates = dates[t_idx + 1: tn_idx + 1]
        if len(seg_dates) == 0:
            continue
        sel_new = sorted(sel[T])
        seg = ret_cc.loc[seg_dates, sel_new].mean(axis=1)
        # 空仓→建仓也须收全额成本 (与定型引擎口径一致: 2022-03-01 教训)
        if i > 0 and rl[i - 1] in sel:
            repl = len(sel[T] - sel[rl[i - 1]]) / len(sel[T])
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
    fund = pd.read_parquet(FUND_FP)
    fund["pubDate"] = pd.to_datetime(fund["pubDate"])
    fund["statDate"] = pd.to_datetime(fund["statDate"])
    out: dict = {}
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        dates = sorted(df["date"].unique())
        s = pd.Series(pd.DatetimeIndex(dates))
        rl_q = quarter_ends(dates)
        rl_m = s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()
        close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        ret_cc = close.pct_change(fill_method=None)
        out[window] = {"objective": OBJECTIVE[window]}

        # A 定型臂
        score = cold_score(df)
        sel_a = {}
        d2 = df.copy()
        d2["score"] = score.values
        day_score = {T: g.dropna(subset=["score"]).set_index("symbol")["score"]
                     for T, g in d2.groupby("date")}
        held: list = []
        for T in rl_m:
            row = day_score.get(T, pd.Series(dtype=float))
            if len(row) < 50:
                sel_a[T] = set(held)
                continue
            ranked = row.sort_values(ascending=False)
            keepzone = ranked.head(KEEP_ZONE).index.tolist()
            new = [c for c in held if c in keepzone]
            for c in ranked.head(TOP_K).index.tolist():
                if len(new) >= TOP_K:
                    break
                if c not in new:
                    new.append(c)
            sel_a[T] = set(new)
            held = new
        dr_a = run_arm(ret_cc, rl_m, sel_a, TOP_K)
        m_a = H.compute_metrics(dr_a)
        out[window]["A"] = {"total": round(m_a["total"], 4), "maxdd": round(m_a["maxdd"], 4)}

        # V 价值质量臂
        f_in_window = fund[(fund["pubDate"] >= pd.Timestamp(dates[0]) - pd.Timedelta(days=400))
                           & (fund["pubDate"] <= pd.Timestamp(dates[-1]))]
        out[window]["no_data"] = bool(len(f_in_window) == 0)
        sel_v = value_selections(df, fund, rl_q)
        dr_v = run_arm(ret_cc, rl_q, sel_v, V_TOP_K)
        if dr_v.empty:
            out[window]["V"] = {"total": None, "maxdd": None, "note": "no_data or empty"}
        else:
            m_v = H.compute_metrics(dr_v)
            out[window]["V"] = {"total": round(m_v["total"], 4), "maxdd": round(m_v["maxdd"], 4)}

        # 匹配 universe (同基础过滤全池等权, 季频换仓口径)
        uni_sel = {T: set(g.set_index("symbol").index) for T, g in df.groupby("date")
                   if T in set(rl_q)}
        dr_u = run_arm(ret_cc, rl_q, uni_sel, 1e9)
        m_u = H.compute_metrics(dr_u)
        out[window]["universe"] = {"total": round(m_u["total"], 4), "maxdd": round(m_u["maxdd"], 4)}

        if out[window]["V"]["total"] is not None:
            out[window]["V_minus_U"] = round(out[window]["V"]["total"] - out[window]["universe"]["total"], 4)
        ovs = [len(sel_v.get(T, set()) & sel_a.get(T, set())) / TOP_K
               for T in rl_q if sel_v.get(T) and sel_a.get(T)]
        out[window]["V_A_overlap"] = round(float(np.mean(ovs)), 3) if ovs else None
        t = out[window]["V"]["total"]
        print(f"{window:<12} [{out[window]['objective']}] A={out[window]['A']['total']*100:>+8.1f}% "
              f"V={'no_data' if t is None else f'{t*100:>+8.1f}%'} "
              f"U={out[window]['universe']['total']*100:>+8.1f}% "
              f"V−U={'—' if t is None else f'{out[window]['V_minus_U']*100:>+7.2f}pp'} "
              f"ddV={'—' if t is None else f'{out[window]['V']['maxdd']*100:>6.1f}%'}")

    # Gate
    avail = [w for w in H.WINDOWS if out[w]["V"]["total"] is not None
             and OBJECTIVE[w] != "纯牛"]
    bulls = [w for w in H.WINDOWS if out[w]["V"]["total"] is not None
             and OBJECTIVE[w] == "纯牛"]
    if avail:
        diffs = [out[w]["V_minus_U"] for w in avail]
        wins = sum(1 for x in diffs if x >= 0)
        avg = float(np.mean(diffs))
        need = int(np.ceil(2 / 3 * len(avail)))
        main_pass = bool(wins >= need and avg >= 0.005)
    else:
        wins, avg, need, main_pass = 0, 0.0, 0, False
    if bulls:
        dd_b = float(np.mean([out[w]["V"]["maxdd"] for w in bulls]))
        ret_b = float(np.mean([out[w]["V"]["total"] for w in bulls]))
        bull_pass = bool(dd_b >= -0.40 and ret_b >= -0.10)
    else:
        dd_b = ret_b = None
        bull_pass = False
    gate_pass = bool(main_pass and bull_pass)
    summary = {
        "gate": {"main": {"wins": f"{wins}/{len(avail)}", "need": f"{need}/{len(avail)}",
                          "avg_diff_pp": round(avg * 100, 2), "PASS": main_pass},
                 "bull_guard": {"avg_dd": None if dd_b is None else round(dd_b, 4),
                                "avg_ret": None if ret_b is None else round(ret_b, 4),
                                "required": "dd>=-40% & ret>=-10%", "PASS": bull_pass},
                 "PASS": gate_pass},
        "no_data_windows": [w for w in H.WINDOWS if out[w].get("no_data")],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "value_quality_result.json"
    fp.write_text(json.dumps({"prereg": "prereg_value_quality.md",
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
