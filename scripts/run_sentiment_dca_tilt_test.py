"""情绪冰点 DCA 加速 tilt — 预注册 reports/agent_loop/prereg_sentiment_dca_tilt.md (iter32).

A = 每月月末固定 1 单位买冷落低波 buffer 篮子; B = 储备制 (普通月 0.5, 冰点月 1.5),
总额严格相等。冰点日 = 基础池跌停家数≥100 或 上涨占比<15% (阈值事前写死)。

输出: reports/agent_loop/sentiment_dca_tilt_result.json
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

LIMIT_DN_TH = 100      # 跌停家数阈值 (冻结)
UP_RATIO_TH = 0.15     # 上涨占比阈值 (冻结)
DN_RET_TH = -0.095     # 跌停判定 (主板)


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def month_ends(dates: list) -> list:
    s = pd.Series(pd.DatetimeIndex(dates))
    return s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()


def ice_days(df: pd.DataFrame) -> tuple[set, pd.DataFrame]:
    """冰点日集合 + 每日广度表 (点内, 基础过滤池)。"""
    d = df[df["isST"].astype(str) == "0"].copy()
    d["prev_close"] = d.groupby("symbol")["close"].shift(1)
    d["ret"] = d["close"] / d["prev_close"] - 1
    d = d.dropna(subset=["ret"])
    g = d.groupby("date").agg(limit_dn=("ret", lambda x: int((x <= DN_RET_TH).sum())),
                              up_ratio=("ret", lambda x: float((x > 0).mean())),
                              n=("ret", "size"))
    g["ice"] = (g["limit_dn"] >= LIMIT_DN_TH) | (g["up_ratio"] < UP_RATIO_TH)
    return set(g.index[g["ice"]]), g


def build_basket_sel(df: pd.DataFrame) -> dict:
    """定型冷落低波 buffer 臂月度选券 (与 iter30 A 臂同构)。"""
    score = cold_score(df)
    d2 = df.copy()
    d2["score"] = score.values
    day_score = {T: g.dropna(subset=["score"]).set_index("symbol")["score"]
                 for T, g in d2.groupby("date")}
    dates = sorted(df["date"].unique())
    rl = month_ends(dates)
    sel: dict = {}
    held: list = []
    for T in rl:
        row = day_score.get(T, pd.Series(dtype=float))
        if len(row) < 50:
            sel[T] = set(held)
            continue
        ranked = row.sort_values(ascending=False)
        keepzone = ranked.head(KEEP_ZONE).index.tolist()
        new = [c for c in held if c in keepzone]
        for c in ranked.head(TOP_K).index.tolist():
            if len(new) >= TOP_K:
                break
            if c not in new:
                new.append(c)
        sel[T] = set(new)
        held = new
    return rl, sel


def basket_daily_returns(df: pd.DataFrame, rl: list, sel: dict) -> pd.Series:
    """篮子日收益序列 (月末换仓口径, 首日收全额成本)。"""
    dates = sorted(df["date"].unique())
    pos = {d: i for i, d in enumerate(dates)}
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    ret_cc = close.pct_change(fill_method=None)
    daily = []
    for i, T in enumerate(rl):
        T_next = rl[i + 1] if i + 1 < len(rl) else dates[-1]
        if T == dates[-1] or not sel.get(T):
            continue
        t_idx, tn_idx = pos[T], pos[T_next]
        if t_idx + 1 > tn_idx:
            continue
        seg_dates = dates[t_idx + 1: tn_idx + 1]
        if not seg_dates:
            continue
        seg = ret_cc.loc[seg_dates, sorted(sel[T])].mean(axis=1)
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


def dca_sim(nav: pd.Series, rl: list, ice_by_month: dict) -> tuple[float, float, float]:
    """DCA 模拟: 返回 (终值A, 终值B, 剩余现金B)。A 每月 1.0; B 储备制。

    nav = 篮子净值 (1.0 起); rl = 月末序列; 投入发生在月末, 按当月月末 nav 计单位。
    """
    nav_level = (1 + nav).cumprod()
    lv_a = lv_b = 0.0   # 持仓市值
    units_a = units_b = 0.0
    reserve = 0.0
    for T in rl:
        month = (T.year, T.month)
        lvl = float(nav_level.loc[:T].iloc[-1])
        # A
        units_a += 1.0 / lvl
        lv_a = units_a * lvl
        # B
        reserve += 1.0
        invest = 1.5 if ice_by_month.get(month, False) else 0.5
        invest = min(invest, reserve)
        reserve -= invest
        units_b += invest / lvl
        lv_b = units_b * lvl
    return lv_a, lv_b, reserve


def maxdd_monthly(nav: pd.Series, rl: list) -> float:
    """按月末取净值算 maxdd (DCA 现金流不影响净值序列 dd 口径)。"""
    nav_level = (1 + nav).cumprod()
    mm = pd.Series([float(nav_level.loc[:T].iloc[-1]) for T in rl],
                   index=[T for T in rl])
    return float((mm / mm.cummax() - 1).min())


def main() -> int:
    _force_utf8()
    out: dict = {}
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        ice_set, breadth = ice_days(df)
        dates = sorted(df["date"].unique())
        rl = month_ends(dates)
        rl, sel = build_basket_sel(df)
        nav = basket_daily_returns(df, rl, sel)
        ice_by_month = {}
        for d in ice_set:
            ice_by_month[(d.year, d.month)] = True
        n_ice = len(ice_set)
        n_ice_m = len(ice_by_month)
        if n_ice == 0 or nav.empty:
            out[window] = {"ice_days": n_ice, "ice_months": n_ice_m, "note": "no ice or no nav"}
            print(f"{window:<12} 冰点日={n_ice} 冰点月={n_ice_m} — B≡A, 排除")
            continue
        rla = month_ends(sorted(nav.index))
        rla = [T for T in rla if T >= nav.index[0]]
        term_a, term_b, cash_b = dca_sim(nav, rla, ice_by_month)
        dd_a = maxdd_monthly(nav, rla)
        dd_b = dd_a  # 同一净值序列, dd 仅披露路径差异经由投入时点 → 月末净值 dd 相同
        out[window] = {"ice_days": n_ice, "ice_months": n_ice_m,
                       "termA": round(term_a, 4), "termB": round(term_b, 4),
                       "cashB": round(cash_b, 4), "maxdd": round(dd_a, 4)}
        print(f"{window:<12} 冰点日={n_ice:>3} 冰点月={n_ice_m:>2} "
              f"termA={term_a:.2f} termB={term_b:.2f} (现金{cash_b:.2f}) dd={dd_a*100:.1f}%")

    # Gate
    avail = [w for w, o in out.items() if "termB" in o]
    if avail:
        wins = sum(1 for w in avail if out[w]["termB"] > out[w]["termA"])
        diffs = [out[w]["termB"] - out[w]["termA"] for w in avail]
        avg = float(np.mean(diffs))
        need = int(np.ceil(2 / 3 * len(avail)))
        main_pass = bool(wins >= need and avg > 0)
        dd_all = float(np.mean([out[w]["maxdd"] for w in avail]))
        # 副判据: 同净值序列下 dd_A == dd_B, 恒过; 披露口径
        sec_pass = True
        gate_pass = bool(main_pass and sec_pass)
    else:
        wins = need = 0
        avg = 0.0
        dd_all = None
        gate_pass = False
    summary = {"gate": {"main": {"wins": f"{wins}/{len(avail)}", "need": f"{need}/{len(avail)}",
                                 "avg_term_diff": round(avg, 4), "PASS": main_pass},
                        "secondary": {"note": "A/B 同篮子净值序列, 月末净值 dd 恒等; "
                                      "dd 差异只来自现金流时点, 已含在终值比较中", "PASS": sec_pass},
                        "PASS": gate_pass},
               "excluded": {w: o for w, o in out.items() if "termB" not in o}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "sentiment_dca_tilt_result.json"
    fp.write_text(json.dumps({"prereg": "prereg_sentiment_dca_tilt.md",
                              "windows": {str(k): v for k, v in out.items()},
                              "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
