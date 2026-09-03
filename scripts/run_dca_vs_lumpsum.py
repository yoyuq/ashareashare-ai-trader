"""DCA (月度定投) vs 一次性投入 — 冷落低波篮子, 8 窗真实回测.

预注册: reports/agent_loop/prereg_dca_vs_lumpsum.md (gate 写死于跑之前).
复用已 PASS 的冷落低波实现 (run_own_strategies.py::_cold_lowvol 同公式):
月末收盘重选 top5 (低换手+低波+小市值), 65bp 现实成本 (1万资金真实费率).

两臂 (同现金流对比):
  LUMP: t0 一次性投 G = 全部现金流总额
  DCA:  t0 投 1万, 其后每个月末调仓日追加 1千 (65bp 买入成本)
资金效率 = 年化 IRR (XIRR, 二分求解); maxdd = nav 累计路径回撤 (两臂同口径).

用法: python scripts/run_dca_vs_lumpsum.py
输出: reports/agent_loop/dca_vs_lumpsum_result.json
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

TOP_K = 5
COST = 65.0 / 1e4          # 1万资金真实往返费率 (reports/cost_reality_result.json)
INIT = 10000.0
MONTHLY = 1000.0
WINDOWS = {k: v for k, v in H.WINDOWS.items() if k in (
    "2018熊", "2019牛", "2020牛转崩", "2021白马转小盘",
    "2022熊", "2023震荡", "2024震荡", "2025-26现期")}


def factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", "symbol", "turn", "amount", "close"]].copy()
    s = df.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    out["std20"] = s.reindex(df.index)
    fmkt = df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8
    out["log_mkt"] = np.log1p(fmkt)
    return out


def cold_lowvol_score(df: pd.DataFrame) -> pd.Series:
    f = factors(df)
    r_turn = f["turn"].groupby(df["date"]).rank(pct=True)
    r_std = f["std20"].groupby(df["date"]).rank(pct=True)
    r_mkt = f["log_mkt"].groupby(df["date"]).rank(pct=True)
    return -(r_turn + r_std + r_mkt)


def month_end_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(dates)
    return s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()


def run_window(fp: str) -> dict:
    """返回 (月度调仓日列表, 每个 (T, T_next] 期间的篮子日收益 DataFrame)。"""
    df = H.load_base_window(fp)
    if df.empty:
        raise RuntimeError(f"窗口数据为空: {fp}")
    score = cold_lowvol_score(df)
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    ret = close.pct_change(fill_method=None)
    dates = close.index
    me = month_end_dates(dates)

    # 选券: 每个月末调仓日取 score 最大的 5 只 (有效值 ≥5 才成立)
    sel: dict[pd.Timestamp, list[str]] = {}
    for T in me:
        row = score[df["date"].to_numpy() == T]
        row = pd.Series(row.values, index=df.loc[row.index, "symbol"]).dropna()
        if len(row) >= TOP_K:
            sel[T] = row.nlargest(TOP_K).index.tolist()

    valid = [T for T in me if T in sel]
    if not valid:
        raise RuntimeError("无有效调仓日 (因子预热不足)")

    # 逐期间日收益 (T 收盘入场 → T_next 收盘)
    rows = []
    for i, T in enumerate(valid):
        T_next = valid[i + 1] if i + 1 < len(valid) else dates[-1]
        idx = ret.index[(ret.index > T) & (ret.index <= T_next)]
        if len(idx) == 0:
            continue
        seg = ret.loc[idx, sel[T]]
        nav_ret = seg.mean(axis=1).fillna(0.0)  # 停牌票当日 0 (fffill 等价)
        # 换仓成本: 相对上期替换比例 × 65bp, 扣在期首
        prev = sel[valid[i - 1]] if i > 0 else []
        replaced = len(set(sel[T]) - set(prev)) / TOP_K
        nav_ret.iloc[0] -= replaced * COST
        rows.append(nav_ret)
    nav = pd.concat(rows).sort_index()
    return valid, sel, nav, dates


def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    """二分求年化 IRR; cashflows: (日期, 金额), 投入为负/回收为正。"""
    t0 = min(d for d, _ in cashflows)
    def npv(r: float) -> float:
        return sum(cf / (1.0 + r) ** ((d - t0).days / 365.0) for d, cf in cashflows)
    lo, hi = -0.95, 20.0
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def maxdd(nav: pd.Series) -> float:
    eq = (1.0 + nav).cumprod()  # nav 是日收益序列
    return float((eq / eq.cummax() - 1.0).min())


def main() -> int:
    out = {}
    for wname, fp in WINDOWS.items():
        valid, sel, nav, dates = run_window(fp)
        t0 = valid[0]
        end = dates[-1]
        n_contrib = len([T for T in valid if T > t0])

        # DCA: t0 1万 + 每月末 1千 (65bp 买入成本)
        units, cf = 0.0, []
        nav_pos = 1.0
        dates_list = list(nav.index)
        nav_by_date = nav.to_dict()
        # nav 累计
        cum = nav.cumprod()
        units = INIT / 1.0  # nav(t0)=1 (cum 首值=1+r_{t0+1}; 以 t0 为基准归一)
        # 重新归一: t0 当日 nav=1, 其后逐日 cum
        cum = pd.concat([pd.Series([1.0], index=[t0]), (1.0 + nav).cumprod()]).sort_index()
        cum = cum[~cum.index.duplicated()]
        units = INIT
        value_at = {}
        for T in valid:
            navT = float(cum.loc[T])
            if T == t0:
                units = INIT
            else:
                units += MONTHLY * (1 - COST) / navT
            value_at[T] = units * navT
        v_dca = units * float(cum.iloc[-1])
        cf = [(t0, -INIT)] + [(T, -MONTHLY) for T in valid if T > t0] + [(end, v_dca)]
        irr_dca = xirr(cf)
        dd_dca = maxdd(nav)

        # LUMP: G 全额 t0 投入
        G = INIT + n_contrib * MONTHLY
        v_lump = G * float(cum.iloc[-1])
        irr_lump = (v_lump / G) ** (365.0 / max((end - t0).days, 1)) - 1.0
        dd_lump = maxdd(nav)

        days = (end - t0).days
        out[wname] = {
            "t0": str(t0.date()), "end": str(end.date()), "days": days,
            "n_monthly_contrib": n_contrib, "G_lump": G,
            "dca_terminal": round(v_dca, 0), "dca_total_in": round(INIT + n_contrib * MONTHLY, 0),
            "lump_terminal": round(v_lump, 0),
            "irr_dca": round(irr_dca, 4), "irr_lump": round(irr_lump, 4),
            "irr_diff_pp": round((irr_dca - irr_lump) * 100, 2),
            "maxdd": round(dd_dca, 4),
        }
        print(f"{wname}: t0={t0.date()} DCA_IRR={irr_dca:+.1%} LUMP_IRR={irr_lump:+.1%} "
              f"diff={out[wname]['irr_diff_pp']:+.1f}pp dd={dd_dca:.1%}", flush=True)

    diffs = [v["irr_diff_pp"] for v in out.values()]
    dca_wins = sum(1 for v in out.values() if v["irr_dca"] >= v["irr_lump"])
    avg_diff = float(np.mean(diffs))
    verdict = bool(dca_wins >= 5 and avg_diff >= -1.0)
    summary = {"dca_wins": f"{dca_wins}/8", "avg_irr_diff_pp": round(avg_diff, 2),
               "PASS_定投叙事": verdict}
    print(json.dumps(summary, ensure_ascii=False))
    result = {"preregistration": "reports/agent_loop/prereg_dca_vs_lumpsum.md",
              "windows": out, "summary": summary}
    fp = ROOT / "reports" / "agent_loop" / "dca_vs_lumpsum_result.json"
    fp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
