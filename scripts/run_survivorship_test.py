"""幸存者偏差重测 + ST/面值排雷 (预注册) — 补回退市股后冷落 tilt 是否仍跑赢上证。

背景: 冷落 beta 5/5 跑赢上证, 但 replay 窗口只含「存活」票 (load_snapshot_basic 当前快照),
1178 只退市股缺失。本脚本把退市股 (replay_data/delisted_daily.parquet) 补回各窗口, 重测:
  问题2 (幸存者偏差): 补回退市股后, 冷落 bottom-100 是否仍跑赢上证?
  问题3 (排雷是否有用): 现在篮子里真有「低换手+快死(ST/面值退市)」的票了, ST/面值排雷是否降险不伤收益?

排雷过滤器 (预注册, 首原则, point-in-time 全覆盖):
  剔除 isST==1 (ST/*ST = 正式退市风险标志) 或 close<3.0 (面值退市风险区)。
  「只删不调权」: 剔除后剩余等权, 不回填第 101 冷票。

方法: 每月(20交易日)按换手率取 bottom-100 等权再平衡, 扣费 31bp/次, 与之前同口径。
  退市股 delisting 后价格 ffill 冻结在最后收盘 (面值退市型主导, 其崩溃已体现在 <1 元价格里;
  冻结残留为 lower-bound 幸存者偏差, 真实影响略更大 — 如实披露)。

预注册判据 (核心 5 非重叠窗口):
  - 幸存者: 补回退市股后 return 跑赢上证 >= 4/5 → 冷落溢价稳健; 否则幸存者偏差证伪。
  - 排雷: 补回后 +排雷 的 maxDD 不劣于补回后基线 (<=+0.5pp) 且 return 不劣 (>= -2pp) >= 4/5
    → 排雷在真实退市风险下有效。

用法: python scripts/run_survivorship_test.py
输出: reports/survivorship_test_result.json + .md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "survivorship_test_result.json"
K = 100
UNIVERSE_N = 800
REBAL = 20
COST_BPS = 31.0
PRICE_FLOOR = 3.0
DELIST_FP = ROOT / "replay_data" / "delisted_daily.parquet"

WINDOWS = {
    "2018熊":     {"win": "replay_data/daily_2018-01-01_2018-12-31.parquet", "core": True},
    "2019牛":     {"win": "replay_data/daily_2019-01-01_2019-12-31.parquet", "core": True},
    "2020牛转崩": {"win": "replay_data/daily_2020-06-01_2021-02-28.parquet", "core": True},
    "2024震荡":   {"win": "replay_data/daily_2024-01-01_2024-12-31.parquet", "core": True},
    "2025-26现期": {"win": "replay_data/daily_2025-10-08_2026-07-31.parquet", "core": True},
}


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {"return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
    dd = float((eq / eq.cummax() - 1.0).min()) * 100.0
    dr = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0
    return {"return_pct": round(ret, 3), "max_dd_pct": round(dd, 3), "sharpe": round(sharpe, 3)}


def _liquid_universe(df: pd.DataFrame, top_n: int = UNIVERSE_N) -> list[str]:
    df = df[df.symbol != "sh.000001"].copy()
    df["isST"] = df["isST"].astype(str)
    inv = df[(df.isST == "0") & (df.is_trade == 1) & (df.peTTM > 0) & (df.pbMRQ > 0)]
    med = inv.groupby("symbol")["amount"].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()


def _rebalanced_curve(df: pd.DataFrame, uni: list[str], every: int, cost_bps: float,
                      keep_fn=None) -> tuple[pd.Series, float]:
    dts = sorted(df["date"].unique())
    piv = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    piv = piv.reindex(dts).reindex(columns=[s for s in uni if s in piv.columns])
    raw = piv.copy()
    piv = piv.ffill()
    turnp = df.pivot_table(index="date", columns="symbol", values="turn", aggfunc="last").reindex(dts)
    stp = df.pivot_table(index="date", columns="symbol", values="isST", aggfunc="last").reindex(dts)
    R = piv.pct_change()

    reb = list(range(0, len(dts), every))
    reb_set = set(reb)
    port_ret = pd.Series(0.0, index=dts)
    sizes = []
    for i, ri in enumerate(reb):
        nxt = reb[i + 1] if i + 1 < len(reb) else len(dts)
        t = dts[ri]
        turn_t = turnp.loc[t].reindex(piv.columns).dropna()
        sel = turn_t.nsmallest(K).index.tolist()
        if keep_fn is not None:
            sel = keep_fn(sel, t, raw, stp)
        if not sel:
            sizes.append(0)
            continue
        sizes.append(len(sel))
        end = min(nxt, len(dts) - 1)
        for j in range(ri + 1, end + 1):
            port_ret.iloc[j] = float(R.iloc[j].reindex(sel).mean())

    cost = cost_bps / 10000.0
    eq = 1.0
    vals = []
    for j in range(len(dts)):
        if j > 0:
            eq *= (1.0 + float(port_ret.iloc[j]))
        if j in reb_set:
            eq *= (1.0 - cost)
        vals.append(eq)
    return pd.Series(vals, index=dts), float(np.mean(sizes))


def _no_filter(sel, t, raw, stp): return sel


def _st_price_filter(sel, t, raw, stp):
    """剔除 isST==1 或 close<3.0 (面值退市/ST 退市风险)。"""
    px = raw.loc[t].reindex(sel)
    st = stp.loc[t].reindex(sel).astype(str)
    keep = []
    for s in sel:
        is_st = st[s] == "1"
        low_px = pd.notna(px[s]) and px[s] < PRICE_FLOOR
        if not is_st and not low_px:
            keep.append(s)
    return keep


def main() -> int:
    _force_utf8()
    if not DELIST_FP.exists():
        print(f"缺 {DELIST_FP} — 先跑 scripts/fetch_delisted_daily.py")
        return 1
    delist = pd.read_parquet(DELIST_FP)
    delist["date"] = pd.to_datetime(delist["date"])

    idx = pd.read_parquet(ROOT / "replay_data" / "index_series.parquet")
    idx["date"] = pd.to_datetime(idx["date"])

    report = {"date": datetime.now().isoformat(timespec="seconds"), "K": K,
              "REBAL": REBAL, "COST_BPS": COST_BPS, "PRICE_FLOOR": PRICE_FLOOR,
              "windows": [], "verdict": {}}

    print("=" * 118)
    print(f"幸存者偏差重测: 补回退市股后 冷落 bottom-{K} 每月再平衡 扣费{COST_BPS}bp vs 上证")
    print("=" * 118)

    for label, cfg in WINDOWS.items():
        df = pd.read_parquet(ROOT / cfg["win"])
        df["date"] = pd.to_datetime(df["date"])
        T, T_end = df.date.min(), df.date.max()

        idx_win = idx[(idx["date"] >= T) & (idx["date"] <= T_end)]
        sh = idx_win[idx_win.symbol == "sh.000001"].set_index("date")["close"]
        m_sh = _metrics(sh)

        # 退市股中在该窗口内仍有交易的票 (切片到窗口区间)
        ddel = delist[(delist["date"] >= T) & (delist["date"] <= T_end)]
        n_delist_sym = ddel["symbol"].nunique()

        surv = df[df.symbol != "sh.000001"].copy()
        merged = pd.concat([surv, ddel], ignore_index=True) if not ddel.empty else surv
        for _c in ("turn", "peTTM", "pbMRQ", "amount", "close"):
            if _c in merged.columns:
                merged[_c] = pd.to_numeric(merged[_c], errors="coerce")
        if "isST" in merged.columns:
            merged["isST"] = merged["isST"].astype(str)
        if "is_trade" not in merged.columns:
            merged["is_trade"] = 1

        uni_surv = _liquid_universe(surv)
        uni_full = _liquid_universe(merged)

        arms = {
            "A幸存者": (surv, uni_surv, _no_filter),
            "B补回退市": (merged, uni_full, _no_filter),
            "C补回+排雷": (merged, uni_full, _st_price_filter),
        }
        row = {"window": label, "index": m_sh, "delisted_symbols_in_window": n_delist_sym,
               "arms": {}}
        print(f"\n### {label}  ({T.date()} -> {T_end.date()}, 窗口内退市股 {n_delist_sym} 只)")
        print(f"  上证综指:    ret={m_sh['return_pct']:>7.2f}%  dd={m_sh['max_dd_pct']:>7.2f}%  sh={m_sh['sharpe']:>5.2f}")
        for aname, (d, uni, fn) in arms.items():
            eq, sz = _rebalanced_curve(d, uni, REBAL, COST_BPS, fn)
            m = _metrics(eq)
            dret = m["return_pct"] - m_sh["return_pct"]
            row["arms"][aname] = {**m, "dret_pp": round(dret, 2), "basket_size": round(sz, 1)}
            print(f"  {aname:>8}: ret={m['return_pct']:>7.2f}%  dd={m['max_dd_pct']:>7.2f}%  "
                  f"sh={m['sharpe']:>5.2f}  篮={sz:>4.1f}  Δret={dret:>+7.2f}pp")
        report["windows"].append(row)

    ws = report["windows"]
    n = len(ws)

    def _beats_idx(arm):
        return sum(1 for w in ws if (not np.isnan(w["arms"][arm]["return_pct"])
                                     and w["arms"][arm]["return_pct"] > w["index"]["return_pct"]))

    def _not_worse(arm, base, mode="return_pct", tol=2.0):
        c = 0
        for w in ws:
            a = w["arms"][arm][mode]
            b = w["arms"][base][mode]
            if np.isnan(a) or np.isnan(b):
                continue
            if mode == "max_dd_pct":
                if a <= b + 0.5:
                    c += 1
            else:
                if a >= b - tol:
                    c += 1
        return c

    report["verdict"] = {
        "survivor_B_beats_index": f"{_beats_idx('B补回退市')}/{n}",
        "survivor_A_beats_index": f"{_beats_idx('A幸存者')}/{n}",
        "survivorship_impact": f"B vs A return: 见 windows (Δ)",
        "filter_C_dd_not_worse_B": f"{_not_worse('C补回+排雷', 'B补回退市', 'max_dd_pct')}/{n}",
        "filter_C_ret_not_worse_B": f"{_not_worse('C补回+排雷', 'B补回退市', 'return_pct')}/{n}",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 118)
    print("核心判据:")
    for k, v in report["verdict"].items():
        print(f"  {k}: {v}")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
