"""精细排雷过滤器 A/B (预注册) — 冷落篮子 vs +精细排雷 (连续亏损/利润暴跌/营收收缩)。

背景: 尾险排雷 A/B (run_tailrisk_filter_ab.py) 只测了「roe<=0」一个粗糙信号, 结论「粗糙排雷不增值」。
这不能否定「精细排雷(欺诈/退市/负催化识别)」的价值 (问题3)。本脚本用扩展后的 point-in-time 基本面
(2017Q3~2026Q2 全季度, pubDate 不偷看未来), 测三个精细信号:

精细排雷信号 (预注册, 全部 as-of 窗口起点 T, 用 pubDate <= T 取最新可得报告, 杜绝未来函数):
  - F_contloss 连续两年亏损: 最近 2 个已披露年报(Q4) netProfit 均 < 0  (ST/退市前兆)
  - F_crash    利润暴跌:      最新可得报告 YOYNI < -50%                  (负催化)
  - F_shrink   营收收缩:      最新可得报告 MBRevenue < 去年同期同季度      (业务萎缩)
  「只删不调权」: 剔除后剩余等权, 不回填第 101 冷票。

方法: 每月(20交易日)按换手率取 bottom-100 等权再平衡, 扣费 31bp/次, 与 run_tailrisk_filter_ab.py 同口径。

预注册判据 (核心 5 个非重叠窗口):
  - 主: 精细排雷(任一/合并) 后 maxDD 不劣于基线 (<=基线+0.5pp) 且 return 不劣于基线 (>=基线-2pp) >= 4/5
    → 若成立且在某窗口显著降 dd → 精细排雷有效; 若 dd/ret 与基线几乎无差 → 精细排雷也无效果(与粗糙排雷同)。
  - 副: 排雷后 return 跑赢上证 窗口数 不劣于基线。

用法: python scripts/run_refined_tailrisk_ab.py
前置: fundamentals.parquet 已含全 point-in-time 季度 (build_fundamental_panel.py 全量跑完)
输出: reports/refined_tailrisk_ab_result.json + .md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "refined_tailrisk_ab_result.json"
K = 100
UNIVERSE_N = 800
REBAL = 20
COST_BPS = 31.0
CRASH_YOYNI = -50.0  # 利润暴跌阈值 (同比腰斩)

WINDOWS = {
    "2018熊":     {"win": "replay_data/daily_2018-01-01_2018-12-31.parquet", "core": True},
    "2019牛":     {"win": "replay_data/daily_2019-01-01_2019-12-31.parquet", "core": True},
    "2020牛转崩": {"win": "replay_data/daily_2020-06-01_2021-02-28.parquet", "core": True},
    "2024震荡":   {"win": "replay_data/daily_2024-01-01_2024-12-31.parquet", "core": True},
    "2025-26现期": {"win": "replay_data/daily_2025-10-08_2026-07-31.parquet", "core": True},
    "2018Q4+2019": {"win": "replay_data/daily_2018-10-01_2019-12-31.parquet", "core": False},
    "2026H1":     {"win": "replay_data/daily_2026-02-21_2026-07-31.parquet", "core": False},
    "lookback2020": {"win": "replay_data/lookback_2020.parquet", "core": False},
    "lookback2024": {"win": "replay_data/lookback_2024.parquet", "core": False},
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


def _asof_refined(fund: pd.DataFrame, T: pd.Timestamp) -> tuple[set, set, set, str]:
    """as-of 窗口起点 T 计算三个精细排雷信号集合 (point-in-time, pubDate<=T)。"""
    fund = fund.copy()
    fund["statDate"] = pd.to_datetime(fund["statDate"])
    fund["pubDate"] = pd.to_datetime(fund["pubDate"], errors="coerce")
    avail = fund[fund["pubDate"].notna() & (fund["pubDate"] <= T)]
    fallback = ""
    if avail.empty:
        avail = fund
        fallback = " (无pubDate, 回退全量/轻微偷看)"
    used = avail["statDate"].max()

    contloss: set[str] = set()
    crash: set[str] = set()
    shrink: set[str] = set()

    # F_contloss: 最近2个年报(Q4) netProfit<0
    annual = avail[avail["statDate"].dt.month == 12].sort_values("statDate")
    for sym, grp in annual.groupby("symbol"):
        vals = grp["netProfit"].dropna().tolist()
        if len(vals) >= 2 and vals[-1] < 0 and vals[-2] < 0:
            contloss.add(sym)

    # F_crash / F_shrink: 最新可得报告
    latest = avail.sort_values("statDate").drop_duplicates("symbol", keep="last")
    for sym, row in latest.iterrows():
        yoy = row.get("YOYNI")
        if pd.notna(yoy) and yoy < CRASH_YOYNI:
            crash.add(sym)
        mb = row.get("MBRevenue")
        sd = row.get("statDate")
        if pd.notna(mb) and pd.notna(sd):
            prev_sd = sd - pd.DateOffset(years=1)
            prev = avail[(avail["symbol"] == sym) & (avail["statDate"] == prev_sd)]["MBRevenue"]
            if len(prev) and pd.notna(prev.iloc[0]) and mb < prev.iloc[0]:
                shrink.add(sym)

    flag = f"as-of={used.date()}{fallback}"
    return contloss, crash, shrink, flag


def _rebalanced_curve(df: pd.DataFrame, uni: list[str], every: int,
                      cost_bps: float, keep_fn=None) -> tuple[pd.Series, float]:
    dts = sorted(df["date"].unique())
    piv = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    piv = piv.reindex(dts).reindex(columns=[s for s in uni if s in piv.columns])
    raw = piv.copy()
    piv = piv.ffill()
    turnp = df.pivot_table(index="date", columns="symbol", values="turn", aggfunc="last").reindex(dts)
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
            sel = keep_fn(sel, t, raw)
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
    mean_size = float(np.mean(sizes)) if sizes else 0.0
    return pd.Series(vals, index=dts), mean_size


def main() -> int:
    _force_utf8()
    idx = pd.read_parquet(ROOT / "replay_data" / "index_series.parquet")
    idx["date"] = pd.to_datetime(idx["date"])
    fund = pd.read_parquet(ROOT / "replay_data" / "fundamentals.parquet")

    report = {"date": datetime.now().isoformat(timespec="seconds"), "K": K,
              "REBAL": REBAL, "COST_BPS": COST_BPS, "CRASH_YOYNI": CRASH_YOYNI,
              "windows": [], "core_win_counts": {}, "verdict": {}}

    print("=" * 120)
    print(f"精细排雷 A/B: 冷落 bottom-{K} 每月再平衡 vs +精细排雷(连续亏损/利润暴跌<{CRASH_YOYNI}/营收收缩)")
    print("=" * 120)

    for label, cfg in WINDOWS.items():
        df = pd.read_parquet(ROOT / cfg["win"])
        df["date"] = pd.to_datetime(df["date"])
        if df.empty:
            print(f"\n### {label}: 空数据, 跳过")
            continue
        T, T_end = df.date.min(), df.date.max()
        idx_win = idx[(idx["date"] >= T) & (idx["date"] <= T_end)]
        sh = idx_win[idx_win.symbol == "sh.000001"].set_index("date")["close"]
        m_sh = _metrics(sh)

        uni = _liquid_universe(df)
        d0 = df[df.date == T].set_index("symbol")
        uni = [s for s in uni if s in d0.index]

        contloss, crash, shrink, flag = _asof_refined(fund, T)
        n_flagged = len(contloss | crash | shrink)

        def base_fn(sel, t, raw): return sel
        def contloss_fn(sel, t, raw): return [s for s in sel if s not in contloss]
        def crash_fn(sel, t, raw): return [s for s in sel if s not in crash]
        def shrink_fn(sel, t, raw): return [s for s in sel if s not in shrink]
        def all_fn(sel, t, raw): return [s for s in sel if s not in (contloss | crash | shrink)]

        arms = {
            "基线": (base_fn, "base"),
            "+连续亏损": (contloss_fn, "contloss"),
            "+利润暴跌": (crash_fn, "crash"),
            "+营收收缩": (shrink_fn, "shrink"),
            "+合并排雷": (all_fn, "all"),
        }
        row = {"window": label, "core": cfg["core"], "index": m_sh, "asof": flag,
               "n_flagged": n_flagged,
               "contloss": len(contloss), "crash": len(crash), "shrink": len(shrink),
               "arms": {}}
        print(f"\n### {label}  ({T.date()} -> {T_end.date()}, universe={len(uni)}, {flag})")
        print(f"  上证综指:      ret={m_sh['return_pct']:>7.2f}%  dd={m_sh['max_dd_pct']:>7.2f}%  sh={m_sh['sharpe']:>5.2f}")
        print(f"  精细排雷命中: 连续亏损{len(contloss)} 利润暴跌{len(crash)} 营收收缩{len(shrink)} 合计{n_flagged}")
        for aname, (fn, _tag) in arms.items():
            eq, sz = _rebalanced_curve(df, uni, REBAL, COST_BPS, fn)
            m = _metrics(eq)
            dret = m["return_pct"] - m_sh["return_pct"]
            row["arms"][aname] = {**m, "dret_pp": round(dret, 2), "basket_size": round(sz, 1)}
            print(f"  {aname:>8}: ret={m['return_pct']:>7.2f}%  dd={m['max_dd_pct']:>7.2f}%  "
                  f"sh={m['sharpe']:>5.2f}  篮={sz:>4.1f}  Δret={dret:>+7.2f}pp")
        report["windows"].append(row)

    core = [w for w in report["windows"] if w["core"]]
    n = len(core)

    def _not_worse(arm, mode="return_pct", tol=2.0):
        c = 0
        for w in core:
            b = w["arms"]["基线"][mode]
            a = w["arms"][arm][mode]
            if np.isnan(b) or np.isnan(a):
                continue
            if mode == "max_dd_pct":
                if a <= b + 0.5:
                    c += 1
            else:
                if a >= b - tol:
                    c += 1
        return c

    def _beats_idx(arm):
        return sum(1 for w in core if (not np.isnan(w["arms"][arm]["return_pct"])
                                       and w["arms"][arm]["return_pct"] > w["index"]["return_pct"]))

    for arm in ["基线", "+连续亏损", "+利润暴跌", "+营收收缩", "+合并排雷"]:
        report["core_win_counts"][arm] = {
            "dd_not_worse": f"{_not_worse(arm, 'max_dd_pct')}/{n}",
            "ret_not_worse": f"{_not_worse(arm, 'return_pct')}/{n}",
            "beats_index_return": f"{_beats_idx(arm)}/{n}",
        }

    dd_ok = _not_worse("+合并排雷", "max_dd_pct")
    ret_ok = _not_worse("+合并排雷", "return_pct")
    report["verdict"] = {
        "all_dd_not_worse": f"{dd_ok}/{n}",
        "all_ret_not_worse": f"{ret_ok}/{n}",
        "all_beats_index": f"{_beats_idx('+合并排雷')}/{n}",
        "baseline_beats_index": f"{_beats_idx('基线')}/{n}",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 120)
    print("核心 5 非重叠窗口判据 (X/5):")
    for arm, cnt in report["core_win_counts"].items():
        print(f"  {arm:>8}: dd不劣 {cnt['dd_not_worse']}  ret不劣 {cnt['ret_not_worse']}  "
              f"跑赢指数 {cnt['beats_index_return']}")
    print("合并排雷结论:", report["verdict"])
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
