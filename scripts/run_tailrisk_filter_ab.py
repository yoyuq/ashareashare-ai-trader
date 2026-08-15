"""尾险排雷过滤器 A/B (预注册) — 纯规则冷落篮子 vs +排雷, 全部历史窗口。

背景: 冷落 beta (低换手 bottom-100) 已证 5/5 窗口跑赢上证 (见 run_cold_tilt_rebalance.py)。
本脚本回答「全靠规则能行吗」的延伸: 冷落篮子会不会拿一堆「快死的票」(退市/亏损) 而承担看不见的尾险?
测试在冷落篮子上叠加「排雷过滤器」, 看是否降尾险 (maxDD) 而不伤收益。

排雷过滤器 (预注册, 首原则, 非调参追赢):
  - F1 面值退市排雷: 剔除 close < 3.0 元 (A股面值退市/仙股风险区; 阈值沿用 PreScreener 既有 hard filter price>=3.0)。
  - F2 基本面排雷: 剔除 roeAvg <= 0 (亏损; 用窗口起点前最新可得基本面快照, as-of 不偷看未来)。
  - 「只删不调权」: 剔除后剩余等权持有, 不回填第 101 冷票, 不因变少而补仓。

方法: 每月(20交易日)按换手率取 bottom-100 等权再平衡, 扣费 31bp/次 (full-turnover 上界), 与 run_cold_tilt_rebalance.py 同口径。

预注册判据 (核心 5 个非重叠窗口):
  - 主: 排雷后 maxDD 不劣于基线 (<= 基线+0.5pp) 且 return 不劣于基线 (>= 基线-2pp) >= 4/5 窗口 → 「排雷不伤收益」。
  - 副: 排雷后 return 跑赢上证 的窗口数 不劣于 基线 (不破坏跑赢指数能力)。
  - 若主/副都不成立且 return/dd 与基线几乎无差 → 「排雷无效果」= 冷落篮子本就干净, 无需叠加 (同样是有价值的诚实结论)。

用法: python scripts/run_tailrisk_filter_ab.py
输出: reports/tailrisk_filter_ab_result.json + .md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "tailrisk_filter_ab_result.json"
K = 100
UNIVERSE_N = 800
REBAL = 20  # 每月再平衡
COST_BPS = 31.0
PRICE_FLOOR = 3.0  # 面值退市排雷阈值

# 全部可用历史窗口 (含重叠/lookback, 按「把大多数历史回测做进去」)。core=True 为 5 个非重叠核心窗。
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


def _asof_roe(fund: pd.DataFrame, window_start: pd.Timestamp) -> tuple[dict, str]:
    """取窗口起点前最新可得基本面快照的 roeAvg (as-of 不偷看未来)。返回 (symbol->roeAvg, 所用 statDate 标注)。"""
    fund = fund.copy()
    fund["statDate"] = pd.to_datetime(fund["statDate"])
    snaps = sorted(fund["statDate"].unique())
    before = [s for s in snaps if s <= window_start]
    if before:
        chosen = max(before)
        flag = ""
    else:
        chosen = min(snaps)
        flag = f" (无 as-of 快照, 回退最早 {chosen.date()}, 近似/轻微偷看)"
    sub = fund[fund["statDate"] == chosen]
    roe = sub.groupby("symbol")["roeAvg"].last().to_dict()
    return roe, f"{chosen.date()}{flag}"


def _rebalanced_curve(df: pd.DataFrame, uni: list[str], every: int,
                      cost_bps: float, keep_fn=None) -> tuple[pd.Series, float]:
    """每月按换手率取 bottom-K 等权再平衡; keep_fn(sel, t, close_at_t) 可选排雷过滤。"""
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
              "REBAL": REBAL, "COST_BPS": COST_BPS, "PRICE_FLOOR": PRICE_FLOOR,
              "windows": [], "core_win_counts": {}, "verdict": {}}

    print("=" * 120)
    print(f"尾险排雷过滤器 A/B: 冷落 bottom-{K} 每月再平衡 扣费{COST_BPS}bp vs +排雷(price<{PRICE_FLOOR} 或 roeAvg<=0)")
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

        roe, roe_flag = _asof_roe(fund, T)

        def base_fn(sel, t, raw): return sel
        def price_fn(sel, t, raw):
            px = raw.loc[t].reindex(sel)
            return [s for s in sel if pd.notna(px[s]) and px[s] >= PRICE_FLOOR]
        def fund_fn(sel, t, raw):
            return [s for s in sel if (s not in roe) or pd.isna(roe.get(s)) or roe[s] > 0]
        def both_fn(sel, t, raw):
            return fund_fn(price_fn(sel, t, raw), t, raw)

        arms = {
            "基线": (base_fn, None),
            "+面值排雷": (price_fn, "price"),
            "+基本面排雷": (fund_fn, "roe"),
            "+双重排雷": (both_fn, "both"),
        }
        row = {"window": label, "core": cfg["core"], "index": m_sh, "roe_snapshot": roe_flag,
               "arms": {}}
        print(f"\n### {label}  ({T.date()} -> {T_end.date()}, universe={len(uni)}, "
              f"基本面 as-of={roe_flag})")
        print(f"  上证综指:      ret={m_sh['return_pct']:>7.2f}%  dd={m_sh['max_dd_pct']:>7.2f}%  sh={m_sh['sharpe']:>5.2f}")
        for aname, (fn, _tag) in arms.items():
            eq, sz = _rebalanced_curve(df, uni, REBAL, COST_BPS, fn)
            m = _metrics(eq)
            dret = m["return_pct"] - m_sh["return_pct"]
            row["arms"][aname] = {**m, "dret_pp": round(dret, 2), "basket_size": round(sz, 1)}
            print(f"  {aname:>8}: ret={m['return_pct']:>7.2f}%  dd={m['max_dd_pct']:>7.2f}%  "
                  f"sh={m['sharpe']:>5.2f}  篮={sz:>4.1f}  Δret={dret:>+7.2f}pp")
        report["windows"].append(row)

    # 判据聚合 (仅核心 5 非重叠窗口)
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

    for arm in ["基线", "+面值排雷", "+基本面排雷", "+双重排雷"]:
        report["core_win_counts"][arm] = {
            "dd_not_worse": f"{_not_worse(arm, 'max_dd_pct')}/{n}",
            "ret_not_worse": f"{_not_worse(arm, 'return_pct')}/{n}",
            "beats_index_return": f"{_beats_idx(arm)}/{n}",
        }

    # 主判据: 排雷后 (双重) dd不劣 且 ret不劣 >= 4/5
    dd_ok = _not_worse("+双重排雷", "max_dd_pct")
    ret_ok = _not_worse("+双重排雷", "return_pct")
    beats = _beats_idx("+双重排雷")
    base_beats = _beats_idx("基线")
    if dd_ok >= 4 and ret_ok >= 4:
        v = "排雷不伤收益且未降险(持平) — 冷落篮子本就干净, 排雷非必需" if dd_ok >= 4 and ret_ok >= 4 else "排雷有效"
    else:
        v = "排雷伤收益或未达标"
    report["verdict"] = {
        "double_dd_not_worse": f"{dd_ok}/{n}",
        "double_ret_not_worse": f"{ret_ok}/{n}",
        "double_beats_index": f"{beats}/{n}",
        "baseline_beats_index": f"{base_beats}/{n}",
        "conclusion": v,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 120)
    print("核心 5 非重叠窗口判据 (X/5):")
    for arm, cnt in report["core_win_counts"].items():
        print(f"  {arm:>8}: dd不劣 {cnt['dd_not_worse']}  ret不劣 {cnt['ret_not_worse']}  "
              f"跑赢指数 {cnt['beats_index_return']}")
    print("结论:", report["verdict"])
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
