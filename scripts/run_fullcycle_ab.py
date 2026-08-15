"""全周期 A/B (路线1+2 预注册) — 结构性风险溢价 vs 指数 + 市场择时 overlay。

理论背景 (回应「从理论上来说这个真的做不到吗」):
  - 路线2 结构性风险溢价: 之前所有 A/B 的基线是 top-800 流动性**等权**(已吃掉小盘溢价), 但「跑赢市场」
    的正确定义是跑赢**市值加权指数**(上证综指 sh.000001 / 沪深300 sh.000300)。小市值/价值/低波溢价是
    系统性 beta, 应在全周期(非仅牛市)检验其能否跑赢指数, 以及回撤代价。
  - 路线1 市场择时/仓位: 跑赢牛市的正确杠杆是「牛市满仓/熊市减仓」, 而非选股。测试趋势择时 overlay
    (价格>MA(N) 满仓, 否则空仓) 能否在全周期改善风险调整收益。

数据 (全程零模拟, 真实 Baostock/replay parquet):
  - 指数: replay_data/index_series.parquet (sh.000001/sh.000300, 2018-01~2026-07)
  - 窗口日线: replay_data/daily_*.parquet (2018熊/2019牛/2020牛转崩/2024震荡/2025-26 现期)
  - lookback (低波/低换手需): daily_2018-10-01_2019-12-31, lookback_2020, lookback_2024

预注册判据 (不挑窗/不调参追赢):
  - 路线2: 等权(小盘)溢价跑赢指数 return 的窗口数 >= 3/5 → 小盘溢价成立(regime相关)。
  - 路线1: N=120(半年线) 择时 overlay 改善最大回撤 >= 4/5 窗口 且 改善 Sharpe >= 3/5 窗口 → 择时降险成立。
  - 组合(改进系统候选): 等权小盘 + N=120 择时 overlay 跑赢指数 return >= 3/5 窗口 且 回撤<=指数 >= 4/5 窗口
    → 全周期跑赢市场成立。

用法: python scripts/run_fullcycle_ab.py
输出: reports/fullcycle_ab_result.json + reports/fullcycle_ab.md
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # 使 analysis 包可导入
from analysis.attribution import (  # noqa: E402
    attribute_equity_curves,
    two_factor_attribute_equity_curves,
)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "fullcycle_ab_result.json"
TOP_K = 30
TIMING_NS = [20, 60, 120, 200]
CANONICAL_N = 120

# 窗口: label -> {win(窗口日线), lookback(因子所需, 可无), T(窗口首日)}
WINDOWS = {
    "2018熊":    {"win": "replay_data/daily_2018-01-01_2018-12-31.parquet", "lookback": None},
    "2019牛":    {"win": "replay_data/daily_2019-01-01_2019-12-31.parquet",
                  "lookback": "replay_data/daily_2018-10-01_2019-12-31.parquet"},
    "2020牛转崩": {"win": "replay_data/daily_2020-06-01_2021-02-28.parquet",
                  "lookback": "replay_data/lookback_2020.parquet"},
    "2024震荡":  {"win": "replay_data/daily_2024-01-01_2024-12-31.parquet",
                  "lookback": "replay_data/lookback_2024.parquet"},
    "2025-26现期": {"win": "replay_data/daily_2025-10-08_2026-07-31.parquet", "lookback": None},
}


# ── 指标 ──
def _series_metrics(eq: pd.Series) -> dict:
    """从净值序列算 return%/maxDD%/sharpe (等权/指数通用)。"""
    eq = eq.dropna()
    if len(eq) < 2:
        return {"return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
    dd = float((eq / eq.cummax() - 1.0).min()) * 100.0
    dr = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0
    return {"return_pct": round(ret, 3), "max_dd_pct": round(dd, 3), "sharpe": round(sharpe, 3)}


def _liquid_universe(df: pd.DataFrame, top_n: int = 800) -> list[str]:
    df = df[df.symbol != "sh.000001"].copy()
    df["isST"] = df["isST"].astype(str)
    inv = df[(df.isST == "0") & (df.is_trade == 1) & (df.peTTM > 0) & (df.pbMRQ > 0)]
    med = inv.groupby("symbol")["amount"].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()


def _equalweight_curve(df: pd.DataFrame, symbols: list[str]) -> pd.Series:
    """等权净值曲线 (首日归一)。"""
    sub = df[df.symbol.isin(symbols)][["date", "symbol", "close"]].dropna(subset=["close"])
    piv = sub.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    piv = piv.reindex(columns=[s for s in symbols if s in piv.columns])
    norm = piv / piv.iloc[0]
    norm = norm.ffill().dropna(axis=1, how="all")
    return norm.mean(axis=1)


def _factor_tilt_curve(df: pd.DataFrame, combined: pd.DataFrame, symbols: list[str],
                       factor: str, top_k: int = TOP_K) -> pd.Series:
    """在窗口首日 T 处按因子选 top/bottom-K, 等权买入持有。factor ∈ {low_pe, low_vol, low_turn}。"""
    T = df.date.min()
    d0 = df[df.date == T].set_index("symbol")
    uni = [s for s in symbols if s in d0.index]
    if factor == "low_pe":
        s = d0["peTTM"].reindex(uni).dropna()
        s = s[s > 0]
        sel = s.nsmallest(top_k)
    elif factor == "low_vol":
        close = combined.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
        if T not in close.index:
            return pd.Series(dtype=float)
        pos = close.index.get_loc(T)
        if pos < 40:
            return pd.Series(dtype=float)
        vol = close.iloc[pos - 40:pos + 1].pct_change().std()
        sel = vol.reindex(uni).dropna().nsmallest(top_k)
    elif factor == "low_turn":
        s = d0["turn"].reindex(uni).dropna()
        sel = s.nsmallest(top_k)
    else:
        raise ValueError(factor)
    if sel.empty:
        return pd.Series(dtype=float)
    return _equalweight_curve(df, sel.index.tolist())


def _timing_curve(eq: pd.Series, n: int) -> pd.Series:
    """趋势择时 overlay: 价格>MA(n) 满仓, 否则空仓 (信号 t-1 决定 t 仓位, 无未来函数)。"""
    ma = eq.rolling(n).mean()
    sig = (eq > ma).astype(float).shift(1).fillna(0.0)
    ret = eq.pct_change().fillna(0.0)
    tret = ret * sig
    return (1.0 + tret).cumprod()


# ── 主流程 ──
def _force_utf8():
    import sys
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main() -> int:
    _force_utf8()
    idx = pd.read_parquet(ROOT / "replay_data" / "index_series.parquet")
    idx["date"] = pd.to_datetime(idx["date"])

    report = {"date": datetime.now().isoformat(timespec="seconds"),
              "TOP_K": TOP_K, "TIMING_NS": TIMING_NS, "CANONICAL_N": CANONICAL_N,
              "windows": [], "verdict": {}}

    print("=" * 100)
    print("全周期 A/B: 结构性风险溢价(路线2) + 市场择时(路线1) + 组合 vs 指数")
    print("=" * 100)

    for label, cfg in WINDOWS.items():
        df = pd.read_parquet(ROOT / cfg["win"])
        df["date"] = pd.to_datetime(df["date"])
        win_dates = sorted(df.date.unique())
        T = win_dates[0]
        T_end = win_dates[-1]

        # 指数切片 (上证综指 主基准, 沪深300 次)
        idx_win = idx[(idx["date"] >= T) & (idx["date"] <= T_end)]
        sh = idx_win[idx_win.symbol == "sh.000001"].set_index("date")["close"]
        hs = idx_win[idx_win.symbol == "sh.000300"].set_index("date")["close"]

        uni = _liquid_universe(df)
        d0 = df[df.date == T].set_index("symbol")["close"]
        uni = [s for s in uni if s in d0.index]

        eq = _equalweight_curve(df, uni)
        eqm = _series_metrics(eq)
        shm = _series_metrics(sh) if len(sh) > 1 else {"return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
        hsm = _series_metrics(hs) if len(hs) > 1 else {"return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
        # 归因: 等权小盘跑赢指数 = 多少 beta(小盘/结构性溢价) / 多少 alpha(选股)
        attr_ew_sh = attribute_equity_curves(eq, sh)
        attr_ew_hs = attribute_equity_curves(eq, hs)

        print(f"\n### {label}  (universe={len(uni)}, {T.date()} -> {T_end.date()})")
        print(f"  上证综指: ret={shm['return_pct']:>7.2f}%  dd={shm['max_dd_pct']:>7.2f}%  sh={shm['sharpe']:>5.2f}")
        print(f"  沪深300:  ret={hsm['return_pct']:>7.2f}%  dd={hsm['max_dd_pct']:>7.2f}%  sh={hsm['sharpe']:>5.2f}")
        print(f"  等权小盘: ret={eqm['return_pct']:>7.2f}%  dd={eqm['max_dd_pct']:>7.2f}%  sh={eqm['sharpe']:>5.2f}  Δvs上证={eqm['return_pct']-shm['return_pct']:>+7.2f}pp")
        print(f"      归因vs上证: β={attr_ew_sh['beta']:.2f}  beta贡献={attr_ew_sh['beta_contribution_pct']:+.1f}%  alpha贡献={attr_ew_sh['alpha_contribution_pct']:+.1f}%  R²={attr_ew_sh['r_squared']:.2f}")

        win_rec = {"window": label, "universe": len(uni), "index": shm, "hs300": hsm,
                   "equalweight": eqm,
                   "attribution": {"equalweight_vs_sh": attr_ew_sh, "equalweight_vs_hs": attr_ew_hs},
                   "tilts": {}, "timing": {}, "combined": {}}

        # 路线2: 因子 tilt vs 指数
        combined = df
        if cfg["lookback"] and Path(ROOT / cfg["lookback"]).exists():
            lb = pd.read_parquet(ROOT / cfg["lookback"])
            lb["date"] = pd.to_datetime(lb["date"])
            combined = pd.concat([lb, df], ignore_index=True).drop_duplicates(subset=["date", "symbol"])
        combined = combined.sort_values(["symbol", "date"])

        for fac in ["low_pe", "low_vol", "low_turn"]:
            curve = _factor_tilt_curve(df, combined, uni, fac)
            if curve.empty:
                continue
            m = _series_metrics(curve)
            attr = attribute_equity_curves(curve, sh)  # 因子 tilt 的 beta/alpha 分解
            attr2f = two_factor_attribute_equity_curves(curve, sh, eq)  # 双因子: 拆小盘 beta vs 真 alpha
            win_rec["tilts"][fac] = {**m, "attribution": attr, "attribution_2f": attr2f}
            dsh = m["return_pct"] - shm["return_pct"]
            print(f"  {fac:8s}: ret={m['return_pct']:>7.2f}%  dd={m['max_dd_pct']:>7.2f}%  sh={m['sharpe']:>5.2f}  Δvs上证={dsh:>+7.2f}pp  β={attr['beta']:.2f} α贡献={attr['alpha_contribution_pct']:+.1f}%")
            print(f"            双因子: β_mkt={attr2f['beta_mkt']:.2f}  β_smb={attr2f['beta_smb']:.2f}  市场贡献={attr2f['market_contribution_pct']:+.1f}%  小盘贡献={attr2f['size_contribution_pct']:+.1f}%  真α={attr2f['alpha_contribution_pct']:+.1f}%  R²={attr2f['r_squared']:.2f}")

        # 路线1: 趋势择时 overlay (等权聚合上)
        for n in TIMING_NS:
            tc = _timing_curve(eq, n)
            m = _series_metrics(tc)
            win_rec["timing"][str(n)] = m
            if n == CANONICAL_N:
                print(f"  择时MA{n:>3d}: ret={m['return_pct']:>7.2f}%  dd={m['max_dd_pct']:>7.2f}%  sh={m['sharpe']:>5.2f}  Δdd={m['max_dd_pct']-eqm['max_dd_pct']:>+7.2f}pp")

        # 组合: 等权 + 择时 (改进系统候选)  vs 指数
        tc_canon = _timing_curve(eq, CANONICAL_N)
        comb_m = _series_metrics(tc_canon)
        win_rec["combined"] = comb_m
        print(f"  组合(等权+MA{CANONICAL_N}): ret={comb_m['return_pct']:>7.2f}%  dd={comb_m['max_dd_pct']:>7.2f}%  sh={comb_m['sharpe']:>5.2f}  Δvs上证ret={comb_m['return_pct']-shm['return_pct']:>+7.2f}pp")

        report["windows"].append(win_rec)

    # ── 判据汇总 ──
    ws = report["windows"]
    n = len(ws)

    def _wins(extract):
        return sum(1 for w in ws if extract(w))

    # 路线2: 等权跑赢指数 return
    ew_beats_idx = _wins(lambda w: (not np.isnan(w["equalweight"]["return_pct"])
                                    and w["equalweight"]["return_pct"] > w["index"]["return_pct"]))
    # 路线1: 择时(MA120) 改善回撤 & Sharpe
    t_dd_better = _wins(lambda w: (not np.isnan(w["timing"][str(CANONICAL_N)]["max_dd_pct"])
                                   and w["timing"][str(CANONICAL_N)]["max_dd_pct"] >= w["equalweight"]["max_dd_pct"] - 1e-9))
    t_sh_better = _wins(lambda w: (not np.isnan(w["timing"][str(CANONICAL_N)]["sharpe"])
                                   and w["timing"][str(CANONICAL_N)]["sharpe"] >= w["equalweight"]["sharpe"] - 1e-9))
    # 组合: 跑赢指数 return 且 回撤<=指数
    comb_beats_idx = _wins(lambda w: (not np.isnan(w["combined"]["return_pct"])
                                      and w["combined"]["return_pct"] > w["index"]["return_pct"]))
    comb_dd_ok = _wins(lambda w: (not np.isnan(w["combined"]["max_dd_pct"])
                                  and w["combined"]["max_dd_pct"] >= w["index"]["max_dd_pct"] - 1e-9))

    report["win_counts"] = {
        "equalweight_beats_index": f"{ew_beats_idx}/{n}",
        "timing_dd_better": f"{t_dd_better}/{n}",
        "timing_sharpe_better": f"{t_sh_better}/{n}",
        "combined_beats_index": f"{comb_beats_idx}/{n}",
        "combined_dd_ok": f"{comb_dd_ok}/{n}",
    }
    report["verdict"]["route2_smallcap"] = "成立" if ew_beats_idx >= 3 else "不稳健"
    report["verdict"]["route1_timing"] = "降险成立" if (t_dd_better >= 4 and t_sh_better >= 3) else "不稳健"
    report["verdict"]["combined_beat_market"] = ("成立" if (comb_beats_idx >= 3 and comb_dd_ok >= 4) else "未满足")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("判据汇总:")
    for k, v in report["win_counts"].items():
        print(f"  {k}: {v}")
    print("结论:")
    for k, v in report["verdict"].items():
        print(f"  {k}: {v}")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
