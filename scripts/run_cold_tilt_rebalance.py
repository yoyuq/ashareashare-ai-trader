"""冷落 beta 的「实际操作」测试 (预注册) — 定期再平衡 + 交易成本 vs 上证综指。

背景: run_improved_portfolio_ab.py 证低换手 bottom-100 等权「一次性买入持有」5/5 窗口跑赢上证。
但那是一次性持有 (每窗口只再平衡 1 次), 未回答「实际怎么操作」:
  1. 定期再平衡 (每月/每季/每周) 会不会破坏优势?
  2. 交易成本 (每次换仓的佣金/印花税/滑点) 会不会吃掉冷落溢价?

本脚本用真实 Baostock replay parquet, 逐日模拟「每 N 交易日按换手率取 bottom-K 等权再平衡」,
扣费 (full-turnover 上界), 对比上证综指。全程零模拟, 缺数据报错不兜底。

方法 (close-to-close, 与原始 A/B 同口径):
  - universe = top-800 流动性可投资 (非ST/可交易/pe>0/pb>0)。
  - 再平衡日 t: 用当日 turn 取 bottom-K, 等权; 持有期 t+1..t+N 的日收益 = 篮子内等权日收益。
  - 停牌股 ffill (价格不变, 贡献 0 收益), 不失真。
  - 成本模型 (记忆 [[transaction-costs-unified]]): 佣金万3(双边)+印花税0.05%(卖)+滑点10bp(双边)
    → full-turnover 往返 ≈ 31bp; 作为上界按每次再平衡全额扣 (真实换仓 <100%, 实际更低)。

预注册判据 (11 窗口 = 5 原 + 6 新增, 覆盖 2015-2026 全 regime):
  - 主: 每月(20交易日)再平衡, 扣费后 return 跑赢上证 >= 4/5 窗口 (80% 门槛, 11 窗口 → >=9) → 「可操作」。
  - 副: 每季(60日)再平衡同样达标 → 更低换手率更稳。
  - regime-gap 关键观察: 2017漂亮50(小盘逆向年) 与 2015牛转股灾(流动性尾部) 的 Δreturn 符号/幅度 —
    这两窗是回答「beta 是否无条件有用」的核心, 若明显跑输则如实标注 beta 非无条件。

用法: python scripts/run_cold_tilt_rebalance.py
输出: reports/cold_tilt_rebalance_result.json + .md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # 使 analysis 包可导入
from analysis.attribution import attribute_equity_curves  # noqa: E402
from analysis.benchmark import (  # noqa: E402
    dividend_contribution_pct,
    matched_universe_curve,
    matched_universe_total_return_curve,
    total_return_daily_factor,
)
OUT = ROOT / "reports" / "cold_tilt_rebalance_result.json"
K = 100
UNIVERSE_N = 800
COST_BPS = 31.0  # full-turnover 往返上界 (佣金万3双边+印花0.05%卖+滑点10bp双边)
CADENCES = [5, 10, 20, 40, 60]  # 再平衡周期 (交易日): 周/双周/月/双月/季

WINDOWS = {
    "2015牛转股灾": "replay_data/daily_2015-01-01_2015-12-31.parquet",
    "2016熔断震荡": "replay_data/daily_2016-01-01_2016-12-31.parquet",
    "2017漂亮50":   "replay_data/daily_2017-01-01_2017-12-31.parquet",
    "2018熊":       "replay_data/daily_2018-01-01_2018-12-31.parquet",
    "2019牛":       "replay_data/daily_2019-01-01_2019-12-31.parquet",
    "2020牛转崩":   "replay_data/daily_2020-06-01_2021-02-28.parquet",
    "2021白马转小盘": "replay_data/daily_2021-01-01_2021-12-31.parquet",
    "2022熊":       "replay_data/daily_2022-01-01_2022-12-31.parquet",
    "2023震荡":     "replay_data/daily_2023-01-01_2023-12-31.parquet",
    "2024震荡":     "replay_data/daily_2024-01-01_2024-12-31.parquet",
    "2025-26现期":  "replay_data/daily_2025-10-08_2026-07-31.parquet",
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


def _rebalanced_curve(df: pd.DataFrame, uni: list[str], every: int | None,
                      cost_bps: float, factor: pd.DataFrame | None = None) -> pd.Series:
    """每 `every` 交易日按换手率取 bottom-K 等权再平衡; every=None → 单次买入持有。

    factor: 每日全收益因子矩阵 (benchmark.total_return_daily_factor) — 传入时策略净值含分红
    (全收益口径), 缺省 None 时用不复权价格收益 (与上证价格指数同口径)。
    """
    dts = sorted(df["date"].unique())
    piv = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    piv = piv.reindex(dts).reindex(columns=[s for s in uni if s in piv.columns]).ffill()
    turnp = df.pivot_table(index="date", columns="symbol", values="turn", aggfunc="last").reindex(dts)
    if factor is None:
        R = piv.pct_change()  # R.iloc[i] = 进入 dts[i] 的日收益 (t-1 close → t close)
    else:
        # 全收益日收益 = 因子 - 1; 与价格路径同 date/columns 对齐
        R = (factor.reindex(index=dts, columns=piv.columns) - 1.0)

    reb = [0] if every is None else list(range(0, len(dts), every))
    reb_set = set(reb)
    port_ret = pd.Series(0.0, index=dts)
    for i, ri in enumerate(reb):
        nxt = reb[i + 1] if i + 1 < len(reb) else len(dts)
        turn_t = turnp.loc[dts[ri]].reindex(piv.columns).dropna()
        if turn_t.empty:
            continue
        sel = turn_t.nsmallest(K).index
        # 持有期: 进入 dts[ri+1 .. nxt] 的日收益 (再平衡日 close 建仓, 次日开始计收益);
        # 末次再平衡无「下一个再平衡日」, 持有到窗口最后一天 (index len(dts)-1)。
        end = min(nxt, len(dts) - 1)
        for j in range(ri + 1, end + 1):
            port_ret.iloc[j] = float(R.iloc[j].reindex(sel).mean())

    # 复利 + 每次再平衡扣费 (含首次建仓; 换仓费在再平衡日 close 支付, 影响次日起)
    cost = cost_bps / 10000.0
    eq = 1.0
    vals = []
    for j in range(len(dts)):
        if j > 0:
            eq *= (1.0 + float(port_ret.iloc[j]))
        if j in reb_set:
            eq *= (1.0 - cost)
        vals.append(eq)
    return pd.Series(vals, index=dts)


def main() -> int:
    _force_utf8()
    idx = pd.read_parquet(ROOT / "replay_data" / "index_series.parquet")
    idx["date"] = pd.to_datetime(idx["date"])

    # 分红数据 (全收益基准): 缺失则退回价格收益并显式标注 (不伪造)
    div_path = ROOT / "replay_data" / "dividends.parquet"
    dividends = pd.read_parquet(div_path) if div_path.exists() else None
    if dividends is not None:
        dividends["dividOperateDate"] = pd.to_datetime(dividends["dividOperateDate"])
        print(f"[基准纪律] 已加载真实分红 {dividends['code'].nunique()} 只 → 全收益基准可用")
    else:
        print("[基准纪律] 无分红数据 (replay_data/dividends.parquet 缺失) → 匹配universe基准用价格收益 (非全收益)")

    report = {"date": datetime.now().isoformat(timespec="seconds"), "K": K,
              "UNIVERSE_N": UNIVERSE_N, "COST_BPS": COST_BPS, "cadences": CADENCES,
              "windows": [], "win_counts": {}, "verdict": {}}

    print("=" * 110)
    print(f"冷落 beta 实际操作测试: bottom-{K} 等权, 定期再平衡, 扣费 {COST_BPS}bp/次 vs 上证综指")
    print("=" * 110)

    for label, fp in WINDOWS.items():
        df = pd.read_parquet(ROOT / fp)
        df["date"] = pd.to_datetime(df["date"])
        T, T_end = df.date.min(), df.date.max()
        idx_win = idx[(idx["date"] >= T) & (idx["date"] <= T_end)]
        sh = idx_win[idx_win.symbol == "sh.000001"].set_index("date")["close"]
        m_sh = _metrics(sh)

        uni = _liquid_universe(df)
        d0 = df[df.date == T].set_index("symbol")
        uni = [s for s in uni if s in d0.index]

        print(f"\n### {label}  (universe={len(uni)}, {T.date()} -> {T_end.date()})")
        print(f"  上证综指:      ret={m_sh['return_pct']:>7.2f}%  dd={m_sh['max_dd_pct']:>7.2f}%  sh={m_sh['sharpe']:>5.2f}")

        row = {"window": label, "index": m_sh, "cadences": {}}

        # 基准纪律 (#107): 匹配 universe (策略可投池子的等权) 价格 + 全收益 (分红加回)
        eq_price = matched_universe_curve(df, uni)
        eq_tr = matched_universe_total_return_curve(df, uni, dividends)
        m_eq = _metrics(eq_price)
        m_eqtr = _metrics(eq_tr)
        div_contrib = dividend_contribution_pct(eq_price, eq_tr)
        row["matched_universe"] = {"price": m_eq, "total_return": m_eqtr,
                                   "dividend_contribution_pp": round(div_contrib, 3) if np.isfinite(div_contrib) else None}
        print(f"  匹配universe(等权): price ret={m_eq['return_pct']:>7.2f}%  全收益ret={m_eqtr['return_pct']:>7.2f}%  "
              f"红利贡献={div_contrib:+.2f}pp")
        # 全收益日因子矩阵 (分红加回), 供策略篮子与基准同口径比较
        factor = total_return_daily_factor(df, uni, dividends)
        for every in CADENCES + [None]:
            tag = "持有" if every is None else f"{every}日"
            eq = _rebalanced_curve(df, uni, every, COST_BPS)                       # 价格收益 (vs 上证价格指数同口径)
            eq_tr_curve = _rebalanced_curve(df, uni, every, COST_BPS, factor=factor)  # 全收益 (分红加回)
            m = _metrics(eq)
            m_tr = _metrics(eq_tr_curve)
            dret = m["return_pct"] - m_sh["return_pct"]
            attr = attribute_equity_curves(eq, sh)  # 归因 vs 上证: 多少 beta / 多少 alpha
            row["cadences"][tag] = {**m, "total_return_pct": m_tr["return_pct"],
                                    "dret_pp": round(dret, 2), "attribution": attr}
            print(f"  {tag:>6}再平衡:   ret={m['return_pct']:>7.2f}%  dd={m['max_dd_pct']:>7.2f}%  "
                  f"sh={m['sharpe']:>5.2f}  Δret={dret:>+7.2f}pp  全收益ret={m_tr['return_pct']:>7.2f}%")
            print(f"           归因vs上证: β={attr['beta']:.2f}  年化α={attr['alpha_annualized_pct']:+.1f}%  "
                  f"beta贡献={attr['beta_contribution_pct']:+.1f}%  alpha贡献={attr['alpha_contribution_pct']:+.1f}%  R²={attr['r_squared']:.2f}")
        report["windows"].append(row)

    ws = report["windows"]
    n = len(ws)

    def _cnt(cad, mode="return_pct"):
        return sum(1 for w in ws if (not np.isnan(w["cadences"][cad][mode])
                                     and w["cadences"][cad][mode] > w["index"][mode]))

    # 基准纪律 (#107): 匹配 universe 跑赢计数。
    #   price 口径: 策略价格收益 vs 匹配universe价格收益 (选股 alpha, 与上证价格指数同口径可比)
    #   total 口径: 策略全收益 vs 匹配universe全收益 (诚实回答「跑赢市场总回报」, 分红加回)
    def _cnt_vs_universe(cad, mode="price"):
        uw = "price" if mode == "price" else "total_return"
        cw = "return_pct" if mode == "price" else "total_return_pct"
        return sum(1 for w in ws if (not np.isnan(w["cadences"][cad][cw])
                                     and not np.isnan(w["matched_universe"][uw]["return_pct"])
                                     and w["cadences"][cad][cw] > w["matched_universe"][uw]["return_pct"]))

    for cad in [f"{c}日" for c in CADENCES] + ["持有"]:
        report["win_counts"][cad] = {
            "return": f"{_cnt(cad, 'return_pct')}/{n}",
            "sharpe": f"{_cnt(cad, 'sharpe')}/{n}",
            "vs_matched_universe_price": f"{_cnt_vs_universe(cad, 'price')}/{n}",
            "vs_matched_universe_total": f"{_cnt_vs_universe(cad, 'total')}/{n}",
        }

    # 主判据: 每月(20日)扣费后 return 跑赢 >= 4/5 窗口 (80% 门槛, 分数口径通用于 11 窗口).
    # 预注册门槛 = ceil(0.8 * n), 11 窗口 → >=9 窗口.
    thr = int(np.ceil(0.8 * n))
    report["verdict"] = {
        "threshold": f">={thr}/{n} 窗口 (80%)",
        "monthly_20d_operable": "可操作" if _cnt("20日", "return_pct") >= thr else "未满足",
        "quarterly_60d_operable": "可操作" if _cnt("60日", "return_pct") >= thr else "未满足",
        "hold_operable": "可操作" if _cnt("持有", "return_pct") >= thr else "未满足",
        # 基准纪律: 冷落tilt 是否真跑赢匹配universe (不是仅跑赢价格指数)
        #   price = 选股 alpha (价格口径); total = 跑赢市场总回报 (分红加回)
        "hold_vs_matched_universe_price": f"{_cnt_vs_universe('持有', 'price')}/{n}",
        "monthly_vs_matched_universe_price": f"{_cnt_vs_universe('20日', 'price')}/{n}",
        "hold_vs_matched_universe_total": f"{_cnt_vs_universe('持有', 'total')}/{n}",
        "monthly_vs_matched_universe_total": f"{_cnt_vs_universe('20日', 'total')}/{n}",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 110)
    print("扣费后 return 跑赢上证 (X/11 窗口) [vs匹配universe价格/全收益]:")
    for cad, cnt in report["win_counts"].items():
        print(f"  {cad:>6}再平衡: return {cnt['return']}  sharpe {cnt['sharpe']}  "
              f"vs匹配universe价格 {cnt['vs_matched_universe_price']}  全收益 {cnt['vs_matched_universe_total']}")
    print("结论:", report["verdict"])
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
