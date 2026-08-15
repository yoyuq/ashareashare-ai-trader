"""改进系统候选组合 A/B (预注册) — 低换手(冷落)小盘 tilt + 软择时 overlay vs 指数。

背景: 全周期 A/B (run_fullcycle_ab.py) 已证:
  - 等权小盘溢价 4/5 跑赢上证, 但回撤大 (2018 -37% / 2025-26 -31%)。
  - 低换手(冷落) tilt 4/5 跑赢上证且 Sharpe 4/5 更优 — 最稳健的"跑赢指数"防御 tilt。
  - 全有/全无 MA120 择时降回撤 5/5 但砍光牛市收益 (太粗暴)。

本脚本验证改进系统: 低换手 bottom-K 等权 + **软择时** (市场跌破 MA120 → 减至 50% 仓位, 不空仓),
目标 = 跑赢上证综指(return 且 Sharpe), 且回撤不劣于等权。

预注册 (不挑窗/不调参追赢):
  - 组合 A = 低换手 bottom-100 等权 (无择时)
  - 组合 B = 低换手 bottom-100 等权 + 软择时 (MA120, 破位减半仓)
  - 组合 C = 低换手 bottom-100 等权 + 硬择时 (MA120, 破位空仓) [参考]
  - 判据: A 或 B 在 return 且 Sharpe 均跑赢上证综指 >= 4/5 窗口 → 满足; 否则不满足。

用法: python scripts/run_improved_portfolio_ab.py
输出: reports/improved_portfolio_ab_result.json + .md
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
from analysis.attribution import (  # noqa: E402
    attribute_equity_curves,
    two_factor_attribute_equity_curves,
)
OUT = ROOT / "reports" / "improved_portfolio_ab_result.json"
BOTTOM_K = 100
MA_N = 120
SOFT_FLOOR = 0.5

WINDOWS = {
    "2018熊":     {"win": "replay_data/daily_2018-01-01_2018-12-31.parquet"},
    "2019牛":     {"win": "replay_data/daily_2019-01-01_2019-12-31.parquet"},
    "2020牛转崩": {"win": "replay_data/daily_2020-06-01_2021-02-28.parquet"},
    "2024震荡":   {"win": "replay_data/daily_2024-01-01_2024-12-31.parquet"},
    "2025-26现期": {"win": "replay_data/daily_2025-10-08_2026-07-31.parquet"},
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


def _liquid_universe(df: pd.DataFrame, top_n: int = 800) -> list[str]:
    df = df[df.symbol != "sh.000001"].copy()
    df["isST"] = df["isST"].astype(str)
    inv = df[(df.isST == "0") & (df.is_trade == 1) & (df.peTTM > 0) & (df.pbMRQ > 0)]
    med = inv.groupby("symbol")["amount"].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()


def _curve(df: pd.DataFrame, symbols: list[str]) -> pd.Series:
    sub = df[df.symbol.isin(symbols)][["date", "symbol", "close"]].dropna(subset=["close"])
    piv = sub.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    piv = piv.reindex(columns=[s for s in symbols if s in piv.columns])
    norm = piv / piv.iloc[0]
    norm = norm.ffill().dropna(axis=1, how="all")
    return norm.mean(axis=1)


def _timing_exposure(eq: pd.Series, n: int, floor: float) -> pd.Series:
    """软择时暴露: 市场 > MA(n) → 1.0, 否则 floor (floor=0 即硬择时空仓)。信号 t-1 决定 t 仓位。"""
    ma = eq.rolling(n).mean()
    sig = (eq > ma).astype(float).shift(1).fillna(1.0)
    return pd.Series(np.where(sig == 1.0, 1.0, floor), index=eq.index)


def _apply_timing(portfolio: pd.Series, exposure: pd.Series) -> pd.Series:
    ret = portfolio.pct_change().fillna(0.0)
    ex = exposure.reindex(ret.index).ffill().fillna(1.0)
    return (1.0 + ret * ex).cumprod()


def main() -> int:
    _force_utf8()
    idx = pd.read_parquet(ROOT / "replay_data" / "index_series.parquet")
    idx["date"] = pd.to_datetime(idx["date"])

    report = {"date": datetime.now().isoformat(timespec="seconds"),
              "BOTTOM_K": BOTTOM_K, "MA_N": MA_N, "SOFT_FLOOR": SOFT_FLOOR,
              "windows": [], "verdict": {}}

    print("=" * 100)
    print(f"改进系统候选 A/B: 低换手 bottom-{BOTTOM_K} + 软择时(MA{MA_N}, 破位减半) vs 上证综指")
    print("=" * 100)

    for label, cfg in WINDOWS.items():
        df = pd.read_parquet(ROOT / cfg["win"])
        df["date"] = pd.to_datetime(df["date"])
        T, T_end = df.date.min(), df.date.max()

        idx_win = idx[(idx["date"] >= T) & (idx["date"] <= T_end)]
        sh = idx_win[idx_win.symbol == "sh.000001"].set_index("date")["close"]

        uni = _liquid_universe(df)
        d0 = df[df.date == T].set_index("symbol")
        uni = [s for s in uni if s in d0.index]

        # 低换手 tilt: bottom-K turn (窗口首日)
        turn = d0["turn"].reindex(uni).dropna()
        lt_sel = turn.nsmallest(BOTTOM_K).index.tolist()
        lt = _curve(df, lt_sel)

        # 市场信号 = 等权聚合
        eq = _curve(df, uni)

        shm = _metrics(sh)
        m_lt = _metrics(lt)
        # A: 无择时; B: 软择时; C: 硬择时
        eq_B = _apply_timing(lt, _timing_exposure(eq, MA_N, SOFT_FLOOR))
        eq_C = _apply_timing(lt, _timing_exposure(eq, MA_N, 0.0))
        m_B = _metrics(eq_B)
        m_C = _metrics(eq_C)
        # 归因 vs 上证: 多少 beta (市场/小盘 tilt) / 多少 alpha (选股+择时技能)
        attr_A = attribute_equity_curves(lt, sh)
        attr_B = attribute_equity_curves(eq_B, sh)
        attr_C = attribute_equity_curves(eq_C, sh)
        # 双因子归因 (市场 + 小盘/等权): 把「冷落 tilt 跑赢上证」拆成
        # 市场 beta + 小盘 beta(相对全 universe 等权) + 真实选股 alpha。回答「跑赢是不是只是小盘 beta」。
        attr2f_A = two_factor_attribute_equity_curves(lt, sh, eq)

        print(f"\n### {label}  (universe={len(uni)}, {T.date()} -> {T_end.date()})")
        print(f"  上证综指:    ret={shm['return_pct']:>7.2f}%  dd={shm['max_dd_pct']:>7.2f}%  sh={shm['sharpe']:>5.2f}")
        print(f"  A 低换手:    ret={m_lt['return_pct']:>7.2f}%  dd={m_lt['max_dd_pct']:>7.2f}%  sh={m_lt['sharpe']:>5.2f}  Δret={m_lt['return_pct']-shm['return_pct']:>+7.2f}pp")
        print(f"      归因A: β={attr_A['beta']:.2f}  beta贡献={attr_A['beta_contribution_pct']:+.1f}%  alpha贡献={attr_A['alpha_contribution_pct']:+.1f}%  R²={attr_A['r_squared']:.2f}")
        print(f"      双因子A: β_mkt={attr2f_A['beta_mkt']:.2f}  β_smb={attr2f_A['beta_smb']:.2f}  市场贡献={attr2f_A['market_contribution_pct']:+.1f}%  小盘贡献={attr2f_A['size_contribution_pct']:+.1f}%  真α贡献={attr2f_A['alpha_contribution_pct']:+.1f}%  R²={attr2f_A['r_squared']:.2f}")
        print(f"  B 低换手+软择: ret={m_B['return_pct']:>7.2f}%  dd={m_B['max_dd_pct']:>7.2f}%  sh={m_B['sharpe']:>5.2f}  Δret={m_B['return_pct']-shm['return_pct']:>+7.2f}pp")
        print(f"      归因B: β={attr_B['beta']:.2f}  beta贡献={attr_B['beta_contribution_pct']:+.1f}%  alpha贡献={attr_B['alpha_contribution_pct']:+.1f}%  R²={attr_B['r_squared']:.2f}")
        print(f"  C 低换手+硬择: ret={m_C['return_pct']:>7.2f}%  dd={m_C['max_dd_pct']:>7.2f}%  sh={m_C['sharpe']:>5.2f}  Δret={m_C['return_pct']-shm['return_pct']:>+7.2f}pp")
        print(f"      归因C: β={attr_C['beta']:.2f}  beta贡献={attr_C['beta_contribution_pct']:+.1f}%  alpha贡献={attr_C['alpha_contribution_pct']:+.1f}%  R²={attr_C['r_squared']:.2f}")

        report["windows"].append({"window": label, "universe": len(uni), "index": shm,
                                  "A_lowturn": {**m_lt, "attribution": attr_A, "attribution_2f": attr2f_A},
                                  "B_lowturn_soft": {**m_B, "attribution": attr_B},
                                  "C_lowturn_hard": {**m_C, "attribution": attr_C}})

    ws = report["windows"]
    n = len(ws)

    def _count(key, mode):
        return sum(1 for w in ws if (not np.isnan(w[key][mode]) and w[key][mode] > w["index"][mode]))

    keys = {"A": "A_lowturn", "B": "B_lowturn_soft", "C": "C_lowturn_hard"}
    report["win_counts"] = {}
    for k, key in keys.items():
        report["win_counts"][f"{k}_ret_beats_idx"] = f"{_count(key, 'return_pct')}/{n}"
        report["win_counts"][f"{k}_sharpe_beats_idx"] = f"{_count(key, 'sharpe')}/{n}"
        r = _count(key, "return_pct")
        s = _count(key, "sharpe")
        report["verdict"][k] = "满足" if (r >= 4 and s >= 4) else "未满足"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    for k, v in report["win_counts"].items():
        print(f"  {k}: {v}")
    print("结论:")
    for k, v in report["verdict"].items():
        print(f"  {k}: {v}")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
