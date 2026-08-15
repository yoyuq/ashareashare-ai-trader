"""方向B 副线实验 (阶段3 #110): 指数绝对动量择时加 beta — 预注册 A/B。

假设 (reports/bull_market_target_preregistration.md §3): A 股普涨牛是 beta 行情, 用指数绝对动量择时
(上证综指 close > MA20 → 满仓市场 beta; close ≤ MA20 → 现金) 在牛市加 beta、熊市退现金, 可把「牛市跑输」
翻成「牛市跟上 + 熊市不崩」。动机 = factor_backtest_timing.md 单窗口 mkt_timing_20 +34.8% vs 持有 -5.0%,
但**未跨窗口泛化验证** —— 本脚本补这个验证。

市场 beta = 匹配 universe (top-800 流动性等权, 与 #107 同口径)。择时 overlay 与「买入持有同一 universe」
对拍, 11 窗口 (2015-2026 全 regime)。

预注册判据 (已冻结, 禁止事后挑 MA 参数 / 换窗口 / 调门槛):
  主:   ≥80% 窗口 (11 → ≥9) 择时 overlay 全收益 ≥ 买入持有 且 最大回撤 ≤ 买入持有。
  副1 (牛市跟上): 纯牛窗口 (2019牛/2021白马转小盘/2025-26现期) overlay 全收益 ≥ 匹配universe全收益。
  副2 (熊市降险): 熊市/牛转崩窗口 (2015牛转股灾/2018熊/2020牛转崩/2022熊) overlay 回撤 ≤ 持有的一半。

信号口径 (无前视): 第 t 日收盘算 close_t vs MA20_t, 决定第 t+1 日仓位 (现金↔满仓),
每次仓位切换扣 31bp full-turnover 上界 (与 [[transaction-costs-unified]] 一致)。

用法: python scripts/run_index_timing_ab.py
输出: reports/index_timing_ab_result.json + .md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from analysis.benchmark import matched_universe_curve  # noqa: E402

OUT = ROOT / "reports" / "index_timing_ab_result.json"
COST_BPS = 31.0
UNIVERSE_N = 800
MA_WINDOW = 20  # 预注册: 只测 MA20, 不事后挑 10/30/60

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

# 预注册窗口分类 (纯牛 / 熊市牛转崩), 用于副判据
PURE_BULL = {"2019牛", "2021白马转小盘", "2025-26现期"}
BEAR_CRASH = {"2015牛转股灾", "2018熊", "2020牛转崩", "2022熊"}


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


def _timing_overlay(eq: pd.Series, idx_close: pd.Series, ma20: pd.Series, cost_bps: float) -> pd.Series:
    """指数绝对动量择时 overlay 净值 (无前视): sig_t 决定 t+1 仓位, 切换扣 cost_bps。

    eq: 匹配 universe 买入持有价格净值 (索引 date, 首值 1.0)。
    idx_close / ma20: 与 eq.index 对齐的上证综指收盘 / MA20。
    """
    dts = list(eq.index)
    daily = eq.pct_change().fillna(0.0)          # daily.iloc[i] = 进入 dts[i] 的收益
    sig = (idx_close.reindex(dts) > ma20.reindex(dts)).astype(float).fillna(0.0)
    cost = cost_bps / 10000.0

    eq_ov = 1.0
    pos = 0.0   # 进入当前日的仓位 (由前一交易日收盘信号决定)
    vals = []
    for i in range(len(dts)):
        if i > 0:
            eq_ov *= (1.0 + pos * float(daily.iloc[i]))
        vals.append(eq_ov)
        # 当日收盘更新次日仓位; 切换 (现金↔满仓) 在当日收盘扣费
        new_pos = sig.iloc[i]
        if new_pos != pos:
            eq_ov *= (1.0 - cost)
            pos = new_pos
    return pd.Series(vals, index=dts)


def main() -> int:
    _force_utf8()
    idx = pd.read_parquet(ROOT / "replay_data" / "index_series.parquet")
    idx["date"] = pd.to_datetime(idx["date"])
    sh = idx[idx.symbol == "sh.000001"].set_index("date")["close"].sort_index()
    # MA20 在全量指数序列上算 (含窗口前 lookback), 再切片到各窗口
    ma20_full = sh.rolling(MA_WINDOW).mean()

    report = {"date": datetime.now().isoformat(timespec="seconds"),
              "MA_WINDOW": MA_WINDOW, "COST_BPS": COST_BPS, "UNIVERSE_N": UNIVERSE_N,
              "windows": [], "verdict": {}}

    print("=" * 96)
    print(f"方向B 副线: 指数绝对动量择时 (close>MA{MA_WINDOW} 满仓 / ≤MA{MA_WINDOW} 现金) vs 买入持有 universe")
    print("=" * 96)

    for label, fp in WINDOWS.items():
        df = pd.read_parquet(ROOT / fp)
        df["date"] = pd.to_datetime(df["date"])
        T, T_end = df.date.min(), df.date.max()

        uni = _liquid_universe(df)
        d0 = df[df.date == T].set_index("symbol")
        uni = [s for s in uni if s in d0.index]
        eq = matched_universe_curve(df, uni)   # 买入持有 universe 价格净值

        win_sh = sh[(sh.index >= T) & (sh.index <= T_end)]
        win_ma = ma20_full[(ma20_full.index >= T) & (ma20_full.index <= T_end)]
        ov = _timing_overlay(eq, win_sh, win_ma, COST_BPS)

        m_eq = _metrics(eq)
        m_ov = _metrics(ov)
        cls = "纯牛" if label in PURE_BULL else ("熊市牛转崩" if label in BEAR_CRASH else "震荡/其他")
        row = {"window": label, "class": cls,
               "buy_hold": m_eq, "timing": m_ov,
               "timing_beats_return": bool(m_ov["return_pct"] >= m_eq["return_pct"]),
               # max_dd_pct 为负 (如 -52.52); 回撤更优 = 更不负 = 更大
               "timing_beats_dd": bool(m_ov["max_dd_pct"] >= m_eq["max_dd_pct"])}
        report["windows"].append(row)
        print(f"\n### {label} [{cls}]  (universe={len(uni)}, {T.date()}->{T_end.date()})")
        print(f"  买入持有: ret={m_eq['return_pct']:>7.2f}%  dd={m_eq['max_dd_pct']:>7.2f}%  sh={m_eq['sharpe']:>5.2f}")
        print(f"  择时overlay: ret={m_ov['return_pct']:>7.2f}%  dd={m_ov['max_dd_pct']:>7.2f}%  sh={m_ov['sharpe']:>5.2f}  "
              f"[收益{'胜' if m_ov['return_pct'] >= m_eq['return_pct'] else '负'} 回撤{'优' if m_ov['max_dd_pct'] >= m_eq['max_dd_pct'] else '劣'}]")

    ws = report["windows"]
    n = len(ws)

    # 主判据: ≥80% 窗口 overlay 全收益≥持有 且 回撤≤持有
    both = sum(1 for w in ws if w["timing_beats_return"] and w["timing_beats_dd"])
    thr = int(np.ceil(0.8 * n))

    # 副1 牛市跟上: 纯牛窗口 overlay 全收益 ≥ universe (买入持有)
    bull_wins = sum(1 for w in ws if w["class"] == "纯牛" and w["timing_beats_return"])
    bull_n = sum(1 for w in ws if w["class"] == "纯牛")
    # 副2 熊市降险: 熊市/牛转崩窗口 overlay 回撤 ≤ 持有的一半
    # max_dd_pct 为负: |overlay dd| <= |buy dd| / 2  ⟺  overlay dd >= buy dd / 2
    def _dd_half(w):
        return w["timing"]["max_dd_pct"] >= w["buy_hold"]["max_dd_pct"] / 2.0
    bear_wins = sum(1 for w in ws if w["class"] == "熊市牛转崩" and _dd_half(w))
    bear_n = sum(1 for w in ws if w["class"] == "熊市牛转崩")

    report["verdict"] = {
        "threshold": f">={thr}/{n} 窗口 (80%)",
        "main_both_windows": f"{both}/{n}",
        "main_pass": "通过" if both >= thr else "未通过",
        "sub1_bull_keepup": f"{bull_wins}/{bull_n} 纯牛窗口全收益≥universe",
        "sub2_bear_drawdown_half": f"{bear_wins}/{bear_n} 熊市窗口回撤≤持有的一半",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    print("预注册判据结果:")
    for k, v in report["verdict"].items():
        print(f"  {k}: {v}")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
