"""冷落低波 + 转债ETF 511380 哑铃 — 预注册 reports/agent_loop/prereg_dumbbell_511380.md.

A = 100% 冷落低波 buffer 臂; B = 70/30 月末纪律再平衡 (股票65bp/ETF15bp, 再平衡日一次)。
窗口 = 511380 覆盖内 6 窗。输出: reports/agent_loop/dumbbell_511380_result.json
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
from run_rotation_vs_hold import _score, run_arm  # noqa: E402

W_STOCK = 0.70
COST_STOCK = 65.0 / 1e4
COST_ETF = 15.0 / 1e4
ETF_FP = ROOT / "replay_data" / "etf_511380_daily.parquet"
WINDOWS = {  # 511380 覆盖内 (路径与 harness WINDOWS 一致, 预注册写死)
    "2020牛转崩后段": "replay_data/daily_2020-06-01_2021-02-28.parquet",
    "2021白马转小盘": "replay_data/daily_2021-01-01_2021-12-31.parquet",
    "2022熊": "replay_data/daily_2022-01-01_2022-12-31.parquet",
    "2023震荡": "replay_data/daily_2023-01-01_2023-12-31.parquet",
    "2024震荡": "replay_data/daily_2024-01-01_2024-12-31.parquet",
    "2025-26现期": "replay_data/daily_2025-10-08_2026-07-31.parquet",
}
# 2020 窗起算日 = ETF 上市后首个可用日 (2020-04-07), 预注册写死
ETF_START = pd.Timestamp("2020-04-07")


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def etf_returns() -> pd.Series:
    d = pd.read_parquet(ETF_FP)
    d["date"] = pd.to_datetime(d["日期"])
    d = d.sort_values("date")
    close = pd.to_numeric(d["收盘"], errors="raise")
    ret = close.pct_change().to_numpy()
    return pd.Series(ret, index=d["date"].to_numpy()).dropna()


def month_ends(dates: list) -> list:
    s = pd.Series(pd.DatetimeIndex(dates))
    return s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()


def metrics(dr: pd.Series) -> dict:
    r = dr.dropna()
    total = float((1.0 + r).prod() - 1.0)
    n = len(r)
    ann = (1.0 + total) ** (252.0 / n) - 1.0 if n else np.nan
    std = float(r.std())
    sharpe = float(r.mean() / std * np.sqrt(252.0)) if std > 1e-12 else np.nan
    eq = (1.0 + r).cumprod()
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {"total": round(total, 4), "ann": round(ann, 4),
            "sharpe": round(sharpe, 3), "maxdd": round(maxdd, 4), "n_days": n}


def run_dumbbell(ret_s: pd.Series, ret_e: pd.Series) -> tuple[pd.Series, list]:
    """B 臂: 70/30, 月末纪律再平衡。两腿 nav 表示 (初始 nav_s=0.7, nav_e=0.3),
    日收益 = P_t/P_{t-1} − 1; 月末把两腿拉回 0.7P/0.3P, 成本按漂移腿价值记当日一次。"""
    idx = ret_s.index.intersection(ret_e.index)
    rs = ret_s.reindex(idx).fillna(0.0)
    re = ret_e.reindex(idx).fillna(0.0)
    rl = set(month_ends(sorted(idx)))
    nav_s, nav_e = W_STOCK, 1.0 - W_STOCK
    prev_P = nav_s + nav_e  # = 1.0
    vals = []
    for d in idx:
        nav_s *= 1.0 + rs[d]
        nav_e *= 1.0 + re[d]
        P = nav_s + nav_e
        r_b = P / prev_P - 1.0
        if d in rl:
            t_s, t_e = W_STOCK * P, (1.0 - W_STOCK) * P
            cost = abs(t_s - nav_s) * COST_STOCK + abs(t_e - nav_e) * COST_ETF
            r_b -= cost / P
            nav_s, nav_e = t_s, t_e
            P = nav_s + nav_e
        vals.append((d, r_b))
        prev_P = P
    return pd.Series(dict(vals)).sort_index(), []


def main() -> int:
    _force_utf8()
    etf = etf_returns()
    print(f"511380 returns: {len(etf)} days {etf.index.min().date()} -> {etf.index.max().date()}")
    out = {}
    print(f"{'window':<14} {'arm':<8} {'total':>8} {'ann':>8} {'sharpe':>7} {'maxdd':>8}")
    for wname, fp in WINDOWS.items():
        df = H.load_base_window(fp)
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {fp}")
        start = max(df["date"].min(), ETF_START) if "2020" in wname else df["date"].min()
        score = _score(df)
        dr_s, _ = run_arm(df, score, "buffer")
        dr_s = dr_s[dr_s.index >= start]
        ret_e = etf[etf.index >= start]
        dr_b, _ = run_dumbbell(dr_s, ret_e)
        mA, mB = metrics(dr_s), metrics(dr_b)
        out[wname] = {"A_stock": mA, "B_dumbbell": mB,
                      "diff_total": round(mB["total"] - mA["total"], 4),
                      "dd_improved": bool(mB["maxdd"] > mA["maxdd"]),
                      "sharpe_improved": bool(mB["sharpe"] > mA["sharpe"])}
        print(f"{wname:<14} {'A':<8} {mA['total']*100:>+7.1f}% {mA['ann']*100:>+7.1f}% "
              f"{mA['sharpe']:>7.2f} {mA['maxdd']*100:>7.1f}%")
        print(f"{wname:<14} {'B(70/30)':<8} {mB['total']*100:>+7.1f}% {mB['ann']*100:>+7.1f}% "
              f"{mB['sharpe']:>7.2f} {mB['maxdd']*100:>7.1f}%")

    diffs = [out[w]["diff_total"] for w in WINDOWS]
    dd_wins = sum(1 for w in WINDOWS if out[w]["dd_improved"])
    sh_wins = sum(1 for w in WINDOWS if out[w]["sharpe_improved"])
    avg_diff = float(np.mean(diffs))
    gate = bool(avg_diff >= -0.01 and (dd_wins >= 4) + (sh_wins >= 4) >= 2)
    summary = {"avg_total_diff": round(avg_diff, 4), "dd_improved_wins": f"{dd_wins}/6",
               "sharpe_improved_wins": f"{sh_wins}/6",
               "gate": "PASS" if gate else "FAIL",
               "rule": "avg差≥−1pp 且 (dd改善≥4/6 与 sharpe改善≥4/6 至少二)"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "dumbbell_511380_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
