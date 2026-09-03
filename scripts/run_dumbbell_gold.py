"""冷落低波 + 黄金ETF 518880 哑铃 (85/15) — 预注册 reports/agent_loop/prereg_dumbbell_gold.md.

A = 100% 冷落低波 buffer 臂; B = 85/15 月末纪律再平衡 (股65bp/金15bp, 再平衡日一次)。
窗口 = harness 全部 11 窗。输出: reports/agent_loop/dumbbell_gold_result.json
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

W_STOCK = 0.85
COST_STOCK = 65.0 / 1e4
COST_GOLD = 15.0 / 1e4
ETF_FP = ROOT / "replay_data" / "etf_518880_daily.parquet"
ETF_START = pd.Timestamp("2013-07-29")
WINDOWS = H.WINDOWS  # 11 窗全量 (预注册写死)


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


def run_dumbbell(ret_s: pd.Series, ret_e: pd.Series) -> pd.Series:
    idx = ret_s.index.intersection(ret_e.index)
    rs = ret_s.reindex(idx).fillna(0.0)
    re = ret_e.reindex(idx).fillna(0.0)
    rl = set(month_ends(sorted(idx)))
    nav_s, nav_e = W_STOCK, 1.0 - W_STOCK
    prev_P = nav_s + nav_e
    vals = []
    for d in idx:
        nav_s *= 1.0 + rs[d]
        nav_e *= 1.0 + re[d]
        P = nav_s + nav_e
        r_b = P / prev_P - 1.0
        if d in rl:
            t_s, t_e = W_STOCK * P, (1.0 - W_STOCK) * P
            cost = abs(t_s - nav_s) * COST_STOCK + abs(t_e - nav_e) * COST_GOLD
            r_b -= cost / P
            nav_s, nav_e = t_s, t_e
            P = nav_s + nav_e
        vals.append((d, r_b))
        prev_P = P
    return pd.Series(dict(vals)).sort_index()


def main() -> int:
    _force_utf8()
    etf = etf_returns()
    print(f"518880 returns: {len(etf)} days {etf.index.min().date()} -> {etf.index.max().date()}")
    out = {}
    print(f"{'window':<14} {'arm':<10} {'total':>8} {'sharpe':>7} {'maxdd':>8}")
    for wname, fp in WINDOWS.items():
        df = H.load_base_window(fp)
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {fp}")
        score = _score(df)
        dr_s, _ = run_arm(df, score, "buffer")
        start = max(dr_s.index.min(), ETF_START)
        dr_s = dr_s[dr_s.index >= start]
        ret_e = etf[etf.index >= start]
        dr_b = run_dumbbell(dr_s, ret_e)
        mA, mB = metrics(dr_s), metrics(dr_b)
        out[wname] = {"A_stock": mA, "B_dumbbell": mB,
                      "diff_total": round(mB["total"] - mA["total"], 4),
                      "dd_improved": bool(mB["maxdd"] > mA["maxdd"]),
                      "sharpe_improved": bool(mB["sharpe"] > mA["sharpe"])}
        print(f"{wname:<14} {'A':<10} {mA['total']*100:>+7.1f}% {mA['sharpe']:>7.2f} {mA['maxdd']*100:>7.1f}%")
        print(f"{wname:<14} {'B(85/15)':<10} {mB['total']*100:>+7.1f}% {mB['sharpe']:>7.2f} {mB['maxdd']*100:>7.1f}%")

    diffs = [out[w]["diff_total"] for w in WINDOWS]
    dd_wins = sum(1 for w in WINDOWS if out[w]["dd_improved"])
    sh_wins = sum(1 for w in WINDOWS if out[w]["sharpe_improved"])
    avg_diff = float(np.mean(diffs))
    gate = bool(avg_diff >= -0.01 and (dd_wins >= 7) + (sh_wins >= 7) >= 2)
    summary = {"avg_total_diff": round(avg_diff, 4), "dd_improved_wins": f"{dd_wins}/11",
               "sharpe_improved_wins": f"{sh_wins}/11",
               "gate": "PASS" if gate else "FAIL",
               "rule": "avg差≥−1pp 且 (dd改善≥7/11 与 sharpe改善≥7/11 至少二)"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "dumbbell_gold_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
