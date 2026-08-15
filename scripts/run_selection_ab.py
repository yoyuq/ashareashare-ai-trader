"""截面选股 A/B — 低估质量选股能否跑赢等权市场 (方案3c, 预注册)。

回答用户假设: "牛市里分析局势 + 选优质股能跑赢市场(买入持有)"。
用系统已验证的 low_pe_value 规则(低 PE + 低 PB)作为"优质股"忠实代理
(数据仅 peTTM/pbMRQ/换手/成交额, 无 ROE/营收增速 — 如实声明)。

全程真实数据 (replay_data/*.parquet), 无模拟。buy-and-hold 向量化, 基线/处理组同构
(均无换手/无费率/单期持有), 差分隔离纯选股效应。

用法:
    python scripts/run_selection_ab.py
输出: reports/cross_sectional_selection_ab_result.json + 控制台报告
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
WINDOWS = {
    "2019牛": "replay_data/daily_2019-01-01_2019-12-31.parquet",
    "2020牛": "replay_data/daily_2020-06-01_2021-02-28.parquet",
    "2024震荡": "replay_data/daily_2024-01-01_2024-12-31.parquet",
}
IDX_FILE = "replay_data/daily_2020-06-01_2021-02-28_idx.parquet"
OUT = ROOT / "reports" / "cross_sectional_selection_ab_result.json"

# ── 预注册参数 (禁止事后调参) ──
MAX_PE = 15.0   # tester low_pe_value 忠实默认
MAX_PB = 2.0    # tester low_pe_value 忠实默认
TOP_K = 30      # T2 纯低 PE 排序取前 K


def _load(fp: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / fp)
    return df[df.symbol != "sh.000001"].copy()  # 剔除指数符号


def _investable(df: pd.DataFrame) -> pd.DataFrame:
    """窗口首日截面 → 可投资 universe (symbol + 首日 pe/pb)。"""
    d0 = df.date.min()
    s = df[df.date == d0].copy()
    s["isST"] = s["isST"].astype(str)
    inv = s[(s.isST == "0") & (s.is_trade == 1) & (s.peTTM > 0) & (s.pbMRQ > 0)]
    return inv[["symbol", "peTTM", "pbMRQ"]]


def _metrics(df: pd.DataFrame, symbols: list[str]) -> dict:
    """等权买入持有组合: return%, maxDD%, sharpe。symbols 空 → 全 NaN。"""
    if not symbols:
        return {"n": 0, "return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
    sub = df[df.symbol.isin(symbols)][["date", "symbol", "close"]].dropna(subset=["close"])
    piv = sub.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    norm = piv / piv.iloc[0]
    norm = norm.ffill()                       # 买入持有: 停牌日沿用前值
    norm = norm.dropna(axis=1, how="all")     # 去掉首日无价的票
    if norm.shape[1] == 0:
        return {"n": 0, "return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
    pf = norm.mean(axis=1)                    # 等权日净值
    ret = float(pf.iloc[-1] - 1.0) * 100.0
    dd = float((pf / pf.cummax() - 1.0).min()) * 100.0
    dr = pf.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0
    return {"n": int(norm.shape[1]), "return_pct": round(ret, 3),
            "max_dd_pct": round(dd, 3), "sharpe": round(sharpe, 3)}


def _index_return(window_key: str, df_start: pd.Timestamp, df_end: pd.Timestamp) -> dict | None:
    """上证指数 sh.000001 在同一窗口内的买入持有收益 (仅 2020 有数据)。"""
    if window_key != "2020牛":
        return None
    idx = pd.read_parquet(ROOT / IDX_FILE)
    ix = idx[idx.symbol == "sh.000001"].sort_values("date")
    win = ix[(ix.date >= df_start) & (ix.date <= df_end)]
    if win.empty:
        return None
    ret = float(win.close.iloc[-1] / win.close.iloc[0] - 1.0) * 100.0
    return {"return_pct": round(ret, 3)}


def main() -> int:
    report = {"date": datetime.now().isoformat(timespec="seconds"),
              "pre_registered": {"MAX_PE": MAX_PE, "MAX_PB": MAX_PB, "TOP_K": TOP_K,
                                 "criterion": "return_T > return_B1 的窗口数 >= 2/3 → 选股alpha成立"},
              "windows": []}
    print("=" * 78)
    print("截面选股 A/B (方案3c): 低估质量选股 vs 等权市场")
    print("=" * 78)
    for name, fp in WINDOWS.items():
        df = _load(fp)
        inv = _investable(df)
        base_syms = inv.symbol.tolist()
        t1_syms = inv[(inv.peTTM < MAX_PE) & (inv.pbMRQ < MAX_PB)].symbol.tolist()
        t2_syms = inv.nsmallest(TOP_K, "peTTM").symbol.tolist()

        b1 = _metrics(df, base_syms)
        t1 = _metrics(df, t1_syms)
        t2 = _metrics(df, t2_syms)
        b2 = _index_return(name, df.date.min(), df.date.max())

        print(f"\n### {name}  (universe investable={len(base_syms)})")
        print(f"  B1 等权市场    : n={b1['n']:>5}  return={b1['return_pct']:>8.2f}%  dd={b1['max_dd_pct']:>7.2f}%  sharpe={b1['sharpe']:>6.2f}")
        print(f"  T1 pe<15&pb<2 : n={t1['n']:>5}  return={t1['return_pct']:>8.2f}%  dd={t1['max_dd_pct']:>7.2f}%  sharpe={t1['sharpe']:>6.2f}   Δret={t1['return_pct']-b1['return_pct']:>+7.2f}pp")
        print(f"  T2 top30低PE  : n={t2['n']:>5}  return={t2['return_pct']:>8.2f}%  dd={t2['max_dd_pct']:>7.2f}%  sharpe={t2['sharpe']:>6.2f}   Δret={t2['return_pct']-b1['return_pct']:>+7.2f}pp")
        if b2:
            print(f"  B2 上证指数    : return={b2['return_pct']:>8.2f}%")

        report["windows"].append({
            "window": name,
            "universe_investable": len(base_syms),
            "B1": b1, "T1": t1, "T2": t2,
            "T1_delta_vs_B1": round(t1["return_pct"] - b1["return_pct"], 3),
            "T2_delta_vs_B1": round(t2["return_pct"] - b1["return_pct"], 3),
            "B2_index": b2,
        })

    # ── 判据汇总 ──
    wins = {k: 0 for k in ("T1", "T2")}
    for w in report["windows"]:
        for k in ("T1", "T2"):
            d = w[f"{k}_delta_vs_B1"]
            if not np.isnan(d) and d > 0:
                wins[k] += 1
    total = len(report["windows"])
    verdict = {}
    for k in ("T1", "T2"):
        v = wins[k]
        verdict[k] = "选股alpha成立" if v >= 2 else ("inconclusive" if v == 1 else "rejected")
    report["verdict"] = verdict
    report["win_counts"] = {k: f"{wins[k]}/{total}" for k in wins}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'=' * 78}")
    print(f"判据 (return > B1 的窗口数 >= 2/3):")
    for k, v in verdict.items():
        print(f"  {k}: {wins[k]}/{total} 窗口跑赢 → {v}")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
