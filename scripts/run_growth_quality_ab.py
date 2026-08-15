"""成长质量选股 A/B (方案3d, 预注册) — 高ROE×高增速选股能否跑赢市场。

背景: 方案3c 只能用低估值(pe/pb)代理"优质股", 且基线等权市场小盘重、与市值加权指数鸿沟大。
本脚本用 Baostock 基本面 (roeAvg/YOYNI/totalShare) 做 point-in-time 选股, 检验用户假设
"选优质成长股跑赢市场"。

预注册因子 (point-in-time, 公告日 pubDate <= 窗口起点, 取最新报告期):
  - Q 质量 = roeAvg (ROE, 年初至今累计)
  - G 成长 = YOYNI (净利润同比, 单季)
  - QG 复合 = z(roeAvg) + z(YOYNI)  (横截面 z-score)

处理组 (各 top-K=30, 等权买入持有):
  - T_Q : roeAvg 最高 top-30
  - T_G : YOYNI 最高 top-30
  - T_QG: 复合 z-score 最高 top-30

基线:
  - B1 等权 : 等权买入持有 top-800 流动性可投资 universe
  - B2 市值加权 : 按 totalShare×close 加权买入持有 (更接近真实指数)

判据 (同方案3c, 预注册): return_T > return_B1 的窗口数 >= 2/3 → 选股alpha成立; 1/3 inconclusive; 0/3 rejected。
诚实局限: roeAvg 为年初至今累计 (2020Q1 为单季, 有季节噪声); YOYNI 为单季同比。

用法: python scripts/run_growth_quality_ab.py
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
FUND = ROOT / "replay_data" / "fundamentals.parquet"
OUT = ROOT / "reports" / "growth_quality_ab_result.json"
TOP_K = 30


def _metrics(df: pd.DataFrame, symbols: list[str], weights: dict | None = None) -> dict:
    """买入持有组合: return%, maxDD%, sharpe。weights=None → 等权; dict → 起始市值加权。"""
    if not symbols:
        return {"n": 0, "return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
    sub = df[df.symbol.isin(symbols)][["date", "symbol", "close"]].dropna(subset=["close"])
    piv = sub.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    piv = piv.reindex(columns=[s for s in symbols if s in piv.columns])
    norm = piv / piv.iloc[0]
    norm = norm.ffill().dropna(axis=1, how="all")
    if norm.shape[1] == 0:
        return {"n": 0, "return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
    if weights is None:
        pf = norm.mean(axis=1)
    else:
        w = pd.Series(weights).reindex(norm.columns).fillna(0.0)
        w = w / w.sum()
        pf = (norm * w).sum(axis=1)
    ret = float(pf.iloc[-1] - 1.0) * 100.0
    dd = float((pf / pf.cummax() - 1.0).min()) * 100.0
    dr = pf.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0
    return {"n": int(norm.shape[1]), "return_pct": round(ret, 3),
            "max_dd_pct": round(dd, 3), "sharpe": round(sharpe, 3)}


def _liquid_universe(df: pd.DataFrame, top_n: int = 800) -> list[str]:
    df = df[df.symbol != "sh.000001"].copy()
    df["isST"] = df["isST"].astype(str)
    inv = df[(df.isST == "0") & (df.is_trade == 1) & (df.peTTM > 0) & (df.pbMRQ > 0)]
    med = inv.groupby("symbol")["amount"].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()


def _point_in_time(fund: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    f = fund.copy()
    f["pubDate"] = pd.to_datetime(f["pubDate"], errors="coerce")
    f = f[f.pubDate <= T]
    if f.empty:
        return f
    return f.sort_values(["symbol", "statDate"]).groupby("symbol").tail(1)


def main() -> int:
    fund = pd.read_parquet(FUND)
    report = {"date": datetime.now().isoformat(timespec="seconds"),
              "TOP_K": TOP_K, "windows": [], "verdict": {}}
    print("=" * 82)
    print("成长质量选股 A/B (方案3d): 高ROE×高增速 vs 市场")
    print("=" * 82)
    for name, fp in WINDOWS.items():
        df = pd.read_parquet(ROOT / fp)
        T = df.date.min()
        uni = _liquid_universe(df)
        pit = _point_in_time(fund, T)
        # universe ∩ 有基本面 (point-in-time) ∩ 窗口首日可交易
        # (可交易过滤: 剔除次新股上市前招股书高ROE的幸存者偏差; 对处理组与基线同等施加)
        fmap = pit.set_index("symbol")
        d0 = df[df.date == T].set_index("symbol")["close"]
        tradeable = set(d0.index)
        uni_f_all = [s for s in uni if s in fmap.index]
        uni_f = [s for s in uni_f_all if s in tradeable]
        excluded_nt = len(uni_f_all) - len(uni_f)
        # 起始市值 (totalShare × close at start)
        cap = {s: float(fmap.loc[s, "totalShare"] * d0[s])
               for s in uni_f if pd.notna(fmap.loc[s, "totalShare"])}

        b1 = _metrics(df, uni_f)
        b2 = _metrics(df, list(cap.keys()), weights=cap) if cap else {"n": 0}

        # 因子截面 (仅在有基本面且市值可算的 universe 内)
        sc = pd.DataFrame({
            "symbol": uni_f,
            "roe": [fmap.loc[s, "roeAvg"] for s in uni_f],
            "yoy": [fmap.loc[s, "YOYNI"] for s in uni_f],
        }).dropna(subset=["roe", "yoy"])
        z_roe = (sc["roe"] - sc["roe"].mean()) / sc["roe"].std()
        z_yoy = (sc["yoy"] - sc["yoy"].mean()) / sc["yoy"].std()
        sc["zqg"] = z_roe + z_yoy

        t_q = sc.nlargest(TOP_K, "roe").symbol.tolist()
        t_g = sc.nlargest(TOP_K, "yoy").symbol.tolist()
        t_qg = sc.nlargest(TOP_K, "zqg").symbol.tolist()

        m_b1, m_b2 = b1, b2
        m_q = _metrics(df, t_q)
        m_g = _metrics(df, t_g)
        m_qg = _metrics(df, t_qg)

        print(f"\n### {name}  (liquid universe={len(uni)}, 有基本面={len(uni_f_all)}, 可交易后={len(uni_f)}, 剔除次新/停牌={excluded_nt})")
        print(f"  B1 等权市场   : n={m_b1['n']:>4}  return={m_b1['return_pct']:>8.2f}%  dd={m_b1['max_dd_pct']:>7.2f}%  sh={m_b1['sharpe']:>5.2f}")
        if cap:
            print(f"  B2 市值加权   : n={m_b2['n']:>4}  return={m_b2['return_pct']:>8.2f}%  dd={m_b2['max_dd_pct']:>7.2f}%  sh={m_b2['sharpe']:>5.2f}")
        for tag, m in [("Q 高ROE", m_q), ("G 高增速", m_g), ("QG 复合", m_qg)]:
            print(f"  T_{tag:<6}: n={m['n']:>4}  return={m['return_pct']:>8.2f}%  dd={m['max_dd_pct']:>7.2f}%  sh={m['sharpe']:>5.2f}  Δret_vsB1={m['return_pct']-m_b1['return_pct']:>+7.2f}pp")

        report["windows"].append({
            "window": name, "universe": len(uni), "universe_with_fund": len(uni_f),
            "excluded_not_tradeable": excluded_nt,
            "B1": m_b1, "B2": m_b2, "T_Q": m_q, "T_G": m_g, "T_QG": m_qg,
            "Q_delta_vs_B1": round(m_q["return_pct"] - m_b1["return_pct"], 3),
            "G_delta_vs_B1": round(m_g["return_pct"] - m_b1["return_pct"], 3),
            "QG_delta_vs_B1": round(m_qg["return_pct"] - m_b1["return_pct"], 3),
        })

    wins = {k: 0 for k in ("Q", "G", "QG")}
    for w in report["windows"]:
        for k in ("Q", "G", "QG"):
            if not np.isnan(w[f"{k}_delta_vs_B1"]) and w[f"{k}_delta_vs_B1"] > 0:
                wins[k] += 1
    total = len(report["windows"])
    for k, v in wins.items():
        report["verdict"][k] = "选股alpha成立" if v >= 2 else ("inconclusive" if v == 1 else "rejected")
    report["win_counts"] = {k: f"{wins[k]}/{total}" for k in wins}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'=' * 82}")
    print("判据 (return > B1 等权市场 的窗口数 >= 2/3):")
    for k, v in report["verdict"].items():
        print(f"  T_{k}: {report['win_counts'][k]} → {v}")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
