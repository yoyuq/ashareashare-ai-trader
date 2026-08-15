"""横截面因子 A/B (方案3e, 预注册) — 低换手/短期反转/短期动量/低波 能否跑赢等权市场。

背景: 撞四次墙 (价值/成长/择时/进化 均非牛市alpha)。学习闭环规则测试器无法忠实表达
横截面"按因子排序选top-K", 故本脚本直接测这四个未测过的交易类因子。

因子 (point-in-time, 窗口起点 T 用 lookback 计算):
  low_turn  低换手  = turn(T) 升序 bottom-K (被市场冷落的票)
  reversal  短期反转 = 过去20日收益 升序 bottom-K (买超跌)
  momentum  短期动量 = 过去20日收益 降序 top-K (买强势; 研究员称A股短期动量强)
  low_vol   低波动  = 过去40日日收益std 升序 bottom-K

基线: B1 等权 = top-800 流动性可投资 universe ∩ 窗口首日可交易 (剔除次新幸存者偏差)。
判据 (预注册): return_T > return_B1 的窗口数 >= 2/3 → 因子alpha成立; 1/3 inconclusive; 0/3 rejected。
每个因子独立判定, 不合并挑优。

lookback: 2019 用 daily_2018-10-01_2019-12-31.parquet; 2020/2024 用 lookback_*.parquet
(由 fetch_factor_lookback.py 产出; 缺失则该窗口只能算 low_turn, 其余因子如实标注缺窗口)。

用法: python scripts/run_factor_ab.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
TOP_K = 30
N_REV = 20   # 短期反转/动量: 过去20交易日
N_VOL = 40   # 低波: 过去40交易日
OUT = ROOT / "reports" / "factor_ab_result.json"

WINDOWS = {
    "2019牛": {
        "win": "replay_data/daily_2019-01-01_2019-12-31.parquet",
        "lookback": "replay_data/daily_2018-10-01_2019-12-31.parquet",
        "T": "2019-01-02",
    },
    "2020牛": {
        "win": "replay_data/daily_2020-06-01_2021-02-28.parquet",
        "lookback": "replay_data/lookback_2020.parquet",
        "T": "2020-06-01",
    },
    "2024震荡": {
        "win": "replay_data/daily_2024-01-01_2024-12-31.parquet",
        "lookback": "replay_data/lookback_2024.parquet",
        "T": "2024-01-02",
    },
}


def _metrics(df: pd.DataFrame, symbols: list[str]) -> dict:
    """等权买入持有: return%/maxDD%/sharpe。"""
    if not symbols:
        return {"n": 0, "return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
    sub = df[df.symbol.isin(symbols)][["date", "symbol", "close"]].dropna(subset=["close"])
    piv = sub.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    piv = piv.reindex(columns=[s for s in symbols if s in piv.columns])
    norm = piv / piv.iloc[0]
    norm = norm.ffill().dropna(axis=1, how="all")
    if norm.shape[1] == 0:
        return {"n": 0, "return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
    pf = norm.mean(axis=1)
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


def _factors(combined: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    """在 T 处对每票算因子: turn(T), ret20, vol40。"""
    close = combined.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    turn = combined[combined.date == T].set_index("symbol")["turn"]
    if T not in close.index:
        return pd.DataFrame(index=close.columns)
    pos = close.index.get_loc(T)
    fac = pd.DataFrame(index=close.columns)
    fac["turn"] = turn.reindex(close.columns)
    if pos >= N_REV:
        fac["ret20"] = close.iloc[pos] / close.iloc[pos - N_REV] - 1.0
    if pos >= N_VOL:
        fac["vol40"] = close.iloc[pos - N_VOL:pos + 1].pct_change().std()
    return fac


def main() -> int:
    report = {"date": datetime.now().isoformat(timespec="seconds"),
              "TOP_K": TOP_K, "N_REV": N_REV, "N_VOL": N_VOL,
              "windows": [], "verdict": {}}
    print("=" * 84)
    print("横截面因子 A/B (方案3e): 低换手/短期反转/动量/低波 vs 等权市场")
    print("=" * 84)
    for name, cfg in WINDOWS.items():
        win_df = pd.read_parquet(ROOT / cfg["win"])
        lb_fp = ROOT / cfg["lookback"]
        has_lb = lb_fp.exists()
        combined = win_df
        if has_lb:
            lb = pd.read_parquet(lb_fp)
            combined = pd.concat([lb, win_df], ignore_index=True).drop_duplicates(subset=["date", "symbol"])
        combined = combined.sort_values(["symbol", "date"])
        T = pd.Timestamp(cfg["T"])
        uni = _liquid_universe(win_df)
        d0 = win_df[win_df.date == T].set_index("symbol")["close"]
        uni = [s for s in uni if s in d0.index]
        fac = _factors(combined, T).reindex(uni)

        b1 = _metrics(win_df, uni)
        print(f"\n### {name}  (universe={len(uni)}, lookback={'有' if has_lb else '无'})")
        print(f"  B1 等权市场: n={b1['n']:>4}  return={b1['return_pct']:>8.2f}%  dd={b1['max_dd_pct']:>7.2f}%  sh={b1['sharpe']:>5.2f}")

        win_rec = {"window": name, "universe": len(uni), "B1": b1, "factors": {}}
        lt = fac.dropna(subset=["turn"]).nsmallest(TOP_K, "turn")
        m_lt = _metrics(win_df, lt.index.tolist())
        win_rec["factors"]["low_turn"] = m_lt
        print(f"  low_turn 低换手: n={m_lt['n']:>4}  return={m_lt['return_pct']:>8.2f}%  Δret_vsB1={m_lt['return_pct']-b1['return_pct']:>+7.2f}pp")
        if "ret20" in fac.columns:
            rev = fac.dropna(subset=["ret20"]).nsmallest(TOP_K, "ret20")
            mom = fac.dropna(subset=["ret20"]).nlargest(TOP_K, "ret20")
            m_rev = _metrics(win_df, rev.index.tolist())
            m_mom = _metrics(win_df, mom.index.tolist())
            win_rec["factors"]["reversal"] = m_rev
            win_rec["factors"]["momentum"] = m_mom
            print(f"  reversal 反转 : n={m_rev['n']:>4}  return={m_rev['return_pct']:>8.2f}%  Δret_vsB1={m_rev['return_pct']-b1['return_pct']:>+7.2f}pp")
            print(f"  momentum 动量 : n={m_mom['n']:>4}  return={m_mom['return_pct']:>8.2f}%  Δret_vsB1={m_mom['return_pct']-b1['return_pct']:>+7.2f}pp")
        if "vol40" in fac.columns:
            lv = fac.dropna(subset=["vol40"]).nsmallest(TOP_K, "vol40")
            m_lv = _metrics(win_df, lv.index.tolist())
            win_rec["factors"]["low_vol"] = m_lv
            print(f"  low_vol  低波 : n={m_lv['n']:>4}  return={m_lv['return_pct']:>8.2f}%  Δret_vsB1={m_lv['return_pct']-b1['return_pct']:>+7.2f}pp")
        report["windows"].append(win_rec)

    factors = ["low_turn", "reversal", "momentum", "low_vol"]
    wins = {f: 0 for f in factors}
    counts = {f: 0 for f in factors}
    for w in report["windows"]:
        for f in factors:
            m = w["factors"].get(f)
            if m and not np.isnan(m["return_pct"]):
                counts[f] += 1
                if m["return_pct"] > w["B1"]["return_pct"]:
                    wins[f] += 1
    total = len(report["windows"])
    report["win_counts"] = {}
    for f in factors:
        c = counts[f]
        report["win_counts"][f] = f"{wins[f]}/{c}"
        if c < total:
            report["verdict"][f] = "incomplete(缺窗口lookback)"
        elif wins[f] >= 2:
            report["verdict"][f] = "因子alpha成立"
        elif wins[f] == 1:
            report["verdict"][f] = "inconclusive"
        else:
            report["verdict"][f] = "rejected"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'=' * 84}")
    print("判据 (return > B1 的窗口数 >= 2/3):")
    for f in factors:
        print(f"  {f:9s}: {report['win_counts'][f]} → {report['verdict'][f]}")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
