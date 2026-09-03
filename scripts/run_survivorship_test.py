"""幸存者偏差修正 (换数据源, 预注册 §九): 把真退市股(type1/2)合并进存活集, 重跑冷落低波精选。

问题: 冷落低波精选回测用的是存活股集, 不含退市股 → 幸存者偏差。冷落票(低换手/低波/小市值)
恰是最易退市的票, 若溢价部分来自"退市票被剔除", 则真实溢价被高估。

判据 (预注册 §九, 跑完不调): 修正后仍成立 ⇔
  (a) 非纯牛 5 窗(2018/2021/2022/2023/2025-26) ≥4/5 跑赢 universe; 且
  (b) 可修正 8 窗均 vs_universe > 0。

用法: python scripts/run_survivorship_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import strategy_research_harness as H  # noqa: E402

OBJECTIVE = {
    "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
    "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
    "2024震荡": "纯牛", "2025-26现期": "震荡",
}
# 可修正窗口 (退市数据 2018 起)
COVERED = ["2018熊", "2019牛", "2020牛转崩", "2021白马转小盘", "2022熊", "2023震荡", "2024震荡", "2025-26现期"]


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# 与 run_llm_filter_ab.py / run_own_strategies.py 一致的冷落低波 score
def _factors(df):
    out = df[["date", "symbol", "turn", "amount", "close"]].copy()
    s = df.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    out["std20"] = s.reindex(df.index)
    fmkt = df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8
    out["log_mkt"] = np.log1p(fmkt)
    return out


def _cold_lowvol(df):
    f = _factors(df)
    r_turn = f["turn"].groupby(df["date"]).rank(pct=True)
    r_std = f["std20"].groupby(df["date"]).rank(pct=True)
    r_mkt = f["log_mkt"].groupby(df["date"]).rank(pct=True)
    return -(r_turn + r_std + r_mkt)


def _norm_code(x) -> str:
    """任何 code 格式 → 6 位数字字符串."""
    s = str(x)
    return s.split(".")[-1]


def load_delisted_codes() -> set[str]:
    st = pd.read_parquet(ROOT / "replay_data" / "delisted_stocks.parquet")
    st = st[st["type"].astype(str).isin(["1", "2"])]  # 真退市 (type 为字符串, 排除 type4 吸收合并)
    return {_norm_code(c) for c in st["code"]}


def load_delisted_window(fp: str, delisted_codes: set[str]) -> pd.DataFrame:
    """加载与窗口日期范围重叠、且通过同口径过滤的退市股日线."""
    base = pd.read_parquet(ROOT / fp, columns=["date"])
    base["date"] = pd.to_datetime(base["date"])
    t0, t1 = base["date"].min(), base["date"].max()

    d = pd.read_parquet(ROOT / "replay_data" / "delisted_daily.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= t0) & (d["date"] <= t1)]
    d = d[d["code"].map(_norm_code).isin(delisted_codes)]
    # 同 load_base_window 口径过滤
    d["isST"] = d["isST"].astype(str)
    d = d[d["is_trade"] == 1]
    d = d[d["isST"] == "0"]
    d = d[d["symbol"].str.startswith(tuple(H.MAIN_BOARD))]
    d = d.sort_values(["date", "symbol"]).reset_index(drop=True)
    if not d.empty:
        d["ret"] = d.groupby("symbol")["close"].pct_change()
    return d


def _top5_picks_with_delisted(df, score, freq, delisted_symbols: set[str]):
    """返回 top5 选中退市股的再平衡日数与总数."""
    df = df.copy()
    df["score"] = score.values
    dates = H.pick_rebalance_dates(df["date"].tolist(), freq)
    n_delisted_days = 0
    delisted_picked = set()
    for T in dates:
        day = df[df["date"] == T].dropna(subset=["score"])
        if day.empty:
            continue
        top = day.nlargest(5, "score")["symbol"].tolist()
        hit = [s for s in top if s in delisted_symbols]
        if hit:
            n_delisted_days += 1
            delisted_picked.update(hit)
    return n_delisted_days, len(dates), delisted_picked


def main() -> int:
    _force_utf8()
    delisted_codes = load_delisted_codes()
    idx = H.load_index()

    # 全量退市股日线, 供 symbol 映射
    dd_all = pd.read_parquet(ROOT / "replay_data" / "delisted_daily.parquet")
    dd_all["date"] = pd.to_datetime(dd_all["date"])

    results = {"covered_windows": COVERED, "survivor_only": {}, "survivor_plus_delisted": {}, "picks": {}}
    print("=" * 92)
    print("幸存者偏差修正 — 冷落低波精选 (top5 月再平衡, 31bp)")
    print("=" * 92)
    print(f"{'窗口':<14} {'regime':<6} {'原vs_univ':>9} {'修正vs_univ':>11} {'Δ(pp)':>7}  退市股 选中")
    print("-" * 92)

    for window in COVERED:
        fp = H.WINDOWS[window]
        df_surv = H.load_base_window(fp)

        # 原回测 (存活集)
        m_surv = H.run_window_ranking(df_surv, idx, _cold_lowvol, 5, "monthly", window)

        # 修正回测 (存活集 + 退市股)
        d_del = load_delisted_window(fp, delisted_codes)
        n_added = d_del["symbol"].nunique() if not d_del.empty else 0
        df_full = pd.concat([df_surv, d_del], ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
        m_full = H.run_window_ranking(df_full, idx, _cold_lowvol, 5, "monthly", window)

        # 选中退市股诊断
        delisted_symbols = set(d_del["symbol"].unique()) if not d_del.empty else set()
        score_full = _cold_lowvol(df_full)
        n_hit, n_rebal, picked = _top5_picks_with_delisted(df_full, score_full, "monthly", delisted_symbols)

        results["survivor_only"][window] = m_surv
        results["survivor_plus_delisted"][window] = m_full
        results["picks"][window] = {"n_delisted_added": n_added, "rebal_days_with_delisted_pick": n_hit,
                                    "n_rebalance": n_rebal, "delisted_picked": sorted(picked)}

        dv = (m_full["vs_universe"] - m_surv["vs_universe"]) * 100
        print(f"{window:<14} {OBJECTIVE[window]:<6} {m_surv['vs_universe']*100:>+8.2f}pp "
              f"{m_full['vs_universe']*100:>+10.2f}pp {dv:>+6.2f}   {n_added:>3}只   {n_hit}/{n_rebal}日")

    # 判据
    full = results["survivor_plus_delisted"]
    surv = results["survivor_only"]
    nonniu = [w for w in COVERED if OBJECTIVE[w] != "纯牛"]
    c1_win = sum(1 for w in nonniu if full[w]["vs_universe"] >= 0)
    c1 = c1_win >= 4
    mean_full = float(np.mean([full[w]["vs_universe"] for w in COVERED]))
    mean_surv = float(np.mean([surv[w]["vs_universe"] for w in COVERED]))
    c2 = mean_full > 0

    print("\n" + "=" * 92)
    print("预注册判据 (§九)")
    print("=" * 92)
    print(f"  (a) 非纯牛 5 窗跑赢 universe: {c1_win}/5 → {'PASS' if c1 else 'FAIL'} (门槛 ≥4/5)")
    print(f"  (b) 8 窗均 vs_universe: 修正后 {mean_full*100:+.2f}pp vs 原 {mean_surv*100:+.2f}pp → "
          f"{'PASS (>0)' if c2 else 'FAIL (转负)'}")
    verdict = "冷落溢价经幸存者偏差修正后仍成立" if (c1 and c2) else "冷落溢价(至少部分)是幸存者偏差 → 下调信心"
    print(f"  → 结论: {verdict}")

    # 副诊断: 退市股被选中后的实际影响
    print("\n副诊断 (不 gate):")
    for window in COVERED:
        p = results["picks"][window]
        if p["rebal_days_with_delisted_pick"] > 0:
            print(f"  {window:<14} 选中退市股 {p['rebal_days_with_delisted_pick']}/{p['n_rebalance']} 个再平衡日: "
                  f"{p['delisted_picked']}")

    H.dump_json(results, ROOT / "reports" / "survivorship_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
