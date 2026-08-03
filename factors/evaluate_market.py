"""市场形势因子评估 — 哪些市场级信号预测市场方向

与 `factors.evaluate.py` 的**截面 IC** 不同, 市场形势信号每交易日只有一个值
(全市场状态), 无法做截面相关。改用**时间序列相关**:
  信号值(T) 与 市场 N 日前向收益 的 Spearman/Pearson 秩相关 + 方向命中率。

判定: |rho|>0.1 且 命中率>55% → 该信号有市场方向预测力。
       (样本仅约60个交易日, 属初筛而非统计证明。)

用法: python -m factors.evaluate_market [--days 60] [--lookahead 5]
                                        [--output reports/market_situation.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from factors.market_situation import (  # noqa: E402
    MARKET_FACTOR_NAMES,
    build_market_close_series,
    build_market_panel,
)


def evaluate_signal(sig: pd.Series, fwd: pd.Series) -> dict:
    """单个信号对市场前向收益的时间序列相关。"""
    valid = sig.notna() & fwd.notna() & np.isfinite(sig) & np.isfinite(fwd)
    xs = sig[valid]
    ys = fwd[valid]
    if len(xs) < 20:
        return {"rho": None, "pearson": None, "hit_rate": None, "n": len(xs),
                "verdict": "数据不足"}
    rho, _ = spearmanr(xs, ys)
    pr, _ = pearsonr(xs, ys)
    # 方向命中率: 信号高于中位 → 市场涨; 低于中位 → 市场跌
    med = xs.median()
    hit = ((xs > med) & (ys > 0) | (xs <= med) & (ys <= 0)).mean()
    verdict = "有效" if abs(rho) > 0.1 and hit > 0.55 else "无效"
    return {"rho": float(rho), "pearson": float(pr), "hit_rate": float(hit),
            "n": int(len(xs)), "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60, help="回看交易日数")
    ap.add_argument("--lookahead", type=int, default=5, help="市场N日前向收益")
    ap.add_argument("--window-end", type=str, default="2026-07-31")
    ap.add_argument("--output", type=str, default=None, help="结果JSON输出路径")
    args = ap.parse_args()

    df = pd.read_parquet("replay_data/daily_2026-02-21_2026-07-31.parquet")
    panel = build_market_panel(df)
    mkt_cum = build_market_close_series(df)

    end = pd.Timestamp(args.window_end)
    start = end - pd.Timedelta(days=int(args.days * 1.5) + 30)
    panel = panel[(panel.index >= start) & (panel.index <= end)]
    mkt_fwd = mkt_cum.shift(-args.lookahead) / mkt_cum - 1
    mkt_fwd = mkt_fwd.reindex(panel.index)

    panel = panel.iloc[-args.days:]
    mkt_fwd = mkt_fwd.iloc[-args.days:]

    print(f"市场形势评估: {args.days}日, 市场{args.lookahead}日前向收益\n")
    print(f"{'信号':<24}{'Spearman':>9}{'Pearson':>9}{'命中率':>8}{'样本':>6}  判定")
    print("-" * 66)
    results = []
    for name in MARKET_FACTOR_NAMES:
        if name not in panel.columns:
            continue
        r = evaluate_signal(panel[name], mkt_fwd)
        results.append({"name": name, **r})
        if r["rho"] is None:
            print(f"{name:<24}{'—':>9}  数据不足")
            continue
        print(f"{name:<24}{r['rho']:>9.3f}{r['pearson']:>9.3f}"
              f"{r['hit_rate']:>8.0%}{r['n']:>6}  {r['verdict']}")

    print("\n有效市场信号 (|rho|>0.1 且 命中率>55%):")
    passed = [r for r in results if r["verdict"] == "有效"]
    for r in passed:
        print(f"  {r['name']}: rho={r['rho']:.3f} 命中={r['hit_rate']:.0%}"
              f" ({'看多' if r['rho'] > 0 else '看空'}信号)")
    if not passed:
        print("  (无)")

    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "days": args.days, "lookahead": args.lookahead,
                "window_end": args.window_end, "signals": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {p}")


if __name__ == "__main__":
    main()
