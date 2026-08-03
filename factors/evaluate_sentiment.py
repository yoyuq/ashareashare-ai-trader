"""市场情绪温度计评估 — 恐慌/贪婪分数能否预测市场方向

两个角度:
  1. 时间序列相关: 情绪分/成分 对 市场 N 日前向收益 的 Spearman/命中率
     (与 evaluate_market 一致, 但关注复合分的预测力)
  2. 极端区制条件收益: 当情绪进入 恐慌(≤20) / 贪婪(>80) 时,
     市场未来 N 日平均收益 — 恐慌后是否反弹, 贪婪后是否回落 (均值回归检验)。

用法: python -m factors.evaluate_sentiment [--days 80] [--lookahead 5]
                                            [--output reports/market_sentiment.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from factors.market_sentiment import (  # noqa: E402
    SENTIMENT_COMPONENTS,
    build_sentiment_panel,
    sentiment_label,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=80, help="回看交易日数")
    ap.add_argument("--lookahead", type=int, default=5, help="市场N日前向收益")
    ap.add_argument("--window-end", type=str, default="2026-07-31")
    ap.add_argument("--replay", type=str, default="replay_data/daily_2026-02-21_2026-07-31.parquet")
    ap.add_argument("--output", type=str, default=None, help="结果JSON输出路径")
    args = ap.parse_args()

    df = pd.read_parquet(args.replay)
    from factors.market_situation import build_market_close_series
    panel = build_sentiment_panel(df)
    mkt_cum = build_market_close_series(df)

    end = pd.Timestamp(args.window_end)
    start = end - pd.Timedelta(days=int(args.days * 1.5) + 30)
    panel = panel[(panel.index >= start) & (panel.index <= end)].iloc[-args.days:]
    mkt_fwd = mkt_cum.shift(-args.lookahead) / mkt_cum - 1
    mkt_fwd = mkt_fwd.reindex(panel.index)

    print(f"情绪温度计评估: {len(panel)}日, 市场{args.lookahead}日前向收益\n")

    # ── 1. 时间序列相关 ──
    names = SENTIMENT_COMPONENTS + ["sent_composite"]
    print(f"{'信号':<20}{'Spearman':>9}{'命中率':>8}{'样本':>6}  判定")
    print("-" * 60)
    results = []
    for name in names:
        sig = panel[name]
        valid = sig.notna() & mkt_fwd.notna() & np.isfinite(sig) & np.isfinite(mkt_fwd)
        xs, ys = sig[valid], mkt_fwd[valid]
        if len(xs) < 20:
            continue
        rho, _ = spearmanr(xs, ys)
        med = xs.median()
        hit = ((xs > med) & (ys > 0) | (xs <= med) & (ys <= 0)).mean()
        verdict = "有效" if abs(rho) > 0.1 and hit > 0.55 else "无效"
        results.append({"name": name, "rho": float(rho), "hit_rate": float(hit),
                        "n": int(len(xs)), "verdict": verdict})
        print(f"{name:<20}{rho:>9.3f}{hit:>8.0%}{len(xs):>6}  {verdict}")

    # ── 2. 极端区制条件收益 ──
    # 主口径用平滑分 (情绪位置), 原始分太抖
    comp = panel["sent_composite_smoothed"] if "sent_composite_smoothed" in panel.columns else panel["sent_composite"]
    print(f"\n极端区制条件收益 (lookahead={args.lookahead}日):")
    print(f"{'区制':<12}{'样本':>6}{'平均前向收益':>12}{'中位前向收益':>12}  解读")
    print("-" * 66)
    zones = {
        "极度恐慌(≤20)": comp <= 20,
        "恐慌(≤40)": (comp > 20) & (comp <= 40),
        "中性(40-60)": (comp > 40) & (comp <= 60),
        "贪婪(>60)": (comp > 60) & (comp <= 80),
        "极度贪婪(>80)": comp > 80,
    }
    zone_stats = {}
    for label, mask in zones.items():
        fr = mkt_fwd[mask].dropna()
        if len(fr) == 0:
            print(f"{label:<12}{0:>6}    —    (无样本)")
            continue
        avg = fr.mean()
        med = fr.median()
        interp = ""
        if avg > 0.01:
            interp = "反弹预期"
        elif avg < -0.01:
            interp = "回落风险"
        zone_stats[label] = {"n": int(len(fr)), "avg_fwd": float(avg), "med_fwd": float(med)}
        print(f"{label:<12}{len(fr):>6}{avg:>11.2%}{med:>12.2%}  {interp}")

    # ── 3. 最近状态 ──
    last = comp.iloc[-1]
    print(f"\n最近一日情绪: {last:.0f}/100 ({sentiment_label(last)})  @ {comp.index[-1].date()}")

    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "days": len(panel), "lookahead": args.lookahead,
                "window_end": args.window_end, "replay": args.replay,
                "signals": results, "zones": zone_stats,
                "latest": {"date": str(comp.index[-1].date()), "score": float(last),
                           "label": sentiment_label(last)},
            }, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {p}")


if __name__ == "__main__":
    main()
