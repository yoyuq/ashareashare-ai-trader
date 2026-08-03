"""Regime 自适应验证 — 因子方向在多 regime 下是否稳定 (v3.2)

背景: 现有因子评估只在 ~110 日单 regime 窗口 (2026-02~07, 熊市)。
长回放 (830 只 SH 大盘, 199 日, 2025-10~2026-07) 覆盖两段:
  牛市 2025-10 ~ 2026-02 (等权 +13%) 与 熊市 2026-03 ~ 2026-07 (-17%)。

方法:
  1. 等权合成指数 OHLCV (截面均值, volume=求和) → MarketRegimeDetector 每日判定
  2. 按 regime 聚合交易日 → 牛/熊/震荡 三桶
  3. 每桶内跑截面 IC (factors.panels 向量化)
  4. 对比因子方向: 同号=稳健 (regime 无关); 异号=regime 依赖 (需动态权重)

局限: 830 只仅覆盖沪市 600000-601377 大盘 (首段连续代码), 非全市场;
       IC 幅度可能偏离全市场, 但方向稳定性结论在此样本上有效。

用法:
  python -m factors.regime_analysis [--replay replay_data/daily_2025-10-08_2026-07-31.parquet]
                                    [--lookahead 5] [--output reports/regime_factor_analysis.md]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.regime import MarketRegimeDetector  # noqa: E402
from factors.panels import compute_factor_panels, compute_forward_returns  # noqa: E402
from factors.sources import BASELINE_FACTORS, WAVE1_FACTORS  # noqa: E402

ALL_FACTORS = BASELINE_FACTORS + WAVE1_FACTORS

# regime → 宏观桶
BUCKET = {
    "strong_bull": "牛", "weak_bull": "牛",
    "range_bound": "震荡",
    "weak_bear": "熊", "strong_bear": "熊", "crisis": "熊",
}


def build_synthetic_index(df: pd.DataFrame) -> pd.DataFrame:
    """等权合成指数 OHLCV (截面均值, volume=求和) → 供 regime 检测。"""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    cols = {}
    for c in ("open", "high", "low", "close"):
        cols[c] = d.pivot_table(index="date", columns="symbol", values=c).mean(axis=1)
    idx = pd.DataFrame(cols)
    idx["volume"] = d.pivot_table(index="date", columns="symbol", values="volume").sum(axis=1)
    return idx


def build_breadth_df(df: pd.DataFrame) -> pd.DataFrame:
    """每日 上涨/下跌/20日新高/新低 计数 → 供 regime 检测的广度维度。"""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    piv_c = d.pivot_table(index="date", columns="symbol", values="close")
    r = piv_c.pct_change()
    out = pd.DataFrame(index=piv_c.index)
    out["advance"] = (r > 0).sum(axis=1)
    out["decline"] = (r <= 0).sum(axis=1)
    g = d.sort_values("date").groupby("symbol")
    d["high20"] = g["close"].transform(lambda s: s.rolling(20).max())
    d["low20"] = g["close"].transform(lambda s: s.rolling(20).min())
    d["at_high"] = d["close"] == d["high20"]
    d["at_low"] = d["close"] == d["low20"]
    dd = d.groupby("date").agg(new_high=("at_high", "sum"), new_low=("at_low", "sum"))
    return out.join(dd)


def detect_regime_series(idx: pd.DataFrame, br: pd.DataFrame | None = None) -> pd.Series:
    """逐日运行 MarketRegimeDetector, 返回 {date: 宏观桶}。br 可为 None (广度维度权重0)。"""
    det = MarketRegimeDetector()
    out = {}
    for i in range(20, len(idx)):
        d = idx.index[i]
        br_part = br.iloc[: i + 1] if br is not None else None
        r = det.detect(idx.iloc[: i + 1], br_part)
        out[pd.Timestamp(d)] = BUCKET[r.regime.value]
    return pd.Series(out)


def bucket_ic(panel, fwd, dates_in_bucket: list) -> dict:
    """某桶内: 截面 Spearman IC 均值/ICIR/正IC占比/方向。"""
    ics = []
    for T in dates_in_bucket:
        T = pd.Timestamp(T)
        if T not in panel.index or T not in fwd.index:
            continue
        fv, fr = panel.loc[T], fwd.loc[T]
        valid = fv.notna() & fr.notna() & np.isfinite(fv) & np.isfinite(fr)
        xs, ys = fv[valid], fr[valid]
        if len(xs) >= 30:
            ic, _ = spearmanr(xs, ys)
            ics.append(ic)
    if not ics:
        return {"mean_ic": None, "n": 0}
    a = np.array(ics)
    return {
        "mean_ic": float(a.mean()),
        "icir": float(a.mean() / (a.std() + 1e-9)),
        "pos_pct": float((a > 0).mean()),
        "n": len(a),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=str,
                    default="replay_data/daily_2025-10-08_2026-07-31.parquet")
    ap.add_argument("--lookahead", type=int, default=5)
    ap.add_argument("--min-symbols", type=int, default=30)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.replay)
    df["date"] = pd.to_datetime(df["date"])
    print(f"回放: {args.replay} | {df['symbol'].nunique()} 只 | "
          f"{df['date'].nunique()} 交易日 | {df['date'].min().date()}~{df['date'].max().date()}")

    # ── 1. regime 序列 ──
    idx = build_synthetic_index(df)
    br = build_breadth_df(df)
    regime_series = detect_regime_series(idx, br)

    # 合并到数据 (每股注入 mkt_close 供 rel_strength)
    from factors.market_situation import build_market_close_series
    mkt_cum = build_market_close_series(df)
    data = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date").set_index("date")
        g["mkt_close"] = mkt_cum.reindex(g.index)
        data[sym] = g

    print(f"regime 分布: {dict(Counter(regime_series.values))}")
    # regime 时段概览
    prev = None
    segs = []
    for d, b in regime_series.items():
        if b != prev:
            segs.append([b, d, d])
            prev = b
        else:
            segs[-1][2] = d
    print("regime 时段:")
    for b, s, e in segs:
        print(f"  {b}: {s.date()} ~ {e.date()}")

    # ── 2. 每桶 IC ──
    panels = compute_factor_panels(data, ALL_FACTORS)
    fwd = compute_forward_returns(data, args.lookahead)
    buckets = {}
    for b in set(regime_series.values):
        buckets[b] = [d for d in regime_series.index if regime_series[d] == b]

    print(f"\n每桶截面 IC (lookahead={args.lookahead}日):\n")
    print(f"{'因子':<18}{'牛IC':>8}{'牛n':>5}{'熊IC':>8}{'熊n':>5}{'震荡IC':>9}{'震荡n':>6}  方向稳定性")
    print("-" * 78)
    rows = []
    for f in ALL_FACTORS:
        panel = panels[f]
        res = {b: bucket_ic(panel, fwd, ds) for b, ds in buckets.items()}
        rb, rb_n = res.get("牛", {}).get("mean_ic"), res.get("牛", {}).get("n", 0)
        rs, rs_n = res.get("熊", {}).get("mean_ic"), res.get("熊", {}).get("n", 0)
        rr, rr_n = res.get("震荡", {}).get("mean_ic"), res.get("震荡", {}).get("n", 0)

        def fmt(v):
            return f"{v:+.3f}" if v is not None else "  — "

        # 方向稳定性: 牛熊都有样本且均值同号 → 稳健; 异号 → regime 依赖
        stability = "数据不足"
        if rb is not None and rs is not None and rb_n >= 10 and rs_n >= 10:
            if rb > 0 and rs > 0:
                stability = "稳定多头"
            elif rb < 0 and rs < 0:
                stability = "稳定反向"
            else:
                stability = "★ regime依赖"
        rows.append((f, rb, rb_n, rs, rs_n, rr, rr_n, stability))
        print(f"{f:<18}{fmt(rb):>8}{rb_n:>5}{fmt(rs):>8}{rs_n:>5}"
              f"{fmt(rr):>9}{rr_n:>6}  {stability}")

    # ── 3. 输出 ──
    if args.output:
        out_lines = [
            "# Regime 自适应验证 — 因子方向稳定性 (v3.2)\n",
            f"回放: {args.replay} ({df['symbol'].nunique()} 只 SH 大盘, "
            f"{df['date'].nunique()} 交易日, {df['date'].min().date()}~{df['date'].max().date()})",
            f"市场代理: 等权合成指数 | regime: MarketRegimeDetector (6态→牛/熊/震荡) | "
            f"lookahead={args.lookahead}\n",
            "## regime 时段\n",
        ]
        for b, s, e in segs:
            out_lines.append(f"- **{b}**: {s.date()} ~ {e.date()}")
        out_lines += [
            "",
            "## 各桶截面 IC\n",
            "| 因子 | 牛IC | 牛n | 熊IC | 熊n | 震荡IC | 震荡n | 方向稳定性 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for f, rb, rb_n, rs, rs_n, rr, rr_n, st in rows:
            f1 = f"{rb:+.3f}" if rb is not None else "—"
            f2 = f"{rs:+.3f}" if rs is not None else "—"
            f3 = f"{rr:+.3f}" if rr is not None else "—"
            out_lines.append(f"| {f} | {f1} | {rb_n} | {f2} | {rs_n} | {f3} | {rr_n} | {st} |")
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"\n报告: {p}")


if __name__ == "__main__":
    main()
