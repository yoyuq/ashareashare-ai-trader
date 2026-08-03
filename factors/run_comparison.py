"""因子对比分析 — 基线 vs 第二波, 生成 reports/factor_comparison.md

对比内容:
  1. 全因子 IC 一览表 (baseline 8 + wave1 13)
  2. 基线 vs 扩展: 新增了哪些有效因子
  3. 有效因子两两相关 (去冗余)
  4. 复合 IC: 等权排名组合 "仅基线" vs "基线+wave1" 的截面预测力
  5. 市场形势因子榜 (读 evaluate_market 输出)

用法: python -m factors.run_comparison [--days 60] [--lookahead 5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from factors.evaluate import load_pivots, run_evaluation  # noqa: E402
from factors.panels import compute_factor_panels, compute_forward_returns  # noqa: E402
from factors.sources import BASELINE_FACTORS, WAVE1_FACTORS  # noqa: E402


def pooled_factor_corr(panels: dict, names: list) -> dict:
    """有效因子两两之间的全池 Spearman 相关 (去冗余参考)。"""
    long = {f: panels[f].stack() for f in names}
    out = {}
    for i, f in enumerate(names):
        for g in names[i + 1:]:
            common = long[f].index.intersection(long[g].index)
            x = long[f].loc[common]
            y = long[g].loc[common]
            valid = x.notna() & y.notna()
            rho = spearmanr(x[valid], y[valid])[0] if valid.sum() > 100 else 0.0
            out[f"{f}~{g}"] = float(rho)
    return out


def composite_ic(data, dates, factor_names: list, lookahead: int) -> list:
    """等权排名组合的每日截面 IC (因子取截面 rank 均值 → 与前瞻收益 Spearman)。"""
    panels = compute_factor_panels(data, factor_names)
    fwd = compute_forward_returns(data, lookahead)
    ics = []
    for T_str in dates:
        T = pd.Timestamp(T_str)
        if T not in fwd.index:
            continue
        frames = []
        for f in factor_names:
            if T in panels[f].index:
                r = panels[f].loc[T].rank(pct=True) - 0.5
                frames.append(r)
        if not frames:
            continue
        comp = pd.concat(frames, axis=1).mean(axis=1)
        fr = fwd.loc[T]
        valid = comp.notna() & fr.notna() & np.isfinite(comp) & np.isfinite(fr)
        xs, ys = comp[valid], fr[valid]
        if len(xs) >= 30:
            ic, _ = spearmanr(xs, ys)
            ics.append(ic)
    return ics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--lookahead", type=int, default=5)
    ap.add_argument("--window-end", type=str, default="2026-07-31")
    ap.add_argument("--out", type=str, default="reports/factor_comparison.md")
    args = ap.parse_args()

    data, dates = load_pivots(args.window_end, args.days, args.lookahead)
    dates = dates[-args.days:]

    all_names = BASELINE_FACTORS + WAVE1_FACTORS
    base_res = run_evaluation(BASELINE_FACTORS, data, dates, args.lookahead)
    full_res = run_evaluation(all_names, data, dates, args.lookahead)
    res_map = {r["name"]: r for r in full_res}

    passed = [r["name"] for r in full_res if r["verdict"] == "有效"]
    base_passed = [r["name"] for r in base_res if r["verdict"] == "有效"]
    new_passed = [f for f in passed if f in WAVE1_FACTORS]

    # 有效因子相关矩阵
    corr = {}
    if len(passed) >= 2:
        panels = compute_factor_panels(data, passed)
        corr = pooled_factor_corr(panels, passed)

    # 复合 IC: 仅基线 vs 基线+wave1
    comp_base = composite_ic(data, dates, BASELINE_FACTORS, args.lookahead)
    comp_full = composite_ic(data, dates, all_names, args.lookahead)

    # 市场形势榜 (从 evaluate_market 输出读取, 若存在)
    mkt_path = Path("reports/market_situation.json")
    mkt_signals = []
    if mkt_path.exists():
        mkt_signals = json.loads(mkt_path.read_text(encoding="utf-8"))["signals"]

    # ── 写 markdown ──
    L = []
    L.append("# 因子对比分析 — 基线 8 因子 vs 第二波 13 因子\n")
    L.append(f"窗口: {args.days} 个交易日 (至 {args.window_end}), "
             f"前向收益 {args.lookahead} 日, 全市场 {len(data)} 只股票\n")
    L.append("判定标准: 平均|IC|>0.02 且 正IC占比>55% → 有效\n")

    L.append("## 1. 全因子 IC 一览\n")
    L.append("| 因子 | 分组 | 平均IC | ICIR | 正IC占比 | 全池Spearman | 判定 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in full_res:
        grp = "baseline" if r["name"] in BASELINE_FACTORS else "wave1"
        if r["mean_ic"] is None:
            L.append(f"| {r['name']} | {grp} | — | — | — | — | 数据不足 |")
            continue
        L.append(f"| {r['name']} | {grp} | {r['mean_ic']:.3f} | {r['icir']:.2f} | "
                 f"{r['pos_pct']:.0%} | {r['pooled_s']:.3f} | {r['verdict']} |")

    L.append("\n## 2. 新增有效因子\n")
    if new_passed:
        for f in new_passed:
            r = res_map[f]
            direction = "多头(值越大越涨)" if r["mean_ic"] > 0 else "反向(值越大越跌)"
            L.append(f"- **{f}** 平均IC={r['mean_ic']:.3f}, ICIR={r['icir']:.2f}, "
                     f"正IC={r['pos_pct']:.0%} → {direction}")
    else:
        L.append("- 第二波因子中暂无通过 IC 阈值的因子。")

    L.append("\n## 3. 基线 vs 扩展复合 IC (等权排名组合)\n")
    L.append("| 组合 | 因子数 | 平均IC | ICIR | 正IC占比 |")
    L.append("|---|---|---|---|---|")
    for label, ics in [("仅基线", comp_base), ("基线+wave1", comp_full)]:
        if ics:
            L.append(f"| {label} | {len(BASELINE_FACTORS) if label=='仅基线' else len(all_names)} | "
                     f"{np.mean(ics):.3f} | {np.mean(ics)/(np.std(ics)+1e-9):.2f} | "
                     f"{(np.asarray(ics)>0).mean():.0%} |")
        else:
            L.append(f"| {label} | — | 数据不足 | — | — |")

    L.append("\n## 4. 有效因子两两相关 (|rho|>0.7 视为冗余)\n")
    if corr:
        L.append("| 因子对 | Spearman |")
        L.append("|---|---|")
        for pair, rho in sorted(corr.items(), key=lambda kv: abs(kv[1]), reverse=True):
            flag = " ⚠️冗余" if abs(rho) > 0.7 else ""
            L.append(f"| {pair} | {rho:.3f}{flag} |")
    else:
        L.append("- 无有效因子, 无法计算相关。")

    L.append("\n## 5. 市场形势因子榜 (时间序列相关, 对市场前向收益)\n")
    L.append("判定: |Spearman|>0.1 且 命中率>55% → 有效\n")
    if mkt_signals:
        L.append("| 信号 | Spearman | Pearson | 命中率 | 判定 |")
        L.append("|---|---|---|---|---|")
        for s in sorted(mkt_signals, key=lambda x: abs(x.get("rho") or 0), reverse=True):
            rho = f"{s['rho']:.3f}" if s["rho"] is not None else "—"
            pr = f"{s['pearson']:.3f}" if s["pearson"] is not None else "—"
            hit = f"{s['hit_rate']:.0%}" if s["hit_rate"] is not None else "—"
            L.append(f"| {s['name']} | {rho} | {pr} | {hit} | {s['verdict']} |")
    else:
        L.append("- 无市场形势评估结果 (先运行 `python -m factors.evaluate_market`)。")

    L.append("\n## 6. 结论与建议注入因子集\n")
    L.append("(见下方控制台输出的自动结论)\n")

    md = "\n".join(L) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")

    # ── 控制台结论 ──
    print(f"基线有效因子: {base_passed or '无'}")
    print(f"新增有效因子 (wave1): {new_passed or '无'}")
    if comp_base and comp_full:
        print(f"复合IC 仅基线: {np.mean(comp_base):.3f} (正IC{(np.asarray(comp_base)>0).mean():.0%})")
        print(f"复合IC 基线+wave1: {np.mean(comp_full):.3f} (正IC{(np.asarray(comp_full)>0).mean():.0%})")
    print(f"对比报告已生成: {args.out}")


if __name__ == "__main__":
    main()
