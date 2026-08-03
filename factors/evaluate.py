"""因子 IC 评估 — 判定哪些因子真正预测 N 日前向收益

方法:
  对历史窗口每个交易日 T:
    1. 每只股票计算因子分 (用 ≤T 数据)
    2. 计算 N 日前向收益 close[T+N]/close[T]-1
    3. 当日截面 Spearman 相关 (因子分 vs 前向收益) = 当日 IC
  汇总: 平均 IC / ICIR(IC均值/IC标准差) / 正IC天数占比 / 全池 Spearman

判定: |平均IC|>0.02 且 正IC占比>55% → 因子有真实预测力, 值得注入 LLM。
       ICIR>0.3 更佳。IC≈0 或负 → 剔除。

wave1 的市场相对因子 (rel_strength_*) 依赖等权市场指数, 由
`factors.market_situation.build_market_close_series` 注入为每股切片的
`mkt_close` 列。注册表契约 fn(df) 不变。

用法: python -m factors.evaluate [--days 40] [--lookahead 5] [--group all]
                                  [--window-end 2026-07-31] [--output reports/factor_evaluation.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from factors.sources import BASELINE_FACTORS, WAVE1_FACTORS  # noqa: E402


def load_pivots(
    window_end: str, days: int, lookahead: int, with_market: bool = True
):
    """加载日K → 每只股票的 close/volume/turn/peTTM/pbMRQ 时间序列 (datetime index)

    FIX(pandas 3.0.3): 交易日列表在 groupby **之前**向量化提取 —
    groupby 之后迭代 datetime64 列会 yield 绑定方法, 比较报错。
    """
    data = {}
    end = pd.Timestamp(window_end)
    start = end - pd.Timedelta(days=int(days * 1.5) + 30)
    df = pd.read_parquet("replay_data/daily_2026-02-21_2026-07-31.parquet")
    df["date"] = pd.to_datetime(df["date"])

    # FIX: 向量化提取窗口内交易日 (避免 groupby 后迭代出错)
    mask = (df["date"] >= start) & (df["date"] <= end)
    dates = sorted(set(pd.Timestamp(d).date() for d in df.loc[mask, "date"]))

    if with_market:
        from factors.market_situation import build_market_close_series
        market_close = build_market_close_series(df)
    else:
        market_close = None

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date")
        g = g[(g["date"] >= start) & (g["date"] <= end)]
        if len(g) < 60:
            continue
        g = g.set_index("date")
        if market_close is not None:
            g["mkt_close"] = market_close.reindex(g.index)
        data[sym] = g
    return data, dates


def asof_slice(sym_df: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    return sym_df[sym_df.index <= T]


def forward_return(sym_df: pd.DataFrame, T: pd.Timestamp, n: int):
    """T 日起第 n 个交易日的收盘收益"""
    fut = sym_df[sym_df.index > T].head(n)
    if len(fut) < n:
        return None
    entry = sym_df.loc[sym_df.index <= T, "close"].iloc[-1]
    exit_px = fut["close"].iloc[-1]
    if entry and entry > 0 and exit_px and exit_px > 0:
        return exit_px / entry - 1
    return None


def run_evaluation(
    factor_names: list, data: dict, dates: list, lookahead: int
) -> list:
    """对给定因子集跑完整 IC 评估, 返回每因子汇总 dict 列表。

    性能: 用 `factors.panels` 向量化计算每只股票的因子时序 + 前向收益面板,
    再按日取截面算 Spearman (避免 日×股 双重循环)。

    Returns:
        [{name, mean_ic, icir, pos_pct, pooled_s, n_days, verdict}]
    """
    from factors.panels import compute_factor_panels, compute_forward_returns

    panels = compute_factor_panels(data, factor_names)  # {f: DataFrame(date×sym)}
    fwd = compute_forward_returns(data, lookahead)      # DataFrame(date×sym)

    results = []
    for f in factor_names:
        panel = panels[f]
        ics = []
        xs_all, ys_all = [], []
        n_days_used = 0
        for T_str in dates:
            T = pd.Timestamp(T_str)
            if T not in panel.index or T not in fwd.index:
                continue
            fv = panel.loc[T]
            fr = fwd.loc[T]
            valid = fv.notna() & fr.notna() & np.isfinite(fv) & np.isfinite(fr)
            xs = fv[valid]
            ys = fr[valid]
            if len(xs) >= 30:
                ic, _ = spearmanr(xs, ys)
                ics.append(ic)
                n_days_used += 1
                xs_all.extend(xs.tolist())
                ys_all.extend(ys.tolist())
        ic_arr = np.array(ics)
        if len(ic_arr) == 0:
            results.append({
                "name": f, "mean_ic": None, "icir": None, "pos_pct": None,
                "pooled_s": None, "n_days": 0, "verdict": "数据不足",
            })
            continue
        mean_ic = np.mean(ic_arr)
        icir = mean_ic / (np.std(ic_arr) + 1e-9)
        pos_pct = (ic_arr > 0).mean()
        pooled_s = 0.0
        if len(xs_all) > 50:
            pooled_s = spearmanr(xs_all, ys_all)[0]
        verdict = "有效" if abs(mean_ic) > 0.02 and pos_pct > 0.55 else "无效"
        results.append({
            "name": f, "mean_ic": float(mean_ic), "icir": float(icir),
            "pos_pct": float(pos_pct), "pooled_s": float(pooled_s),
            "n_days": n_days_used, "verdict": verdict,
        })
    return results


def print_table(results: list, n_days: int, title: str):
    print(f"{title} (有效交易日: {n_days})\n")
    print(f"{'因子':<18}{'平均IC':>8}{'ICIR':>7}{'正IC占比':>9}{'全池Spearman':>14}  判定")
    print("-" * 70)
    for r in results:
        if r["mean_ic"] is None:
            print(f"{r['name']:<18}{'—':>8}  数据不足")
            continue
        print(
            f"{r['name']:<18}{r['mean_ic']:>8.3f}{r['icir']:>7.2f}"
            f"{r['pos_pct']:>9.0%}{r['pooled_s']:>14.3f}  {r['verdict']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=40, help="回看交易日数")
    ap.add_argument("--lookahead", type=int, default=5, help="N日前向收益")
    ap.add_argument("--window-end", type=str, default="2026-07-31")
    ap.add_argument("--group", type=str, default="all",
                    choices=["all", "baseline", "wave1"], help="因子分组")
    ap.add_argument("--output", type=str, default=None, help="结果JSON输出路径")
    args = ap.parse_args()

    data, dates = load_pivots(args.window_end, args.days, args.lookahead)
    dates = dates[-args.days:]

    groups = {"all": BASELINE_FACTORS + WAVE1_FACTORS,
              "baseline": BASELINE_FACTORS, "wave1": WAVE1_FACTORS}
    factor_names = groups[args.group]
    print(f"因子评估: {args.days}日 × {len(data)} 只, lookahead={args.lookahead}日")
    print(f"分组: {args.group} ({len(factor_names)}因子)\n")

    results = run_evaluation(factor_names, data, dates, args.lookahead)
    print_table(results, results[0]["n_days"] if results else 0, f"[{args.group}] IC 结果")

    print("\n有效因子 (平均|IC|>0.02 且 正IC占比>55%):")
    passed = [r for r in results if r["verdict"] == "有效"]
    for r in passed:
        print(f"  {r['name']}: 平均IC={r['mean_ic']:.3f} ICIR={r['icir']:.2f} "
              f"正IC={r['pos_pct']:.0%} (方向: {'多头' if r['mean_ic'] > 0 else '反向'})")
    if not passed:
        print("  (无 — 需尝试更多因子)")

    if args.output:
        out = {
            "group": args.group, "days": args.days, "lookahead": args.lookahead,
            "window_end": args.window_end, "n_symbols": len(data),
            "factors": results,
        }
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {p}")


if __name__ == "__main__":
    main()
