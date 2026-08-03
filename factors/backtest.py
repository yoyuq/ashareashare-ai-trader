"""因子选股回测 — 基线因子 vs 扩展因子 vs 纯估值 vs 市场等权基准

用回放全市场日K (replay_data/daily_*.parquet) 做截面因子选股回测:
  每 N 个交易日调仓: 按因子组合得分排名, 买入 Top-K (目标等权),
  用 AShareBroker 以**次日开盘价**成交 (T+1 / 涨跌停 / 停牌 / 费用 / 整手),
  消除"收盘决策+收盘成交"的时序乐观偏差。

对比策略 (因子组合见 factors.sources):
  baseline  仅基线有效因子: momentum_5 + bias_ma20
  extended  基线 + wave1 新增: + ep + bp + rel_strength_5d + sharpe_20
  value     纯估值: ep + bp
  market    市场等权买入持有基准 (无交易成本, 供参考)

用法: python -m factors.backtest [--topk 20] [--rebalance 5] [--warmup 25]
                                  [--output reports/factor_backtest.md]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.regime import MarketRegimeDetector  # noqa: E402  (resolve_names 保留参考)
from backtest.broker import AShareBroker  # noqa: E402
from factors.market_situation import build_market_close_series  # noqa: E402
from factors.panels import compute_factor_panels  # noqa: E402
from factors.regime_analysis import BUCKET  # noqa: E402  (resolve_names 保留参考)

# 策略 → 因子组合
# NOTE (v3.2 实测): "regime_adaptive" 动态切因子集策略已弃用 —
# 回测证明它不如纯估值 (长窗口 -8.8% vs value -6.9%; 全窗口 -14.0% vs -3.7%),
# 复杂度是负债。见 reports/factor_v32_assessment.md。代码保留供参考, 不再默认运行。
STRATEGIES = {
    "baseline": ["momentum_5", "bias_ma20"],
    "extended": ["momentum_5", "bias_ma20", "ep", "bp", "rel_strength_5d", "sharpe_20"],
    "value": ["ep", "bp"],
    "rec": ["ep", "bp", "rel_strength_5d"],  # 报告推荐的去相关注入集
}

# [弃用] v3.2 regime 自适应因子集 (来自 regime_factor_analysis.md 的方向稳定性):
#   牛: 估值 + 长周期动量 (momentum_20 牛 +0.020) + 跳空
#   震荡: 纯估值 + 跳空 (ep/bp 震荡最强 +0.093/+0.055)
#   熊: 估值 + 反转 + 超卖 (reversal_1d 熊 +0.025, rsi_extreme 熊 +0.043)
# 回测证明动态切换不优于静态估值核心, 保留代码仅作参考。
REGIME_FACTOR_SETS = {
    "牛": ["ep", "bp", "momentum_20", "gap_strength"],
    "震荡": ["ep", "bp", "gap_strength"],
    "熊": ["ep", "bp", "reversal_1d", "rsi_extreme"],
}


def load_universe(window_end: str = "2026-07-31", min_rows: int = 60,
                  replay: str = "replay_data/daily_2026-02-21_2026-07-31.parquet",
                  window_start: str = None):
    """加载回放全市场 → 每股日K (date index, 含 broker 所需字段)。

    过滤: 剔除 ST (isST==1)、历史不足 min_rows 的股票。
    window_start: 可选, 截取窗口起点 (用于单独验证牛市/熊市段)。
    """
    end = pd.Timestamp(window_end)
    df = pd.read_parquet(replay)
    df["date"] = pd.to_datetime(df["date"])
    if window_start:
        df = df[df["date"] >= pd.Timestamp(window_start)]
    df = df[df["date"] <= end]
    df = df[pd.to_numeric(df["isST"], errors="coerce") == 0]  # 剔除 ST

    data = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date")
        if len(g) < min_rows:
            continue
        g = g.set_index("date")
        g["pre_close"] = g["close"].shift(1)
        g["pct_change"] = g["pctChg"]  # broker 用百分比口径
        data[sym] = g
    return data


def score_of(panels: dict, T: pd.Timestamp, names: list):
    """多因子等权排名组合得分 (rank pct 均值, 越大越优)。"""
    frames = []
    for f in names:
        if T in panels[f].index:
            frames.append(panels[f].loc[T].rank(pct=True))
    if not frames:
        return None
    return pd.concat(frames, axis=1).mean(axis=1)


def bar_for(row: pd.Series) -> dict:
    """把某日一行转成 broker 需要的 bar dict。"""
    return {
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "pre_close": float(row["pre_close"]) if pd.notna(row["pre_close"]) else float(row["close"]),
        "pct_change": float(row["pct_change"]) if pd.notna(row["pct_change"]) else 0.0,
        "is_trade": int(row["is_trade"]) if pd.notna(row["is_trade"]) else 1,
    }


def resolve_names(names, regime_index, Tdt) -> list:
    """固定策略返回 names; regime_adaptive (names=="REGIME") 按 T 时刻 regime 选因子集。

    PIT 安全: 只用 ≤T 的合成指数数据判 regime。
    """
    if names != "REGIME":
        return names
    if regime_index is not None:
        sofar = regime_index[regime_index.index <= Tdt]
        if len(sofar) >= 20:
            try:
                r = MarketRegimeDetector().detect(sofar, None)
                return REGIME_FACTOR_SETS[BUCKET[r.regime.value]]
            except Exception:
                return REGIME_FACTOR_SETS["震荡"]
    return REGIME_FACTOR_SETS["震荡"]


def run_strategy(
    data: dict,
    day_closes: dict,
    panels: dict,
    dates: list,
    names: list,
    topk: int,
    rebalance_every: int,
    warmup: int,
    initial: float = 100000.0,
    regime_index=None,
):
    """跑单个因子策略, 返回 (equity_series, broker)。

    names 传 "REGIME" 时按调仓日检测到的 regime 动态选因子集 (regime 自适应)。
    """
    broker = AShareBroker(initial_capital=initial)
    broker.deferred_execution = True  # 收盘决策 → 次日开盘成交

    equity_idx, equity_vals = [], []
    for i, T in enumerate(dates):
        Tdt = pd.Timestamp(T)
        day = Tdt.date()

        # 1) 需要价格的符号: 挂单 + 持仓
        need = {o.symbol for o in broker._pending_orders}
        need |= set(broker.account.positions.keys())
        prices = {}
        for sym in need:
            g = data.get(sym)
            if g is None or Tdt not in g.index:
                continue
            prices[sym] = bar_for(g.loc[Tdt])

        broker.set_date(day)
        broker.set_prices(prices)
        broker.execute_pending_orders()  # 次日开盘成交昨日挂单

        # 2) 收盘调仓决策
        if i >= warmup and (i - warmup) % rebalance_every == 0:
            names_cur = resolve_names(names, regime_index, Tdt)
            score = score_of(panels, Tdt, names_cur)
            closes = day_closes.get(Tdt, pd.Series(dtype=float))
            if score is not None and len(closes):
                valid = score.reindex(closes.index).dropna()
                top = valid.sort_values(ascending=False).head(topk)
                equity_now = broker.account.total_asset(closes.to_dict())
                target_notional = equity_now / max(len(top), 1)
                target_qty = {}
                for sym in top.index:
                    px = float(closes[sym])
                    qty = int(target_notional / px / 100) * 100 if px > 0 else 0
                    if qty >= 100:
                        target_qty[sym] = qty

                # 卖出: 不在目标 / 超配
                for sym, pos in list(broker.account.positions.items()):
                    if sym not in target_qty:
                        broker.sell(sym, pos.quantity)
                    elif pos.quantity > target_qty[sym]:
                        broker.sell(sym, pos.quantity - target_qty[sym])
                # 买入: 目标中低配/未持
                for sym, qty in target_qty.items():
                    pos = broker.account.positions.get(sym)
                    have = pos.quantity if pos else 0
                    if qty > have:
                        broker.buy(sym, qty - have)

        # 3) 记录日度权益 (持仓用当日收盘)
        closes = {s: float(day_closes[Tdt][s]) for s in broker.account.positions
                  if Tdt in day_closes and s in day_closes[Tdt].index}
        eq = broker.account.total_asset(closes)
        equity_idx.append(Tdt)
        equity_vals.append(eq)

    return pd.Series(equity_vals, index=equity_idx), broker


def equity_metrics(eq: pd.Series) -> dict:
    """从日度权益序列计算绩效指标。"""
    if len(eq) < 2:
        return {}
    rets = eq.pct_change().dropna()
    n = len(eq)
    years = n / 252
    total = eq.iloc[-1] / eq.iloc[0] - 1
    annual = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    return {
        "total_return": float(total),
        "annual_return": float(annual),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "n_days": n,
    }


def run(
    topk: int = 20,
    rebalance_every: int = 5,
    warmup: int = 25,
    window_end: str = "2026-07-31",
    initial: float = 100000.0,
    out_path: str = "reports/factor_backtest.md",
    replay: str = "replay_data/daily_2026-02-21_2026-07-31.parquet",
    window_start: str = None,
):
    print("加载回放全市场数据...")
    data = load_universe(window_end, replay=replay, window_start=window_start)
    dates = sorted(set().union(*[set(g.index) for g in data.values()]))
    dates = [pd.Timestamp(d) for d in dates]
    print(f"股票 {len(data)} 只 | 交易日 {len(dates)} | 窗口 "
          f"{window_start or '起点'} ~ {window_end}")

    # 每交易日收盘价 (仅可交易: close>0)
    day_closes = {}
    for T in dates:
        day_closes[T] = pd.Series({
            sym: g.loc[T, "close"] for sym, g in data.items()
            if T in g.index and pd.notna(g.loc[T, "close"]) and g.loc[T, "close"] > 0
        })

    # 注入市场指数 (rel_strength_* 需要)
    full = pd.read_parquet(replay)
    full["date"] = pd.to_datetime(full["date"])
    mkt = build_market_close_series(full)
    for sym, g in data.items():
        g["mkt_close"] = mkt.reindex(g.index)

    # 计算所需因子面板
    all_factors = sorted({f for names in STRATEGIES.values() for f in names})
    print(f"计算因子面板 ({len(all_factors)}因子)...")
    panels = compute_factor_panels(data, all_factors)

    results = {}
    curves = {}
    for name, names in STRATEGIES.items():
        print(f"回测 [{name}] top{topk} 每{rebalance_every}日调仓...")
        eq, broker = run_strategy(
            data, day_closes, panels, dates, names,
            topk, rebalance_every, warmup, initial,
        )
        m = equity_metrics(eq)
        perf = broker.get_performance()
        m["win_rate"] = perf.get("win_rate", 0)
        m["total_trades"] = perf.get("total_trades", 0)
        m["total_fees"] = perf.get("total_commission", 0) + perf.get("total_stamp_duty", 0)
        results[name] = m
        curves[name] = eq
        print(f"   总收益 {m['total_return']:.1%} | 夏普 {m['sharpe']:.2f} | "
              f"回撤 {m['max_drawdown']:.1%} | 胜率 {m['win_rate']:.0%} | 交易 {m['total_trades']}")

    # 市场等权基准 (无成本, 供参考): 每日收盘均价的归一化曲线
    closes_mean = pd.Series({T: float(day_closes[T].mean()) for T in dates if len(day_closes[T])})
    bench = closes_mean / closes_mean.iloc[0] * initial
    mkt_m = equity_metrics(bench)
    results["market"] = mkt_m
    curves["market"] = bench

    # ── 输出 ──
    print("\n" + "=" * 66)
    print(f"{'策略':<10}{'总收益':>9}{'年化':>9}{'夏普':>7}{'回撤':>8}{'胜率':>7}{'交易':>6}{'费用':>8}")
    print("-" * 66)
    for name, m in results.items():
        print(f"{name:<10}{m['total_return']:>8.1%}{m['annual_return']:>8.1%}"
              f"{m['sharpe']:>7.2f}{m['max_drawdown']:>7.1%}"
              f"{m.get('win_rate', float('nan')):>6.0%}{m.get('total_trades', 0):>6}"
              f"{m.get('total_fees', 0):>8.0f}")
    print("=" * 66)

    # 写 markdown
    if out_path:
        L = ["# 因子选股回测 — 基线 vs 扩展 vs 纯估值 vs 市场\n",
             f"窗口: {dates[warmup].date()} ~ {dates[-1].date()} "
             f"({len(dates) - warmup} 个交易日), Top-{topk} 等权, "
             f"每 {rebalance_every} 日调仓, 次日开盘成交\n",
             "| 策略 | 因子组合 | 总收益 | 年化 | 夏普 | 最大回撤 | 胜率 | 交易数 | 费用(¥) |",
             "|---|---|---|---|---|---|---|---|---|"]
        combos = {"baseline": "momentum_5+bias_ma20", "extended": "基线+ep+bp+rel_strength_5d+sharpe_20",
                  "value": "ep+bp", "rec": "ep+bp+rel_strength_5d",
                  "market": "等权买入持有(无成本)"}
        for name, m in results.items():
            L.append(f"| {name} | {combos[name]} | {m['total_return']:.1%} | {m['annual_return']:.1%} | "
                     f"{m['sharpe']:.2f} | {m['max_drawdown']:.1%} | "
                     f"{m.get('win_rate', float('nan')):.0%} | {m.get('total_trades', 0)} | "
                     f"{m.get('total_fees', 0):.0f} |")
        L.append("\n## 日度权益序列\n")
        eq_df = pd.DataFrame(curves)
        L.append("```")
        L.append(eq_df.round(0).tail(20).to_string())
        L.append("```")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("\n".join(L) + "\n", encoding="utf-8")
        # 权益序列 CSV
        eq_df.to_csv(Path(out_path).with_suffix(".csv"), float_format="%.0f")
        print(f"\n回测报告: {out_path}")
        print(f"权益序列: {Path(out_path).with_suffix('.csv')}")

    return results, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--rebalance", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--window-end", type=str, default="2026-07-31")
    ap.add_argument("--capital", type=float, default=100000.0)
    ap.add_argument("--output", type=str, default="reports/factor_backtest.md")
    ap.add_argument("--replay", type=str,
                    default="replay_data/daily_2026-02-21_2026-07-31.parquet",
                    help="回放文件 (可用长窗口多regime)")
    ap.add_argument("--window-start", type=str, default=None,
                    help="可选窗口起点 (验证牛市/熊市单段)")
    args = ap.parse_args()
    run(topk=args.topk, rebalance_every=args.rebalance, warmup=args.warmup,
        window_end=args.window_end, initial=args.capital, out_path=args.output,
        replay=args.replay, window_start=args.window_start)


if __name__ == "__main__":
    main()
