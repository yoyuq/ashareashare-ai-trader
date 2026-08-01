#!/usr/bin/env python3
"""
回测对比: 旧版风控 vs 新版(Kelly+ATR+RSI+Regime)

验证 v2.11-2.13 改动是否真的改进性能。
使用 Baostock 历史数据 + 策略回测结果中的胜率/夏普。

使用:
    python scripts/backtest_compare.py                    # 默认3年回测
    python scripts/backtest_compare.py --years 5          # 5年回测
    python scripts/backtest_compare.py --symbols 5        # 只测5只
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def fetch_historical_data(symbols: List[str], years: int = 3) -> Dict[str, pd.DataFrame]:
    """从 Baostock 拉取历史日K线"""
    import baostock as bs

    bs.login()
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    data = {}
    for sym in symbols:
        try:
            # Baostock 需要完整代码: sh.600519
            code = sym  # 保持 sh.600519 格式
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume",
                start_date=start, end_date=end,
                frequency="d", adjustflag="2",
            )
            if rs.error_code == "0":
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    df = df.dropna()
                    if len(df) > 100:
                        data[sym] = df
                        print(f"  {sym}: {len(df)} rows")
        except Exception as e:
            print(f"  {sym}: error - {e}")
    bs.logout()
    return data


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    n = len(close)

    # RSI(14)
    rsi = np.full(n, 50.0)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    for i in range(14, n):
        avg_gain = np.mean(gain[i - 14:i])
        avg_loss = np.mean(loss[i - 14:i])
        if avg_loss > 0:
            rs_val = avg_gain / avg_loss
            rsi[i] = 100 - 100 / (1 + rs_val)

    # ATR(14)
    atr = np.full(n, np.nan)
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    for i in range(13, n):
        atr[i] = np.mean(tr[i - 13:i + 1])

    # Trend score (price vs MA20)
    ma20 = pd.Series(close).rolling(20).mean().values
    trend_score = np.where(close > ma20, 0.5, -0.5)
    # Strengthen with recent momentum
    mom5 = np.zeros(n)
    mom5[5:] = close[5:] / close[:-5] - 1
    trend_score = trend_score + np.clip(mom5 * 5, -0.3, 0.3)

    df = df.copy()
    df["rsi_14"] = rsi
    df["atr_14"] = atr
    df["trend_score"] = np.clip(trend_score, -1, 1)
    df["ma20"] = ma20
    df["close"] = close
    return df


# ═══════════════════════════════════════════════════════════════
# 旧版风控 (v2.10 之前)
# ═══════════════════════════════════════════════════════════════

def old_stop_loss(price: float, score: float) -> Tuple[float, float]:
    """旧版固定百分比止损"""
    if score >= 80:
        sl, tp = price * 0.95, price * 1.12
    elif score >= 50:
        sl, tp = price * 0.93, price * 1.10
    else:
        sl, tp = price * 0.90, price * 1.08
    return round(sl, 2), round(tp, 2)


def old_position_pct() -> float:
    """旧版固定仓位"""
    return 0.20


def old_max_positions(_regime: str = "range_bound") -> int:
    """旧版固定最大持仓"""
    return 10


def old_rsi_filter(rsi: float) -> bool:
    """旧版无RSI过滤"""
    return False  # 从不跳过


# ═══════════════════════════════════════════════════════════════
# 新版风控 (v2.13)
# ═══════════════════════════════════════════════════════════════

def new_stop_loss(price: float, atr: float, regime: str,
                  rsi: float, _score: float = 0) -> Tuple[float, float]:
    """新版ATR自适应止损"""
    from scripts.shared import calc_atr_stop_loss
    return calc_atr_stop_loss(price, atr, regime, rsi)


def new_position_pct(win_rate: float, conviction: float, sharpe: float,
                     regime_single_pct: float = 0.20) -> float:
    """新版凯利公式仓位"""
    from scripts.shared import calc_kelly_position_pct
    kelly = calc_kelly_position_pct(win_rate, conviction, sharpe=sharpe)
    return min(kelly, regime_single_pct) if kelly > 0 else regime_single_pct


REGIME_LIMITS = {
    "strong_bull": 10, "weak_bull": 8, "range_bound": 6,
    "weak_bear": 3, "strong_bear": 1, "crisis": 1,
}


def new_max_positions(regime: str) -> int:
    """新版市场自适应仓位上限"""
    return REGIME_LIMITS.get(regime, 6)


def new_rsi_filter(rsi: float, trend: float) -> bool:
    """新版RSI超买过滤"""
    from scripts.shared import check_rsi_filter
    skip, _ = check_rsi_filter(rsi, trend)
    return skip


# ═══════════════════════════════════════════════════════════════
# 简易回测
# ═══════════════════════════════════════════════════════════════

def simulate(old_mode: bool, data: Dict[str, pd.DataFrame],
             years: int = 3) -> Dict:
    """在历史数据上模拟买入持有+止盈止损"""
    regime = "range_bound"
    initial_capital = 100000.0
    cash = initial_capital
    positions = {}  # symbol → {entry, shares, sl, tp, entry_date}

    all_trades = []
    daily_equity = []

    # 获取所有交易日
    all_dates = set()
    for df in data.values():
        all_dates.update(df["date"].tolist())
    dates = sorted(all_dates)

    for i, date_str in enumerate(dates):
        # 检查持仓止损止盈
        closed = []
        for sym, pos in positions.items():
            df = data.get(sym)
            if df is None:
                continue
            row = df[df["date"] == date_str]
            if row.empty:
                continue
            price = float(row["close"].iloc[0])
            entry = pos["entry"]
            sl = pos["sl"]
            tp = pos["tp"]

            # 检查卖出
            sell_reason = None
            if price <= sl:
                sell_reason = "stop_loss"
            elif price >= tp:
                sell_reason = "take_profit"
            elif pos["holding_days"] > 60:
                sell_reason = "time_exit"  # 最多持有60天

            if sell_reason:
                pnl = (price - entry) * pos["shares"]
                pnl_pct = (price / entry - 1) * 100
                cash += price * pos["shares"]
                all_trades.append({
                    "symbol": sym, "entry": entry, "exit": price,
                    "pnl": pnl, "pnl_pct": pnl_pct, "reason": sell_reason,
                    "holding_days": pos["holding_days"],
                })
                closed.append(sym)
            else:
                pos["holding_days"] += 1

        for sym in closed:
            del positions[sym]

        # 检查买入信号 (每20个交易日检查)
        if len(positions) < (new_max_positions(regime) if not old_mode else old_max_positions(regime)):
            if i % 20 == 0 and cash > initial_capital * 0.1:
                for sym, df in list(data.items())[:20]:
                    if sym in positions:
                        continue
                    row = df[df["date"] == date_str]
                    if row.empty:
                        continue
                    price = float(row["close"].iloc[0])
                    # v3.1 修复: 用 row.iloc[0] 而非 label 索引, 避免指标 df 索引与位置不对齐导致越界
                    atr_val = float(row["atr_14"].iloc[0]) if "atr_14" in row.columns and not pd.isna(row["atr_14"].iloc[0]) else price * 0.03
                    rsi = float(row["rsi_14"].iloc[0]) if "rsi_14" in row.columns and not pd.isna(row["rsi_14"].iloc[0]) else 50
                    trend = float(row["trend_score"].iloc[0]) if "trend_score" in row.columns and not pd.isna(row["trend_score"].iloc[0]) else 0.5

                    if pd.isna(price) or price <= 0:
                        continue
                    if pd.isna(atr_val) or atr_val <= 0:
                        atr_val = price * 0.03

                    if old_mode:
                        if old_rsi_filter(rsi):
                            continue
                        max_pos = old_max_positions(regime)
                        pct = old_position_pct()
                        sl, tp = old_stop_loss(price, 70)
                    else:
                        if new_rsi_filter(rsi, trend):
                            continue
                        max_pos = new_max_positions(regime)
                        # v2.14: 使用实际回测胜率和夏普 (而非硬编码0.55/0.60/1.5)
                        # 对9个策略运行回测取平均胜率
                        from analysis.recommender import STRATEGY_BACKTESTERS
                        wr_vals = [bt_func(df).get("win_rate", 0.5) for bt_func in STRATEGY_BACKTESTERS.values()]
                        sr_vals = [bt_func(df).get("sharpe", 1.0) for bt_func in STRATEGY_BACKTESTERS.values()]
                        avg_wr = np.mean(wr_vals) if wr_vals else 0.55
                        avg_sr = np.mean(sr_vals) if sr_vals else 1.5
                        pct = new_position_pct(avg_wr, 0.60, avg_sr)
                        sl, tp = new_stop_loss(price, atr_val, regime, rsi)

                    if len(positions) >= max_pos:
                        break

                    # 买入
                    max_amount = cash * pct
                    shares = int(max_amount / price / 100) * 100
                    if shares < 100:
                        continue
                    cost = shares * price
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[sym] = {
                        "entry": price, "shares": shares,
                        "sl": sl, "tp": tp, "holding_days": 1,
                    }

        # 记录每日权益
        pos_value = sum(
            positions[p]["shares"] * float(
                data[p][data[p]["date"] == date_str]["close"].iloc[0]
            )
            for p in positions
            if not data[p][data[p]["date"] == date_str].empty
        )
        daily_equity.append(cash + pos_value)

    # 清算
    final_equity = daily_equity[-1] if daily_equity else initial_capital

    # 统计
    returns = pd.Series(daily_equity).pct_change().dropna()
    total_ret = (final_equity / initial_capital - 1) * 100
    annual_ret = total_ret / years
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    max_dd = float(((pd.Series(daily_equity).cummax() - pd.Series(daily_equity)) / pd.Series(daily_equity).cummax()).max() * 100)

    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    total_trades = len(all_trades)
    avg_win = np.mean([t["pnl_pct"] for t in all_trades if t["pnl"] > 0]) if wins > 0 else 0
    avg_loss = np.mean([t["pnl_pct"] for t in all_trades if t["pnl"] <= 0]) if (total_trades - wins) > 0 else 0
    profit_factor = abs(sum(t["pnl"] for t in all_trades if t["pnl"] > 0) / sum(t["pnl"] for t in all_trades if t["pnl"] < 0)) if sum(t["pnl"] for t in all_trades if t["pnl"] < 0) != 0 else float("inf")

    return {
        "final_equity": round(final_equity, 2),
        "total_return": round(total_ret, 2),
        "annual_return": round(annual_ret, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "total_trades": total_trades,
        "win_rate": round(wins / total_trades * 100, 1) if total_trades > 0 else 0,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 99.9,
    }


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="回测对比: 旧版 vs 新版风控")
    parser.add_argument("--years", type=int, default=3, help="回测年数")
    parser.add_argument("--symbols", type=int, default=10, help="标的数量")
    args = parser.parse_args()

    # 精选标的 (各行业代表)
    test_symbols = [
        "sh.600519",  # 茅台
        "sh.600036",  # 招商银行
        "sz.000333",  # 美的集团
        "sz.300750",  # 宁德时代
        "sh.601318",  # 中国平安
        "sh.601088",  # 中国神华
        "sz.002594",  # 比亚迪
        "sh.600900",  # 长江电力
        "sz.002415",  # 海康威视
        "sh.600276",  # 恒瑞医药
        "sh.601899",  # 紫金矿业
        "sz.000858",  # 五粮液
        "sh.600031",  # 三一重工
        "sz.002027",  # 分众传媒
        "sh.601225",  # 陕西煤业
    ][:args.symbols]

    print(f"=== 回测对比: 旧版 vs 新版风控 ===")
    print(f"标的: {len(test_symbols)}只 | 回测年限: {args.years}年")
    print()

    # 拉取数据
    print("正在从 Baostock 拉取历史数据...")
    data = fetch_historical_data(test_symbols, args.years)
    print(f"成功获取: {len(data)}只")

    if len(data) < 3:
        print("数据不足, 无法回测")
        return

    # 计算指标
    print("计算技术指标...")
    for sym in list(data.keys()):
        try:
            data[sym] = compute_indicators(data[sym])
        except Exception as e:
            print(f"  {sym}: 指标计算失败 - {e}")
            del data[sym]

    # 回测
    print(f"\n回测中 (旧版: 固定20%仓位, 固定止损, 无RSI过滤)...")
    old_result = simulate(old_mode=True, data=data, years=args.years)

    print(f"回测中 (新版: Kelly仓位, ATR止损, RSI过滤, 市场自适应)...")
    new_result = simulate(old_mode=False, data=data, years=args.years)

    # 对比输出
    print("\n" + "=" * 70)
    print(f"{'指标':<20} {'旧版 v2.10':>12} {'新版 v2.13':>12} {'改进':>12}")
    print("-" * 70)

    metrics = [
        ("总收益率(%)", "total_return"),
        ("年化收益(%)", "annual_return"),
        ("夏普比率", "sharpe"),
        ("最大回撤(%)", "max_drawdown"),
        ("总交易次数", "total_trades"),
        ("胜率(%)", "win_rate"),
        ("平均盈利(%)", "avg_win"),
        ("平均亏损(%)", "avg_loss"),
        ("盈利因子", "profit_factor"),
        ("最终权益", "final_equity"),
    ]

    for label, key in metrics:
        old_val = old_result[key]
        new_val = new_result[key]
        if isinstance(old_val, float):
            if key in ("total_return", "annual_return", "sharpe", "win_rate",
                       "avg_win", "profit_factor", "final_equity"):
                improvement = new_val - old_val
                arrow = "+" if improvement > 0 else ""
            elif key in ("max_drawdown", "avg_loss"):
                improvement = old_val - new_val  # 负值是改进
                arrow = "+" if improvement > 0 else ""
            else:
                improvement = new_val - old_val
                arrow = "+" if improvement > 0 else ""
            print(f"{label:<20} {old_val:>12.2f} {new_val:>12.2f} {arrow}{improvement:>11.2f}")
        else:
            print(f"{label:<20} {old_val!s:>12} {new_val!s:>12}")
    print("=" * 70)

    # 结论
    print()
    old_sharpe = old_result["sharpe"]
    new_sharpe = new_result["sharpe"]
    old_dd = old_result["max_drawdown"]
    new_dd = new_result["max_drawdown"]

    if new_sharpe > old_sharpe and new_dd < old_dd:
        print("结论: 新版风控全面优于旧版 (更高夏普 + 更低回撤)")
    elif new_sharpe > old_sharpe:
        print("结论: 新版风控夏普更高, 但回撤未必更优")
    elif new_dd < old_dd:
        print("结论: 新版风控回撤更低, 但夏普未必更优")
    else:
        print("结论: 两版差异不大, 需更多数据验证")

    # 🆕 v2.14: 过拟合检测
    print(f"\n--- 过拟合检测 (OverfittingGuard) ---")
    try:
        from backtest.overfitting import OverfittingGuard
        guard = OverfittingGuard(n_simulations=500)

        # 用新版的日收益率序列
        new_daily = pd.Series(new_result.get("_daily_equity", [])).pct_change().dropna() if "_daily_equity" in new_result else None

        if new_daily is not None and len(new_daily) > 50:
            report = guard.evaluate(new_daily)
            print(f"  PBO: {report.pbo:.3f} "
                  f"({'⚠️ 高风险' if report.pbo > 0.1 else '✅ 通过'})")
            print(f"  Deflated Sharpe: {report.deflated_sharpe:.3f} "
                  f"(p={report.deflated_sharpe_pvalue:.4f})")
            print(f"  Monte Carlo p-value: {report.monte_carlo_pvalue:.4f}")
            print(f"  综合判定: {report.verdict()}")
        else:
            print("  跳过 (数据不足)")
    except Exception as e:
        print(f"  过拟合检测跳过: {e}")


if __name__ == "__main__":
    main()
