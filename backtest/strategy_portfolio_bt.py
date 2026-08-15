"""
策略组合级事件驱动回测 — 把 signal-level 回测器的信号接入 EventDrivenBacktestEngine。

背景 (健康体检 P0-2): `strategy_backtest.py` 用简化回测器按「单票」聚合信号排名,
完整的 `AShareBroker` (T+1 / 涨跌停 / 费率 / 滑点 / 共享资金 / 持仓上限) 从未参与策略评级。
本模块把每个信号策略的逐 bar entry/exit 信号抽出, 灌进事件驱动引擎做「组合级」回测,
让真实券商摩擦与组合约束 (共享资金 + max_positions) 参与选策略。

信号来源: 各回测器 (recommender._backtest_* / strategies_v3.*_v3 / optimized_strategies.*)
现额外返回 `entry_sig` / `exit_sig` (bar index 列表, 信号检测时点 = T日收盘)。
引擎 `deferred_execution=True` → T日信号 → T+1 开盘成交 (与简化回测器同口径)。

与简化回测器的口径差异 (如实披露, 见 reports/model_advantage_roadmap.md 阶段0):
  1. 涨跌停阈值: 引擎用 broker 的 limit-0.2, 简化回测器 `_exec_prices` 用 limit-0.7 (更保守)。
  2. 卖出封板重试: 引擎每交易日重试被拒的卖出直到成交; 简化回测器对一次性交叉信号不重试。
     引擎口径更贴近真实交易 (封板打开即卖)。
  3. 组合级: 共享资金 + 持仓上限, 简化回测器每票独立虚拟资金、无上限。

诚实边界: 这是「信号级 → 组合级」的第二个意见 (second opinion), 不替换信号级排名,
只补上组合摩擦维度。幸存者偏差与信号级同源 (股票池=当前上市快照)。
"""
from __future__ import annotations

from datetime import date
from typing import Callable, Dict, Optional, Set

import numpy as np
import pandas as pd
from loguru import logger

from .engine import BacktestConfig, BacktestResult, EventDrivenBacktestEngine


def _bar_close(bar) -> float:
    """从当日 bar (pd.Series 或 dict) 安全取收盘价"""
    try:
        if isinstance(bar, pd.Series):
            return float(bar.get("close", 0.0))
        if isinstance(bar, dict):
            return float(bar.get("close") or 0.0)
        return float(bar)
    except (TypeError, ValueError):
        return 0.0


def _extract_signal_dates(
    bt_func: Callable, df: pd.DataFrame, symbol: str
) -> tuple[Set[date], Set[date]]:
    """跑一次信号回测器, 把 entry_sig/exit_sig (bar index) 映射为日期集合。

    返回 (entry_dates, exit_dates), 元素为 datetime.date。
    """
    entry_dates: Set[date] = set()
    exit_dates: Set[date] = set()
    try:
        res = bt_func(df, symbol=symbol)
        if not isinstance(res, dict):
            return entry_dates, exit_dates
        if "date" not in df.columns:
            return entry_dates, exit_dates
        dts = pd.to_datetime(df["date"]).dt.date.tolist()
        n = len(dts)
        for i in res.get("entry_sig", []) or []:
            if isinstance(i, (int, np.integer)) and 0 <= i < n:
                entry_dates.add(dts[i])
        for i in res.get("exit_sig", []) or []:
            if isinstance(i, (int, np.integer)) and 0 <= i < n:
                exit_dates.add(dts[i])
    except Exception as e:  # noqa: BLE001 — 信号抽取失败不阻断组合级评级
        logger.warning(f"信号抽取失败 {symbol}: {e}")
    return entry_dates, exit_dates


def _make_portfolio_strategy(
    entry_dates: Dict[str, Set[date]],
    exit_dates: Dict[str, Set[date]],
    config: BacktestConfig,
) -> Callable:
    """构造事件驱动引擎的 strategy_func — 组合级 (共享资金 + max_positions)。"""
    max_pos = max(1, config.max_positions)
    capital = max(config.initial_capital, 1.0)
    pending_exits: Set[str] = set()

    def strategy_func(today: date, bars: Dict[str, pd.Series], broker) -> None:
        # 1. 重试被拒的卖出 (封板卖不出 → 每交易日重试直到成交)
        for sym in list(pending_exits):
            pos = broker.account.positions.get(sym)
            if pos is None or pos.quantity <= 0:
                pending_exits.discard(sym)
            else:
                broker.sell(sym, pos.quantity)

        # 2. 当日新信号 (先卖后买; 买入受持仓上限约束)
        for sym, bar in bars.items():
            pos = broker.account.positions.get(sym)
            holding = pos is not None and pos.quantity > 0
            if today in exit_dates.get(sym, ()) and holding:
                broker.sell(sym, pos.quantity)
                pending_exits.add(sym)
            elif today in entry_dates.get(sym, ()) and not holding:
                if len(broker.account.positions) >= max_pos:
                    continue
                close = _bar_close(bar)
                if close <= 0:
                    continue
                qty = int(capital / max_pos / close / 100) * 100
                if qty >= 100:
                    broker.buy(sym, qty)

    return strategy_func


def run_portfolio_event_backtest(
    bt_func: Callable,
    stock_dfs: Dict[str, pd.DataFrame],
    config: BacktestConfig,
) -> Optional[BacktestResult]:
    """对单个策略跑组合级事件驱动回测。

    Args:
        bt_func: 信号回测器 (df, symbol=...) -> dict (含 entry_sig/exit_sig)
        stock_dfs: {symbol: 日K DataFrame} (需含 date 列)
        config: 回测配置 (initial_capital / max_positions / start/end / 费率)

    Returns:
        BacktestResult 或 None (无有效信号/数据不足)
    """
    entry_dates: Dict[str, Set[date]] = {}
    exit_dates: Dict[str, Set[date]] = {}
    valid = False
    for sym, df in stock_dfs.items():
        if df is None or len(df) < 30 or "date" not in df.columns:
            continue
        e, x = _extract_signal_dates(bt_func, df, sym)
        entry_dates[sym] = e
        exit_dates[sym] = x
        if e or x:
            valid = True

    if not valid:
        return None

    engine = EventDrivenBacktestEngine(config)
    for sym, df in stock_dfs.items():
        if df is None or len(df) < 30 or "date" not in df.columns:
            continue
        engine.load_data(sym, df)

    strategy_func = _make_portfolio_strategy(entry_dates, exit_dates, config)
    return engine.run(strategy_func, progress_bar=False)


def summarize(result: BacktestResult) -> Dict:
    """把 BacktestResult 压缩成可 JSON 序列化的摘要"""
    return {
        "total_return_pct": result.total_return,
        "annual_return_pct": result.annual_return,
        "sharpe": result.sharpe_ratio,
        "sortino": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "trading_days": result.trading_days,
    }
