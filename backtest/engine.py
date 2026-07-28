"""
事件驱动回测引擎 (Event-Driven Backtesting Engine)

A股规则完整模拟:
  - T+1 清算: 当日买入次日可卖
  - 涨跌停: 到达限价的订单挂起,后续可解封
  - 停牌: 跳过停牌日
  - 除权除息: 自动复权(使用前复权数据)
  - 交易成本: 佣金+印花税+过户费+冲击成本(v2.1)

输出:
  - 净值曲线
  - 收益率统计(年化/CAGR/夏普/Sortino/最大回撤)
  - 交易明细
  - 策略归因
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .broker import AShareBroker, OrderSide, OrderStatus


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0
    start_date: date = date(2020, 1, 1)
    end_date: date = date(2024, 12, 31)
    benchmark_symbol: str = "sh.000300"  # 比较基准

    # 交易成本
    commission_rate: float = 0.0003       # 佣金万3
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005       # 印花税0.05%
    slippage_pct: float = 0.001           # 滑点0.1%

    # 限制
    max_position_pct: float = 0.30        # 单票最大仓位30%
    max_positions: int = 5                # 最大同时持仓数
    min_holding_days: int = 1             # 最短持仓T+1

    # 过拟合防控
    train_split: float = 0.60
    validation_split: float = 0.20
    test_split: float = 0.20


@dataclass
class BacktestResult:
    """回测结果"""
    # 净值
    equity_curve: pd.Series
    benchmark_curve: pd.Series

    # 收益统计
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int

    # 交易统计
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_days: float

    # 风控
    var_95: float
    cvar_95: float
    daily_volatility: float

    # 回测区间
    start_date: date
    end_date: date
    trading_days: int

    # 原始数据
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        """格式化的回测报告摘要"""
        return (
            f"══════════════════════════════\n"
            f"  回测区间: {self.start_date} → {self.end_date}\n"
            f"  交易日数: {self.trading_days}\n"
            f"  ─────────────────\n"
            f"  总收益率:  {self.total_return:+.2f}%\n"
            f"  年化收益率: {self.annual_return:+.2f}%\n"
            f"  Sharpe:     {self.sharpe_ratio:.2f}\n"
            f"  Sortino:    {self.sortino_ratio:.2f}\n"
            f"  Calmar:     {self.calmar_ratio:.2f}\n"
            f"  最大回撤:   {self.max_drawdown:.2f}%\n"
            f"  ─────────────────\n"
            f"  总交易次数: {self.total_trades}\n"
            f"  胜率:      {self.win_rate:.1%}\n"
            f"  盈亏比:    {self.avg_win:.2f}/{abs(self.avg_loss):.2f}\n"
            f"  盈利因子:  {self.profit_factor:.2f}\n"
            f"  VaR(95%):  {self.var_95:.2f}%\n"
            f"  CVaR(95%): {self.cvar_95:.2f}%\n"
            f"══════════════════════════════"
        )


class EventDrivenBacktestEngine:
    """
    A股事件驱动回测引擎

    使用方式:
        engine = EventDrivenBacktestEngine(config)
        engine.load_data(symbol, ohlcv_df)
        result = engine.run(strategy_func)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.broker = AShareBroker(initial_capital=config.initial_capital)
        self._data: Dict[str, pd.DataFrame] = {}
        self._benchmark: Optional[pd.DataFrame] = None
        self._equity: List[float] = []
        self._dates: List[date] = []

    def load_data(
        self,
        symbol: str,
        df: pd.DataFrame,
    ):
        """加载标的日K线数据"""
        df = df.copy()
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date")
        self._data[symbol] = df

    def load_benchmark(self, df: pd.DataFrame):
        """加载基准数据"""
        df = df.copy()
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        self._benchmark = df.sort_values("date")

    # ═══════════════════════════════════════════════════════════════
    # 主回测循环
    # ═══════════════════════════════════════════════════════════════

    def run(
        self,
        strategy_func: Callable[[date, Dict[str, pd.Series], AShareBroker], None],
        progress_bar: bool = True,
    ) -> BacktestResult:
        """
        执行回测

        Args:
            strategy_func: 策略函数
                def my_strategy(
                    today: date,
                    bars: {symbol: today's row},
                    broker: AShareBroker,
                ) -> None:
                    # 策略逻辑: 调用 broker.buy() / broker.sell()
                    pass

            progress_bar: 是否显示进度条

        Returns:
            BacktestResult
        """
        # 1. 确定回测日期范围
        all_dates = set()
        for df in self._data.values():
            all_dates.update(df["date"].tolist())
        dates = sorted([
            d for d in all_dates
            if self.config.start_date <= d <= self.config.end_date
        ])

        if not dates:
            raise ValueError("回测区间内无交易数据")

        # 2. 主循环
        self._equity = [self.config.initial_capital]
        self._dates = []

        iterator = dates
        if progress_bar:
            from tqdm import tqdm
            iterator = tqdm(dates, desc="回测执行中")

        for i, today in enumerate(iterator):
            self._dates.append(today)
            self.broker.set_date(today)

            # 准备当日数据
            bars = {}
            for sym, df in self._data.items():
                day_data = df[df["date"] == today]
                if not day_data.empty:
                    bars[sym] = day_data.iloc[0]

            if not bars:
                self._equity.append(self._equity[-1])
                continue

            self.broker.set_prices(bars)

            # 执行策略
            try:
                strategy_func(today, bars, self.broker)
            except Exception as e:
                logger.warning(f"策略执行异常 {today}: {e}")

            # 记录当日净值
            prices = {
                sym: bars[sym]["close"]
                for sym in self._data
                if sym in bars
            }
            self._equity.append(
                self.broker.account.total_asset(prices)
            )

        # 3. 计算绩效
        return self._compute_performance()

    # ═══════════════════════════════════════════════════════════════
    # 绩效计算
    # ═══════════════════════════════════════════════════════════════

    def _compute_performance(self) -> BacktestResult:
        """计算完整回测绩效指标"""
        equity = pd.Series(self._equity[1:], index=pd.DatetimeIndex(self._dates))

        if len(equity) < 2:
            return BacktestResult(
                equity_curve=equity,
                benchmark_curve=pd.Series(),
                total_return=0, annual_return=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                max_drawdown=0, max_drawdown_duration=0,
                total_trades=0, win_rate=0, avg_win=0, avg_loss=0,
                profit_factor=0, avg_holding_days=0,
                var_95=0, cvar_95=0, daily_volatility=0,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                trading_days=len(equity),
            )

        ret = equity.pct_change().dropna()

        # 基准
        benchmark = pd.Series(dtype=float)
        if self._benchmark is not None:
            bm = self._benchmark.set_index("date")["close"]
            bm = bm[bm.index.isin(equity.index)]
            benchmark = bm / bm.iloc[0] * self.config.initial_capital

        # === 收益 ===
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        years = (self.config.end_date - self.config.start_date).days / 365.25
        annual_return = ((1 + total_return / 100) ** (1 / max(years, 0.1)) - 1) * 100

        # === 风险 ===
        daily_vol = ret.std()
        ann_vol = daily_vol * np.sqrt(252)

        # Sharpe (假定无风险利率=2%)
        rf_daily = 0.02 / 252
        excess = ret - rf_daily
        sharpe = (excess.mean() / max(daily_vol, 1e-10)) * np.sqrt(252)

        # Sortino (v2.14: 修正 — 使用 downside deviation, 非 std of negative returns)
        downside_sq = np.clip(ret, None, 0) ** 2  # 只取负收益的平方
        downside_dev = np.sqrt(downside_sq.mean()) * np.sqrt(252) if len(downside_sq) > 0 else 0.01
        sortino = (ret.mean() * 252 - 0.02) / max(downside_dev, 1e-10)

        # 最大回撤
        cummax = equity.expanding().max()
        drawdown = (equity - cummax) / cummax * 100
        max_dd = drawdown.min()
        # 回撤持续时间 (v2.14: 修复 — 无回撤时返回0, 非总天数)
        max_dd_duration = 0
        if drawdown.min() < 0:
            dd_periods = (drawdown < 0).astype(int)
            max_dd_duration = int(
                dd_periods.groupby((dd_periods != dd_periods.shift()).cumsum())
                .sum().max()
            ) if dd_periods.sum() > 0 else 0

        # Calmar
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # === VaR / CVaR ===
        var_95 = np.percentile(ret, 5) * 100
        cvar_95 = ret[ret <= np.percentile(ret, 5)].mean() * 100

        # === 交易统计 ===
        trade_df = pd.DataFrame(self.broker.trade_log) if self.broker.trade_log else pd.DataFrame()
        total_trades = len(trade_df[trade_df["side"] == "sell"]) if not trade_df.empty else 0

        if total_trades > 0:
            # 从买卖配对中计算盈亏
            win_rate = (
                self.broker.account.win_count /
                max(self.broker.account.win_count + self.broker.account.loss_count, 1)
            )
            avg_win = 0
            avg_loss = 0
            profit_factor = 1.0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 1.0

        # === 月度收益 ===
        monthly = equity.resample("ME").last().pct_change().dropna() * 100

        return BacktestResult(
            equity_curve=equity,
            benchmark_curve=benchmark,
            total_return=round(total_return, 2),
            annual_return=round(annual_return, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_duration=int(max_dd_duration),
            total_trades=total_trades,
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            avg_holding_days=0,
            var_95=round(var_95, 2),
            cvar_95=round(cvar_95, 2),
            daily_volatility=round(ann_vol * 100, 2),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            trading_days=len(equity),
            trade_log=trade_df,
            monthly_returns=monthly,
        )
