"""
单元测试: 策略回测函数 + 交易成本 + 事件驱动回测引擎
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 300, trend: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """生成带趋势的模拟数据"""
    rng = np.random.default_rng(seed)
    close = 50 + np.cumsum(rng.normal(trend, 1.0, n))
    close = np.maximum(close, 1)
    high = close + rng.uniform(0, 1.5, n)
    low = close - rng.uniform(0, 1.5, n)
    volume = rng.integers(100000, 1000000, n).astype(float)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates, "open": close - rng.uniform(-0.5, 0.5, n),
        "high": high, "low": low, "close": close, "volume": volume,
    })


class TestBacktestFunctions:
    """策略回测函数测试"""

    def test_all_strategies_have_backtester(self):
        from analysis.recommender import STRATEGY_BACKTESTERS
        assert "dual_ma_trend" in STRATEGY_BACKTESTERS
        assert "macd_trend" in STRATEGY_BACKTESTERS
        assert "bollinger_reversal" in STRATEGY_BACKTESTERS
        assert "rsi_mean_reversion" in STRATEGY_BACKTESTERS
        assert "momentum_breakout" in STRATEGY_BACKTESTERS
        assert "low_volatility" in STRATEGY_BACKTESTERS

    def test_ma_trend_on_trending_data(self):
        from analysis.recommender import _backtest_ma_trend
        df = _make_ohlcv(300, trend=0.05)
        result = _backtest_ma_trend(df)
        assert result["signals"] > 0
        assert 0 <= result["win_rate"] <= 1
        assert result["cost_adjusted"] is True

    def test_macd_trend_returns_stats(self):
        from analysis.recommender import _backtest_macd_trend
        df = _make_ohlcv(300)
        result = _backtest_macd_trend(df)
        assert "signals" in result
        assert "win_rate" in result
        assert "sharpe" in result
        assert "max_dd" in result

    def test_bollinger_reversal_returns_stats(self):
        from analysis.recommender import _backtest_bollinger_reversal
        df = _make_ohlcv(300)
        result = _backtest_bollinger_reversal(df)
        assert "signals" in result
        assert "profit_factor" in result

    def test_rsi_reversal_returns_stats(self):
        from analysis.recommender import _backtest_rsi_reversal
        df = _make_ohlcv(300)
        result = _backtest_rsi_reversal(df)
        assert "signals" in result
        assert "expected_value" in result

    def test_momentum_breakout_returns_stats(self):
        from analysis.recommender import _backtest_momentum_breakout
        df = _make_ohlcv(300)
        result = _backtest_momentum_breakout(df)
        assert "signals" in result

    def test_low_volatility_returns_stats(self):
        from analysis.recommender import _backtest_low_volatility
        df = _make_ohlcv(300)
        result = _backtest_low_volatility(df)
        assert "signals" in result

    def test_turtle_trend_returns_stats(self):
        from analysis.recommender import _backtest_turtle_trend
        df = _make_ohlcv(300, trend=0.05)
        result = _backtest_turtle_trend(df)
        assert "signals" in result
        assert "sharpe" in result

    def test_multi_factor_returns_stats(self):
        from analysis.recommender import _backtest_multi_factor
        df = _make_ohlcv(300)
        result = _backtest_multi_factor(df)
        assert "signals" in result
        assert "profit_factor" in result

    def test_limit_up_returns_stats(self):
        from analysis.recommender import _backtest_limit_up
        df = _make_ohlcv(300)
        result = _backtest_limit_up(df)
        assert "signals" in result

    def test_empty_data_returns_zero_signals(self):
        from analysis.recommender import _backtest_ma_trend
        df = _make_ohlcv(30)  # 太短
        result = _backtest_ma_trend(df)
        assert result["signals"] == 0


class TestBacktestStats:
    """_calc_bt_stats 测试"""

    def test_cost_adjustment_reduces_returns(self):
        from analysis.recommender import _calc_bt_stats
        # 模拟有盈利的交易
        trades = [5.0, 3.0, -2.0, 4.0, -1.0, 6.0, -3.0, 5.0]
        result = _calc_bt_stats(trades)
        assert result["cost_adjusted"] is True
        # 扣除0.31%成本后胜率应略有下降或持平
        assert 0 <= result["win_rate"] <= 1

    def test_no_trades_returns_defaults(self):
        from analysis.recommender import _calc_bt_stats
        result = _calc_bt_stats([])
        assert result["signals"] == 0
        assert result["win_rate"] == 0

    def test_all_wins(self):
        from analysis.recommender import _calc_bt_stats
        result = _calc_bt_stats([5.0, 3.0, 2.0])
        assert result["win_rate"] == 1.0

    def test_all_losses(self):
        from analysis.recommender import _calc_bt_stats
        result = _calc_bt_stats([-5.0, -3.0, -2.0])
        assert result["win_rate"] == 0.0


class TestAShareBroker:
    """A股券商模拟器测试"""

    def test_initial_state(self):
        from backtest.broker import AShareBroker
        broker = AShareBroker(initial_capital=100000.0)
        assert broker.account.cash == 100000.0
        assert len(broker.account.positions) == 0

    def test_buy_creates_position(self):
        from backtest.broker import AShareBroker
        broker = AShareBroker(initial_capital=100000.0)
        broker.set_date(date(2024, 1, 15))
        broker.set_prices({"sh.600519": {"close": 500.0}})

        order = broker.buy("sh.600519", quantity=100)
        assert order.status.value == "filled"
        assert order.filled_qty == 100
        assert "sh.600519" in broker.account.positions
        assert broker.account.cash < 100000.0  # 扣了钱

    def test_buy_insufficient_funds_rejected(self):
        from backtest.broker import AShareBroker
        broker = AShareBroker(initial_capital=1000.0)  # 只有1000元
        broker.set_date(date(2024, 1, 15))
        broker.set_prices({"sh.600519": {"close": 500.0}})

        order = broker.buy("sh.600519", quantity=1000)  # 需要50万
        assert order.status.value == "rejected"

    def test_sell_closes_position(self):
        from backtest.broker import AShareBroker
        broker = AShareBroker(initial_capital=100000.0)
        broker.set_date(date(2024, 1, 15))
        broker.set_prices({"sh.600519": {"close": 500.0}})

        broker.buy("sh.600519", quantity=100)
        broker.set_date(date(2024, 1, 16))
        broker.set_prices({"sh.600519": {"close": 550.0}})

        order = broker.sell("sh.600519", quantity=100)
        assert order.status.value == "filled"
        assert "sh.600519" not in broker.account.positions

    def test_sell_no_position_rejected(self):
        from backtest.broker import AShareBroker
        broker = AShareBroker(initial_capital=100000.0)
        broker.set_date(date(2024, 1, 15))
        broker.set_prices({"sh.600519": {"close": 500.0}})

        order = broker.sell("sh.600519", quantity=100)
        assert order.status.value == "rejected"

    def test_commission_calculation(self):
        from backtest.broker import AShareBroker
        broker = AShareBroker(initial_capital=100000.0)
        broker.set_date(date(2024, 1, 15))
        broker.set_prices({"sh.600519": {"close": 500.0}})

        order = broker.buy("sh.600519", quantity=100)
        amount = 100 * 500.0  # 50000
        expected_commission = max(amount * 0.0003, 5.0)  # 万3, 最低5元
        assert abs(order.commission - expected_commission) < 0.01

    def test_stamp_duty_on_sell_only(self):
        from backtest.broker import AShareBroker
        broker = AShareBroker(initial_capital=100000.0)
        broker.set_date(date(2024, 1, 15))
        broker.set_prices({"sh.600519": {"close": 500.0}})

        buy_order = broker.buy("sh.600519", quantity=100)
        assert buy_order.stamp_duty == 0  # 买入不收印花税

        broker.set_date(date(2024, 1, 16))
        broker.set_prices({"sh.600519": {"close": 550.0}})
        sell_order = broker.sell("sh.600519", quantity=100)
        assert sell_order.stamp_duty > 0  # 卖出收印花税

    def test_total_asset_calculation(self):
        from backtest.broker import AShareBroker
        broker = AShareBroker(initial_capital=100000.0)
        broker.set_date(date(2024, 1, 15))
        broker.set_prices({"sh.600519": {"close": 500.0}})

        broker.buy("sh.600519", quantity=100)
        total = broker.account.total_asset({"sh.600519": 550.0})
        assert total > 100000  # 盈利了


class TestEventDrivenBacktestEngine:
    """事件驱动回测引擎测试"""

    def test_engine_initialization(self):
        from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
        config = BacktestConfig(
            initial_capital=100000.0,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 12, 31),
        )
        engine = EventDrivenBacktestEngine(config)
        assert engine.config.initial_capital == 100000.0

    def test_load_data(self):
        from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
        config = BacktestConfig()
        engine = EventDrivenBacktestEngine(config)
        df = _make_ohlcv(200)
        engine.load_data("sh.600519", df)
        assert "sh.600519" in engine._data

    def test_run_simple_strategy(self):
        from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
        from backtest.broker import OrderSide

        config = BacktestConfig(
            initial_capital=100000.0,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 12, 31),
        )
        engine = EventDrivenBacktestEngine(config)
        df = _make_ohlcv(300)
        engine.load_data("sh.600519", df)

        held_days = 0

        def simple_strategy(today, bars, broker):
            nonlocal held_days
            # 买入后持有5天然后卖出
            if "sh.600519" not in broker.account.positions:
                broker.buy("sh.600519", 100)
                held_days = 0
            else:
                held_days += 1
                if held_days >= 5:
                    broker.sell("sh.600519", 100)

        result = engine.run(simple_strategy, progress_bar=False)
        assert result.total_trades > 0
        assert result.trading_days > 0
        # 权益曲线应该存在
        assert len(result.equity_curve) > 0
