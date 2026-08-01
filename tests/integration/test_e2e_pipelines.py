"""
端到端集成测试 — 核心数据流: 数据获取 → 指标计算 → 策略回测 → 信号分级 → 模拟交易

不依赖外部 API (使用模拟数据), 验证所有模块之间的数据流完整性。
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolate_portfolio(tmp_path, monkeypatch):
    """隔离真实 simulation_data/portfolio.json: 集成测试一律写入临时目录,
    避免 reset()/execute_buy() 污染生产持仓状态 (此前实测会清空真实账户)。"""
    monkeypatch.setenv("PORTFOLIO_PATH", str(tmp_path / "portfolio.json"))


# ═══════════════════════════════════════════════════════════════
# 测试数据工厂
# ═══════════════════════════════════════════════════════════════

def _make_ohlcv(n: int = 300, trend: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """生成带趋势的模拟OHLCV数据 (含3年牛市+震荡+熊市)"""
    rng = np.random.default_rng(seed)
    # 分段趋势: 前1/3上涨, 中1/3震荡, 后1/3下跌
    n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
    trend_segments = np.concatenate([
        np.linspace(0, 20, n1),
        np.linspace(20, 23, n2),
        np.linspace(23, 5, n3),
    ])
    close = 50 + trend_segments + rng.normal(0, 1.5, n)
    close = np.maximum(close, 2)
    high = close + rng.uniform(0.2, 2.0, n)
    low = close - rng.uniform(0.2, 2.0, n)
    open_price = close - rng.uniform(-0.5, 0.5, n)
    volume = rng.integers(100000, 5000000, n).astype(float)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates, "open": open_price, "high": high,
        "low": low, "close": close, "volume": volume,
    })


# ═══════════════════════════════════════════════════════════════
# Pipeline 1: 数据获取 → 指标计算 → 策略回测
# ═══════════════════════════════════════════════════════════════

class TestDataToBacktestPipeline:
    """端到端: 数据→指标→回测→统计"""

    def test_full_pipeline_single_stock(self):
        from analysis.indicators import TechnicalAnalyzer
        from analysis.recommender import STRATEGY_BACKTESTERS, _calc_bt_stats

        df = _make_ohlcv(300)

        # Step 1: 计算指标
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(df, symbol="sh.600519")
        indicators_df = result.to_dataframe()
        assert not indicators_df.empty
        assert "rsi_14" in indicators_df.columns
        assert "trend_score" in indicators_df.columns

        # Step 2: 策略回测 (所有9个策略)
        for strategy_id, bt_func in STRATEGY_BACKTESTERS.items():
            bt = bt_func(df)
            assert "signals" in bt, f"{strategy_id} missing signals"
            assert "win_rate" in bt, f"{strategy_id} missing win_rate"
            assert "sharpe" in bt, f"{strategy_id} missing sharpe"
            assert bt.get("cost_adjusted", False), f"{strategy_id} missing cost flag"

    def test_pipeline_multi_stock(self):
        from analysis.indicators import TechnicalAnalyzer
        from analysis.recommender import _backtest_ma_trend, _backtest_bollinger_reversal

        symbols = ["sh.600519", "sz.300750", "sh.600036"]
        results = {}

        for i, sym in enumerate(symbols):
            df = _make_ohlcv(250, trend=0.03 + i * 0.01, seed=100 + i)
            analyzer = TechnicalAnalyzer()
            ind = analyzer.compute_all(df, symbol=sym)
            last = ind.to_dataframe().iloc[-1]

            bt_ma = _backtest_ma_trend(df)
            bt_bb = _backtest_bollinger_reversal(df)

            results[sym] = {
                "indicators": {
                    "rsi": round(float(last.get("rsi_14", 50)), 1),
                    "trend": round(float(last.get("trend_score", 0)), 2),
                    "composite": round(float(last.get("composite_score", 50)), 1),
                },
                "ma_trend": {
                    "win_rate": bt_ma["win_rate"],
                    "sharpe": bt_ma["sharpe"],
                    "signals": bt_ma["signals"],
                },
                "bollinger": {
                    "win_rate": bt_bb["win_rate"],
                    "sharpe": bt_bb["sharpe"],
                    "signals": bt_bb["signals"],
                },
            }

        assert len(results) == 3
        # 每个标的都有指标和回测结果
        for sym, data in results.items():
            assert -1 <= data["indicators"]["trend"] <= 1
            assert 0 <= data["indicators"]["composite"] <= 100
            assert 0 <= data["ma_trend"]["win_rate"] <= 1


# ═══════════════════════════════════════════════════════════════
# Pipeline 2: 信号分级 → 仓位计算 → 风控检查
# ═══════════════════════════════════════════════════════════════

class TestSignalToExecutionPipeline:
    """端到端: 信号分级→仓位→风控"""

    def test_signal_grade_to_position(self):
        from analysis.risk_controls import SignalGrader
        from analysis.winrate import WinRateAnalyzer

        # Step 1: 信号分级
        card = SignalGrader.grade(
            composite_score=72, win_rate=0.58, regime="weak_bull",
            northbound_signal="neutral", sector_rank=3,
        )
        assert card.level.label in ("BUY", "STRONG_BUY", "WATCH")

        # Step 2: 仓位计算
        pos_amt, shares = WinRateAnalyzer.optimal_position_size(
            capital=100000, win_rate=0.58, avg_win_pct=4.5, avg_loss_pct=2.8,
            regime="weak_bull",
        )
        assert pos_amt > 0
        assert shares >= 100  # 至少一手

        # 仓位不超过总资金
        assert pos_amt <= 100000
        # 熊市更保守
        pos_bear, _ = WinRateAnalyzer.optimal_position_size(
            capital=100000, win_rate=0.58, avg_win_pct=4.5, avg_loss_pct=2.8,
            regime="strong_bear",
        )
        assert pos_bear <= pos_amt

    def test_risk_controls_integration(self):
        from analysis.risk_controls import PortfolioRiskManager

        rm = PortfolioRiskManager(initial_capital=100000)

        # 模拟正常市场
        daily_returns = pd.Series(np.random.normal(0.001, 0.012, 60))
        state = rm.update(105000, daily_returns)  # 盈利5%
        assert not state.circuit_breaker_active
        assert not state.crisis_mode
        assert state.suggested_exposure_pct > 0.3

        # 模拟大幅回撤
        state2 = rm.update(90000, daily_returns)  # 亏损10%
        assert state2.circuit_breaker_active
        assert state2.risk_multiplier < 0.8

    def test_enhanced_risk_layers(self):
        from analysis.risk_controls import PortfolioRiskManager

        rm = PortfolioRiskManager()

        # L5: 防踩踏
        triggered, _ = rm.check_stampede_risk(
            {"A": None, "B": None, "C": None},
            {"A": -0.04, "B": -0.05, "C": -0.03},
        )
        assert triggered

        # L7: 跌停检测
        triggered, _ = rm.check_limit_down("sh.600519", -9.9, 0.03)
        assert triggered

        # L8: 相关性
        rng = np.random.default_rng(42)
        base = rng.normal(0.001, 0.015, 100)
        returns = {"X": base, "Y": base * 0.98 + rng.normal(0, 0.002, 100)}
        warnings = PortfolioRiskManager.check_correlation(returns)
        assert len(warnings) >= 1


# ═══════════════════════════════════════════════════════════════
# Pipeline 3: 模拟交易完整流程
# ═══════════════════════════════════════════════════════════════

class TestPaperTradingPipeline:
    """端到端: 模拟交易买入→MTM→卖出"""

    def test_buy_mtm_sell_flow(self):
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine

        manager = PortfolioManager()
        manager.reset()
        engine = PaperTradingEngine(manager)

        # 买入 (价格设低以确保小账户足够一手)
        trade = engine.execute_buy(
            symbol="sh.600519", name="贵州茅台", price=100.0,
            recommendation={
                "strategy_id": "dual_ma_trend",
                "strategy_name": "双均线趋势",
                "win_rate": 0.55,
                "conviction": 0.65,
                "verdict_summary": "技术面多头排列",
                "key_reasons": ["MA金叉", "放量突破"],
                "risks": ["大盘回调"],
            },
        )
        assert trade is not None
        assert trade.side == "buy"
        assert trade.quantity >= 100
        assert "sh.600519" in engine.state.positions

        # Mark-to-Market (涨了)
        mtm = engine.mark_to_market({"sh.600519": 110.0})
        assert "sh.600519" in mtm
        assert mtm["sh.600519"]["unrealized_pnl"] > 0

        # Mark-to-Market (跌了)
        mtm2 = engine.mark_to_market({"sh.600519": 95.0})
        assert mtm2["sh.600519"]["unrealized_pnl"] < 0

        # 手动卖出 (T+1: 模拟持仓为前一交易日买入)
        engine.state.positions["sh.600519"].buy_date = "2020-01-01"
        sell_trade = engine.execute_sell(
            symbol="sh.600519", exit_reason="manual", price=105.0,
        )
        assert sell_trade is not None
        assert sell_trade.side == "sell"
        assert sell_trade.pnl is not None
        assert "sh.600519" not in engine.state.positions

    def test_exit_conditions(self):
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine

        manager = PortfolioManager()
        manager.reset()
        engine = PaperTradingEngine(manager)

        engine.execute_buy(
            symbol="sh.600519", name="贵州茅台", price=100.0,
            recommendation={
                "stop_loss": 93.0, "take_profit": 110.0,
            },
        )

        # 触发止损 (paper_trader 输出 dynamic_stop 原因)
        triggers = engine.check_exit_conditions({"sh.600519": 90.0})
        assert len(triggers) == 1
        assert triggers[0]["reason"] == "dynamic_stop"

        # 正常持有 (不触发)
        manager.reset()
        engine = PaperTradingEngine(manager)
        engine.execute_buy(
            symbol="sh.600519", name="贵州茅台", price=100.0,
            recommendation={
                "stop_loss": 90.0, "take_profit": 120.0,
            },
        )
        triggers2 = engine.check_exit_conditions({"sh.600519": 105.0})
        assert len(triggers2) == 0

    def test_daily_snapshot_tracks_performance(self):
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine

        manager = PortfolioManager()
        manager.reset()
        engine = PaperTradingEngine(manager)

        engine.execute_buy(
            symbol="sh.600519", name="贵州茅台", price=100.0,
            recommendation={"stop_loss": 90, "take_profit": 120},
        )
        engine.mark_to_market({"sh.600519": 110.0})

        snapshot = engine.record_daily_snapshot()
        assert snapshot.total_value > 100000  # 有盈利
        assert snapshot.positions_count == 1
        assert snapshot.daily_return_pct != 0

    def test_duplicate_buy_prevented(self):
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine

        manager = PortfolioManager()
        manager.reset()
        engine = PaperTradingEngine(manager)

        t1 = engine.execute_buy("sh.600519", "贵州茅台", 100.0)
        assert t1 is not None

        # 重复买入应被阻止
        t2 = engine.execute_buy("sh.600519", "贵州茅台", 100.0)
        assert t2 is None


# ═══════════════════════════════════════════════════════════════
# Pipeline 4: 过拟合检测 + 回测对比
# ═══════════════════════════════════════════════════════════════

class TestOverfittingAndCompare:
    """端到端: 过拟合检测 + 策略对比"""

    def test_overfitting_guard_on_returns(self):
        from backtest.overfitting import OverfittingGuard

        guard = OverfittingGuard(n_simulations=200)

        # 有真实alpha的假数据 (>0 mean)
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0008, 0.012, 252))
        report = guard.evaluate(returns)

        assert hasattr(report, "is_overfit")
        assert np.isnan(report.pbo) or 0 <= report.pbo <= 1  # 无变体矩阵时PBO为NaN
        assert hasattr(report, "monte_carlo_pvalue")

    def test_time_split_maintains_order(self):
        from backtest.overfitting import OverfittingGuard

        df = _make_ohlcv(300)
        train, val, test = OverfittingGuard.split_time_series(df)

        assert len(train) > len(val)
        assert len(val) > 0
        assert len(test) > 0
        # 时间顺序不能乱
        assert train["date"].max() <= val["date"].min()
        assert val["date"].max() <= test["date"].min()

    def test_all_strategies_consistent(self):
        """所有策略回测函数返回一致的数据结构"""
        from analysis.recommender import STRATEGY_BACKTESTERS

        df = _make_ohlcv(300)
        required_fields = {"signals", "win_rate", "sharpe", "max_dd",
                          "profit_factor", "expected_value", "cost_adjusted"}

        for sid, bt_func in STRATEGY_BACKTESTERS.items():
            result = bt_func(df)
            missing = required_fields - set(result.keys())
            assert not missing, f"{sid} missing fields: {missing}"
            assert 0 <= result["win_rate"] <= 1, f"{sid} win_rate out of range"
            if result["signals"] > 0 and result["win_rate"] > 0:
                assert result["profit_factor"] >= 0, f"{sid} profit_factor={result['profit_factor']} invalid"


# ═══════════════════════════════════════════════════════════════
# Pipeline 5: ATR/Kelly/止损 端到端
# ═══════════════════════════════════════════════════════════════

class TestSharedUtilsIntegration:
    """端到端: ATR止损+凯利+RSI过滤+移动止盈"""

    def test_atr_to_kelly_to_execution(self):
        from scripts.shared import (calc_atr_stop_loss, calc_kelly_position_pct,
                                   check_rsi_filter, calc_trailing_stop)

        # 场景: 强牛市中的茅台
        entry = 1500.0
        atr = 25.0
        regime = "strong_bull"
        rsi = 58

        # ATR 止损止盈
        sl, tp = calc_atr_stop_loss(entry, atr, regime, rsi)
        assert sl < entry < tp
        assert tp > sl

        # RSI 过滤
        skip, _ = check_rsi_filter(rsi, trend_score=0.7)
        assert not skip  # RSI=58 正常, 不跳过

        # 凯利仓位
        kelly = calc_kelly_position_pct(win_rate=0.55, conviction=0.65,
                                        payoff_ratio=2.0, sharpe=1.5)
        assert 0 < kelly <= 0.25

        # 模拟价格新高后回落 → 移动止盈触发
        should_sell, _, detail = calc_trailing_stop(
            entry, current_price=entry * 1.12, highest_since_entry=entry * 1.18,
            atr=atr, regime=regime,
        )
        assert should_sell  # 从高点回落应触发

    def test_full_risk_flow_bear_market(self):
        from scripts.shared import calc_atr_stop_loss, calc_kelly_position_pct, check_rsi_filter

        # 场景: 强熊市, RSI超卖
        entry = 100.0
        sl, tp = calc_atr_stop_loss(entry, atr=5.0, regime="strong_bear", rsi=25)
        # 熊市止损更紧
        assert sl > entry * 0.85  # 不低于-15%

        # RSI超卖不阻止买入 (可能是抄底机会)
        skip, reason = check_rsi_filter(rsi=25, trend_score=-0.6)
        assert not skip  # 超卖不应该阻止

        # 熊市凯利更保守
        kelly = calc_kelly_position_pct(win_rate=0.40, conviction=0.35, payoff_ratio=1.5)
        assert kelly <= 0.1  # 低胜率应给出极低仓位


# ═══════════════════════════════════════════════════════════════
# Pipeline 6: 市场状态 → 策略参数 端到端
# ═══════════════════════════════════════════════════════════════

class TestRegimeToStrategy:
    """端到端: 市场状态识别→参数建议→策略权重"""

    def test_regime_detection_to_params(self):
        from analysis.regime import MarketRegimeDetector, MarketRegime, get_regime_parameters

        # 牛市场景 (降低噪声确保不会被误判为crisis)
        rng = np.random.default_rng(1)
        n = 252
        close = 50 + np.linspace(0, 30, n) + rng.normal(0, 0.8, n)
        close = np.maximum(close, 5)
        df_bull = pd.DataFrame({
            "open": close - 0.3, "high": close + 0.8, "low": close - 0.8,
            "close": close, "volume": rng.integers(500000, 2000000, n).astype(float),
        })
        detector = MarketRegimeDetector()
        result = detector.detect(df_bull)
        assert result.regime in (MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL, MarketRegime.RANGE_BOUND)

        params = get_regime_parameters(result.regime)
        assert params["position_multiplier"] >= 0.5
        assert params["trend_strategy_weight"] > params["mean_reversion_weight"]

        # 震荡场景
        rng = np.random.default_rng(99)
        close = 50 + np.cumsum(rng.normal(0, 1.5, 252))
        close = np.maximum(close, 5)
        df_range = pd.DataFrame({
            "open": close - 0.3, "high": close + 1.0, "low": close - 1.0,
            "close": close, "volume": rng.integers(100000, 1000000, 252).astype(float),
        })
        result2 = detector.detect(df_range)
        # 震荡市参数
        params2 = get_regime_parameters(result2.regime)
        assert params2["mean_reversion_weight"] >= params2["trend_strategy_weight"]
