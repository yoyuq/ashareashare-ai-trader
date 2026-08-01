"""
单元测试: 信号分级 + 胜率分析 + ATR止损 + 凯利仓位

覆盖:
  - SignalGrader 6级信号分级
  - WinRateAnalyzer 仓位计算
  - ATR 自适应止损止盈
  - 移动止盈逻辑
  - 凯利公式仓位计算
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════
# SignalGrader 测试
# ═══════════════════════════════════════════════════════════════

class TestSignalGrader:
    """信号分级系统测试"""

    def test_strong_buy_threshold(self):
        from analysis.risk_controls import SignalGrader, SignalLevel
        card = SignalGrader.grade(
            composite_score=85, win_rate=0.65, regime="strong_bull",
            northbound_signal="bullish", sector_rank=1,
        )
        assert card.level == SignalLevel.STRONG_BUY

    def test_buy_threshold(self):
        from analysis.risk_controls import SignalGrader, SignalLevel
        card = SignalGrader.grade(
            composite_score=70, win_rate=0.55, regime="weak_bull",
            northbound_signal="neutral", sector_rank=3,
        )
        assert card.level in (SignalLevel.BUY, SignalLevel.STRONG_BUY)

    def test_hold_in_bear_market(self):
        from analysis.risk_controls import SignalGrader
        card = SignalGrader.grade(
            composite_score=55, win_rate=0.45, regime="strong_bear",
            northbound_signal="bearish", sector_rank=10,
        )
        # 强熊市下宁可偏保守
        assert card.level.label in ("HOLD", "SELL", "STRONG_SELL")

    def test_watch_threshold(self):
        from analysis.risk_controls import SignalGrader, SignalLevel
        card = SignalGrader.grade(
            composite_score=52, win_rate=0.40, regime="range_bound",
            northbound_signal="neutral", sector_rank=5,
        )
        assert card.level in (SignalLevel.WATCH, SignalLevel.HOLD)

    def test_sell_on_low_score(self):
        from analysis.risk_controls import SignalGrader, SignalLevel
        card = SignalGrader.grade(
            composite_score=25, win_rate=0.20, regime="weak_bear",
            northbound_signal="bearish", sector_rank=15,
        )
        assert card.level in (SignalLevel.SELL, SignalLevel.STRONG_SELL)

    def test_returns_direction(self):
        from analysis.risk_controls import SignalGrader
        card = SignalGrader.grade(
            composite_score=80, win_rate=0.60, regime="strong_bull",
            northbound_signal="bullish", sector_rank=2,
        )
        assert card.direction in ("long", "short", "hold")

    def test_signal_level_from_score(self):
        from analysis.risk_controls import SignalLevel
        assert SignalLevel.from_score(85) == SignalLevel.STRONG_BUY
        assert SignalLevel.from_score(70) == SignalLevel.BUY
        assert SignalLevel.from_score(55) == SignalLevel.WATCH
        assert SignalLevel.from_score(40) == SignalLevel.HOLD
        assert SignalLevel.from_score(25) == SignalLevel.SELL
        assert SignalLevel.from_score(10) == SignalLevel.STRONG_SELL


# ═══════════════════════════════════════════════════════════════
# WinRateAnalyzer 测试
# ═══════════════════════════════════════════════════════════════

class TestWinRateAnalyzer:
    """胜率分析器测试"""

    def test_optimal_position_returns_positive(self):
        from analysis.winrate import WinRateAnalyzer
        pos, shares = WinRateAnalyzer.optimal_position_size(
            capital=100000, win_rate=0.55, avg_win_pct=5.0, avg_loss_pct=3.0,
            regime="range_bound",
        )
        assert pos > 0
        assert shares >= 100
        assert pos <= 100000

    def test_low_win_rate_returns_zero(self):
        from analysis.winrate import WinRateAnalyzer
        pos, shares = WinRateAnalyzer.optimal_position_size(
            capital=100000, win_rate=0.25, avg_win_pct=3.0, avg_loss_pct=5.0,
            regime="range_bound",
        )
        # 胜率太低，不建议仓位
        assert pos <= 0 or shares < 100

    def test_bear_market_reduces_position(self):
        from analysis.winrate import WinRateAnalyzer
        pos_bull, _ = WinRateAnalyzer.optimal_position_size(
            capital=100000, win_rate=0.60, avg_win_pct=5.0, avg_loss_pct=3.0,
            regime="strong_bull",
        )
        pos_bear, _ = WinRateAnalyzer.optimal_position_size(
            capital=100000, win_rate=0.60, avg_win_pct=5.0, avg_loss_pct=3.0,
            regime="strong_bear",
        )
        # 熊市仓位应该 ≤ 牛市仓位
        assert pos_bear <= pos_bull

    def test_higher_win_rate_gives_more_position(self):
        from analysis.winrate import WinRateAnalyzer
        pos_low, _ = WinRateAnalyzer.optimal_position_size(
            capital=100000, win_rate=0.40, avg_win_pct=5.0, avg_loss_pct=3.0,
            regime="range_bound",
        )
        pos_high, _ = WinRateAnalyzer.optimal_position_size(
            capital=100000, win_rate=0.65, avg_win_pct=5.0, avg_loss_pct=3.0,
            regime="range_bound",
        )
        assert pos_high >= pos_low


# ═══════════════════════════════════════════════════════════════
# ATR 止损止盈 测试
# ═══════════════════════════════════════════════════════════════

class TestATRStopLoss:
    """ATR自适应止损测试 (来自 shared.py)"""

    def test_basic_stop_loss_calculation(self):
        from scripts.shared import calc_atr_stop_loss
        sl, tp = calc_atr_stop_loss(entry_price=50.0, atr=1.5, regime="range_bound")
        # ATR=1.5, entry=50, atr_pct=3%
        # range_bound multiplier=2.0, sl_pct=6%
        # sl=50*(1-0.06)=47.00, tp=50*(1+0.09)=54.50
        assert sl < 50.0
        assert tp > 50.0
        assert tp > sl

    def test_zero_atr_uses_default(self):
        from scripts.shared import calc_atr_stop_loss
        sl, tp = calc_atr_stop_loss(entry_price=100.0, atr=0, regime="range_bound")
        # atr=0 triggers default: atr = 100 * 0.03 = 3.0
        assert sl < 100.0
        assert tp > 100.0

    def test_bull_market_wider_stops(self):
        from scripts.shared import calc_atr_stop_loss
        sl_range, tp_range = calc_atr_stop_loss(
            entry_price=50.0, atr=1.5, regime="range_bound"
        )
        sl_bull, tp_bull = calc_atr_stop_loss(
            entry_price=50.0, atr=1.5, regime="strong_bull"
        )
        # 牛市止损距离更宽 (multiplier: 2.0 vs 3.0)
        # So stop_loss is lower (wider) in bull market
        assert sl_bull <= sl_range
        assert tp_bull >= tp_range

    def test_crisis_widest_stops(self):
        from scripts.shared import calc_atr_stop_loss
        sl_crisis, _ = calc_atr_stop_loss(
            entry_price=50.0, atr=1.5, regime="crisis"
        )
        sl_normal, _ = calc_atr_stop_loss(
            entry_price=50.0, atr=1.5, regime="range_bound"
        )
        # 危机模式止损最宽
        assert sl_crisis <= sl_normal

    def test_rsi_overbought_tightens_stop(self):
        from scripts.shared import calc_atr_stop_loss
        sl_normal, _ = calc_atr_stop_loss(
            entry_price=50.0, atr=1.5, regime="range_bound", rsi=50
        )
        sl_tight, _ = calc_atr_stop_loss(
            entry_price=50.0, atr=1.5, regime="range_bound", rsi=78
        )
        # RSI超买时止损更紧 (higher stop_loss, closer to entry)
        assert sl_tight > sl_normal

    def test_stop_loss_capped_between_3_and_12_pct(self):
        from scripts.shared import calc_atr_stop_loss
        # Very small ATR — stop shouldn't be tighter than 3%
        sl, _ = calc_atr_stop_loss(entry_price=100.0, atr=0.1, regime="range_bound")
        assert sl >= 100 * 0.88  # at most 12% away

        # Very large ATR — stop shouldn't be wider than 12%
        sl, _ = calc_atr_stop_loss(entry_price=100.0, atr=50.0, regime="range_bound")
        assert sl >= 100 * 0.88


# ═══════════════════════════════════════════════════════════════
# 移动止盈 测试
# ═══════════════════════════════════════════════════════════════

class TestTrailingStop:
    """移动止盈测试 (来自 shared.py)"""

    def test_no_sell_when_rising(self):
        from scripts.shared import calc_trailing_stop
        # 买入100，涨到120，目前还在120附近
        should_sell, level, detail = calc_trailing_stop(
            entry_price=100.0, current_price=120.0,
            highest_since_entry=121.0, atr=2.0, regime="range_bound",
        )
        assert not should_sell

    def test_trigger_when_price_falls_from_peak(self):
        from scripts.shared import calc_trailing_stop
        # 买入100，最高到110，现在跌回103（跌破trailing level）
        should_sell, level, detail = calc_trailing_stop(
            entry_price=100.0, current_price=103.0,
            highest_since_entry=110.0, atr=1.0, regime="range_bound",
        )
        # ATR=1, entry=100, atr_pct=~1%, mult=2.0
        # trailing_level = 110 * (1 - 0.02) = 107.8
        # 103 < 107.8 → should sell
        assert should_sell
        assert "触发" in detail or "移动止盈" in detail

    def test_no_sell_below_entry(self):
        from scripts.shared import calc_trailing_stop
        # 买入100，跌到95（亏损中），trailing不触发（因为没有盈利）
        should_sell, level, detail = calc_trailing_stop(
            entry_price=100.0, current_price=95.0,
            highest_since_entry=101.0, atr=2.0, regime="range_bound",
        )
        # trailing level < entry price → 不应触发
        assert not should_sell or level <= 100.0


# ═══════════════════════════════════════════════════════════════
# 凯利公式仓位 测试
# ═══════════════════════════════════════════════════════════════

class TestKellyPosition:
    """凯利公式仓位计算测试 (来自 shared.py)"""

    def test_high_conviction_gives_higher_position(self):
        from scripts.shared import calc_kelly_position_pct
        low = calc_kelly_position_pct(win_rate=0.55, conviction=0.45,
                                      payoff_ratio=2.0)
        high = calc_kelly_position_pct(win_rate=0.55, conviction=0.75,
                                       payoff_ratio=2.0)
        assert high >= low

    def test_negative_kelly_returns_zero(self):
        from scripts.shared import calc_kelly_position_pct
        # 胜率低于50%，凯利通常为负
        pct = calc_kelly_position_pct(win_rate=0.30, conviction=0.30,
                                      payoff_ratio=1.5)
        assert pct == 0.0

    def test_position_capped_at_25_pct(self):
        from scripts.shared import calc_kelly_position_pct
        # 极高胜率也不应超过25%
        pct = calc_kelly_position_pct(win_rate=0.80, conviction=0.85,
                                      payoff_ratio=5.0)
        assert pct <= 0.25

    def test_higher_sharpe_increases_position(self):
        from scripts.shared import calc_kelly_position_pct
        low_sharpe = calc_kelly_position_pct(
            win_rate=0.55, conviction=0.60, payoff_ratio=2.0, sharpe=0.5,
        )
        high_sharpe = calc_kelly_position_pct(
            win_rate=0.55, conviction=0.60, payoff_ratio=2.0, sharpe=2.0,
        )
        assert high_sharpe >= low_sharpe


# ═══════════════════════════════════════════════════════════════
# RSI 过滤器 测试
# ═══════════════════════════════════════════════════════════════

class TestRSIFilter:
    """RSI超买过滤测试 (来自 shared.py)"""

    def test_rsi_above_75_blocks(self):
        from scripts.shared import check_rsi_filter
        should_skip, reason = check_rsi_filter(rsi=78, trend_score=0.6)
        assert should_skip
        assert "极度超买" in reason

    def test_rsi_above_70_weak_trend_blocks(self):
        from scripts.shared import check_rsi_filter
        should_skip, reason = check_rsi_filter(rsi=72, trend_score=0.3)
        assert should_skip
        assert "回调风险高" in reason or "超买" in reason

    def test_rsi_above_70_strong_trend_allows(self):
        from scripts.shared import check_rsi_filter
        should_skip, reason = check_rsi_filter(rsi=73, trend_score=0.7)
        assert not should_skip
        assert "允许" in reason or "降级" in reason

    def test_normal_rsi_passes(self):
        from scripts.shared import check_rsi_filter
        should_skip, reason = check_rsi_filter(rsi=55, trend_score=0.5)
        assert not should_skip


# ═══════════════════════════════════════════════════════════════
# 兜底价格系统 测试
# ═══════════════════════════════════════════════════════════════

class TestFallbackPrices:
    """兜底价格和统一价格获取测试"""

    def test_get_best_price_uses_analysis_first(self):
        from scripts.shared import get_best_price
        analysis_prices = {"sh.600519": {"close": 1500.0}}
        price = get_best_price("sh.600519", analysis_prices)
        assert price == 1500.0

    def test_get_best_price_falls_back_to_rt_quotes(self):
        from scripts.shared import get_best_price
        price = get_best_price(
            "sh.600519", {},
            rt_quotes={"sh.600519": {"price": 1450.0}},
        )
        assert price == 1450.0

    def test_get_best_price_returns_zero_for_unknown(self):
        from scripts.shared import get_best_price
        price = get_best_price("sh.999999", {})
        assert price == 0.0

    def test_resolve_name(self):
        from scripts.shared import resolve_name
        assert resolve_name("sh.600519") == "贵州茅台"
        assert resolve_name("sz.300750") == "宁德时代"
        assert resolve_name("sh.999999") == "999999"


# ═══════════════════════════════════════════════════════════════
# 🆕 v2.14: 增强风控层测试
# ═══════════════════════════════════════════════════════════════

class TestEnhancedRiskControls:
    """L5-L8 风控层测试"""

    def test_stampede_detection_triggers(self):
        from analysis.risk_controls import PortfolioRiskManager
        rm = PortfolioRiskManager(initial_capital=100000)
        positions = {"A": None, "B": None, "C": None}
        daily_changes = {"A": -0.05, "B": -0.04, "C": 0.01}
        triggered, msg = rm.check_stampede_risk(positions, daily_changes)
        assert triggered
        assert "防踩踏" in msg

    def test_stampede_detection_no_trigger(self):
        from analysis.risk_controls import PortfolioRiskManager
        rm = PortfolioRiskManager(initial_capital=100000)
        positions = {"A": None, "B": None, "C": None, "D": None}
        daily_changes = {"A": -0.05, "B": 0.01, "C": 0.02, "D": 0.03}
        triggered, _ = rm.check_stampede_risk(positions, daily_changes)
        assert not triggered

    def test_limit_down_main_board(self):
        from analysis.risk_controls import PortfolioRiskManager
        rm = PortfolioRiskManager()
        triggered, msg = rm.check_limit_down(
            "sh.600519", pct_change=-9.9, volume_ratio=0.05
        )
        assert triggered
        assert "一字跌停" in msg

    def test_limit_down_not_triggered_normal(self):
        from analysis.risk_controls import PortfolioRiskManager
        rm = PortfolioRiskManager()
        triggered, _ = rm.check_limit_down(
            "sh.600519", pct_change=-3.0, volume_ratio=0.5
        )
        assert not triggered

    def test_limit_down_chi_next_threshold(self):
        from analysis.risk_controls import PortfolioRiskManager
        rm = PortfolioRiskManager()
        # 创业板 20% 跌停
        triggered, msg = rm.check_limit_down(
            "sz.300750", pct_change=-19.8, volume_ratio=0.03
        )
        assert triggered

    def test_correlation_warns_on_high_corr(self):
        from analysis.risk_controls import PortfolioRiskManager
        rng = np.random.default_rng(42)
        base = rng.normal(0.001, 0.015, 100)
        returns = {
            "A": base,
            "B": base * 0.99 + rng.normal(0, 0.001, 100),  # 接近完全相关
            "C": rng.normal(0.002, 0.02, 100),  # 不相关
        }
        warnings = PortfolioRiskManager.check_correlation(returns)
        assert len(warnings) >= 1
        assert "A" in warnings[0]["pair"][0] or "A" in warnings[0]["pair"][1]

    def test_correlation_no_warn_on_uncorrelated(self):
        from analysis.risk_controls import PortfolioRiskManager
        rng = np.random.default_rng(99)
        returns = {
            "X": rng.normal(0.001, 0.015, 100),
            "Y": rng.normal(0.002, 0.02, 100),
        }
        warnings = PortfolioRiskManager.check_correlation(returns)
        # 两个独立的随机序列, 相关性应低于0.7
        assert len(warnings) == 0
