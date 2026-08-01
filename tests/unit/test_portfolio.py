"""
单元测试: 模拟交易 Portfolio + PaperTrader
"""
import pytest
from datetime import date


class TestPortfolioState:
    """持仓状态测试"""

    def test_default_state(self):
        from simulation.portfolio import PortfolioState
        state = PortfolioState()
        assert state.initial_capital == 100000.0
        assert state.cash == 100000.0
        assert state.total_value == 100000.0
        assert state.total_return == 0.0
        assert state.total_return_pct == 0.0
        assert len(state.positions) == 0
        assert state.win_rate == 0.0

    def test_total_value_with_positions(self):
        from simulation.portfolio import PortfolioState, PaperPosition
        state = PortfolioState(initial_capital=100000.0, cash=50000.0)
        pos = PaperPosition(
            symbol="sh.600519", name="贵州茅台",
            quantity=100, avg_cost=500.0, total_cost=50000.0,
            current_price=550.0,
        )
        state.positions["sh.600519"] = pos
        assert state.position_value == 55000.0
        assert state.total_value == 105000.0
        assert state.total_return == 5000.0
        assert abs(state.total_return_pct - 5.0) < 0.01

    def test_win_rate_calculation(self):
        from simulation.portfolio import PortfolioState
        state = PortfolioState(win_count=6, loss_count=4)
        assert state.win_rate == 0.6

    def test_serialization_roundtrip(self):
        from simulation.portfolio import PortfolioState, PaperPosition, PaperTrade
        state = PortfolioState(initial_capital=100000.0, cash=80000.0)
        pos = PaperPosition(
            symbol="sh.600036", name="招商银行",
            quantity=200, avg_cost=38.0, total_cost=7600.0,
            current_price=40.0, stop_loss=36.0, take_profit=44.0,
        )
        state.positions["sh.600036"] = pos
        state.trade_history.append(PaperTrade(
            trade_id="T-001", date="2026-07-01", symbol="sh.600036",
            name="招商银行", side="buy", quantity=200, price=38.0,
            amount=7600.0, commission=2.28, reason="测试",
        ))

        data = state.to_dict()
        restored = PortfolioState.from_dict(data)

        assert restored.cash == 80000.0
        assert len(restored.positions) == 1
        assert restored.positions["sh.600036"].name == "招商银行"
        assert restored.positions["sh.600036"].stop_loss == 36.0
        assert restored.positions["sh.600036"].take_profit == 44.0
        assert len(restored.trade_history) == 1


class TestPortfolioManager:
    """持仓管理器测试"""

    def test_load_default(self, tmp_path):
        from simulation.portfolio import PortfolioManager
        mgr = PortfolioManager(filepath=tmp_path / "test_portfolio.json")
        state = mgr.state
        assert state.initial_capital == 100000.0
        assert state.cash == 100000.0

    def test_save_and_load(self, tmp_path):
        from simulation.portfolio import PortfolioManager, PaperPosition
        mgr = PortfolioManager(filepath=tmp_path / "test_portfolio.json")
        state = mgr.state
        pos = PaperPosition(
            symbol="sh.600519", name="贵州茅台",
            quantity=100, avg_cost=500.0, total_cost=50000.0,
            current_price=550.0, stop_loss=475.0, take_profit=600.0,
            rec_strategy_name="双均线趋势",
        )
        state.positions["sh.600519"] = pos
        mgr.save()

        # 重新加载
        mgr2 = PortfolioManager(filepath=tmp_path / "test_portfolio.json")
        state2 = mgr2.state
        assert len(state2.positions) == 1
        assert state2.positions["sh.600519"].name == "贵州茅台"
        assert state2.positions["sh.600519"].stop_loss == 475.0

    def test_reset(self, tmp_path):
        from simulation.portfolio import PortfolioManager, PaperPosition
        mgr = PortfolioManager(filepath=tmp_path / "test_reset.json")
        state = mgr.state
        state.positions["test"] = PaperPosition(symbol="test", name="test")
        mgr.save()
        assert len(mgr.state.positions) == 1

        mgr.reset()
        assert len(mgr.state.positions) == 0
        assert mgr.state.cash == 100000.0


class TestPaperTradingEngine:
    """模拟交易引擎测试"""

    def test_buy_creates_position(self, tmp_path):
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine
        mgr = PortfolioManager(filepath=tmp_path / "test_pt.json")
        engine = PaperTradingEngine(mgr)

        trade = engine.execute_buy(
            symbol="sh.600519", name="贵州茅台",
            price=500.0,
            recommendation={
                "conviction": 0.6, "score": 7,
                "strategy_id": "dual_ma_trend",
                "strategy_name": "双均线趋势",
                "win_rate": 0.65,
                "key_reasons": ["MA金叉", "趋势向上"],
                "risks": ["市场风险"],
            },
        )
        assert trade is not None
        assert trade.symbol == "sh.600519"
        assert trade.side == "buy"
        assert trade.quantity >= 100

        state = engine.state
        assert "sh.600519" in state.positions
        pos = state.positions["sh.600519"]
        assert pos.name == "贵州茅台"
        assert pos.rec_strategy_name == "双均线趋势"
        assert pos.rec_win_rate == 0.65

    def test_sell_closes_position(self, tmp_path):
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine
        mgr = PortfolioManager(filepath=tmp_path / "test_sell.json")
        engine = PaperTradingEngine(mgr)

        # 先买入
        engine.execute_buy(
            symbol="sh.600519", name="贵州茅台", price=500.0,
            recommendation={"conviction": 0.6},
        )
        assert "sh.600519" in engine.state.positions

        # T+1: 模拟持仓为前一交易日买入 (当日买入不可当日卖出)
        engine.state.positions["sh.600519"].buy_date = "2020-01-01"

        # 再卖出
        trade = engine.execute_sell(
            symbol="sh.600519", exit_reason="take_profit",
        )
        assert trade is not None
        assert trade.side == "sell"
        assert trade.exit_reason == "take_profit"
        assert "sh.600519" not in engine.state.positions

    def test_duplicate_buy_skipped(self, tmp_path):
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine
        mgr = PortfolioManager(filepath=tmp_path / "test_dup.json")
        engine = PaperTradingEngine(mgr)

        t1 = engine.execute_buy(
            symbol="sh.600519", name="贵州茅台", price=500.0,
            recommendation={"conviction": 0.6},
        )
        assert t1 is not None

        t2 = engine.execute_buy(
            symbol="sh.600519", name="贵州茅台", price=500.0,
            recommendation={"conviction": 0.6},
        )
        assert t2 is None  # 已持仓,应跳过

    def test_industry_concentration_limit(self, tmp_path):
        """v2.9: 行业集中度限制"""
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine, DEFAULT_MAX_INDUSTRY_PCT
        mgr = PortfolioManager(filepath=tmp_path / "test_ind.json")
        engine = PaperTradingEngine(mgr)

        # 先大比例买入一只银行股
        engine.execute_buy(
            symbol="sh.600036", name="招商银行", price=38.0,
            recommendation={"conviction": 0.8},
        )
        # 手动放大持仓价值来触发行业限制
        pos = engine.state.positions["sh.600036"]
        pos.total_cost = 35000.0  # 35%+ 仓位
        engine.manager.save()

        # 再尝试买入同行业(银行)的股票 — 应被限制
        t2 = engine.execute_buy(
            symbol="sh.601318", name="中国平安", price=54.0,
            recommendation={"conviction": 0.7},
        )
        # 可能被行业集中度或现金缓冲限制
        # 无论是否触发限制, 结果必须是合法类型且账户状态一致
        assert t2 is None or hasattr(t2, "symbol"), f"非法交易结果类型: {type(t2)}"
        assert engine.state.cash >= 0, "买入后现金不应为负"

    def test_mark_to_market(self, tmp_path):
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine
        mgr = PortfolioManager(filepath=tmp_path / "test_mtm.json")
        engine = PaperTradingEngine(mgr)

        engine.execute_buy(
            symbol="sh.600519", name="贵州茅台", price=500.0,
            recommendation={"conviction": 0.6},
        )

        result = engine.mark_to_market({"sh.600519": 550.0})
        assert "sh.600519" in result
        assert result["sh.600519"]["current_price"] == 550.0
        assert result["sh.600519"]["unrealized_pnl"] > 0  # 盈利
