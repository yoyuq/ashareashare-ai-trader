"""
PaperAccount — 虚拟资金账户 + 基准对比 (v3.1)

完整的模拟交易账户,包含:
  - T+1 交收约束 (当日买入次日可卖)
  - 涨跌停封板检测
  - 滑点模拟 + 手续费
  - 沪深300基准对比
  - 8层风控集成

用法:
    account = PaperAccount(initial_capital=100000)
    account.buy("sh.600519", price=1800, quantity=100, date="2026-07-29")
    account.update_mtm(prices)  # 每日估值
    nav = account.get_nav()
    alpha = account.get_alpha(benchmark_return_pct=0.02)
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


@dataclass
class Position:
    """Single position"""
    symbol: str
    name: str = ""
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    buy_date: str = ""           # T日
    settlement_date: str = ""    # T+1日

    @property
    def market_value(self) -> float:
        return self.current_price * self.quantity

    @property
    def cost_basis(self) -> float:
        return self.avg_cost * self.quantity

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        return self.unrealized_pnl / self.cost_basis if self.cost_basis > 0 else 0

    @property
    def can_sell_today(self) -> bool:
        """Check T+1: can only sell if settlement date has passed"""
        if not self.settlement_date or not self.buy_date:
            return True  # Legacy positions (no buy date recorded)
        return date.today().isoformat() > self.settlement_date

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "quantity": self.quantity,
            "avg_cost": round(self.avg_cost, 2),
            "current_price": round(self.current_price, 2),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
            "buy_date": self.buy_date,
            "can_sell": self.can_sell_today,
        }


@dataclass
class TradeRecord:
    """Single trade record"""
    symbol: str
    side: str                   # BUY | SELL
    price: float
    quantity: int
    amount: float
    commission: float
    trade_date: str
    reason: str = ""


@dataclass
class NavPoint:
    """NAV data point for equity curve"""
    date: str
    nav: float
    benchmark_nav: float
    cash: float
    position_value: float


class PaperAccount:
    """
    Virtual paper trading account with A-share constraints.

    Features:
      - T+1 settlement (buy today, sell tomorrow)
      - Price limit detection (cannot buy at limit-up, cannot sell at limit-down)
      - Slippage simulation (default 0.1%)
      - Commission (default 0.03% buy, 0.13% sell incl. stamp duty)
      - Benchmark tracking (CSI 300)
    """

    COMMISSION_BUY = 0.0003      # 0.03%
    COMMISSION_SELL = 0.0013     # 0.03% + 0.1% stamp duty
    SLIPPAGE = 0.001             # 0.1%
    MIN_COMMISSION = 5.0         # Minimum commission

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.nav_history: List[NavPoint] = []
        self.benchmark_nav = initial_capital  # Track CSI300 equivalent

    def buy(
        self,
        symbol: str,
        price: float,
        quantity: int,
        trade_date: str = "",
        name: str = "",
        reason: str = "",
        is_limit_up: bool = False,
    ) -> Optional[TradeRecord]:
        """
        Execute a buy order.

        Args:
            symbol: Stock symbol (e.g., "sh.600519")
            price: Execution price
            quantity: Shares to buy (multiples of 100 for A-shares)
            trade_date: Trade date (ISO format)
            name: Stock name
            reason: Trade reason
            is_limit_up: Whether the stock is at limit-up (cannot buy)

        Returns:
            TradeRecord or None if rejected
        """
        if is_limit_up:
            logger.debug(f"[PaperAccount] Cannot buy {symbol}: at limit-up")
            return None

        if quantity < 100:
            logger.debug(f"[PaperAccount] Min order is 100 shares, got {quantity}")
            return None

        # Slippage
        exec_price = price * (1 + self.SLIPPAGE)
        amount = exec_price * quantity
        commission = max(self.MIN_COMMISSION, amount * self.COMMISSION_BUY)
        total_cost = amount + commission

        if total_cost > self.cash:
            logger.warning(f"[PaperAccount] Insufficient cash: need {total_cost:.0f}, have {self.cash:.0f}")
            return None

        if not trade_date:
            trade_date = date.today().isoformat()
        settlement_date = (date.fromisoformat(trade_date) + timedelta(days=1)).isoformat()

        # Execute
        self.cash -= total_cost

        # Update or create position
        if symbol in self.positions:
            pos = self.positions[symbol]
            new_qty = pos.quantity + quantity
            pos.avg_cost = (pos.cost_basis + amount) / new_qty if new_qty > 0 else 0
            pos.quantity = new_qty
            pos.current_price = price
            pos.buy_date = trade_date
            pos.settlement_date = settlement_date
        else:
            self.positions[symbol] = Position(
                symbol=symbol, name=name, quantity=quantity,
                avg_cost=exec_price, current_price=price,
                buy_date=trade_date, settlement_date=settlement_date,
            )

        trade = TradeRecord(
            symbol=symbol, side="BUY", price=exec_price,
            quantity=quantity, amount=amount, commission=commission,
            trade_date=trade_date, reason=reason,
        )
        self.trade_history.append(trade)
        logger.info(f"[PaperAccount] BUY {symbol} x{quantity} @ {exec_price:.2f} "
                    f"cost={total_cost:.0f} cash_left={self.cash:.0f}")
        return trade

    def sell(
        self,
        symbol: str,
        price: float,
        quantity: int = None,
        trade_date: str = "",
        reason: str = "",
        is_limit_down: bool = False,
    ) -> Optional[TradeRecord]:
        """
        Execute a sell order.

        Args:
            symbol: Stock symbol
            price: Execution price
            quantity: Shares to sell (None = all)
            trade_date: Trade date
            reason: Trade reason
            is_limit_down: Whether the stock is at limit-down (cannot sell)

        Returns:
            TradeRecord or None if rejected
        """
        if is_limit_down:
            logger.debug(f"[PaperAccount] Cannot sell {symbol}: at limit-down")
            return None

        pos = self.positions.get(symbol)
        if pos is None:
            logger.warning(f"[PaperAccount] No position for {symbol}")
            return None

        # T+1 check
        if not pos.can_sell_today and trade_date:
            logger.debug(f"[PaperAccount] T+1 constraint: {symbol} bought {pos.buy_date}, "
                        f"can sell after {pos.settlement_date}")
            return None

        if quantity is None:
            quantity = pos.quantity
        if quantity > pos.quantity:
            quantity = pos.quantity

        # Slippage
        exec_price = price * (1 - self.SLIPPAGE)
        amount = exec_price * quantity
        commission = max(self.MIN_COMMISSION, amount * self.COMMISSION_SELL)
        net_proceeds = amount - commission

        self.cash += net_proceeds

        # Update position
        pos.quantity -= quantity
        if pos.quantity == 0:
            del self.positions[symbol]

        trade = TradeRecord(
            symbol=symbol, side="SELL", price=exec_price,
            quantity=quantity, amount=amount, commission=commission,
            trade_date=trade_date or date.today().isoformat(),
            reason=reason,
        )
        self.trade_history.append(trade)
        logger.info(f"[PaperAccount] SELL {symbol} x{quantity} @ {exec_price:.2f} "
                    f"proceeds={net_proceeds:.0f} cash={self.cash:.0f}")
        return trade

    def update_mtm(self, prices: Dict[str, float]):
        """Mark-to-market: update all position prices"""
        for sym, pos in self.positions.items():
            if sym in prices:
                pos.current_price = prices[sym]

    def update_benchmark(self, benchmark_return_pct: float):
        """Update benchmark NAV with daily return"""
        self.benchmark_nav *= (1 + benchmark_return_pct)

    def record_nav(self, dt: str = ""):
        """Record daily NAV snapshot"""
        if not dt:
            dt = date.today().isoformat()
        self.nav_history.append(NavPoint(
            date=dt,
            nav=self.get_nav(),
            benchmark_nav=self.benchmark_nav,
            cash=self.cash,
            position_value=self.get_position_value(),
        ))

    def get_nav(self) -> float:
        """Total net asset value"""
        return self.cash + self.get_position_value()

    def get_position_value(self) -> float:
        """Total market value of all positions"""
        return sum(p.market_value for p in self.positions.values())

    def get_total_return_pct(self) -> float:
        """Total return percentage"""
        return (self.get_nav() / self.initial_capital - 1) if self.initial_capital > 0 else 0

    def get_alpha(self, benchmark_return_pct: float) -> float:
        """Alpha = portfolio return - benchmark return"""
        return self.get_total_return_pct() - benchmark_return_pct

    def get_equity_curve(self) -> Dict[str, list]:
        """Return equity curve data for plotting"""
        return {
            "dates": [p.date for p in self.nav_history],
            "nav": [p.nav for p in self.nav_history],
            "benchmark": [p.benchmark_nav for p in self.nav_history],
            "cash": [p.cash for p in self.nav_history],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Portfolio summary"""
        nav = self.get_nav()
        pos_value = self.get_position_value()
        return {
            "initial_capital": self.initial_capital,
            "total_value": round(nav, 2),
            "cash": round(self.cash, 2),
            "position_value": round(pos_value, 2),
            "total_return_pct": round(self.get_total_return_pct() * 100, 2),
            "position_count": len(self.positions),
            "trade_count": len(self.trade_history),
            "positions": [p.to_dict() for p in self.positions.values()],
        }

    def check_risk_limits(self, regime_limits: Dict = None) -> List[str]:
        """Check against risk limits, returns violations"""
        violations = []
        nav = self.get_nav()
        pos_value = self.get_position_value()

        # Max drawdown check (from peaks)
        if self.nav_history:
            peak_nav = max(p.nav for p in self.nav_history)
            dd_pct = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0
            if dd_pct < -0.10:
                violations.append(f"Drawdown {dd_pct:.1%} exceeds -10% circuit breaker")

        # Position count limit
        if regime_limits:
            max_pos = regime_limits.get("max_positions", 10)
            if len(self.positions) >= max_pos:
                violations.append(f"Position count {len(self.positions)} >= limit {max_pos}")

            # Single position cap
            single_pct = regime_limits.get("single_pct", 0.20)
            for sym, pos in self.positions.items():
                weight = pos.market_value / nav if nav > 0 else 0
                if weight > single_pct:
                    violations.append(f"{sym} weight {weight:.1%} > limit {single_pct:.0%}")

        return violations


class Benchmark:
    """Benchmark comparison utilities"""

    @staticmethod
    def compare_to_csi300(portfolio_nav: List[float], benchmark_nav: List[float]) -> Dict:
        """Compute alpha, beta, tracking error vs CSI300"""
        if len(portfolio_nav) < 2 or len(benchmark_nav) < 2:
            return {}

        port_returns = np.diff(portfolio_nav) / portfolio_nav[:-1]
        bench_returns = np.diff(benchmark_nav) / benchmark_nav[:-1]
        min_len = min(len(port_returns), len(bench_returns))
        port_returns = port_returns[-min_len:]
        bench_returns = bench_returns[-min_len:]

        excess = port_returns - bench_returns
        alpha = float(np.mean(excess) * 252)
        tracking_error = float(np.std(excess) * np.sqrt(252))

        cov = np.cov(port_returns, bench_returns)
        beta = cov[0][1] / cov[1][1] if cov.shape == (2, 2) and cov[1][1] > 0 else 1.0

        # Sharpe
        risk_free = 0.025 / 252
        sharpe = float((np.mean(port_returns) - risk_free) / np.std(port_returns) * np.sqrt(252))

        return {
            "alpha": round(alpha, 4),
            "beta": round(beta, 2),
            "sharpe": round(sharpe, 2),
            "tracking_error": round(tracking_error, 4),
            "information_ratio": round(alpha / tracking_error, 2) if tracking_error > 0 else 0,
        }

    @staticmethod
    def max_drawdown(nav_series: List[float]) -> float:
        """Compute maximum drawdown"""
        peaks = np.maximum.accumulate(nav_series)
        drawdowns = (np.array(nav_series) - peaks) / peaks
        return float(np.min(drawdowns))
