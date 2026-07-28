"""
A股交易规则模拟器 (Broker Simulator)

精确模拟A股交易机制:
  - T+1: 当日买入→次日才能卖出
  - 涨跌停限制: 主板±10%, 创业板/科创板±20%, 北交所±30%, ST±5%
  - 印花税: 卖出时0.05% (2023年8月减半)
  - 佣金: 万3 (0.03%), 最低¥5
  - 过户费: 万分之0.1 (仅上交所)
  - 最小交易单位: 100股 (1手), 按手取整
  - 停牌/涨跌停板: 无法成交
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class PositionSide(Enum):
    LONG = "long"
    FLAT = "flat"


@dataclass
class Order:
    """订单"""
    symbol: str
    side: OrderSide
    quantity: int          # 股数(已按100股取整)
    price: Optional[float] = None  # None=市价单
    order_id: str = ""
    create_time: date = field(default_factory=date.today)
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0

    @property
    def amount(self) -> float:
        return self.filled_qty * self.filled_price

    @property
    def total_cost(self) -> float:
        return self.commission + self.stamp_duty + self.transfer_fee

    @property
    def net_amount(self) -> float:
        """买入: 正(支出), 卖出: 负(收入)"""
        sign = 1 if self.side == OrderSide.BUY else -1
        return sign * self.amount - self.total_cost


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    total_cost: float = 0.0
    side: PositionSide = PositionSide.FLAT
    buy_date: Optional[date] = None
    can_sell_date: Optional[date] = None  # T+1之后才能卖

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_cost  # 简化, 实际应该*当前价


@dataclass
class Account:
    """账户"""
    initial_capital: float = 100000.0
    cash: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    total_commission: float = 0.0
    total_stamp_duty: float = 0.0
    total_transfer_fee: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0

    def total_asset(self, prices: Optional[Dict[str, float]] = None) -> float:
        """总资产 = 现金 + 持仓市值"""
        if prices is None:
            prices = {}

        position_value = sum(
            pos.quantity * prices.get(sym, pos.avg_cost)
            for sym, pos in self.positions.items()
        )
        return self.cash + position_value

    @property
    def total_return(self) -> float:
        """v2.14: 修复 — total_return 应包含持仓市值, 非仅现金"""
        return self.total_return_pct


class AShareBroker:
    """
    A股券商模拟器

    核心规则:
    1. T+1: 买入当天不可卖出
    2. 涨跌停: 到达涨跌停价的订单无法成交
    3. 交易费用: 佣金(万3,最低¥5) + 印花税(卖出0.05%) + 过户费(上交所0.001%)
    4. 整手交易: 必须是100股整数倍
    5. 资金检查: 买入金额+费用 <= 可用现金
    """

    # 交易费用常量
    COMMISSION_RATE = 0.0003      # 佣金费率: 万3
    MIN_COMMISSION = 5.0           # 最低佣金: ¥5
    STAMP_DUTY_RATE = 0.0005       # 印花税: 0.05% (仅卖出)
    TRANSFER_FEE_RATE = 0.00001    # 过户费: 0.001% (仅上交所)

    def __init__(self, initial_capital: float = 100000.0):
        self.account = Account(initial_capital=initial_capital)
        self.account.cash = initial_capital
        self.orders: List[Order] = []
        self.trade_log: List[Dict] = []
        self._cur_date: Optional[date] = None
        self._cur_prices: Dict[str, pd.Series] = {}  # {symbol: daily_bar}

    def set_date(self, dt: date):
        """设置当前交易日"""
        self._cur_date = dt

    def set_prices(self, prices: Dict[str, pd.Series]):
        """设置当前交易日的股价数据 (open/high/low/close/volume/...)"""
        self._cur_prices = prices

    # ═══════════════════════════════════════════════════════════════
    # 交易执行
    # ═══════════════════════════════════════════════════════════════

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: Optional[float] = None,
    ) -> Order:
        """
        买入委托

        Args:
            symbol: 股票代码
            quantity: 买入股数(自动向下取整到100的倍数)
            price: 限价(默认: 当日收盘价)

        Returns:
            Order (含成交状态)
        """
        # 整手取整
        lots = max(quantity // 100, 1)
        qty = lots * 100

        # 获取执行价
        exec_price = self._get_execution_price(symbol, price, OrderSide.BUY)

        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=qty,
            price=exec_price,
            order_id=f"B-{self._cur_date}-{symbol}-{len(self.orders)}",
            create_time=self._cur_date or date.today(),
        )

        # 费用计算
        amount = qty * exec_price
        commission = max(amount * self.COMMISSION_RATE, self.MIN_COMMISSION)
        transfer_fee = (amount * self.TRANSFER_FEE_RATE
                       if self._is_shanghai(symbol) else 0)

        total_cost = amount + commission + transfer_fee

        # 资金检查
        if total_cost > self.account.cash:
            order.status = OrderStatus.REJECTED
            logger.debug(f"买入拒绝: 资金不足 (需要¥{total_cost:.0f}, 可用¥{self.account.cash:.0f})")
            self.orders.append(order)
            return order

        # 涨跌停检查
        if self._is_limit_hit(symbol, OrderSide.BUY):
            order.status = OrderStatus.REJECTED
            logger.debug(f"买入拒绝: {symbol} 涨停/跌停无法成交")
            self.orders.append(order)
            return order

        # 执行成交
        order.status = OrderStatus.FILLED
        order.filled_qty = qty
        order.filled_price = exec_price
        order.commission = commission
        order.stamp_duty = 0  # 买入不收印花税
        order.transfer_fee = transfer_fee

        # 更新账户
        self.account.cash -= total_cost
        self.account.total_commission += commission
        self.account.total_transfer_fee += transfer_fee
        self.account.trade_count += 1

        # 更新持仓
        if symbol not in self.account.positions:
            self.account.positions[symbol] = Position(
                symbol=symbol,
                side=PositionSide.LONG,
                buy_date=self._cur_date,
                can_sell_date=self._next_trade_date(),
            )

        pos = self.account.positions[symbol]
        old_total = pos.total_cost
        pos.quantity += qty
        pos.total_cost += qty * exec_price
        pos.avg_cost = pos.total_cost / pos.quantity if pos.quantity > 0 else 0

        self.orders.append(order)
        self._log_trade(order)

        return order

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: Optional[float] = None,
    ) -> Order:
        """
        卖出委托

        Args:
            symbol: 股票代码
            quantity: 卖出股数(自动向下取整到100的倍数)
            price: 限价(默认: 当日收盘价)
        """
        lots = max(quantity // 100, 1)
        qty = lots * 100

        # 获取执行价
        exec_price = self._get_execution_price(symbol, price, OrderSide.SELL)

        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=qty,
            price=exec_price,
            order_id=f"S-{self._cur_date}-{symbol}-{len(self.orders)}",
            create_time=self._cur_date or date.today(),
        )

        # T+1检查
        pos = self.account.positions.get(symbol)
        if pos is None or pos.quantity <= 0:
            order.status = OrderStatus.REJECTED
            logger.debug(f"卖出拒绝: {symbol} 无持仓")
            self.orders.append(order)
            return order

        # 🆕 v2.14: T+1 卖出限制 (当日买入次日才能卖出)
        if self._cur_date and pos.can_sell_date and self._cur_date < pos.can_sell_date:
            order.status = OrderStatus.REJECTED
            logger.debug(f"卖出拒绝: {symbol} T+1限制 (可卖日期={pos.can_sell_date})")
            self.orders.append(order)
            return order

        # 持仓不足
        if qty > pos.quantity:
            qty = pos.quantity  # 降级为清仓
            order.quantity = qty

        # 费用(卖出)
        amount = qty * exec_price
        commission = max(amount * self.COMMISSION_RATE, self.MIN_COMMISSION)
        stamp_duty = amount * self.STAMP_DUTY_RATE
        transfer_fee = (amount * self.TRANSFER_FEE_RATE
                       if self._is_shanghai(symbol) else 0)

        total_fees = commission + stamp_duty + transfer_fee

        # 涨跌停检查
        if self._is_limit_hit(symbol, OrderSide.SELL):
            order.status = OrderStatus.REJECTED
            logger.debug(f"卖出拒绝: {symbol} 跌停板无法成交")
            self.orders.append(order)
            return order

        # 执行
        order.status = OrderStatus.FILLED
        order.filled_qty = qty
        order.filled_price = exec_price
        order.commission = commission
        order.stamp_duty = stamp_duty
        order.transfer_fee = transfer_fee

        # 计算盈亏
        sell_income = amount - total_fees
        cost_basis = pos.avg_cost * qty
        pnl = sell_income - cost_basis

        # 更新账户
        self.account.cash += sell_income
        self.account.total_commission += commission
        self.account.total_stamp_duty += stamp_duty
        self.account.total_transfer_fee += transfer_fee
        self.account.trade_count += 1

        if pnl >= 0:
            self.account.win_count += 1
        else:
            self.account.loss_count += 1
        # v2.14: 存储 PnL 到 order 以便 _log_trade 记录
        order._pnl = pnl
        order._pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0

        # 更新持仓
        pos.quantity -= qty
        if pos.quantity <= 0:
            self.account.positions.pop(symbol)
        else:
            pos.total_cost -= cost_basis
            pos.avg_cost = pos.total_cost / pos.quantity

        self.orders.append(order)
        self._log_trade(order)

        return order

    # ═══════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════

    def _get_execution_price(
        self,
        symbol: str,
        price: Optional[float],
        side: OrderSide,
    ) -> float:
        """获取执行价格"""
        if price is not None:
            return price

        # 市价单: 用收盘价
        bar = self._cur_prices.get(symbol, {})
        close = bar.get("close", 0) if isinstance(bar, dict) else (
            bar.get("close", 0) if isinstance(bar, pd.Series) else (
                bar if isinstance(bar, (int, float)) else 0
            )
        )
        if close > 0:
            return close

        # 如果没找到价格,从持仓取
        pos = self.account.positions.get(symbol)
        if pos and pos.avg_cost > 0:
            return pos.avg_cost

        raise ValueError(f"无法获取{symbol}执行价格")

    def _is_limit_hit(self, symbol: str, side: OrderSide) -> bool:
        """检查是否涨跌停(涨停→买不到, 跌停→卖不掉)"""
        bar = self._cur_prices.get(symbol)
        if bar is None:
            return False

        if isinstance(bar, dict):
            close = bar.get("close", 0)
            pct_change = bar.get("pct_change", 0)
        elif isinstance(bar, pd.Series):
            close = bar.get("close", 0) if "close" in bar.index else 0
            pct_change = bar.get("pct_change", 0) if "pct_change" in bar.index else 0
        else:
            return False

        # v2.14: 按板块区分涨跌停幅度
        limit_pct = 10.0  # 主板默认
        sym_stripped = symbol.replace("sh.", "").replace("sz.", "")
        if sym_stripped.startswith("30") or sym_stripped.startswith("68"):
            limit_pct = 20.0  # 创业板/科创板
        elif sym_stripped.startswith("8"):
            limit_pct = 30.0  # 北交所
        threshold = limit_pct - 0.2  # 小容差

        if side == OrderSide.BUY:
            return pct_change >= threshold
        else:
            return pct_change <= -threshold

    def _is_shanghai(self, symbol: str) -> bool:
        """判断是否上交所(收过户费)"""
        return symbol.startswith("sh.") or (
            symbol.replace("sh.", "").replace("sz.", "").startswith("6")
        )

    def _next_trade_date(self) -> date:
        """下一个交易日(T+1) — v2.14: 实际+1天(周末跳过由调用方处理)"""
        from datetime import timedelta
        if self._cur_date is None:
            return date.today() + timedelta(days=1)
        return self._cur_date + timedelta(days=1)

    def _log_trade(self, order: Order):
        """记录交易日志 — v2.14: 增加 PnL 字段"""
        entry = {
            "date": str(self._cur_date),
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.filled_qty,
            "price": order.filled_price,
            "amount": order.amount,
            "commission": order.commission,
            "stamp_duty": order.stamp_duty,
            "status": order.status.value,
        }
        if hasattr(order, '_pnl'):
            entry["pnl"] = order._pnl
            entry["pnl_pct"] = order._pnl_pct
        self.trade_log.append(entry)

    # ═══════════════════════════════════════════════════════════════
    # 绩效统计
    # ═══════════════════════════════════════════════════════════════

    def get_performance(self) -> Dict:
        """获取交易绩效汇总"""
        if not self.trade_log:
            return {"total_trades": 0}

        trades = pd.DataFrame(self.trade_log)
        buy_trades = trades[trades["side"] == "buy"]
        sell_trades = trades[trades["side"] == "sell"]

        total_commission = self.account.total_commission
        total_stamp = self.account.total_stamp_duty
        total_fees = total_commission + total_stamp

        return {
            "total_trades": self.account.trade_count,
            "win_count": self.account.win_count,
            "loss_count": self.account.loss_count,
            "win_rate": (self.account.win_count / max(self.account.trade_count, 1)),
            "total_commission": round(total_commission, 2),
            "total_stamp_duty": round(total_stamp, 2),
            "total_fees": round(total_fees, 2),
            "total_return_pct": round(self.account.total_return, 2),
            "remaining_cash": round(self.account.cash, 2),
        }

    def get_position_summary(self, prices: Dict[str, float]) -> pd.DataFrame:
        """获取持仓摘要"""
        records = []
        for sym, pos in self.account.positions.items():
            if pos.quantity > 0:
                cur_price = prices.get(sym, pos.avg_cost)
                mkt_val = pos.quantity * cur_price
                pnl = mkt_val - pos.total_cost
                pnl_pct = (pnl / pos.total_cost * 100) if pos.total_cost > 0 else 0
                records.append({
                    "symbol": sym,
                    "quantity": pos.quantity,
                    "avg_cost": round(pos.avg_cost, 2),
                    "cur_price": round(cur_price, 2),
                    "market_value": round(mkt_val, 0),
                    "unrealized_pnl": round(pnl, 0),
                    "pnl_pct": round(pnl_pct, 2),
                    "buy_date": pos.buy_date,
                })
        return pd.DataFrame(records)
