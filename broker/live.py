"""
实盘交易抽象层 — 统一 Paper / QMT / 其他券商接口

使用:
    from broker.live import get_broker
    broker = get_broker("paper")   # 模拟交易
    broker = get_broker("qmt")     # QMT实盘 (需要xtquant)

所有 broker 实现统一的 TradeBroker 接口:
  - connect() / disconnect()
  - get_positions() / get_account()
  - submit_order() / cancel_order()
  - get_orders() / get_trades()
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class LiveOrder:
    """实盘订单"""
    symbol: str
    side: OrderSide
    quantity: int
    price: float = 0.0
    order_type: str = "limit"  # limit / market
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    submitted_at: str = ""
    filled_at: str = ""


@dataclass
class LivePosition:
    """实盘持仓"""
    symbol: str
    name: str = ""
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class AccountInfo:
    """账户信息"""
    account_id: str = ""
    total_asset: float = 0.0
    available_cash: float = 0.0
    frozen_cash: float = 0.0
    position_value: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0


# ═══════════════════════════════════════════════════════════════
# 抽象接口
# ═══════════════════════════════════════════════════════════════

class TradeBroker(ABC):
    """实盘交易接口 — 所有券商适配器必须实现"""

    @abstractmethod
    async def connect(self) -> bool:
        """连接交易系统"""
        ...

    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        ...

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """获取账户信息"""
        ...

    @abstractmethod
    async def get_positions(self) -> Dict[str, LivePosition]:
        """获取当前持仓"""
        ...

    @abstractmethod
    async def submit_order(self, symbol: str, side: OrderSide,
                          quantity: int, price: float = 0.0,
                          order_type: str = "limit") -> LiveOrder:
        """提交订单"""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        ...

    @abstractmethod
    async def get_orders(self) -> List[LiveOrder]:
        """获取当日订单"""
        ...

    @abstractmethod
    async def get_trades(self) -> List[LiveOrder]:
        """获取当日成交"""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @property
    @abstractmethod
    def broker_type(self) -> str:
        ...


# ═══════════════════════════════════════════════════════════════
# Paper Trading 适配器 (复用现有 PaperTradingEngine)
# ═══════════════════════════════════════════════════════════════

class PaperBroker(TradeBroker):
    """模拟交易适配器 — 包装 PaperTradingEngine"""

    def __init__(self):
        self._connected = False
        self._engine = None
        self._manager = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def broker_type(self) -> str:
        return "paper"

    async def connect(self) -> bool:
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine
        self._manager = PortfolioManager()
        self._engine = PaperTradingEngine(self._manager)
        self._connected = True
        logger.info("PaperBroker 已连接 (模拟交易)")
        return True

    async def disconnect(self):
        self._connected = False
        logger.info("PaperBroker 已断开")

    async def get_account(self) -> AccountInfo:
        if not self._engine:
            await self.connect()
        state = self._engine.state
        return AccountInfo(
            account_id="paper-001",
            total_asset=state.total_value,
            available_cash=state.cash,
            position_value=state.position_value,
            total_return=state.total_return,
            total_return_pct=state.total_return_pct,
        )

    async def get_positions(self) -> Dict[str, LivePosition]:
        if not self._engine:
            await self.connect()
        result = {}
        for sym, p in self._engine.state.positions.items():
            result[sym] = LivePosition(
                symbol=sym, name=p.name, quantity=p.quantity,
                avg_cost=p.avg_cost, current_price=p.current_price,
                market_value=p.market_value,
                unrealized_pnl=p.unrealized_pnl,
                unrealized_pnl_pct=p.unrealized_pnl_pct,
            )
        return result

    async def submit_order(self, symbol: str, side: OrderSide,
                          quantity: int, price: float = 0.0,
                          order_type: str = "limit") -> LiveOrder:
        if not self._engine:
            await self.connect()

        state = self._engine.state
        today = date.today().isoformat()

        if side == OrderSide.BUY:
            trade = self._engine.execute_buy(
                symbol=symbol, name=symbol.split(".")[-1], price=price,
            )
        else:
            trade = self._engine.execute_sell(
                symbol=symbol, quantity=quantity, price=price if price > 0 else None,
                exit_reason="api_order",
            )

        if trade is None:
            return LiveOrder(symbol=symbol, side=side, quantity=quantity,
                           price=price, status=OrderStatus.REJECTED,
                           order_id=f"PAPER-REJECTED-{today}")

        return LiveOrder(
            symbol=symbol, side=side, quantity=trade.quantity,
            price=trade.price, status=OrderStatus.FILLED,
            order_id=trade.trade_id, filled_qty=trade.quantity,
            filled_price=trade.price, commission=trade.commission,
            submitted_at=f"{trade.date}T{trade.time}",
            filled_at=f"{trade.date}T{trade.time}",
        )

    async def cancel_order(self, order_id: str) -> bool:
        logger.info(f"PaperBroker: 模拟模式不支持撤单 ({order_id})")
        return False

    async def get_orders(self) -> List[LiveOrder]:
        return []

    async def get_trades(self) -> List[LiveOrder]:
        if not self._engine:
            await self.connect()
        result = []
        for t in self._engine.state.trade_history[-20:]:
            result.append(LiveOrder(
                symbol=t.symbol,
                side=OrderSide.BUY if t.side == "buy" else OrderSide.SELL,
                quantity=t.quantity, price=t.price,
                status=OrderStatus.FILLED,
                order_id=t.trade_id, filled_qty=t.quantity,
                filled_price=t.price, commission=t.commission,
                submitted_at=f"{t.date}T{t.time}",
                filled_at=f"{t.date}T{t.time}",
            ))
        return result


# ═══════════════════════════════════════════════════════════════
# QMT / miniQMT 适配器
# ═══════════════════════════════════════════════════════════════

class QMTBroker(TradeBroker):
    """
    QMT/miniQMT 实盘适配器

    前置条件:
      1. 安装 xtquant: pip install xtquant (券商提供)
      2. 启动 QMT 客户端并登录
      3. 在 QMT 设置中开启 "极简模式" (miniQMT)

    使用:
      broker = QMTBroker(account_id="your-account", mini=True)
      await broker.connect()
      account = await broker.get_account()
    """

    def __init__(self, account_id: str = "", mini: bool = True):
        self._account_id = account_id or os.getenv("QMT_ACCOUNT_ID", "")
        self._mini = mini
        self._connected = False
        self._xt_trader = None
        self._xt_account = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def broker_type(self) -> str:
        return "qmt"

    async def connect(self) -> bool:
        try:
            from xtquant import xtdata, xttrader

            # miniQMT: 连接本地xtquant服务
            if self._mini:
                path = os.getenv("QMT_PATH", r"C:\国金QMT交易端\userdata_mini")
                session_id = int(os.getenv("QMT_SESSION_ID", "1"))
                self._xt_trader = xttrader.XtQuantTrader(path, session_id)
                self._xt_trader.start()
                ret = self._xt_trader.connect()
                if ret != 0:
                    logger.error(f"QMT连接失败: 返回码={ret}")
                    return False

                # 订阅账户
                accounts = self._xt_trader.query_stock_asset()
                if not accounts:
                    logger.error("QMT: 未找到账户")
                    return False
                self._xt_account = accounts[0]
                self._connected = True
                logger.info(f"QMT已连接 (miniQMT): 账户={self._xt_account.account_id}")
                return True
            else:
                logger.warning("QMT: 完整版需要手动登录QMT客户端后连接")
                return False

        except ImportError:
            logger.error("QMT: xtquant未安装。请联系券商获取并安装: pip install xtquant")
            return False
        except Exception as e:
            logger.error(f"QMT连接异常: {e}")
            return False

    async def disconnect(self):
        if self._xt_trader:
            self._xt_trader.stop()
        self._connected = False

    async def get_account(self) -> AccountInfo:
        if not self._connected:
            await self.connect()

        try:
            asset = self._xt_trader.query_stock_asset()
            if not asset:
                return AccountInfo()
            a = asset[0]
            return AccountInfo(
                account_id=str(getattr(a, 'account_id', '')),
                total_asset=float(getattr(a, 'total_asset', 0)),
                available_cash=float(getattr(a, 'cash', 0)),
                frozen_cash=float(getattr(a, 'frozen_cash', 0)),
                position_value=float(getattr(a, 'market_value', 0)),
            )
        except Exception as e:
            logger.error(f"QMT查询账户失败: {e}")
            return AccountInfo()

    async def get_positions(self) -> Dict[str, LivePosition]:
        if not self._connected:
            await self.connect()

        try:
            positions = self._xt_trader.query_stock_positions()
            result = {}
            for p in positions:
                code = str(getattr(p, 'stock_code', ''))
                sym = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
                result[sym] = LivePosition(
                    symbol=sym,
                    name=str(getattr(p, 'stock_name', '')),
                    quantity=int(getattr(p, 'volume', 0)),
                    avg_cost=float(getattr(p, 'cost_price', 0)),
                    current_price=float(getattr(p, 'last_price', 0)),
                    market_value=float(getattr(p, 'market_value', 0)),
                    unrealized_pnl=float(getattr(p, 'profit', 0)),
                    unrealized_pnl_pct=float(getattr(p, 'profit_ratio', 0)),
                )
            return result
        except Exception as e:
            logger.error(f"QMT查询持仓失败: {e}")
            return {}

    async def submit_order(self, symbol: str, side: OrderSide,
                          quantity: int, price: float = 0.0,
                          order_type: str = "limit") -> LiveOrder:
        if not self._connected:
            await self.connect()

        try:
            code = symbol.replace("sh.", "").replace("sz.", "")
            order_id = self._xt_trader.order_stock(
                stock_code=code,
                order_type=xtquant.xtconstant.STOCK_BUY if side == OrderSide.BUY
                else xtquant.xtconstant.STOCK_SELL,
                order_volume=quantity,
                price=price if price > 0 else 0,
                strategy_name="ashare_ai_trader",
                remark="AI自动下单",
            )
            return LiveOrder(
                symbol=symbol, side=side, quantity=quantity, price=price,
                order_id=str(order_id), status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"QMT下单失败: {e}")
            return LiveOrder(symbol=symbol, side=side, quantity=quantity,
                           price=price, status=OrderStatus.REJECTED)

    async def cancel_order(self, order_id: str) -> bool:
        if not self._connected:
            return False
        try:
            self._xt_trader.cancel_order_stock(int(order_id))
            return True
        except Exception as e:
            logger.error(f"QMT撤单失败: {e}")
            return False

    async def get_orders(self) -> List[LiveOrder]:
        if not self._connected:
            return []
        try:
            return []  # QMT order query需要通过回调异步获取
        except Exception:
            return []

    async def get_trades(self) -> List[LiveOrder]:
        if not self._connected:
            return []
        try:
            return []  # QMT trade query需要通过回调异步获取
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════
# Broker 工厂
# ═══════════════════════════════════════════════════════════════

_broker_instance: Optional[TradeBroker] = None


def get_broker(broker_type: str = "paper", **kwargs) -> TradeBroker:
    """获取交易接口实例 (单例)"""
    global _broker_instance

    if _broker_instance is not None:
        return _broker_instance

    broker_type = broker_type or os.getenv("BROKER_TYPE", "paper")

    if broker_type == "qmt":
        _broker_instance = QMTBroker(**kwargs)
    else:
        _broker_instance = PaperBroker()

    return _broker_instance
