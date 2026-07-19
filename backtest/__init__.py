"""
回测引擎层 (Backtesting Engine)

核心组件:
- AShareBroker: A股交易规则完整模拟(T+1/涨跌停/印花税/佣金/过户费)
- EventDrivenBacktestEngine: 事件驱动回测引擎
- OverfittingGuard: 6层过拟合防控(PBO/DSR/Walk-Forward/参数敏感性/Monte Carlo)
- ImpactCostEngine: 平方根律冲击成本建模(v2.1)
- BacktestDiscipline: 48小时冷静期+密封评估(v2.1)
"""

from .broker import AShareBroker, Account, Order, OrderSide, OrderStatus, Position
from .engine import BacktestConfig, BacktestResult, EventDrivenBacktestEngine
from .impact import ImpactCostEngine, ImpactCostResult
from .overfitting import (
    BacktestDiscipline,
    OverfittingGuard,
    OverfittingReport,
    SealedEvaluation,
)

__all__ = [
    "AShareBroker",
    "Account",
    "Order",
    "OrderSide",
    "OrderStatus",
    "Position",
    "BacktestConfig",
    "BacktestResult",
    "EventDrivenBacktestEngine",
    "ImpactCostEngine",
    "ImpactCostResult",
    "OverfittingGuard",
    "OverfittingReport",
    "BacktestDiscipline",
    "SealedEvaluation",
]
