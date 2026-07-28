"""
持仓状态管理 + JSON持久化

数据模型: PaperPosition, PaperTrade, DailySnapshot, PortfolioState
管理器: PortfolioManager (原子读写, 默认状态创建)
"""

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# 默认状态文件路径
DEFAULT_STATE_PATH = Path(__file__).parent.parent / "simulation_data" / "portfolio.json"


@dataclass
class PaperPosition:
    """模拟持仓"""
    symbol: str
    name: str = ""
    quantity: int = 0
    avg_cost: float = 0.0
    total_cost: float = 0.0
    buy_date: str = ""
    current_price: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    rec_strategy_id: str = ""
    rec_strategy_name: str = ""
    rec_confidence: str = ""
    rec_win_rate: float = 0.0
    rec_quality_score: float = 0.0
    rec_reasons: List[str] = field(default_factory=list)        # AI分析的关键理由
    rec_verdict: str = ""                                         # AI综合判断
    rec_risks: List[str] = field(default_factory=list)           # AI提示的风险
    rec_entry_score: float = 0.0                                  # 入场评分(0-100)
    rec_entry_rationale: str = ""                                 # 入场逻辑简述

    @property
    def market_value(self) -> float:
        if self.current_price > 0:
            return self.quantity * self.current_price
        return self.quantity * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.total_cost

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.total_cost > 0:
            return (self.market_value / self.total_cost - 1) * 100
        return 0.0

    @property
    def allocation_pct(self) -> float:
        """返回持仓占总资产的比例 — 在PortfolioState中计算"""
        return 0.0


@dataclass
class PaperTrade:
    """模拟交易记录"""
    trade_id: str = ""
    date: str = ""
    time: str = ""
    symbol: str = ""
    name: str = ""
    side: str = ""  # "buy" | "sell"
    quantity: int = 0
    price: float = 0.0
    amount: float = 0.0
    commission: float = 0.0
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0
    net_amount: float = 0.0
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None  # "manual" | "stop_loss" | "take_profit" | "sell_signal"
    reason: str = ""  # 交易原因简述


@dataclass
class DailySnapshot:
    """每日快照"""
    date: str = ""
    total_value: float = 0.0
    cash: float = 0.0
    position_value: float = 0.0
    daily_pnl: float = 0.0
    daily_return_pct: float = 0.0
    cumulative_return_pct: float = 0.0
    positions_count: int = 0


@dataclass
class PortfolioState:
    """完整持仓状态"""
    initial_capital: float = 100000.0
    cash: float = 100000.0
    total_commission: float = 0.0
    total_stamp_duty: float = 0.0
    total_transfer_fee: float = 0.0
    total_realized_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    trade_history: List[PaperTrade] = field(default_factory=list)
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    last_analysis_date: str = ""
    pending_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @property
    def position_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.position_value

    @property
    def total_return(self) -> float:
        return self.total_value - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital > 0:
            return (self.total_value / self.initial_capital - 1) * 100
        return 0.0

    @property
    def invested_amount(self) -> float:
        return sum(p.total_cost for p in self.positions.values())

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        if total > 0:
            return self.win_count / total
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "meta": {
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "version": "1.0",
                "initial_capital": self.initial_capital,
            },
            "account": {
                "cash": self.cash,
                "total_commission": self.total_commission,
                "total_stamp_duty": self.total_stamp_duty,
                "total_transfer_fee": self.total_transfer_fee,
                "total_realized_pnl": self.total_realized_pnl,
                "trade_count": self.trade_count,
                "win_count": self.win_count,
                "loss_count": self.loss_count,
            },
            "positions": {
                sym: {
                    "symbol": p.symbol,
                    "name": p.name,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "total_cost": p.total_cost,
                    "buy_date": p.buy_date,
                    "current_price": p.current_price,
                    "entry_price": p.entry_price,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "rec_strategy_id": p.rec_strategy_id,
                    "rec_strategy_name": p.rec_strategy_name,
                    "rec_confidence": p.rec_confidence,
                    "rec_win_rate": p.rec_win_rate,
                    "rec_quality_score": p.rec_quality_score,
                    "rec_reasons": p.rec_reasons,
                    "rec_verdict": p.rec_verdict,
                    "rec_risks": p.rec_risks,
                    "rec_entry_score": p.rec_entry_score,
                    "rec_entry_rationale": p.rec_entry_rationale,
                }
                for sym, p in self.positions.items()
            },
            "trade_history": [
                {
                    "trade_id": t.trade_id,
                    "date": t.date,
                    "time": t.time,
                    "symbol": t.symbol,
                    "name": t.name,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "amount": t.amount,
                    "commission": t.commission,
                    "stamp_duty": t.stamp_duty,
                    "transfer_fee": t.transfer_fee,
                    "net_amount": t.net_amount,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                    "reason": t.reason,
                }
                for t in self.trade_history
            ],
            "daily_snapshots": [
                {
                    "date": s.date,
                    "total_value": s.total_value,
                    "cash": s.cash,
                    "position_value": s.position_value,
                    "daily_pnl": s.daily_pnl,
                    "daily_return_pct": s.daily_return_pct,
                    "cumulative_return_pct": s.cumulative_return_pct,
                    "positions_count": s.positions_count,
                }
                for s in self.daily_snapshots
            ],
            "last_analysis_date": self.last_analysis_date,
            "pending_recommendations": self.pending_recommendations,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortfolioState":
        """从字典反序列化"""
        meta = d.get("meta", {})
        acct = d.get("account", {})
        state = cls(
            initial_capital=meta.get("initial_capital", 10000.0),
            cash=acct.get("cash", 10000.0),
            total_commission=acct.get("total_commission", 0.0),
            total_stamp_duty=acct.get("total_stamp_duty", 0.0),
            total_transfer_fee=acct.get("total_transfer_fee", 0.0),
            total_realized_pnl=acct.get("total_realized_pnl", 0.0),
            trade_count=acct.get("trade_count", 0),
            win_count=acct.get("win_count", 0),
            loss_count=acct.get("loss_count", 0),
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
            last_analysis_date=d.get("last_analysis_date", ""),
        )

        # 恢复持仓
        for sym, pdata in d.get("positions", {}).items():
            state.positions[sym] = PaperPosition(
                symbol=pdata.get("symbol", sym),
                name=pdata.get("name", ""),
                quantity=pdata.get("quantity", 0),
                avg_cost=pdata.get("avg_cost", 0.0),
                total_cost=pdata.get("total_cost", 0.0),
                buy_date=pdata.get("buy_date", ""),
                current_price=pdata.get("current_price", 0.0),
                entry_price=pdata.get("entry_price", 0.0),
                stop_loss=pdata.get("stop_loss", 0.0),
                take_profit=pdata.get("take_profit", 0.0),
                rec_strategy_id=pdata.get("rec_strategy_id", ""),
                rec_strategy_name=pdata.get("rec_strategy_name", ""),
                rec_confidence=pdata.get("rec_confidence", ""),
                rec_win_rate=pdata.get("rec_win_rate", 0.0),
                rec_quality_score=pdata.get("rec_quality_score", 0.0),
                rec_reasons=pdata.get("rec_reasons", []),
                rec_verdict=pdata.get("rec_verdict", ""),
                rec_risks=pdata.get("rec_risks", []),
                rec_entry_score=pdata.get("rec_entry_score", 0.0),
                rec_entry_rationale=pdata.get("rec_entry_rationale", ""),
            )

        # 恢复交易历史
        for tdata in d.get("trade_history", []):
            state.trade_history.append(PaperTrade(
                trade_id=tdata.get("trade_id", ""),
                date=tdata.get("date", ""),
                time=tdata.get("time", ""),
                symbol=tdata.get("symbol", ""),
                name=tdata.get("name", ""),
                side=tdata.get("side", ""),
                quantity=tdata.get("quantity", 0),
                price=tdata.get("price", 0.0),
                amount=tdata.get("amount", 0.0),
                commission=tdata.get("commission", 0.0),
                stamp_duty=tdata.get("stamp_duty", 0.0),
                transfer_fee=tdata.get("transfer_fee", 0.0),
                net_amount=tdata.get("net_amount", 0.0),
                pnl=tdata.get("pnl"),
                pnl_pct=tdata.get("pnl_pct"),
                exit_reason=tdata.get("exit_reason"),
                reason=tdata.get("reason", ""),
            ))

        # 恢复快照
        for sdata in d.get("daily_snapshots", []):
            state.daily_snapshots.append(DailySnapshot(
                date=sdata.get("date", ""),
                total_value=sdata.get("total_value", 0.0),
                cash=sdata.get("cash", 0.0),
                position_value=sdata.get("position_value", 0.0),
                daily_pnl=sdata.get("daily_pnl", 0.0),
                daily_return_pct=sdata.get("daily_return_pct", 0.0),
                cumulative_return_pct=sdata.get("cumulative_return_pct", 0.0),
                positions_count=sdata.get("positions_count", 0),
            ))

        # 恢复待处理推荐
        state.pending_recommendations = d.get("pending_recommendations", [])

        return state


class PortfolioManager:
    """持仓状态管理器 — 原子JSON读写"""

    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = Path(filepath) if filepath else DEFAULT_STATE_PATH
        self._state: Optional[PortfolioState] = None

    @property
    def state(self) -> PortfolioState:
        if self._state is None:
            self._state = self.load()
        return self._state

    def load(self) -> PortfolioState:
        """加载持仓状态, 文件不存在时创建默认状态"""
        if not self.filepath.exists():
            logger.info(f"状态文件不存在, 创建默认状态: {self.filepath}")
            state = self._create_default()
            self.save(state)
            return state

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = PortfolioState.from_dict(data)
            logger.info(f"加载持仓状态: {state.position_count}只持仓, "
                       f"现金¥{state.cash:,.2f}, 总资产¥{state.total_value:,.2f}")
            return state
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"状态文件损坏: {e}, 重建默认状态")
            state = self._create_default()
            self.save(state)
            return state

    def save(self, state: Optional[PortfolioState] = None) -> None:
        """原子保存 (先写临时文件, 再替换)"""
        if state is None:
            state = self.state

        state.updated_at = datetime.now().isoformat()
        data = state.to_dict()

        # 确保目录存在
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # 原子写入
        tmp_path = self.filepath.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            shutil.move(str(tmp_path), str(self.filepath))
            self._state = state
            logger.debug(f"状态已保存: {self.filepath}")
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _create_default(self) -> PortfolioState:
        """创建默认状态 (¥100,000初始资金)"""
        now = datetime.now().isoformat()
        today = date.today().isoformat()
        state = PortfolioState(
            initial_capital=100000.0,
            cash=100000.0,
            created_at=now,
            updated_at=now,
        )
        # 添加初始快照
        state.daily_snapshots.append(DailySnapshot(
            date=today,
            total_value=100000.0,
            cash=100000.0,
            position_value=0.0,
            daily_pnl=0.0,
            daily_return_pct=0.0,
            cumulative_return_pct=0.0,
            positions_count=0,
        ))
        return state

    def reset(self) -> PortfolioState:
        """重置所有持仓 (清空)"""
        logger.warning("重置模拟交易状态!")
        state = self._create_default()
        self.save(state)
        return state
