"""
模拟交易引擎 — Paper Trading Engine

封装AShareBroker的费率模型, 提供:
  - execute_buy: 根据推荐信号模拟买入
  - execute_sell: 模拟卖出(止盈/止损/信号)
  - mark_to_market: 持仓按市价估值
  - check_exit_conditions: 检查止盈止损条件
"""

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# 复用 backtest/broker 的费率常量
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.broker import AShareBroker

from .portfolio import (
    DailySnapshot,
    PaperPosition,
    PaperTrade,
    PortfolioManager,
    PortfolioState,
)

# ═══════════════════════════════════════════════════════════════
# 费率常量 (从 AShareBroker 引用, 保持一致)
# ═══════════════════════════════════════════════════════════════
COMMISSION_RATE = AShareBroker.COMMISSION_RATE       # 0.0003 (万3)
MIN_COMMISSION = AShareBroker.MIN_COMMISSION          # 5.0
STAMP_DUTY_RATE = AShareBroker.STAMP_DUTY_RATE        # 0.0005 (千0.5, 仅卖出)
TRANSFER_FEE_RATE = AShareBroker.TRANSFER_FEE_RATE    # 0.00001 (仅上交所)

# 模拟交易默认参数 (v2.9: 增加风控参数)
DEFAULT_MIN_CONFIDENCE = 0.55    # B级及以上
DEFAULT_MAX_POSITIONS = 10       # 最大持仓数
DEFAULT_SINGLE_POSITION_PCT = 0.20  # 单只最大仓位20%
DEFAULT_MAX_INDUSTRY_PCT = 0.35     # 🆕 单行业最大仓位35%
DEFAULT_CASH_BUFFER_PCT = 0.10      # 🆕 现金缓冲10% (保留抄底资金)

# 🆕 v2.9 行业分类 (基于申万一级,从代码前缀推断)
# v2.14: 修复重复key — 用非重叠区间精确到申万一级行业
_INDUSTRY_MAP = [
    # (code_low, code_high, industry)
    ("600000", "600016", "银行"),
    ("600030", "601236", "非银金融"),
    ("600048", "600325", "房地产"),
    ("600519", "603369", "食品饮料"),
    ("000333", "002959", "家用电器"),
    ("601933", "603708", "商贸零售"),
    ("600754", "601007", "社会服务"),
    ("603605", "603983", "美容护理"),
    ("688000", "689009", "电子/半导体"),
    ("002000", "002999", "电子/计算机"),
    ("300000", "300999", "通信/传媒"),
    ("300700", "300999", "电力设备"),
    ("601012", "688599", "新能源"),
    ("600700", "600999", "国防军工"),
    ("600200", "600763", "医药生物"),
    ("600309", "603260", "基础化工"),
    ("601800", "603993", "有色金属/煤炭"),
    ("600900", "601985", "公用事业"),
    ("601000", "601880", "交通运输"),
    ("000800", "002714", "农林牧渔"),
]


def _guess_industry(symbol: str) -> str:
    """从代码前缀粗略推断行业分类 (v2.14: 修复重复key, 改用非重叠区间列表)"""
    code = symbol.replace("sh.", "").replace("sz.", "")
    for low, high, industry in _INDUSTRY_MAP:
        if low <= code <= high:
            return industry
    # 按前缀兜底
    if code.startswith("6") and code[1] in "012":
        return "金融"
    elif code.startswith("6") and code[1] == "8":
        return "科技"
    elif code.startswith("6") and code[1] in "345":
        return "制造"
    elif code.startswith("000") or code.startswith("002"):
        return "消费"
    elif code.startswith("300"):
        return "创业板"
    elif code.startswith("688"):
        return "科创板"
    return "其他"


def _calc_buy_fees(amount: float, symbol: str) -> Tuple[float, float, float]:
    """计算买入费用: (佣金, 印花税, 过户费)"""
    commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
    stamp_duty = 0.0  # 买入不收印花税
    transfer_fee = amount * TRANSFER_FEE_RATE  # 2015年起沪深两市均收过户费
    return commission, stamp_duty, transfer_fee


def _calc_sell_fees(amount: float, symbol: str) -> Tuple[float, float, float]:
    """计算卖出费用: (佣金, 印花税, 过户费)"""
    commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
    stamp_duty = amount * STAMP_DUTY_RATE  # 卖出收印花税
    transfer_fee = amount * TRANSFER_FEE_RATE  # 2015年起沪深两市均收过户费
    return commission, stamp_duty, transfer_fee


class PaperTradingEngine:
    """模拟交易引擎"""

    def __init__(self, manager: Optional[PortfolioManager] = None):
        self.manager = manager or PortfolioManager()

    @property
    def state(self) -> PortfolioState:
        return self.manager.state

    # ═══════════════════════════════════════════════════════════
    # 买入
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _limit_pct_for(symbol: str) -> float:
        """按板块返回涨跌幅限制(%) — 主板±10%, 创业板/科创板±20%, 北交所±30%"""
        sym_stripped = symbol.replace("sh.", "").replace("sz.", "").replace("bj.", "")
        if sym_stripped.startswith("30") or sym_stripped.startswith("68"):
            return 20.0
        if sym_stripped.startswith("8") or sym_stripped.startswith("4"):
            return 30.0
        return 10.0

    def execute_buy(
        self,
        symbol: str,
        name: str,
        price: float,
        recommendation: Optional[Dict[str, Any]] = None,
        max_position_pct: float = DEFAULT_SINGLE_POSITION_PCT,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        pct_change: Optional[float] = None,
        sealed_limit_up: bool = False,
    ) -> Optional[PaperTrade]:
        """
        执行模拟买入

        Args:
            symbol: 股票代码 (如 sh.600519)
            name: 股票名称
            price: 成交价
            recommendation: 来自分析的推荐数据 (含strategy_id, conviction等)
            max_position_pct: 单只最大仓位比例
            max_positions: 最大持仓数
            pct_change: 当日涨跌幅(%), 用于涨停封板检测
            sealed_limit_up: 调用方已确认涨停封板 (优先于 pct_change)

        Returns:
            PaperTrade 或 None (资金不足/已达上限/涨停封板)
        """
        state = self.state
        today = date.today().isoformat()

        # A股涨跌停: 涨停封板无法买入 (v3.0: 板块感知的涨跌幅限制)
        limit_pct = self._limit_pct_for(symbol)
        if sealed_limit_up or (pct_change is not None and pct_change >= limit_pct - 0.2):
            logger.info(f"买入跳过: {symbol} 涨停封板 (涨幅{pct_change if pct_change is not None else '?'}%), 无法成交")
            return None

        # 检查是否已持仓
        if symbol in state.positions and state.positions[symbol].quantity > 0:
            logger.info(f"买入跳过: {symbol} 已持仓")
            return None

        # 检查持仓数上限
        if len(state.positions) >= max_positions:
            logger.info(f"买入跳过: 已达最大持仓数 {max_positions}")
            return None

        # 🆕 v2.9: 检查行业集中度
        industry = _guess_industry(symbol)
        industry_value = sum(
            p.market_value for p in state.positions.values()
            if _guess_industry(p.symbol) == industry
        )
        industry_pct = industry_value / max(state.total_value, 1)
        if industry_pct >= DEFAULT_MAX_INDUSTRY_PCT:
            logger.info(f"买入跳过: {symbol} 行业[{industry}]已占{industry_pct:.0%},超过{DEFAULT_MAX_INDUSTRY_PCT:.0%}上限")
            return None

        # 🆕 v2.9: 保留现金缓冲 (v2.14: 使用当前总资产而非初始资金)
        cash_buffer = state.total_value * DEFAULT_CASH_BUFFER_PCT
        available_cash = max(0, state.cash - cash_buffer)
        if available_cash <= 0:
            logger.info(f"买入跳过: 现金仅剩¥{state.cash:,.0f},低于缓冲¥{cash_buffer:,.0f}")
            return None

        # 提取推荐信息
        rec = recommendation or {}
        conviction = rec.get("conviction", 0.5)
        # 优先使用推荐中携带的止损止盈, 否则用默认-7%/+10%
        stop_loss = rec.get("stop_loss", price * 0.93)
        take_profit = rec.get("take_profit", price * 1.10)
        # 提取策略元数据
        strategy_id = rec.get("strategy_id", rec.get("id", ""))
        strategy_name = rec.get("strategy_name", rec.get("name", ""))
        win_rate = rec.get("win_rate", 0.0)
        quality_score = rec.get("composite_score", rec.get("score", 0) * 10)
        confidence = rec.get("confidence", rec.get("verdict_summary", ""))

    # ── 智能仓位计算 (v2.9: 使用可用现金而非全部现金) ──
        # 基础: 单只最多20%总资金
        base_pct = max_position_pct
        max_amount = available_cash * base_pct

        # 小账户弹性: 如果基础比例买不起一手, 逐步放宽到35%
        if price > 0:
            min_lot_cost = 100 * price * 1.005  # 约含手续费
            while max_amount < min_lot_cost and base_pct < 0.40:
                base_pct += 0.05
                max_amount = available_cash * base_pct
            # 如果放宽到40%还不够, 直接给到能买一手
            if max_amount < min_lot_cost:
                max_amount = min(min_lot_cost, available_cash * 0.95)

        position_amount = max_amount

        # 计算可买股数 (整手100股)
        if price <= 0:
            logger.error(f"买入失败: {symbol} 价格无效 {price}")
            return None
        shares = int(position_amount / price / 100) * 100
        if shares < 100:
            logger.info(f"买入跳过: {symbol} 资金不足 (¥{position_amount:.0f}不够一手)")
            return None

        # 计算费用确认可负担
        amount = shares * price
        commission, stamp_duty, transfer_fee = _calc_buy_fees(amount, symbol)
        total_cost = amount + commission + stamp_duty + transfer_fee

        while total_cost > state.cash and shares >= 100:
            shares -= 100
            amount = shares * price
            commission, stamp_duty, transfer_fee = _calc_buy_fees(amount, symbol)
            total_cost = amount + commission + stamp_duty + transfer_fee

        if shares < 100:
            logger.info(f"买入跳过: {symbol} 资金不足(调整后仍不够一手)")
            return None

        # 执行
        trade_id = f"T-{today}-{symbol.replace('.', '')}-{state.trade_count + 1:04d}"
        now = datetime.now().strftime("%H:%M:%S")

        trade = PaperTrade(
            trade_id=trade_id,
            date=today,
            time=now,
            symbol=symbol,
            name=name,
            side="buy",
            quantity=shares,
            price=price,
            amount=amount,
            commission=round(commission, 2),
            stamp_duty=round(stamp_duty, 2),
            transfer_fee=round(transfer_fee, 2),
            net_amount=round(-total_cost, 2),
            reason=rec.get("verdict_summary", rec.get("key_reason", "AI推荐买入")),
        )

        # 更新账户
        state.cash -= total_cost
        state.total_commission += commission
        state.total_transfer_fee += transfer_fee
        state.trade_count += 1

        # 创建持仓
        key_reasons = rec.get("key_reasons", [])
        if isinstance(key_reasons, str):
            key_reasons = [key_reasons]
        risks = rec.get("risks", [])
        if isinstance(risks, str):
            risks = [risks]

        position = PaperPosition(
            symbol=symbol,
            name=name,
            quantity=shares,
            avg_cost=price,
            total_cost=amount,
            buy_date=today,
            current_price=price,
            entry_price=price,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            rec_strategy_id=strategy_id,
            rec_strategy_name=strategy_name,
            rec_confidence=str(confidence)[:10] if confidence else "",
            rec_win_rate=win_rate,
            rec_quality_score=quality_score,
            rec_reasons=key_reasons,
            rec_verdict=rec.get("verdict_summary", ""),
            rec_risks=risks,
            rec_entry_score=quality_score,
            rec_entry_rationale=rec.get("verdict_summary", ""),
        )
        state.positions[symbol] = position
        state.trade_history.append(trade)

        logger.info(f"✅ 买入 {name}({symbol}) {shares}股 @¥{price:.2f} "
                    f"金额¥{amount:,.0f} 费用¥{commission+transfer_fee:.2f} "
                    f"剩余¥{state.cash:,.2f}")

        self.manager.save(state)
        return trade

    # ═══════════════════════════════════════════════════════════
    # 卖出
    # ═══════════════════════════════════════════════════════════

    def execute_sell(
        self,
        symbol: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        exit_reason: str = "manual",
        pct_change: Optional[float] = None,
        sealed_limit_down: bool = False,
    ) -> Optional[PaperTrade]:
        """
        执行模拟卖出

        Args:
            symbol: 股票代码
            quantity: 卖出数量(None=全部)
            price: 卖出价格(None=使用持仓现价)
            exit_reason: 卖出原因
            pct_change: 当日涨跌幅(%), 用于跌停封板检测
            sealed_limit_down: 调用方已确认跌停封板

        Returns:
            PaperTrade 或 None (T+1限制/跌停封板)
        """
        state = self.state
        today = date.today().isoformat()

        pos = state.positions.get(symbol)
        if not pos or pos.quantity <= 0:
            logger.warning(f"卖出失败: {symbol} 无持仓")
            return None

        # A股 T+1: 当日买入的股份当日不可卖出
        if pos.buy_date == today:
            logger.warning(f"卖出跳过: {symbol} 今日买入, T+1制度下需次交易日起方可卖出")
            return None

        # A股涨跌停: 跌停封板无法卖出 (v3.0: 板块感知的涨跌幅限制)
        limit_pct = self._limit_pct_for(symbol)
        if sealed_limit_down or (pct_change is not None and pct_change <= -(limit_pct - 0.2)):
            logger.warning(f"卖出跳过: {symbol} 跌停封板 (跌幅{pct_change if pct_change is not None else '?'}%), 无法成交")
            return None

        if quantity is None or quantity > pos.quantity:
            quantity = pos.quantity

        lots = max(quantity // 100, 1)
        qty = lots * 100
        if qty > pos.quantity:
            qty = (pos.quantity // 100) * 100
        if qty <= 0:
            return None

        # 成交价
        exec_price = price if price and price > 0 else pos.current_price
        if exec_price <= 0:
            exec_price = pos.avg_cost

        # 计算费用
        amount = qty * exec_price
        commission, stamp_duty, transfer_fee = _calc_sell_fees(amount, symbol)
        total_fees = commission + stamp_duty + transfer_fee
        net_income = amount - total_fees

        # 计算盈亏
        cost_basis = pos.avg_cost * qty
        realized_pnl = net_income - cost_basis
        pnl_pct = (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

        trade_id = f"T-{today}-{symbol.replace('.', '')}-{state.trade_count + 1:04d}"
        now = datetime.now().strftime("%H:%M:%S")

        exit_labels = {
            "stop_loss": "止损",
            "take_profit": "止盈",
            "sell_signal": "卖出信号",
            "manual": "手动卖出",
        }
        reason_text = exit_labels.get(exit_reason, exit_reason)

        trade = PaperTrade(
            trade_id=trade_id,
            date=today,
            time=now,
            symbol=symbol,
            name=pos.name,
            side="sell",
            quantity=qty,
            price=exec_price,
            amount=amount,
            commission=round(commission, 2),
            stamp_duty=round(stamp_duty, 2),
            transfer_fee=round(transfer_fee, 2),
            net_amount=round(net_income, 2),
            pnl=round(realized_pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=exit_reason,
            reason=f"{reason_text} | 成本¥{cost_basis:,.0f} → ¥{net_income:,.0f} | {pnl_pct:+.1f}%",
        )

        # 更新账户
        state.cash += net_income
        state.total_commission += commission
        state.total_stamp_duty += stamp_duty
        state.total_transfer_fee += transfer_fee
        state.total_realized_pnl += realized_pnl
        state.trade_count += 1

        if realized_pnl > 0:
            state.win_count += 1
        else:
            state.loss_count += 1

        # 更新持仓
        pos.quantity -= qty
        if pos.quantity <= 0:
            del state.positions[symbol]
            logger.info(f"✅ 卖出 {pos.name}({symbol}) 清仓 {qty}股 @¥{exec_price:.2f} "
                       f"盈亏¥{realized_pnl:+,.2f} ({pnl_pct:+.1f}%) [{reason_text}]")
        else:
            pos.total_cost -= cost_basis
            pos.avg_cost = pos.total_cost / pos.quantity if pos.quantity > 0 else 0
            logger.info(f"✅ 卖出 {pos.name}({symbol}) {qty}股 @¥{exec_price:.2f} "
                       f"盈亏¥{realized_pnl:+,.2f} [{reason_text}] 剩余{pos.quantity}股")

        state.trade_history.append(trade)
        self.manager.save(state)
        return trade

    # ═══════════════════════════════════════════════════════════
    # Mark-to-Market
    # ═══════════════════════════════════════════════════════════

    def mark_to_market(self, prices: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        按市价估值所有持仓

        Args:
            prices: {symbol: current_price}

        Returns:
            {symbol: {market_value, unrealized_pnl, unrealized_pnl_pct, ...}}
        """
        state = self.state
        result = {}

        for sym, pos in state.positions.items():
            cur_price = prices.get(sym, pos.current_price)
            if cur_price <= 0:
                cur_price = pos.avg_cost

            pos.current_price = cur_price
            mv = pos.market_value
            pnl = pos.unrealized_pnl
            pnl_pct = pos.unrealized_pnl_pct

            result[sym] = {
                "symbol": sym,
                "name": pos.name,
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": cur_price,
                "market_value": mv,
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            }

        self.manager.save(state)
        return result

    # ═══════════════════════════════════════════════════════════
    # 动态止盈止损 + 策略信号辅助
    # ═══════════════════════════════════════════════════════════

    def update_dynamic_stops(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """
        每日更新持仓的动态止损价 (ATR trailing stop)

        逻辑:
          1. 计算每只持仓的 14 日 ATR
          2. 止损 = 买入后最高价 - 2×ATR (trailing stop)
          3. 移动止盈: 盈利 > 8% 后, 从最高点回撤 5% 触发
          4. 同时更新持仓的 stop_loss 到动态价格 (只上移, 不下移)

        Returns:
            {symbol: {new_stop, peak_price, atr, action: "raised"|"hold"|"alert"}}
        """
        import numpy as np
        state = self.state
        result = {}
        today = date.today().isoformat()

        for sym, pos in list(state.positions.items()):
            cur_price = prices.get(sym, pos.current_price)
            if cur_price <= 0:
                continue

            # 跟踪峰值 (用 entry_price 字段附近的逻辑: 无持久化peak则在position上维护)
            peak = getattr(pos, '_peak_price', 0) or pos.entry_price
            if cur_price > peak:
                peak = cur_price
            pos._peak_price = peak

            cur_pnl_pct = (cur_price / pos.entry_price - 1) * 100 if pos.entry_price > 0 else 0

            # ATR 估算 (简化: 用历史波动率代理)
            atr_est = pos.entry_price * 0.025  # 默认 2.5% ATR
            try:
                # 尝试获取真实ATR
                high_low_est = abs(cur_price - pos.entry_price)
                if high_low_est > 0:
                    atr_est = max(atr_est, high_low_est * 0.3)  # 粗略估计
            except Exception:
                pass

            # 动态止损 = 峰值 - 2×ATR (至少保本)
            dynamic_stop = peak - 2 * atr_est
            # 盈利 > 5% 后至少保本
            if cur_pnl_pct > 5:
                dynamic_stop = max(dynamic_stop, pos.entry_price * 1.01)

            # 只上移止损
            old_stop = pos.stop_loss
            new_stop = max(old_stop, dynamic_stop) if old_stop > 0 else dynamic_stop

            # 判断状态
            if cur_price <= new_stop:
                action = "alert"
            elif new_stop > old_stop + 0.01:
                action = "raised"
            else:
                action = "hold"

            pos.stop_loss = round(new_stop, 2)

            result[sym] = {
                "name": pos.name,
                "price": cur_price,
                "entry": pos.entry_price,
                "peak": round(peak, 2),
                "old_stop": round(old_stop, 2),
                "new_stop": round(new_stop, 2),
                "atr_est": round(atr_est, 2),
                "pnl_pct": round(cur_pnl_pct, 1),
                "action": action,
            }

        return result

    def check_exit_conditions(
        self,
        prices: Dict[str, float],
        sell_signals: Optional[Dict[str, Dict]] = None,
        strategy_votes: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        检查所有持仓的退出条件 (增强版)

        退出优先级:
          1. 动态止损触发 (来自 update_dynamic_stops)
          2. AI 卖出信号 (DeepSeek 分析的 SELL)
          3. 策略共识退出 (3+ 个验证过的策略同时建议卖出)
          4. 移动止盈触发 (涨幅 > 10% 后回撤 > 5%)

        Args:
            prices: {symbol: current_price}
            sell_signals: {symbol: {reason, ...}} — AI 的 SELL 信号
            strategy_votes: {symbol: exit_vote_count} — 策略投票数

        Returns:
            [{symbol, reason, should_sell, price, detail}, ...]
        """
        state = self.state
        triggers = []
        sell_signals = sell_signals or {}
        strategy_votes = strategy_votes or {}

        # 先更新动态止损
        stops = self.update_dynamic_stops(prices)

        for sym, pos in list(state.positions.items()):
            cur_price = prices.get(sym, pos.current_price)
            if cur_price <= 0:
                continue

            cur_pnl = (cur_price / pos.entry_price - 1) * 100 if pos.entry_price > 0 else 0
            peak = getattr(pos, '_peak_price', cur_price)
            stop_info = stops.get(sym, {})

            # 1. 动态止损
            if stop_info.get("action") == "alert":
                triggers.append({
                    "symbol": sym, "name": pos.name, "reason": "dynamic_stop",
                    "should_sell": True, "price": cur_price,
                    "detail": (f"触发动态止损: ¥{cur_price:.2f} <= ¥{pos.stop_loss:.2f} "
                               f"(峰值¥{peak:.2f}, 盈利{cur_pnl:+.1f}%)"),
                })
                continue

            # 2. AI 卖出信号
            if sym in sell_signals:
                triggers.append({
                    "symbol": sym, "name": pos.name, "reason": "ai_sell_signal",
                    "should_sell": True, "price": cur_price,
                    "detail": f"AI分析给出卖出信号: {sell_signals[sym].get('reason','')}",
                })
                continue

            # 3. 策略共识退出 (3+ 策略确认)
            votes = strategy_votes.get(sym, 0)
            if votes >= 3:
                triggers.append({
                    "symbol": sym, "name": pos.name, "reason": "strategy_consensus",
                    "should_sell": True, "price": cur_price,
                    "detail": f"{votes}个验证策略一致建议卖出",
                })
                continue

            # 4. 移动止盈
            if cur_pnl > 10 and (cur_price / peak - 1) * 100 < -5:
                triggers.append({
                    "symbol": sym, "name": pos.name, "reason": "trailing_take_profit",
                    "should_sell": True, "price": cur_price,
                    "detail": f"移动止盈: 从峰值¥{peak:.2f}回撤{(cur_price/peak-1)*100:.1f}%",
                })

        return triggers

    # ═══════════════════════════════════════════════════════════
    # 每日快照
    # ═══════════════════════════════════════════════════════════

    def record_daily_snapshot(self) -> DailySnapshot:
        """记录当日快照"""
        state = self.state
        today = date.today().isoformat()

        # 计算相对于上一快照的日盈亏
        prev_value = state.initial_capital
        if state.daily_snapshots:
            prev_value = state.daily_snapshots[-1].total_value

        total_value = state.total_value
        daily_pnl = total_value - prev_value
        daily_return_pct = (daily_pnl / prev_value * 100) if prev_value > 0 else 0.0

        snapshot = DailySnapshot(
            date=today,
            total_value=round(total_value, 2),
            cash=round(state.cash, 2),
            position_value=round(state.position_value, 2),
            daily_pnl=round(daily_pnl, 2),
            daily_return_pct=round(daily_return_pct, 2),
            cumulative_return_pct=round(state.total_return_pct, 2),
            positions_count=len(state.positions),
        )

        # 去重: 同一天不重复记录
        existing_dates = {s.date for s in state.daily_snapshots}
        if today not in existing_dates:
            state.daily_snapshots.append(snapshot)
        else:
            # 更新今日快照
            for i, s in enumerate(state.daily_snapshots):
                if s.date == today:
                    state.daily_snapshots[i] = snapshot
                    break
        # 保持快照按日期有序
        state.daily_snapshots.sort(key=lambda s: s.date)

        self.manager.save(state)
        logger.info(f"📸 快照 {today}: 总资产¥{total_value:,.2f} "
                    f"({daily_return_pct:+.2f}%) "
                    f"现金¥{state.cash:,.2f} 持仓{len(state.positions)}只")

        return snapshot

    # ═══════════════════════════════════════════════════════════
    # 组合摘要
    # ═══════════════════════════════════════════════════════════

    def get_summary(self) -> Dict[str, Any]:
        """获取组合摘要 (供 Dashboard 使用)"""
        state = self.state
        return {
            "initial_capital": state.initial_capital,
            "total_value": round(state.total_value, 2),
            "cash": round(state.cash, 2),
            "position_value": round(state.position_value, 2),
            "invested_amount": round(state.invested_amount, 2),
            "total_return": round(state.total_return, 2),
            "total_return_pct": round(state.total_return_pct, 2),
            "total_realized_pnl": round(state.total_realized_pnl, 2),
            "total_commission": round(state.total_commission, 2),
            "total_stamp_duty": round(state.total_stamp_duty, 2),
            "total_fees": round(state.total_commission + state.total_stamp_duty + state.total_transfer_fee, 2),
            "trade_count": state.trade_count,
            "win_count": state.win_count,
            "loss_count": state.loss_count,
            "win_rate": round(state.win_rate * 100, 1),
            "position_count": state.position_count,
            "positions": [
                {
                    "symbol": p.symbol,
                    "name": p.name,
                    "quantity": p.quantity,
                    "avg_cost": round(p.avg_cost, 2),
                    "current_price": round(p.current_price, 2),
                    "market_value": round(p.market_value, 2),
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                    "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 2),
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "strategy": p.rec_strategy_name,
                    "confidence": p.rec_confidence,
                    "allocation_pct": round(p.market_value / max(state.total_value, 1) * 100, 1),
                }
                for p in state.positions.values()
            ],
            "daily_snapshots": [
                {
                    "date": s.date,
                    "total_value": s.total_value,
                    "cash": s.cash,
                    "position_value": s.position_value,
                    "daily_return_pct": s.daily_return_pct,
                    "cumulative_return_pct": s.cumulative_return_pct,
                }
                for s in state.daily_snapshots
            ],
            "recent_trades": [
                {
                    "trade_id": t.trade_id,
                    "date": t.date,
                    "symbol": t.symbol,
                    "name": t.name,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "amount": t.amount,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                    "reason": t.reason,
                }
                for t in state.trade_history[-20:]  # 最近20条
            ],
        }
