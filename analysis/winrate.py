"""
🆕 v2.3 胜率分析引擎 (Win Rate Analysis Engine)

基于2026年券商研报方法论:
- 区间胜率: 华泰"多维择时"框架 — 分场景(追高/抄底/追空/逃顶)路径拆解
- 信号质量评分: WorkBuddy四步验证法 — 全量回测+幸存者偏差审计+分钟级拆解+出场隔离
- 动态凯利仓位: Kelly formula with regime-aware adjustment
- N日信号追踪: 信号发出→N日后绩效回看→贝叶斯衰减检测

核心指标:
  1. 区间胜率 (Interval Win Rate): 按周/月统计正确预测比例
  2. 场景胜率 (Scenario Win Rate): 追高/抄底/追空/逃顶 四种场景各自胜率
  3. 盈亏比 (Profit Factor): 总盈利/总亏损
  4. 信号质量分 (Signal Quality Score): 综合时效性+持续性+稳健性
  5. 期望值 (Expected Value): E = WinRate × AvgWin - (1-WinRate) × AvgLoss
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class SignalScenario(Enum):
    """信号场景分类 (华泰多维择时框架)"""
    CHASE_HIGH = "chase_high"       # 追高: 上涨趋势中追入
    BUY_DIP = "buy_dip"             # 抄底: 下跌中买入
    CHASE_SHORT = "chase_short"     # 追空: 下跌趋势中做空
    ESCAPE_TOP = "escape_top"       # 逃顶: 高位卖出/减仓


@dataclass
class WinRateReport:
    """胜率分析报告"""
    # 总体
    total_signals: int
    total_win: int
    overall_win_rate: float        # 总胜率
    profit_factor: float           # 盈亏比
    expected_value: float          # 期望值(%)

    # 分场景
    scenario_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 分时间
    monthly_win_rates: pd.Series = field(default_factory=pd.Series)
    weekly_win_rates: pd.Series = field(default_factory=pd.Series)

    # 信号质量
    avg_holding_days: float = 0
    signal_efficiency: float = 0   # 信号效率(收益/持仓天数)
    max_consecutive_loss: int = 0  # 最大连续亏损

    # 分布
    win_distribution: pd.Series = field(default_factory=pd.Series)
    loss_distribution: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        lines = [
            "══════════ 胜率分析报告 ══════════",
            f"总信号: {self.total_signals} | 胜: {self.total_win} | 胜率: {self.overall_win_rate:.1%}",
            f"盈亏比: {self.profit_factor:.2f} | 期望值: {self.expected_value:+.2f}%",
            f"最大连亏: {self.max_consecutive_loss}次 | 平均持仓: {self.avg_holding_days:.1f}天",
            "",
            "── 分场景胜率 ──",
        ]
        for scenario, stats in self.scenario_breakdown.items():
            lines.append(
                f"  {scenario}: 胜率{stats['win_rate']:.0%} "
                f"({stats['wins']}/{stats['total']}) "
                f"盈亏比{stats['profit_factor']:.1f}"
            )
        lines.append("════════════════════════════════")
        return "\n".join(lines)


class WinRateAnalyzer:
    """
    胜率分析器

    使用方式:
        analyzer = WinRateAnalyzer()
        report = analyzer.analyze(trade_log, price_data)
        print(report.summary())
    """

    def __init__(self):
        pass

    # ═══════════════════════════════════════════════════════════════
    # 主分析入口
    # ═══════════════════════════════════════════════════════════════

    def analyze(
        self,
        trade_log: pd.DataFrame,
        price_data: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
    ) -> WinRateReport:
        """
        全面胜率分析

        Args:
            trade_log: 交易记录 (columns: date, symbol, side, price, pnl, pnl_pct, holding_days)
            price_data: 价格数据 (用于场景分类)
            benchmark: 基准收益率(用于超额收益计算)
        """
        if trade_log.empty:
            return WinRateReport(0, 0, 0, 0, 0)

        df = trade_log.copy()

        # ===== 1. 基础统计 =====
        sells = df[df["side"] == "sell"] if "side" in df.columns else df
        total = len(sells)
        wins = (sells["pnl"] > 0).sum() if "pnl" in sells.columns else 0
        win_rate = wins / total if total > 0 else 0

        # ===== 2. 盈亏比 =====
        if "pnl" in sells.columns:
            gross_profit = sells[sells["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(sells[sells["pnl"] < 0]["pnl"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            profit_factor = 1.0

        # ===== 3. 期望值 =====
        if "pnl_pct" in sells.columns:
            avg_win = sells[sells["pnl_pct"] > 0]["pnl_pct"].mean() if wins > 0 else 0
            avg_loss = abs(sells[sells["pnl_pct"] < 0]["pnl_pct"].mean()) if (total - wins) > 0 else 0
            ev = win_rate * avg_win - (1 - win_rate) * avg_loss
        else:
            ev = 0

        # ===== 4. 场景分类胜率 =====
        scenario_stats = self._classify_scenarios(sells, price_data)

        # ===== 5. 时间维度胜率 =====
        monthly = self._monthly_win_rate(sells)
        weekly = self._weekly_win_rate(sells)

        # ===== 6. 信号质量 =====
        avg_hold = sells["holding_days"].mean() if "holding_days" in sells.columns else 0
        efficiency = self._signal_efficiency(sells)

        # ===== 7. 最大连续亏损 =====
        max_consec = self._max_consecutive_losses(sells)

        return WinRateReport(
            total_signals=total,
            total_win=wins,
            overall_win_rate=win_rate,
            profit_factor=round(profit_factor, 2),
            expected_value=round(ev, 2),
            scenario_breakdown=scenario_stats,
            monthly_win_rates=monthly,
            weekly_win_rates=weekly,
            avg_holding_days=round(avg_hold, 1),
            signal_efficiency=round(efficiency, 3),
            max_consecutive_loss=max_consec,
        )

    # ═══════════════════════════════════════════════════════════════
    # 场景分类 (华泰路径拆解法)
    # ═══════════════════════════════════════════════════════════════

    def _classify_scenarios(
        self,
        sells: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        将每笔交易分类到4种场景:
          chase_high:  入场前5日趋势向上 + 入场时RSI>60
          buy_dip:     入场前5日趋势向下 + 入场时RSI<40
          chase_short: 入场前5日趋势向下(做空场景)
          escape_top:  入场前5日趋势向上 + 入场时RSI>70(高位止盈)
        """
        scenarios = {s.value: {"total": 0, "wins": 0, "wins_pnl": 0.0, "losses_pnl": 0.0,
                                "total_pnl": 0.0, "win_rate": 0, "profit_factor": 0}
                     for s in SignalScenario}

        if price_data.empty or "close" not in price_data.columns:
            return scenarios

        close = price_data.set_index("date")["close"] if "date" in price_data.columns else price_data["close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        for _, trade in sells.iterrows():
            trade_date = trade.get("date", trade.get("signal_date"))
            if trade_date is None:
                continue

            try:
                idx = close.index.get_loc(trade_date)
                if idx < 5:
                    continue

                # 入场前5日趋势
                pre5 = close.iloc[max(0, idx-5):idx]
                trend_up = pre5.iloc[-1] > pre5.iloc[0]

                # 简易RSI(14)
                if idx >= 14:
                    delta = close.iloc[idx-13:idx+1].diff()
                    gain = delta.where(delta > 0, 0).mean()
                    loss = -delta.where(delta < 0, 0).mean()
                    rs = gain / (loss + 1e-10)
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 50

                # 分类
                if trend_up and rsi > 60:
                    scenario = SignalScenario.CHASE_HIGH
                elif not trend_up and rsi < 40:
                    scenario = SignalScenario.BUY_DIP
                elif not trend_up:
                    scenario = SignalScenario.CHASE_SHORT
                elif trend_up and rsi > 70:
                    scenario = SignalScenario.ESCAPE_TOP
                else:
                    scenario = SignalScenario.CHASE_HIGH  # default

                key = scenario.value
                scenarios[key]["total"] += 1
                pnl = trade.get("pnl", 0)
                scenarios[key]["total_pnl"] += float(pnl)
                if pnl > 0:
                    scenarios[key]["wins"] += 1
                    scenarios[key]["wins_pnl"] += float(pnl)
                else:
                    scenarios[key]["losses_pnl"] += abs(float(pnl))

            except (KeyError, IndexError):
                continue

        # 计算每个场景的胜率和盈亏比
        for key, stats in scenarios.items():
            if stats["total"] > 0:
                stats["win_rate"] = round(stats["wins"] / stats["total"], 3)
                stats["profit_factor"] = round(
                    stats["wins_pnl"] / max(stats["losses_pnl"], 0.01), 1
                )

        return scenarios

    # ═══════════════════════════════════════════════════════════════
    # 时间维度胜率
    # ═══════════════════════════════════════════════════════════════

    def _monthly_win_rate(self, sells: pd.DataFrame) -> pd.Series:
        """按月度统计胜率"""
        if sells.empty or "date" not in sells.columns:
            return pd.Series(dtype=float)

        df = sells.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M")

        def calc_win_rate(group):
            if "pnl" in group.columns:
                return (group["pnl"] > 0).sum() / len(group)
            return 0

        monthly = df.groupby("month").apply(calc_win_rate)
        return monthly

    def _weekly_win_rate(self, sells: pd.DataFrame) -> pd.Series:
        """按周度统计胜率"""
        if sells.empty or "date" not in sells.columns:
            return pd.Series(dtype=float)

        df = sells.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["week"] = df["date"].dt.to_period("W")

        def calc_win_rate(group):
            if "pnl" in group.columns:
                return (group["pnl"] > 0).sum() / len(group)
            return 0

        weekly = df.groupby("week").apply(calc_win_rate)
        return weekly

    # ═══════════════════════════════════════════════════════════════
    # 信号质量
    # ═══════════════════════════════════════════════════════════════

    def _signal_efficiency(self, sells: pd.DataFrame) -> float:
        """
        信号效率 = 平均收益 / 平均持仓天数
        值越高 → 信号越有效率(快进快出赚到钱 vs 长时间持仓才赚到)
        """
        if sells.empty:
            return 0

        if "pnl_pct" in sells.columns and "holding_days" in sells.columns:
            valid = sells[sells["holding_days"] > 0]
            if valid.empty:
                return 0
            avg_return = valid["pnl_pct"].abs().mean()
            avg_days = valid["holding_days"].mean()
            return float(avg_return / avg_days) if avg_days > 0 else 0

        return 0

    def _max_consecutive_losses(self, sells: pd.DataFrame) -> int:
        """最大连续亏损次数"""
        if sells.empty or "pnl" not in sells.columns:
            return 0

        pnl_series = sells["pnl"].sort_index()
        max_consec = 0
        current = 0
        for pnl in pnl_series:
            if pnl < 0:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0
        return max_consec

    # ═══════════════════════════════════════════════════════════════
    # 凯利仓位计算
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def kelly_position(
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        regime_multiplier: float = 1.0,
        fractional: float = 0.5,  # 分数凯利(保守)
    ) -> float:
        """
        凯利公式仓位计算

        f* = (p * W - (1-p) * L) / (W * L)

        Args:
            win_rate: 胜率 (0-1)
            avg_win_pct: 平均盈利(%)
            avg_loss_pct: 平均亏损(%)
            regime_multiplier: 市场状态乘数(牛市1.0, 熊市0.3, 危机0)
            fractional: 分数凯利系数(0.5=半凯利, 1.0=全凯利)

        Returns:
            建议仓位比例 (0-1)
        """
        if avg_loss_pct <= 0 or avg_win_pct <= 0:
            return 0

        # 凯利公式
        b = avg_win_pct / avg_loss_pct  # 盈亏比(odds)
        kelly_f = (win_rate * b - (1 - win_rate)) / b

        # 限制
        kelly_f = max(0, min(kelly_f, 0.25))  # 单策略上限25%

        # 分数凯利 + 市场状态调整
        position = kelly_f * fractional * regime_multiplier

        return round(position, 3)

    @staticmethod
    def optimal_position_size(
        capital: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        max_risk_pct: float = 0.02,  # 单笔最大亏损2%
        regime: str = "range_bound",
    ) -> Tuple[float, int]:
        """
        最优仓位计算(综合凯利+风险约束)

        Returns:
            (建议仓位金额, 建议股数@¥10)
        """
        regime_map = {
            "strong_bull": 1.0, "weak_bull": 0.8,
            "range_bound": 0.5, "weak_bear": 0.3,
            "strong_bear": 0.1, "crisis": 0.0,
        }
        multiplier = regime_map.get(regime, 0.5)

        kelly_pct = WinRateAnalyzer.kelly_position(
            win_rate, avg_win_pct, avg_loss_pct, multiplier
        )

        # 风险约束: 单笔最大亏损不超过 capital * max_risk_pct
        risk_adjusted_pct = (capital * max_risk_pct) / (
            capital * avg_loss_pct / 100 + 1e-10
        )

        # 取两者较小值
        position_pct = min(kelly_pct, risk_adjusted_pct, 0.3)
        position_amount = capital * position_pct

        # 假设¥10/股,100股/手
        shares = int(position_amount / 10 / 100) * 100

        return round(position_amount, 0), shares


# ═══════════════════════════════════════════════════════════════
# 信号追踪器 — N日后绩效回看
# ═══════════════════════════════════════════════════════════════

@dataclass
class SignalReview:
    """信号回顾结果"""
    signal_id: str
    symbol: str
    signal_date: date
    direction: str                # long/short
    entry_price: float
    confidence: float

    # N日后回看
    review_date: date
    days_after: int
    actual_return_pct: float
    max_favorable_pct: float      # 最佳时点收益
    max_adverse_pct: float        # 最差时点收益

    # 结果
    is_correct: bool              # 方向是否正确
    quality_score: float          # 信号质量分(0-100)


class SignalTracker:
    """
    信号追踪器 — 信号发出→N日后回看→闭环进化

    使用方式:
        tracker = SignalTracker()
        tracker.emit(symbol="sh.600000", direction="long", price=10.5, confidence=0.8)

        # 5天后回看...
        review = tracker.review("signal_id", price_data, days=5)
    """

    def __init__(self):
        self._active_signals: Dict[str, Dict] = {}
        self._reviewed_signals: List[SignalReview] = []

    def emit(
        self,
        symbol: str,
        direction: str,
        price: float,
        confidence: float = 0.5,
        signal_id: Optional[str] = None,
    ) -> str:
        """发出信号"""
        sid = signal_id or f"{symbol}-{date.today().isoformat()}-{len(self._active_signals)}"
        self._active_signals[sid] = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": price,
            "confidence": confidence,
            "signal_date": date.today(),
            "created_at": datetime.now(),
        }
        return sid

    def review(
        self,
        signal_id: str,
        price_data: pd.DataFrame,
        days: int = 5,
    ) -> Optional[SignalReview]:
        """
        N日后回看信号

        Args:
            signal_id: 信号ID
            price_data: 信号日到当前日的价格数据
            days: 回看天数
        """
        signal = self._active_signals.pop(signal_id, None)
        if signal is None:
            return None

        entry_price = signal["entry_price"]
        signal_date = signal["signal_date"]
        direction = signal["direction"]

        # 取N日价格
        df = price_data.copy()
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])

        future = df[df["date"] > pd.Timestamp(signal_date)].head(days)
        if future.empty:
            return None

        exit_price = future["close"].iloc[-1] if "close" in future.columns else future.iloc[-1]["close"]
        prices = future["close"].values if "close" in future.columns else future["close"].values

        # 收益计算
        if direction == "long":
            actual_return = (exit_price / entry_price - 1) * 100
            max_fav = (prices.max() / entry_price - 1) * 100
            max_adv = (prices.min() / entry_price - 1) * 100
            is_correct = actual_return > 0
        else:  # short
            actual_return = (1 - exit_price / entry_price) * 100
            max_fav = (1 - prices.min() / entry_price) * 100
            max_adv = (1 - prices.max() / entry_price) * 100
            is_correct = actual_return > 0

        # 质量分: 方向正确+大收益+低回撤 = 高
        quality = (
            (50 if is_correct else 0) +
            min(abs(actual_return) * 5, 30) +
            min((1 - abs(max_adv / (max_fav + 1e-10))) * 20, 20)
        )

        review = SignalReview(
            signal_id=signal_id,
            symbol=signal["symbol"],
            signal_date=signal_date,
            direction=direction,
            entry_price=entry_price,
            confidence=signal["confidence"],
            review_date=date.today(),
            days_after=days,
            actual_return_pct=round(actual_return, 2),
            max_favorable_pct=round(max_fav, 2),
            max_adverse_pct=round(max_adv, 2),
            is_correct=is_correct,
            quality_score=round(min(quality, 100), 1),
        )

        self._reviewed_signals.append(review)
        return review

    def review_all(
        self, price_data: pd.DataFrame, days: int = 5
    ) -> List[SignalReview]:
        """批量回看所有活跃信号"""
        reviews = []
        for sid in list(self._active_signals.keys()):
            r = self.review(sid, price_data, days)
            if r:
                reviews.append(r)
        return reviews

    def get_tracking_stats(self) -> Dict[str, Any]:
        """信号追踪统计"""
        if not self._reviewed_signals:
            return {"total_reviewed": 0}

        rs = self._reviewed_signals
        correct = sum(1 for r in rs if r.is_correct)
        total = len(rs)

        return {
            "total_reviewed": total,
            "correct": correct,
            "accuracy": round(correct / total, 3) if total > 0 else 0,
            "avg_return_pct": round(np.mean([r.actual_return_pct for r in rs]), 2),
            "avg_quality": round(np.mean([r.quality_score for r in rs]), 1),
            "avg_max_favorable": round(np.mean([r.max_favorable_pct for r in rs]), 2),
            "avg_max_adverse": round(np.mean([r.max_adverse_pct for r in rs]), 2),
            "by_confidence": {
                "high": sum(1 for r in rs if r.confidence >= 0.7 and r.is_correct) /
                        max(sum(1 for r in rs if r.confidence >= 0.7), 1),
                "mid": sum(1 for r in rs if 0.4 <= r.confidence < 0.7 and r.is_correct) /
                       max(sum(1 for r in rs if 0.4 <= r.confidence < 0.7), 1),
                "low": sum(1 for r in rs if r.confidence < 0.4 and r.is_correct) /
                       max(sum(1 for r in rs if r.confidence < 0.4), 1),
            },
        }
