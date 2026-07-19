"""
🆕 v2.3 多时间框架分析 + 因子IC衰减监控

Multi-Timeframe Analysis:
- 日/周/月K线联动分析
- 多周期共振检测 (日线+周线同向=强信号)
- 周线枢轴点 (Pivot Points)
- 时间框架分歧预警 (日线看多但周线看空=假信号风险)

Factor IC Decay Monitor:
- 贝叶斯变点检测 (Bayesian Change Point Detection)
- 因子IC滚动追踪
- 自动退休机制 (P(decay) > 95% → 标记为inactive)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 多时间框架分析
# ═══════════════════════════════════════════════════════════════

@dataclass
class TimeframeSignal:
    """单个时间框架的信号"""
    timeframe: str           # daily / weekly / monthly
    trend: str               # bullish / bearish / neutral
    strength: float          # 0-1 趋势强度
    key_level: float         # 关键价位
    indicators: Dict[str, float] = field(default_factory=dict)


@dataclass
class MultiFrameResult:
    """多时间框架分析结果"""
    symbol: str
    daily: TimeframeSignal
    weekly: Optional[TimeframeSignal] = None
    monthly: Optional[TimeframeSignal] = None

    # 共振分析
    resonance: str = "none"        # bullish_resonance / bearish_resonance / divergence
    resonance_strength: float = 0  # 0-1 共振强度
    divergence_warning: bool = False

    # 周线枢轴点
    pivot_points: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        daily_emoji = "🟢" if self.daily.trend == "bullish" else ("🔴" if self.daily.trend == "bearish" else "⚪")
        weekly_emoji = "🟢" if self.weekly and self.weekly.trend == "bullish" else ("🔴" if self.weekly and self.weekly.trend == "bearish" else "⚪")
        monthly_emoji = "🟢" if self.monthly and self.monthly.trend == "bullish" else ("🔴" if self.monthly and self.monthly.trend == "bearish" else "⚪")

        return (
            f"{self.symbol} 多周期: 日{daily_emoji} 周{weekly_emoji} 月{monthly_emoji} | "
            f"共振: {self.resonance} ({self.resonance_strength:.0%})"
            + (" ⚠️分歧!" if self.divergence_warning else "")
        )


class MultiFrameAnalyzer:
    """
    多时间框架分析器

    原则: 大周期定方向, 小周期找买点
    """

    def __init__(self):
        pass

    def analyze(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        weekly_df: Optional[pd.DataFrame] = None,
        monthly_df: Optional[pd.DataFrame] = None,
    ) -> MultiFrameResult:
        """多时间框架综合分析"""
        from .indicators import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()

        # 日线
        daily_ind = analyzer.compute_all(daily_df, symbol=symbol)
        daily_signal = self._extract_signal("daily", daily_ind)

        # 周线
        weekly_signal = None
        if weekly_df is not None and len(weekly_df) >= 20:
            weekly_ind = analyzer.compute_all(weekly_df, symbol=symbol)
            weekly_signal = self._extract_signal("weekly", weekly_ind)

        # 月线
        monthly_signal = None
        if monthly_df is not None and len(monthly_df) >= 12:
            monthly_ind = analyzer.compute_all(monthly_df, symbol=symbol)
            monthly_signal = self._extract_signal("monthly", monthly_ind)

        # 共振分析
        resonance, res_strength, divergence = self._analyze_resonance(
            daily_signal, weekly_signal, monthly_signal
        )

        # 枢轴点
        pivots = self._calc_pivot_points(daily_df) if not daily_df.empty else {}

        return MultiFrameResult(
            symbol=symbol,
            daily=daily_signal,
            weekly=weekly_signal,
            monthly=monthly_signal,
            resonance=resonance,
            resonance_strength=res_strength,
            divergence_warning=divergence,
            pivot_points=pivots,
        )

    def _extract_signal(self, tf: str, ind) -> TimeframeSignal:
        """从指标结果提取时间框架信号"""
        last = ind.to_dataframe().iloc[-1]

        trend_score = float(last.get("trend_score", 0))
        rsi = float(last.get("rsi_14", 50))

        if trend_score > 0.2:
            trend = "bullish"
            strength = min(abs(trend_score), 1.0)
        elif trend_score < -0.2:
            trend = "bearish"
            strength = min(abs(trend_score), 1.0)
        else:
            trend = "neutral"
            strength = 0.3

        return TimeframeSignal(
            timeframe=tf,
            trend=trend,
            strength=strength,
            key_level=float(last.get("close", 0)),
            indicators={
                "rsi": rsi,
                "trend_score": trend_score,
                "macd_hist": float(last.get("macd_hist", 0)),
                "vol_ratio": float(last.get("vol_ratio_5", 1)),
            },
        )

    def _analyze_resonance(
        self,
        daily: TimeframeSignal,
        weekly: Optional[TimeframeSignal],
        monthly: Optional[TimeframeSignal],
    ) -> Tuple[str, float, bool]:
        """
        共振分析:
        - bullish_resonance: 日+周同向看多
        - bearish_resonance: 日+周同向看空
        - divergence: 日线方向与周线相反
        """
        signals = [daily]
        if weekly: signals.append(weekly)
        if monthly: signals.append(monthly)

        bull_count = sum(1 for s in signals if s.trend == "bullish")
        bear_count = sum(1 for s in signals if s.trend == "bearish")
        total = len(signals)

        if bull_count == total:
            return "bullish_resonance", bull_count / total, False
        elif bear_count == total:
            return "bearish_resonance", bear_count / total, False
        elif bull_count > 0 and bear_count > 0:
            # 分歧
            strength = max(bull_count, bear_count) / total
            return "divergence", strength, True
        else:
            return "none", 0.3, False

    def _calc_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算周线枢轴点 (Floor Trader Pivots)"""
        if df.empty:
            return {}

        high = df["high"].iloc[-1] if "high" in df.columns else df["close"].iloc[-1]
        low = df["low"].iloc[-1] if "low" in df.columns else df["close"].iloc[-1]
        close = df["close"].iloc[-1]

        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)
        s1 = 2 * pp - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)

        return {
            "R3": round(r3, 2), "R2": round(r2, 2), "R1": round(r1, 2),
            "PP": round(pp, 2),
            "S1": round(s1, 2), "S2": round(s2, 2), "S3": round(s3, 2),
        }


# ═══════════════════════════════════════════════════════════════
# 贝叶斯因子IC衰减检测
# ═══════════════════════════════════════════════════════════════

@dataclass
class FactorHealth:
    """因子健康状态"""
    name: str
    rolling_ic: float              # 当前滚动IC
    ic_trend: str                  # improving / stable / declining
    decay_probability: float       # P(decay) — 贝叶斯后验概率
    warning_level: str             # green / yellow / red
    should_retire: bool = False
    days_to_retirement: int = 999


class BayesianICDecayDetector:
    """
    🆕 v2.3 贝叶斯IC衰减检测器

    方法: Bayesian Change Point Detection
      - 先验: P(decay) = 0.1 (假设10%的因子可能衰减)
      - 似然: IC序列相对于历史均值的偏离程度
      - 后验: P(decay | recent_IC) = P(recent_IC | decay) * P(decay) / P(recent_IC)

    触发条件:
      - P(decay) > 0.95 → 标记为red,建议退休
      - P(decay) > 0.80 → 标记为yellow,预警
      - P(decay) < 0.80 → 标记为green,正常
    """

    def __init__(self, prior_decay: float = 0.1, window: int = 60):
        self.prior_decay = prior_decay
        self.window = window
        self._factor_history: Dict[str, List[float]] = {}

    def update(self, factor_name: str, ic_value: float) -> FactorHealth:
        """更新因子IC并检测衰减"""
        if factor_name not in self._factor_history:
            self._factor_history[factor_name] = []

        history = self._factor_history[factor_name]
        history.append(ic_value)

        if len(history) < self.window // 2:
            return FactorHealth(
                name=factor_name, rolling_ic=ic_value,
                ic_trend="stable", decay_probability=0,
                warning_level="green",
            )

        # 滚动IC
        recent = history[-self.window:]
        rolling_ic = np.mean(recent)
        historical_mean = np.mean(history)
        historical_std = np.std(history)

        if historical_std < 1e-10:
            historical_std = 0.01

        # 贝叶斯更新
        # 似然: 近期IC低于历史均值多少标准差
        z_score = (rolling_ic - historical_mean) / historical_std
        # P(recent_IC | decay): IC下降越显著,似然越大
        likelihood_decay = 1 / (1 + np.exp(-z_score - 1))  # sigmoid shifted
        likelihood_no_decay = 1 - likelihood_decay

        # 后验: P(decay | recent_IC)
        posterior = (
            likelihood_decay * self.prior_decay /
            (likelihood_decay * self.prior_decay +
             likelihood_no_decay * (1 - self.prior_decay))
        )

        # 趋势判断
        if len(recent) >= 20:
            first_half = np.mean(recent[:len(recent)//2])
            second_half = np.mean(recent[len(recent)//2:])
            if second_half > first_half + 0.01:
                trend = "improving"
            elif second_half < first_half - 0.01:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # 告警级别
        if posterior > 0.95:
            level = "red"
            retire = True
        elif posterior > 0.80:
            level = "yellow"
            retire = False
        else:
            level = "green"
            retire = False

        # 距退休天数估算
        if trend == "declining" and posterior > 0.5:
            decay_rate = abs(second_half - first_half) if len(recent) >= 20 else 0.01
            days_left = int((0.95 - posterior) / decay_rate * 5) if decay_rate > 0 else 999
        else:
            days_left = 999

        return FactorHealth(
            name=factor_name,
            rolling_ic=round(rolling_ic, 4),
            ic_trend=trend,
            decay_probability=round(posterior, 4),
            warning_level=level,
            should_retire=retire,
            days_to_retirement=days_left,
        )

    def get_all_health(self) -> Dict[str, FactorHealth]:
        """获取所有因子健康状态"""
        return {
            name: self.update(name, self._factor_history[name][-1])
            for name in self._factor_history
            if self._factor_history[name]
        }


# ═══════════════════════════════════════════════════════════════
# 综合信号评分器
# ═══════════════════════════════════════════════════════════════

class SignalQualityScorer:
    """信号质量综合评分器 (0-100分, A-F等级)"""

    @staticmethod
    def score(
        win_rate: float,
        profit_factor: float,
        signal_count: int,
        avg_holding_days: float,
        max_consecutive_loss: int,
        monthly_stability: float,  # 月度胜率标准差(越小越稳定)
    ) -> Tuple[float, str]:
        """
        综合评分:
          - 胜率: 30分 (胜率>50%→15分, >60%→25分, >70%→30分)
          - 盈亏比: 25分 (PF>1.5→15分, >2.0→20分, >3.0→25分)
          - 样本量: 15分 (信号数>30→10分, >100→15分)
          - 效率: 15分 (信号效率得分)
          - 稳定性: 15分 (月度胜率CV<0.3→15分, <0.5→10分)
        """
        score = 0.0

        # 胜率 (30分)
        if win_rate > 0.7: score += 30
        elif win_rate > 0.6: score += 25
        elif win_rate > 0.5: score += 15
        else: score += max(0, win_rate * 20)

        # 盈亏比 (25分)
        if profit_factor > 3.0: score += 25
        elif profit_factor > 2.0: score += 20
        elif profit_factor > 1.5: score += 15
        elif profit_factor > 1.0: score += 10

        # 样本量 (15分)
        if signal_count > 100: score += 15
        elif signal_count > 50: score += 10
        elif signal_count > 30: score += 7
        elif signal_count > 10: score += 3

        # 效率 (15分) — 收益/持仓天
        if avg_holding_days > 0:
            efficiency = min(abs(win_rate - 0.5) * 2 / avg_holding_days * 100, 15)
            score += efficiency

        # 稳定性 (15分)
        if monthly_stability < 0.15: score += 15
        elif monthly_stability < 0.30: score += 10
        elif monthly_stability < 0.50: score += 5

        # 最大连亏惩罚
        if max_consecutive_loss > 10: score -= 10
        elif max_consecutive_loss > 5: score -= 5

        score = max(0, min(100, score))

        # 等级
        if score >= 85: grade = "A"
        elif score >= 70: grade = "B"
        elif score >= 55: grade = "C"
        elif score >= 40: grade = "D"
        else: grade = "F"

        return round(score, 1), grade
