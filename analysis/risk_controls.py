"""
🆕 v2.6 组合风控 + 信号分级系统

信号分级 (6级):
  STRONG_BUY(>=80) → BUY(>=65) → WATCH(>=50) → HOLD(>=35) → SELL(>=20) → STRONG_SELL(<20)

组合风控:
  - 回撤断路器: -8%触发减仓50% / -15%触发清仓
  - 波动率目标: 年化15%波动率上限
  - 单票上限: 20%
  - 集中度控制: 最大行业暴露40%
  - 危机模式: 波动率>30% + 广度<35% → 95%现金
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class SignalLevel(Enum):
    """信号等级"""
    STRONG_BUY = ("STRONG_BUY", 80, "🟢🟢")
    BUY = ("BUY", 65, "🟢")
    WATCH = ("WATCH", 50, "🟡")
    HOLD = ("HOLD", 35, "⚪")
    SELL = ("SELL", 20, "🔴")
    STRONG_SELL = ("STRONG_SELL", 0, "🔴🔴")

    @property
    def label(self) -> str: return self.value[0]

    @property
    def threshold(self) -> int: return self.value[1]

    @property
    def emoji(self) -> str: return self.value[2]

    @classmethod
    def from_score(cls, score: float) -> "SignalLevel":
        for level in cls:
            if score >= level.value[1]:
                return level
        return cls.STRONG_SELL


@dataclass
class SignalCard:
    """信号卡片"""
    symbol: str
    name: str = ""
    score: float = 0
    level: SignalLevel = SignalLevel.HOLD
    direction: str = "long"
    action: str = ""          # "buy 600 shares at 10.5"
    reason: str = ""


@dataclass
class PortfolioRiskState:
    """组合风险状态"""
    # 净值
    current_capital: float
    peak_capital: float
    drawdown_pct: float

    # 波动率
    current_volatility: float      # 年化波动率
    target_volatility: float = 0.15

    # 市场广度
    breadth_ratio: float = 1.0     # 涨跌比

    # 风控触发
    circuit_breaker_active: bool = False
    crisis_mode: bool = False
    risk_multiplier: float = 1.0   # 当前风险乘数(1.0=正常)

    # 建议
    suggested_exposure_pct: float = 0.3
    suggested_cash_pct: float = 0.7
    warning_messages: List[str] = field(default_factory=list)


class PortfolioRiskManager:
    """
    组合风控管理器

    规则(基于2026年研究):
      1. 回撤断路器: DD>-8%→减仓50%, DD>-15%→清仓
      2. 波动率目标: 年化15%, 超过则等比缩仓
      3. 危机模式: IV>30% AND 广度<35% → 95%现金
      4. 单票上限: 20%
      5. 行业集中度: 40%
    """

    # 风控参数
    DD_WARNING = -0.05       # -5%回撤预警
    DD_REDUCE = -0.08        # -8%减仓50%
    DD_LIQUIDATE = -0.15     # -15%清仓
    VOL_TARGET = 0.15        # 年化15%波动率目标
    VOL_CRISIS = 0.30        # 年化30%=危机
    BREADTH_CRISIS = 0.35    # 广度<35%=危机
    MAX_SINGLE_POSITION = 0.20  # 单票最大20%
    MAX_SECTOR_EXPOSURE = 0.40  # 行业最大40%

    def __init__(self, initial_capital: float = 100000):
        self.peak_capital = initial_capital
        self._daily_values: List[float] = [initial_capital]

    def update(self, current_capital: float, daily_returns: pd.Series,
               market_breadth: float = 0.5) -> PortfolioRiskState:
        """
        更新组合风险状态

        Args:
            current_capital: 当前总资金
            daily_returns: 近期日收益率序列
            market_breadth: 市场广度(涨跌比,0-1)

        Returns:
            PortfolioRiskState
        """
        self._daily_values.append(current_capital)
        self.peak_capital = max(self.peak_capital, current_capital)

        # 1. 回撤计算
        dd = (current_capital / self.peak_capital - 1)

        # 2. 波动率
        vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 5 else 0.15

        # 3. 风险乘数
        risk_mult = 1.0
        warnings = []

        # 回撤断路器
        circuit = False
        if dd <= self.DD_LIQUIDATE:
            risk_mult = 0.05
            circuit = True
            warnings.append(f"⚠️ 回撤{dd:.1%}触发清仓线(-15%),风险乘数降至0.05")
        elif dd <= self.DD_REDUCE:
            risk_mult = 0.5
            circuit = True
            warnings.append(f"⚡ 回撤{dd:.1%}触发减仓线(-8%),风险乘数降至0.5")
        elif dd <= self.DD_WARNING:
            risk_mult = 0.8
            warnings.append(f"回撤{dd:.1%}接近预警线(-5%),风险乘数降至0.8")

        # 波动率调整
        if vol > self.VOL_TARGET:
            vol_adj = self.VOL_TARGET / vol
            risk_mult *= vol_adj
            warnings.append(f"波动率{vol:.1%}超过目标{self.VOL_TARGET:.0%},乘数×{vol_adj:.2f}")

        # 危机模式
        crisis = (vol > self.VOL_CRISIS and market_breadth < self.BREADTH_CRISIS)
        if crisis:
            risk_mult = min(risk_mult, 0.05)
            warnings.append("🚨 危机模式: 高波动+低广度,强制95%现金")

        # 建议仓位
        suggested_exposure = max(0, min(1.0, risk_mult))
        suggested_cash = 1.0 - suggested_exposure

        return PortfolioRiskState(
            current_capital=current_capital,
            peak_capital=self.peak_capital,
            drawdown_pct=round(dd * 100, 2),
            current_volatility=round(vol * 100, 1),
            target_volatility=self.VOL_TARGET,
            breadth_ratio=market_breadth,
            circuit_breaker_active=circuit,
            crisis_mode=crisis,
            risk_multiplier=round(risk_mult, 3),
            suggested_exposure_pct=round(suggested_exposure, 3),
            suggested_cash_pct=round(suggested_cash, 3),
            warning_messages=warnings,
        )

    def check_position_limit(self, position_value: float,
                             total_capital: float) -> Tuple[bool, str]:
        """检查单票仓位是否超限"""
        pct = position_value / total_capital if total_capital > 0 else 1
        if pct > self.MAX_SINGLE_POSITION:
            return False, f"单票仓位{pct:.1%}超过上限{self.MAX_SINGLE_POSITION:.0%}"
        return True, "OK"

    def check_sector_exposure(self, sector_values: Dict[str, float],
                              total_capital: float) -> Tuple[bool, str]:
        """检查行业集中度"""
        for sector, value in sector_values.items():
            pct = value / total_capital if total_capital > 0 else 1
            if pct > self.MAX_SECTOR_EXPOSURE:
                return False, f"{sector}仓位{pct:.1%}超过行业上限{self.MAX_SECTOR_EXPOSURE:.0%}"
        return True, "OK"

    def size_positions(
        self,
        recommendations: List[Any],
        total_capital: float,
        risk_state: PortfolioRiskState,
    ) -> List[Any]:
        """
        根据风控状态调整推荐仓位

        Args:
            recommendations: TradeRecommendation列表
            total_capital: 总资金
            risk_state: 当前风险状态

        Returns:
            调整后的推荐列表
        """
        if risk_state.crisis_mode:
            # 危机模式: 所有仓位归零
            for r in recommendations:
                r.suggested_amount = 0
                r.position_pct = 0
                r.shares = 0
                r.risks.append("🚨 危机模式:强制空仓")
            logger.warning("危机模式: 所有仓位清零")
            return recommendations

        # 应用风险乘数
        for r in recommendations:
            adjusted_amount = r.suggested_amount * risk_state.risk_multiplier
            # 单票上限检查
            max_amount = total_capital * self.MAX_SINGLE_POSITION
            if adjusted_amount > max_amount:
                adjusted_amount = max_amount
            r.suggested_amount = adjusted_amount
            r.position_pct = adjusted_amount / total_capital if total_capital > 0 else 0
            r.shares = max(0, int(adjusted_amount / r.entry_price / 100) * 100)

            if risk_state.circuit_breaker_active:
                r.risks.append(f"回撤断路器激活(DD={risk_state.drawdown_pct}%),仓位已缩减")

        return recommendations


# ═══════════════════════════════════════════════════════════════
# 信号分级器
# ═══════════════════════════════════════════════════════════════

class SignalGrader:
    """信号分级器 — 将原始评分映射到可操作信号"""

    @staticmethod
    def grade(
        composite_score: float,
        win_rate: float,
        regime: str,
        northbound_signal: str = "neutral",
        sector_rank: int = 0,
    ) -> SignalCard:
        """
        综合分级

        输入:
          composite_score: 技术面综合评分(0-100)
          win_rate: 策略胜率
          regime: 市场状态
          northbound_signal: 北向信号
          sector_rank: 板块排名(越小越好)
        """
        score = composite_score

        # 胜率调整
        if win_rate > 0.7: score += 8
        elif win_rate > 0.6: score += 4
        elif win_rate < 0.4: score -= 8
        elif win_rate < 0.5: score -= 4

        # 北向调整
        if northbound_signal == "strong_inflow": score += 6
        elif northbound_signal == "inflow": score += 3
        elif northbound_signal == "strong_outflow": score -= 8
        elif northbound_signal == "outflow": score -= 4

        # 板块调整
        if sector_rank > 0:
            if sector_rank <= 5: score += 6
            elif sector_rank <= 15: score += 3
            elif sector_rank >= 40: score -= 5

        # 市场状态调整
        regime_adj = {
            "strong_bull": 5, "weak_bull": 3,
            "range_bound": 0, "weak_bear": -3,
            "strong_bear": -8, "crisis": -15,
        }
        score += regime_adj.get(regime, 0)

        score = min(100, max(0, score))
        level = SignalLevel.from_score(score)

        # 方向
        if level in (SignalLevel.STRONG_BUY, SignalLevel.BUY):
            direction = "long"
            action = "买入"
        elif level == SignalLevel.WATCH:
            direction = "long"
            action = "轻仓试探"
        elif level == SignalLevel.HOLD:
            direction = "flat"
            action = "持有/观望"
        elif level == SignalLevel.SELL:
            direction = "flat"
            action = "减仓"
        else:
            direction = "flat"
            action = "清仓"

        return SignalCard(
            symbol="",
            score=round(score, 1),
            level=level,
            direction=direction,
            action=action,
        )
