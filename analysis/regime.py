"""
市场状态识别 (Market Regime Detection)

6种市场状态:
  - strong_bull:  强牛市 (趋势向上,低波动,成交活跃)
  - weak_bull:    弱牛市 (趋势偏上但动量减弱)
  - range_bound:  震荡市 (无明显方向,波动收窄)
  - weak_bear:    弱熊市 (趋势偏下,卖压增大)
  - strong_bear:  强熊市 (趋势下行,恐慌性抛售)
  - crisis:       危机模式 (极端波动+流动性枯竭)

识别方法: 多维度规则 + 层次投票
  维度1: 价格结构 (MA排列 / 高低点序列)
  维度2: 动量强度 (MACD / RSI / ADX)
  维度3: 波动率状态 (HV / BB宽度 / ATR)
  维度4: 成交量特征 (放量上涨 / 缩量下跌 / 量价配合)
  维度5: 市场广度 (涨跌比 / 创新高vs创新低)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class MarketRegime(Enum):
    """市场状态枚举"""
    STRONG_BULL = "strong_bull"
    WEAK_BULL = "weak_bull"
    RANGE_BOUND = "range_bound"
    WEAK_BEAR = "weak_bear"
    STRONG_BEAR = "strong_bear"
    CRISIS = "crisis"


@dataclass
class RegimeResult:
    """市场状态识别结果"""
    regime: MarketRegime
    confidence: float  # 置信度 (0-1)
    scores: Dict[str, float] = field(default_factory=dict)  # 各维度得分
    details: Dict[str, str] = field(default_factory=dict)   # 可读描述
    detected_at: datetime = field(default_factory=datetime.now)

    @property
    def is_bullish(self) -> bool:
        return self.regime in (MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL)

    @property
    def is_bearish(self) -> bool:
        return self.regime in (MarketRegime.STRONG_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.CRISIS)

    @property
    def is_ranging(self) -> bool:
        return self.regime == MarketRegime.RANGE_BOUND

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 3),
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "details": self.details,
            "detected_at": self.detected_at.isoformat(),
        }


class MarketRegimeDetector:
    """
    市场状态检测器

    基于多维度打分+加权投票,实时识别当前市场处于哪种状态。

    使用方式:
        detector = MarketRegimeDetector()
        regime = detector.detect(index_daily_df, market_breadth_df)
    """

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    # ═══════════════════════════════════════════════════════════════
    # 主检测入口
    # ═══════════════════════════════════════════════════════════════

    def detect(
        self,
        index_df: pd.DataFrame,
        market_breadth: Optional[pd.DataFrame] = None,
    ) -> RegimeResult:
        """
        检测当前市场状态

        Args:
            index_df: 指数日K线 (含open/high/low/close/volume)
            market_breadth: 市场广度数据 (可选,含 adv/dec/new_high/new_low)

        Returns:
            RegimeResult
        """
        if index_df.empty or len(index_df) < 20:
            return RegimeResult(
                regime=MarketRegime.RANGE_BOUND,
                confidence=0.1,
                details={"error": "数据不足"},
            )

        scores = {}
        details = {}

        # ===== 维度1: 价格结构 (权重: 25%) =====
        struct_score, struct_detail = self._evaluate_price_structure(index_df)
        scores["price_structure"] = struct_score
        details["price_structure"] = struct_detail

        # ===== 维度2: 动量强度 (权重: 25%) =====
        momentum_score, momentum_detail = self._evaluate_momentum(index_df)
        scores["momentum"] = momentum_score
        details["momentum"] = momentum_detail

        # ===== 维度3: 波动率状态 (权重: 20%) =====
        vol_score, vol_detail = self._evaluate_volatility(index_df)
        scores["volatility"] = vol_score
        details["volatility"] = vol_detail

        # ===== 维度4: 成交量特征 (权重: 15%) =====
        volume_score, volume_detail = self._evaluate_volume(index_df)
        scores["volume"] = volume_score
        details["volume"] = volume_detail

        # ===== 维度5: 市场广度 (权重: 15%) =====
        if market_breadth is not None and not market_breadth.empty:
            breadth_score, breadth_detail = self._evaluate_breadth(market_breadth)
        else:
            breadth_score, breadth_detail = 0, "无广度数据"
        scores["market_breadth"] = breadth_score
        details["market_breadth"] = breadth_detail

        # ===== 加权综合 =====
        weights = {
            "price_structure": 0.25,
            "momentum": 0.25,
            "volatility": 0.20,
            "volume": 0.15,
            "market_breadth": 0.15,
        }
        composite = sum(
            scores[k] * weights[k]
            for k in weights
        )

        # ===== 映射到状态 =====
        # composite range: [-1.0 (最熊) ... +1.0 (最牛)]
        # 同时检查危机模式
        regime, confidence = self._classify_regime(composite, scores)

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            scores=scores,
            details=details,
        )

    # ═══════════════════════════════════════════════════════════════
    # 维度1: 价格结构 (-1 ~ +1)
    # ═══════════════════════════════════════════════════════════════

    def _evaluate_price_structure(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        评估价格结构:
          - MA排列: 多头排列 / 空头排列
          - N日高低点: 创60日新高 / 新低
          - 价格距60日均线距离
        """
        close = df["close"]
        latest = len(df) - 1

        # MA排列
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        ma120 = close.rolling(120).mean() if len(df) >= 120 else ma60

        idx = latest
        bull_alignment = (
            (ma5.iloc[idx] > ma20.iloc[idx]) +
            (ma20.iloc[idx] > ma60.iloc[idx]) +
            (ma60.iloc[idx] > ma120.iloc[idx])
        )
        # bull_alignment: 0=完全空头, 3=完全多头

        # 距离MA60
        dist_ma60 = (close.iloc[idx] / ma60.iloc[idx] - 1) * 100

        # 60日高低点
        is_60d_high = close.iloc[idx] >= close.tail(60).max()
        is_60d_low = close.iloc[idx] <= close.tail(60).min()

        # 综合评分
        score = (
            ((bull_alignment / 3) - 0.5) * 0.5 +   # MA排列: 0~1 → 映射到-0.25~0.25
            np.clip(dist_ma60 / 20, -1, 1) * 0.3 +  # 距MA60距离
            (0.2 if is_60d_high else (-0.2 if is_60d_low else 0))
        )
        score = np.clip(score, -1, 1)

        detail = (
            f"MA排列:{bull_alignment}/3, 距MA60:{dist_ma60:.1f}%, "
            f"{'60日新高' if is_60d_high else ('60日新低' if is_60d_low else '区间内')}"
        )
        return score, detail

    # ═══════════════════════════════════════════════════════════════
    # 维度2: 动量强度 (-1 ~ +1)
    # ═══════════════════════════════════════════════════════════════

    def _evaluate_momentum(self, df: pd.DataFrame) -> Tuple[float, str]:
        """评估动量: RSI + MACD + 收益趋势"""
        close = df["close"]
        latest = len(df) - 1

        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[latest]

        # MACD柱子方向
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = 2 * (macd - signal)
        macd_trend = 1 if hist.iloc[latest] > hist.iloc[latest - 5:latest].min() else -1

        # 20日收益
        ret_20 = (close.iloc[latest] / close.iloc[max(0, latest - 20)] - 1) * 100

        # 综合
        score = (
            ((rsi_val - 50) / 50) * 0.35 +       # RSI: 50→0, <30→-0.4, >70→0.4
            (macd_trend * 0.35) +                   # MACD方向
            np.clip(ret_20 / 20, -1, 1) * 0.3       # 20日收益
        )
        score = np.clip(score, -1, 1)

        detail = f"RSI:{rsi_val:.0f}, MACD:{'↑' if macd_trend>0 else '↓'}, 20日收益:{ret_20:.1f}%"
        return score, detail

    # ═══════════════════════════════════════════════════════════════
    # 维度3: 波动率状态 (-1 ~ +1)
    # ═══════════════════════════════════════════════════════════════

    def _evaluate_volatility(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        评估波动率:
          低波动=牛市常态 / 中波动=震荡 / 极高波动=危机
          注意: 此处返回的是 "波动率对多头的友好程度", 非波动率本身
        """
        close = df["close"]
        ret = close.pct_change().dropna()
        latest = len(df) - 1

        # 20日历史波动率
        hv_20 = ret.tail(20).std() * np.sqrt(252) * 100

        # 60日HV
        hv_60 = ret.tail(min(60, len(ret))).std() * np.sqrt(252) * 100

        # 波动率变化率
        hv_change = hv_20 / (hv_60 + 1e-10) - 1

        # 评分逻辑:
        # 低波动(↓) + 波动收敛(↓) = 牛市常态 → +score
        # 中等波动 + 波动扩大 = 震荡 → 0
        # 极高高波动 + 波动暴涨 = 危机 → -score
        if hv_20 < 15 and hv_change < 0.2:
            score = 0.6
            desc = "低波动(牛市常态)"
        elif hv_20 < 25:
            score = 0.2
            desc = "中低波动"
        elif hv_20 < 35:
            score = -0.2
            desc = "中高波动"
        elif hv_20 < 50:
            score = -0.6
            desc = "高波动(警惕)"
        else:
            score = -1.0
            desc = "极端波动(危机模式!)"

        detail = f"HV20:{hv_20:.1f}%, HV变化:{hv_change:+.0%}, {desc}"
        return score, detail

    # ═══════════════════════════════════════════════════════════════
    # 维度4: 成交量 (-1 ~ +1)
    # ═══════════════════════════════════════════════════════════════

    def _evaluate_volume(self, df: pd.DataFrame) -> Tuple[float, str]:
        """评估量价配合"""
        close = df["close"]
        volume = df["volume"]
        latest = len(df) - 1

        # 近5日量价关系
        vol_up = volume.iloc[-5:].mean() > volume.iloc[-20:].mean()
        price_up = close.iloc[latest] > close.iloc[max(0, latest - 5)]

        # 量价配合评分
        if price_up and vol_up:
            score = 0.5  # 价涨量增(健康)
            desc = "价涨量增"
        elif price_up and not vol_up:
            score = -0.3  # 价涨量缩(背离)
            desc = "价涨量缩(背离!)"
        elif not price_up and vol_up:
            score = -0.2  # 价跌量增(抛压)
            desc = "价跌量增(抛压)"
        else:
            score = -0.4  # 价跌量缩(低迷)
            desc = "价跌量缩(低迷)"

        # 量比修正
        vol_ratio = volume.iloc[latest] / volume.tail(20).mean()
        if vol_ratio > 2:
            score += (0.2 if price_up else -0.2)
            desc += ", 放量"

        detail = f"{desc}, 量比:{vol_ratio:.1f}x"
        return np.clip(score, -1, 1), detail

    # ═══════════════════════════════════════════════════════════════
    # 维度5: 市场广度 (-1 ~ +1)
    # ═══════════════════════════════════════════════════════════════

    def _evaluate_breadth(self, breadth_df: pd.DataFrame) -> Tuple[float, str]:
        """
        评估市场广度:
          - 涨跌比 (advance/decline)
          - 创新高/创新低比
        """
        latest = len(breadth_df) - 1

        # 涨跌比
        if "advance" in breadth_df.columns and "decline" in breadth_df.columns:
            adv = breadth_df["advance"].iloc[latest]
            dec = breadth_df["decline"].iloc[latest]
            adr = adv / (dec + 1e-10)
        else:
            adr = 1.0

        # 创新高/创新低比
        if "new_high" in breadth_df.columns and "new_low" in breadth_df.columns:
            nh = breadth_df["new_high"].iloc[latest]
            nl = breadth_df["new_low"].iloc[latest]
            hl_ratio = nh / (nl + 1e-10)
        else:
            hl_ratio = 1.0

        # 综合评分
        adr_score = np.clip(np.log2(adr + 1e-10) / 3, -1, 1)  # adr=1→0, adr=4→0.67, adr=8→1
        hl_score = np.clip(np.log2(hl_ratio + 1e-10) / 3, -1, 1)
        score = adr_score * 0.5 + hl_score * 0.5

        detail = f"涨跌比:{adr:.2f}, 新高/新低比:{hl_ratio:.2f}"
        return score, detail

    # ═══════════════════════════════════════════════════════════════
    # 分类
    # ═══════════════════════════════════════════════════════════════

    def _classify_regime(
        self,
        composite: float,
        scores: Dict[str, float],
    ) -> Tuple[MarketRegime, float]:
        """将综合得分映射到市场状态"""
        # 先检查危机模式
        if scores.get("volatility", 0) <= -0.8:
            return MarketRegime.CRISIS, abs(scores["volatility"])

        # 按composite区间分类
        if composite >= 0.5:
            regime = MarketRegime.STRONG_BULL
        elif composite >= 0.1:
            regime = MarketRegime.WEAK_BULL
        elif composite >= -0.1:
            regime = MarketRegime.RANGE_BOUND
        elif composite >= -0.5:
            regime = MarketRegime.WEAK_BEAR
        else:
            regime = MarketRegime.STRONG_BEAR

        # 置信度 = |composite| 的绝对值 × (1 - 各维度分歧度)
        divergence = np.std(list(scores.values()))
        confidence = max(0.0, min(abs(composite) * (1 - divergence / 2), 1.0))  # v2.14: clamp to [0,1]

        return regime, confidence

    # ═══════════════════════════════════════════════════════════════
    # 快速检测 (仅用价格数据)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def quick_detect(index_df: pd.DataFrame) -> MarketRegime:
        """
        快速检测: 仅用MA排列,返回粗略状态
        (适用于大量股票扫描时快速决定策略)
        """
        close = index_df["close"]
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        ma120 = close.rolling(120).mean() if len(close) >= 120 else ma60

        idx = len(close) - 1
        c = close.iloc[idx]
        above_ma20 = c > ma20.iloc[idx]
        above_ma60 = c > ma60.iloc[idx]
        ma_bull = ma20.iloc[idx] > ma60.iloc[idx] > ma120.iloc[idx]

        if above_ma20 and above_ma60 and ma_bull:
            return MarketRegime.STRONG_BULL
        elif above_ma20 and above_ma60:
            return MarketRegime.WEAK_BULL
        elif not above_ma20 and not above_ma60 and not ma_bull:
            return MarketRegime.STRONG_BEAR
        elif not above_ma20:
            return MarketRegime.WEAK_BEAR
        else:
            return MarketRegime.RANGE_BOUND


# ═══════════════════════════════════════════════════════════════
# 辅助: 按市场状态调整策略参数
# ═══════════════════════════════════════════════════════════════

def get_regime_parameters(regime: MarketRegime) -> Dict[str, float]:
    """
    根据市场状态返回策略参数建议

    核心原则:
      - 牛市: 放宽止损,提高仓位,趋势策略加权重
      - 熊市: 收紧止损,降低仓位,均值回归策略加权重
      - 震荡: 收紧止盈,反趋势策略加权重
      - 危机: 现金为王,所有信号降低置信度
    """
    params = {
        MarketRegime.STRONG_BULL: {
            "position_multiplier": 1.0,
            "stop_loss_multiplier": 1.2,     # 放宽止损(趋势中回调正常)
            "take_profit_multiplier": 1.3,    # 提高止盈目标
            "trend_strategy_weight": 0.6,
            "mean_reversion_weight": 0.1,
            "breakout_weight": 0.3,
            "max_positions": 8,
            "confidence_threshold": 0.6,
        },
        MarketRegime.WEAK_BULL: {
            "position_multiplier": 0.8,
            "stop_loss_multiplier": 1.0,
            "take_profit_multiplier": 1.1,
            "trend_strategy_weight": 0.5,
            "mean_reversion_weight": 0.2,
            "breakout_weight": 0.3,
            "max_positions": 6,
            "confidence_threshold": 0.65,
        },
        MarketRegime.RANGE_BOUND: {
            "position_multiplier": 0.5,
            "stop_loss_multiplier": 0.8,      # 收紧止损(震荡突破是陷阱)
            "take_profit_multiplier": 0.7,     # 降低止盈目标
            "trend_strategy_weight": 0.2,
            "mean_reversion_weight": 0.6,
            "breakout_weight": 0.2,
            "max_positions": 4,
            "confidence_threshold": 0.7,
        },
        MarketRegime.WEAK_BEAR: {
            "position_multiplier": 0.3,
            "stop_loss_multiplier": 0.7,
            "take_profit_multiplier": 0.6,
            "trend_strategy_weight": 0.3,
            "mean_reversion_weight": 0.4,
            "breakout_weight": 0.3,
            "max_positions": 2,
            "confidence_threshold": 0.75,
        },
        MarketRegime.STRONG_BEAR: {
            "position_multiplier": 0.1,
            "stop_loss_multiplier": 0.5,       # 极紧止损
            "take_profit_multiplier": 0.5,
            "trend_strategy_weight": 0.5,       # 顺势做空思维
            "mean_reversion_weight": 0.4,
            "breakout_weight": 0.1,
            "max_positions": 0,                 # 建议空仓
            "confidence_threshold": 0.85,
        },
        MarketRegime.CRISIS: {
            "position_multiplier": 0.0,
            "stop_loss_multiplier": 1.0,
            "take_profit_multiplier": 1.0,
            "trend_strategy_weight": 0.0,
            "mean_reversion_weight": 0.0,
            "breakout_weight": 0.0,
            "max_positions": 0,                 # 空仓!现金为王
            "confidence_threshold": 0.95,
        },
    }
    return params.get(regime, params[MarketRegime.RANGE_BOUND])
