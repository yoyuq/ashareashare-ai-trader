"""
分析引擎层 (Analysis Engine)

核心组件:
- TechnicalAnalyzer: 130+技术指标(趋势/动量/波动率/成交量/形态/因子)
- MarketRegimeDetector: 市场状态识别(6种状态)
- IncrementalComputer: 增量计算调度器+数据异常检测
"""

from .indicators import IndicatorResult, TechnicalAnalyzer
from .incremental import IncrementalComputer
from .regime import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeResult,
    get_regime_parameters,
)

__all__ = [
    "TechnicalAnalyzer",
    "IndicatorResult",
    "MarketRegime",
    "MarketRegimeDetector",
    "RegimeResult",
    "get_regime_parameters",
    "IncrementalComputer",
]
