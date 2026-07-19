"""
分析引擎层 (Analysis Engine)

核心组件:
- TechnicalAnalyzer: 130+技术指标(趋势/动量/波动率/成交量/形态/因子)
- MarketRegimeDetector: 市场状态识别(6种状态)
- WinRateAnalyzer: 胜率分析(分场景路径拆解+凯利仓位)
- SignalTracker: N日信号追踪+闭环回看
- MultiFrameAnalyzer: 多时间框架共振分析
- BayesianICDecayDetector: 贝叶斯因子衰减检测
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
from .winrate import (
    SignalScenario,
    SignalTracker,
    SignalReview,
    WinRateAnalyzer,
    WinRateReport,
)
from .multiframe import (
    BayesianICDecayDetector,
    FactorHealth,
    MultiFrameAnalyzer,
    MultiFrameResult,
    SignalQualityScorer,
    TimeframeSignal,
)

__all__ = [
    "TechnicalAnalyzer",
    "IndicatorResult",
    "MarketRegime",
    "MarketRegimeDetector",
    "RegimeResult",
    "get_regime_parameters",
    "IncrementalComputer",
    # v2.3
    "WinRateAnalyzer",
    "WinRateReport",
    "SignalTracker",
    "SignalReview",
    "SignalScenario",
    "MultiFrameAnalyzer",
    "MultiFrameResult",
    "TimeframeSignal",
    "BayesianICDecayDetector",
    "FactorHealth",
    "SignalQualityScorer",
]
