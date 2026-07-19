"""
Agent状态定义 — LangGraph共享状态

整个分析流水线各个节点通过TypedDict共享状态,
每个节点读取前人产出→处理→写入新字段。
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, TypedDict

from analysis.regime import MarketRegime


class MarketAnalysisState(TypedDict, total=False):
    """
    LangGraph工作流共享状态

    每个字段由特定Agent负责填充,下游Agent读取。
    """

    # === 任务元信息 ===
    task_id: str                        # 任务ID (uuid)
    task_type: str                      # daily_scan / stock_analysis / strategy_backtest
    date: str                           # 分析日期 (YYYY-MM-DD)
    symbols: List[str]                  # 待分析标的列表

    # === 数据层产出 ===
    market_data: Dict[str, Any]         # 市场行情数据 (DataRouter输出)
    alternative_data: Dict[str, Any]    # 另类数据快照

    # === 分析引擎层产出 ===
    technical_indicators: Dict[str, Any]  # 技术指标计算结果
    market_regime: str                    # 市场状态 (strong_bull/weak_bull/...)
    regime_confidence: float              # 状态置信度
    regime_params: Dict[str, float]       # 市场状态对应的策略参数

    # === 多空辩论层产出 (v2.1) ===
    debate_result: Dict[str, Any]         # 辩论结果
    bull_arguments: List[Dict]            # 多头论据
    bear_arguments: List[Dict]            # 空头论据
    debate_score: float                   # 辩论综合评分(bull优势=正,bear优势=负)

    # === 代码即推理层产出 (v2.1) ===
    computed_numbers: Dict[str, float]    # {变量名: 值} — 所有数值由代码计算
    computed_metrics: Dict[str, Any]      # 计算指标汇总

    # === Agent产出 ===
    scan_results: List[Dict]              # 市场扫描结果
    analysis_reports: Dict[str, str]      # {symbol: 分析报告}
    strategy_matches: Dict[str, List]     # {symbol: [匹配策略]}
    backtest_results: Dict[str, Any]      # 回测结果
    risk_assessments: Dict[str, Dict]     # 风险评估

    # === 综合研判 ===
    trade_suggestions: List[Dict]         # 交易建议列表
    final_report: str                     # 最终研判报告
    report_provenance: Dict[str, str]     # 报告数据溯源表 (数字→来源)

    # === 元信息 ===
    model_trace: List[Dict]               # 模型调用追踪 (tier/cost/tokens)
    errors: List[Dict]                    # 各节点错误汇总
    execution_time_ms: float              # 总执行时间

    # === 信号归档 ===
    signals_archived: bool                # 是否已归档信号
