"""
LangGraph工作流 — AI Agent编排引擎

分析流水线:
  数据准备 → 市场扫描 → 技术分析 → 策略匹配 → 回测验证 → 多空辩论 → 综合研判

每个节点是LangGraph的Node,通过State在节点间传递数据。
"""

import asyncio
import json
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, StateGraph
from loguru import logger

from .state import MarketAnalysisState


# ═══════════════════════════════════════════════════════════════
# LangGraph 工作流定义
# ═══════════════════════════════════════════════════════════════

class AnalysisWorkflow:
    """
    市场分析LangGraph工作流

    使用方式:
        wf = AnalysisWorkflow(router, knowledge_manager)
        state = await wf.run_daily_scan(symbols=["sh.600519", "sz.000858"])
        print(state["final_report"])
    """

    def __init__(self, router=None, knowledge_manager=None):
        self.router = router
        self.knowledge = knowledge_manager
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建LangGraph有向图"""
        workflow = StateGraph(MarketAnalysisState)

        # 添加节点
        workflow.add_node("data_preparation", self._data_preparation_node)
        workflow.add_node("market_scanner", self._market_scanner_node)
        workflow.add_node("technical_analysis", self._technical_analysis_node)
        workflow.add_node("strategy_matching", self._strategy_matching_node)
        workflow.add_node("backtest_verification", self._backtest_verification_node)
        workflow.add_node("adversarial_debate", self._adversarial_debate_node)
        workflow.add_node("synthesis", self._synthesis_node)

        # 定义边: 数据准备 → 并行(扫描 + 技术分析) → 策略匹配 → 回测 → 辩论 → 综合
        workflow.set_entry_point("data_preparation")

        workflow.add_edge("data_preparation", "market_scanner")
        workflow.add_edge("data_preparation", "technical_analysis")

        # 扫描和技术分析完成后→策略匹配
        workflow.add_edge("market_scanner", "strategy_matching")
        workflow.add_edge("technical_analysis", "strategy_matching")

        workflow.add_edge("strategy_matching", "backtest_verification")

        # 回测完成后→辩论→综合
        workflow.add_edge("backtest_verification", "adversarial_debate")
        workflow.add_edge("adversarial_debate", "synthesis")

        workflow.add_edge("synthesis", END)

        return workflow.compile()

    # ═══════════════════════════════════════════════════════════════
    # 便捷运行方法
    # ═══════════════════════════════════════════════════════════════

    async def run_daily_scan(
        self,
        symbols: Optional[List[str]] = None,
        pool_name: str = "default",
    ) -> MarketAnalysisState:
        """运行每日市场扫描"""
        initial_state: MarketAnalysisState = {
            "task_id": str(uuid.uuid4())[:8],
            "task_type": "daily_scan",
            "date": date.today().isoformat(),
            "symbols": symbols or [],
            "errors": [],
            "model_trace": [],
            "signals_archived": False,
        }
        result = await self.graph.ainvoke(initial_state)
        return result

    async def run_stock_analysis(
        self, symbol: str
    ) -> MarketAnalysisState:
        """运行单只股票深度分析"""
        initial_state: MarketAnalysisState = {
            "task_id": str(uuid.uuid4())[:8],
            "task_type": "stock_analysis",
            "date": date.today().isoformat(),
            "symbols": [symbol],
            "errors": [],
            "model_trace": [],
            "signals_archived": False,
        }
        result = await self.graph.ainvoke(initial_state)
        return result

    # ═══════════════════════════════════════════════════════════════
    # 节点1: 数据准备
    # ═══════════════════════════════════════════════════════════════

    async def _data_preparation_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """
        数据准备节点:
        - 拉取市场行情数据
        - 计算技术指标
        - 识别市场状态
        - 拉取另类数据
        """
        logger.info(f"[数据准备] task={state['task_id']}")

        try:
            from data import get_data_router
            from analysis import TechnicalAnalyzer, MarketRegimeDetector

            router = get_data_router()
            analyzer = TechnicalAnalyzer()

            # 拉取指数数据(判断市场状态)
            from data.providers.base import DataFrequency, DataRequest
            index_request = DataRequest(
                symbol="sh.000300",
                start_date=date.today().replace(year=date.today().year - 1),
                end_date=date.today(),
                frequency=DataFrequency.DAILY,
                adjust="qfq",
            )
            index_result = await router.get_daily_kline(index_request)
            index_df = index_result.data

            # 市场状态检测
            detector = MarketRegimeDetector()
            regime_result = detector.detect(index_df)

            state["market_regime"] = regime_result.regime.value
            state["regime_confidence"] = regime_result.confidence
            state["regime_params"] = (
                __import__("analysis.regime", fromlist=["get_regime_parameters"])
                .get_regime_parameters(regime_result.regime)
            )

            # 拉取标的行情
            market_data = {}
            for sym in state.get("symbols", []):
                try:
                    req = DataRequest(
                        symbol=sym,
                        start_date=date.today().replace(year=date.today().year - 3),
                        end_date=date.today(),
                        frequency=DataFrequency.DAILY,
                        adjust="qfq",
                    )
                    result = await router.get_daily_kline(req)
                    if not result.data.empty:
                        market_data[sym] = result.data
                except Exception as e:
                    state["errors"].append({"node": "data_prep", "symbol": sym, "error": str(e)})

            state["market_data"] = market_data
            state["market_regime"] = regime_result.regime.value

            logger.info(
                f"[数据准备] 完成: {len(market_data)}只标的, "
                f"市场状态={regime_result.regime.value}"
            )
        except Exception as e:
            state["errors"].append({"node": "data_preparation", "error": str(e)})
            logger.error(f"[数据准备] 异常: {e}")

        return state

    # ═══════════════════════════════════════════════════════════════
    # 节点2: 市场扫描
    # ═══════════════════════════════════════════════════════════════

    async def _market_scanner_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """市场扫描节点"""
        logger.info(f"[市场扫描] task={state['task_id']}")

        if self.router is None:
            state["scan_results"] = []
            return state

        try:
            # 用LLM做市场扫描解读
            if self.knowledge:
                system_prompt = self.knowledge.get_system_prompt("market_scanner")
            else:
                system_prompt = "你是A股市场扫描Agent。"

            # 准备上下文
            regime = state.get("market_regime", "unknown")
            context = f"当前市场状态: {regime}\n标的数量: {len(state.get('symbols', []))}"

            result = await self.router.route(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                task_type="simple_qa",
            )

            state["scan_results"] = [{"raw_analysis": result.response}]
            state["model_trace"].append({
                "node": "market_scanner",
                "tier": result.tier.value,
                "cost": result.cost,
            })
        except Exception as e:
            state["errors"].append({"node": "market_scanner", "error": str(e)})

        return state

    # ═══════════════════════════════════════════════════════════════
    # 节点3: 技术分析
    # ═══════════════════════════════════════════════════════════════

    async def _technical_analysis_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """技术分析节点"""
        logger.info(f"[技术分析] task={state['task_id']}")

        try:
            from analysis import TechnicalAnalyzer

            analyzer = TechnicalAnalyzer()
            reports = {}
            indicators = {}

            for sym, df in state.get("market_data", {}).items():
                if df.empty:
                    continue
                result = analyzer.compute_all(df, symbol=sym)
                indicators[sym] = result.to_dataframe()

                # 生成分析报告
                if self.router:
                    system_prompt = (
                        self.knowledge.get_system_prompt("technical_analyst")
                        if self.knowledge else
                        "你是A股技术分析Agent。"
                    )

                    # 提取关键指标摘要
                    last_row = result.to_dataframe().iloc[-1] if not result.to_dataframe().empty else {}
                    summary = json.dumps({
                        k: round(float(v), 2) if isinstance(v, (int, float)) else str(v)
                        for k, v in last_row.items()
                        if k in ["close", "ma_5", "ma_20", "ma_60", "rsi_14",
                                 "macd_dif", "macd_dea", "bb_pct_20", "atr_14",
                                 "trend_score", "composite_score"]
                    }, ensure_ascii=False)

                    result_llm = await self.router.route(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"标的: {sym}\n指标数据: {summary}"},
                        ],
                        task_type="technical_analysis",
                    )
                    reports[sym] = result_llm.response

            state["technical_indicators"] = {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in indicators.items()
            }
            state["analysis_reports"] = reports

        except Exception as e:
            state["errors"].append({"node": "technical_analysis", "error": str(e)})

        return state

    # ═══════════════════════════════════════════════════════════════
    # 节点4-6: 策略匹配/回测/辩论/综合 (占位,待实现)
    # ═══════════════════════════════════════════════════════════════

    async def _strategy_matching_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """策略匹配节点 — 根据技术指标和市场状态匹配策略"""
        logger.info(f"[策略匹配] task={state['task_id']}")

        state["strategy_matches"] = {}
        regime = state.get("market_regime", "range_bound")

        for sym in state.get("symbols", []):
            matches = []
            # 简单规则: 趋势市场→趋势策略, 震荡→均值回归
            if regime in ("strong_bull", "weak_bull"):
                matches.append({"strategy": "dual_ma_trend", "fit_score": 0.8})
                matches.append({"strategy": "momentum_breakout", "fit_score": 0.7})
            elif regime == "range_bound":
                matches.append({"strategy": "bollinger_reversal", "fit_score": 0.75})
                matches.append({"strategy": "rsi_mean_reversion", "fit_score": 0.7})
            else:
                matches.append({"strategy": "low_volatility", "fit_score": 0.5})

            state["strategy_matches"][sym] = matches

        return state

    async def _backtest_verification_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """回测验证节点"""
        state["backtest_results"] = {"note": "回测验证待完善"}
        return state

    async def _adversarial_debate_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """多空辩论节点 (v2.1)"""
        state["debate_result"] = {
            "bull_score": 0.5,
            "bear_score": 0.5,
            "summary": "辩论节点待完善",
        }
        return state

    async def _synthesis_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """综合研判节点"""
        logger.info(f"[综合研判] 生成最终报告")

        if self.router and self.knowledge:
            try:
                system_prompt = self.knowledge.get_system_prompt("synthesis")
                context = json.dumps({
                    "regime": state.get("market_regime"),
                    "scan": state.get("scan_results", []),
                    "reports": state.get("analysis_reports", {}),
                    "strategies": state.get("strategy_matches", {}),
                    "debate": state.get("debate_result", {}),
                }, ensure_ascii=False, default=str)[:8000]

                result = await self.router.route(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context},
                    ],
                    task_type="daily_synthesis",
                )
                state["final_report"] = result.response
            except Exception as e:
                state["final_report"] = f"综合研判生成失败: {e}"
        else:
            state["final_report"] = "## 综合研判报告\n\n模型路由未初始化,报告待生成。"

        return state
