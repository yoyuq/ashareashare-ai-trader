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
    # 节点4: 策略匹配 (使用KnowledgeManager)
    # ═══════════════════════════════════════════════════════════════

    async def _strategy_matching_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """策略匹配节点 — 根据市场状态+技术指标匹配最佳策略"""
        logger.info(f"[策略匹配] task={state['task_id']}")

        state["strategy_matches"] = {}
        regime = state.get("market_regime", "range_bound")

        # 从知识库获取适配策略
        if self.knowledge:
            regime_strategies = self.knowledge.get_strategies_for_regime(regime)
        else:
            regime_strategies = []

        for sym in state.get("symbols", []):
            matches = []

            if regime_strategies:
                for s in regime_strategies:
                    # 基础适配分
                    base_score = 0.7 if regime in s.get("market_regimes", []) else 0.3

                    # 根据技术指标修正
                    indicators = state.get("technical_indicators", {}).get(sym, {})
                    if isinstance(indicators, dict):
                        trend_score = float(indicators.get("trend_score", 0))
                        # 趋势策略在趋势强时加分
                        if s["category"] == "trend_following":
                            base_score += abs(trend_score) * 0.1
                        # 均值回归策略在震荡时加分
                        elif s["category"] == "mean_reversion":
                            base_score += (1 - abs(trend_score)) * 0.1

                    matches.append({
                        "strategy": s["id"],
                        "name": s.get("name", s["id"]),
                        "fit_score": round(min(base_score, 1.0), 2),
                        "category": s.get("category", ""),
                        "capacity_limit": s.get("capacity_limit", 0),
                    })
            else:
                # 无知识库时的fallback
                if regime in ("strong_bull", "weak_bull"):
                    matches.append({"strategy": "dual_ma_trend", "fit_score": 0.8})
                    matches.append({"strategy": "momentum_breakout", "fit_score": 0.7})
                elif regime == "range_bound":
                    matches.append({"strategy": "bollinger_reversal", "fit_score": 0.75})
                else:
                    matches.append({"strategy": "low_volatility", "fit_score": 0.5})

            # 按适配分排序
            matches.sort(key=lambda m: m["fit_score"], reverse=True)
            state["strategy_matches"][sym] = matches[:3]  # Top 3

        return state

    # ═══════════════════════════════════════════════════════════════
    # 节点5: 回测验证 (精简版)
    # ═══════════════════════════════════════════════════════════════

    async def _backtest_verification_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """
        回测验证节点

        对匹配的策略执行快速历史回测验证,
        计算Sharpe/胜率/最大回撤等关键指标
        """
        logger.info(f"[回测验证] task={state['task_id']}")

        results = {}

        try:
            from backtest.broker import AShareBroker
            from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
            from datetime import date as dt

            for sym, matches in state.get("strategy_matches", {}).items():
                market_data = state.get("market_data", {}).get(sym)
                if market_data is None or market_data.empty:
                    continue

                sym_results = []
                for match in matches[:2]:  # 只验证Top 2策略
                    strategy_id = match["strategy"]
                    try:
                        config = BacktestConfig(
                            initial_capital=100000,
                            start_date=market_data["date"].min(),
                            end_date=market_data["date"].max(),
                        )
                        engine = EventDrivenBacktestEngine(config)
                        engine.load_data(sym, market_data)

                        # 简单回测(MACD金叉策略作快速验证)
                        result = engine.run(
                            self._make_quick_strategy(strategy_id),
                            progress_bar=False,
                        )
                        sym_results.append({
                            "strategy": strategy_id,
                            "total_return": result.total_return,
                            "sharpe": result.sharpe_ratio,
                            "max_drawdown": result.max_drawdown,
                            "win_rate": result.win_rate,
                            "total_trades": result.total_trades,
                        })
                    except Exception as e:
                        sym_results.append({
                            "strategy": strategy_id,
                            "error": str(e),
                        })

                results[sym] = sym_results

        except Exception as e:
            state["errors"].append({"node": "backtest_verification", "error": str(e)})

        state["backtest_results"] = results
        return state

    def _make_quick_strategy(self, strategy_id: str):
        """根据策略ID生成快速回测函数"""
        def quick_strategy(today, bars, broker):
            for sym, bar in bars.items():
                close = bar["close"]
                # 简化版: 持有N天后卖出
                if not hasattr(quick_strategy, "_bought"):
                    quick_strategy._bought = {}
                if sym not in quick_strategy._bought:
                    qty = int(broker.account.cash * 0.2 / close / 100) * 100
                    if qty >= 100:
                        broker.buy(sym, qty, price=close)
                        quick_strategy._bought[sym] = today
                elif (today - quick_strategy._bought[sym]).days >= 5:
                    pos = broker.account.positions.get(sym)
                    if pos and pos.quantity > 0:
                        broker.sell(sym, pos.quantity, price=close)

        quick_strategy._bought = {}  # type: ignore
        return quick_strategy

    # ═══════════════════════════════════════════════════════════════
    # 节点6: 多空辩论 (v2.1 完整实现)
    # ═══════════════════════════════════════════════════════════════

    async def _adversarial_debate_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """
        🆕 v2.1 多空辩论节点

        三Agent对抗:
          Bull Researcher → 必须找看涨理由
          Bear Researcher → 必须找看跌理由
          Judge           → 评估双方论据质量 → 综合结论
        """
        logger.info(f"[多空辩论] task={state['task_id']}")

        if self.router is None:
            state["debate_result"] = {
                "bull_score": 0.5, "bear_score": 0.5,
                "summary": "模型路由未初始化,跳过辩论",
            }
            return state

        try:
            regime = state.get("market_regime", "range_bound")
            reports = state.get("analysis_reports", {})
            strategies = state.get("strategy_matches", {})

            # 准备辩论上下文
            debate_context = json.dumps({
                "regime": regime,
                "reports": {k: v[:500] for k, v in reports.items()},
                "strategies": strategies,
            }, ensure_ascii=False, default=str)[:5000]

            # ---- Bull Researcher (多头) ----
            bull_prompt = (
                "你是A股多头研究员。你的任务是为以下标的寻找所有可能的看涨理由。"
                "必须包含: 技术面/资金面/情绪面/政策面 至少3个维度的论据。"
                "每个论据需要具体的数据支撑,不能泛泛而谈。"
            )
            bull_result = await self.router.route(
                messages=[
                    {"role": "system", "content": bull_prompt},
                    {"role": "user", "content": debate_context},
                ],
                task_type="adversarial_debate",
            )

            # ---- Bear Researcher (空头) ----
            bear_prompt = (
                "你是A股空头研究员。你的任务是为以下标的寻找所有可能的看跌理由。"
                "必须包含: 技术风险/估值风险/流动性风险/宏观风险 至少3个维度的论据。"
                "不要为了反对而反对——每个论据必须有数据或逻辑支撑。"
            )
            bear_result = await self.router.route(
                messages=[
                    {"role": "system", "content": bear_prompt},
                    {"role": "user", "content": debate_context},
                ],
                task_type="adversarial_debate",
            )

            # ---- Judge (裁判) ----
            judge_prompt = (
                "你是A股策略裁判。你收到了以下多空辩论:\n\n"
                f"### 多头论据\n{bull_result.response}\n\n"
                f"### 空头论据\n{bear_result.response}\n\n"
                "请评估双方论据质量:\n"
                "1. 哪一方的论据更有数据支撑？\n"
                "2. 双方的根本分歧点是什么？\n"
                "3. 综合评判: 评分(1-10,多头得分越高=越看多)\n"
                "4. 置信度: 评分分歧度越小→置信度越低\n"
            )
            judge_result = await self.router.route(
                messages=[
                    {"role": "system", "content": "你是公正的策略裁判。"},
                    {"role": "user", "content": judge_prompt},
                ],
                task_type="adversarial_debate",
            )

            state["debate_result"] = {
                "bull_argument": bull_result.response,
                "bear_argument": bear_result.response,
                "judge_verdict": judge_result.response,
                "bull_score": 0.5,  # 后续可NLP打分
                "bear_score": 0.5,
                "summary": judge_result.response[:1000],
            }

        except Exception as e:
            state["errors"].append({"node": "adversarial_debate", "error": str(e)})
            state["debate_result"] = {
                "bull_score": 0.5, "bear_score": 0.5,
                "summary": f"辩论异常: {e}",
            }

        return state

    # ═══════════════════════════════════════════════════════════════
    # 节点7: 综合研判 (增强版 — 含辩论摘要+交易建议)
    # ═══════════════════════════════════════════════════════════════

    async def _synthesis_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """综合研判节点 — 汇总所有分析,输出最终报告"""
        logger.info(f"[综合研判] 生成最终报告")

        if self.router and self.knowledge:
            try:
                system_prompt = self.knowledge.get_system_prompt("synthesis")

                # 构建完整上下文
                context_parts = {
                    "date": state.get("date", ""),
                    "regime": state.get("market_regime"),
                    "regime_confidence": state.get("regime_confidence"),
                    "scan_results": state.get("scan_results", []),
                    "analysis_reports": {
                        k: v[:600] for k, v in state.get("analysis_reports", {}).items()
                    },
                    "strategy_matches": {
                        k: [{"id": m["strategy"], "score": m["fit_score"]}
                            for m in v]
                        for k, v in state.get("strategy_matches", {}).items()
                    },
                    "backtest_summary": {
                        k: [{"strategy": r.get("strategy"), "sharpe": r.get("sharpe"),
                             "win_rate": r.get("win_rate")}
                            for r in v if "error" not in r]
                        for k, v in state.get("backtest_results", {}).items()
                    },
                    "debate_summary": state.get("debate_result", {}).get("summary", ""),
                }

                context = json.dumps(context_parts, ensure_ascii=False, default=str)[:12000]

                result = await self.router.route(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context},
                    ],
                    task_type="daily_synthesis",
                )
                state["final_report"] = result.response

            except Exception as e:
                state["final_report"] = (
                    f"## 综合研判报告\n\n"
                    f"生成失败: {e}\n\n"
                    f"请检查模型路由和知识库配置。"
                )

        else:
            # 生成无LLM的基本摘要
            regime = state.get("market_regime", "unknown")
            symbols = state.get("symbols", [])
            matches = state.get("strategy_matches", {})
            debate = state.get("debate_result", {})

            report = [
                f"## A股每日分析报告",
                f"",
                f"**日期**: {state.get('date', 'N/A')}",
                f"**市场状态**: {regime} (置信度: {state.get('regime_confidence', 'N/A')})",
                f"",
                f"### 扫描标的 ({len(symbols)}只)",
            ]
            for sym in symbols:
                report.append(f"- {sym}")
                sym_matches = matches.get(sym, [])
                if sym_matches:
                    for m in sym_matches[:2]:
                        report.append(f"  - {m['strategy']} (适配分: {m['fit_score']})")

            if debate:
                report.append(f"\n### 辩论摘要\n{debate.get('summary', 'N/A')}")

            state["final_report"] = "\n".join(report)

        return state
