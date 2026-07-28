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
from analysis.regime import get_regime_parameters


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
        """
        构建LangGraph有向图 (v2.2 改进: 显式fan-in文档 + 图校验)

        工作流拓扑 (v2.8 串行):

            data_preparation → technical_analysis → market_scanner
            (拉数据+算指标)    (代码计算指标+LLM解读)  (LLM市况扫描)
                ↓
            strategy_matching → backtest_verification → adversarial_debate
            (市场适配策略)       (STRATEGY_BACKTESTERS)   (逐股Bull/Bear/Judge)
                ↓
            synthesis → END
            (含 stock_recommendations)
        """
        workflow = StateGraph(MarketAnalysisState)

        # ---- 注册所有节点 ----
        workflow.add_node("data_preparation", self._data_preparation_node)
        workflow.add_node("market_scanner", self._market_scanner_node)
        workflow.add_node("technical_analysis", self._technical_analysis_node)
        workflow.add_node("strategy_matching", self._strategy_matching_node)
        workflow.add_node("backtest_verification", self._backtest_verification_node)
        workflow.add_node("adversarial_debate", self._adversarial_debate_node)
        workflow.add_node("synthesis", self._synthesis_node)

        # ---- 边定义 ----
        workflow.set_entry_point("data_preparation")

        # v2.8 串行执行 (避免 LangGraph 并行分支的 channel 冲突)
        # market_scanner 和 technical_analysis 共享 ModelRouter, 无法真正并行
        workflow.add_edge("data_preparation", "technical_analysis")
        workflow.add_edge("technical_analysis", "market_scanner")

        # 汇聚到策略匹配
        workflow.add_edge("market_scanner", "strategy_matching")

        # Phase 2→3: 串行 — 策略匹配 → 回测验证
        workflow.add_edge("strategy_matching", "backtest_verification")

        # Phase 3→4: 串行 — 回测 → 多空辩论 → 综合研判
        workflow.add_edge("backtest_verification", "adversarial_debate")
        workflow.add_edge("adversarial_debate", "synthesis")

        # 终点
        workflow.add_edge("synthesis", END)

        compiled = workflow.compile()
        try:
            graph_info = compiled.get_graph()
            node_count = len(graph_info.nodes) if hasattr(graph_info, 'nodes') else len(list(graph_info.nodes))
            logger.info(f"[Workflow] 图编译完成, 节点数={node_count}")
        except Exception:
            logger.info("[Workflow] 图编译完成")
        return compiled

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
        数据准备节点 (v2.9: 并发拉取 + DataRequest.recent)

        - 拉取市场行情数据 (并发)
        - 识别市场状态
        - 拉取另类数据
        """
        logger.info(f"[数据准备] task={state['task_id']}")

        try:
            from data import get_data_router
            from analysis import TechnicalAnalyzer, MarketRegimeDetector
            from data.providers.base import DataFrequency, DataRequest

            router = get_data_router()
            analyzer = TechnicalAnalyzer()
            detector = MarketRegimeDetector()
            symbols = state.get("symbols", [])

            # 并发: 指数数据 + 所有标的行情
            async def fetch_index():
                req = DataRequest.recent("sh.000300", days=252)
                result = await router.get_daily_kline(req)
                return detector.detect(result.data)

            async def fetch_symbol(sym):
                try:
                    req = DataRequest.recent(sym, days=365 * 3)
                    result = await router.get_daily_kline(req)
                    if not result.data.empty:
                        return sym, result.data
                except Exception as e:
                    state["errors"].append({"node": "data_prep", "symbol": sym, "error": str(e)})
                return sym, None

            # 并发执行所有数据拉取
            index_task = fetch_index()
            symbol_tasks = [fetch_symbol(sym) for sym in symbols]
            all_tasks = [index_task] + symbol_tasks

            results = await asyncio.gather(*all_tasks)

            # 第一个结果是市场状态
            regime_result = results[0]
            state["market_regime"] = regime_result.regime.value
            state["regime_confidence"] = regime_result.confidence
            state["regime_params"] = get_regime_parameters(regime_result.regime)

            # 其余是标的行情
            market_data = {}
            for sym, df in results[1:]:
                if df is not None:
                    market_data[sym] = df

            state["market_data"] = market_data

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
        """技术分析节点 (v2.9: 并发处理, 进度日志)"""
        logger.info(f"[技术分析] task={state['task_id']}")

        try:
            from analysis import TechnicalAnalyzer

            analyzer = TechnicalAnalyzer()
            reports = {}
            indicators = {}
            semaphore = asyncio.Semaphore(6)
            total = len(state.get("market_data", {}))
            completed = 0

            async def _process_one(sym, df):
                nonlocal completed
                async with semaphore:
                    if df.empty:
                        return
                    result = analyzer.compute_all(df, symbol=sym, skip_patterns=True)
                    indicators[sym] = result.to_dataframe()
                    reports[sym] = ""

                    if self.router:
                        system_prompt = (
                            self.knowledge.get_system_prompt("technical_analyst")
                            if self.knowledge else
                            "你是A股技术分析Agent。"
                        )
                        last_row = result.to_dataframe().iloc[-1] if not result.to_dataframe().empty else {}
                        summary = json.dumps({
                            k: round(float(v), 2) if isinstance(v, (int, float)) else str(v)
                            for k, v in last_row.items()
                            if k in ["close", "ma_5", "ma_20", "ma_60", "rsi_14",
                                     "macd_dif", "macd_dea", "bb_pct_20", "atr_14",
                                     "trend_score", "composite_score"]
                        }, ensure_ascii=False)

                        try:
                            result_llm = await self.router.route(
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"标的: {sym}\n指标数据: {summary}"},
                                ],
                                task_type="technical_analysis",
                            )
                            reports[sym] = result_llm.response
                        except Exception:
                            reports[sym] = f"[{sym} API错误]"
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        logger.info(f"  [技术分析] {completed}/{total} 完成")

            tasks = [_process_one(sym, df) for sym, df in state.get("market_data", {}).items()]
            await asyncio.gather(*tasks)

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
                    base_score = 0.7 if regime in s.get("market_regimes", []) else 0.3

                    indicators_raw = state.get("technical_indicators", {}).get(sym, {})
                    trend_score = self._get_indicator_value(indicators_raw, "trend_score", 0.0)
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
        回测验证节点 (v2.2 修复: 使用真实策略回测器)

        对匹配的策略执行快速历史回测验证,
        使用 analysis.recommender.STRATEGY_BACKTESTERS 中的真实策略逻辑,
        计算Sharpe/胜率/最大回撤等关键指标。
        """
        logger.info(f"[回测验证] task={state['task_id']}")

        results = {}

        try:
            from analysis.recommender import STRATEGY_BACKTESTERS

            for sym, matches in state.get("strategy_matches", {}).items():
                market_data = state.get("market_data", {}).get(sym)
                if market_data is None or market_data.empty:
                    continue

                sym_results = []
                for match in matches[:2]:  # 只验证Top 2策略
                    strategy_id = match["strategy"]
                    try:
                        bt_func = STRATEGY_BACKTESTERS.get(strategy_id)
                        if bt_func is None:
                            sym_results.append({
                                "strategy": strategy_id,
                                "error": f"策略{strategy_id}无注册回测函数",
                            })
                            continue

                        bt = bt_func(market_data)
                        sym_results.append({
                            "strategy": strategy_id,
                            "total_return": round(bt.get("expected_value", 0) * max(bt.get("signals", 1), 1), 2),
                            "sharpe": round(bt.get("sharpe", 0), 3),
                            "max_drawdown": round(bt.get("max_dd", 0), 1),
                            "win_rate": round(bt.get("win_rate", 0), 3),
                            "profit_factor": round(bt.get("profit_factor", 1), 2),
                            "total_trades": bt.get("signals", 0),
                        })
                        logger.debug(
                            f"[回测验证] {sym} {strategy_id}: "
                            f"wr={bt['win_rate']:.1%} sr={bt['sharpe']:.2f} "
                            f"trades={bt['signals']}"
                        )
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

    # ═══════════════════════════════════════════════════════════════
    # 节点6: 多空辩论 (v2.1 完整实现)
    # ═══════════════════════════════════════════════════════════════

    async def _adversarial_debate_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """
        🆕 v2.8 多空辩论节点 — 逐股辩论 + 数据驱动

        三Agent对抗:
          Bull Researcher → 基于量化数据找看涨论据
          Bear Researcher → 基于同一份量化数据找看跌论据
          Judge           → 评估论据质量 → 输出逐股交易建议

        v2.8 核心改进:
          1. 每只股票独立辩论（不复用LLM分析文本）
          2. 辩论上下文注入真实指标和回测数据
          3. Bull/Bear看到同一份原始数据，各自解读
          4. Judge输出结构化推荐: {action, conviction, key_reasons, risks}
        """
        logger.info(f"[多空辩论] task={state['task_id']}")

        if self.router is None:
            state["debate_result"] = {
                "bull_argument": "", "bear_argument": "", "judge_verdict": "",
                "bull_score": 0.5, "bear_score": 0.5,
                "direction": "neutral", "confidence": 0.0, "divergence": 1.0,
                "summary": "模型路由未初始化,跳过辩论",
            }
            state["stock_recommendations"] = {}
            return state

        regime = state.get("market_regime", "range_bound")
        regime_params = state.get("regime_params", {})
        symbols = state.get("symbols", [])
        market_data = state.get("market_data", {})
        indicators = state.get("technical_indicators", {})
        backtests = state.get("backtest_results", {})
        matches = state.get("strategy_matches", {})

        if not symbols:
            state["debate_result"] = {
                "direction": "neutral", "confidence": 0,
                "divergence": 1.0, "summary": "无待分析标的",
            }
            state["stock_recommendations"] = {}
            return state

        stock_recommendations: Dict[str, Dict] = {}
        all_bull_args: List[str] = []
        all_bear_args: List[str] = []
        debate_errors = []
        semaphore = asyncio.Semaphore(5)
        total_syms = len(symbols)
        completed_syms = 0

        async def _debate_one(sym):
            nonlocal completed_syms
            async with semaphore:
                try:
                    logger.info(f"  [辩论] {sym} ({completed_syms+1}/{total_syms})...")
                    rec = await self._debate_single_stock(
                        sym, regime, regime_params,
                        market_data.get(sym),
                        indicators.get(sym, {}),
                        backtests.get(sym, []),
                        matches.get(sym, []),
                    )
                    completed_syms += 1
                    if completed_syms % 10 == 0 or completed_syms == total_syms:
                        logger.info(f"  [辩论] {completed_syms}/{total_syms} 完成")
                    return sym, rec, None
                except Exception as e:
                    completed_syms += 1
                    return sym, None, str(e)

        try:
            results = await asyncio.gather(*[_debate_one(sym) for sym in symbols])
            for sym, rec, err in results:
                if err:
                    debate_errors.append({"symbol": sym, "error": err})
                    stock_recommendations[sym] = {
                        "action": "HOLD", "conviction": 0.0,
                        "key_reasons": [f"辩论异常: {err}"],
                        "risks": ["无法完成分析"],
                    }
                else:
                    stock_recommendations[sym] = rec
                    all_bull_args.append(f"[{sym}] {rec.get('bull_summary', '')}")
                    all_bear_args.append(f"[{sym}] {rec.get('bear_summary', '')}")

            # ---- 汇总裁判 ----
            if len(symbols) > 1 and all_bull_args and all_bear_args:
                try:
                    summary_prompt = (
                        "你是A股策略裁判。以下是多只标的的多空辩论摘要:\n\n"
                        f"市场状态: {regime} (多头仓位建议: {regime_params.get('position_multiplier', 0.5):.0%})\n\n"
                        f"### 多头论据汇总\n" + "\n".join(all_bull_args[:5000]) + "\n\n"
                        f"### 空头论据汇总\n" + "\n".join(all_bear_args[:5000]) + "\n\n"
                        "请完成:\n"
                        "1. 整体市场方向判断 (BULLISH/BEARISH/NEUTRAL)\n"
                        "2. 综合评分: 1-10 (10=强烈看多)\n"
                        "3. 当前环境下最值得关注的前3只标的及理由\n"
                        "4. 主要风险提示"
                    )
                    judge_result = await self.router.route(
                        messages=[
                            {"role": "system", "content": "你是公正的策略裁判,只基于具体数据做出判断。"},
                            {"role": "user", "content": summary_prompt},
                        ],
                        task_type="adversarial_debate",
                    )

                    scores = self._parse_debate_scores(judge_result.response)
                    state["debate_result"] = {
                        "bull_argument": "\n\n".join(all_bull_args[:3]),
                        "bear_argument": "\n\n".join(all_bear_args[:3]),
                        "judge_verdict": judge_result.response,
                        "bull_score": scores["bull_score"],
                        "bear_score": scores["bear_score"],
                        "direction": scores["direction"],
                        "confidence": scores["confidence"],
                        "divergence": scores["divergence"],
                        "summary": judge_result.response[:1000],
                    }
                except Exception as e:
                    logger.warning(f"汇总裁判失败: {e}")
                    state["debate_result"] = {
                        "direction": "neutral", "confidence": 0, "divergence": 1.0,
                        "summary": f"汇总异常: {e}",
                    }
            elif len(symbols) == 1:
                # 单只股票: 逐股辩论的结果就是最终结果
                rec = stock_recommendations.get(symbols[0], {})
                direction_map = {"BUY": "bullish", "SELL": "bearish", "HOLD": "neutral"}
                bull_score = {"BUY": 0.7, "SELL": 0.3, "HOLD": 0.5}.get(rec.get("action", "HOLD"), 0.5)
                state["debate_result"] = {
                    "bull_argument": rec.get("bull_summary", ""),
                    "bear_argument": rec.get("bear_summary", ""),
                    "judge_verdict": rec.get("judge_verdict", ""),
                    "bull_score": bull_score,
                    "bear_score": 1.0 - bull_score,
                    "direction": direction_map.get(rec.get("action", "HOLD"), "neutral"),
                    "confidence": rec.get("conviction", 0),
                    "divergence": 1.0 - rec.get("conviction", 0),
                    "summary": rec.get("judge_verdict", "")[:1000],
                }

        except Exception as e:
            state["errors"].append({"node": "adversarial_debate", "error": str(e)})
            state["debate_result"] = {
                "bull_argument": "", "bear_argument": "", "judge_verdict": "",
                "bull_score": 0.5, "bear_score": 0.5,
                "direction": "neutral", "confidence": 0.0, "divergence": 1.0,
                "summary": f"辩论异常: {e}",
            }

        state["stock_recommendations"] = stock_recommendations
        if debate_errors:
            state["errors"].extend(
                {"node": "adversarial_debate", "symbol": e["symbol"], "error": e["error"]}
                for e in debate_errors
            )
        return state

    async def _debate_single_stock(
        self, sym: str, regime: str, regime_params: dict,
        df, indicators_dict, backtest_list, strategy_list,
    ) -> Dict[str, Any]:
        """
        🆕 v2.8 单只股票的完整辩论流程

        给 Bull/Bear 注入同一份量化数据,
        让双方基于真实数字而非LLM文本进行对抗论证。
        """
        # 构建数据上下文 — 真实指标, 非LLM文本
        data_context = self._build_stock_data_context(
            sym, regime, regime_params, df, indicators_dict, backtest_list, strategy_list
        )

        # ---- Bull Researcher ----
        bull_prompt = (
            "你是A股多头研究员。以下是一只股票的量化数据,请基于这些数据找出看涨的理由。\n\n"
            "分析维度 (每个维度必须引用具体数据):\n"
            "1. 技术面: 均线排列/金叉死叉/RSI位置/布林带位置→有什么看涨信号?\n"
            "2. 回测面: 策略胜率/夏普比率/盈亏比→是否支持做多?\n"
            "3. 市场适配: 当前市场状态下的策略适配度→是否有利于该股票?\n\n"
            "输出格式:\n"
            "### 看涨论据\n"
            "1. [论据] (数据支撑: 引用具体数值)\n"
            "2. [论据] (数据支撑: ...)\n"
            "3. [论据] (数据支撑: ...)\n"
            "### 多头确信度: 0.0-1.0"
        )

        # ---- Bear Researcher ----
        bear_prompt = (
            "你是A股空头研究员。以下是一只股票的量化数据,请基于这些数据找出看跌的理由。\n\n"
            "分析维度 (每个维度必须引用具体数据):\n"
            "1. 技术风险: 均线空排/RSI超买/背离/布林带收窄→有什么看跌信号?\n"
            "2. 策略风险: 回测最大回撤/胜率不足/信号稀疏→有什么风险?\n"
            "3. 市场风险: 当前市场状态对该股票的负面影响→有什么隐患?\n\n"
            "输出格式:\n"
            "### 看跌论据\n"
            "1. [论据] (数据支撑: 引用具体数值)\n"
            "2. [论据] (数据支撑: ...)\n"
            "3. [论据] (数据支撑: ...)\n"
            "### 空头确信度: 0.0-1.0"
        )

        # 并行调用 Bull + Bear
        import asyncio
        bull_task = self.router.route(
            messages=[
                {"role": "system", "content": bull_prompt},
                {"role": "user", "content": data_context},
            ],
            task_type="adversarial_debate",
        )
        bear_task = self.router.route(
            messages=[
                {"role": "system", "content": bear_prompt},
                {"role": "user", "content": data_context},
            ],
            task_type="adversarial_debate",
        )

        bull_result, bear_result = await asyncio.gather(bull_task, bear_task)

        # ---- Judge ----
        judge_prompt = (
            f"你是A股策略裁判。以下是 {sym} 的多空辩论,双方基于同一份量化数据:\n\n"
            f"=== 原始数据 ===\n{data_context[:2000]}\n\n"
            f"=== 多头论据 ===\n{bull_result.response[:1200]}\n\n"
            f"=== 空头论据 ===\n{bear_result.response[:1200]}\n\n"
            "请评估并输出 JSON (不要输出其他内容, 只输出 JSON):\n"
            "{\n"
            f'  "action": "BUY" | "HOLD" | "SELL",\n'
            '  "conviction": 0.0-1.0,\n'
            '  "score": 1-10 (10=强烈看多),\n'
            '  "key_reasons": ["论据1", "论据2", "论据3"],\n'
            '  "risks": ["风险1", "风险2"],\n'
            '  "bull_quality": 1-5 (多头论据质量),\n'
            '  "bear_quality": 1-5 (空头论据质量),\n'
            '  "verdict_summary": "一句话总结裁判意见"\n'
            "}"
        )
        judge_result = await self.router.route(
            messages=[
                {"role": "system", "content": "你是公正的策略裁判。只输出JSON,不要输出其他内容。"},
                {"role": "user", "content": judge_prompt},
            ],
            task_type="adversarial_debate",
        )

        # 解析 Judge 的 JSON
        judge_json = {}
        try:
            judge_text = judge_result.response.strip()
            # 提取 JSON (可能被包裹在 ```json ... ``` 中)
            if "```json" in judge_text:
                judge_text = judge_text.split("```json")[1].split("```")[0]
            elif "```" in judge_text:
                judge_text = judge_text.split("```")[1].split("```")[0]
            judge_json = json.loads(judge_text)
        except (json.JSONDecodeError, IndexError):
            # JSON 解析失败, 尝试从文本中提取关键词
            judge_json = {
                "action": "HOLD",
                "conviction": 0.3,
                "score": 5,
                "key_reasons": ["Judge JSON解析失败, 使用默认"],
                "risks": ["裁判结论不可用"],
                "bull_quality": 3,
                "bear_quality": 3,
                "verdict_summary": judge_result.response[:200],
            }

        return {
            "action": judge_json.get("action", "HOLD"),
            "conviction": judge_json.get("conviction", 0.5),
            "score": judge_json.get("score", 5),
            "key_reasons": judge_json.get("key_reasons", []),
            "risks": judge_json.get("risks", []),
            "bull_quality": judge_json.get("bull_quality", 3),
            "bear_quality": judge_json.get("bear_quality", 3),
            "verdict_summary": judge_json.get("verdict_summary", ""),
            "bull_summary": (bull_result.response or "")[:600],
            "bear_summary": (bear_result.response or "")[:600],
            "judge_verdict": judge_result.response[:1000],
        }

    @staticmethod
    def _get_indicator_value(indicators_dict, key: str, default=0.0) -> float:
        """
        🆕 v2.8 从可能嵌套的指标字典中安全提取标量值

        TechnicalAnalyzer.to_dataframe().iloc[-1] 返回 Series → 直接 float
        IndicatorResult.to_dict() 可能返回嵌套 dict → 取最后一个有效值
        """
        if not isinstance(indicators_dict, dict):
            return default
        val = indicators_dict.get(key, default)
        if isinstance(val, dict):
            vals = [v for v in val.values() if isinstance(v, (int, float))]
            return float(vals[-1]) if vals else default
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _build_stock_data_context(
        self, sym: str, regime: str, regime_params: dict,
        df, indicators_dict, backtest_list, strategy_list,
    ) -> str:
        """🆕 v2.8 构建单只股票的量化数据上下文 — 喂给Bull/Bear"""
        parts = [f"## {sym} 量化数据\n"]
        parts.append(f"市场状态: {regime}")

        # 趋势信息
        if df is not None and hasattr(df, 'iloc') and len(df) > 0:
            try:
                close = float(df["close"].iloc[-1])
                parts.append(f"最新收盘价: {close:.2f}")
                if len(df) >= 20:
                    chg_5d = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
                    chg_20d = (df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100
                    parts.append(f"近5日涨跌: {chg_5d:+.1f}% | 近20日: {chg_20d:+.1f}%")
            except Exception:
                pass

        # 技术指标
        if isinstance(indicators_dict, dict) and indicators_dict:
            parts.append("\n### 技术指标")
            key_fields = [
                ("rsi_14", "RSI(14)", ".1f"),
                ("trend_score", "趋势评分(-1~+1)", ".2f"),
                ("composite_score", "综合评分(0-100)", ".1f"),
                ("bias_ma20", "偏离MA20(%)", ".2f"),
                ("vol_ratio_5", "量比(5日)", ".2f"),
            ]
            for field, label, fmt in key_fields:
                val = self._get_indicator_value(indicators_dict, field, None)
                if val is not None:
                    parts.append(f"- {label}: {val:{fmt}}")

            # 信号检测
            signals = []
            rsi = self._get_indicator_value(indicators_dict, "rsi_14", 50)
            if rsi > 70: signals.append("RSI超买(>70)")
            elif rsi < 30: signals.append("RSI超卖(<30)")
            elif rsi > 60: signals.append("RSI偏强(>60)")
            elif rsi < 40: signals.append("RSI偏弱(<40)")

            try:
                macd_gc = int(self._get_indicator_value(indicators_dict, "macd_golden_cross", 0))
                if macd_gc > 0: signals.append("MACD金叉")
            except: pass
            try:
                macd_dc = int(self._get_indicator_value(indicators_dict, "macd_death_cross", 0))
                if macd_dc > 0: signals.append("MACD死叉")
            except: pass

            trend = self._get_indicator_value(indicators_dict, "trend_score", 0)
            if trend > 0.5: signals.append("多头排列(趋势>0.5)")
            elif trend < -0.5: signals.append("空头排列(趋势<-0.5)")

            if signals:
                parts.append(f"信号: {', '.join(signals)}")

        # 回测结果
        if backtest_list:
            parts.append("\n### 策略回测验证")
            for bt in backtest_list:
                if "error" in bt:
                    parts.append(f"- {bt['strategy']}: ❌ {bt['error']}")
                else:
                    parts.append(
                        f"- {bt['strategy']}: "
                        f"胜率={bt.get('win_rate', 0):.1%} | "
                        f"夏普={bt.get('sharpe', 0):.2f} | "
                        f"回撤={bt.get('max_drawdown', 0):.1f}% | "
                        f"盈亏比={bt.get('profit_factor', 1):.1f}x | "
                        f"交易{bt.get('total_trades', 0)}次"
                    )

        # 策略匹配
        if strategy_list:
            parts.append("\n### 策略适配度")
            for s in strategy_list[:3]:
                parts.append(
                    f"- {s['strategy']}: 适配分={s.get('fit_score', 0):.0%} | "
                    f"类别={s.get('category', 'N/A')}"
                )

        parts.append(f"\n市场参数: 仓位系数={regime_params.get('position_multiplier', 0.5):.0%}")

        return "\n".join(parts)

    @staticmethod
    def _parse_debate_scores(judge_text: str) -> dict:
        """
        v2.2 从裁判评论文本中提取结构化评分

        解析策略:
        1. 正则匹配 "评分: X/10" 或 "综合评分: X" 等模式获取数值
        2. 检测看多/看空关键词确定方向
        3. 通过评分离散度和分歧描述词估算置信度

        Returns:
            {bull_score, bear_score, direction, confidence, divergence}
        """
        import re

        bull_score = 0.5
        bear_score = 0.5
        direction = "neutral"
        confidence = 0.5

        # ---- Step 1: 提取数值评分 ----
        # 匹配 "评分: 7/10", "综合评分: 6", "得分: 5.5/10" 等
        score_patterns = [
            r'评[分估][:：]\s*(\d+\.?\d*)\s*(?:/\s*10)?',
            r'得[分][:：]\s*(\d+\.?\d*)\s*(?:/\s*10)?',
            r'综合[评比][分估][:：]\s*(\d+\.?\d*)\s*(?:/\s*10)?',
            r'(\d+\.?\d*)\s*/\s*10',
            r'(\d+\.?\d*)\s*分',
        ]

        numeric_score = None
        for pat in score_patterns:
            m = re.search(pat, judge_text)
            if m:
                val = float(m.group(1))
                # 归一化到 0-1
                if val > 10:
                    val = val / 10 if val <= 100 else val / 100
                numeric_score = max(0, min(1.0, val / 10))
                break

        if numeric_score is not None:
            # 1-10 映射: 1=强烈看空, 5.5=中性, 10=强烈看多
            bull_score = round(numeric_score, 3)
            bear_score = round(1.0 - numeric_score, 3)

        # ---- Step 2: 方向检测 ----
        bullish_keywords = ['看多', '做多', '多头', '看涨', '买入', 'bull', '乐观', '积极', '上行']
        bearish_keywords = ['看空', '做空', '空头', '看跌', '卖出', 'bear', '悲观', '下行', '风险']

        bull_hits = sum(1 for kw in bullish_keywords if kw in judge_text)
        bear_hits = sum(1 for kw in bearish_keywords if kw in judge_text)

        if bull_hits > bear_hits:
            direction = "bullish"
        elif bear_hits > bull_hits:
            direction = "bearish"
        else:
            direction = "neutral"

        # 如果没有数值评分,从方向估算
        if numeric_score is None:
            if direction == "bullish":
                bull_score = 0.65
                bear_score = 0.35
            elif direction == "bearish":
                bull_score = 0.35
                bear_score = 0.65

        # ---- Step 3: 置信度 & 分歧度 ----
        # 分歧词越多 → 置信度越低
        divergence_keywords = ['分歧', '不确定', '矛盾', '模棱两可', '难以判断',
                               '各有道理', '不确定性', '两难', '争议']
        divergence_hits = sum(1 for kw in divergence_keywords if kw in judge_text)
        divergence = min(1.0, divergence_hits * 0.25)

        # 置信度: 评分越极端→置信越高, 分歧越大→置信越低
        score_extremity = abs(bull_score - 0.5) * 2  # 偏离中性的程度
        confidence = max(0.1, min(1.0, 0.5 + score_extremity * 0.4 - divergence * 0.3))

        return {
            "bull_score": bull_score,
            "bear_score": bear_score,
            "direction": direction,
            "confidence": round(confidence, 3),
            "divergence": round(divergence, 3),
        }

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

                # 构建完整上下文 — v2.8: 含逐股推荐
                stock_recs = state.get("stock_recommendations", {})
                stock_summary = {
                    sym: {
                        "action": rec.get("action", "HOLD"),
                        "conviction": rec.get("conviction", 0),
                        "key_reasons": rec.get("key_reasons", [])[:2],
                        "risks": rec.get("risks", [])[:2],
                    }
                    for sym, rec in stock_recs.items()
                }

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
                             "win_rate": r.get("win_rate"), "profit_factor": r.get("profit_factor"),
                             "total_trades": r.get("total_trades")}
                            for r in v if "error" not in r]
                        for k, v in state.get("backtest_results", {}).items()
                    },
                    "debate": {
                        "direction": state.get("debate_result", {}).get("direction", "neutral"),
                        "confidence": state.get("debate_result", {}).get("confidence", 0),
                        "divergence": state.get("debate_result", {}).get("divergence", 0),
                        "summary": state.get("debate_result", {}).get("summary", ""),
                    },
                    "stock_recommendations": stock_summary,  # 🆕 v2.8 逐股推荐
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
            stock_recs = state.get("stock_recommendations", {})

            report = [
                f"## A股每日分析报告",
                f"",
                f"**日期**: {state.get('date', 'N/A')}",
                f"**市场状态**: {regime} (置信度: {state.get('regime_confidence', 'N/A')})",
                f"",
            ]

            if debate:
                direction = debate.get("direction", "unknown")
                report.append(f"### 整体方向: {direction} | 置信度: {debate.get('confidence', 0):.0%}")

            # 🆕 v2.8: 逐股推荐
            if stock_recs:
                report.append(f"\n### 逐股推荐 ({len(stock_recs)}只)")
                action_emoji = {"BUY": "🟢", "HOLD": "⚪", "SELL": "🔴"}
                for sym, rec in stock_recs.items():
                    code = sym.replace("sh.", "").replace("sz.", "")
                    emoji = action_emoji.get(rec.get("action", "HOLD"), "❓")
                    report.append(
                        f"- {emoji} **{code}**: {rec.get('action', 'HOLD')} "
                        f"(确信度: {rec.get('conviction', 0):.0%})"
                    )
                    if rec.get("key_reasons"):
                        for r in rec["key_reasons"][:2]:
                            report.append(f"  - {r}")
                    if rec.get("risks"):
                        for r in rec["risks"][:1]:
                            report.append(f"  - ⚠️ {r}")

            report.append(f"\n### 标的详情 ({len(symbols)}只)")
            for sym in symbols:
                code = sym.replace("sh.", "").replace("sz.", "")
                report.append(f"- {code}")
                sym_matches = matches.get(sym, [])
                if sym_matches:
                    for m in sym_matches[:2]:
                        report.append(f"  - {m['strategy']} (适配分: {m['fit_score']})")

            if debate:
                report.append(f"\n### 辩论摘要\n{debate.get('summary', 'N/A')[:300]}")

            state["final_report"] = "\n".join(report)

        return state
