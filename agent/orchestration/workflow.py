"""
LangGraph工作流 — AI Agent编排引擎 (v3.1)

分析流水线:
  数据准备 → 技术分析 → 市场扫描 → 策略匹配 → 回测验证
  → 多空辩论 → Critic审计 → 综合研判(含反思注入)
  → 决策持久化 → END

v3.1 新增:
  - CriticAgent 独立审计节点 (对抗性验证)
  - DecisionLogger 决策持久化 (SQLite)
  - AgentMemory 反思注入 (历史决策→Prompt)
  - CheckpointManager 断点续跑 (SqliteSaver)
"""

import asyncio
import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, StateGraph
from loguru import logger

from .state import MarketAnalysisState
from .memory import AgentMemory
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
        # v3.0: 实例化正式Agent类 (替代内联LLM调用)
        self._agent_instances = {}
        # v3.1: 决策记忆与自进化
        self._memory = AgentMemory()
        self._decision_logger = None  # 延迟初始化
        self.graph = self._build_graph()

    def _init_decision_logger(self):
        """v3.1: 延迟初始化 DecisionLogger"""
        if self._decision_logger is None:
            from agent.orchestration.decision_log import DecisionLogger
            self._decision_logger = DecisionLogger()
            self._memory.set_decision_logger(self._decision_logger)

    def _get_agent(self, name: str):
        """v3.0: 懒加载Agent实例"""
        if name not in self._agent_instances:
            from agent.sub_agents import create_agent
            self._agent_instances[name] = create_agent(
                name, knowledge_manager=self.knowledge, model_router=self.router
            )
        return self._agent_instances[name]

    def _build_graph(self) -> StateGraph:
        """
        构建LangGraph有向图 (v3.1: 新增 Critic审计 + Checkpoint断点续跑)

        工作流拓扑 (v3.1):

            data_preparation → technical_analysis → market_scanner
            (拉数据+算指标)    (代码计算指标+LLM解读)  (LLM市况扫描)
                ↓
            strategy_matching → backtest_verification → adversarial_debate
            (市场适配策略)       (STRATEGY_BACKTESTERS)   (逐股Bull/Bear/Judge)
                ↓
            critic_audit → synthesis → END
            (CriticAgent (含反思注入+
             5维审计)    决策持久化)
        """
        workflow = StateGraph(MarketAnalysisState)

        # ---- 注册所有节点 ----
        workflow.add_node("data_preparation", self._data_preparation_node)
        workflow.add_node("market_scanner", self._market_scanner_node)
        workflow.add_node("technical_analysis", self._technical_analysis_node)
        workflow.add_node("strategy_matching", self._strategy_matching_node)
        workflow.add_node("backtest_verification", self._backtest_verification_node)
        workflow.add_node("adversarial_debate", self._adversarial_debate_node)
        workflow.add_node("critic_audit", self._critic_node)  # v3.1 新增
        workflow.add_node("synthesis", self._synthesis_node)
        # 🆕 v3.1-deerflow: 反思-修订循环 (Producer→Evaluator→Revise)
        workflow.add_node("evaluation", self._evaluation_node)
        workflow.add_node("synthesis_revision", self._synthesis_revision_node)

        # ---- 边定义 ----
        workflow.set_entry_point("data_preparation")

        # v2.8 串行执行 (避免 LangGraph 并行分支的 channel 冲突)
        workflow.add_edge("data_preparation", "technical_analysis")
        workflow.add_edge("technical_analysis", "market_scanner")

        # 汇聚到策略匹配
        workflow.add_edge("market_scanner", "strategy_matching")

        # Phase 2→3: 策略匹配 → 回测验证
        workflow.add_edge("strategy_matching", "backtest_verification")

        # Phase 3→4: 回测 → 多空辩论 → Critic审计 → 综合研判
        workflow.add_edge("backtest_verification", "adversarial_debate")
        workflow.add_edge("adversarial_debate", "critic_audit")     # v3.1 新增
        workflow.add_edge("critic_audit", "synthesis")              # v3.1 修改

        # v3.1-deerflow: synthesis → evaluation → (通过=END | 不通过=回炉修订, 最多2轮)
        workflow.add_edge("synthesis", "evaluation")
        workflow.add_conditional_edges(
            "evaluation",
            self._route_after_evaluation,
            {"synthesis_revision": "synthesis_revision", END: END},
        )
        workflow.add_edge("synthesis_revision", "evaluation")

        # v3.1: 尝试注册 SqliteSaver checkpoint
        checkpointer = None
        try:
            from agent.orchestration.checkpoint import CheckpointManager
            cpm = CheckpointManager()
            checkpointer = cpm.get_saver()
            if checkpointer:
                logger.info("[Workflow] Checkpoint 已启用 (SqliteSaver)")
        except Exception as e:
            logger.debug(f"[Workflow] Checkpoint 未启用: {e}")

        compiled = workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()
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
        """运行每日市场扫描 (v3.1: 支持 checkpoint 恢复)"""
        task_id = str(uuid.uuid4())[:8]
        initial_state: MarketAnalysisState = {
            "task_id": task_id,
            "task_type": "daily_scan",
            "date": date.today().isoformat(),
            "symbols": symbols or [],
            "errors": [],
            "model_trace": [],
            "signals_archived": False,
            "reflection_round": 0,
            "synthesis_revisions": [],
        }
        config = {"configurable": {"thread_id": task_id}}
        result = await self.graph.ainvoke(initial_state, config=config)
        return result

    async def run_stock_analysis(
        self, symbol: str
    ) -> MarketAnalysisState:
        """运行单只股票深度分析 (v3.1: 支持 checkpoint 恢复)"""
        task_id = str(uuid.uuid4())[:8]
        initial_state: MarketAnalysisState = {
            "task_id": task_id,
            "task_type": "stock_analysis",
            "date": date.today().isoformat(),
            "symbols": [symbol],
            "errors": [],
            "model_trace": [],
            "signals_archived": False,
            "reflection_round": 0,
            "synthesis_revisions": [],
        }
        config = {"configurable": {"thread_id": task_id}}
        result = await self.graph.ainvoke(initial_state, config=config)
        return result

    def _load_agent_prompt(self, agent_name: str) -> Optional[str]:
        """
        v3.0-competition: 从知识库加载Agent提示词

        如果KnowledgeManager可用,使用其加载提示词(含模板注入);
        否则返回 None (调用方使用 fallback 默认值)

        Args:
            agent_name: Agent名称 (如 'bull_researcher', 'judge')

        Returns:
            完整提示词文本,或 None
        """
        if self.knowledge:
            prompt = self.knowledge.get_system_prompt(agent_name)
            if prompt and "Prompt文件缺失" not in prompt:
                # 去除 YAML frontmatter (如果存在)
                if prompt.startswith("---"):
                    try:
                        end = prompt.index("---", 3)
                        prompt = prompt[end + 3:].strip()
                    except ValueError:
                        pass
                return prompt
        return None

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

            results = await asyncio.gather(*all_tasks, return_exceptions=True)

            # 第一个结果是市场状态
            regime_result = results[0]
            if isinstance(regime_result, Exception):
                logger.warning(f"[DataPrep] Market regime detection failed: {regime_result}")
                state["market_regime"] = "range_bound"
                state["regime_confidence"] = 0.5
                state["regime_params"] = get_regime_parameters("range_bound")
            else:
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

            # v3.1.2: 附加全市场广度快照 + v3.2 情绪温度计 (失败则降级为原提示词, 零行为变化)
            try:
                from analysis.market_breadth import format_snapshot, live_market_snapshot
                snap = await live_market_snapshot()
                snap_txt = format_snapshot(snap)
                if snap_txt:
                    context += f"\n市场形势快照: {snap_txt}"
                # v3.2: 情绪温度计 (固定锚点标定, 无需历史z)
                from factors.market_sentiment import format_live_sentiment_snapshot
                sent_txt = format_live_sentiment_snapshot(snap)
                if sent_txt:
                    context += f"\n{sent_txt}"
            except Exception as e:
                logger.warning(f"市场快照注入失败: {e}")

            # v3.3: RAG 知识注入 — 当前 regime 作战手册 + 向量检索 (让 AI 判断结合历史实证)
            try:
                if self.knowledge is not None and regime != "unknown":
                    pb = self.knowledge.get_regime_playbook(regime)
                    if pb:
                        context += f"\n【{regime} 作战手册 (历史实证)】\n{pb}"
                    rag = self.knowledge.rag_query(
                        f"{regime} 市场 操作策略 选股 风险 止损", top_k=2, regime=regime)
                    if rag:
                        context += f"\n{rag[:2000]}"
            except Exception as e:
                logger.warning(f"regime 知识注入失败: {e}")

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

                    # 🆕 v3.0: Code-as-Reasoning 集成 — 锁定数字,防止LLM幻觉
                    code_as_reasoning_report = None
                    if self.router and not result.to_dataframe().empty:
                        try:
                            from agent.tools.code_executor import CodeAsReasoningPipeline
                            pipeline = CodeAsReasoningPipeline()
                            code_report = await pipeline.run(
                                plan=f"验证{sym}的关键技术指标计算",
                                context={
                                    "ohlcv_df": df.tail(60),
                                    "indicators": result.to_dataframe().to_dict(),
                                },
                                router=self.router,
                            )
                            if code_report.num_safe and code_report.computed_numbers:
                                code_as_reasoning_report = code_report
                                logger.debug(f"[{sym}] Code-as-Reasoning: {len(code_report.computed_numbers)}个数值已锁定")
                        except Exception as e:
                            logger.debug(f"[{sym}] Code-as-Reasoning跳过: {e}")

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
                            # v3.0: 增强上下文 — 包含Code-as-Reasoning锁定的数值
                            user_content = f"标的: {sym}\n指标数据: {summary}"
                            if code_as_reasoning_report:
                                locked_nums = {
                                    k: {"value": v.value, "unit": v.unit, "formula": v.formula}
                                    for k, v in code_as_reasoning_report.computed_numbers.items()
                                }
                                user_content += (
                                    f"\n\n🔒 代码验证锁定的数值 (必须引用,不可编造):\n"
                                    f"{json.dumps(locked_nums, ensure_ascii=False, indent=2)}"
                                )
                            result_llm = await self.router.route(
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_content},
                                ],
                                task_type="technical_analysis",
                            )
                            reports[sym] = result_llm.response

                            # 数字安全校验: 验证最终报告中的数值与Python计算的指标数据一致
                            # v3.0: 修复 last_row 为 pandas Series 时 isinstance(dict) 恒 False + numpy 未导入
                            # 导致校验从不执行的问题; 校验失败不再静默吞掉。
                            rows = last_row.to_dict() if hasattr(last_row, "to_dict") else (last_row or {})
                            if reports[sym] and rows:
                                try:
                                    import numpy as np
                                    from agent.tools.code_executor import ComputedNumber, NumericSafetyChecker
                                    computed = {
                                        k: ComputedNumber(name=k, value=float(v),
                                                          source=f"computed({k})", formula=k)
                                        for k, v in rows.items()
                                        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))
                                    }
                                    checker = NumericSafetyChecker(computed)
                                    safe, violations = checker.validate_report(reports[sym])
                                    if not safe:
                                        logger.warning(f"[{sym}] 数字校验未通过: {'; '.join(violations[:3])}")
                                except Exception as e:
                                    logger.debug(f"[{sym}] 数字校验异常: {e}")
                        except Exception:
                            reports[sym] = f"[{sym} API错误]"
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        logger.info(f"  [技术分析] {completed}/{total} 完成")

            tasks = [_process_one(sym, df) for sym, df in state.get("market_data", {}).items()]
            await asyncio.gather(*tasks, return_exceptions=True)

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

    # 🆕 v3.1-deerflow: 多视角发散 — 3个分析视角 × 多空双阵营 = 6路并行论证
    LENS_SPECS = {
        "trend": {
            "label": "技术趋势",
            "bull_focus": ("均线排列/金叉死叉/RSI位置/布林带位置/动量指标 → 找出看涨的技术信号"
                           "(如多头排列、RSI从超卖回升、MACD金叉)"),
            "bear_focus": ("均线空排/RSI超买/顶背离/布林带收窄/动量衰竭 → 找出看跌的技术信号"
                           "(如死叉、超买回落、趋势破位)"),
        },
        "valuation": {
            "label": "估值与基本面",
            "bull_focus": ("PE/PB/ROE/增速/现金流/估值分位 → 找估值洼地、盈利改善、安全边际"
                           "(如PE处于低位、ROE持续改善)"),
            "bear_focus": ("估值偏高/增速放缓/现金流恶化/盈利质量存疑 → 找基本面隐患"
                           "(如高估值透支成长、ROE下滑)"),
        },
        "risk": {
            "label": "风险与市场",
            "bull_focus": ("市场状态(regime)/策略适配度/回测胜率/夏普/盈亏比 → 找顺势做多的环境支撑"
                           "(如牛市状态+策略适配+回测胜率达标)"),
            "bear_focus": ("市场风险/回测最大回撤/胜率不足/信号稀疏/板块集中 → 找环境与回测层面的隐患"
                           "(如震荡市追高风险、回测回撤过大)"),
        },
    }

    async def _debate_single_stock(
        self, sym: str, regime: str, regime_params: dict,
        df, indicators_dict, backtest_list, strategy_list,
    ) -> Dict[str, Any]:
        """
        🆕 v3.1-deerflow 单只股票的"头脑风暴式"多视角辩论

        DeerFlow Brainstormer 协作模式移植:
          1. 3个独立分析视角(技术/估值/风险) × 多空 = 6路并行论证, 彼此隔离上下文
          2. Judge 对全部论据综合评估 → 输出裁决 + 指出最弱视角
          3. 若需改进, 只定向修订最弱视角的多头/空头论据 (reflexion循环, 最多2轮)
          4. 所有论据基于同一份量化数据, 防止LLM编造数字
        """
        # 构建数据上下文 — 真实指标, 非LLM文本
        data_context = self._build_stock_data_context(
            sym, regime, regime_params, df, indicators_dict, backtest_list, strategy_list
        )

        import asyncio

        def _lens_prompt(lens: str, side: str) -> str:
            """按视角+阵营构造提示词 (v3.1-deerflow)"""
            spec = self.LENS_SPECS[lens]
            role = "多头研究员" if side == "bull" else "空头研究员"
            camp_word = "看涨" if side == "bull" else "看跌"
            focus = spec["bull_focus"] if side == "bull" else spec["bear_focus"]
            return (
                f"你是A股{role},采用【{spec['label']}】视角。以下是一只股票的量化数据,"
                f"请只从这个视角找出{spec['label']}相关的{camp_word}理由。\n\n"
                f"分析重点:\n{focus}\n\n"
                f"要求 (必须引用具体数值, 不得编造数据):\n"
                f"1. 给出 2-3 条该视角下的{'看涨' if side == 'bull' else '看跌'}论据\n"
                f"2. 每条论据后标注其数据支撑\n"
                f"3. 该视角{'多头' if side == 'bull' else '空头'}确信度: 0.0-1.0\n\n"
                f"输出格式:\n### {spec['label']} {'看涨' if side == 'bull' else '看跌'}论据\n"
                f"1. [论据] (数据支撑: 引用具体数值)\n..."
            )

        lenses = list(self.LENS_SPECS.keys())

        # ---- 第一轮: 6路并行论证 (3视角 × 多空) ----
        bull_results: Dict[str, Any] = {}
        bear_results: Dict[str, Any] = {}
        first_round_tasks = []
        for lens in lenses:
            first_round_tasks.append((
                ("bull", lens), self.router.route(
                    messages=[
                        {"role": "system", "content": _lens_prompt(lens, "bull")},
                        {"role": "user", "content": data_context},
                    ],
                    task_type="adversarial_debate",
                )
            ))
            first_round_tasks.append((
                ("bear", lens), self.router.route(
                    messages=[
                        {"role": "system", "content": _lens_prompt(lens, "bear")},
                        {"role": "user", "content": data_context},
                    ],
                    task_type="adversarial_debate",
                )
            ))
        first_results = await asyncio.gather(*[t for _, t in first_round_tasks])
        for (side, lens), res in zip([k for k, _ in first_round_tasks], first_results):
            (bull_results if side == "bull" else bear_results)[lens] = res

        def _camp_text(camp_results: Dict[str, Any]) -> str:
            """拼接某阵营所有视角的论据文本"""
            parts = []
            for lens in lenses:
                resp = camp_results.get(lens)
                if resp and resp.response:
                    parts.append(f"[{self.LENS_SPECS[lens]['label']}]\n{resp.response[:600]}")
            return "\n\n".join(parts)

        # ---- 多轮辩论迭代 (reflexion, 最多2轮) ----
        max_rounds = 2
        judge_result = None
        for round_num in range(1, max_rounds + 1):
            judge_system = self._load_agent_prompt("judge") or "你是公正的策略裁判。只输出JSON。"
            judge_prompt = (
                f"你是A股策略裁判。以下是 {sym} 的多视角辩论第{round_num}轮:\n\n"
                f"=== 原始数据 ===\n{data_context[:2000]}\n\n"
                f"=== 多头论据 ({len(lenses)}视角) ===\n{_camp_text(bull_results)[:2500]}\n\n"
                f"=== 空头论据 ({len(lenses)}视角) ===\n{_camp_text(bear_results)[:2500]}\n\n"
                "请整合全部论据, 识别最强的论据与最弱的视角, 输出JSON:\n"
                "{\"action\":\"BUY|HOLD|SELL\",\"conviction\":0.0,\"score\":1-10,"
                "\"key_reasons\":[],\"risks\":[],\"bull_quality\":1-5,\"bear_quality\":1-5,"
                "\"improvement_needed\":true|false,"
                "\"weakest_bull_lens\":\"trend|valuation|risk\",\"weakest_bear_lens\":\"trend|valuation|risk\","
                "\"critique_for_bull\":\"\",\"critique_for_bear\":\"\"}"
            )
            judge_result = await self.router.route(
                messages=[
                    {"role": "system", "content": judge_system},
                    {"role": "user", "content": judge_prompt},
                ],
                task_type="judge_verdict",
            )

            judge_json = {}
            try:
                judge_text = judge_result.response.strip()
                if "```json" in judge_text:
                    judge_text = judge_text.split("```json")[1].split("```")[0]
                elif "```" in judge_text:
                    judge_text = judge_text.split("```")[1].split("```")[0]
                judge_json = json.loads(judge_text)
            except (json.JSONDecodeError, IndexError):
                judge_json = {"action": "HOLD", "conviction": 0.3, "improvement_needed": False}

            if not judge_json.get("improvement_needed", False) or round_num >= max_rounds:
                break

            # 定向修订最弱视角 (DeerFlow reflexion: 不重跑全部, 只改短板)
            crit_bull = judge_json.get("critique_for_bull", "")
            crit_bear = judge_json.get("critique_for_bear", "")
            weak_bull_lens = judge_json.get("weakest_bull_lens") or "trend"
            weak_bear_lens = judge_json.get("weakest_bear_lens") or "trend"
            if weak_bull_lens not in lenses:
                weak_bull_lens = "trend"
            if weak_bear_lens not in lenses:
                weak_bear_lens = "trend"

            revise_tasks = []
            if crit_bull:
                revise_tasks.append(self.router.route(
                    messages=[
                        {"role": "system", "content": _lens_prompt(weak_bull_lens, "bull")},
                        {"role": "user", "content": data_context},
                        {"role": "assistant", "content": bull_results[weak_bull_lens].response},
                        {"role": "user", "content": f"裁判批评: {crit_bull}\n请从【{self.LENS_SPECS[weak_bull_lens]['label']}】视角修改你的看涨论据,回应批评。只输出修改后的完整论据。"},
                    ],
                    task_type="adversarial_debate",
                ))
            if crit_bear:
                revise_tasks.append(self.router.route(
                    messages=[
                        {"role": "system", "content": _lens_prompt(weak_bear_lens, "bear")},
                        {"role": "user", "content": data_context},
                        {"role": "assistant", "content": bear_results[weak_bear_lens].response},
                        {"role": "user", "content": f"裁判批评: {crit_bear}\n请从【{self.LENS_SPECS[weak_bear_lens]['label']}】视角修改你的看跌论据,回应批评。只输出修改后的完整论据。"},
                    ],
                    task_type="adversarial_debate",
                ))
            if revise_tasks:
                revised = await asyncio.gather(*revise_tasks)
                idx = 0
                if crit_bull:
                    bull_results[weak_bull_lens] = revised[idx]; idx += 1
                if crit_bear:
                    bear_results[weak_bear_lens] = revised[idx]

        # 解析最终 Judge JSON
        judge_json = {}
        if judge_result is not None:
            try:
                judge_text = judge_result.response.strip()
                if "```json" in judge_text:
                    judge_text = judge_text.split("```json")[1].split("```")[0]
                elif "```" in judge_text:
                    judge_text = judge_text.split("```")[1].split("```")[0]
                judge_json = json.loads(judge_text)
            except (json.JSONDecodeError, IndexError):
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

        # 汇总: 多头/空头论据按视角拼接, 标注来源视角
        def _camp_summary(camp_results: Dict[str, Any]) -> str:
            parts = []
            for lens in lenses:
                resp = camp_results.get(lens)
                if resp and resp.response:
                    parts.append(f"[{self.LENS_SPECS[lens]['label']}] {resp.response.strip()[:200]}")
            return " | ".join(parts)[:600]

        return {
            "action": judge_json.get("action", "HOLD"),
            "conviction": judge_json.get("conviction", 0.5),
            "score": judge_json.get("score", 5),
            "key_reasons": judge_json.get("key_reasons", []),
            "risks": judge_json.get("risks", []),
            "bull_quality": judge_json.get("bull_quality", 3),
            "bear_quality": judge_json.get("bear_quality", 3),
            "verdict_summary": judge_json.get("verdict_summary", ""),
            "bull_summary": _camp_summary(bull_results),
            "bear_summary": _camp_summary(bear_results),
            "judge_verdict": (judge_result.response if judge_result else "")[:1000],
            "lenses_used": lenses,
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
                if macd_gc > 0: signals.append("MACD golden cross")
            except (ValueError, TypeError):
                pass
            try:
                macd_dc = int(self._get_indicator_value(indicators_dict, "macd_death_cross", 0))
                if macd_dc > 0: signals.append("MACD death cross")
            except (ValueError, TypeError):
                pass

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

        # 🆕 v3.0: ChromaDB RAG — 检索相似历史K线形态
        if self.knowledge and self.knowledge.chroma_available:
            try:
                # 构建当前K线形态的查询向量
                query_text = f"{sym} K线形态: "
                if rsi := self._get_indicator_value(indicators_dict, "rsi_14", None):
                    query_text += f"RSI={rsi:.0f} "
                if trend := self._get_indicator_value(indicators_dict, "trend_score", None):
                    query_text += f"趋势={trend:+.2f} "
                if composite := self._get_indicator_value(indicators_dict, "composite_score", None):
                    query_text += f"评分={composite:.1f}"

                similar = self.knowledge.search_similar_klines(
                    query_text, top_k=3
                )
                if similar and len(similar) > 0:
                    parts.append("\n### 🔍 历史相似形态 (ChromaDB检索)")
                    for i, match in enumerate(similar, 1):
                        parts.append(
                            f"{i}. {match.get('pattern', '未知形态')} "
                            f"(相似度: {match.get('similarity', 0):.0%}) "
                            f"— {match.get('description', '')}"
                        )
                    logger.debug(f"[{sym}] ChromaDB RAG: 检索到{len(similar)}个相似形态")
            except Exception as e:
                logger.debug(f"[{sym}] ChromaDB RAG跳过: {e}")

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
    # 节点7: Critic 独立审计 (v3.1 新增)
    # ═══════════════════════════════════════════════════════════════

    async def _critic_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """
        🆕 v3.1 CriticAgent 独立审计节点

        在辩论完成后、综合研判之前,用规则引擎+LLM对每个推荐进行5维审计:
          1. Look-Ahead Bias
          2. Regime Cherry-Picking
          3. Overfitting
          4. Cost Optimism
          5. Fragility

        审计结果注入 state,供 synthesis 节点参考。
        """
        logger.info(f"[Critic审计] task={state['task_id']}")

        critic_results = {}
        stock_recs = state.get("stock_recommendations", {})
        backtest_results = state.get("backtest_results", {})
        regime = state.get("market_regime", "unknown")

        if not stock_recs:
            state["critic_results"] = {}
            return state

        try:
            from agent.sub_agents.critic import CriticAgent, CriticResult

            critic = CriticAgent(knowledge_manager=self.knowledge, model_router=self.router)

            for sym, rec in stock_recs.items():
                bt_data = backtest_results.get(sym, [])

                # 构建审计输入 — 尽可能注入真实回测数据而非硬编码常量
                bt_metrics: Dict[str, Any] = {}
                for b in bt_data:
                    if not isinstance(b, dict):
                        continue
                    for k in ("win_rate", "max_drawdown", "signals", "sharpe"):
                        if k in b:
                            bt_metrics.setdefault(k, b[k])
                bt_metrics.setdefault("strategies_tested", len(bt_data) if isinstance(bt_data, list) else 0)

                ctx = critic._start_context(
                    task_id=state.get("task_id", ""),
                    symbol=sym,
                    backtest_results={"metrics": {
                        "win_rate": bt_metrics.get("win_rate", rec.get("conviction", 0.5)),
                        "max_drawdown": bt_metrics.get("max_drawdown",
                            max((b.get("max_drawdown", 0) for b in bt_data if isinstance(b, dict) and "max_drawdown" in b), default=0)),
                        "pbo": bt_metrics.get("pbo", 0.0),
                        "best_sharpe": bt_metrics.get("sharpe", 0.0),
                        "strategies_tested": bt_metrics.get("strategies_tested", 0),
                        "signal_count": bt_metrics.get("signals", 0),
                    }},
                    strategy_config={
                        "transaction_cost_pct": 0.0031,
                        "slippage_enabled": True,
                        "pit_enabled": True,
                        "strategy_id": rec.get("strategy_id", rec.get("id", "")),
                    },
                    regime_history={
                        "distribution": {regime: rec.get("win_rate", 0.5)},
                        "years_covered": state.get("hist_years",
                            len(bt_data) if isinstance(bt_data, list) else 3),
                    },
                    indicators=state.get("technical_indicators", {}).get(sym, {}),
                )

                audit_result = await critic.run(ctx)
                critic_results[sym] = audit_result.data if audit_result.success else {}

            state["critic_results"] = critic_results
            total_flaws = sum(
                c.get("flaw_count", 0)
                for c in critic_results.values()
            )
            avg_robustness = (
                sum(c.get("robustness_score", 100) for c in critic_results.values())
                / max(len(critic_results), 1)
            )
            logger.info(
                f"[Critic审计] 完成: {len(critic_results)}只标的, "
                f"发现{total_flaws}个缺陷, 平均稳健性={avg_robustness:.0f}/100"
            )

        except Exception as e:
            state["errors"].append({"node": "critic_audit", "error": str(e)})
            logger.warning(f"[Critic审计] 跳过: {e}")
            state["critic_results"] = {}

        return state

    # ═══════════════════════════════════════════════════════════════
    # 节点8: 综合研判 (v3.1 增强 — 含反思注入+Critic审计+决策持久化)
    # ═══════════════════════════════════════════════════════════════

    async def _synthesis_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """综合研判节点 (v3.1) — 汇总所有分析 + 反思注入 + 决策持久化"""
        logger.info(f"[综合研判] 生成最终报告")

        # v3.1: 加载历史记忆,注入反思上下文
        symbols = state.get("symbols", [])
        reflection_contexts = {}
        for sym in symbols:
            self._memory.load_historical(sym, days=5)
            reflection = self._memory.get_reflection(sym, days=5)
            if reflection:
                reflection_contexts[sym] = reflection

        if self.router and self.knowledge:
            try:
                system_prompt = self.knowledge.get_system_prompt("synthesis")

                # 构建完整上下文 — v3.1: 含逐股推荐 + Critic审计 + 反思
                stock_recs = state.get("stock_recommendations", {})
                critic_results = state.get("critic_results", {})
                # v3.0: 代码计算交易参数(入场/止损/止盈/仓位), 消除 LLM 编造价格
                # 容错提取: 兼容 technical_indicators 键格式差异, 回退 market_data OHLCV 末行
                tech_ind = state.get("technical_indicators", {})
                market_data = state.get("market_data", {})
                trade_params = {}
                for sym in stock_recs.keys():
                    ind = tech_ind.get(sym) or tech_ind.get(sym.split(".")[-1]) or {}
                    last = {}
                    if ind:
                        for k, v in ind.items():
                            if isinstance(v, dict) and v:
                                last[k] = v[max(v)]  # DataFrame.to_dict: 取最后一行
                            elif isinstance(v, list) and v:
                                last[k] = v[-1]
                            elif isinstance(v, (int, float)):
                                last[k] = v
                    elif not last:
                        # 回退: 直接取 market_data OHLCV 末行 (一定含 close)
                        # 注意: 不能用 `dfx = a or b` (DataFrame 的 or 触发歧义真值错误)
                        dfx = market_data.get(sym)
                        if dfx is None:
                            dfx = market_data.get(sym.split(".")[-1])
                        if dfx is not None and not getattr(dfx, "empty", True):
                            last = dfx.iloc[-1].to_dict()
                    close = float(last.get("close") or 0)
                    if not close:
                        # 指标 df 不含 OHLC 列 (只有 atr_14 等) → 回退 market_data 末行拿 close
                        # 注意: 不能用 `dfx = a or b` (DataFrame 的 or 触发歧义真值错误)
                        dfx = market_data.get(sym)
                        if dfx is None:
                            dfx = market_data.get(sym.split(".")[-1])
                        if dfx is not None and not getattr(dfx, "empty", True):
                            _lr = dfx.iloc[-1].to_dict()
                            close = float(_lr.get("close") or 0)
                            if not last.get("atr_14"):
                                last["atr_14"] = float(_lr.get("atr_14") or 0) or close * 0.02
                    atr = float(last.get("atr_14") or 0) or close * 0.02
                    conv = float(stock_recs[sym].get("conviction", 0.5))
                    if close > 0:
                        trade_params[sym] = {
                            "entry_price": round(close, 2),
                            "stop_loss": round(close - 2 * atr, 2),
                            "take_profit": round(close + 1.5 * atr, 2),
                            "position_pct": round(min(0.20, max(0.03, 0.05 + conv * 0.15)), 3),
                            "note": "代码计算: 入场=最新收盘, 止损=收盘-2×ATR, 止盈=收盘+1.5×ATR, 仓位=5%+15%×确信度",
                        }
                    else:
                        trade_params[sym] = {}

                # 🆕 v3.1-deerflow: 修复 trade_params 未写回 bug — 把代码计算的交易参数
                # 合并进 stock_recommendations, 让下游 morning_buy/daily_runner 直接使用
                # (此前只注入 prompt, state 里的推荐没有交易参数)
                for _sym, _tp in trade_params.items():
                    if stock_recs.get(_sym):
                        stock_recs[_sym] = {**stock_recs[_sym], "trade_params": _tp}
                state["stock_recommendations"] = stock_recs

                # 🆕 v3.1-deerflow: 决策验证器 — 执行前硬约束校验 + 拒绝日志落盘
                # 仅首轮运行 (trade_params 是代码确定值, 回炉轮次结果不变, 避免 journal 重复)
                if state.get("reflection_round", 0) == 0:
                    try:
                        from agent.sub_agents.validator import DecisionValidator
                        vd = DecisionValidator()
                        v_result = await vd.run(vd._start_context(
                            task_id=state.get("task_id", ""),
                            trading_params=trade_params,
                            stock_recommendations=stock_recs,
                            market_data=market_data,
                        ))
                        state["validation_results"] = (
                            v_result.data.get("validation_results", {}) if v_result.success else {}
                        )
                        _rejects = v_result.data.get("reject_count", 0) if v_result.success else 0
                        if _rejects:
                            logger.warning(
                                f"[验证器] {_rejects}只标的推荐未通过硬约束校验, "
                                f"详情见 validation_journal"
                            )
                    except Exception as _ve:
                        state["errors"].append({"node": "validator", "error": str(_ve)})
                        logger.warning(f"[验证器] 跳过: {_ve}")

                stock_summary = {}
                for sym, rec in stock_recs.items():
                    critic = critic_results.get(sym, {})
                    stock_summary[sym] = {
                        "action": rec.get("action", "HOLD"),
                        "conviction": rec.get("conviction", 0),
                        "key_reasons": rec.get("key_reasons", [])[:2],
                        "risks": rec.get("risks", [])[:2],
                        "critic_flaws": critic.get("flaw_count", 0),
                        "robustness": critic.get("robustness_score", 100),
                        "reflection": reflection_contexts.get(sym, "")[:300],
                        "trade_params": trade_params.get(sym, {}),
                    }

                context_parts = {
                    "date": state.get("date", ""),
                    "regime": state.get("market_regime"),
                    "regime_confidence": state.get("regime_confidence"),
                    # v3.0: 代码计算的交易参数放最前, 避免被长上下文截断 (此前在 stock_summary
                    # 尾部, 上下文 [:N] 截断后 LLM 只能输出 null)
                    "trading_params": trade_params,
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
                    "stock_recommendations": stock_summary,
                    "reflections": reflection_contexts,  # v3.1 反思上下文
                }

                # v3.0: v4-flash 支持 1M 上下文, 放宽截断 (此前 12K 会切掉尾部关键字段)
                context = json.dumps(context_parts, ensure_ascii=False, default=str)[:30000]

                # 🆕 v3.1-deerflow: 回炉修订 — 注入 Evaluator 批评
                critique = state.get("synthesis_critique", "")
                if critique:
                    context = (
                        f"【上一次报告未通过质量评估, 请按以下批评修订后重新输出完整报告】\n"
                        f"{critique}\n\n{context}"
                    )
                    # 修订批评已消费, 避免下一轮重复注入
                    state["synthesis_critique"] = ""

                result = await self.router.route(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context},
                    ],
                    task_type="daily_synthesis",
                )
                state["final_report"] = result.response

                # v3.0: 代码兜底覆盖交易参数 — LLM 常输出 0/null (此前 null→0.0),
                # 用 computed trade_params 强制填入, 消除日报幻觉价格
                try:
                    _parsed = json.loads(result.response)
                    _recs = _parsed.get("recommendations", [])
                    _filled = 0
                    for _rec in _recs:
                        _tp = trade_params.get(_rec.get("symbol"))
                        if _tp:
                            for _k in ("entry_price", "stop_loss", "take_profit", "position_pct"):
                                if not _rec.get(_k):
                                    _rec[_k] = _tp[_k]
                                    _filled += 1
                    state["final_report"] = json.dumps(_parsed, ensure_ascii=False, indent=2)
                    # 仅在实际计算失败(推荐对应 trade_params 为空)时告警
                    if _recs and not any(trade_params.get(r.get("symbol")) for r in _recs):
                        logger.warning(
                            f"[综合研判] trade_params 计算为空 (tech_keys={list(tech_ind.keys())[:3]}, "
                            f"mkt_keys={list(market_data.keys())[:3]})"
                        )
                except Exception as _e:
                    logger.debug(f"[综合研判] 交易参数兜底失败: {_e}")

            except Exception as e:
                state["final_report"] = (
                    f"## 综合研判报告\n\n"
                    f"生成失败: {e}\n\n"
                    f"请检查模型路由和知识库配置。"
                )

        else:
            # 生成无LLM的基本摘要
            regime = state.get("market_regime", "unknown")
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

        # ── v3.1: Signal tracking + 决策持久化 ──
        try:
            from analysis.winrate import SignalTracker
            tracker = SignalTracker()
            for sym, rec in stock_recs.items():
                if rec.get("action") in ("BUY", "SELL"):
                    tracker.emit(
                        symbol=sym,
                        signal=rec.get("action", "HOLD"),
                        confidence=rec.get("conviction", 0),
                        strategy_id=rec.get("strategy_id", ""),
                        market_regime=state.get("market_regime", "unknown"),
                        bull_score=debate.get("bull_score", 0.5),
                        bear_score=debate.get("bear_score", 0.5),
                        debate_result=debate.get("direction", "neutral"),
                        report_snippet=(rec.get("verdict_summary", "") or "")[:200],
                    )
            logger.debug(f"[SignalTracker] {len(stock_recs)} signals emitted for future review")
        except Exception as e:
            logger.debug(f"[SignalTracker] Skip: {e}")
        # ──
        try:
            self._init_decision_logger()
            debate = state.get("debate_result", {})
            critic = state.get("critic_results", {})
            analysis_date = state.get("date", date.today().isoformat())

            for sym, rec in stock_recs.items():
                rec_critic = critic.get(sym, {})
                self._decision_logger.log_analysis(
                    symbol=sym,
                    analysis_date=analysis_date,
                    task_id=state.get("task_id", ""),
                    market_regime=state.get("market_regime", "unknown"),
                    regime_confidence=state.get("regime_confidence", 0),
                    bull_score=debate.get("bull_score", 0.5),
                    bear_score=debate.get("bear_score", 0.5),
                    debate_result=debate.get("direction", "neutral"),
                    bull_key_points={"summary": rec.get("bull_summary", "")},
                    bear_key_points={"summary": rec.get("bear_summary", "")},
                    final_signal=rec.get("action", "HOLD"),
                    confidence=rec.get("conviction", 0),
                    reasoning=rec.get("verdict_summary", ""),
                    risk_flaws=rec_critic.get("flaw_count", 0),
                    robustness_score=rec_critic.get("robustness_score", 100),
                )
            logger.info(f"[综合研判] {len(stock_recs)}条决策已持久化到 DecisionLogger")
        except Exception as e:
            logger.warning(f"[综合研判] 决策持久化跳过: {e}")

        return state

    # ═══════════════════════════════════════════════════════════════
    # 🆕 v3.1-deerflow: 全局反思-修订循环 (Producer→Evaluator→Revise)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _route_after_evaluation(state: MarketAnalysisState) -> str:
        """
        评估节点后的路由决策 (DeerFlow Reflexion 循环):

          - 评估通过 → END
          - 评估不通过 且 已达最多2轮修订 → END (防止无限回炉)
          - 否则 → synthesis_revision (带批评回炉重写)
        """
        ev = state.get("evaluation_result", {}) or {}
        if ev.get("passed"):
            return END
        if state.get("reflection_round", 0) >= 2:
            logger.warning(
                f"[评估] 达到最大修订轮次({state.get('reflection_round', 0)}轮), "
                f"报告仍不达标 (score={ev.get('score', 0)}), 停止回炉"
            )
            return END
        return "synthesis_revision"

    async def _evaluation_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """
        🆕 v3.1-deerflow EvaluatorAgent 评估节点

        对 synthesis 产出的最终报告逐项打分 (trade_params缺失/自相矛盾/幻觉数字/
        稳健性低等), 结果存入 state, 驱动是否回炉。
        """
        logger.info(f"[评估] task={state['task_id']}")

        if self.router is None:
            # 无 LLM 时 synthesis 产出的是模板报告, 跳过评估
            state["evaluation_result"] = {"passed": True, "score": 100.0,
                                          "issues": [], "critique": ""}
            return state

        try:
            from agent.sub_agents.evaluator import EvaluatorAgent

            # 代码计算的交易参数 (synthesis 内已算好, 从 final_report 兜底逻辑复用的同款数据)
            tech_ind = state.get("technical_indicators", {})
            market_data = state.get("market_data", {})
            trade_params = self._compute_trade_params(
                state.get("stock_recommendations", {}),
                tech_ind, market_data,
            )

            evaluator = EvaluatorAgent(knowledge_manager=self.knowledge, model_router=self.router)
            result = await evaluator.run(evaluator._start_context(
                task_id=state.get("task_id", ""),
                final_report=state.get("final_report", ""),
                trading_params=trade_params,
                stock_recommendations=state.get("stock_recommendations", {}),
                critic_results=state.get("critic_results", {}),
            ))
            state["evaluation_result"] = result.data.get("evaluation_result", {}) if result.success else {}
            if not result.success:
                state["evaluation_result"] = {"passed": True, "score": 100.0,
                                              "issues": [], "critique": ""}
            logger.info(
                f"[评估] passed={state['evaluation_result'].get('passed')} "
                f"score={state['evaluation_result'].get('score')} "
                f"issues={state['evaluation_result'].get('issue_count')}"
            )
        except Exception as e:
            state["errors"].append({"node": "evaluation", "error": str(e)})
            logger.warning(f"[评估] 跳过, 按通过处理: {e}")
            state["evaluation_result"] = {"passed": True, "score": 100.0,
                                          "issues": [], "critique": ""}

        return state

    async def _synthesis_revision_node(
        self, state: MarketAnalysisState
    ) -> MarketAnalysisState:
        """
        🆕 v3.1-deerflow 回炉修订节点

        将 Evaluator 的批评注入 synthesis 上下文, 重新生成报告 (重跑 synthesis),
        然后回到评估节点进行第二轮评估。
        """
        round_no = state.get("reflection_round", 0) + 1
        state["reflection_round"] = round_no
        critique = (state.get("evaluation_result", {}) or {}).get("critique", "")
        state["synthesis_critique"] = critique
        state["synthesis_revisions"].append(critique)
        logger.info(f"[回炉修订] 第{round_no}轮, 注入批评 {len(critique)}字符")
        # 回炉 = 带着批评重跑 synthesis, 产生新报告后再评估
        return await self._synthesis_node(state)

    @staticmethod
    def _compute_trade_params(
        stock_recs: Dict[str, Dict],
        tech_ind: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, Dict]:
        """
        代码计算交易参数 (v3.1-deerflow 抽取为独立方法)

        与 synthesis 内逻辑一致: 入场=最新收盘, 止损=收盘-2×ATR,
        止盈=收盘+1.5×ATR, 仓位=5%+15%×确信度 (封顶20%)。
        供 Evaluator 用代码计算的 ground-truth 校验报告。
        """
        trade_params: Dict[str, Dict] = {}
        for sym, rec in stock_recs.items():
            ind = tech_ind.get(sym) or tech_ind.get(sym.split(".")[-1]) or {}
            last = {}
            if ind:
                for k, v in ind.items():
                    if isinstance(v, dict) and v:
                        last[k] = v[max(v)]
                    elif isinstance(v, list) and v:
                        last[k] = v[-1]
                    elif isinstance(v, (int, float)):
                        last[k] = v
            close = float(last.get("close") or 0)
            if not close:
                dfx = market_data.get(sym)
                if dfx is None:
                    dfx = market_data.get(sym.split(".")[-1])
                if dfx is not None and not getattr(dfx, "empty", True):
                    _lr = dfx.iloc[-1].to_dict()
                    close = float(_lr.get("close") or 0)
                    if not last.get("atr_14"):
                        last["atr_14"] = float(_lr.get("atr_14") or 0) or close * 0.02
            atr = float(last.get("atr_14") or 0) or close * 0.02
            conv = float(rec.get("conviction", 0.5))
            if close > 0:
                trade_params[sym] = {
                    "entry_price": round(close, 2),
                    "stop_loss": round(close - 2 * atr, 2),
                    "take_profit": round(close + 1.5 * atr, 2),
                    "position_pct": round(min(0.20, max(0.03, 0.05 + conv * 0.15)), 3),
                }
        return trade_params
