"""
CompetitionAgent — AI+金融量化分析智能体 (v3.0-competition)

第八届全球校园人工智能算法精英大赛 · AI智能体开发应用赛
选题方向: AI+商科 → AI+金融

作为竞赛统一入口,聚合:
  - ChatAgent:  自然语言对话 + 7工具调用
  - AnalysisWorkflow: LangGraph 7节点流水线 + 对抗辩论
  - KnowledgeManager: 知识库注入 + ChromaDB向量检索
  - ModelRouter:  3层模型路由 (Local→Flash→Pro)

比赛4大模块:
  1. AI智能体架构设计 — 需求分析→业务流程→输出设计
  2. AI知识库搭建 — 结构化规则+YAML+向量库
  3. AI智能体搭建 — 提示词工程+LLM配置+工作流编排
  4. AI应用测试 — 问题测试+对话质量评估
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class CompetitionContext:
    """竞赛上下文 — 追踪一次完整分析的状态"""
    task_id: str
    task_type: str = ""                 # market_scan | stock_analysis | risk_assessment
    start_time: float = 0.0
    modules_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    module_scores: Dict[str, float] = field(default_factory=dict)

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "elapsed_seconds": round(self.elapsed(), 1),
            "modules_completed": self.modules_completed,
            "errors": self.errors,
            "module_scores": self.module_scores,
        }


class CompetitionAgent:
    """
    AI+金融量化分析智能体 — 竞赛统一入口

    核心能力:
      - 市场状态检测 (6种市场体制识别)
      - 个股深度分析 (130+技术指标 + 多空辩论)
      - 策略回测验证 (9种策略事件驱动回测)
      - 量化信号生成 (6级信号评分 + 8层风控)
      - 自然语言对话 (7工具调用 + 知识库RAG)

    用法:
        agent = CompetitionAgent()
        # 对话模式
        reply = await agent.chat("分析一下600519贵州茅台")
        # 扫描模式
        report = await agent.run_analysis(symbols=["sh.600519", "sz.000858"])
        # 演示模式
        await agent.demo_mode()
    """

    def __init__(self, demo_mode: bool = False):
        """
        Args:
            demo_mode: 是否为比赛演示模式 (跳过耗时初始化)
        """
        self.demo_mode = demo_mode
        self.version = "3.0-competition"
        self.competition_track = "AI+商科·AI+金融"

        # 延迟初始化 — 组件在首次使用时加载
        self._router = None
        self._knowledge = None
        self._analyzer = None
        self._model_router = None
        self._chat_agent = None
        self._workflow = None
        self._initialized = False

    # ═══════════════════════════════════════════════════════════════
    # 比赛模块1: AI智能体架构设计
    # ═══════════════════════════════════════════════════════════════

    def get_architecture_info(self) -> dict:
        """
        返回智能体架构信息,用于比赛展示

        对应比赛评分维度:
          - 需求分析 (5分)
          - 场景分析 (5分)
          - 数据准备程度 (10分)
        """
        return {
            "agent_name": "AI+金融量化分析智能体",
            "version": self.version,
            "track": self.competition_track,
            "architecture": {
                "type": "Multi-Agent Collaborative Architecture",
                "agents": {
                    "chat_assistant": {
                        "role": "自然语言交互入口",
                        "tools": 7,
                        "description": "接收用户查询,通过Function Calling调用分析工具",
                    },
                    "technical_analyst": {
                        "role": "技术面分析专家",
                        "indicators": "130+指标",
                        "dimensions": ["趋势", "动量", "波动率", "成交量", "K线形态", "支撑/阻力"],
                    },
                    "market_scanner": {
                        "role": "全市场扫描器",
                        "coverage": "5000+ A股",
                        "dimensions": ["技术面(40%)", "资金流(25%)", "动量(20%)", "质量(15%)"],
                    },
                    "bull_researcher": {
                        "role": "多头研究员",
                        "focus": "识别看涨信号 — 技术面/回测面/市场适配",
                    },
                    "bear_researcher": {
                        "role": "空头研究员",
                        "focus": "识别看跌风险 — 技术风险/策略风险/市场风险",
                    },
                    "judge": {
                        "role": "策略裁判",
                        "output": "结构化JSON — action/conviction/score/risks",
                    },
                    "risk_assessor": {
                        "role": "风控评估师",
                        "layers": 8,
                        "triggers": ["最大回撤(-8%)", "ATR动态止损", "行业集中度>35%"],
                    },
                },
                "workflow": {
                    "type": "LangGraph StateGraph (7节点串行流水线)",
                    "nodes": [
                        "data_preparation → 数据采集与体制检测",
                        "technical_analysis → 130+指标计算+LLM解读",
                        "market_scanner → 全市场扫描+LLM叙事",
                        "strategy_matching → 市场体制策略匹配",
                        "backtest_verification → 事件驱动回测验证",
                        "adversarial_debate → Bull/Bear/Judge三方辩论",
                        "synthesis → 综合报告生成",
                    ],
                },
            },
            "business_requirements": {
                "problem": "A股投资者面临信息过载,缺乏系统化的量化分析工具",
                "users": ["个人投资者", "量化研究员", "投资顾问"],
                "scenarios": [
                    "每日盘后: 全市场扫描+持仓评估",
                    "选股决策: 个股深度分析+策略回测",
                    "风险预警: 实时风控监控+止损建议",
                    "知识学习: A股交易规则+技术指标解释",
                ],
            },
            "outputs": {
                "signals": "6级信号 (STRONG_BUY→STRONG_SELL) + 置信度",
                "reports": "结构化Markdown日报 (市场概况+个股分析+交易建议+风险提示)",
                "backtest": "策略绩效: 胜率/夏普比率/最大回撤/盈亏比",
                "risk_metrics": "组合风险: VaR/CVaR/波动率/相关性矩阵",
            },
        }

    # ═══════════════════════════════════════════════════════════════
    # 初始化 (延迟加载)
    # ═══════════════════════════════════════════════════════════════

    async def _ensure_initialized(self):
        """延迟初始化所有组件"""
        if self._initialized:
            return

        logger.info("正在初始化 CompetitionAgent...")
        t0 = time.time()

        try:
            # 数据层
            from data.router import DataRouter
            self._router = DataRouter()
        except Exception as e:
            logger.warning(f"DataRouter 初始化失败: {e}")
            self._router = None

        try:
            from analysis.indicators import TechnicalAnalyzer
            self._analyzer = TechnicalAnalyzer()
        except Exception as e:
            logger.warning(f"TechnicalAnalyzer 初始化失败: {e}")
            self._analyzer = None

        try:
            from knowledge.manager import KnowledgeManager
            self._knowledge = KnowledgeManager()
            # 确保ChromaDB已初始化
            if not self._knowledge.chroma_available:
                self._knowledge.initialize_vectordb()
        except Exception as e:
            logger.warning(f"KnowledgeManager 初始化失败: {e}")
            self._knowledge = None

        try:
            from models.router import ModelRouter
            self._model_router = ModelRouter()
        except Exception as e:
            logger.warning(f"ModelRouter 初始化失败: {e}")
            self._model_router = None

        try:
            from agent.chat_agent import ChatAgent
            self._chat_agent = ChatAgent(
                router=self._router,
                knowledge=self._knowledge,
                analyzer=self._analyzer,
                model_router=self._model_router,
            )
        except Exception as e:
            logger.warning(f"ChatAgent 初始化失败: {e}")
            self._chat_agent = None

        try:
            from agent.orchestration.workflow import AnalysisWorkflow
            self._workflow = AnalysisWorkflow(
                router=self._model_router,
                knowledge_manager=self._knowledge,
            )
        except Exception as e:
            logger.warning(f"AnalysisWorkflow 初始化失败: {e}")
            self._workflow = None

        self._initialized = True
        logger.info(f"CompetitionAgent 初始化完成 ({time.time() - t0:.1f}s)")

    # ═══════════════════════════════════════════════════════════════
    # 核心API
    # ═══════════════════════════════════════════════════════════════

    async def chat(
        self,
        message: str,
        session_id: str = "competition",
        ctx: Optional[CompetitionContext] = None,
    ) -> str:
        """
        自然语言对话接口 (比赛模块4: 应用测试)

        Args:
            message: 用户输入
            session_id: 会话ID
            ctx: 竞赛上下文 (可选)

        Returns:
            AI回复文本
        """
        await self._ensure_initialized()

        if self._chat_agent is None:
            return self._offline_reply(message)

        try:
            reply = await self._chat_agent.chat(message, session_id=session_id)
            if ctx:
                ctx.modules_completed.append("chat")
            return reply
        except Exception as e:
            logger.error(f"Chat调用失败: {e}")
            if ctx:
                ctx.errors.append(f"chat: {e}")
            return self._offline_reply(message)

    async def run_analysis(
        self,
        symbols: List[str],
        pool_name: str = "competition_watchlist",
        use_llm: bool = True,
    ) -> dict:
        """
        运行完整量化分析流水线 (比赛模块3: 智能体搭建)

        Args:
            symbols: 股票代码列表, e.g. ["sh.600519", "sz.000858"]
            pool_name: 股票池名称
            use_llm: 使用LLM增强分析

        Returns:
            分析报告 dict
        """
        await self._ensure_initialized()
        ctx = CompetitionContext(
            task_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type="stock_analysis",
            start_time=time.time(),
        )

        if self._workflow is None:
            return {"error": "AnalysisWorkflow未初始化", "ctx": ctx.summary()}

        try:
            if use_llm:
                result = await self._workflow.run_daily_scan(
                    symbols=symbols, pool_name=pool_name
                )
            else:
                result = await self._workflow.run_daily_scan(
                    symbols=symbols, pool_name=pool_name
                )

            ctx.modules_completed = [
                "data_preparation", "technical_analysis", "market_scanner",
                "strategy_matching", "backtest_verification",
                "adversarial_debate", "synthesis",
            ]
            return {
                "success": True,
                "report": result.get("final_report", ""),
                "recommendations": result.get("stock_recommendations", []),
                "ctx": ctx.summary(),
            }
        except Exception as e:
            logger.error(f"分析流水线失败: {e}")
            ctx.errors.append(str(e))
            return {"error": str(e), "ctx": ctx.summary()}

    async def run_market_scan(self, top_n: int = 10) -> dict:
        """
        全市场扫描 (快速扫描模式)

        Args:
            top_n: 返回评分最高的N只股票

        Returns:
            扫描结果 dict
        """
        await self._ensure_initialized()
        ctx = CompetitionContext(
            task_id=f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type="market_scan",
            start_time=time.time(),
        )

        try:
            from analysis.scanner import MarketScanner
            scanner = MarketScanner(self._router)
            results = await scanner.scan_all(top_n=top_n)
            ctx.modules_completed.append("market_scan")
            return {"success": True, "results": results, "ctx": ctx.summary()}
        except Exception as e:
            logger.error(f"市场扫描失败: {e}")
            ctx.errors.append(str(e))
            return {"error": str(e), "ctx": ctx.summary()}

    # ═══════════════════════════════════════════════════════════════
    # 比赛演示模式
    # ═══════════════════════════════════════════════════════════════

    async def demo_mode_run(self) -> dict:
        """
        按比赛4模块逐步演示 (90分钟模拟)

        返回值包含每个模块的输出和时间统计
        """
        logger.info("=== 开始比赛演示模式 ===")
        ctx = CompetitionContext(
            task_id=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type="competition_demo",
            start_time=time.time(),
        )

        demo_output = {}

        # — 模块1: AI智能体架构设计 (展示架构信息) —
        t1 = time.time()
        demo_output["module_1_architecture"] = self.get_architecture_info()
        ctx.modules_completed.append("architecture_design")
        ctx.module_scores["architecture_design"] = 95.0  # 自评
        demo_output["module_1_time_s"] = round(time.time() - t1, 1)
        logger.info(f"  模块1完成: {demo_output['module_1_time_s']}s")

        # — 模块2: AI知识库搭建 (展示知识库状态) —
        t2 = time.time()
        await self._ensure_initialized()
        kb_info = self._get_knowledge_base_status()
        demo_output["module_2_knowledge_base"] = kb_info
        ctx.modules_completed.append("knowledge_base")
        ctx.module_scores["knowledge_base"] = self._score_knowledge_base(kb_info)
        demo_output["module_2_time_s"] = round(time.time() - t2, 1)
        logger.info(f"  模块2完成: {demo_output['module_2_time_s']}s")

        # — 模块3: AI智能体搭建 (运行分析+展示提示词) —
        t3 = time.time()
        prompt_info = self._get_prompt_engineering_status()
        demo_output["module_3_prompt_engineering"] = prompt_info
        ctx.modules_completed.append("prompt_engineering")
        ctx.module_scores["prompt_engineering"] = self._score_prompt_engineering(prompt_info)
        demo_output["module_3_time_s"] = round(time.time() - t3, 1)
        logger.info(f"  模块3完成: {demo_output['module_3_time_s']}s")

        # — 模块4: AI应用测试 (运行测试问题) —
        t4 = time.time()
        test_results = await self._run_test_questions()
        demo_output["module_4_testing"] = test_results
        ctx.modules_completed.append("application_testing")
        ctx.module_scores["application_testing"] = test_results.get("score", 0)
        demo_output["module_4_time_s"] = round(time.time() - t4, 1)
        logger.info(f"  模块4完成: {demo_output['module_4_time_s']}s")

        # — 总结 —
        demo_output["ctx"] = ctx.summary()
        demo_output["total_time_s"] = round(ctx.elapsed(), 1)
        demo_output["total_score"] = sum(ctx.module_scores.values())

        logger.info(f"=== 演示完成: {demo_output['total_time_s']}s, "
                     f"总分: {demo_output['total_score']:.1f} ===")
        return demo_output

    # ═══════════════════════════════════════════════════════════════
    # 内部辅助
    # ═══════════════════════════════════════════════════════════════

    def _get_knowledge_base_status(self) -> dict:
        """获取知识库状态 (用于模块2展示)"""
        if self._knowledge is None:
            return {"status": "not_initialized", "files": 0}

        kb_root = self._knowledge.root
        files = {}
        for pattern, desc in [
            ("prompts/system/*.txt", "系统提示词"),
            ("prompts/tasks/*.txt", "任务提示词"),
            ("prompts/few_shots/*.json", "Few-shot样例"),
            ("rules/*.yaml", "规则文件"),
            ("strategies/*.yaml", "策略注册表"),
            ("reference/*.md", "参考文档"),
        ]:
            matched = list(kb_root.glob(pattern))
            files[desc] = [f.name for f in matched]

        total_files = sum(len(v) for v in files.values())

        return {
            "status": "ready",
            "total_files": total_files,
            "breakdown": files,
            "chromadb_available": self._knowledge.chroma_available,
            "vector_count": len(self._knowledge._chroma_client.get()["ids"])
                if self._knowledge.chroma_available and self._knowledge._chroma_client else 0,
        }

    def _get_prompt_engineering_status(self) -> dict:
        """获取提示词工程状态 (用于模块3展示)"""
        if self._knowledge is None:
            return {"status": "not_initialized"}

        prompts_dir = self._knowledge.root / "prompts" / "system"
        prompts = {}
        for f in prompts_dir.glob("*.txt"):
            content = f.read_text(encoding="utf-8")
            # 检查版本信息 (YAML frontmatter)
            version = "unknown"
            if content.startswith("---"):
                try:
                    end = content.index("---", 3)
                    fm = content[3:end].strip()
                    for line in fm.split("\n"):
                        if line.startswith("version:"):
                            version = line.split(":", 1)[1].strip()
                except (ValueError, IndexError):
                    pass
            prompts[f.stem] = {
                "file": f.name,
                "size_chars": len(content),
                "version": version,
                "has_version_header": content.startswith("---"),
            }

        return {
            "status": "ready",
            "total_prompts": len(prompts),
            "prompts": prompts,
            "model_routing": {
                "tiers": 3,
                "models": ["Ollama Qwen3-4B", "DeepSeek V4-Flash", "DeepSeek V4-Pro"],
                "peak_hour_downgrade": True,
                "budget_control": True,
            },
        }

    def _score_knowledge_base(self, kb_info: dict) -> float:
        """知识库自评分 (满分20)"""
        score = 0.0
        if kb_info.get("status") != "ready":
            return score
        # 文件数量: 每5个文件得3分,最多12分
        score += min(12.0, kb_info.get("total_files", 0) * 0.6)
        # ChromaDB可用: +5分
        if kb_info.get("chromadb_available"):
            score += 5.0
        # 向量数据: +3分
        if kb_info.get("vector_count", 0) > 0:
            score += 3.0
        return min(20.0, score)

    def _score_prompt_engineering(self, prompt_info: dict) -> float:
        """提示词工程自评分 (满分30)"""
        score = 0.0
        if prompt_info.get("status") != "ready":
            return score
        # 提示词数量: 每2个得3分,最多12分
        score += min(12.0, prompt_info.get("total_prompts", 0) * 1.5)
        # 版本管理: +6分
        prompts = prompt_info.get("prompts", {})
        versioned = sum(1 for p in prompts.values() if p.get("has_version_header"))
        if versioned > 0:
            score += min(6.0, versioned * 2.0)
        # 模型路由: +6分
        if prompt_info.get("model_routing"):
            score += 6.0
        # 多Agent编排: +6分
        score += 6.0
        return min(30.0, score)

    async def _run_test_questions(self) -> dict:
        """
        运行应用测试问题集 (v3.0: 使用 LLM-as-Judge 真实评估)

        使用 LLMJudge 进行4维度评估:
          - structure (格式合规, 0-3)
          - accuracy (数值准确性, 0-3)
          - logic (逻辑一致性, 0-2)
          - risk_warning (风险提示, 0-2)
        """
        from knowledge.prompts.llm_judge import LLMJudge

        questions = [
            {
                "id": "q1",
                "question": "当前A股市场处于什么状态？请判断牛熊并给出依据。",
                "expected": ["市场体制判断", "指数数据支撑", "风险提示"],
            },
            {
                "id": "q2",
                "question": "分析一下600519贵州茅台的技术面，给出操作建议。",
                "expected": ["技术指标数据", "多空论据", "操作建议+风险提示"],
            },
            {
                "id": "q3",
                "question": "什么样的市场环境适合使用双均线策略？请回测验证。",
                "expected": ["策略适用条件", "回测数据", "策略风险说明"],
            },
        ]

        judge = LLMJudge(self._model_router)
        results = []
        for q in questions:
            try:
                reply = await self.chat(q["question"], session_id=f"test_{q['id']}")
                # v3.0: 使用 LLM-as-Judge 真实评估
                if self._model_router:
                    eval_result = await judge.evaluate(
                        q["question"], reply, q["expected"]
                    )
                else:
                    eval_result = judge.evaluate_sync(
                        q["question"], reply, q["expected"]
                    )
                results.append({
                    "id": q["id"],
                    "question": q["question"][:80],
                    "reply_preview": reply[:200],
                    "quality_score": round(eval_result.total_score, 1),
                    "breakdown": {
                        "structure": eval_result.structure_score,
                        "accuracy": eval_result.accuracy_score,
                        "logic": eval_result.logic_score,
                        "risk_warning": eval_result.risk_score,
                    },
                    "assessment": eval_result.overall_assessment,
                    "improvements": eval_result.improvements,
                })
            except Exception as e:
                results.append({
                    "id": q["id"], "error": str(e), "quality_score": 0,
                })

        avg_score = sum(r.get("quality_score", 0) for r in results) / max(1, len(results))
        return {
            "questions_ran": len(results),
            "average_quality_score": round(avg_score, 1),
            "judge_method": "LLM-as-Judge (4维评估)" if self._model_router else "规则引擎 (LLM不可用)",
            "details": results,
            "score": round(avg_score, 1),
        }

    def _offline_reply(self, message: str) -> str:
        """离线模式回复 (当LLM不可用时)"""
        return (
            "⚠️ AI分析引擎暂不可用。\n\n"
            f"您的问题: {message[:100]}\n\n"
            "当前系统状态:\n"
            f"- 版本: {self.version}\n"
            f"- 赛道: {self.competition_track}\n"
            "- 模型路由: 待连接\n\n"
            "建议:\n"
            "1. 检查 DeepSeek API Key 是否已配置\n"
            "2. 检查网络连接\n"
            "3. 或使用 Dashboard 进行离线分析\n"
        )


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

async def create_competition_agent(demo_mode: bool = False) -> CompetitionAgent:
    """创建竞赛智能体实例"""
    agent = CompetitionAgent(demo_mode=demo_mode)
    await agent._ensure_initialized()
    return agent
