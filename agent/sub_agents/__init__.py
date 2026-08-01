"""
AI+金融智能体 — 子Agent模块 (v3.1-deerflow)

注册的子Agent (均有独立类+run()方法):
  - TechnicalAnalystAgent: 技术面分析 (6维度)
  - MarketScannerAgent:    市场扫描 (4维度)
  - SynthesisAgent:        综合研判
  - RiskAssessorAgent:     风控评估 (8层)
  - CriticAgent:           策略审计 (5维)
  - EvaluatorAgent:        报告质量评估 (驱动反思回炉)
  - DecisionValidator:     交易计划硬约束校验

v3.1-deerflow: bull_researcher/bear_researcher/judge 模块类层为死代码 (仅 __init__
再导出, 无任何调用方; 多空辩论由 workflow 内联 + knowledge/prompts 驱动), 已删除。

基类:
  - BaseAgent:   抽象基类 (统一接口)
  - AgentContext: 执行上下文
  - AgentResult:  执行结果
"""

# ── 基类 ──
from agent.sub_agents.base import BaseAgent, AgentContext, AgentResult  # noqa: F401

# ── 分析类Agent ──
from agent.sub_agents.technical_analyst import TechnicalAnalystAgent  # noqa: F401
from agent.sub_agents.market_scanner import MarketScannerAgent  # noqa: F401

# ── 综合类Agent ──
from agent.sub_agents.synthesis import SynthesisAgent  # noqa: F401
from agent.sub_agents.risk_assessor import RiskAssessorAgent  # noqa: F401
from agent.sub_agents.critic import CriticAgent, CriticResult, FlawReport  # noqa: F401

# ── v3.1-deerflow: 反射循环 + 决策验证 ──
from agent.sub_agents.evaluator import EvaluatorAgent, EvaluationResult, EvaluationIssue  # noqa: F401
from agent.sub_agents.validator import (  # noqa: F401
    DecisionValidator, ValidationResult, ValidationViolation, limit_pct_for_symbol,
)

# ── Agent注册表 (v3.1: 优先使用可配置注册表,fallback到硬编码) ──
# 保留硬编码 AGENT_REGISTRY 作为 fallback
AGENT_REGISTRY = {
    "technical_analyst": TechnicalAnalystAgent,
    "market_scanner": MarketScannerAgent,
    "synthesis": SynthesisAgent,
    "risk_assessor": RiskAssessorAgent,
    "critic": CriticAgent,
    "evaluator": EvaluatorAgent,
    "validator": DecisionValidator,
}

# v3.1: 可配置Agent注册表 (从 config/agents.yaml 加载)
_config_registry = None


def _get_config_registry():
    """懒加载可配置注册表"""
    global _config_registry
    if _config_registry is None:
        try:
            from agent.config.agent_registry import ConfigurableAgentRegistry
            _config_registry = ConfigurableAgentRegistry()
            _config_registry.load_config()
        except Exception:
            _config_registry = None
    return _config_registry


def create_agent(name: str, knowledge_manager=None, model_router=None) -> BaseAgent:
    """
    工厂函数 — 按名称创建Agent实例 (v3.1)

    优先使用可配置注册表 (ConfigurableAgentRegistry),
    fallback到硬编码的 AGENT_REGISTRY。

    用法:
        agent = create_agent("technical_analyst",
                            knowledge_manager=km,
                            model_router=mr)
    """
    # 优先: 可配置注册表
    registry = _get_config_registry()
    if registry is not None and registry.is_registered(name):
        return registry.create_agent(
            name,
            knowledge_manager=knowledge_manager,
            model_router=model_router,
        )

    # Fallback: 硬编码注册表
    agent_cls = AGENT_REGISTRY.get(name)
    if agent_cls is None:
        available = (
            registry.get_available_names()
            if registry else list(AGENT_REGISTRY.keys())
        )
        raise ValueError(f"未知Agent: '{name}'. 可用: {available}")
    return agent_cls(knowledge_manager=knowledge_manager, model_router=model_router)


def list_available_agents() -> list:
    """列出所有可用的 Agent (v3.1)"""
    registry = _get_config_registry()
    if registry is not None:
        return registry.list_agents()
    return [{"name": k, "display_name": k} for k in AGENT_REGISTRY.keys()]
