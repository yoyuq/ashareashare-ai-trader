"""API 组件工厂 (v6.0 拆分自 server.py)

延迟初始化的组件单例与工厂函数; 任务/会话内存态也在此。
"""
from typing import Any, Dict, List

_router = None
_knowledge = None
_analyzer = None
_detector = None
_chat_agent = None
_tasks: Dict[str, Dict] = {}
_sessions: Dict[str, List[Dict]] = {}  # v2.8: 会话存储


def get_router():
    global _router
    if _router is None:
        from data.router import get_data_router
        _router = get_data_router()
    return _router


def get_knowledge():
    global _knowledge
    if _knowledge is None:
        from knowledge.manager import KnowledgeManager
        _knowledge = KnowledgeManager()
    return _knowledge


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from analysis.indicators import TechnicalAnalyzer
        _analyzer = TechnicalAnalyzer()
    return _analyzer


def get_detector():
    global _detector
    if _detector is None:
        from analysis.regime import MarketRegimeDetector
        _detector = MarketRegimeDetector()
    return _detector


def get_chat_agent():
    """ChatAgent 懒加载 (v2.9: 使用 ModelRouter)"""
    global _chat_agent
    if _chat_agent is None:
        try:
            from models.router import ModelRouter
            model_router = ModelRouter()
        except Exception:
            model_router = None
        from agent.chat_agent import ChatAgent
        _chat_agent = ChatAgent(
            router=get_router(),
            knowledge=get_knowledge(),
            analyzer=get_analyzer(),
            model_router=model_router,
        )
    return _chat_agent


def get_tasks() -> Dict[str, Dict]:
    """分析任务表 (内存态)"""
    return _tasks


def get_sessions() -> Dict[str, List[Any]]:
    """会话存储 (内存态, v2.8)"""
    return _sessions
