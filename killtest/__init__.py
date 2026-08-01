"""kill-test — 三路信号对比 (LLM / 规则 / 随机) 的前向收益验证框架 (v3.1-deerflow)"""

from killtest.arms import RuleArm, RandomArm, LLMArm, Signal
from killtest.tracker import KillTestTracker, Outcome
from killtest.report import summarize, compare, render
from killtest.runner import run_killtest, DEFAULT_UNIVERSE

__all__ = [
    "RuleArm", "RandomArm", "LLMArm", "Signal",
    "KillTestTracker", "Outcome",
    "summarize", "compare", "render", "run_killtest", "DEFAULT_UNIVERSE",
]
