"""
BaseAgent — 子Agent基类 (v3.0-competition)

所有子Agent的抽象基类,定义统一的接口:
  - run(): 执行Agent逻辑
  - system_prompt(): 加载系统提示词
  - memory: Agent间共享记忆
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentContext:
    """Agent执行上下文"""
    agent_name: str
    task_id: str = ""
    start_time: float = 0.0
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    model_trace: List[Dict] = field(default_factory=list)

    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000 if self.start_time else 0


@dataclass
class AgentResult:
    """Agent执行结果"""
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0
    model_used: str = ""


class BaseAgent(ABC):
    """
    子Agent抽象基类

    每个子Agent必须实现:
      - agent_name: Agent名称
      - default_system_prompt: 默认系统提示词
      - run(): 执行Agent逻辑

    可选实现:
      - load_prompt(): 从知识库加载提示词
      - pre_process(): 输入预处理
      - post_process(): 输出后处理
    """

    agent_name: str = "base"
    agent_icon: str = "🤖"
    default_system_prompt: str = ""

    def __init__(self, knowledge_manager=None, model_router=None):
        """
        Args:
            knowledge_manager: KnowledgeManager 实例
            model_router: ModelRouter 实例
        """
        self.knowledge = knowledge_manager
        self.router = model_router
        self._context: Optional[AgentContext] = None

    @abstractmethod
    async def run(self, ctx: AgentContext) -> AgentResult:
        """
        执行Agent逻辑

        Args:
            ctx: Agent执行上下文 (含输入数据)

        Returns:
            AgentResult: Agent执行结果
        """
        ...

    def load_prompt(self) -> str:
        """
        加载系统提示词

        优先从知识库加载,fallback到默认值
        """
        if self.knowledge:
            prompt = self.knowledge.get_system_prompt(self.agent_name)
            if prompt and "Prompt文件缺失" not in prompt:
                # 去除YAML frontmatter
                if prompt.startswith("---"):
                    try:
                        end = prompt.index("---", 3)
                        prompt = prompt[end + 3:].strip()
                    except ValueError:
                        pass
                return prompt
        return self.default_system_prompt

    def _start_context(self, task_id: str = "", **inputs) -> AgentContext:
        """创建并启动执行上下文"""
        self._context = AgentContext(
            agent_name=self.agent_name,
            task_id=task_id,
            start_time=time.time(),
            input_data=inputs,
        )
        return self._context

    def _finish_context(self, success: bool = True, error: str = "", **outputs) -> AgentResult:
        """完成执行上下文,返回结果"""
        if self._context is None:
            self._context = AgentContext(agent_name=self.agent_name)
        self._context.output_data = outputs
        if error:
            self._context.errors.append(error)
        return AgentResult(
            success=success,
            data=outputs,
            error=error,
            latency_ms=self._context.elapsed_ms(),
        )
