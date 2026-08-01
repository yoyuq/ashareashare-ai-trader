"""
ConfigurableAgentRegistry — 可配置Agent注册表 (v3.1)

替代 agent/sub_agents/__init__.py 中硬编码的 AGENT_REGISTRY dict。
从 config/agents.yaml 动态加载 Agent 定义,支持:
  - YAML 声明式定义 (类路径/模型/提示词/工具集)
  - 动态创建 Agent 实例
  - 热重载配置
  - Agent 列表查询

用法:
    registry = ConfigurableAgentRegistry()
    registry.load_config()              # 从 config/agents.yaml 加载
    agent = registry.create_agent("technical_analyst",
                                   knowledge_manager=km,
                                   model_router=mr)
    agents = registry.list_agents()     # 列出所有已注册Agent
"""

import importlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger


class AgentConfig:
    """单个 Agent 配置"""
    def __init__(self, name: str, config: dict):
        self.name = name
        self.display_name = config.get("name", name)
        self.class_path = config.get("class", "")
        self.icon = config.get("icon", "")
        self.model = config.get("model", "flash")
        self.system_prompt = config.get("system_prompt", f"{name}.txt")
        self.tools = config.get("tools", [])
        self.description = config.get("description", "")
        self.parallel_allowed = config.get("parallel_allowed", True)
        self.timeout_seconds = config.get("timeout_seconds", 120)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "icon": self.icon,
            "model": self.model,
            "tools": self.tools,
            "description": self.description,
            "parallel_allowed": self.parallel_allowed,
        }


class ConfigurableAgentRegistry:
    """
    可配置Agent注册表 — 从 YAML 加载,支持热重载

    单例模式,全局共享。
    """

    _instance: Optional["ConfigurableAgentRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._configs: Dict[str, AgentConfig] = {}
                    cls._instance._loaded = False
        return cls._instance

    def load_config(self, config_path: str = None):
        """
        从 YAML 文件加载 Agent 配置

        Args:
            config_path: YAML 文件路径,默认 config/agents.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            agents_data = data.get("agents", {})
            self._configs.clear()

            for agent_name, agent_cfg in agents_data.items():
                self._configs[agent_name] = AgentConfig(agent_name, agent_cfg)

            self._loaded = True
            logger.info(f"[AgentRegistry] 已从 {config_path} 加载 {len(self._configs)} 个Agent")
            return len(self._configs)

        except Exception as e:
            logger.error(f"[AgentRegistry] 配置加载失败: {e}")
            return 0

    def reload(self):
        """热重载配置"""
        self._loaded = False
        return self.load_config()

    def create_agent(
        self,
        name: str,
        knowledge_manager=None,
        model_router=None,
    ):
        """
        从配置创建 Agent 实例

        Args:
            name: Agent 名称
            knowledge_manager: KnowledgeManager 实例
            model_router: ModelRouter 实例

        Returns:
            BaseAgent 子类实例

        Raises:
            ValueError: Agent 未注册
            ImportError: 类路径无效
        """
        if not self._loaded:
            self.load_config()

        config = self._configs.get(name)
        if config is None:
            raise ValueError(
                f"未知Agent: '{name}'。可用: {list(self._configs.keys())}"
            )

        # 动态导入类
        class_path = config.class_path
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            agent_cls = getattr(module, class_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(f"无法加载 '{class_path}': {e}")

        # 实例化 (所有 BaseAgent 子类接受 knowledge_manager + model_router)
        instance = agent_cls(
            knowledge_manager=knowledge_manager,
            model_router=model_router,
        )

        # 注入配置覆盖
        instance.agent_name = name
        if config.icon:
            instance.agent_icon = config.icon

        logger.debug(f"[AgentRegistry] 已创建 Agent: {name} ({config.display_name})")
        return instance

    def get_config(self, name: str) -> Optional[AgentConfig]:
        """获取单个 Agent 配置"""
        if not self._loaded:
            self.load_config()
        return self._configs.get(name)

    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有已注册的 Agent"""
        if not self._loaded:
            self.load_config()
        return [
            {
                "name": name,
                "display_name": cfg.display_name,
                "icon": cfg.icon,
                "model": cfg.model,
                "tools": cfg.tools,
                "description": cfg.description,
            }
            for name, cfg in self._configs.items()
        ]

    def get_available_names(self) -> List[str]:
        """获取所有可用 Agent 名称"""
        if not self._loaded:
            self.load_config()
        return list(self._configs.keys())

    def is_registered(self, name: str) -> bool:
        """检查 Agent 是否已注册"""
        if not self._loaded:
            self.load_config()
        return name in self._configs
