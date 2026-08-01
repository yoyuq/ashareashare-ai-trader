"""
Tool Registry — 装饰器模式工具注册系统 (v3.0-competition)

使用 @tool 装饰器自动注册工具,替代硬编码的 if/elif 链:

    from agent.tools.registry import tool, ToolRegistry

    @tool(
        name="analyze_stock",
        description="深度分析指定股票",
        parameters={"symbol": {"type": "string", "description": "股票代码"}},
        required=["symbol"],
    )
    async def analyze_stock(symbol: str, executor: "ToolExecutor") -> str:
        '''分析股票的技术面和资金面'''
        ...

特点:
  - 自动生成 OpenAI Function Calling schema
  - 字典分发替代 if/elif 链
  - 支持动态添加/移除工具
  - 向后兼容现有的 TOOLS 列表
"""

from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class ToolRegistry:
    """
    工具注册中心 — 单例模式

    用法:
        registry = ToolRegistry()

        @registry.register(
            name="my_tool",
            description="我的工具",
            parameters={"param1": {"type": "string", "description": "参数1"}},
            required=["param1"],
        )
        async def my_tool(param1: str, executor=None) -> str:
            return f"结果: {param1}"
    """

    _instance: Optional["ToolRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, Callable] = {}
            cls._instance._schemas: Dict[str, dict] = {}
            cls._instance._tool_list: List[dict] = []
        return cls._instance

    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None,
    ):
        """
        工具注册装饰器

        Args:
            name: 工具名称 (用于 function calling)
            description: 工具描述
            parameters: 参数schema {"param": {"type": "string", "description": "..."}}
            required: 必填参数列表

        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            # 构建 OpenAI Function Calling Schema
            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters or {},
                        "required": required or [],
                    },
                },
            }
            self._tools[name] = func
            self._schemas[name] = schema

            # 更新工具列表 (移除旧版同名工具)
            self._tool_list = [s for s in self._tool_list
                             if s["function"]["name"] != name]
            self._tool_list.append(schema)

            return func
        return decorator

    def get_tool(self, name: str) -> Optional[Callable]:
        """获取工具函数"""
        return self._tools.get(name)

    def get_schema(self, name: str) -> Optional[dict]:
        """获取工具的OpenAI Schema"""
        return self._schemas.get(name)

    def get_all_schemas(self) -> List[dict]:
        """获取全部工具的OpenAI Schema列表"""
        return list(self._tool_list)

    def get_all_names(self) -> List[str]:
        """获取全部工具名称"""
        return list(self._tools.keys())

    def list_tools(self) -> List[Dict[str, str]]:
        """列出所有注册的工具 (名称+描述)"""
        return [
            {"name": name, "description": schema["function"]["description"]}
            for name, schema in self._schemas.items()
        ]

    def is_registered(self, name: str) -> bool:
        """检查工具是否已注册"""
        return name in self._tools

    def unregister(self, name: str) -> bool:
        """移除工具"""
        if name not in self._tools:
            return False
        del self._tools[name]
        del self._schemas[name]
        self._tool_list = [s for s in self._tool_list
                         if s["function"]["name"] != name]
        return True

    def to_mcp_format(self) -> List[dict]:
        """
        v3.1: 将注册工具转为 MCP (Model Context Protocol) Tool spec

        返回符合 MCP Tool 规范的 schema 列表,
        可直接用于 MCP Server 的 list_tools() 响应。
        """
        mcp_tools = []
        for name, schema in self._schemas.items():
            func = schema["function"]
            mcp_tools.append({
                "name": func["name"],
                "description": func["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": func["parameters"].get("properties", {}),
                    "required": func["parameters"].get("required", []),
                },
            })
        return mcp_tools

    def from_config(self, tools_config: List[dict], executor_instance=None):
        """
        v3.1: 从配置字典批量注册工具

        支持从 YAML/JSON 配置动态注册,配合 ConfigurableAgentRegistry 使用。

        Args:
            tools_config: [{"name": "...", "description": "...", "parameters": {...}}, ...]
            executor_instance: ToolExecutor 实例,用于执行工具
        """
        registered = 0
        for tc in tools_config:
            name = tc.get("name")
            if not name or name in self._tools:
                continue

            async def _dynamic_tool(executor=None, _name=name, **kwargs):
                if executor and hasattr(executor, 'execute_by_name'):
                    return await executor.execute_by_name(_name, **kwargs)
                return f"Tool '{_name}': no executor configured"

            _dynamic_tool.__name__ = name
            self.register(
                name=name,
                description=tc.get("description", ""),
                parameters=tc.get("parameters", {}),
                required=tc.get("required", []),
            )(_dynamic_tool)
            registered += 1

        logger.info(f"[ToolRegistry] from_config: {registered} tools registered")
        return registered

    def clear(self):
        """清空所有注册"""
        self._tools.clear()
        self._schemas.clear()
        self._tool_list.clear()


# ── 模块级便捷函数 ──

def tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
):
    """
    @tool 装饰器 — 注册工具到全局 ToolRegistry

    用法:
        from agent.tools.registry import tool

        @tool(name="my_tool", description="...", parameters={...})
        async def my_tool(executor=None, **kwargs):
            ...
    """
    return ToolRegistry().register(name, description, parameters, required)


def get_tool_registry() -> ToolRegistry:
    """获取全局 ToolRegistry 实例"""
    return ToolRegistry()


# ── 向后兼容: 从现有 TOOLS 列表导入 ──

def import_from_dict_list(tool_list: List[dict], executor_instance=None):
    """
    从现有的 TOOLS 字典列表批量注册到 ToolRegistry

    用法:
        from agent.chat_agent import TOOLS
        from agent.tools.registry import import_from_dict_list
        import_from_dict_list(TOOLS)
    """
    registry = ToolRegistry()
    for t in tool_list:
        name = t["function"]["name"]
        desc = t["function"]["description"]
        params = t["function"]["parameters"].get("properties", {})
        req = t["function"]["parameters"].get("required", [])

        # 创建一个通用的异步处理函数
        async def _generic_tool(executor=None, _name=name, **kwargs):
            if executor and hasattr(executor, 'execute_by_name'):
                return await executor.execute_by_name(_name, **kwargs)
            return f"工具 {_name} 未配置执行器"

        _generic_tool.__name__ = name
        registry.register(name=name, description=desc,
                        parameters=params, required=req)(_generic_tool)
