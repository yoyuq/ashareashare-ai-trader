"""
MCP Server — 将 A股智能分析Agent 的工具暴露给外部 AI IDE (v3.1)

启动:
    python -m api.mcp_server

配置 (添加到 Claude Code / Cursor 的 mcp.json):
    {
      "mcpServers": {
        "ashare-ai-trader": {
          "command": "python",
          "args": ["-m", "api.mcp_server"],
          "cwd": "<project_root>"
        }
      }
    }

暴露工具:
  - get_market_overview: 市场总览 (状态/指数/北向资金)
  - analyze_stock: 个股深度分析
  - run_backtest: 策略历史回测
  - scan_market: 全市场扫描
  - list_strategies: 策略列表
  - get_win_rate: 策略胜率查询
  - explain_concept: A股概念术语解释
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# 导入核心组件
from agent.chat_agent import TOOLS, ToolExecutor
from knowledge.manager import KnowledgeManager

# ── MCP Server ──
try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationCapabilities
    from mcp.server.stdio import stdio_server
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    logger.warning("mcp SDK 未安装, pip install mcp 以启用 MCP Server")


# ── 工具执行器能力 ──
# 只暴露纯数据查询的工具,不暴露会触发交易的
SAFE_TOOLS = [
    "get_market_overview",
    "analyze_stock",
    "scan_market",
    "list_strategies",
    "explain_concept",
    "get_win_rate",
    "run_backtest",     # 回测仅分析历史,不涉及实盘
]

# 需要运行时初始化的组件
_executor = None
_knowledge = None


def _get_executor() -> ToolExecutor:
    global _executor
    if _executor is None:
        try:
            from data.router import get_data_router
            data_router = get_data_router()
        except Exception:
            data_router = None

        try:
            from analysis.indicators import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer()
        except Exception:
            analyzer = None

        _executor = ToolExecutor(
            router=data_router,
            knowledge=_get_knowledge(),
            analyzer=analyzer,
        )
    return _executor


def _get_knowledge() -> KnowledgeManager:
    global _knowledge
    if _knowledge is None:
        _knowledge = KnowledgeManager()
    return _knowledge


def _filter_tools() -> list:
    """过滤出安全的工具列表"""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "inputSchema": {
                "type": "object",
                "properties": t["function"]["parameters"].get("properties", {}),
                "required": t["function"]["parameters"].get("required", []),
            },
        }
        for t in TOOLS
        if t["function"]["name"] in SAFE_TOOLS
    ]


# ── MCP Server 实现 ──
if HAS_MCP:
    mcp_server = Server("ashare-ai-trader")

    @mcp_server.list_tools()
    async def handle_list_tools():
        """列出所有可用的 MCP 工具"""
        return _filter_tools()

    @mcp_server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list:
        """
        执行 MCP 工具调用

        将 MCP Tool 调用路由到 ChatAgent 的 ToolExecutor,
        返回文本内容列表。
        """
        if name not in SAFE_TOOLS:
            return [{"type": "text", "text": f"Error: tool '{name}' is not available"}]

        try:
            executor = _get_executor()
            result = await executor.execute(name, arguments)
            return [{"type": "text", "text": str(result)}]
        except Exception as e:
            logger.error(f"MCP tool '{name}' failed: {e}")
            return [{"type": "text", "text": f"Error executing '{name}': {e}"}]

    @mcp_server.list_resources()
    async def handle_list_resources():
        """列出知识资源 (策略/规则/参考文档)"""
        km = _get_knowledge()
        strategies = km.list_strategies()
        prompts = km.list_all_prompts()

        resources = []
        for s in strategies:
            resources.append({
                "uri": f"strategy://{s['id']}",
                "name": s.get("name", s["id"]),
                "mimeType": "application/json",
                "description": s.get("description", ""),
            })
        for p in prompts:
            resources.append({
                "uri": f"prompt://{p['name']}",
                "name": f"System Prompt: {p['name']}",
                "mimeType": "text/plain",
                "description": f"v{p.get('version', '?')} {p.get('date', '')}",
            })
        return resources

    @mcp_server.read_resource()
    async def handle_read_resource(uri: str):
        """读取知识资源内容"""
        km = _get_knowledge()
        if uri.startswith("strategy://"):
            strategy_id = uri.replace("strategy://", "")
            strategy = km.get_strategy(strategy_id)
            if strategy:
                return [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(strategy, ensure_ascii=False, indent=2),
                }]
        elif uri.startswith("prompt://"):
            prompt_name = uri.replace("prompt://", "")
            prompt = km.get_system_prompt(prompt_name)
            if prompt and "Prompt file missing" not in prompt:
                return [{
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": prompt,
                }]
        raise ValueError(f"Resource not found: {uri}")

    @mcp_server.list_prompts()
    async def handle_list_prompts():
        """列出可用的 MCP 提示词模板"""
        return [
            {
                "name": "analyze-stock",
                "description": "Analyze a single A-share stock",
                "arguments": [
                    {"name": "symbol", "description": "Stock code, e.g. 600519", "required": True},
                ],
            },
            {
                "name": "market-overview",
                "description": "Get current A-share market overview",
                "arguments": [],
            },
            {
                "name": "scan-opportunities",
                "description": "Scan market for trading opportunities",
                "arguments": [
                    {"name": "top_n", "description": "Number of top stocks to return", "required": False},
                ],
            },
        ]

    @mcp_server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict):
        """生成提示词模板"""
        if name == "analyze-stock":
            symbol = arguments.get("symbol", "600519")
            return [{
                "role": "user",
                "content": f"Please perform a comprehensive technical analysis of A-share stock {symbol}. "
                          f"Include: 1) Current indicators (RSI/MACD/MA/Bollinger) "
                          f"2) Pattern detection 3) Market regime fit 4) Risk assessment 5) Trading recommendation.",
            }]
        elif name == "market-overview":
            return [{
                "role": "user",
                "content": "Analyze the current A-share market status including: "
                          "1) Market regime (current state detection) "
                          "2) Major index performance 3) North-bound capital flow "
                          "4) Sector rotation 5) Risk alerts.",
            }]
        elif name == "scan-opportunities":
            top_n = arguments.get("top_n", 10)
            return [{
                "role": "user",
                "content": f"Scan the A-share market for the top {top_n} trading opportunities. "
                          "Use the composite scoring system (technical 40%, fund flow 25%, momentum 20%, quality 15%).",
            }]
        raise ValueError(f"Prompt not found: {name}")


# ── 主入口 ──
async def main():
    if not HAS_MCP:
        logger.error("MCP SDK not installed. Run: pip install mcp")
        sys.exit(1)

    logger.info("A股智能分析Agent MCP Server starting...")
    logger.info(f"Exposing {len(SAFE_TOOLS)} tools: {SAFE_TOOLS}")

    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            InitializationCapabilities(
                sampling=None,
                experimental=None,
            ),
            NotificationOptions(),
            raise_exceptions=False,
        )


if __name__ == "__main__":
    asyncio.run(main())
