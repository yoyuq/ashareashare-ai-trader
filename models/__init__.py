"""
模型层 — 单模型统一调度 (v3.0)

- ModelRouter: 全任务统一路由到 DeepSeek V4-Flash + 预算追踪
- CostMonitor: 日/月成本追踪+预算告警
"""

import os

from .cost_monitor import CostMonitor
from .router import ModelRouter, RouteResult

# 模型名 (经环境变量覆盖; v3.0 全量使用 deepseek-v4-flash — 2026-07 旧名 deepseek-chat/reasoner 已弃用)
DEEPSEEK_FLASH_MODEL = os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")

__all__ = [
    "ModelRouter", "RouteResult", "CostMonitor",
    "DEEPSEEK_FLASH_MODEL",
]
