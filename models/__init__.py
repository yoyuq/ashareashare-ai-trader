"""
模型层 — 三层漏斗混合调度 (v2.0+)

- ModelRouter: 任务复杂度评估→自动路由→高峰降级→预算控制
- CostMonitor: 日/月成本追踪+预算告警
"""

import os

from .cost_monitor import CostMonitor
from .router import ModelRouter, ModelTier, RouteResult

# 模型名 (经环境变量覆盖; 默认使用 DeepSeek V4 规范名 — 2026-07 旧名 deepseek-chat/reasoner 已弃用)
DEEPSEEK_FLASH_MODEL = os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")
DEEPSEEK_PRO_MODEL = os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro")

__all__ = [
    "ModelRouter", "ModelTier", "RouteResult", "CostMonitor",
    "DEEPSEEK_FLASH_MODEL", "DEEPSEEK_PRO_MODEL",
]
