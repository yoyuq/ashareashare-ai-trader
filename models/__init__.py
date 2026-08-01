"""
模型层 — 单模型统一调度 (v3.0)

- ModelRouter: 全任务统一路由到 DeepSeek V4-Flash + 预算追踪
- CostMonitor: 日/月成本追踪+预算告警
"""

import os

from .cost_monitor import CostMonitor
from .router import ModelRouter, ModelTier, RouteResult

# 模型名 (经环境变量覆盖; v3.0 全量使用 deepseek-v4-flash — 2026-07 旧名 deepseek-chat/reasoner 已弃用)
DEEPSEEK_FLASH_MODEL = os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")
# 兼容别名: v2 曾有 PRO 层, v3 统一 flash (保留导出避免破坏既有 import)
DEEPSEEK_PRO_MODEL = os.getenv("DEEPSEEK_PRO_MODEL", DEEPSEEK_FLASH_MODEL)

__all__ = [
    "ModelRouter", "ModelTier", "RouteResult", "CostMonitor",
    "DEEPSEEK_FLASH_MODEL", "DEEPSEEK_PRO_MODEL",
]
