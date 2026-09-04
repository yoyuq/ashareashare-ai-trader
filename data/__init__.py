"""
数据基础层 (Data Layer)

核心组件:
- DataRouter: 多源数据路由+自动降级+交叉验证
- AKShareProvider: 主数据源(行情/财务/板块)
- BaostockProvider: 备用数据源
- PITProcessor: Point-in-Time处理
"""

from .router import DataRouter, get_data_router
from .providers.base import (
    DataFrequency,
    DataProvider,
    DataRequest,
    DataResult,
    DataSource,
)
from .providers.akshare_provider import AKShareProvider
from .providers.baostock_provider import BaostockProvider
from .processors.pit import PITProcessor

__all__ = [
    # Router
    "DataRouter",
    "get_data_router",
    # Base
    "DataSource",
    "DataFrequency",
    "DataRequest",
    "DataResult",
    "DataProvider",
    # Providers
    "AKShareProvider",
    "BaostockProvider",
    # Processing
    "PITProcessor",
]
