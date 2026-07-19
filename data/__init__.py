"""
数据基础层 (Data Layer)

核心组件:
- DataRouter: 多源数据路由+自动降级+交叉验证
- AKShareProvider: 主数据源(行情/财务/板块)
- BaostockProvider: 备用数据源
- AlternativeDataProvider: 另类数据(v2.1)
- CacheLayer: Redis缓存
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
from .providers.alternative import AlternativeDataProvider
from .cache import CacheLayer, get_cache
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
    "AlternativeDataProvider",
    # Cache
    "CacheLayer",
    "get_cache",
    # Processing
    "PITProcessor",
]
