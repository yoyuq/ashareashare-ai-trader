"""Data providers package (v3.1)"""

from .base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource
from .akshare_provider import AKShareProvider
from .baostock_provider import BaostockProvider
from .eastmoney_provider import EastMoneyProvider  # v3.1
from .tencent_provider import TencentFinanceProvider  # v3.1
from .fundamentals import FundamentalsProvider  # v3.1
from .tushare_provider import TushareProvider  # v3.0: 主源 (可靠性最高, 需 TUSHARE_TOKEN)

__all__ = [
    "DataSource",
    "DataFrequency",
    "DataRequest",
    "DataResult",
    "DataProvider",
    "AKShareProvider",
    "BaostockProvider",
    "EastMoneyProvider",
    "TencentFinanceProvider",
    "FundamentalsProvider",
    "TushareProvider",
]
