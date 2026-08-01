"""Data providers package (v3.1)"""

from .base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource
from .akshare_provider import AKShareProvider
from .baostock_provider import BaostockProvider
from .alternative import AlternativeDataProvider
from .eastmoney_provider import EastMoneyProvider  # v3.1
from .tencent_provider import TencentFinanceProvider  # v3.1
from .fundamentals import FundamentalsProvider  # v3.1

__all__ = [
    "DataSource",
    "DataFrequency",
    "DataRequest",
    "DataResult",
    "DataProvider",
    "AKShareProvider",
    "BaostockProvider",
    "AlternativeDataProvider",
    "EastMoneyProvider",
    "TencentFinanceProvider",
    "FundamentalsProvider",
]
