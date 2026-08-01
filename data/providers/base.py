"""
数据提供器抽象基类

所有数据源(原生/AKShare/Baostock/Tushare/另类数据)统一实现此接口,
确保上层不依赖于具体数据源,方便多源路由和降级切换。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class DataSource(Enum):
    """数据源枚举"""
    AKSHARE = "akshare"
    BAOSTOCK = "baostock"
    TUSHARE = "tushare"
    EFINANCE = "efinance"
    EASTMONEY = "eastmoney"
    TENCENT = "tencent"
    ALTERNATIVE = "alternative"


class DataFrequency(Enum):
    """数据频率"""
    TICK = "tick"
    MIN1 = "1min"
    MIN5 = "5min"
    MIN15 = "15min"
    MIN30 = "30min"
    MIN60 = "60min"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class DataRequest:
    """统一数据请求结构"""
    symbol: str                          # 股票代码 (sh.600000 / sz.000001)
    start_date: date
    end_date: date
    frequency: DataFrequency = DataFrequency.DAILY
    fields: Optional[List[str]] = None   # None = 全部字段
    adjust: str = "qfq"                  # 复权方式: qfq/hfq/None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def recent(cls, symbol: str, days: int = 365,
               frequency: DataFrequency = DataFrequency.DAILY,
               adjust: str = "qfq") -> "DataRequest":
        """工厂方法: 获取最近N天的数据请求 (消除全项目重复的date.today()-timedelta模式)"""
        from datetime import date, timedelta
        today = date.today()
        return cls(
            symbol=symbol,
            start_date=today - timedelta(days=days),
            end_date=today,
            frequency=frequency,
            adjust=adjust,
        )


@dataclass
class DataResult:
    """统一数据返回结构"""
    symbol: str
    source: DataSource
    frequency: DataFrequency
    data: pd.DataFrame
    fetched_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 标准化列名映射 (所有数据源输出统一列名)
    STANDARD_COLUMNS = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
        "turnover": "turnover",
        "adj_factor": "adj_factor",
    }

    def standardize(self) -> "DataResult":
        """将各数据源的列名统一标准化"""
        # AKShare标准输出已接近,主要处理Baostock等差异
        col_map = {
            "code": "symbol",
            "trade_date": "date",
            "tradeDate": "date",
            "openPrice": "open",
            "highPrice": "high",
            "lowPrice": "low",
            "closePrice": "close",
            "tradestatus": "trade_status",
            "isST": "is_st",
            # Baostock 涨跌幅/昨收列 — pct_change 供回测涨跌停判定使用
            # (此前缺失映射导致 _is_limit_hit 恒取默认0, 涨跌停永不触发)
            "pctChg": "pct_change",
            "preclose": "pre_close",
        }
        self.data = self.data.rename(columns={
            k: v for k, v in col_map.items() if k in self.data.columns
        })
        return self


class DataProvider(ABC):
    """
    数据提供器抽象基类

    每个具体实现负责:
    1. 从特定数据源获取数据
    2. 统一返回 DataResult 格式
    3. 声明自身的可用性状态
    """

    def __init__(self, name: str, source: DataSource):
        self.name = name
        self.source = source
        self._healthy = True

    @abstractmethod
    async def get_daily_kline(self, request: DataRequest) -> DataResult:
        """获取日K线数据"""
        ...

    @abstractmethod
    async def get_minute_kline(self, request: DataRequest) -> DataResult:
        """获取分钟K线数据"""
        ...

    @abstractmethod
    async def get_stock_list(self) -> pd.DataFrame:
        """获取全市场股票列表"""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查: 数据源是否可用"""
        ...

    async def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """
        获取实时行情 (可选实现,默认空返回)
        """
        return pd.DataFrame()

    async def get_financials(self, symbol: str) -> pd.DataFrame:
        """
        获取财务数据 (可选实现,默认空返回)
        """
        return pd.DataFrame()

    def is_healthy(self) -> bool:
        return self._healthy

    def _validate_response(self, df: pd.DataFrame, request: DataRequest) -> bool:
        """校验返回数据的基本质量"""
        if df is None or df.empty:
            return False
        required_cols = {"date", "open", "high", "low", "close"}
        actual_cols = set(df.columns)
        if not required_cols.issubset(actual_cols):
            return False
        if len(df) < 5:  # 至少5条数据
            return False
        return True
