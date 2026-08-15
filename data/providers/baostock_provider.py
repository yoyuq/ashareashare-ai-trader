"""
Baostock数据提供器 — 备用数据源

Baostock是免费证券数据平台,数据质量稳定,作为AKShare的备份。
当AKShare不可用时自动降级到Baostock。

特点:
- 数据准确度较高(来源: 上交所/深交所)
- 支持历史日/周/月/5/15/30/60分钟K线
- v2.0: 会话池 — 只login一次,所有请求共享会话,分析完再logout
"""

import asyncio
import threading
from datetime import date, datetime
from typing import List

import baostock as bs
import pandas as pd

from .base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource

# 全局查询锁 — baostock的query_history_k_data_plus不是线程安全的
# 即使共享同一个login session,并发查询也会导致死锁或返回空数据
_query_lock = threading.Lock()


def _locked_query(fn, *args, **kwargs):
    """在锁保护下执行baostock查询,防止并发死锁"""
    with _query_lock:
        return fn(*args, **kwargs)


class BaostockSession:
    """Baostock 会话池 — 全局单例,login一次,所有请求共享"""

    _instance = None
    _lock = threading.Lock()
    _refcount = 0

    def __init__(self):
        self._logged_in = False

    @classmethod
    def acquire(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            cls._refcount += 1
            if not cls._instance._logged_in:
                # v5.6 P1-5: 校验 login 返回码 (此前忽略, 登录失败会静默进入坏会话)
                lg = bs.login()
                if lg.error_code != "0" or "success" not in lg.error_msg.lower():
                    cls._refcount -= 1
                    raise RuntimeError(
                        f"Baostock登录失败: error_code={lg.error_code} "
                        f"error_msg={lg.error_msg}"
                    )
                cls._instance._logged_in = True
            return cls._instance

    @classmethod
    def release(cls):
        with cls._lock:
            if cls._refcount > 0:
                cls._refcount -= 1
            if cls._refcount == 0 and cls._instance and cls._instance._logged_in:
                try:
                    bs.logout()
                except Exception:
                    pass
                cls._instance._logged_in = False
                cls._instance = None


class BaostockProvider(DataProvider):
    """Baostock数据提供器 (备用数据源,优先级2)"""

    def __init__(self):
        super().__init__(name="BaoStock", source=DataSource.BAOSTOCK)
        self._session = None

    def __del__(self):
        self._release_session()

    def _ensure_session(self):
        if self._session is None:
            self._session = BaostockSession.acquire()

    def _release_session(self):
        if self._session:
            BaostockSession.release()
            self._session = None

    async def close(self):
        """手动释放会话 (用于async context)"""
        self._release_session()

    async def get_daily_kline(self, request: DataRequest) -> DataResult:
        """获取日K线 (v2.0: 会话池复用)"""
        symbol_raw = self._normalize_symbol(request.symbol)

        try:
            self._ensure_session()
            rs = await asyncio.to_thread(
                _locked_query,
                bs.query_history_k_data_plus,
                symbol_raw,
                "date,code,open,high,low,close,preclose,volume,amount,"
                "adjustflag,turn,tradestatus,pctChg,isST,peTTM,pbMRQ",
                start_date=request.start_date.strftime("%Y-%m-%d"),
                end_date=request.end_date.strftime("%Y-%m-%d"),
                frequency="d",
                # v3.0: 尊重请求的复权方式 (此前硬编码前复权, hfq/raw 请求被静默返回 qfq 数据)
                adjustflag={"qfq": "2", "hfq": "1"}.get(request.adjust, "3"),
            )
            data_list = []
            while (rs.error_code == "0") and rs.next():
                data_list.append(rs.get_row_data())

            df = pd.DataFrame(data_list, columns=rs.fields)

            if df.empty:
                return DataResult(
                    symbol=request.symbol,
                    source=self.source,
                    frequency=DataFrequency.DAILY,
                    data=df,
                )

            # 类型转换
            numeric_cols = ["open", "high", "low", "close", "preclose",
                          "volume", "amount", "turn", "pctChg", "peTTM", "pbMRQ"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return DataResult(
                symbol=request.symbol,
                source=self.source,
                frequency=DataFrequency.DAILY,
                data=df,
                metadata={"raw_symbol": symbol_raw},
            ).standardize()

        except Exception as e:
            self._healthy = False
            raise RuntimeError(f"Baostock获取{symbol_raw}日K线失败: {e}") from e

    async def get_minute_kline(self, request: DataRequest) -> DataResult:
        """获取分钟K线"""
        symbol_raw = self._normalize_symbol(request.symbol)
        freq_map = {
            DataFrequency.MIN5: "5",
            DataFrequency.MIN15: "15",
            DataFrequency.MIN30: "30",
            DataFrequency.MIN60: "60",
        }
        baostock_freq = freq_map.get(request.frequency, "5")

        try:
            self._ensure_session()
            rs = await asyncio.to_thread(
                _locked_query,
                bs.query_history_k_data_plus,
                symbol_raw,
                "date,time,code,open,high,low,close,volume,amount,adjustflag",
                start_date=request.start_date.strftime("%Y-%m-%d"),
                end_date=request.end_date.strftime("%Y-%m-%d"),
                frequency=baostock_freq,
                adjustflag={"qfq": "2", "hfq": "1"}.get(request.adjust, "3"),
            )
            data_list = []
            while (rs.error_code == "0") and rs.next():
                data_list.append(rs.get_row_data())

            df = pd.DataFrame(data_list, columns=rs.fields)
            if not df.empty:
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

            return DataResult(
                symbol=request.symbol,
                source=self.source,
                frequency=request.frequency,
                data=df,
                metadata={"raw_symbol": symbol_raw},
            ).standardize()

        except Exception as e:
            self._healthy = False
            raise RuntimeError(f"Baostock获取{symbol_raw}分钟K线失败: {e}") from e

    async def get_stock_list(self) -> pd.DataFrame:
        """获取全市场股票列表"""
        try:
            self._ensure_session()
            rs = await asyncio.to_thread(
                _locked_query, bs.query_stock_basic, code_name=""
            )
            data_list = []
            while (rs.error_code == "0") and rs.next():
                data_list.append(rs.get_row_data())

            df = pd.DataFrame(data_list, columns=rs.fields)
            df.columns = ["symbol", "name", "ipo_date", "out_date", "type", "status"]
            return df
        except Exception as e:
            raise RuntimeError(f"Baostock获取股票列表失败: {e}") from e

    async def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """Baostock不支持实时行情,返回空"""
        return pd.DataFrame()

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            self._ensure_session()
            rs = _locked_query(
                bs.query_history_k_data_plus,
                "sh.000001",
                "date,close",
                start_date=date.today().strftime("%Y-%m-%d"),
                end_date=date.today().strftime("%Y-%m-%d"),
                frequency="d",
            )
            self._healthy = rs.error_code == "0"
        except Exception:
            self._healthy = False
        return self._healthy

    def _normalize_symbol(self, symbol: str) -> str:
        """
        标准化 → Baostock格式: sh.600000 / sz.000001 / bj.8xxxxx
        v3.0: 修复北交所代码 (8/4 开头) 此前被错误映射到 sz 的问题
        """
        symbol = symbol.strip()
        # 已经是bs格式
        if symbol.startswith(("sh.", "sz.", "bj.")):
            return symbol
        # 纯数字
        code = symbol.replace("sh.", "").replace("sz.", "").replace("bj.", "").split(".")[-1]
        if code.startswith(("6", "9")):
            return f"sh.{code}"
        if code.startswith(("8", "4")):
            return f"bj.{code}"  # 北交所 (8/4 开头)
        return f"sz.{code}"
