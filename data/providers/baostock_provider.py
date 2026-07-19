"""
Baostock数据提供器 — 备用数据源

Baostock是免费证券数据平台,数据质量稳定,作为AKShare的备份。
当AKShare不可用时自动降级到Baostock。

特点:
- 数据准确度较高(来源: 上交所/深交所)
- 支持历史日/周/月/5/15/30/60分钟K线
- 需先 login() 再 logout()
- 无条件单次请求,不支持异步长连接
"""

import asyncio
from datetime import date, datetime
from typing import List

import baostock as bs
import pandas as pd

from .base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource


class BaostockProvider(DataProvider):
    """Baostock数据提供器 (备用数据源,优先级2)"""

    def __init__(self):
        super().__init__(name="BaoStock", source=DataSource.BAOSTOCK)

    async def get_daily_kline(self, request: DataRequest) -> DataResult:
        """获取日K线 (Baostock格式)"""
        symbol_raw = self._normalize_symbol(request.symbol)

        try:
            bs.login()
            rs = await asyncio.to_thread(
                bs.query_history_k_data_plus,
                symbol_raw,
                "date,code,open,high,low,close,preclose,volume,amount,"
                "adjustflag,turn,tradestatus,pctChg,isST",
                start_date=request.start_date.strftime("%Y-%m-%d"),
                end_date=request.end_date.strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="2",  # 前复权
            )
            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())

            bs.logout()

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
                          "volume", "amount", "turn", "pctChg"]
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
            try:
                bs.logout()
            except Exception:
                pass
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
            bs.login()
            rs = await asyncio.to_thread(
                bs.query_history_k_data_plus,
                symbol_raw,
                "date,time,code,open,high,low,close,volume,amount,adjustflag",
                start_date=request.start_date.strftime("%Y-%m-%d"),
                end_date=request.end_date.strftime("%Y-%m-%d"),
                frequency=baostock_freq,
                adjustflag="2",
            )
            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())

            bs.logout()

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
            try:
                bs.logout()
            except Exception:
                pass
            raise RuntimeError(f"Baostock获取{symbol_raw}分钟K线失败: {e}") from e

    async def get_stock_list(self) -> pd.DataFrame:
        """获取全市场股票列表"""
        try:
            bs.login()
            rs = await asyncio.to_thread(
                bs.query_stock_basic, code_name=""
            )
            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())

            bs.logout()

            df = pd.DataFrame(data_list, columns=rs.fields)
            df.columns = ["symbol", "name", "ipo_date", "out_date", "type", "status"]
            return df
        except Exception as e:
            try:
                bs.logout()
            except Exception:
                pass
            raise RuntimeError(f"Baostock获取股票列表失败: {e}") from e

    async def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """Baostock不支持实时行情,返回空"""
        return pd.DataFrame()

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            bs.login()
            rs = bs.query_history_k_data_plus(
                "sh.000001",
                "date,close",
                start_date=date.today().strftime("%Y-%m-%d"),
                end_date=date.today().strftime("%Y-%m-%d"),
                frequency="d",
            )
            self._healthy = rs.error_code == "0"
            bs.logout()
        except Exception:
            self._healthy = False
        return self._healthy

    def _normalize_symbol(self, symbol: str) -> str:
        """
        标准化 → Baostock格式: sh.600000 或 sz.000001
        """
        symbol = symbol.strip()
        # 已经是bs格式
        if symbol.startswith("sh.") or symbol.startswith("sz."):
            return symbol
        # 纯数字
        code = symbol.replace("sh.", "").replace("sz.", "").split(".")[-1]
        if code.startswith(("6", "9")):
            return f"sh.{code}"
        else:
            return f"sz.{code}"
