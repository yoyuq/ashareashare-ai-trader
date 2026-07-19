"""
AKShare数据提供器 — 主数据源

AKShare是免费开源金融数据接口,覆盖A股全市场行情/财务/指数/新闻等。
优先级最高,作为默认主数据源。
"""

import asyncio
from datetime import date, datetime
from typing import List, Optional

import akshare as ak
import pandas as pd

from .base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource


class AKShareProvider(DataProvider):
    """AKShare数据提供器 (主数据源,优先级1)"""

    def __init__(self):
        super().__init__(name="AKShare", source=DataSource.AKSHARE)
        self._symbol_cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None

    # ═══════════════════════════════════════════════════════════════
    # K线数据
    # ═══════════════════════════════════════════════════════════════

    async def get_daily_kline(self, request: DataRequest) -> DataResult:
        """获取A股日K线数据(含前复权)"""
        symbol_raw = self._normalize_symbol(request.symbol)

        try:
            df = await asyncio.to_thread(
                ak.stock_zh_a_hist,
                symbol=symbol_raw,
                period="daily",
                start_date=request.start_date.strftime("%Y%m%d"),
                end_date=request.end_date.strftime("%Y%m%d"),
                adjust=request.adjust or "qfq",
            )

            df = self._standardize_columns(df)

            return DataResult(
                symbol=request.symbol,
                source=self.source,
                frequency=DataFrequency.DAILY,
                data=df,
                metadata={"raw_symbol": symbol_raw, "adjust": request.adjust},
            ).standardize()

        except Exception as e:
            self._healthy = False
            raise RuntimeError(f"AKShare获取{symbol_raw}日K线失败: {e}") from e

    async def get_minute_kline(self, request: DataRequest) -> DataResult:
        """获取分钟K线数据"""
        symbol_raw = self._normalize_symbol(request.symbol)
        period_map = {
            DataFrequency.MIN1: "1",
            DataFrequency.MIN5: "5",
            DataFrequency.MIN15: "15",
            DataFrequency.MIN30: "30",
            DataFrequency.MIN60: "60",
        }

        try:
            df = await asyncio.to_thread(
                ak.stock_zh_a_hist_min_em,
                symbol=symbol_raw,
                period=period_map.get(request.frequency, "5"),
                start_date=request.start_date.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=request.end_date.strftime("%Y-%m-%d %H:%M:%S"),
                adjust=request.adjust or "",
            )

            return DataResult(
                symbol=request.symbol,
                source=self.source,
                frequency=request.frequency,
                data=df,
                metadata={"raw_symbol": symbol_raw},
            ).standardize()

        except Exception as e:
            self._healthy = False
            raise RuntimeError(f"AKShare获取{symbol_raw}分钟K线失败: {e}") from e

    # ═══════════════════════════════════════════════════════════════
    # 实时行情
    # ═══════════════════════════════════════════════════════════════

    async def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时行情(东方财富实时接口)"""
        try:
            df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
            if symbols:
                raw_symbols = [self._normalize_symbol(s) for s in symbols]
                df = df[df["代码"].isin(raw_symbols)]
            return df
        except Exception as e:
            return pd.DataFrame({"error": [str(e)]})

    # ═══════════════════════════════════════════════════════════════
    # 股票列表
    # ═══════════════════════════════════════════════════════════════

    async def get_stock_list(self) -> pd.DataFrame:
        """获取全A股股票列表(带缓存)"""
        if self._symbol_cache is not None and self._cache_time is not None:
            if (datetime.now() - self._cache_time).seconds < 86400:  # 1天缓存
                return self._symbol_cache

        try:
            df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
            df = df[["代码", "名称", "最新价", "涨跌幅", "总市值", "流通市值",
                     "市盈率-动态", "市净率", "60日涨跌幅", "年初至今涨跌幅"]]
            df.columns = ["symbol", "name", "price", "pct_change", "total_mv",
                         "float_mv", "pe_ttm", "pb", "pct_60d", "pct_ytd"]
            self._symbol_cache = df
            self._cache_time = datetime.now()
            return df
        except Exception as e:
            if self._symbol_cache is not None:
                return self._symbol_cache  # 降级: 返回旧缓存
            raise RuntimeError(f"AKShare获取股票列表失败: {e}") from e

    # ═══════════════════════════════════════════════════════════════
    # 板块/行业/指数
    # ═══════════════════════════════════════════════════════════════

    async def get_index_daily(self, index_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """获取指数日K线 (上证/深证/创业板等)"""
        idx_map = {
            "sh.000001": "000001",   # 上证综指
            "sz.399001": "399001",   # 深证成指
            "sz.399006": "399006",   # 创业板指
            "sh.000688": "000688",   # 科创50
            "sh.000300": "000300",   # 沪深300
            "sh.000905": "000905",   # 中证500
        }
        code = idx_map.get(index_code, index_code.replace("sh.", "").replace("sz.", ""))
        df = await asyncio.to_thread(
            ak.stock_zh_index_daily_em,
            symbol=f"sh{code}" if index_code.startswith("sh.") else f"sz{code}",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )
        return df

    async def get_sector_list(self) -> pd.DataFrame:
        """获取行业板块列表"""
        return await asyncio.to_thread(ak.stock_board_industry_name_em)

    async def get_sector_stocks(self, sector_name: str) -> pd.DataFrame:
        """获取某行业板块成分股"""
        return await asyncio.to_thread(
            ak.stock_board_industry_cons_em, symbol=sector_name
        )

    # ═══════════════════════════════════════════════════════════════
    # 财务数据
    # ═══════════════════════════════════════════════════════════════

    async def get_financials(self, symbol: str) -> pd.DataFrame:
        """获取财务指标"""
        raw = self._normalize_symbol(symbol)
        try:
            df = await asyncio.to_thread(
                ak.stock_financial_analysis_indicator, symbol=raw
            )
            return df
        except Exception:
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════
    # 健康检查
    # ═══════════════════════════════════════════════════════════════

    async def health_check(self) -> bool:
        """检查AKShare是否可用"""
        try:
            df = await asyncio.to_thread(
                ak.stock_zh_a_hist,
                symbol="000001",
                period="daily",
                start_date=date.today().strftime("%Y%m%d"),
                end_date=date.today().strftime("%Y%m%d"),
                adjust="qfq",
            )
            self._healthy = df is not None and not df.empty
        except Exception:
            self._healthy = False
        return self._healthy

    # ═══════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════

    def _normalize_symbol(self, symbol: str) -> str:
        """
        标准化代码格式
        sh.600000 / sz.000001 → 600000 / 000001 (纯数字)
        """
        return symbol.replace("sh.", "").replace("sz.", "").split(".")[-1]

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
