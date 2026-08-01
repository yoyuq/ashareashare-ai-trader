"""
Tushare 数据提供器 (v3.0)

Tushare 可靠性最高 (2026 实测可用性 ~99.9%), 作为日K线主源注册。
需要 `TUSHARE_TOKEN` 环境变量 (积分制); 未配置时优雅降级, 不阻塞路由。
"""

import asyncio
import os
from typing import List

import pandas as pd
from loguru import logger

from .base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource


class TushareProvider(DataProvider):
    """Tushare 数据源 — 主源 (可靠性最高)"""

    def __init__(self):
        super().__init__("tushare", DataSource.TUSHARE)
        self._ts = None
        self._token = os.getenv("TUSHARE_TOKEN", "")
        self._init_ts()

    def _init_ts(self):
        if not self._token:
            logger.info("TUSHARE_TOKEN 未配置, Tushare 不可用 (路由将自动跳过)")
            self._healthy = False
            return
        try:
            import tushare as ts
            ts.set_token(self._token)
            self._ts = ts
            logger.info("Tushare 客户端就绪")
        except Exception as e:
            logger.warning(f"Tushare 初始化失败: {e}")
            self._healthy = False

    @staticmethod
    def _to_ts_code(symbol: str) -> str:
        """sh.600519 → 600519.SH, sz.000001 → 000001.SZ, bj.8xxxxx → 8xxxxx.BJ"""
        market, code = symbol.split(".")
        suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(market, "SH")
        return f"{code}.{suffix}"

    @staticmethod
    def _from_ts_code(ts_code: str) -> str:
        """600519.SH → sh.600519"""
        code, mkt = ts_code.split(".")
        return {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(mkt, "sh") + "." + code

    async def get_daily_kline(self, request: DataRequest) -> DataResult:
        if self._ts is None:
            raise RuntimeError("Tushare 未配置/未初始化")
        ts_code = self._to_ts_code(request.symbol)
        adj_map = {"qfq": "qfq", "hfq": "hfq"}  # 其余 (None/raw) → 不复权

        try:
            def _sync():
                return self._ts.pro_bar(
                    ts_code=ts_code,
                    start_date=request.start_date.strftime("%Y%m%d"),
                    end_date=request.end_date.strftime("%Y%m%d"),
                    adj=adj_map.get(request.adjust, ""),
                )

            df = await asyncio.to_thread(_sync)
            if df is None or df.empty:
                return DataResult(
                    symbol=request.symbol, source=self.source,
                    frequency=request.frequency, data=pd.DataFrame(),
                )

            df = df.sort_values("trade_date").reset_index(drop=True)
            df = df.rename(columns={"trade_date": "date", "pct_chg": "pct_change", "vol": "volume"})
            # Tushare vol 单位为"手", 统一为"股" (与 Baostock 口径一致)
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100

            return DataResult(
                symbol=request.symbol, source=self.source,
                frequency=request.frequency, data=df,
                metadata={"ts_code": ts_code, "adjust": request.adjust},
            ).standardize()

        except Exception as e:
            self._healthy = False
            raise RuntimeError(f"Tushare获取{ts_code}日K线失败: {e}") from e

    async def get_minute_kline(self, request: DataRequest) -> DataResult:
        # Tushare 分钟线需较高积分等级, 暂不支持 (路由会降级到支持方)
        return DataResult(
            symbol=request.symbol, source=self.source,
            frequency=request.frequency, data=pd.DataFrame(),
        )

    async def get_stock_list(self) -> pd.DataFrame:
        if self._ts is None:
            return pd.DataFrame()
        try:
            df = await asyncio.to_thread(
                self._ts.stock_basic, exchange="", list_status="L",
                fields="ts_code,symbol,name,list_date",
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={"ts_code": "symbol"})
            df["symbol"] = df["symbol"].map(self._from_ts_code)
            return df
        except Exception as e:
            logger.warning(f"Tushare 股票列表失败: {e}")
            return pd.DataFrame()

    async def health_check(self) -> bool:
        return self._ts is not None
