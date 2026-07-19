"""
🆕 v2.1 另类数据提供器

采集7个低拥挤度另类数据源,提供传统价量数据之外的信号维度:
1. 互动易问答 — 投资者互动平台(Q&A情感分析)
2. 股东户数变化 — 筹码集中度(季报附注)
3. 涨停原因归因 — 同花顺人工标注涨停逻辑
4. 融资融券明细 — 杠杆资金情绪
5. 限售解禁日历 — 供给冲击预警
6. 大宗交易溢价率 — 机构动向
7. 个股热度排名 — 散户关注度
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import List, Optional

import akshare as ak
import pandas as pd
from loguru import logger

from .base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource


class AlternativeDataProvider(DataProvider):
    """🆕 v2.1 另类数据提供器"""

    def __init__(self):
        super().__init__(name="AlternativeData", source=DataSource.ALTERNATIVE)

    # ═══════════════════════════════════════════════════════════════
    # 1. 互动易问答 (拥挤度: 🟢 极低)
    # ═══════════════════════════════════════════════════════════════

    async def get_irm_qa(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """
        投资者互动平台问答数据

        Returns:
            columns: [提问时间, 提问内容, 回复时间, 回复内容, 回复速度(hours)]
        """
        try:
            df = await asyncio.to_thread(
                ak.stock_irm_ans, symbol=self._normalize(symbol)
            )
            if df.empty:
                return df

            # 过滤时间范围
            if "提问时间" in df.columns:
                df["提问时间"] = pd.to_datetime(df["提问时间"], errors="coerce")
                cutoff = datetime.now() - timedelta(days=days)
                df = df[df["提问时间"] >= cutoff]

            # 计算回复速度
            if "提问时间" in df.columns and "回复时间" in df.columns:
                df["回复时间"] = pd.to_datetime(df["回复时间"], errors="coerce")
                df["回复速度_hours"] = (
                    (df["回复时间"] - df["提问时间"]).dt.total_seconds() / 3600
                )

            return df

        except Exception as e:
            logger.warning(f"互动易数据获取失败({symbol}): {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════
    # 2. 股东户数变化 (拥挤度: 🟢 低)
    # ═══════════════════════════════════════════════════════════════

    async def get_shareholder_count(self, symbol: str) -> pd.DataFrame:
        """
        股东户数(季报附注) + 衍生指标

        Returns:
            columns: [日期, 股东户数, 户均持股数, 环比变化%]
        """
        try:
            df = await asyncio.to_thread(
                ak.stock_zh_a_gdhs, symbol=self._normalize(symbol)
            )
            if df.empty:
                return df

            # 计算衍生指标
            if "holder_num" in df.columns:
                df = df.sort_values("日期" if "日期" in df.columns else "date")
                df["holder_change_pct"] = df["holder_num"].pct_change() * 100
                # 股东户数减少→筹码集中→利好信号
                df["concentration_signal"] = (
                    df["holder_change_pct"].apply(
                        lambda x: 1 if x < -5 else (-1 if x > 10 else 0)
                    )
                )
            return df
        except Exception as e:
            logger.warning(f"股东户数数据获取失败({symbol}): {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════
    # 3. 涨停原因题材归因 (拥挤度: 🟢 低)
    # ═══════════════════════════════════════════════════════════════

    async def get_limit_up_reason(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """
        涨停原因 + 题材标签 + 连板统计

        Returns:
            columns: [symbol, 涨停原因, 题材标签, 连板数, 封板时间, 开板次数, 封单额/成交额]
        """
        if trade_date is None:
            trade_date = date.today().strftime("%Y%m%d")

        try:
            # AKShare涨停板数据
            df = await asyncio.to_thread(
                ak.stock_zt_pool_em, date=trade_date
            )
            if df.empty:
                return df

            # 增强: 连板统计
            df["连板数"] = df.groupby("代码")["代码"].transform("count")

            return df
        except Exception as e:
            logger.warning(f"涨停归因数据获取失败({trade_date}): {e}")
            return pd.DataFrame()

    async def get_stock_zt_pool_strong_em(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """强势股池(炸板/烂板/回封)"""
        if trade_date is None:
            trade_date = date.today().strftime("%Y%m%d")
        try:
            return await asyncio.to_thread(
                ak.stock_zt_pool_strong_em, date=trade_date
            )
        except Exception:
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════
    # 4. 融资融券明细 (拥挤度: 🟡 中)
    # ═══════════════════════════════════════════════════════════════

    async def get_margin_detail(self, symbol: str) -> pd.DataFrame:
        """
        融资融券日度明细

        关键指标:
        - 融资余额变化率: 杠杆资金情绪(增长=看多, 减少=看空)
        - 融资买入占比: 激进程度(=融资买入额/总成交额)
        """
        try:
            df = await asyncio.to_thread(
                ak.stock_margin_detail_sse, date=""
            )
            sym = self._normalize(symbol)
            if not df.empty and "股票代码" in df.columns:
                df = df[df["股票代码"] == sym]

            # 计算融资买入占比
            if "融资买入额" in df.columns and "成交额" in df.columns:
                df["margin_buy_ratio"] = (
                    pd.to_numeric(df["融资买入额"], errors="coerce") /
                    pd.to_numeric(df["成交额"], errors="coerce")
                )
            return df
        except Exception as e:
            logger.warning(f"融资融券数据获取失败({symbol}): {e}")
            return pd.DataFrame()

    async def get_margin_summary(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """全市场融资融券汇总(融资余额/融券余额/融资买入额)"""
        if trade_date is None:
            trade_date = date.today().strftime("%Y%m%d")
        try:
            return await asyncio.to_thread(
                ak.stock_margin_sz_em, date=trade_date
            )
        except Exception:
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════
    # 5. 限售解禁日历 (拥挤度: 🟢 极低)
    # ═══════════════════════════════════════════════════════════════

    async def get_unlock_calendar(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        限售解禁日历

        关注: 解禁市值/流通市值 > 5% → 供给冲击风险
        """
        if start_date is None:
            start_date = date.today().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            df = await asyncio.to_thread(
                ak.stock_restricted_release_queue_em,
                symbol="全部",
                start_date=start_date,
                end_date=end_date,
            )
            return df
        except Exception as e:
            logger.warning(f"限售解禁数据获取失败: {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════
    # 6. 大宗交易 (拥挤度: 🟡 中)
    # ═══════════════════════════════════════════════════════════════

    async def get_block_trades(
        self, symbol: Optional[str] = None, days: int = 30
    ) -> pd.DataFrame:
        """
        大宗交易明细

        关注: 溢价率(折价/溢价=机构态度)
        """
        try:
            # 获取最近N天大宗交易
            today = date.today()
            start = (today - timedelta(days=days)).strftime("%Y%m%d")
            end = today.strftime("%Y%m%d")

            df = await asyncio.to_thread(
                ak.stock_dzjy_mrmx,
                start_date=start,
                end_date=end,
            )
            if not df.empty and symbol:
                sym = self._normalize(symbol)
                if "证券代码" in df.columns:
                    df = df[df["证券代码"] == sym]

            # 计算溢价率
            if "成交价" in df.columns and "当日收盘价" in df.columns:
                df["premium_rate"] = (
                    (pd.to_numeric(df["成交价"], errors="coerce") /
                     pd.to_numeric(df["当日收盘价"], errors="coerce") - 1) * 100
                )
            return df
        except Exception as e:
            logger.warning(f"大宗交易数据获取失败: {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════
    # 7. 个股热度排名 (拥挤度: 🟡 中)
    # ═══════════════════════════════════════════════════════════════

    async def get_stock_heat_rank(self) -> pd.DataFrame:
        """同花顺个股热度排名 (散户关注度代理变量)"""
        try:
            df = await asyncio.to_thread(ak.stock_hot_rank_em)
            return df
        except Exception as e:
            logger.warning(f"个股热度排名获取失败: {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════
    # 综合另类数据快照
    # ═══════════════════════════════════════════════════════════════

    async def get_alternative_snapshot(self, symbol: str) -> dict:
        """
        获取某只股票的所有另类数据快照(并发请求)

        Returns:
            {
                "irm_qa_count": N条近期问答,
                "shareholder_trend": "concentrating/dispersing/stable",
                "margin_trend": "increase/decrease/stable",
                "upcoming_unlock": [解禁事件列表],
                "recent_block_trades": N条大宗交易,
                "heat_rank": 排名,
            }
        """
        irm, shareholder, margin, hot = await asyncio.gather(
            self.get_irm_qa(symbol),
            self.get_shareholder_count(symbol),
            self.get_margin_detail(symbol),
            self.get_stock_heat_rank(),
            return_exceptions=True,
        )

        snapshot = {
            "symbol": symbol,
            "fetched_at": datetime.now().isoformat(),
        }

        # 互动易
        if isinstance(irm, pd.DataFrame):
            snapshot["irm_qa_count"] = len(irm)
            snapshot["irm_avg_response_hours"] = (
                irm["回复速度_hours"].mean() if "回复速度_hours" in irm.columns else None
            )
        else:
            snapshot["irm_qa_count"] = 0

        # 股东户数趋势
        if isinstance(shareholder, pd.DataFrame) and not shareholder.empty:
            recent = shareholder.tail(2)
            if "holder_change_pct" in recent.columns:
                chg = recent["holder_change_pct"].iloc[-1]
                snapshot["shareholder_trend"] = (
                    "concentrating" if chg < -3 else
                    ("dispersing" if chg > 5 else "stable")
                )
                snapshot["shareholder_change_pct"] = chg

        # 融资趋势
        if isinstance(margin, pd.DataFrame) and not margin.empty:
            if "margin_buy_ratio" in margin.columns:
                snapshot["avg_margin_buy_ratio"] = margin["margin_buy_ratio"].mean()

        # 热度
        if isinstance(hot, pd.DataFrame) and not hot.empty:
            sym = self._normalize(symbol)
            match = hot[hot["代码"] == sym] if "代码" in hot.columns else pd.DataFrame()
            if not match.empty:
                snapshot["heat_rank"] = int(match.index[0]) + 1 if "排名" not in match.columns else match.iloc[0].get("排名")

        return snapshot

    # ═══════════════════════════════════════════════════════════════
    # 接口实现 (非主要用途,提供基础支持)
    # ═══════════════════════════════════════════════════════════════

    async def get_daily_kline(self, request: DataRequest) -> DataResult:
        return DataResult(
            symbol=request.symbol,
            source=self.source,
            frequency=request.frequency,
            data=pd.DataFrame(),
        )

    async def get_minute_kline(self, request: DataRequest) -> DataResult:
        return DataResult(
            symbol=request.symbol,
            source=self.source,
            frequency=request.frequency,
            data=pd.DataFrame(),
        )

    async def get_stock_list(self) -> pd.DataFrame:
        return pd.DataFrame()

    async def health_check(self) -> bool:
        try:
            await self.get_stock_heat_rank()
            self._healthy = True
        except Exception:
            self._healthy = False
        return self._healthy

    def _normalize(self, symbol: str) -> str:
        return symbol.replace("sh.", "").replace("sz.", "").split(".")[-1]
