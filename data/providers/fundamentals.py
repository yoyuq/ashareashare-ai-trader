"""
基本面数据提供器 (v2.9)

提供 PE/PB/ROE/营收增速/现金流/股息率等基本面数据。
数据来源: AKShare (东方财富接口)

使用:
    provider = FundamentalsProvider()
    pe_data = await provider.get_valuation("sh.600519")
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class FundamentalsProvider:
    """
    基本面数据提供器

    覆盖:
      - 估值: PE(TTM), PB, PS, 股息率
      - 盈利: ROE, 毛利率, 净利率, 营收增速, 净利润增速
      - 财务健康: 资产负债率, 流动比率, 经营现金流
      - 成长性: 3年营收CAGR, 3年净利润CAGR
    """

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=6)  # 基本面数据6小时有效

    async def get_valuation(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取估值数据

        Returns:
            {pe_ttm, pb, ps_ttm, dividend_yield, total_mv, circ_mv}
            或 None
        """
        cache_key = f"val_{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        code = symbol.replace("sh.", "").replace("sz.", "")

        try:
            import akshare as ak

            # 东方财富个股信息
            df = await asyncio.to_thread(
                ak.stock_individual_info_em, symbol=code
            )
            if df is None or df.empty:
                return None

            info = {}
            for _, row in df.iterrows():
                key = row["item"]
                val = row["value"]
                info[key] = val

            result = {
                "symbol": symbol,
                "pe_ttm": self._safe_float(info.get("市盈率-动态")),
                "pb": self._safe_float(info.get("市净率")),
                "total_mv": self._safe_float(info.get("总市值")),
                "circ_mv": self._safe_float(info.get("流通市值")),
                "revenue_per_share": self._safe_float(info.get("营业收入")),
                "eps": self._safe_float(info.get("每股收益")),
                "net_assets_per_share": self._safe_float(info.get("每股净资产")),
                "roe": self._safe_float(info.get("净资产收益率")),
                "gross_margin": self._safe_float(info.get("毛利率")),
                "net_margin": self._safe_float(info.get("净利率")),
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.debug(f"获取{symbol}基本面失败: {e}")
            return None

    async def get_financial_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取财务摘要 (含成长性指标)

        Returns:
            {pe_ttm, pb, roe, revenue_growth, profit_growth, debt_ratio, ...}
        """
        cache_key = f"fin_{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        code = symbol.replace("sh.", "").replace("sz.", "")

        try:
            import akshare as ak

            # 财务指标
            df = await asyncio.to_thread(
                ak.stock_financial_abstract_ths, symbol=code
            )

            if df is None or df.empty:
                return await self.get_valuation(symbol)

            latest = df.iloc[0] if len(df) > 0 else {}

            # 营收增速 (同比)
            rev_growth = 0.0
            if len(df) >= 2:
                try:
                    rev_now = float(df.iloc[0].get("营业收入", 0) or 0)
                    rev_prev = float(df.iloc[1].get("营业收入", 0) or 0)
                    if rev_prev > 0:
                        rev_growth = round((rev_now / rev_prev - 1) * 100, 1)
                except (ValueError, TypeError):
                    pass

            # 净利润增速
            profit_growth = 0.0
            if len(df) >= 2:
                try:
                    profit_now = float(df.iloc[0].get("净利润", 0) or 0)
                    profit_prev = float(df.iloc[1].get("净利润", 0) or 0)
                    if profit_prev > 0:
                        profit_growth = round((profit_now / profit_prev - 1) * 100, 1)
                except (ValueError, TypeError):
                    pass

            result = {
                "symbol": symbol,
                "revenue_growth_yoy": rev_growth,
                "profit_growth_yoy": profit_growth,
                "debt_ratio": self._safe_float(latest.get("资产负债率")),
                "current_ratio": self._safe_float(latest.get("流动比率")),
                "roe": self._safe_float(latest.get("净资产收益率")),
                "gross_margin": self._safe_float(latest.get("销售毛利率")),
                "net_margin": self._safe_float(latest.get("销售净利率")),
                "operating_cash_flow": self._safe_float(latest.get("经营活动现金流")),
            }

            # 补充估值数据
            val = await self.get_valuation(symbol)
            if val:
                result.update(val)

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.debug(f"获取{symbol}财务摘要失败: {e}")
            return await self.get_valuation(symbol)

    async def get_fundamental_score(self, symbol: str) -> float:
        """
        计算基本面综合评分 (0-100)

        维度:
          - 估值合理性 (PE/PB 分位数) : 30%
          - 盈利能力 (ROE)            : 25%
          - 成长性 (营收/利润增速)     : 25%
          - 财务健康 (负债率)          : 20%
        """
        data = await self.get_financial_summary(symbol)
        if data is None:
            return 50.0  # 无数据 → 中性分

        score = 0.0
        weights = 0.0

        # 估值
        pe = data.get("pe_ttm", 0)
        pb = data.get("pb", 0)
        if pe and pe > 0:
            # PE 15-25 合理区间
            if pe <= 15:
                score += 30
            elif pe <= 25:
                score += 25
            elif pe <= 40:
                score += 15
            elif pe <= 80:
                score += 8
            else:
                score += 3
            weights += 30

        # 盈利能力
        roe = data.get("roe", 0)
        if roe and roe > 0:
            if roe >= 20:
                score += 25
            elif roe >= 15:
                score += 20
            elif roe >= 10:
                score += 15
            elif roe >= 5:
                score += 10
            else:
                score += 5
            weights += 25

        # 成长性
        rev_g = data.get("revenue_growth_yoy", 0)
        profit_g = data.get("profit_growth_yoy", 0)
        if rev_g is not None or profit_g is not None:
            avg_growth = ((rev_g or 0) + (profit_g or 0)) / 2
            if avg_growth >= 30:
                score += 25
            elif avg_growth >= 15:
                score += 20
            elif avg_growth >= 5:
                score += 15
            elif avg_growth >= 0:
                score += 10
            else:
                score += 5
            weights += 25

        # 财务健康
        debt = data.get("debt_ratio", 50)
        if debt is not None:
            if debt <= 30:
                score += 20
            elif debt <= 50:
                score += 15
            elif debt <= 70:
                score += 10
            else:
                score += 5
            weights += 20

        return round(score / weights * 100) if weights > 0 else 50.0

    def _get_cache(self, key: str) -> Optional[Dict]:
        if self._cache_time and (datetime.now() - self._cache_time) < self._cache_ttl:
            return self._cache.get(key)
        # 缓存过期,清空
        if self._cache_time and (datetime.now() - self._cache_time) >= self._cache_ttl:
            self._cache.clear()
            self._cache_time = None
        return None

    def _set_cache(self, key: str, data: Dict):
        self._cache[key] = data
        if self._cache_time is None:
            self._cache_time = datetime.now()

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
            # v2.14: 0 是有效值 (如零增长、零利润), 不应视为缺失
            return f
        except (ValueError, TypeError):
            return None
