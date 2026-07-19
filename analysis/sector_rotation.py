"""
🆕 v2.6 板块轮动分析 (Sector Rotation Analysis)

A股板块轮动是资金方向切换的核心指标。
- 板块热力图: 全行业收益率排名+动量
- 板块情绪六阶段: 启动→加速→主升→高潮→退潮→冰点
- 龙头识别: 板块趋势龙头+核心标的
- 板块资金流向: 大小盘/成长价值风格切换

数据来源: AKShare stock_board_* 系列
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SectorInfo:
    """单个板块信息"""
    name: str
    code: str = ""
    pct_change: float = 0            # 今日涨跌幅
    pct_5d: float = 0                # 5日涨跌幅
    pct_20d: float = 0               # 20日涨跌幅
    leading_stock: str = ""           # 龙头股
    leading_pct: float = 0           # 龙头涨幅
    up_count: int = 0                # 上涨家数
    down_count: int = 0              # 下跌家数
    turnover: float = 0              # 换手率
    fund_flow: float = 0             # 资金净流入(亿)
    phase: str = "neutral"           # 板块阶段

    @property
    def strength(self) -> float:
        """板块强度(0-100)"""
        return min(100, max(0,
            (self.pct_5d * 3 + self.pct_20d * 1.5 + self.pct_change * 2) + 50
        ))


@dataclass
class SectorHeatmap:
    """板块轮动热力图"""
    date: str
    sectors: List[SectorInfo] = field(default_factory=list)
    top_sectors: List[SectorInfo] = field(default_factory=list)
    bottom_sectors: List[SectorInfo] = field(default_factory=list)
    hot_themes: List[str] = field(default_factory=list)

    def get_sector_rank(self, sector_name: str) -> int:
        """获取某板块排名"""
        for i, s in enumerate(self.sectors):
            if s.name == sector_name:
                return i + 1
        return -1


class SectorRotationAnalyzer:
    """板块轮动分析器"""

    def __init__(self):
        self._cache: Optional[SectorHeatmap] = None
        self._cache_time: Optional[datetime] = None

    async def analyze(self) -> SectorHeatmap:
        """分析当前板块轮动"""
        # 缓存30分钟
        if self._cache and self._cache_time:
            if (datetime.now() - self._cache_time).seconds < 1800:
                return self._cache

        try:
            import akshare as ak

            # 1. 行业板块行情
            df = await asyncio.to_thread(ak.stock_board_industry_name_em)
            if df.empty:
                return SectorHeatmap(date=date.today().isoformat())

            sectors = []
            for _, row in df.iterrows():
                name = str(row.get("板块名称", row.get("name", "")))
                pct = float(row.get("涨跌幅", row.get("pct_change", 0)))

                # 5日/20日涨幅(从板块K线获取)
                pct_5d, pct_20d, phase, up_count, down_count = (
                    await self._get_sector_detail(name)
                )

                sectors.append(SectorInfo(
                    name=name,
                    pct_change=pct,
                    pct_5d=pct_5d,
                    pct_20d=pct_20d,
                    phase=phase,
                    up_count=up_count,
                    down_count=down_count,
                ))

            # 按强度排序
            sectors.sort(key=lambda s: s.strength, reverse=True)

            top = [s for s in sectors[:5] if s.strength > 50]
            bottom = [s for s in sectors[-5:] if s.strength < 50]

            # 热门题材(连续走强的板块)
            hot = [
                s.name for s in sectors[:8]
                if s.pct_5d > 3 and s.phase in ("acceleration", "main_surge")
            ]

            heatmap = SectorHeatmap(
                date=date.today().isoformat(),
                sectors=sectors,
                top_sectors=top,
                bottom_sectors=bottom,
                hot_themes=hot,
            )

            self._cache = heatmap
            self._cache_time = datetime.now()
            return heatmap

        except Exception as e:
            logger.warning(f"板块轮动分析失败: {e}")
            return SectorHeatmap(date=date.today().isoformat())

    async def _get_sector_detail(self, name: str) -> Tuple[float, float, str, int, int]:
        """获取板块详细信息"""
        try:
            import akshare as ak

            # 板块K线
            df = await asyncio.to_thread(
                ak.stock_board_industry_hist_em, symbol=name,
                period="日k", start_date=(date.today() - timedelta(days=30)).strftime("%Y%m%d"),
                end_date=date.today().strftime("%Y%m%d"),
            )
            if df.empty:
                return 0, 0, "neutral", 0, 0

            if "收盘" in df.columns:
                closes = df["收盘"].values
            elif "close" in df.columns:
                closes = df["close"].values
            else:
                return 0, 0, "neutral", 0, 0

            pct_5d = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
            pct_20d = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0

            # 阶段判断
            phase = self._classify_phase(pct_5d, pct_20d)

            return round(pct_5d, 2), round(pct_20d, 2), phase, 0, 0

        except Exception:
            return 0, 0, "neutral", 0, 0

    @staticmethod
    def _classify_phase(pct_5d: float, pct_20d: float) -> str:
        """
        板块情绪六阶段:
          start: 启动(5日转正,20日仍跌)
          acceleration: 加速(5日强劲,20日刚转正)
          main_surge: 主升(5日和20日都强劲)
          climax: 高潮(5日过度上涨>10%)
          retreat: 退潮(5日转跌,20日仍涨)
          ice_point: 冰点(5日和20日都跌)
        """
        if pct_5d > 10:
            return "climax"
        elif pct_5d > 3 and pct_20d > 5:
            return "main_surge"
        elif pct_5d > 2 and pct_20d > 0:
            return "acceleration"
        elif pct_5d > 0 > pct_20d:
            return "start"
        elif pct_5d < 0 and pct_20d > 0:
            return "retreat"
        elif pct_5d < 0 and pct_20d < 0:
            return "ice_point"
        else:
            return "neutral"

    def adjust_score_by_sector(
        self, heatmap: SectorHeatmap, stock_industry: str, base_score: float
    ) -> Tuple[float, str]:
        """
        根据板块轮动调整个股评分

        Args:
            heatmap: 板块热力图
            stock_industry: 标的所属行业
            base_score: 原始评分

        Returns:
            (adjusted_score, note)
        """
        if not heatmap.sectors:
            return base_score, "无板块数据"

        rank = heatmap.get_sector_rank(stock_industry)

        if rank < 0:
            return base_score, f"行业'{stock_industry}'不在热力图中"

        if rank <= 3:
            adj = 12
            note = f"板块排名第{rank},龙头板块,加{adj}分"
        elif rank <= 10:
            adj = 8
            note = f"板块排名第{rank},强势板块,加{adj}分"
        elif rank <= 20:
            adj = 4
            note = f"板块排名第{rank},中等偏强"
        elif rank <= 40:
            adj = 0
            note = f"板块排名第{rank},中性"
        elif rank <= 50:
            adj = -3
            note = f"板块排名第{rank},弱势板块"
        else:
            adj = -6
            note = f"板块排名第{rank},最弱板块,减{abs(adj)}分"

        # 热门题材额外加成
        if stock_industry in heatmap.hot_themes:
            adj += 5
            note += ";热门题材+5分"

        return min(100, max(0, base_score + adj)), note
