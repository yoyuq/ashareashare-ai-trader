"""
🆕 v2.5 全市场扫描器 — 覆盖5000+A股，智能排序

扫描流程:
  1. 获取全A股列表 (AKShare实时+缓存)
  2. 基础过滤: 非ST,上市>60天,日均成交>5000万,价格>2
  3. 多维度打分: 技术面40%+资金面25%+动量面20%+质量面15%
  4. 排序 → Top N 进入推荐引擎

输出: 每只标的的综合评分 + 各维度得分 + 推荐策略
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class StockScore:
    """单只标的评分卡"""
    symbol: str
    name: str = ""
    close: float = 0
    market_cap: float = 0             # 总市值(亿)

    # 维度得分 (0-100)
    technical_score: float = 0         # 技术面
    fund_flow_score: float = 0         # 资金面
    momentum_score: float = 0          # 动量面
    quality_score: float = 0           # 质量面(低波动+高流动性)

    # 综合
    composite: float = 0

    # 信号
    signals: List[str] = field(default_factory=list)  # 触发的信号
    matched_strategies: List[str] = field(default_factory=list)

    # 元信息
    industry: str = ""
    rank: int = 0


@dataclass
class ScanResult:
    """扫描结果"""
    scan_date: str
    total_stocks: int
    filtered_stocks: int
    top_scores: List[StockScore] = field(default_factory=list)
    sector_breakdown: Dict[str, int] = field(default_factory=dict)


class MarketScanner:
    """
    全市场扫描器

    使用方式:
        scanner = MarketScanner(router)
        result = await scanner.scan(top_n=50, min_amount=5e7)
        # result.top_scores 包含Top 50标的及评分
    """

    def __init__(self, router=None, analyzer=None):
        self.router = router
        self.analyzer = analyzer
        self._stock_cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None

    # ═══════════════════════════════════════════════════════════════
    # 主扫描入口
    # ═══════════════════════════════════════════════════════════════

    async def scan(
        self,
        top_n: int = 50,
        min_daily_amount: float = 5e7,   # 最低日成交额5000万
        min_price: float = 3.0,
        max_concurrent: int = 15,
    ) -> ScanResult:
        """
        全市场扫描

        Args:
            top_n: 返回前N只
            min_daily_amount: 最低日成交额
            min_price: 最低股价(过滤仙股)
            max_concurrent: 并发分析数
        """
        # Step 1: 获取股票列表
        all_stocks = await self._get_stock_list()
        if all_stocks.empty:
            logger.warning("无法获取股票列表")
            return ScanResult(
                scan_date=date.today().isoformat(),
                total_stocks=0, filtered_stocks=0,
            )

        total = len(all_stocks)

        # Step 2: 基础过滤
        filtered = self._basic_filter(all_stocks, min_daily_amount, min_price)
        logger.info(f"全市场扫描: {total}→{len(filtered)}只 (过滤ST/仙股/僵尸股)")

        # Step 3: 批量评分(并发)
        sem = asyncio.Semaphore(max_concurrent)
        tasks = []
        for _, row in filtered.iterrows():
            tasks.append(self._score_stock(row, sem))

        scores = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 4: 排序
        valid_scores = []
        for s in scores:
            if isinstance(s, Exception):
                continue
            if s is not None and s.composite > 0:
                valid_scores.append(s)

        valid_scores.sort(key=lambda x: x.composite, reverse=True)

        # Step 5: 行业分布
        sectors = {}
        for s in valid_scores[:top_n]:
            ind = s.industry or "其他"
            sectors[ind] = sectors.get(ind, 0) + 1

        return ScanResult(
            scan_date=date.today().isoformat(),
            total_stocks=total,
            filtered_stocks=len(filtered),
            top_scores=valid_scores[:top_n],
            sector_breakdown=sectors,
        )

    # ═══════════════════════════════════════════════════════════════
    # 股票列表
    # ═══════════════════════════════════════════════════════════════

    async def _get_stock_list(self) -> pd.DataFrame:
        """获取全A股列表(缓存1小时)"""
        if self._stock_cache is not None and self._cache_time is not None:
            if (datetime.now() - self._cache_time).seconds < 3600:
                return self._stock_cache

        try:
            import akshare as ak
            df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
            df.columns = [c.strip() for c in df.columns]
            self._stock_cache = df
            self._cache_time = datetime.now()
            logger.info(f"获取全A股列表: {len(df)}只")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            if self._stock_cache is not None:
                return self._stock_cache
            return pd.DataFrame()

    def _basic_filter(
        self, df: pd.DataFrame, min_amount: float, min_price: float
    ) -> pd.DataFrame:
        """基础过滤"""
        if df.empty:
            return df

        # 列名适配
        name_col = "名称" if "名称" in df.columns else "name"
        price_col = "最新价" if "最新价" in df.columns else "price"
        amount_col = "成交额" if "成交额" in df.columns else "amount"
        mv_col = "总市值" if "总市值" in df.columns else "total_mv"

        # 排除ST/退市/N
        if name_col in df.columns:
            df = df[~df[name_col].str.contains("ST|退|N", na=True)]

        # 价格>min_price
        if price_col in df.columns:
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
            df = df[df[price_col] > min_price]

        # 成交额>min_amount
        if amount_col in df.columns:
            df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
            df = df[df[amount_col] > min_amount]

        # 市值>10亿(排除壳股)
        if mv_col in df.columns:
            df[mv_col] = pd.to_numeric(df[mv_col], errors="coerce")
            df = df[df[mv_col] > 1e9]

        # 取Top 500(避免扫描太多)
        if amount_col in df.columns:
            df = df.sort_values(amount_col, ascending=False).head(500)

        return df.reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════
    # 单只评分
    # ═══════════════════════════════════════════════════════════════

    async def _score_stock(
        self, row: pd.Series, sem: asyncio.Semaphore
    ) -> Optional[StockScore]:
        """对单只标的打分"""
        async with sem:
            try:
                code = str(row.get("代码", row.get("code", "")))
                name = str(row.get("名称", row.get("name", "")))
                price = float(row.get("最新价", row.get("price", 0)))
                mv = float(row.get("总市值", row.get("total_mv", 0))) / 1e8

                sym = f"sh.{code}" if code.startswith("6") else f"sz.{code}"

                # 拉取日K线
                from data.providers.base import DataFrequency, DataRequest
                if self.router is None:
                    return None

                req = DataRequest(
                    symbol=sym,
                    start_date=date.today() - timedelta(days=120),
                    end_date=date.today(),
                    frequency=DataFrequency.DAILY,
                )
                result = await self.router.get_daily_kline(req)
                df = result.data

                if df.empty or len(df) < 30:
                    return None

                # 计算指标
                from analysis.indicators import TechnicalAnalyzer
                if self.analyzer is None:
                    self.analyzer = TechnicalAnalyzer()
                ind = self.analyzer.compute_all(df, symbol=sym)
                last = ind.to_dataframe().iloc[-1]

                close_arr = df["close"].values
                volume_arr = df["volume"].values
                current_close = float(close_arr[-1])

                # ===== 1. 技术面评分 (40%) =====
                tech = self._score_technical(last, close_arr, volume_arr)
                technical_score = min(tech, 100)

                # ===== 2. 资金面评分 (25%) =====
                fund = self._score_fund_flow(last, close_arr, volume_arr)
                fund_flow_score = min(fund, 100)

                # ===== 3. 动量面评分 (20%) =====
                mom = self._score_momentum(last, close_arr)
                momentum_score = min(mom, 100)

                # ===== 4. 质量面评分 (15%) =====
                qual = self._score_quality(last, close_arr, volume_arr)
                quality_score = min(qual, 100)

                # ===== 综合 =====
                composite = (
                    technical_score * 0.40 +
                    fund_flow_score * 0.25 +
                    momentum_score * 0.20 +
                    quality_score * 0.15
                )

                # ===== 检测信号 =====
                signals = self._detect_signals(last, df)

                return StockScore(
                    symbol=sym,
                    name=name,
                    close=round(current_close, 2),
                    market_cap=round(mv, 1),
                    technical_score=round(technical_score, 1),
                    fund_flow_score=round(fund_flow_score, 1),
                    momentum_score=round(momentum_score, 1),
                    quality_score=round(quality_score, 1),
                    composite=round(composite, 1),
                    signals=signals,
                )

            except Exception as e:
                logger.debug(f"评分异常 {row.get('代码','?')}: {e}")
                return None

    # ═══════════════════════════════════════════════════════════════
    # 评分子维度
    # ═══════════════════════════════════════════════════════════════

    def _score_technical(self, last: pd.Series, close: np.ndarray,
                         volume: np.ndarray) -> float:
        """技术面评分 (40%)"""
        score = 50.0  # 基准50分

        # MA排列 (+/- 20分)
        ma5 = float(last.get("ma_5", 0))
        ma20 = float(last.get("ma_20", 0))
        ma60 = float(last.get("ma_60", 0))
        price = float(last.get("close", close[-1]))

        if ma5 > ma20 > ma60 and price > ma5:
            score += 20  # 完美多头排列
        elif ma5 > ma20 and price > ma20:
            score += 12
        elif ma5 < ma20 < ma60 and price < ma5:
            score -= 15  # 空头排列
        elif ma5 < ma20:
            score -= 5

        # MACD (+/- 15分)
        macd_hist = float(last.get("macd_hist", 0))
        if macd_hist > 0:
            score += 8
            if macd_hist > float(last.get("macd_hist", 0)):  # 柱线增长
                score += 7
        else:
            score -= 5

        # RSI位置 (+/- 10分)
        rsi = float(last.get("rsi_14", 50))
        if 40 <= rsi <= 65:
            score += 10  # 健康区间
        elif 30 <= rsi < 40:
            score += 8   # 超卖反弹机会
        elif rsi > 75:
            score -= 8   # 超买风险
        elif rsi < 25:
            score -= 3   # 极度弱势

        # 布林带位置 (+/- 10分)
        bb_pct = float(last.get("bb_pct_20", 0.5))
        if 0.2 <= bb_pct <= 0.6:
            score += 8  # 中下轨,有上涨空间
        elif bb_pct > 0.9:
            score -= 5  # 上轨附近,短期见顶

        # ADX趋势强度 (+/- 5分)
        adx = float(last.get("adx_14", 20))
        if adx > 25:
            score += 5  # 趋势明确

        return score

    def _score_fund_flow(self, last: pd.Series, close: np.ndarray,
                         volume: np.ndarray) -> float:
        """资金面评分 (25%)"""
        score = 50.0

        # 量价配合 (+/- 20分)
        vol_ratio = float(last.get("vol_ratio_5", 1.0))
        pct_change = float(last.get("pct_change", 0)) if "pct_change" in last.index else 0

        # 价涨量增(最佳)
        if pct_change > 1 and 1.2 <= vol_ratio <= 2.5:
            score += 20
        elif pct_change > 0 and vol_ratio > 1:
            score += 12
        # 价平量增(资金进场)
        elif abs(pct_change) < 1 and vol_ratio > 1.5:
            score += 15
        # 价跌量缩(正常调整)
        elif pct_change < -1 and vol_ratio < 0.7:
            score += 5
        # 价跌量增(出货)
        elif pct_change < -2 and vol_ratio > 1.8:
            score -= 20
        # 价涨量缩(背离)
        elif pct_change > 1 and vol_ratio < 0.6:
            score -= 10

        # 换手率 (+/- 10分)
        turnover = float(last.get("turnover", last.get("turnover_ma5", 3.0)))
        if 2 <= turnover <= 8:
            score += 10  # 活跃但不狂热
        elif turnover > 15:
            score -= 5   # 过度投机
        elif turnover < 1:
            score -= 8   # 无人问津

        # OBV趋势 (+/- 10分)
        obv_val = float(last.get("obv", 0))
        obv_ma = float(last.get("obv_ma_20", 0))
        if obv_val > obv_ma:
            score += 8

        return score

    def _score_momentum(self, last: pd.Series, close: np.ndarray) -> float:
        """动量面评分 (20%)"""
        score = 50.0

        # 短期动量 (+/- 15分)
        roc_5 = float(last.get("roc_5", 0))
        roc_20 = float(last.get("roc_20", 0))

        # 短期有动量但不极端
        if 2 <= roc_5 <= 8:
            score += 12
        elif 0 < roc_5 < 2:
            score += 5
        elif roc_5 > 12:
            score -= 8  # 短期过热

        # 中期趋势
        if 5 <= roc_20 <= 20:
            score += 8
        elif roc_20 > 25:
            score -= 3

        # 连续涨/跌 (+/- 10分)
        consec_up = int(last.get("consecutive_up", 0))
        consec_down = int(last.get("consecutive_down", 0))
        if 2 <= consec_up <= 5:
            score += 8  # 温和连涨
        elif consec_up > 7:
            score -= 3  # 连涨过多,回调风险
        if consec_down >= 3:
            score -= 5  # 连跌,弱势

        # 创新高/新低 (+/- 10分)
        is_high = int(last.get("is_high_20", 0))
        is_low = int(last.get("is_low_20", 0))
        if is_high and roc_20 > 0:
            score += 10  # 突破新高
        if is_low:
            score -= 10

        return score

    def _score_quality(self, last: pd.Series, close: np.ndarray,
                       volume: np.ndarray) -> float:
        """质量面评分 (15%) — 低波动+高流动性"""
        score = 50.0

        # 波动率 (+/- 10分) — 低波异象
        hv_20 = float(last.get("hv_20", 30))
        if hv_20 < 25:
            score += 10
        elif hv_20 > 50:
            score -= 8

        # 成交量稳定性 (+/- 5分)
        vol_ma20 = float(last.get("vol_ma_20", volume[-1]))
        vol_std = np.std(volume[-20:]) / (vol_ma20 + 1)  # 变异系数
        if vol_std < 0.5:
            score += 5

        # 最大回撤 (+/- 10分)
        dd = float(last.get("drawdown_20", 0))
        if dd > -5:
            score += 5
        elif dd < -15:
            score -= 10

        return score

    # ═══════════════════════════════════════════════════════════════
    # 信号检测
    # ═══════════════════════════════════════════════════════════════

    def _detect_signals(self, last: pd.Series, df: pd.DataFrame) -> List[str]:
        """检测交易信号"""
        signals = []

        # MA金叉
        if int(last.get("macd_golden_cross", 0)) > 0:
            signals.append("MACD金叉")
        if int(last.get("ma_bullish", 0)) > 0 and int(last.get("ma_bullish", 0)) <= 3:
            signals.append("均线转多头")

        # 放量突破
        vol_ratio = float(last.get("vol_ratio_5", 1))
        pct = float(last.get("pct_change", 0)) if "pct_change" in last.index else 0
        if vol_ratio > 1.5 and pct > 2:
            signals.append("放量上涨")

        # 超卖反弹
        rsi = float(last.get("rsi_14", 50))
        if rsi < 30:
            signals.append("RSI超卖")

        # 布林下轨
        bb_pct = float(last.get("bb_pct_20", 0.5))
        if bb_pct < 0.1:
            signals.append("布林下轨")

        # 趋势强势
        trend = float(last.get("trend_score", 0))
        if trend > 0.5:
            signals.append("趋势强势")

        # 底背离(EOD简单判断)
        if rsi < 40 and pct > 2:
            signals.append("潜在反转")

        # K线形态
        patterns = {
            "hammer": "锤子线", "bullish_engulfing": "看涨吞没",
            "morning_star": "晨星", "three_soldiers": "红三兵",
            "piercing_line": "穿刺线",
        }
        for p_key, p_name in patterns.items():
            if hasattr(last, p_key) or p_key in last.index:
                pv = last.get(p_key, 0)
                if hasattr(pv, 'iloc'): pv = pv.iloc[-1] if hasattr(pv, 'iloc') else pv
                if pv and int(pv) > 0:
                    signals.append(p_name)

        return signals
