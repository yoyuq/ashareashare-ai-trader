"""
PreScreener — 全市场多因子初筛引擎 (v3.1)

替代 daily_runner.py 中的简陋 ad-hoc 打分, 提供:
  - 6维度评分 (动量/价值/质量/波动/情绪/规模)
  - 市场体制自适应权重 (牛市中动量权重高, 熊市中质量权重高)
  - 三层漏斗 (流动性→质量→因子打分)
  - 行业中性化排名
  - ST/新股/僵尸股过滤

用法:
    screener = PreScreener()
    top300 = screener.screen(df, regime="weak_bull", top_n=300)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 体制自适应权重 — 不同市场环境, 因子重要性不同
# ═══════════════════════════════════════════════════════════════

REGIME_WEIGHTS = {
    "strong_bull": {
        "momentum": 0.35, "value": 0.05, "quality": 0.10,
        "volatility": 0.05, "sentiment": 0.25, "size": 0.20,
        "rationale": "牛市重势 — 动量+情绪主导, 价格强度至上"
    },
    "weak_bull": {
        "momentum": 0.25, "value": 0.15, "quality": 0.15,
        "volatility": 0.10, "sentiment": 0.15, "size": 0.20,
        "rationale": "弱牛均衡 — 动量减弱, 开始关注基本面和估值"
    },
    "range_bound": {
        "momentum": 0.10, "value": 0.25, "quality": 0.20,
        "volatility": 0.25, "sentiment": 0.10, "size": 0.10,
        "rationale": "震荡重质 — 低波动+估值保护, 反转因子有效"
    },
    "weak_bear": {
        "momentum": 0.05, "value": 0.30, "quality": 0.30,
        "volatility": 0.20, "sentiment": 0.05, "size": 0.10,
        "rationale": "弱熊防守 — 质量+估值优先, 避开高波动"
    },
    "strong_bear": {
        "momentum": 0.00, "value": 0.25, "quality": 0.40,
        "volatility": 0.25, "sentiment": 0.00, "size": 0.10,
        "rationale": "强熊避险 — 质量>低波>估值, 不追动量"
    },
    "crisis": {
        "momentum": 0.00, "value": 0.20, "quality": 0.35,
        "volatility": 0.30, "sentiment": 0.00, "size": 0.15,
        "rationale": "危机模式 — 低波+质量+大盘, 流动性优先"
    },
}


@dataclass
class ScreenResult:
    """初筛结果"""
    df: pd.DataFrame
    total_in: int
    total_out: int
    regime: str
    weights: Dict[str, float]
    score_distribution: Dict[str, float] = field(default_factory=dict)
    filter_stats: Dict[str, int] = field(default_factory=dict)


class PreScreener:
    """
    全市场多因子初筛引擎

    三层漏斗:
      L0: 硬过滤 (ST/新股/僵尸/低价/低流动性) → 5800→~4000
      L1: 质量过滤 (财务风险/持续亏损/退市风险) → ~3500
      L2: 6维因子打分 (体制自适应权重) → 排序取 Top N
    """

    # ── L0 硬过滤参数 ──
    MIN_PRICE = 3.0               # 最低价格 (排除仙股)
    MIN_AMOUNT_RANK = 0.15        # 成交额排名后15%排除 (提高门槛)
    MIN_TURNOVER = 0.20           # 最低换手率 (排除僵尸股)
    MAX_AMPLITUDE = 20.0          # 最高振幅 (排除妖股, 可选)
    MIN_MARKET_CAP = 2e9          # 最低市值20亿 (排除壳股, 提高门槛)
    EXCLUDE_BSE = True            # 排除北交所 (流动性差, 波动极端)

    # ── L1 质量过滤 ──
    MAX_PE_SPIKE = 200            # PE>200可能是亏损/微利
    MIN_PB = 0.2                  # PB<0.2有退市风险
    MAX_PB_SPIKE = 50             # PB>50可能是壳/概念炒作

    def screen(
        self,
        df: pd.DataFrame,
        regime: str = "range_bound",
        top_n: int = 300,
        industry_neutral: bool = False,
    ) -> ScreenResult:
        """
        执行全市场初筛

        Args:
            df: 全市场实时行情 DataFrame (来自 akshare stock_zh_a_spot_em)
                需含列: code, name, price, pct_change, volume, amount,
                        turnover, pe_ttm, pb, total_mv, vol_ratio,
                        pct_60d, amplitude
            regime: 市场体制
            top_n: 最终返回的股票数量
            industry_neutral: 是否启用行业中性化

        Returns:
            ScreenResult with ranked DataFrame
        """
        total_in = len(df)

        # L0: 硬过滤
        df = self._hard_filter(df)
        after_L0 = len(df)

        # L1: 质量过滤
        df = self._quality_filter(df)
        after_L1 = len(df)

        # L2: 6维打分
        weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["range_bound"])
        df = self._compute_scores(df, weights)

        # 行业中性化 (可选)
        if industry_neutral and 'industry' in df.columns:
            df = self._neutralize_industry(df)

        # 排序取 Top N
        df = df.sort_values('composite_score', ascending=False)
        df_top = df.head(top_n).copy()
        df_top['rank'] = range(1, len(df_top) + 1)

        # 10分位统计
        score_dist = {
            "min": round(float(df_top['composite_score'].min()), 1),
            "p25": round(float(df_top['composite_score'].quantile(0.25)), 1),
            "median": round(float(df_top['composite_score'].median()), 1),
            "p75": round(float(df_top['composite_score'].quantile(0.75)), 1),
            "max": round(float(df_top['composite_score'].max()), 1),
        }

        logger.info(
            f"[PreScreener] {total_in} -> L0:{after_L0} -> L1:{after_L1} -> Top{top_n}"
            f" | regime={regime} weight={weights['rationale']}"
        )

        return ScreenResult(
            df=df_top,
            total_in=total_in,
            total_out=top_n,
            regime=regime,
            weights={k: v for k, v in weights.items() if k != "rationale"},
            score_distribution=score_dist,
            filter_stats={
                "total": total_in,
                "after_hard_filter": after_L0,
                "after_quality_filter": after_L1,
                "final": top_n,
            },
        )

    # ═══════════════════════════════════════════════════════════════
    # L0: 硬过滤
    # ═══════════════════════════════════════════════════════════════

    def _hard_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤 ST/新股/僵尸/仙股/流动性枯竭"""
        initial = len(df)

        # 基础字段非空
        req_cols = ['code', 'name', 'price', 'pct_change', 'amount', 'turnover']
        for c in req_cols:
            if c in df.columns:
                df = df[df[c].notna()]

        # ST / *ST 过滤
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退', na=False)]

        # 新股过滤 (< 60个交易日, 用60日涨跌幅为空来判断)
        if 'pct_60d' in df.columns:
            # 保留pct_60d有值的 (非新股) 或者 pct_60d 为0但amount足够大的
            df = df[df['pct_60d'].notna()]

        # 价格
        if 'price' in df.columns:
            df = df[df['price'] >= self.MIN_PRICE]

        # 成交额 (剔除后10%)
        if 'amount' in df.columns:
            threshold = df['amount'].quantile(self.MIN_AMOUNT_RANK)
            df = df[df['amount'] >= threshold]

        # 换手率
        if 'turnover' in df.columns:
            df = df[df['turnover'] >= self.MIN_TURNOVER]

        # 市值
        if 'total_mv' in df.columns:
            df = df[df['total_mv'] >= self.MIN_MARKET_CAP]

        # 北交所排除 (代码以8/9/4开头, 流动性差+波动极端)
        if self.EXCLUDE_BSE and 'code' in df.columns:
            df = df[~df['code'].astype(str).str.match(r'^(8|9|4)\d{5}')]

        logger.debug(f"[L0] Hard filter: {initial} -> {len(df)} "
                     f"(-{initial - len(df)} removed: ST/IPO/BSE/low_liquidity)")

        return df.copy()

    # ═══════════════════════════════════════════════════════════════
    # L1: 质量过滤
    # ═══════════════════════════════════════════════════════════════

    def _quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤财务异常标的"""
        initial = len(df)

        # PE 极端值 (可能是微利股, PE虚高无意义)
        if 'pe_ttm' in df.columns:
            df = df[(df['pe_ttm'] > 0) & (df['pe_ttm'] <= self.MAX_PE_SPIKE)]
            # 同时也保留PE为负但有合理市值和成交量的 (周期股)
            # 这里我们保守保留PE>0的

        # PB 合理范围
        if 'pb' in df.columns:
            df = df[(df['pb'] >= self.MIN_PB) & (df['pb'] <= self.MAX_PB_SPIKE)]

        # 当日振幅合理 (<20%排除妖股)
        if 'amplitude' in df.columns:
            df = df[df['amplitude'] <= self.MAX_AMPLITUDE]

        # 当日涨跌幅在合理范围 (排除一字板涨停/跌停, 无法交易)
        if 'pct_change' in df.columns:
            df = df[(df['pct_change'] > -9.5) & (df['pct_change'] < 9.5)]

        logger.debug(f"[L1] Quality filter: {initial} -> {len(df)} "
                     f"(-{initial - len(df)} removed: PE/PB/amplitude)")

        return df.copy()

    # ═══════════════════════════════════════════════════════════════
    # L2: 6维因子打分
    # ═══════════════════════════════════════════════════════════════

    def _compute_scores(self, df: pd.DataFrame, weights: Dict) -> pd.DataFrame:
        """计算6个维度的因子分并合成综合评分"""
        df = df.copy()

        scores = {}

        # 维度1: 动量 (Momentum) — 价格趋势和强度
        scores["momentum"] = self._score_momentum(df)

        # 维度2: 价值 (Value) — 估值便宜程度
        scores["value"] = self._score_value(df)

        # 维度3: 质量 (Quality) — 经营质量和稳定性
        scores["quality"] = self._score_quality(df)

        # 维度4: 波动率 (Volatility) — 风险调整后收益
        scores["volatility"] = self._score_volatility(df)

        # 维度5: 情绪/资金 (Sentiment/Flow) — 市场关注度
        scores["sentiment"] = self._score_sentiment(df)

        # 维度6: 规模 (Size) — 市值特征
        scores["size"] = self._score_size(df)

        # 加权合成
        composite = pd.Series(0.0, index=df.index)
        for dim, w in weights.items():
            if dim in scores and w > 0:
                composite += scores[dim] * w

        # 归一化到 0-100
        if composite.std() > 0:
            composite = (composite - composite.min()) / (composite.max() - composite.min()) * 100
        else:
            composite = pd.Series(50.0, index=df.index)

        df["composite_score"] = composite

        # 保存各维度得分
        for dim, s in scores.items():
            df[f"score_{dim}"] = s

        return df

    # ── 动量维度 ──

    def _score_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        动量评分: 短期趋势 + 中期趋势 + 相对强度

        因子:
          - 当日涨幅 (正贡献, clip -5~+8)
          - 60日涨幅 (中期动量, clip -20~+50)
          - 当日涨幅 > 0 (上涨家数偏好)
          - 当日涨幅排名分位
        """
        score = pd.Series(0.0, index=df.index)

        if 'pct_change' in df.columns:
            # 非线性映射: 涨得太多扣分(追高风险), 小幅上涨加分最多
            s = df['pct_change'].clip(-5, 8)
            # 最优区间: 1%~5% (温和上涨)
            optimal = ((s >= 1) & (s <= 5)).astype(float) * 10
            normal = s * 2
            score += optimal + normal.clip(-10, 20)

        if 'pct_60d' in df.columns:
            # 中期动量: 正动量加分, 但近期涨太多 (>40%) 扣分
            s60 = df['pct_60d'].clip(-30, 50)
            score += s60 * 0.3
            # 过度上涨惩罚
            overbought = (df['pct_60d'] > 40).astype(float) * -5
            score += overbought

        # 量比确认: 放量上涨加分, 缩量上涨扣分
        if 'vol_ratio' in df.columns and 'pct_change' in df.columns:
            vol_confirmed = (
                (df['vol_ratio'] > 1.2) & (df['pct_change'] > 0)
            ).astype(float) * 5
            score += vol_confirmed

        return self._normalize_score(score, "momentum")

    # ── 价值维度 ──

    def _score_value(self, df: pd.DataFrame) -> pd.Series:
        """
        价值评分: 估值便宜程度

        因子:
          - PE_TTM (越低越好, 20x为中性)
          - PB (越低越好, 2x为中性)
          - 市值适中 (太小有风险, 太大没弹性)
        """
        score = pd.Series(0.0, index=df.index)

        if 'pe_ttm' in df.columns:
            # PE梯度评分: PE在10-30之间最优, PE>100惩罚
            pe = df['pe_ttm'].clip(5, 150)
            # 中心在20, 分布评分
            pe_score = 20 - abs(pe - 18) / 2
            pe_score = pe_score.clip(-20, 25)
            score += pe_score

        if 'pb' in df.columns:
            pb = df['pb'].clip(0.3, 30)
            # PB在0.5~3之间最优
            pb_score = 15 - abs(pb - 1.8) * 3
            pb_score = pb_score.clip(-15, 20)
            score += pb_score

        return self._normalize_score(score, "value")

    # ── 质量维度 ──

    def _score_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        质量评分: 经营质量和稳定性

        (基于实时行情数据的代理指标, 真实质量需要财务报表)
        代理因子:
          - 换手率适中 (1%~8%最优, 太高=投机/太低=无人问津)
          - PE为正且合理 (盈利公司)
          - 市值>中位数 (大盘股更稳定)
          - 振幅不过大 (稳定性)
        """
        score = pd.Series(0.0, index=df.index)

        # 换手率适中
        if 'turnover' in df.columns:
            t = df['turnover'].clip(0.1, 25)
            # 最优区间 1.5%~8%
            in_sweet_spot = ((t >= 1.5) & (t <= 8)).astype(float) * 15
            # 过高或过低都扣分
            too_high = (t > 15).astype(float) * -8
            too_low = (t < 0.5).astype(float) * -5
            score += in_sweet_spot + too_high + too_low

        # PE为正 (盈利公司加分)
        if 'pe_ttm' in df.columns:
            profitable = ((df['pe_ttm'] > 0) & (df['pe_ttm'] < 60)).astype(float) * 10
            score += profitable

        # 振幅不过大
        if 'amplitude' in df.columns:
            amp = df['amplitude']
            low_amp = (amp <= 5).astype(float) * 8
            mid_amp = ((amp > 5) & (amp <= 8)).astype(float) * 3
            high_amp = (amp > 12).astype(float) * -6
            score += low_amp + mid_amp + high_amp

        return self._normalize_score(score, "quality")

    # ── 波动率维度 ──

    def _score_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        波动率评分: 风险调整后收益

        因子:
          - 当日涨跌幅稳定性 (涨幅在-3~+5最优)
          - 振幅适中 (3%~8% = 有交易机会, 但不太妖)
          - 60日涨幅稳定性
        """
        score = pd.Series(0.0, index=df.index)

        if 'pct_change' in df.columns:
            # 偏离0太远 = 高波动风险
            stable = 10 - abs(df['pct_change']) * 0.8
            score += stable.clip(-5, 12)

        if 'amplitude' in df.columns:
            amp = df['amplitude']
            # 3~8%振幅最优: 有波动有利润, 但不过于剧烈
            amp_optimal = ((amp >= 3) & (amp <= 8)).astype(float) * 8
            amp_too_high = (amp > 15).astype(float) * -10
            score += amp_optimal + amp_too_high

        # 60日涨跌幅的稳定性 (用绝对值近似)
        if 'pct_60d' in df.columns:
            # 中期涨幅过大 = 高回撤风险
            extreme_run = (abs(df['pct_60d']) > 50).astype(float) * -8
            score += extreme_run

        return self._normalize_score(score, "volatility")

    # ── 情绪/资金维度 ──

    def _score_sentiment(self, df: pd.DataFrame) -> pd.Series:
        """
        情绪/资金流评分: 市场关注度

        因子:
          - 量比 (vol_ratio > 1 = 今日放量, 资金关注)
          - 换手率排名 (前20% = 市场焦点)
          - 涨幅排名 (当日强势)
          - 成交额排名 (大资金参与)
        """
        score = pd.Series(0.0, index=df.index)

        # 量比
        if 'vol_ratio' in df.columns:
            vr = df['vol_ratio'].clip(0.3, 8)
            # 量比1.2~3最优 (温和放量, 不是巨量出货)
            optimal_vr = ((vr >= 1.2) & (vr <= 3.0)).astype(float) * 12
            slightly_high = ((vr > 3) & (vr <= 5)).astype(float) * 4
            too_high = (vr > 5).astype(float) * -3
            score += optimal_vr + slightly_high + too_high

        # 换手率排名
        if 'turnover' in df.columns:
            turnover_rank = df['turnover'].rank(pct=True)
            # 前30%的换手 = 受关注
            score += ((turnover_rank > 0.7) & (turnover_rank <= 0.95)).astype(float) * 6
            # 最高5%换手 = 过度投机
            score += (turnover_rank > 0.95).astype(float) * -3

        # 成交额排名
        if 'amount' in df.columns:
            amount_rank = df['amount'].rank(pct=True)
            # 前30%成交额 = 主力资金参与
            score += ((amount_rank > 0.7) & (amount_rank <= 0.95)).astype(float) * 5

        return self._normalize_score(score, "sentiment")

    # ── 规模维度 ──

    def _score_size(self, df: pd.DataFrame) -> pd.Series:
        """
        规模评分: 市值特征

        因子:
          - 总市值排名 (中大盘股加分, 但超大市值弹性不足)
          - 价格适中 (10~50元区间最受机构青睐)
        """
        score = pd.Series(0.0, index=df.index)

        if 'total_mv' in df.columns:
            mv_rank = df['total_mv'].rank(pct=True)
            # 前20%~80% = 中盘股 (弹性+流动性平衡)
            mid_cap = ((mv_rank >= 0.2) & (mv_rank <= 0.8)).astype(float) * 8
            # 前20% = 大盘蓝筹 (稳定但弹性不足)
            large_cap = (mv_rank > 0.8).astype(float) * 4
            # 后10% = 微盘 (流动性差)
            micro_cap = (mv_rank < 0.1).astype(float) * -5
            score += mid_cap + large_cap + micro_cap

        if 'price' in df.columns:
            price = df['price']
            # 10~80元最优
            sweet_price = ((price >= 10) & (price <= 80)).astype(float) * 6
            too_low = (price < 5).astype(float) * -3
            score += sweet_price + too_low

        return self._normalize_score(score, "size")

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_score(score: pd.Series, name: str = "") -> pd.Series:
        """归一化到 0-100, 均值50"""
        if score.std() < 1e-8:
            return pd.Series(50.0, index=score.index)
        z = (score - score.mean()) / score.std()
        # 映射到 0-100 (均值50, ±2.5std覆盖0~100)
        normalized = 50 + z * 20
        return normalized.clip(0, 100)

    def _neutralize_industry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        行业中性化: 在每个行业内按得分排序, 取等量

        避免某一行业过度集中 (如全部选入白酒/新能源)
        """
        if 'industry' not in df.columns:
            return df

        # 每个行业取 top N
        industries = df['industry'].unique()
        per_industry = max(3, 300 // len(industries))

        result = []
        for ind in industries:
            ind_df = df[df['industry'] == ind].sort_values(
                'composite_score', ascending=False
            )
            result.append(ind_df.head(per_industry))

        return pd.concat(result, ignore_index=True)
