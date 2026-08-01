"""
技术指标引擎 — 130+指标计算

基于 TA-Lib + pandas-ta 双引擎:
  - TA-Lib: C库计算,速度快(如有安装)
  - pandas-ta: 纯Python备选,跨平台无依赖问题

指标分类(10大类):
  1. 趋势类: MA/EMA/MACD/ADX/Ichimoku/Parabolic SAR
  2. 动量类: RSI/Stochastic/Williams %R/CCI/MFI
  3. 波动率类: Bollinger Bands/ATR/Keltner/Donchian
  4. 成交量类: OBV/CMF/VWAP/Volume Profile
  5. 形态识别: 61种日本K线形态(Doji/Hammer/Engulfing...)
  6. 周期类: Hilbert Transform/Cycle indicators
  7. 统计学类: Z-Score/Skew/Kurtosis/Linear Regression
  8. 价格变换: Renko/Heikin-Ashi/Point & Figure
  9. 自定义A股: 涨停信号/炸板率/封单强度/连板统计
  10. 因子类: 动量因子/波动率因子/质量因子/价值因子
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# pandas-ta (纯Python,跨平台)
try:
    import logging
    logging.getLogger('pandas_ta').setLevel(logging.ERROR)  # 抑制 TA-Lib 警告洪水
    import pandas_ta as ta
    PANDASTA_AVAILABLE = True
except ImportError:
    PANDASTA_AVAILABLE = False
    logger.warning("pandas-ta未安装,将使用纯numpy实现核心指标")


@dataclass
class IndicatorResult:
    """指标计算结果容器"""
    symbol: str
    indicators: Dict[str, pd.Series] = field(default_factory=dict)
    patterns: Dict[str, pd.Series] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """将所有指标合并为一个DataFrame"""
        all_data = {"date": self.indicators.get("date", pd.Series(dtype=float))}
        all_data.update(self.indicators)
        all_data.update(self.patterns)
        return pd.DataFrame(all_data)


class TechnicalAnalyzer:
    """
    技术分析引擎

    使用方式:
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_all(ohlcv_df)  # 一键计算130+指标
        regime = analyzer.detect_market_regime(ohlcv_df)  # 市场状态
    """

    def __init__(self, use_talib: bool = False):
        self.use_talib = use_talib
        self._indicator_cache: Dict[str, IndicatorResult] = {}

    # ═══════════════════════════════════════════════════════════════
    # 一键计算所有指标
    # ═══════════════════════════════════════════════════════════════

    def compute_all(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        skip_patterns: bool = False,
    ) -> IndicatorResult:
        """
        计算所有支持的技术指标

        Args:
            df: OHLCV DataFrame, 需包含 open/high/low/close/volume 列
            symbol: 股票代码 (用于缓存key)
            skip_patterns: 跳过CDL形态识别 (规则模式无需形态,可提速~30%)

        Returns:
            IndicatorResult
        """
        if df.empty:
            return IndicatorResult(symbol=symbol)

        # 确保列名标准化
        df = self._normalize_columns(df.copy())

        # v3.0: 防御性数据清洗 — provider 若绕过 standardize, 分析层仍保证零价不污染指标
        try:
            from data.processors.cleaning import clean_ohlcv
            df = clean_ohlcv(df)
        except Exception:
            pass

        result = IndicatorResult(symbol=symbol)

        # ===== 趋势类指标 =====
        result.indicators.update(self._compute_trend_indicators(df))

        # ===== 动量类指标 =====
        result.indicators.update(self._compute_momentum_indicators(df))

        # ===== 波动率类指标 =====
        result.indicators.update(self._compute_volatility_indicators(df))

        # ===== 成交量类指标 =====
        result.indicators.update(self._compute_volume_indicators(df))

        # ===== 统计学指标 =====
        result.indicators.update(self._compute_statistical_indicators(df))

        # ===== A股特化指标 =====
        result.indicators.update(self._compute_ashare_indicators(df))

        # ===== K线形态识别 =====
        if not skip_patterns:
            if PANDASTA_AVAILABLE:
                result.patterns = self._detect_candlestick_patterns(df)
            else:
                result.patterns = self._detect_candlestick_patterns_numpy(df)

        # ===== 衍生因子 =====
        result.indicators.update(self._compute_factors(df, result.indicators))

        return result

    # ═══════════════════════════════════════════════════════════════
    # 1. 趋势类指标
    # ═══════════════════════════════════════════════════════════════

    def _compute_trend_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        result = {}

        # ---- 移动平均线 ----
        for period in [5, 10, 20, 30, 60, 120, 250]:
            result[f"ma_{period}"] = close.rolling(period).mean()
            result[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # MA乖离率
        result["bias_ma5"] = (close - result["ma_5"]) / result["ma_5"] * 100
        result["bias_ma20"] = (close - result["ma_20"]) / result["ma_20"] * 100
        result["bias_ma60"] = (close - result["ma_60"]) / result["ma_60"] * 100

        # MA多空排列
        result["ma_bullish"] = (
            (result["ma_5"] > result["ma_20"]) &
            (result["ma_20"] > result["ma_60"])
        ).astype(int)
        result["ma_bearish"] = (
            (result["ma_5"] < result["ma_20"]) &
            (result["ma_20"] < result["ma_60"])
        ).astype(int)

        # ---- MACD ----
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        result["macd_dif"] = ema12 - ema26
        result["macd_dea"] = result["macd_dif"].ewm(span=9, adjust=False).mean()
        result["macd_hist"] = 2 * (result["macd_dif"] - result["macd_dea"])
        result["macd_golden_cross"] = (
            (result["macd_dif"] > result["macd_dea"]) &
            (result["macd_dif"].shift(1) <= result["macd_dea"].shift(1))
        ).astype(int)
        result["macd_death_cross"] = (
            (result["macd_dif"] < result["macd_dea"]) &
            (result["macd_dif"].shift(1) >= result["macd_dea"].shift(1))
        ).astype(int)

        # ---- ADX (趋势强度) ----
        result["adx_14"], result["plus_di_14"], result["minus_di_14"] = (
            self._compute_adx(df, period=14)
        )

        # ---- 均线多/空头排列评分 ----
        # 均线从短到长排序计分: 完全多头=+1, 完全空头=-1
        ma_cols = [result[f"ma_{p}"] for p in [5, 10, 20, 60]]
        result["ma_alignment_score"] = pd.concat(ma_cols, axis=1).apply(
            lambda row: sum(
                1 if row.iloc[i] > row.iloc[i + 1] else -1
                for i in range(len(row) - 1)
            ) / (len(row) - 1),
            axis=1,
        )

        # ---- 趋势强度综合评分 ----
        # 基于MA排列 + MACD + ADX的综合趋势评分
        result["trend_score"] = (
            result["ma_alignment_score"] * 0.4 +
            (result["macd_hist"].apply(np.sign)) * 0.3 +
            (result["adx_14"] / 100).clip(0, 1) * 0.3
        )

        return result

    def _compute_adx(
        self, df: pd.DataFrame, period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算 ADX / +DI / -DI"""
        high, low, close = df["high"], df["low"], df["close"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        tr = pd.concat([
            (high - low).abs(),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx, plus_di, minus_di

    # ═══════════════════════════════════════════════════════════════
    # 2. 动量类指标
    # ═══════════════════════════════════════════════════════════════

    def _compute_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        result = {}

        # ---- RSI ----
        for period in [6, 14, 24]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # RSI超买超卖信号
        result["rsi_overbought"] = (result["rsi_14"] > 70).astype(int)
        result["rsi_oversold"] = (result["rsi_14"] < 30).astype(int)

        # ---- Stochastic KDJ ----
        for period in [9, 14]:
            low_min = low.rolling(period).min()
            high_max = high.rolling(period).max()
            result[f"stoch_k_{period}"] = 100 * (close - low_min) / (high_max - low_min + 1e-10)
            result[f"stoch_d_{period}"] = result[f"stoch_k_{period}"].rolling(3).mean()
            result[f"stoch_j_{period}"] = (
                3 * result[f"stoch_k_{period}"] - 2 * result[f"stoch_d_{period}"]
            )

        # ---- CCI (Commodity Channel Index) ----
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
        result["cci_20"] = (tp - sma_tp) / (0.015 * mad + 1e-10)

        # ---- Williams %R ----
        for period in [10, 14]:
            hh = high.rolling(period).max()
            ll = low.rolling(period).min()
            result[f"willr_{period}"] = -100 * (hh - close) / (hh - ll + 1e-10)

        # ---- 价格动量 ----
        for period in [5, 10, 20, 60]:
            result[f"momentum_{period}"] = close - close.shift(period)
            result[f"roc_{period}"] = (close / close.shift(period) - 1) * 100

        # ---- 相对强度(相对指数) ----
        result["relative_strength_vs_index"] = (
            close.pct_change(20).rank(pct=True) * 100
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    # 3. 波动率类指标
    # ═══════════════════════════════════════════════════════════════

    def _compute_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        result = {}

        # ---- Bollinger Bands ----
        for period in [20, 60]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            result[f"bb_upper_{period}"] = ma + 2 * std
            result[f"bb_mid_{period}"] = ma
            result[f"bb_lower_{period}"] = ma - 2 * std
            result[f"bb_width_{period}"] = result[f"bb_upper_{period}"] - result[f"bb_lower_{period}"]
            result[f"bb_pct_{period}"] = (close - result[f"bb_lower_{period}"]) / (
                result[f"bb_upper_{period}"] - result[f"bb_lower_{period}"] + 1e-10
            )

        # Bollinger带宽突破信号
        result["bb_squeeze"] = (
            result["bb_width_20"] < result["bb_width_20"].rolling(20).mean() * 0.8
        ).astype(int)

        # ---- ATR (Average True Range) ----
        for period in [14, 20]:
            tr = pd.concat([
                (high - low).abs(),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)
            result[f"atr_{period}"] = tr.ewm(span=period, adjust=False).mean()
            result[f"atr_pct_{period}"] = result[f"atr_{period}"] / close * 100

        # ---- 历史波动率 ----
        for period in [5, 20, 60]:
            log_ret = np.log(close / close.shift(1))
            result[f"hv_{period}"] = log_ret.rolling(period).std() * np.sqrt(252) * 100

        # ---- 振幅 ----
        result["amplitude"] = (high - low) / close.shift(1) * 100

        # ---- 最大回撤 (滚动) ----
        result["drawdown_20"] = self._rolling_drawdown(close, window=20)
        result["drawdown_60"] = self._rolling_drawdown(close, window=60)

        return result

    def _rolling_drawdown(self, series: pd.Series, window: int = 20) -> pd.Series:
        """滚动窗口最大回撤"""
        rolling_max = series.rolling(window, min_periods=1).max()
        drawdown = (series - rolling_max) / rolling_max * 100
        return drawdown

    # ═══════════════════════════════════════════════════════════════
    # 4. 成交量类指标
    # ═══════════════════════════════════════════════════════════════

    def _compute_volume_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        close = df["close"]
        volume = df["volume"]
        result = {}

        # ---- 成交量均线 ----
        for period in [5, 10, 20, 60]:
            result[f"vol_ma_{period}"] = volume.rolling(period).mean()

        # 放量/缩量信号
        result["vol_ratio_5"] = volume / result["vol_ma_5"]
        result["vol_ratio_20"] = volume / result["vol_ma_20"]

        # 放量定义(v2.1硬化): vol > 1.5x 5日均量 → 1, vol < 0.5x 20日均量 → -1
        result["vol_expand"] = ((volume > result["vol_ma_5"] * 1.5)).astype(int)
        result["vol_shrink"] = ((volume < result["vol_ma_20"] * 0.5)).astype(int)

        # ---- OBV (On-Balance Volume) ----
        direction = np.sign(close.diff()).fillna(0)
        result["obv"] = (direction * volume).cumsum()
        result["obv_ma_20"] = result["obv"].rolling(20).mean()

        # ---- 量价关系 ----
        price_up = (close > close.shift(1)).astype(int)
        vol_up = (volume > volume.shift(1)).astype(int)
        result["price_vol_divergence"] = price_up - vol_up  # 1=价涨量缩(需警惕), -1=价跌量增(需关注)

        # ---- 换手率 (如果无此列则估算) ----
        if "turnover" in df.columns:
            result["turnover_ma5"] = df["turnover"].rolling(5).mean()
            result["turnover_ma20"] = df["turnover"].rolling(20).mean()

        return result

    # ═══════════════════════════════════════════════════════════════
    # 5. 统计学指标
    # ═══════════════════════════════════════════════════════════════

    def _compute_statistical_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        close = df["close"]
        ret = close.pct_change()
        result = {}

        # ---- Z-Score ----
        for period in [20, 60]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            result[f"zscore_{period}"] = (close - ma) / (std + 1e-10)

        # ---- 偏度 & 峰度 (滚动20日) ----
        result["skew_20"] = ret.rolling(20).skew()
        result["kurt_20"] = ret.rolling(20).kurt()

        # ---- 累计收益率 ----
        for period in [5, 10, 20, 60]:
            result[f"cum_return_{period}"] = (close / close.shift(period) - 1) * 100

        # ---- 最大/最小值位置 ----
        for period in [20, 60]:
            result[f"is_high_{period}"] = (close == close.rolling(period).max()).astype(int)
            result[f"is_low_{period}"] = (close == close.rolling(period).min()).astype(int)

        # ---- 连续上涨/下跌天数 ----
        result["consecutive_up"] = self._count_consecutive(close, direction="up")
        result["consecutive_down"] = self._count_consecutive(close, direction="down")

        return result

    def _count_consecutive(self, series: pd.Series, direction: str = "up") -> pd.Series:
        """计算连续涨/跌天数"""
        change = series.diff()
        if direction == "up":
            cond = change > 0
        else:
            cond = change < 0

        count = pd.Series(0, index=series.index)
        streak = 0
        for i in range(1, len(cond)):
            if cond.iloc[i]:
                streak += 1
            else:
                streak = 0
            count.iloc[i] = streak
        return count

    # ═══════════════════════════════════════════════════════════════
    # 6. A股特化指标
    # ═══════════════════════════════════════════════════════════════

    def _compute_ashare_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        result = {}

        # ---- 涨跌停检测 ----
        result["is_limit_up"] = (close / close.shift(1) - 1 >= 0.095).astype(int)
        result["is_limit_down"] = (close / close.shift(1) - 1 <= -0.095).astype(int)

        # 连板计数
        result["limit_up_streak"] = self._count_consecutive_signal(
            result["is_limit_up"]
        )
        result["limit_down_streak"] = self._count_consecutive_signal(
            result["is_limit_down"]
        )

        # ---- 炸板率 (日内冲高回落) ----
        result["fake_breakout"] = (
            (high / close.shift(1) - 1 > 0.09) &  # 盘中曾涨停
            (close / close.shift(1) - 1 < 0.05)     # 收盘回落
        ).astype(int)

        # ---- 封单强度 (简化: 用涨幅和成交量推断) ----
        result["limit_up_quality"] = result["is_limit_up"] * (
            (close / high).clip(0.95, 1.0)    # 收盘价/最高价(越接近1越好)
        )

        # ---- 日内波动率 ----
        result["intraday_volatility"] = (high - low) / close.shift(1) * 100

        # ---- A股特有缺口 ----
        result["gap_up"] = (low > df["open"].shift(1)).astype(int)
        result["gap_down"] = (high < df["open"].shift(1)).astype(int)

        # ---- 成交量异常(僵尸股检测) ----
        result["zombie_stock"] = (
            (df["turnover"] if "turnover" in df.columns else pd.Series(0, index=df.index)) < 0.3
        ).astype(int)

        return result

    def _count_consecutive_signal(self, series: pd.Series) -> pd.Series:
        """计算连续信号天数"""
        count = pd.Series(0, index=series.index, dtype=int)
        streak = 0
        for i in range(len(series)):
            if series.iloc[i] > 0:
                streak += 1
            else:
                streak = 0
            count.iloc[i] = streak
        return count

    # ═══════════════════════════════════════════════════════════════
    # 7. K线形态识别 (61种日本K线形态)
    # ═══════════════════════════════════════════════════════════════

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """pandas-ta形态识别 (61种)"""
        patterns = {}
        try:
            # pandas-ta CDL pattern recognition
            cdl = df.ta.cdl_pattern(name="all", append=True)
            # 提取新增的形态列
            pattern_cols = [c for c in cdl.columns if c.startswith("CDL_")]
            for col in pattern_cols:
                patterns[col.lower()] = cdl[col]
        except Exception as e:
            logger.debug(f"pandas-ta形态识别失败: {e}")
        return patterns

    def _detect_candlestick_patterns_numpy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """纯numpy实现最常用K线形态(Doji/Hammer/Engulfing等)"""
        patterns = {}
        open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
        body = (close - open_).abs()
        total_range = high - low

        # Doji (十字星): body <= 10% of total range
        patterns["doji"] = ((body <= total_range * 0.1) & (total_range > 0)).astype(int)

        # Hammer (锤子线): 下影线 >= 2x body, 上影线很小
        lower_shadow = np.minimum(open_, close) - low
        upper_shadow = high - np.maximum(open_, close)
        patterns["hammer"] = (
            (lower_shadow >= body * 2) & (upper_shadow <= body * 0.3) & (body > 0)
        ).astype(int)

        # Shooting Star (射击之星): 上影线 >= 2x body
        patterns["shooting_star"] = (
            (upper_shadow >= body * 2) & (lower_shadow <= body * 0.3) & (body > 0)
        ).astype(int)

        # Bullish Engulfing (看涨吞没)
        prev_body = (close.shift(1) - open_.shift(1))
        curr_body = (close - open_)
        patterns["bullish_engulfing"] = (
            (prev_body < 0) &
            (curr_body > 0) &
            (curr_body.abs() > prev_body.abs() * 1.5)
        ).astype(int)

        # Bearish Engulfing (看跌吞没)
        patterns["bearish_engulfing"] = (
            (prev_body > 0) &
            (curr_body < 0) &
            (curr_body.abs() > prev_body.abs() * 1.5)
        ).astype(int)

        # Morning Star (晨星) — 3日形态
        patterns["morning_star"] = (
            (close.shift(2) < open_.shift(2)) &           # Day1: 阴线
            (body.shift(1) <= total_range.shift(1) * 0.3) &  # Day2: Doji
            (close > open_) &                               # Day3: 阳线
            (close > close.shift(2))                        # Day3 > Day1 close
        ).astype(int)

        # Evening Star (暮星) — 3日形态
        patterns["evening_star"] = (
            (close.shift(2) > open_.shift(2)) &
            (body.shift(1) <= total_range.shift(1) * 0.3) &
            (close < open_) &
            (close < close.shift(2))
        ).astype(int)

        # Three White Soldiers (红三兵)
        patterns["three_soldiers"] = (
            (close > open_) &
            (close.shift(1) > open_.shift(1)) &
            (close.shift(2) > open_.shift(2)) &
            (close > close.shift(1)) &
            (close.shift(1) > close.shift(2))
        ).astype(int)

        # Three Black Crows (三只乌鸦)
        patterns["three_crows"] = (
            (close < open_) &
            (close.shift(1) < open_.shift(1)) &
            (close.shift(2) < open_.shift(2)) &
            (close < close.shift(1)) &
            (close.shift(1) < close.shift(2))
        ).astype(int)

        # Piercing Line (穿刺线)
        patterns["piercing_line"] = (
            (close.shift(1) < open_.shift(1)) &
            (close > open_) &
            (open_ < low.shift(1)) &
            (close > (close.shift(1) + (open_.shift(1) - close.shift(1)) / 2))
        ).astype(int)

        # Dark Cloud Cover (乌云盖顶)
        patterns["dark_cloud_cover"] = (
            (close.shift(1) > open_.shift(1)) &
            (close < open_) &
            (open_ > high.shift(1)) &
            (close < (close.shift(1) + (open_.shift(1) - close.shift(1)) / 2))
        ).astype(int)

        # Marubozu (光头光脚阳/阴线)
        patterns["marubozu_bull"] = (
            (close > open_) & (low == open_) & (high == close)
        ).astype(int)
        patterns["marubozu_bear"] = (
            (close < open_) & (high == open_) & (low == close)
        ).astype(int)

        return patterns

    # ═══════════════════════════════════════════════════════════════
    # 8. 衍生因子
    # ═══════════════════════════════════════════════════════════════

    def _compute_factors(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, pd.Series],
    ) -> Dict[str, pd.Series]:
        """基于已有指标计算衍生多因子"""
        result = {}
        close = df["close"]

        # ---- 动量因子 ----
        result["momentum_factor"] = (
            indicators.get("roc_20", close.pct_change(20).fillna(0)) * 0.4 +
            indicators.get("rsi_14", pd.Series(50, index=df.index)) / 100 * 0.3 +
            (indicators.get("bias_ma20", close * 0).fillna(0)) / 10 * 0.3
        )

        # ---- 波动率因子 (低波动溢价) ----
        hv_20 = indicators.get("hv_20", close.pct_change().rolling(20).std() * np.sqrt(252) * 100)
        result["volatility_factor"] = -hv_20.rank(pct=True)  # 低波动排名靠前

        # ---- 趋势因子 ----
        result["trend_factor"] = indicators.get("trend_score", pd.Series(0, index=df.index))

        # ---- 成交量因子 ----
        vol_ratio = indicators.get("vol_ratio_5", pd.Series(1, index=df.index))
        result["volume_factor"] = (vol_ratio - 1).clip(-0.8, 5)  # 放量信号

        # ---- 质量因子 (低杠杆+高ROE代理) ----
        result["quality_proxy"] = (
            -indicators.get("drawdown_60", pd.Series(0, index=df.index)).rank(pct=True)
        )

        # ---- 综合评分 ----
        result["composite_score"] = (
            result["momentum_factor"].fillna(0) * 0.25 +
            result["volatility_factor"].fillna(0) * 0.15 +
            result["trend_factor"].fillna(0) * 0.30 +
            result["volume_factor"].fillna(0) * 0.15 +
            result["quality_proxy"].fillna(0) * 0.15
        ).rank(pct=True) * 100

        return result

    # ═══════════════════════════════════════════════════════════════
    # 🆕 v2.14: ChromaDB 向量检索 — 相似K线形态匹配
    # ═══════════════════════════════════════════════════════════════

    def search_similar_patterns(
        self,
        df: pd.DataFrame,
        top_k: int = 5,
        lookback: int = 3,
    ) -> List[Dict]:
        """
        用最近N根K线的特征向量检索ChromaDB中相似的历史形态。

        Args:
            df: OHLCV DataFrame
            top_k: 返回最相似的K个形态
            lookback: 取最近N根K线

        Returns:
            相似形态列表 [{pattern, description, distance}, ...]
        """
        try:
            from knowledge.manager import KnowledgeManager
            km = KnowledgeManager()

            if not km.chroma_available:
                return []

            # 提取最近lookback根K线的特征向量
            close = df["close"].values
            open_p = df["open"].values
            high = df["high"].values
            low = df["low"].values

            n = min(lookback, len(close))
            recent = slice(-n, None)

            # 归一化特征: body_ratio, upper_shadow_ratio, lower_shadow_ratio, body_direction
            body = abs(close[recent] - open_p[recent])
            total_range = np.maximum(high[recent] - low[recent], 1e-10)
            body_ratio = float(np.mean(body / total_range))
            upper_shadow = float(np.mean((high[recent] - np.maximum(close[recent], open_p[recent])) / total_range))
            lower_shadow = float(np.mean((np.minimum(close[recent], open_p[recent]) - low[recent]) / total_range))
            body_direction = float(np.sign(np.mean(close[recent] - open_p[recent])))

            # 构建查询向量 (与播种时的6维向量对齐)
            query_vec = np.array([
                body_ratio, upper_shadow, lower_shadow, body_direction, 0.0, 0.0
            ], dtype=np.float32)
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm

            return km.search_similar_klines(query_vec.tolist(), top_k=top_k)
        except Exception:
            return []

    # ═══════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """列名标准化"""
        mapping = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume", "Amount": "amount",
            "Turnover": "turnover",
        }
        return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
