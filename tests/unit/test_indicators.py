"""
单元测试: TechnicalAnalyzer 技术指标计算
"""
import numpy as np
import pandas as pd
import pytest

# 生成模拟OHLCV数据
def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1)
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close - rng.uniform(-0.5, 0.5, n)
    volume = rng.integers(100000, 1000000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })


class TestTechnicalAnalyzer:
    """TechnicalAnalyzer 单元测试"""

    @classmethod
    def setup_class(cls):
        from analysis.indicators import TechnicalAnalyzer
        cls.analyzer = TechnicalAnalyzer()
        cls.df = _make_ohlcv(200)

    def test_compute_all_returns_result(self):
        result = self.analyzer.compute_all(self.df, symbol="test")
        assert result is not None
        assert result.symbol == "test"

    def test_trend_indicators(self):
        result = self.analyzer.compute_all(self.df)
        df = result.to_dataframe()
        assert "ma_5" in df.columns
        assert "ma_20" in df.columns
        assert "ma_60" in df.columns
        assert "macd_dif" in df.columns
        assert "trend_score" in df.columns

    def test_momentum_indicators(self):
        result = self.analyzer.compute_all(self.df)
        df = result.to_dataframe()
        assert "rsi_14" in df.columns
        assert "stoch_k_9" in df.columns
        assert "cci_20" in df.columns

    def test_volatility_indicators(self):
        result = self.analyzer.compute_all(self.df)
        df = result.to_dataframe()
        assert "bb_upper_20" in df.columns
        assert "atr_14" in df.columns
        assert "hv_20" in df.columns

    def test_volume_indicators(self):
        result = self.analyzer.compute_all(self.df)
        df = result.to_dataframe()
        assert "vol_ratio_5" in df.columns
        assert "obv" in df.columns

    def test_composite_score_range(self):
        result = self.analyzer.compute_all(self.df)
        df = result.to_dataframe()
        score = df["composite_score"].iloc[-1]
        assert 0 <= score <= 100, f"composite_score {score} out of [0,100]"

    def test_no_nan_in_key_indicators(self):
        result = self.analyzer.compute_all(self.df)
        df = result.to_dataframe()
        key_cols = ["rsi_14", "trend_score", "composite_score"]
        # 最后一行(最新数据)不应有 NaN
        last = df[key_cols].iloc[-1]
        assert not last.isna().any(), f"NaN in last row: {last.to_dict()}"

    def test_empty_df_returns_empty(self):
        result = self.analyzer.compute_all(pd.DataFrame(), symbol="empty")
        df = result.to_dataframe()
        assert df.empty

    def test_skip_patterns_faster(self):
        """skip_patterns=True 应跳过形态识别"""
        result = self.analyzer.compute_all(self.df, skip_patterns=True)
        df = result.to_dataframe()
        # 关键指标应该仍在
        assert "rsi_14" in df.columns


class TestADX:
    """ADX 计算测试"""

    def test_adx_positive(self):
        from analysis.indicators import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        df = _make_ohlcv(100)
        adx, plus_di, minus_di = analyzer._compute_adx(df, period=14)
        assert len(adx) == len(df)
        assert (adx.dropna() >= 0).all()
        assert (adx.dropna() <= 100).all()


class TestPatternRecognition:
    """K线形态识别测试"""

    def test_doji_detection(self):
        """十字星检测"""
        from analysis.indicators import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        # 构造一个明确的十字星: open ≈ close, 有上下影线
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "open": [10.0, 10.5, 10.0, 10.0, 10.0],
            "high": [10.5, 11.0, 10.8, 10.6, 10.5],
            "low": [9.5, 10.0, 9.2, 9.4, 9.5],
            "close": [10.2, 10.0, 10.01, 10.05, 10.0],  # 最后一天: open=10, close=10, body≈0
            "volume": [100000, 200000, 150000, 180000, 120000],
        })
        patterns = analyzer._detect_candlestick_patterns_numpy(df)
        assert "doji" in patterns
        # 最后一天应该是十字星
        assert patterns["doji"].iloc[-1] == 1

    def test_hammer_detection(self):
        """锤子线检测"""
        from analysis.indicators import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "open": [10.0, 10.5, 10.0, 10.0, 10.3],
            "high": [10.5, 11.0, 10.8, 10.6, 10.5],
            "low": [9.5, 10.0, 9.2, 9.4, 9.0],      # 长下影线
            "close": [10.2, 10.0, 10.01, 10.05, 10.35],  # 收盘接近开盘, 小实体
            "volume": [100000, 200000, 150000, 180000, 120000],
        })
        patterns = analyzer._detect_candlestick_patterns_numpy(df)
        assert "hammer" in patterns

    def test_bullish_engulfing(self):
        """看涨吞没检测"""
        from analysis.indicators import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "open":  [10.5, 10.0, 10.0, 10.0, 9.5],   # day4: 阴线(open=10, close=9.5)
            "high":  [11.0, 10.5, 10.5, 10.5, 10.8],
            "low":   [10.0, 9.5,  9.5,  9.5,  9.3],
            "close": [10.2, 10.2, 10.1, 9.5,  10.7],   # day5: 阳线(open=9.5, close=10.7) 吞没day4
            "volume": [100000, 200000, 150000, 180000, 200000],
        })
        patterns = analyzer._detect_candlestick_patterns_numpy(df)
        assert "bullish_engulfing" in patterns
        assert patterns["bullish_engulfing"].iloc[-1] == 1


class TestMultiFrameAnalyzer:
    """多时间框架分析器测试"""

    @pytest.fixture
    def mf_analyzer(self):
        from analysis.multiframe import MultiFrameAnalyzer
        return MultiFrameAnalyzer()

    def test_analyze_with_daily_only(self, mf_analyzer):
        """仅日线数据分析"""
        df = _make_ohlcv(200)
        result = mf_analyzer.analyze("sh.600519", df)
        assert result.symbol == "sh.600519"
        assert result.daily is not None
        assert result.daily.trend in ("bullish", "bearish", "neutral")

    def test_auto_resample_weekly(self, mf_analyzer):
        """自动降采样生成周线"""
        df = _make_ohlcv(200)
        result = mf_analyzer.analyze("test", df, auto_resample=True)
        assert result.weekly is not None
        assert result.weekly.timeframe == "weekly"

    def test_resample_to_weekly(self, mf_analyzer):
        """日线→周线降采样"""
        df = _make_ohlcv(200)
        weekly = mf_analyzer._resample_to_weekly(df)
        assert len(weekly) > 0
        assert len(weekly) < len(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in weekly.columns

    def test_resample_to_monthly(self, mf_analyzer):
        """日线→月线降采样"""
        df = _make_ohlcv(300)
        monthly = mf_analyzer._resample_to_monthly(df)
        assert len(monthly) > 0
        assert len(monthly) < len(df) // 20  # 大约每月20个交易日

    def test_pivot_points(self, mf_analyzer):
        """枢轴点计算"""
        df = _make_ohlcv(200)
        result = mf_analyzer.analyze("test", df)
        assert "PP" in result.pivot_points
        assert "R1" in result.pivot_points
        assert "S1" in result.pivot_points
        # R1 > PP > S1
        assert result.pivot_points["R1"] >= result.pivot_points["PP"]
        assert result.pivot_points["PP"] >= result.pivot_points["S1"]

    def test_cross_timeframe_score(self, mf_analyzer):
        """跨时间框架评分"""
        df = _make_ohlcv(200)
        result = mf_analyzer.analyze("test", df, auto_resample=True)
        score = mf_analyzer.cross_timeframe_score(result)
        assert 0 <= score <= 10

    def test_false_breakout_detection(self, mf_analyzer):
        """虚假突破检测"""
        df = _make_ohlcv(200)
        is_false, detail = mf_analyzer.detect_false_breakout("test", df)
        assert isinstance(is_false, bool)
        assert isinstance(detail, str)
