"""纯函数边界回归；固定输入仅用于离线单测，不参与行情获取或研究。"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from analysis.crowding import crowding_penalty
from analysis.market_structure import market_structure, market_structure_series
from data.processors.cleaning import clean_ohlcv


def test_clean_ohlcv_does_not_backfill_leading_missing_prices():
    """首个有效价格以前保持缺失，不能把未来价格填到历史。"""
    frame = pd.DataFrame({"close": ["bad", "0", "-1", "12.5", None]})
    original = frame.copy(deep=True)

    result = clean_ohlcv(frame)

    assert result["close"].iloc[:3].isna().all()
    assert result["close"].iloc[3:].tolist() == [12.5, 12.5]
    assert result["is_trade"].tolist() == [0, 0, 0, 1, 0]
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("column", ["volume", "amount", "turnover", "turn"])
def test_clean_ohlcv_preserves_zero_activity_but_discards_invalid_activity(column):
    frame = pd.DataFrame({
        "close": [10.0] * 4,
        column: ["0", "-1", "bad", "12.5"],
    })

    result = clean_ohlcv(frame)

    assert result[column].iloc[0] == 0.0
    assert result[column].iloc[1:3].isna().all()
    assert result[column].iloc[3] == 12.5


def test_clean_ohlcv_preserves_explicit_trade_flags():
    """成交状态来自上游时，保留停牌标注，即使该行已含沿用价格。"""
    frame = pd.DataFrame({"close": [10.0, 10.0], "is_trade": [1, 0]})

    result = clean_ohlcv(frame)

    assert_frame_equal(result, frame)
    assert result is not frame


def test_clean_ohlcv_sorts_dates_drops_invalid_dates_and_keeps_last_duplicate():
    frame = pd.DataFrame({
        "date": ["2026-01-03", "bad", "2026-01-02", "2026-01-02"],
        "close": [13.0, 99.0, 11.0, 12.0],
    })
    original = frame.copy(deep=True)

    result = clean_ohlcv(frame)

    assert result["date"].tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    assert result["close"].tolist() == [12.0, 13.0]
    assert result["date"].is_unique
    assert_frame_equal(frame, original)


def test_clean_ohlcv_empty_frame_preserves_schema_and_returns_copy():
    frame = pd.DataFrame({"close": pd.Series(dtype="float64")})

    result = clean_ohlcv(frame)

    assert_frame_equal(result, frame)
    assert result is not frame
    assert clean_ohlcv(None) is None


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ([1.0] * 49, "震荡"),
        ([1.0] * 50, "轮动普涨"),
        ([1.0] * 49 + [None, "invalid"], "震荡"),
        ([1.0] * 55 + [-0.5] * 45, "震荡"),
        ([1.0] * 56 + [-0.5] * 44, "轮动普涨"),
        ([2.0] * 10 + [-0.5] * 90, "震荡"),
        ([2.01] * 10 + [-0.5] * 90, "抱团动量"),
        ([3.0] * 45 + [-0.5] * 55, "震荡"),
        ([3.0] * 44 + [-0.5] * 56, "抱团动量"),
        ([1.0] * 30 + [-1.0] * 70, "震荡"),
        ([1.0] * 30 + [-1.01] * 70, "熊"),
    ],
    ids=[
        "insufficient-sample", "minimum-sample", "invalid-values-excluded",
        "breadth-at-threshold", "breadth-above-threshold",
        "leader-at-threshold", "leader-above-threshold",
        "concentration-at-threshold", "concentration-below-threshold",
        "bear-median-at-threshold", "bear-median-below-threshold",
    ],
)
def test_market_structure_sample_and_strict_threshold_boundaries(changes, expected):
    frame = pd.DataFrame({"pct_change": changes})
    original = frame.copy(deep=True)

    assert market_structure(frame) == expected
    assert_frame_equal(frame, original)


def test_market_structure_missing_change_column_is_unclassified():
    assert market_structure(pd.DataFrame({"close": [10.0] * 100})) == "震荡"


def test_market_structure_series_sorts_dates_and_only_votes_on_past_window():
    bull = pd.DataFrame({"pct_change": [1.0] * 50})
    bear = pd.DataFrame({"pct_change": [-2.0] * 50})
    sections = {
        "2026-01-05": bear,
        "2026-01-01": bull,
        "2026-01-03": bull,
        "2026-01-04": bear,
        "2026-01-02": bull,
    }

    result = market_structure_series(sections, window=3)

    assert list(result) == sorted(sections)
    assert list(result.values()) == ["轮动普涨"] * 4 + ["熊"]
    prefix = {day: frame for day, frame in sections.items() if day <= "2026-01-03"}
    assert market_structure_series(prefix, window=3) == {
        day: result[day] for day in sorted(prefix)
    }
    assert market_structure_series({}, window=3) == {}


def test_crowding_penalty_preserves_symbol_index_and_clips_outliers():
    frame = pd.DataFrame(
        {"turn_pct_60d": ["0.97", "0.985", "1.0", "1.2", "bad", None, "-0.1"]},
        index=pd.Index(["at", "half", "full", "above", "invalid", "missing", "below"], name="symbol"),
    )
    original = frame.copy(deep=True)

    result = crowding_penalty(frame)

    expected = pd.Series([0.0, -4.0, -8.0, -8.0, 0.0, 0.0, 0.0], index=frame.index)
    assert_series_equal(result, expected, atol=1e-12, rtol=0)
    assert_frame_equal(frame, original)


def test_crowding_penalty_respects_configured_threshold_and_limit():
    frame = pd.DataFrame({"turn_pct_60d": [0.8, 0.9, 1.0]})

    result = crowding_penalty(frame, extreme_thr=0.8, max_penalty=4.0)

    assert result.tolist() == pytest.approx([0.0, -2.0, -4.0])


def test_crowding_penalty_missing_column_preserves_nondefault_index():
    frame = pd.DataFrame(index=pd.Index([7, 3], name="position"))

    assert_series_equal(crowding_penalty(frame), pd.Series(0.0, index=frame.index))
