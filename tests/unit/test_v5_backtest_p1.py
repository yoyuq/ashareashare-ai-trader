"""v5.6 P1 回测层测试 (P1-10 涨跌停板块感知 / P1-12 费率单一来源)

覆盖:
  P1-10 简化回测器涨跌停按板块 (主板10/创业板科创20/北交所30) 感知
  P1-12 往返成本率单一来源 (broker 常量派生, 不再硬编码 0.0031)
"""

import numpy as np
import pandas as pd
import pytest

from backtest.broker import (
    AShareBroker,
    limit_pct_for_symbol,
    round_trip_cost_rate,
)
from analysis.optimized_strategies import _exec_prices as v2_exec
from analysis.strategies_v3 import _exec as v3_exec
from analysis.recommender import _exec_prices as v1_exec


# ═══════════════════════════════════════════════════════════════
# P1-10: 涨跌停板块感知
# ═══════════════════════════════════════════════════════════════

def test_limit_pct_for_symbol_boards():
    assert limit_pct_for_symbol("sh.600519") == 10.0   # 沪主板
    assert limit_pct_for_symbol("sz.000858") == 10.0   # 深主板
    assert limit_pct_for_symbol("sz.300750") == 20.0   # 创业板
    assert limit_pct_for_symbol("sh.688981") == 20.0   # 科创板
    assert limit_pct_for_symbol("bj.830799") == 30.0   # 北交所
    assert limit_pct_for_symbol("bj.920001") == 30.0   # 北交所 920 新前缀
    assert limit_pct_for_symbol("") == 10.0            # 空 symbol 默认主板


def _make_plus11_df():
    """构造: 前 3 日收盘 10, 第 4 日(索引3) +11% 涨停, 之后回落。

    信号在索引 i 日收盘产生, 于 i+1 日开盘执行 (entry_px[i] = open[i+1])。
    故 +11% 日(索引3) 的涨停状态影响的是 entry_px[2]。
    """
    return pd.DataFrame({
        "open":  [10.0, 10.0, 10.0, 11.0, 10.0, 10.0],
        "close": [10.0, 10.0, 10.0, 11.1, 10.0, 10.0],
        "high":  [10.0, 10.0, 10.0, 11.1, 10.0, 10.0],
        "low":   [10.0, 10.0, 10.0, 10.9, 10.0, 10.0],
    })


def test_v2_exec_board_aware_20pct_not_blocked():
    """创业板(+20%板块) 收盘+11% 不触及涨停 → 当日 entry 不应为 NaN"""
    entry, _ = v2_exec(_make_plus11_df(), symbol="sz.300750")
    assert not np.isnan(entry.iloc[3]), "20%板块 +11% 不应被判为涨停买不进"


def test_v2_exec_board_aware_main_blocked():
    """主板(+10%) 收盘+11% 触及涨停 → 当日 entry 应为 NaN"""
    entry, _ = v2_exec(_make_plus11_df(), symbol="sh.600519")
    assert np.isnan(entry.iloc[3]), "10%板块 +11% 应被判为涨停买不进"


def test_v3_exec_board_aware():
    entry20, _ = v3_exec(_make_plus11_df(), symbol="sz.300750")
    entry10, _ = v3_exec(_make_plus11_df(), symbol="sh.600519")
    assert not np.isnan(entry20.iloc[3])
    assert np.isnan(entry10.iloc[3])


def test_v1_exec_board_aware_sealed_floor():
    """v1 (recommender) 封板下限按板块: +11% 封死时, 主板挡而 20% 板块不挡"""
    df = _make_plus11_df()
    # 20% 板块 floor=19.3, +11% 不触发 → entry[2] 有效
    entry20, _ = v1_exec(df, symbol="sz.300750")
    assert not np.isnan(entry20[2])
    # 主板 floor=9.3, +11% 封板触发 → entry[2] = NaN
    entry10, _ = v1_exec(df, symbol="sh.600519")
    assert np.isnan(entry10[2])


# ═══════════════════════════════════════════════════════════════
# P1-12: 费率单一来源
# ═══════════════════════════════════════════════════════════════

def test_round_trip_cost_rate_matches_broker_constants():
    expected = (
        AShareBroker.COMMISSION_RATE * 2
        + AShareBroker.STAMP_DUTY_RATE
        + AShareBroker.TRANSFER_FEE_RATE * 2
        + AShareBroker.SLIPPAGE_PCT * 2
    )
    assert round_trip_cost_rate() == pytest.approx(expected)
    assert round_trip_cost_rate() == pytest.approx(0.00312)


def test_round_trip_cost_rate_respects_custom_slippage():
    assert round_trip_cost_rate(0.0) == pytest.approx(
        AShareBroker.COMMISSION_RATE * 2
        + AShareBroker.STAMP_DUTY_RATE
        + AShareBroker.TRANSFER_FEE_RATE * 2
    )
