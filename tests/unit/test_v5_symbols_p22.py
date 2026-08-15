"""v5.6 P2-2 代码→交易所前缀单一来源 + 最近交易日

覆盖:
  - data.symbols.market_prefix / to_symbol 补北交所 bj (8/4/920)
  - timeutil.last_trade_date / is_trading_day (周末 + 节假日回退)
"""
from datetime import date

from data.symbols import market_prefix, to_symbol
from timeutil import is_trading_day, last_trade_date


# ═══════════════════════════════════════════════════════════════
# 代码 → 交易所前缀
# ═══════════════════════════════════════════════════════════════

def test_market_prefix_sh_sz_bj():
    assert market_prefix("600519") == "sh"   # 上交所主板
    assert market_prefix("688111") == "sh"   # 科创板
    assert market_prefix("000858") == "sz"   # 深主板
    assert market_prefix("300750") == "sz"   # 创业板
    assert market_prefix("830001") == "bj"   # 北交所 8xxxxx
    assert market_prefix("430047") == "bj"   # 北交所 4xxxxx
    assert market_prefix("920001") == "bj"   # 北交所新码段


def test_to_symbol_idempotent_and_bare():
    assert to_symbol("sh.600519") == "sh.600519"   # 幂等
    assert to_symbol("600519") == "sh.600519"
    assert to_symbol("000858") == "sz.000858"
    assert to_symbol("830001") == "bj.830001"      # 关键: 不再丢北交所
    assert to_symbol("bj.830001") == "bj.830001"


# ═══════════════════════════════════════════════════════════════
# 最近交易日
# ═══════════════════════════════════════════════════════════════

def test_last_trade_date_skips_weekend():
    # 2026-08-15 周六 → 回退到周五 08-14
    assert is_trading_day(date(2026, 8, 14)) is True
    assert is_trading_day(date(2026, 8, 15)) is False
    assert last_trade_date(date(2026, 8, 15)) == date(2026, 8, 14)


def test_last_trade_date_skips_holiday():
    # 2026-10-01 国庆休市 → 回退到 09-30
    assert is_trading_day(date(2026, 10, 1)) is False
    assert last_trade_date(date(2026, 10, 1)) == date(2026, 9, 30)


def test_last_trade_date_returns_same_day_when_trading():
    assert last_trade_date(date(2026, 8, 13)) == date(2026, 8, 13)
