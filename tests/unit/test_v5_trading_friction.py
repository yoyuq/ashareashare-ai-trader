"""v5.6 纸盘面 — 回放真实成交模型单测.

覆盖: 滑点 / 费用(佣金·印花·过户) / 涨跌停成交拦截 / 流动性上限 / 冲击成本.
目的: 回放不再对成交偏乐观, 反映真实交易所受摩擦.
"""

import json

import pytest


# ═══════════════════════════════════════════════════════════════
# 费用计算
# ═══════════════════════════════════════════════════════════════

from scripts.historical_replay import (
    DEFAULT_SLIPPAGE_BPS, MIN_COMMISSION, STAMP_DUTY_RATE, _buy_fee, _sell_fee,
)


def test_buy_fee_min_commission():
    """小额买入: 佣金取最低 ¥5 + 过户费."""
    # 佣金 max(amt*0.0003, 5) = 5 for small amt; 过户费 = amt*0.00001
    fee = _buy_fee(10000)
    assert fee == pytest.approx(5.0 + 10000 * 0.00001, abs=1e-6)
    assert fee >= 5.0


def test_buy_fee_scales_above_min():
    """大额买入: 佣金按比例, 高于最低佣金."""
    fee = _buy_fee(5_000_000)  # 佣金 1500, 无最低限制
    assert fee == pytest.approx(5_000_000 * 0.0003 + 5_000_000 * 0.00001, abs=1e-4)


def test_sell_fee_has_stamp_duty():
    """卖出含印花税 (0.05%), 买入无."""
    sell = _sell_fee(10000)
    buy = _buy_fee(10000)
    assert sell > buy
    diff = sell - buy
    assert diff == pytest.approx(10000 * STAMP_DUTY_RATE, abs=1e-6)


# ═══════════════════════════════════════════════════════════════
# 涨跌停成交拦截
# ═══════════════════════════════════════════════════════════════

from scripts.historical_replay import _limit_blocked


def test_limit_blocked_buy_at_limit_up():
    """主板涨停 (pctChg>=9.8) → 买入被拒 (买不进封死股)."""
    assert _limit_blocked("buy", 9.9, "600519") is True
    assert _limit_blocked("buy", 9.8, "600519") is True


def test_limit_blocked_sell_at_limit_down():
    """主板跌停 (pctChg<=-9.8) → 卖出被拒 (卖不出)."""
    assert _limit_blocked("sell", -9.9, "600519") is True


def test_limit_blocked_mid_allowed():
    """非涨跌停 → 允许成交."""
    assert _limit_blocked("buy", 3.0, "600519") is False
    assert _limit_blocked("sell", -3.0, "600519") is False


def test_limit_blocked_chinext_20pct():
    """创业板涨停阈值 20% (pctChg>=19.8 才拒)."""
    assert _limit_blocked("buy", 15.0, "300750") is False
    assert _limit_blocked("buy", 19.9, "300750") is True


def test_limit_blocked_none_or_bad():
    """缺 pctChg / 非数字 → 不拦截 (不回退到错误拒绝)."""
    assert _limit_blocked("buy", None, "600519") is False
    assert _limit_blocked("sell", "abc", "600519") is False
    assert _limit_blocked("buy", 9.9, None) is False


# ═══════════════════════════════════════════════════════════════
# 冲击 / 流动性
# ═══════════════════════════════════════════════════════════════

from scripts.historical_replay import _impact_bps, IMPACT_CAP_PCT


def test_impact_zero_tiny_order():
    """极小单 (占比低) → 无冲击."""
    assert _impact_bps(1000, 100_000_000) == 0.0


def test_impact_grows_with_ratio():
    """占比越大冲击越高."""
    small = _impact_bps(100_000, 100_000_000)   # 0.1%
    large = _impact_bps(1_000_000, 100_000_000) # 1%
    assert large > small > 0


def test_impact_capped():
    """冲击有上限 (防极端值)."""
    assert _impact_bps(1e9, 1e6) <= 50.0


def test_liquidity_cap_constant():
    """流动性上限 = 1% × 当日成交额."""
    assert IMPACT_CAP_PCT == 0.01


# ═══════════════════════════════════════════════════════════════
# ReplayPortfolio 成交完整性 (滑点+费用+现金正确)
# ═══════════════════════════════════════════════════════════════

from scripts.historical_replay import ReplayPortfolio


def test_buy_applies_slippage_and_fee():
    """买入: 成交价=基准×(1+滑点), 现金扣除(成交额+费用), 费用累计."""
    pf = ReplayPortfolio(capital=100000.0, slippage_bps=100, fees=True)  # 1% 滑点便于断言
    ok = pf.buy("sh.600519", "茅台", 100.0, 100, "2026-08-09")
    assert ok
    # 成交价 = 100 * 1.01 = 101; 成交额 10100; 费用 = 佣金5 + 过户0.101
    fee = _buy_fee(10100)
    assert pf.cash == pytest.approx(100000 - 10100 - fee, abs=1e-6)
    assert pf.fees_paid == pytest.approx(fee, abs=1e-6)
    assert pf.positions["sh.600519"]["entry_price"] == pytest.approx(101.0, abs=1e-6)


def test_sell_applies_slippage_and_fee():
    """卖出: 成交价=基准×(1-滑点), 现金增加(成交额-费用)."""
    pf = ReplayPortfolio(capital=100000.0, slippage_bps=100, fees=True)
    pf.buy("sh.600519", "茅台", 100.0, 100, "2026-08-09")
    pf.sell("sh.600519", 110.0, "2026-08-10")
    # 卖出价 = 110 * 0.99 = 108.9; 成交额 10890; 费用含印花税
    sell_fee = _sell_fee(10890)
    assert pf.cash == pytest.approx(100000 - (10100 + _buy_fee(10100)) + 10890 - sell_fee, abs=1e-6)


def test_zero_slippage_no_fee_keeps_float_cash():
    """关闭摩擦时: 现金保持 Python float, 快照可 JSON 序列化 (float32 回归防护)."""
    pf = ReplayPortfolio(capital=100000.0, slippage_bps=0, fees=False)
    pf.buy("sh.600519", "茅台", 10.0, 100, "2026-08-09")
    assert type(pf.cash) is float
    snap = [{"symbol": s, "price": float(p["entry_price"]), "qty": int(p["qty"]),
             "value": float(p["qty"] * p["entry_price"]), "weight": 0.5}
            for s, p in pf.positions.items()]
    json.dumps(snap)  # 不应抛 TypeError


def test_blocked_orders_counter():
    """被涨跌停/流动性拒的单会计入 blocked_orders."""
    pf = ReplayPortfolio(capital=100000.0)
    assert pf.blocked_orders == 0
    pf.blocked_orders += 1
    assert pf.blocked_orders == 1