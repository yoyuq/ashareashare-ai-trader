"""v5.6 P1-13 组合反事实: 提高阈值 + 同期基准对照 + 限制回流频次

覆盖:
  - 提高基础阈值 (0.05→0.25) 后, 小改善不再被误判 verified (消除同义反复)
  - 同期基准阈值写入结果, 改善须超过组合|收益|×VOL_SCALE 才 verified
  - drag_experiences 冷却期 + 单次条数上限, 限制后视偏差回流频次
"""

import pytest

from agent.evolution.portfolio_counterfactual import (
    PortfolioCounterfactualResult,
    StockContribution,
    accumulate_drag_history,
    drag_experiences,
    portfolio_level_counterfactual,
    IMPROVE_BASE, VOL_SCALE, DRAG_COOLDOWN_DAYS, MAX_DRAG_ITEMS,
)


def _snap(*items):
    return [{"symbol": s, "name": s, "qty": 100, "value": v, "weight": w}
            for s, v, w in items]


# ═══════════════════════════════════════════════════════════════
# 提高阈值 + 同期基准
# ═══════════════════════════════════════════════════════════════

def test_raised_threshold_rejects_small_improvement():
    """平盘日 + 最差票微跌 → 改善不足基础阈值 (0.25%), 不再同义反复 verified."""
    # A(+0.2%)/B(-0.2%) 各 50% → actual ≈ 0; 最差票 B 贡献 -0.1% → improvement 0.1%
    snap = _snap(("sh.A", 5000, 0.5), ("sh.B", 5000, 0.5))
    rets = {"sh.A": 0.2, "sh.B": -0.2}
    r = portfolio_level_counterfactual(snap, rets, date="2026-08-09")
    assert r is not None
    assert r.worst_stock.symbol == "sh.B"
    assert r.improvement_pct == pytest.approx(0.1)
    assert r.verified is False


def test_same_period_benchmark_threshold_exposed():
    """同期基准阈值写入结果, 改善须超过它才 verified."""
    snap = _snap(("sh.GOOD", 6000, 0.6), ("sh.BAD", 4000, 0.4))
    rets = {"sh.GOOD": 3.0, "sh.BAD": -5.0}
    r = portfolio_level_counterfactual(snap, rets, date="2026-08-09")
    # actual = 0.6*3 + 0.4*(-5) = -0.2; improvement = 2.0
    expected_thr = max(IMPROVE_BASE, abs(-0.2) * VOL_SCALE)
    assert r.benchmark_threshold_pct == pytest.approx(expected_thr)
    assert r.improvement_pct > r.benchmark_threshold_pct
    assert r.verified is True


def test_volatility_benchmark_scales_with_move():
    """大波动日同期基准抬高: 同样的 0.5pp 改善不足以通过."""
    snap = _snap(("sh.A", 9000, 0.9), ("sh.B", 1000, 0.1))
    rets = {"sh.A": 5.5, "sh.B": -5.0}  # actual ≈ 4.45, improvement 0.5
    r = portfolio_level_counterfactual(snap, rets, date="2026-08-09")
    assert r.benchmark_threshold_pct > IMPROVE_BASE  # 被同期基准抬高
    assert r.improvement_pct < r.benchmark_threshold_pct
    assert r.verified is False


# ═══════════════════════════════════════════════════════════════
# 限制回流频次
# ═══════════════════════════════════════════════════════════════

def _pcf_result(symbol, imp, verified=True):
    return PortfolioCounterfactualResult(
        date="2026-08-09", portfolio_return_pct=-1.0,
        counterfactual_return_pct=-1.0 + imp, improvement_pct=imp,
        worst_stock=StockContribution(symbol=symbol, name="测试"),
        worst_contribution_pct=-imp, n_positions=3, n_match=3, verified=verified,
    )


def test_drag_cooldown_prevents_daily_respill():
    """同一票回流后, 冷却期内不再重复回流 (防每日刷屏)."""
    h = accumulate_drag_history(
        {}, [_pcf_result("sz.000001", 0.5), _pcf_result("sz.000001", 0.3)], "2026-08-09")
    assert h["sz.000001"]["count"] == 2
    # 第一次回流: count=2 → 1 条
    assert len(drag_experiences(h, "2026-08-09")) == 1
    # 冷却期内 (间隔 < DRAG_COOLDOWN_DAYS) → 0 条
    assert drag_experiences(h, "2026-08-09") == []
    assert drag_experiences(h, "2026-08-10") == []
    # 冷却期过后 → 可再次回流
    assert len(drag_experiences(h, "2026-08-13")) == 1


def test_drag_max_items_cap():
    """单次回流条数上限 (MAX_DRAG_ITEMS)."""
    h = {}
    for i in range(8):
        h[f"sz.00000{i}"] = {"count": 5, "avg_improvement_pct": 0.5, "last_name": f"股{i}"}
    items = drag_experiences(h, "2026-08-09")
    assert len(items) == MAX_DRAG_ITEMS
