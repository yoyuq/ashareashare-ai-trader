"""v3.0 低覆盖模块测试 — 补 daily_runner 市场状态 / knowledge 核心 API

CI 覆盖率快照: daily_runner 12%, knowledge/manager 33% — 本文件补核心纯逻辑。
"""

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# daily_runner._detect_regime (覆盖 12%)
# ═══════════════════════════════════════════════════════════════

def _regime_df(pcts):
    return pd.DataFrame({"pct_change": pcts})


def test_detect_regime_strong_bull():
    from simulation.daily_runner import _detect_regime
    r = _detect_regime(_regime_df([3.0, 4.0, 6.0, 2.5, 3.5]))
    assert r["regime"] == "strong_bull"
    assert r["label"] == "强牛"


def test_detect_regime_strong_bear():
    from simulation.daily_runner import _detect_regime
    r = _detect_regime(_regime_df([-3.0, -4.0, -2.0, -1.5, -3.0]))
    assert r["regime"] == "strong_bear"


def test_detect_regime_range_bound():
    from simulation.daily_runner import _detect_regime
    r = _detect_regime(_regime_df([0.1, -0.2, 0.3, -0.1, 0.2]))
    assert r["regime"] == "range_bound"


def test_detect_regime_crisis_on_extreme_down():
    """极端下跌占比 >10% → crisis (优先于均值判断)"""
    from simulation.daily_runner import _detect_regime
    r = _detect_regime(_regime_df([-6.0, -7.0, 0.5, -6.5, 0.3, -0.2, -6.0, 0.4, -5.5, 0.2]))
    assert r["regime"] == "crisis"


def test_detect_regime_missing_pct_col():
    """无 pct_change 列 → 安全默认 range_bound (不崩溃)"""
    from simulation.daily_runner import _detect_regime
    r = _detect_regime(pd.DataFrame({"close": [10, 11, 12]}))
    assert r["regime"] == "range_bound"


# ═══════════════════════════════════════════════════════════════
# v5.3: 指数趋势 regime (修复单日截面跳变 + 抱团牛误判熊
# ═══════════════════════════════════════════════════════════════

def _idx_series(final, n=30, trend=None):
    """构造一个指数收盘序列: 从 final 逆推, 末段带动量 trend."""
    if trend is None:
        # 默认: 平缓窄幅 (m25≈0, m5≈0)
        vals = [final] * n
    elif trend == "bull":
        # 25日日涨0.3% → m25≈+7.8%
        vals = [final / (1 + 0.003) ** i for i in range(n - 1, -1, -1)]
    elif trend == "bear":
        vals = [final * (1 + 0.003) ** i for i in range(n - 1, -1, -1)]
    else:
        vals = [final] * n
    return pd.Series(vals)


def test_detect_regime_index_bull_overrides_breadth():
    """指数25d强牛 → 即使今日平均跌, 仍判牛 (修复抱团牛误判熊)."""
    from simulation.daily_runner import _detect_regime
    # 今日截面: 平均 -0.5% (广度弱), 但指数25d动量 +7.8% (牛市)
    df = _regime_df([-0.5] * 8 + [0.2] * 2)
    idx = _idx_series(100, trend="bull")
    r = _detect_regime(df, index_close=idx)
    assert r["regime"] in ("strong_bull", "weak_bull")
    assert "index_m25" in r  # 带趋势字段


def test_detect_regime_index_bear_flips():
    """指数25d强熊 → 判熊 (修复反弹日误判强牛)."""
    from simulation.daily_runner import _detect_regime
    df = _regime_df([0.2] * 10)  # 今日平均上涨 (反弹日)
    idx = _idx_series(100, trend="bear")  # 25d -7.8%
    r = _detect_regime(df, index_close=idx)
    assert r["regime"] in ("strong_bear", "weak_bear")


def test_detect_regime_index_crisis_on_5d_crash():
    """指数5d崩 (>6%) → crisis (快速反应)."""
    from simulation.daily_runner import _detect_regime
    df = _regime_df([0.0] * 10)
    # 指数: 前25日平缓, 最后5日急跌 → m5 崩
    base = [100] * 25
    # 5日连跌2%: 末尾最低 (最后一天=90.39), m5 = (90.39/100-1) = -9.6%
    crash5 = [100 * (1 - 0.02) ** i for i in range(1, 6)]  # 98,96,94,92,90.4 (末位最低)
    idx = pd.Series(base + crash5)
    r = _detect_regime(df, index_close=idx)
    assert r["regime"] == "crisis"


def test_detect_regime_fallback_without_index_unchanged():
    """无指数 → 回退单日广度 (旧行为, 不回归)."""
    from simulation.daily_runner import _detect_regime
    r = _detect_regime(_regime_df([3.0, 4.0, 6.0, 2.5, 3.5]))
    assert r["regime"] == "strong_bull"
    assert "index_m25" not in r


# ═══════════════════════════════════════════════════════════════
# knowledge.manager 核心 API (覆盖 33%)
# ═══════════════════════════════════════════════════════════════

def test_get_system_prompt_returns_content():
    from knowledge.manager import KnowledgeManager
    p = KnowledgeManager().get_system_prompt("technical_analyst")
    assert isinstance(p, str) and len(p) > 50


def test_critic_prompt_now_exists():
    """critic.txt 已补齐 (v3.0 实跑发现缺失), 不应再返回空"""
    from knowledge.manager import KnowledgeManager
    p = KnowledgeManager().get_system_prompt("critic")
    assert isinstance(p, str) and len(p) > 50, "critic 提示词不应为空"


def test_get_rule_stamp_duty_rate():
    from knowledge.manager import KnowledgeManager
    r = KnowledgeManager().get_rule("trading_rules.fees.stamp_duty")
    assert isinstance(r, dict)
    assert r.get("rate") == pytest.approx(0.0005), "印花税 0.05% (2023-08 起)"
    assert r.get("side") == "sell_only"


def test_get_strategy_dual_ma_trend():
    from knowledge.manager import KnowledgeManager
    s = KnowledgeManager().get_strategy("dual_ma_trend")
    assert s is not None
    assert isinstance(s, dict)
    assert "name" in s or "description" in s


def test_list_strategies_nonempty():
    from knowledge.manager import KnowledgeManager
    strats = KnowledgeManager().list_strategies()
    assert len(strats) >= 9, "策略注册表应含 9+ 策略"
