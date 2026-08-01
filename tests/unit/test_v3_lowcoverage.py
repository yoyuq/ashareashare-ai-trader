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
