"""v5.7 进化可观测性 + 注入门槛 — 健康追踪 + 经验注入门槛单测.

覆盖:
A. health_tracker.snapshot_memory_health — 记忆库健康快照 (纯函数)
B. health_tracker.check_regime_alignment — regime 跨界注入告警 (防 v5.2 过拟合复发)
C. health_tracker.snapshot_daily_activity — 每日活动计数
D. experience_memory.filter_injectable — 注入门槛 0/1/2 (纯函数)
E. experience_memory.inject_gate — 环境变量读取 (默认关)
F. review_decision 无 client → 走 ModelRouter (分路径)
"""

import json
import os

import pytest

from agent.evolution.daily_review import ExperienceItem, review_decision
from agent.evolution.experience_memory import filter_injectable, inject_gate
from agent.evolution.health_tracker import (
    check_regime_alignment,
    snapshot_daily_activity,
    snapshot_memory_health,
)


def _item(id_, tags=None, scenario="trend_up", verdict="correct", confidence=0.5):
    return ExperienceItem(
        id=id_, date="2026-08-01", scenario_type=scenario, verdict=verdict,
        lesson_title=f"lesson {id_}", lesson_detail="detail",
        master_used="利弗莫尔", risk_level_given=3, actual_outcome="涨",
        confidence=confidence, tags=tags or [],
    )


# ═══════════════════════════════════════════════════════════════
# A. snapshot_memory_health
# ═══════════════════════════════════════════════════════════════

def test_memory_health_empty():
    class _Mem:
        items = []
    assert snapshot_memory_health(_Mem())["total"] == 0


def test_memory_health_counts():
    class _Mem:
        items = [
            _item("1", tags=["cf_verified"], confidence=0.8, verdict="correct"),
            _item("2", tags=["cf_failed"], confidence=0.4, verdict="wrong"),
            _item("3", tags=[], confidence=0.6, verdict="partial", scenario="range"),
        ]
    h = snapshot_memory_health(_Mem())
    assert h["total"] == 3
    assert h["by_verdict"] == {"correct": 1, "wrong": 1, "partial": 1}
    assert h["avg_confidence"] == pytest.approx(0.6, abs=1e-9)
    # ratio 落盘时 round 到 4 位小数
    assert h["cf_verified_ratio"] == pytest.approx(0.3333, abs=1e-4)
    assert h["cf_failed_ratio"] == pytest.approx(0.3333, abs=1e-4)
    assert h["by_scenario"] == {"trend_up": 2, "range": 1}


# ═══════════════════════════════════════════════════════════════
# B. check_regime_alignment
# ═══════════════════════════════════════════════════════════════

def test_regime_alignment_cross_triggers_alert():
    # 熊市注入牛市激进经验 (trend_up bullish vs strong_bear bearish) → 跨界告警
    items = [_item("1", scenario="trend_up"), _item("2", scenario="trend_up")]
    out = check_regime_alignment("strong_bear", items)
    assert out["cross_count"] == 2
    assert out["cross_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert out["alert"] is True


def test_regime_alignment_aligned_no_alert():
    # 熊市注入防守经验 (trend_down bearish) → 对齐, 不告警
    items = [_item("1", scenario="trend_down"), _item("2", scenario="bubble_late")]
    out = check_regime_alignment("strong_bear", items)
    assert out["cross_count"] == 0
    assert out["alert"] is False


def test_regime_alignment_empty():
    out = check_regime_alignment("strong_bear", [])
    assert out["injected_count"] == 0
    assert out["alert"] is False


# ═══════════════════════════════════════════════════════════════
# C. snapshot_daily_activity
# ═══════════════════════════════════════════════════════════════

def test_daily_activity_ratio():
    a = snapshot_daily_activity(date="2026-08-13", cf_verified_count=3, cf_total=6)
    assert a["cf_verified_ratio"] == pytest.approx(0.5, abs=1e-9)
    assert a["audit_count"] == 0


def test_daily_activity_no_cf():
    a = snapshot_daily_activity(cf_verified_count=0, cf_total=0)
    assert a["cf_verified_ratio"] == 0.0


# ═══════════════════════════════════════════════════════════════
# D. filter_injectable (门槛 0/1/2)
# ═══════════════════════════════════════════════════════════════

def test_filter_gate0_keeps_all():
    items = [_item("1"), _item("2", tags=["cf_failed"]), _item("3", tags=["cf_verified"])]
    assert len(filter_injectable(items, 0)) == 3


def test_filter_gate1_removes_cf_failed():
    items = [
        _item("1"),                          # 无标记 → 保留
        _item("2", tags=["cf_failed"]),      # 剔除
        _item("3", tags=["cf_verified"]),    # 保留
    ]
    got = filter_injectable(items, 1)
    assert [it.id for it in got] == ["1", "3"]


def test_filter_gate2_only_verified():
    items = [
        _item("1"),                              # 未验证 → 剔除
        _item("2", tags=["cf_verified"]),        # 保留
        _item("3", tags=["high_confidence"]),    # 保留
        _item("4", tags=["cf_failed"]),          # 剔除
    ]
    got = filter_injectable(items, 2)
    assert {it.id for it in got} == {"2", "3"}


# ═══════════════════════════════════════════════════════════════
# E. inject_gate (环境变量, 默认关)
# ═══════════════════════════════════════════════════════════════

def test_inject_gate_default_zero(monkeypatch):
    monkeypatch.delenv("EXPERIENCE_INJECT_GATE", raising=False)
    assert inject_gate() == 0


def test_inject_gate_reads_env(monkeypatch):
    monkeypatch.setenv("EXPERIENCE_INJECT_GATE", "2")
    assert inject_gate() == 2


def test_inject_gate_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("EXPERIENCE_INJECT_GATE", "abc")
    assert inject_gate() == 0


# ═══════════════════════════════════════════════════════════════
# F. review_decision 无 client → 走 ModelRouter (分路径)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_review_decision_none_client_walks_router(monkeypatch):
    import models.router as mr

    class _FakeRouteResult:
        def __init__(self, response):
            self.response = response

    class _FakeRouter:
        async def route(self, **kwargs):
            assert kwargs.get("task_type") == "evolution_review"
            return _FakeRouteResult(json.dumps({
                "verdict": "correct", "error_type": "none",
                "actual_outcome": "市场上涨2%", "lesson_title": "t",
                "lesson_detail": "d", "confidence": 0.6,
            }))

    monkeypatch.setattr(mr, "get_shared_router", lambda: _FakeRouter())

    from agent.evolution.decision_journal import DecisionRecord
    rec = DecisionRecord(
        date="2026-08-09", market_phase="trend_up", dominant_master="利弗莫尔",
        secondary_master="", risk_level=3, position_multiplier=1.0,
        max_positions_adj=0, key_risks=[], diagnosis="测试",
        market_snapshot={"regime": "weak_bull"}, regime="weak_bull",
        crowding_score=50.0, crowding_signal="unknown",
    )
    out = await review_decision(rec, {"up_ratio": 0.6}, 2.0)
    assert out is not None
    assert out["verdict"] == "correct"
