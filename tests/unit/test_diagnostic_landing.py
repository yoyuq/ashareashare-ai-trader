"""v4.0 大师策略+进化系统"真正落地"单元测试.

覆盖本次落地 (Module 1/2/3/4/6):
1. _load_optimized_base_prompt — 优化器自动部署 (无文件→基座; 有文件+开关→文件内容)
2. build_diagnostic_system_prompt — 注入进化总结/principles/经验/元认知
3. risk_level 硬闸门 — mock 诊断 risk=4/5 强制持仓上限/单票
4. market_phase 信念乘数 — mock 阶段 → 确信度正确折算
5. 编排路径 _ensure_evolution — lazy 初始化共享进化系统
"""

import ast
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


# ═══════════════════════════════════════════════════════════════
# 1. 优化器自动部署 (_load_optimized_base_prompt)
# ═══════════════════════════════════════════════════════════════

def test_load_optimized_prompt_no_file_uses_base(tmp_path, monkeypatch):
    """无优化文件 → 返回基座 _DIAGNOSTIC_SYSTEM_PROMPT"""
    from simulation import daily_runner
    monkeypatch.setenv("DIAG_USE_OPTIMIZED", "1")
    monkeypatch.setattr(daily_runner, "_OPTIMIZED_PROMPT_PATH", tmp_path / "nope" / "best_prompt.txt")
    prompt = daily_runner._load_optimized_base_prompt()
    assert prompt == daily_runner._DIAGNOSTIC_SYSTEM_PROMPT


def test_load_optimized_prompt_uses_file_when_enabled(tmp_path, monkeypatch):
    """DIAG_USE_OPTIMIZED=1 且文件存在 → 返回文件内容"""
    from simulation import daily_runner
    monkeypatch.setenv("DIAG_USE_OPTIMIZED", "1")
    opt_dir = tmp_path / "prompt_optimization"
    opt_dir.mkdir(parents=True)
    (opt_dir / "best_prompt.txt").write_text("===优化后大师诊断基座===", encoding="utf-8")
    monkeypatch.setattr(daily_runner, "_OPTIMIZED_PROMPT_PATH", opt_dir / "best_prompt.txt")
    prompt = daily_runner._load_optimized_base_prompt()
    assert prompt == "===优化后大师诊断基座==="


def test_load_optimized_prompt_disabled_ignores_file(tmp_path, monkeypatch):
    """DIAG_USE_OPTIMIZED 未开 → 即使文件存在也用基座"""
    from simulation import daily_runner
    monkeypatch.setenv("DIAG_USE_OPTIMIZED", "0")
    opt_dir = tmp_path / "prompt_optimization"
    opt_dir.mkdir(parents=True)
    (opt_dir / "best_prompt.txt").write_text("优化内容", encoding="utf-8")
    monkeypatch.setattr(daily_runner, "_OPTIMIZED_PROMPT_PATH", opt_dir / "best_prompt.txt")
    prompt = daily_runner._load_optimized_base_prompt()
    assert prompt == daily_runner._DIAGNOSTIC_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════
# 2. build_diagnostic_system_prompt — 进化注入
# ═══════════════════════════════════════════════════════════════

def test_build_prompt_injects_evolution_and_memory(monkeypatch):
    """注入 principles/master_tips/进化总结/经验/元认知"""
    from simulation import daily_runner

    class _FakeSnapshot:
        summary = {
            "period_summary": "本周期表现稳定",
            "strengths": ["趋势捕捉准"],
            "weaknesses": ["震荡市偏激进"],
            "biases_identified": ["追高"],
            "next_period_focus": "降低震荡市仓位",
        }

    class _FakeEvolution:
        def get_latest_principles_text(self):
            return "## 你自己总结的核心原则\n- 原则1: 趋势市敢上 (置信度 80%)"
        def get_master_tips_text(self):
            return "## 大师使用心得\n- 利弗莫尔: 最适合趋势市"
        def get_summary_text(self):
            s = self._fake.summary
            return (f"## 你上个周期的自我总结\n- 总结: {s['period_summary']}\n"
                    f"- 强项: {'、'.join(s['strengths'])}\n"
                    f"- 待改进: {'、'.join(s['weaknesses'])}\n"
                    f"- 应避免的偏见: {'、'.join(s['biases_identified'])}\n"
                    f"- 下周期重点: {s['next_period_focus']}")
        def __init__(self):
            self._fake = _FakeSnapshot()

    class _FakeMemory:
        def format_for_prompt(self, current_date=None, top_k=6, regime=None):
            return "## 你的历史经验\n1. [2026-08-01] ✅ 正确经验 | 利弗莫尔 | 趋势市加仓"
        def get_metacognition_summary(self, current_date, lookback_days=15):
            return "## 你的近期自我评估\n- 正确: 3次  错误: 1次"

    monkeypatch.setattr(daily_runner, "_load_optimized_base_prompt",
                        lambda: daily_runner._DIAGNOSTIC_SYSTEM_PROMPT)
    prompt = daily_runner.build_diagnostic_system_prompt(
        _FakeEvolution(), _FakeMemory(), current_date="2026-08-10"
    )
    assert "大师使用心得" in prompt
    assert "自己总结的核心原则" in prompt
    assert "上个周期的自我总结" in prompt
    assert "历史经验" in prompt
    assert "近期自我评估" in prompt


def test_build_prompt_no_evolution_returns_base(monkeypatch):
    """evolution/memory 为 None → 只返回基座"""
    from simulation import daily_runner
    monkeypatch.setattr(daily_runner, "_load_optimized_base_prompt",
                        lambda: daily_runner._DIAGNOSTIC_SYSTEM_PROMPT)
    prompt = daily_runner.build_diagnostic_system_prompt(None, None)
    assert prompt == daily_runner._DIAGNOSTIC_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════
# 3. risk_level 硬闸门 (Module 3)
# ═══════════════════════════════════════════════════════════════

def test_risk_gate_level4_caps_positions():
    """risk_level=4 → 最多4只/单票10%"""
    from simulation.daily_runner import _apply_risk_gate

    limits = {"max_positions": 8, "single_pct": 0.25, "label": "weak_bull"}
    _apply_risk_gate({"risk_level": 4}, limits)
    assert limits["max_positions"] == 4
    assert limits["single_pct"] == pytest.approx(0.10)


def test_risk_gate_level5_tightest():
    """risk_level=5 → 最多2只/单票6%"""
    from simulation.daily_runner import _apply_risk_gate

    limits = {"max_positions": 8, "single_pct": 0.25}
    _apply_risk_gate({"risk_level": 5}, limits)
    assert limits["max_positions"] == 2
    assert limits["single_pct"] == pytest.approx(0.06)


def test_risk_gate_level3_no_effect():
    """risk_level=3 → 不触发 (只收不放)"""
    from simulation.daily_runner import _apply_risk_gate

    limits = {"max_positions": 8, "single_pct": 0.25}
    _apply_risk_gate({"risk_level": 3}, limits)
    assert limits["max_positions"] == 8
    assert limits["single_pct"] == pytest.approx(0.25)


def test_risk_gate_level5_never_raises_cap():
    """risk_level=5 单向收紧: 已是更严格的下限不被放宽"""
    from simulation.daily_runner import _apply_risk_gate

    limits = {"max_positions": 1, "single_pct": 0.05}
    _apply_risk_gate({"risk_level": 5}, limits)
    assert limits["max_positions"] == 1
    assert limits["single_pct"] == pytest.approx(0.05)


# ═══════════════════════════════════════════════════════════════
# 4. market_phase 信念乘数 (Module 4)
# ═══════════════════════════════════════════════════════════════

def test_phase_multiplier_trend_up():
    from simulation.daily_runner import _phase_conviction_mult
    assert _phase_conviction_mult("trend_up") == pytest.approx(1.0)
    assert _phase_conviction_mult("range") == pytest.approx(0.8)
    assert _phase_conviction_mult("bubble_late") == pytest.approx(0.5)
    assert _phase_conviction_mult("trend_down") == pytest.approx(0.4)
    assert _phase_conviction_mult("unknown") == pytest.approx(0.8)


# ═══════════════════════════════════════════════════════════════
# 5. 编排路径 _ensure_evolution (Module 6)
# ═══════════════════════════════════════════════════════════════

def test_workflow_ensure_evolution_lazy(tmp_path, monkeypatch):
    """编排路径懒初始化共享进化系统 (journal/memory/evolution)"""
    from agent.orchestration.workflow import AnalysisWorkflow
    from simulation import daily_runner

    monkeypatch.setenv("PORTFOLIO_PATH", str(tmp_path / "portfolio.json"))
    monkeypatch.setenv("EVOLUTION_DIR", str(tmp_path))

    wf = AnalysisWorkflow(router=None, knowledge_manager=None)
    assert wf._evo_journal is None
    wf._ensure_evolution()
    assert wf._evo_journal is not None
    assert wf._evo_memory is not None
    assert wf._evo is not None