"""chat_agent 交互式学习工具 — 单测 (纯函数 + fake, 不联网不调 LLM)。

覆盖:
- TOOLS 注册 4 个学习工具 (search/judge/suggest_windows/learn)
- _dispatch_map 注册对应 handler
- _WINDOW_REGIME 覆盖回放窗口
- _judge_strategy 按 category 正确路由 rule→test_rule / fact→test_fact (monkeypatch)
- 空参数优雅报错
"""
from __future__ import annotations

import asyncio
import json

from agent.chat_agent import TOOLS, ToolExecutor, _WINDOW_REGIME
from agent.learning import tester as T


def _tool_names() -> list[str]:
    return [t["function"]["name"] for t in TOOLS]


# ── 工具注册 ──

def test_learning_tools_registered():
    names = set(_tool_names())
    assert {"search_trading_strategy", "judge_trading_strategy",
            "suggest_backtest_windows", "learn_trading_strategy"} <= names


def test_learning_tools_in_dispatch_map():
    ex = ToolExecutor()
    dmap = ex._dispatch_map
    for n in ("search_trading_strategy", "judge_trading_strategy",
              "suggest_backtest_windows", "learn_trading_strategy"):
        assert n in dmap, f"{n} 未注册到 _dispatch_map"


def test_window_regime_covers_replay_windows():
    assert len(_WINDOW_REGIME) >= 5
    assert "2018-01-01_2018-12-31" in _WINDOW_REGIME  # 熊市窗口必须在


# ── 空参数优雅报错 ──

def test_search_strategy_requires_topic():
    out = json.loads(asyncio.run(ToolExecutor()._search_strategy("")))
    assert "error" in out


def test_judge_strategy_requires_args():
    out = json.loads(asyncio.run(ToolExecutor()._judge_strategy("", "", "rule")))
    assert "error" in out


def test_suggest_windows_requires_strategy():
    out = json.loads(asyncio.run(ToolExecutor()._suggest_windows("")))
    assert "error" in out


def test_learn_strategy_requires_topic():
    out = json.loads(asyncio.run(ToolExecutor()._learn_strategy("")))
    assert "error" in out


# ── 路由正确性 (monkeypatch 掉 LLM/回测) ──

def test_judge_strategy_fact_routes_to_test_fact(monkeypatch):
    called = {}

    async def _fake_test_fact(candidate, km=None):
        called["fn"] = "test_fact"
        return T.TestResult("verified", "一致性核验通过")

    async def _fake_test_rule(candidate, data_files, universe_size=20):
        called["fn"] = "test_rule"
        return T.TestResult("rejected", "x")

    monkeypatch.setattr(T, "test_fact", _fake_test_fact)
    monkeypatch.setattr(T, "test_rule", _fake_test_rule)

    out = json.loads(asyncio.run(
        ToolExecutor()._judge_strategy("涨跌停", "A股有涨跌停制度", "fact")))
    assert called["fn"] == "test_fact"
    assert out["verdict"] == "verified"


def test_judge_strategy_rule_routes_to_test_rule(monkeypatch):
    called = {}

    async def _fake_test_fact(candidate, km=None):
        called["fn"] = "test_fact"
        return T.TestResult("verified", "x")

    async def _fake_test_rule(candidate, data_files, universe_size=20):
        called["fn"] = "test_rule"
        return T.TestResult("rejected", "0/3 窗口改善")

    monkeypatch.setattr(T, "test_fact", _fake_test_fact)
    monkeypatch.setattr(T, "test_rule", _fake_test_rule)

    out = json.loads(asyncio.run(
        ToolExecutor()._judge_strategy("均线", "金叉买入死叉卖出", "rule")))
    assert called["fn"] == "test_rule"
    assert out["verdict"] == "rejected"


def test_suggest_windows_parses_llm_json(monkeypatch):
    """_suggest_windows 能解析 LLM 返回的 JSON (monkeypatch router)。"""
    class _FakeResult:
        response = '{"windows": ["2020-06-01_2021-02-28", "2024-01-01_2024-12-31"], "reasons": "趋势策略测牛市与震荡"}'

    class _FakeRouter:
        async def route(self, **kwargs):
            return _FakeResult()

    import models.router as mr
    monkeypatch.setattr(mr, "get_shared_router", lambda: _FakeRouter())

    out = json.loads(asyncio.run(ToolExecutor()._suggest_windows("双均线趋势跟踪")))
    assert "2020-06-01_2021-02-28" in out["suggested_windows"]
    assert out["reasons"]
