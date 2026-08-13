"""知识持仓择时 (方案3b 消费端) — 单测 (纯函数 + fake, 不联网不调 LLM)。

覆盖:
- recall_verified_rules: gate=0 返回 []; gate=1 取回截面型(low_pe_value, 脏参数夹回)
  并标记时序型(ma_cross) not_applicable。
- _parse_params: 幻觉参数 max_pe=0.5 → 夹回默认 15.0。
- buy_gate_low_pe_value: 买入门 pe>0&pe<max_pe&pb>0&pb<max_pb (忠实 tester cb)。
- sell_signal_low_pe_value: 卖出 pe>max_pe*1.5 (忠实 tester cs)。
- apply_rules_to_cross_section: 买入门 (无保底回退) / 缺列跳过 / 无 applicable 规则。
- sell_signals_for_positions: 持仓高估卖出信号 (OR 叠加)。
"""
from __future__ import annotations

import pandas as pd

from agent.learning.knowledge_apply import (
    recall_verified_rules,
    apply_rules_to_cross_section,
    sell_signals_for_positions,
    buy_gate_low_pe_value,
    sell_signal_low_pe_value,
    _parse_params,
)


# ── fake KnowledgeManager (monkeypatch, 避免碰真实 ChromaDB) ──

class _FakeKM:
    def __init__(self, path):
        pass

    def list_learned(self, categories=None, verdicts=None):
        # 模拟已按 categories=["rule"] + verdicts=["verified"] 过滤后的结果
        return [
            {"concept": "估值指标筛选", "doc": "估值指标筛选\n低PE低PB选股",
             "meta": {"concept": "估值指标筛选", "category": "rule", "verdict": "verified",
                      "template": "low_pe_value",
                      "params": '{"max_pe": 0.5, "max_pb": 1.0}',  # 幻觉参数
                      "confidence": 0.8}},
            {"concept": "均线系统", "doc": "均线系统\n金叉买入",
             "meta": {"concept": "均线系统", "category": "rule", "verdict": "verified",
                      "template": "ma_cross",
                      "params": '{"fast": 5, "slow": 20}',
                      "confidence": 0.85}},
        ]


def _patch_km(monkeypatch):
    monkeypatch.setattr("knowledge.manager.KnowledgeManager", _FakeKM)


# ── recall_verified_rules ──

def test_recall_gate0_empty(monkeypatch):
    monkeypatch.delenv("LEARNED_KNOWLEDGE_GATE", raising=False)
    _patch_km(monkeypatch)
    assert recall_verified_rules(0) == []


def test_recall_gate1_sanitizes_and_marks_timeseries(monkeypatch):
    _patch_km(monkeypatch)
    rules = recall_verified_rules(1)
    by_tpl = {r["template"]: r for r in rules}

    low = by_tpl["low_pe_value"]
    assert low["applicable"] is True
    assert low["params"]["max_pe"] == 15.0  # 幻觉 0.5 被夹回默认
    assert low["params"]["max_pb"] == 1.0   # 区间内原样保留

    ma = by_tpl["ma_cross"]
    assert ma["applicable"] is False        # 时序型, 不强行落地
    assert "时序型" in ma["reason"]


# ── _parse_params (脏参数夹回) ──

def test_parse_params_sanitizes_hallucination():
    out = _parse_params("low_pe_value", '{"max_pe": 0.5, "max_pb": 1.0}')
    assert out["max_pe"] == 15.0
    assert out["max_pb"] == 1.0


def test_parse_params_bad_json_returns_defaults():
    out = _parse_params("low_pe_value", "not-json")
    assert out["max_pe"] == 15.0
    assert out["max_pb"] == 2.0


# ── buy_gate_low_pe_value (忠实 tester cb) ──

def _params(max_pe=15.0, max_pb=2.0):
    return {"max_pe": max_pe, "max_pb": max_pb}


def test_buy_gate_requires_positive_pe_pb_and_below_threshold():
    df = pd.DataFrame({"code": ["a", "b", "c", "d", "e"],
                       "pe_ttm": [10.0, 20.0, -3.0, 8.0, None],
                       "pb": [1.0, 1.0, 1.0, 0.0, 1.0]})
    m = buy_gate_low_pe_value(df, _params())
    # a: pe10<15,pb1<2 ✓ | b: pe20 超 ✗ | c: pe-3 亏损 ✗ | d: pb0 破净缺失 ✗ | e: NaN ✗
    assert list(df.loc[m, "code"]) == ["a"]


def test_buy_gate_missing_column_returns_none():
    df = pd.DataFrame({"code": ["a"], "pe_ttm": [10.0]})  # 缺 pb
    assert buy_gate_low_pe_value(df, _params()) is None


# ── sell_signal_low_pe_value (忠实 tester cs) ──

def test_sell_signal_high_pe():
    df = pd.DataFrame({"code": ["a", "b", "c"],
                       "pe_ttm": [10.0, 22.5, 30.0]})
    m = sell_signal_low_pe_value(df, _params(max_pe=15.0))
    # pe>22.5 才触发; 22.5 恰好等于阈值不触发 (严格 >)
    assert list(df.loc[m, "code"]) == ["c"]


# ── apply_rules_to_cross_section (买入门, 无保底) ──

def _low_pe_rule(**over):
    r = {"concept": "估值指标筛选", "template": "low_pe_value",
         "params": {"max_pe": 15.0, "max_pb": 2.0}, "applicable": True, "reason": ""}
    r.update(over)
    return r


def _df():
    return pd.DataFrame({"code": ["a", "b", "c"],
                         "pe_ttm": [10.0, 20.0, 30.0],
                         "pb": [1.0, 1.0, 1.0]})


def test_apply_buy_gate_keeps_only_low_pe():
    df = _df()
    out, report = apply_rules_to_cross_section(df, [_low_pe_rule()])
    assert list(out["code"]) == ["a"]          # 只留 pe_ttm=10
    assert report["before"] == 3 and report["after"] == 1
    assert report["removed"] == 2
    assert "fallback" not in report            # 已移除保底回退


def test_apply_no_fallback_when_tiny_pool():
    # 方案3b 关键: 买入门只留低估, 无保底回退 (候选池少是特性不是 bug)
    df = pd.DataFrame({"code": ["a", "b", "c"],
                       "pe_ttm": [10.0, 50.0, 60.0],
                       "pb": [1.0, 1.0, 1.0]})
    out, report = apply_rules_to_cross_section(df, [_low_pe_rule()])
    assert len(out) == 1                       # 只留 a, 不回退到 3 只
    assert report["after"] == 1


def test_apply_skips_missing_column():
    df = pd.DataFrame({"code": ["a", "b"], "close": [1.0, 2.0]})  # 无 pe_ttm/pb
    out, report = apply_rules_to_cross_section(df, [_low_pe_rule()])
    assert len(out) == 2                       # 缺列跳过, 不报错不伪造
    assert report["removed"] == 0


def test_apply_no_applicable_rules_returns_original():
    df = _df()
    r = _low_pe_rule(applicable=False, reason="时序型")
    out, report = apply_rules_to_cross_section(df, [r])
    assert report["enabled"] is False
    assert len(out) == 3


# ── sell_signals_for_positions ──

def test_sell_signals_for_positions_or_mask():
    df = pd.DataFrame({"code": ["a", "b", "c"],
                       "pe_ttm": [10.0, 25.0, 30.0]})
    mask = sell_signals_for_positions(df, [_low_pe_rule()])
    assert list(df.loc[mask, "code"]) == ["b", "c"]  # pe>22.5 触发


def test_sell_signals_no_rules_returns_all_false():
    df = _df()
    mask = sell_signals_for_positions(df, [])
    assert not mask.any()
