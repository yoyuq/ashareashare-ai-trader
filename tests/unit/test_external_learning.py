"""自主学习闭环 — 单测 (纯函数 + fake 依赖, 不联网不调 LLM)。

覆盖:
- tester.apply_keep_criterion (预注册留/删判据, 纯函数)
- tester._parse_json / researcher._parse_json_array (容错解析)
- researcher.web_search 无 key 优雅降级 (返回 [])
- knowledge_history.record 经 fake km 落库 (断言 verdict 传入)
"""
from __future__ import annotations

import asyncio

import pytest

from agent.learning.researcher import KnowledgeCandidate, web_search, _parse_json_array
from agent.learning.knowledge_history import (KnowledgeHistory, learned_gate,
                                              _format_learned_items, merge_revalidation)
from agent.learning import tester as T
from agent.learning.tester import apply_keep_criterion, _parse_json, TEMPLATES, _make_strategy, _compute_indicators, resolve_template, _sanitize_params
from scripts.learn_external import _fresh_windows_for_record


# ── 预注册判据 (纯函数) ──

def _win(name, sharpe_delta, dd_delta):
    return {"window": name, "sharpe_delta": sharpe_delta, "dd_delta": dd_delta}


def test_criterion_verified_two_windows_improve():
    deltas = [_win("w1", 0.5, 2.0), _win("w2", 0.3, 1.0), _win("w3", -0.1, -0.5)]
    r = apply_keep_criterion(deltas)
    assert r.verdict == "verified"
    assert r.metric_delta["improved_windows"] == 2


def test_criterion_rejected_zero_improve():
    deltas = [_win("w1", -0.2, -1.0), _win("w2", 0.0, -2.0)]
    r = apply_keep_criterion(deltas)
    assert r.verdict == "rejected"
    assert r.metric_delta["improved_windows"] == 0


def _win2(name, sharpe_delta, dd_delta, return_delta):
    return {"window": name, "sharpe_delta": sharpe_delta, "dd_delta": dd_delta,
            "return_delta": return_delta}


def test_criterion_v2_risk_reduction_verified():
    # 降险通道: 回撤收窄 >1pp 且收益牺牲 ≤3pp, 即使夏普不提升也算改善
    deltas = [_win2("w1", -0.3, 2.5, -1.0), _win2("w2", -0.1, 3.0, -2.0)]
    r = apply_keep_criterion(deltas)
    assert r.verdict == "verified"
    assert r.metric_delta["improved_windows"] == 2
    assert r.metric_delta["risk_reduction_windows"] == 2


def test_criterion_v2_return_sacrifice_too_large_not_improved():
    # 收益牺牲 >3pp → 降险通道不成立
    deltas = [_win2("w1", -0.5, 3.0, -5.0), _win2("w2", -0.2, -1.0, -1.0)]
    r = apply_keep_criterion(deltas)
    assert r.verdict == "rejected"
    assert r.metric_delta["improved_windows"] == 0


def test_criterion_v2_superset_adj_still_wins():
    # v1 风险调整通道仍生效 (回归锁: v2 是 v1 超集)
    deltas = [_win("w1", 0.5, 2.0), _win("w2", 0.3, 1.0), _win("w3", -0.1, -0.5)]
    r = apply_keep_criterion(deltas)
    assert r.verdict == "verified"
    assert r.metric_delta["improved_windows"] == 2


def test_resolve_template_support_stop_and_time_stop():
    t, params, _ = resolve_template({"template": "support_stop", "params": {"support_lookback": 15}})
    assert t == "support_stop" and params["support_lookback"] == 15
    t2, _, _ = resolve_template({"template": "time_stop", "params": {"max_hold_days": 30}})
    assert t2 == "time_stop"


def test_compute_indicators_support_stop_adds_support():
    import pandas as pd
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=30, freq="D"),
                       "close": list(range(30, 0, -1)), "low": list(range(29, -1, -1))})
    out = _compute_indicators(df, "support_stop", {"support_lookback": 5})
    assert "support" in out.columns
    assert out["support"].notna().any()


def test_criterion_inconclusive_single_improve():
    deltas = [_win("w1", 0.4, 1.0), _win("w2", -0.3, -2.0), _win("w3", -0.1, -1.0)]
    r = apply_keep_criterion(deltas)
    assert r.verdict == "inconclusive"


def test_criterion_dd_significant_worsen_blocks_keep():
    # 夏普提升但回撤显著恶化 (dd_delta < -2*sharpe) → 不计数为改善窗口
    deltas = [_win("w1", 0.6, -3.0), _win("w2", 0.6, -3.0)]
    r = apply_keep_criterion(deltas)
    assert r.verdict == "rejected"


def test_criterion_empty():
    r = apply_keep_criterion([])
    assert r.verdict == "rejected"  # 0/0 无改善 → 不留


# ── 容错 JSON 解析 ──

def test_parse_json_plain():
    assert _parse_json('{"template": "rsi_reversal"}') == {"template": "rsi_reversal"}


def test_parse_json_code_fence():
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_wrapped_text():
    assert _parse_json('here is result: {"b": 2} trailing') == {"b": 2}


def test_parse_json_garbage():
    assert _parse_json("not json at all") == {}


def test_parse_json_array_plain():
    assert _parse_json_array('[{"concept": "x"}]') == [{"concept": "x"}]


def test_parse_json_array_object_wraps():
    assert _parse_json_array('{"concept": "x"}') == [{"concept": "x"}]


# ── 联网搜索降级 ──

def test_web_search_no_key_degrades(monkeypatch):
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    out = asyncio.run(web_search("价值投资", count=3))
    assert out == []


def test_web_search_unknown_provider_degrades(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "no_such_backend")
    monkeypatch.setenv("SEARCH_API_KEY", "dummy-key")
    out = asyncio.run(web_search("价值投资", count=3))
    assert out == []


# ── knowledge_history 落库 (fake km) ──

class _FakeKM:
    def __init__(self):
        self.records = []

    def record_learned(self, concept, claim, verdict, category="rule", meta=None):
        self.records.append({"concept": concept, "claim": claim, "verdict": verdict,
                             "category": category, "meta": meta})
        return True

    def recall_learned(self, query, top_k=5):
        return []


def test_history_record_passes_verdict_and_meta():
    km = _FakeKM()
    h = KnowledgeHistory(km)
    cand = KnowledgeCandidate(concept="低PE价值投资", claim="买入PE<10长期跑赢",
                              category="rule", testable="PE<10 买入")
    ok = h.record(cand, "verified", T.TestResult("verified", reason="ok").to_dict(), date="2026-08-13")
    assert ok is True
    rec = km.records[0]
    assert rec["concept"] == "低PE价值投资"
    assert rec["verdict"] == "verified"
    assert rec["category"] == "rule"
    assert rec["meta"]["reason"] == "ok"
    assert rec["meta"]["date"] == "2026-08-13"


def test_history_check_learned_empty_recall_returns_none():
    km = _FakeKM()
    h = KnowledgeHistory(km)
    cand = KnowledgeCandidate(concept="任意", claim="x", category="fact", testable="x")
    # recall 为空 → 未学过, 且不应触发 LLM 判重
    assert asyncio.run(h.check_learned(cand)) is None


# ── P0 消费路径: 注入门槛 + 格式化 (纯函数) ──

def _learned(verdict="verified", confidence=0.5, concept="c"):
    return {"concept": concept, "doc": f"{concept}\nclaim", "verdict": verdict,
            "category": "rule", "confidence": confidence}


def test_learned_gate_default_off(monkeypatch):
    monkeypatch.delenv("LEARNED_KNOWLEDGE_GATE", raising=False)
    assert learned_gate() == 0


def test_learned_gate_parses(monkeypatch):
    monkeypatch.setenv("LEARNED_KNOWLEDGE_GATE", "1")
    assert learned_gate() == 1
    monkeypatch.setenv("LEARNED_KNOWLEDGE_GATE", "abc")
    assert learned_gate() == 0  # 非法值回退默认关


def test_format_learned_items_gate0_empty():
    assert _format_learned_items([_learned()], 0) == ""


def test_format_learned_items_gate1_only_verified():
    items = [_learned("verified"), _learned("rejected"), _learned("inconclusive")]
    out = _format_learned_items(items, 1)
    assert "✅ 已验证" in out
    assert "rejected" not in out and "inconclusive" not in out


def test_format_learned_items_gate2_confidence_threshold():
    hi = _learned("verified", confidence=0.9, concept="高置信规则")
    lo = _learned("verified", confidence=0.5, concept="低置信规则")
    out = _format_learned_items([hi, lo], 2)
    assert "高置信规则" in out
    assert "低置信规则" not in out


# ── P1 忠实测试: 模板库 ──

def test_templates_cover_rule_classes():
    # 忠实模板库应覆盖 趋势/均值回归/估值/动量/止损/移动止损/量能
    for t in ("ma_cross", "rsi_reversal", "low_pe_value", "momentum_rank",
              "stop_loss_fixed", "trailing_stop", "volume_breakout"):
        assert t in TEMPLATES


def test_make_strategy_returns_callable_every_template():
    for t in TEMPLATES:
        fn = _make_strategy(t, {}, position_cash=4500.0)
        assert callable(fn)


@pytest.mark.parametrize("template,expect_cols", [
    ("rsi_reversal", ["rsi"]),
    ("ma_cross", ["ma_fast", "ma_slow"]),
    ("momentum_rank", ["mom"]),
    ("volume_breakout", ["vol_ma20"]),
])
def test_compute_indicators_adds_columns(template, expect_cols):
    import pandas as pd
    n = 60
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "close": pd.Series(range(10, 10 + n), dtype=float),
        "volume": pd.Series([1_000_000.0] * n),
    })
    out = _compute_indicators(df, template, {})
    for c in expect_cols:
        assert c in out.columns


def test_resolve_template_not_yet_testable():
    t, _, reason = resolve_template({"template": "not_yet_testable", "reason": "仓位管理无模板"})
    assert t is None and "仓位管理" in reason


def test_resolve_template_unknown_falls_to_notest():
    t, _, _ = resolve_template({"template": "garbage"})
    assert t is None


def test_resolve_template_valid():
    t, p, _ = resolve_template({"template": "momentum_rank", "params": {"top_n": 3}})
    assert t == "momentum_rank" and p["top_n"] == 3


def test_low_pe_value_no_extra_indicator_needed():
    # low_pe_value 直接读 peTTM/pbMRQ, 不新增指标列
    import pandas as pd
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="B"),
        "close": pd.Series(range(10, 20), dtype=float),
        "volume": pd.Series([1.0] * 10),
        "peTTM": pd.Series([5.0] * 10),
        "pbMRQ": pd.Series([1.0] * 10),
    })
    out = _compute_indicators(df, "low_pe_value", {})
    assert "peTTM" in out.columns and "pbMRQ" in out.columns


# ── v5.10 参数防幻觉 (sanitize) ──

def test_sanitize_params_resets_hallucinated_max_pe():
    # 翻译官幻觉 max_pe=0.5 (PE<0.5 实为空组合) → 回落到模板默认 15
    out = _sanitize_params("low_pe_value", {"max_pe": 0.5, "max_pb": 1.0})
    assert out["max_pe"] == 15.0
    assert out["max_pb"] == 1.0  # 区间内原样保留


def test_sanitize_params_keeps_in_range():
    out = _sanitize_params("low_pe_value", {"max_pe": 20.0, "max_pb": 3.0})
    assert out["max_pe"] == 20.0 and out["max_pb"] == 3.0


def test_sanitize_params_missing_key_uses_default():
    out = _sanitize_params("stop_loss_fixed", {})
    assert out["stop_pct"] == 0.08


def test_sanitize_params_non_numeric_uses_default():
    out = _sanitize_params("time_stop", {"max_hold_days": "abc"})
    assert out["max_hold_days"] == 20


def test_sanitize_params_ma_cross_fast_not_less_than_slow_resets():
    out = _sanitize_params("ma_cross", {"fast": 50, "slow": 10})
    assert out["fast"] == 5 and out["slow"] == 20


# ── P2 实战反馈闭环: 滚动重测 merge_revalidation (纯函数) ──

def test_merge_revalidation_confirmed():
    r = merge_revalidation("verified", "verified", 0.7, 0)
    assert r["verdict"] == "verified" and r["action"] == "confirmed"
    assert r["confidence"] == 0.8 and r["revalidations"] == 1


def test_merge_revalidation_falsified():
    r = merge_revalidation("verified", "rejected", 0.7, 2)
    assert r["verdict"] == "rejected" and r["action"] == "falsified"
    assert r["confidence"] == 0.45 and r["revalidations"] == 3


def test_merge_revalidation_inconclusive_keeps_old():
    r = merge_revalidation("inconclusive", "inconclusive", 0.6, 1)
    assert r["verdict"] == "inconclusive" and r["action"] == "unchanged"
    assert r["confidence"] == 0.55 and r["revalidations"] == 2


def test_merge_revalidation_confidence_capped():
    r = merge_revalidation("verified", "verified", 0.92, 0)
    assert r["confidence"] == 0.95  # 封顶


def test_merge_revalidation_confidence_floored():
    r = merge_revalidation("verified", "rejected", 0.4, 0)
    assert r["confidence"] == 0.30  # 保底


def test_merge_revalidation_untestable_noop():
    r = merge_revalidation("verified", "not_yet_testable", 0.7, 3)
    assert r == {"verdict": "verified", "confidence": 0.7,
                 "revalidations": 3, "action": "untestable"}


# ── P2: TestResult 持久化 template/params (供重测复用, 不重译) ──

def test_test_result_to_dict_carries_template():
    r = T.TestResult("verified", "ok", template="ma_cross", params={"fast": 5, "slow": 20})
    d = r.to_dict()
    assert d["template"] == "ma_cross" and d["params"] == {"fast": 5, "slow": 20}


def test_test_result_to_dict_omits_template_when_none():
    d = T.TestResult("not_yet_testable", "x").to_dict()
    assert "template" not in d


# ── P2: 新鲜 out-of-sample 窗口挑选 ──

def test_fresh_windows_exclude_tested():
    all_files = ["replay_data/daily_2020-06-01_2021-02-28.parquet",
                 "replay_data/daily_2024-01-01_2024-12-31.parquet",
                 "replay_data/daily_2018-01-01_2018-12-31.parquet"]
    md = {"windows_tested": '["2020-06-01_2021-02-28"]'}
    fresh = _fresh_windows_for_record(md, all_files)
    names = [f.split("daily_")[-1].replace(".parquet", "") for f in fresh]
    assert "2020-06-01_2021-02-28" not in names
    assert "2018-01-01_2018-12-31" in names


def test_fresh_windows_no_tested_returns_all():
    all_files = ["replay_data/daily_2020-06-01_2021-02-28.parquet"]
    fresh = _fresh_windows_for_record({}, all_files)  # 无 windows_tested → 全部视为新鲜
    assert fresh == all_files


# ── P3 自动化: 好奇队列 + 元报告 (纯函数) ──

from agent.learning.curiosity import (pick_next_topics, mark_learned,
                                      summarize_learned, format_meta_report)


def test_pick_next_fresh_seeds_first():
    state = {"topics": {"动量策略": {"last_learned": "2026-08-01"}}, "order": ["动量策略"]}
    seeds = ["动量策略", "价值投资", "趋势跟踪"]
    picked = pick_next_topics(state, seeds, n=2, today="2026-08-13")
    # 未学的"价值投资"/"趋势跟踪"排最前
    assert picked == ["价值投资", "趋势跟踪"]


def test_pick_next_stale_by_last_learned():
    state = {"topics": {"A": {"last_learned": "2026-01-01"}, "B": {"last_learned": "2026-07-01"}},
             "order": []}
    picked = pick_next_topics(state, ["A", "B"], n=2, today="2026-08-13")
    assert picked == ["A", "B"]  # A 更久未学 → 优先


def test_mark_learned_updates_state():
    state = {"topics": {}, "order": []}
    mark_learned(state, ["动量策略"], n_learned=1, today="2026-08-13")
    assert state["topics"]["动量策略"]["last_learned"] == "2026-08-13"
    assert state["topics"]["动量策略"]["n_learned"] == 1
    assert state["order"] == ["动量策略"]
    mark_learned(state, ["动量策略"], n_learned=1, today="2026-08-13")
    assert state["topics"]["动量策略"]["n_learned"] == 2  # 累加


def _meta(concept, verdict="rejected", category="rule", confidence=0.7, revalidations=0):
    return {"meta": {"concept": concept, "verdict": verdict, "category": category,
                     "confidence": confidence, "revalidations": revalidations}}


def test_summarize_learned_distribution():
    recs = [_meta("A", "verified", confidence=0.9),
            _meta("B", "rejected", confidence=0.3),
            _meta("C", "not_yet_testable", confidence=0.6),
            _meta("D", "verified", revalidations=1, confidence=0.8)]
    s = summarize_learned(recs)
    assert s["total"] == 4
    assert s["by_verdict"]["verified"] == 2
    assert s["by_verdict"]["rejected"] == 1
    assert s["by_verdict"]["not_yet_testable"] == 1
    assert s["concepts"]["verified"] == ["A", "D"]
    assert s["revalidated"] == ["D"]


def test_format_meta_report_renders():
    text = format_meta_report([_meta("A", "verified"), _meta("B", "rejected")])
    assert "元报告" in text and "已验证" in text and "A" in text


# ── P4 fact 型回写知识库 ──

import agent.learning.kb_writer as KB
from agent.learning.kb_writer import apply_fact_to_kb, _char_jaccard


def _fact(concept="X", claim="c", confidence=0.9, source="s"):
    return KnowledgeCandidate(concept=concept, claim=claim, category="fact",
                              testable=claim, source=source, confidence=confidence)


def test_char_jaccard_identical():
    assert _char_jaccard("abc", "abc") == 1.0


def test_apply_fact_low_conf_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(KB, "LEARNED_FACTS_FILE", str(tmp_path / "facts.md"))
    monkeypatch.setattr(KB, "CONTRADICTIONS_FILE", str(tmp_path / "cont.jsonl"))
    assert apply_fact_to_kb(_fact(confidence=0.5), T.TestResult("verified", "ok")) == "skipped_low_conf"
    assert not (tmp_path / "facts.md").exists()


def test_apply_fact_verified_inserted(tmp_path, monkeypatch):
    monkeypatch.setattr(KB, "LEARNED_FACTS_FILE", str(tmp_path / "facts.md"))
    monkeypatch.setattr(KB, "CONTRADICTIONS_FILE", str(tmp_path / "cont.jsonl"))
    cand = _fact(concept="低PE", claim="PE<10 长期跑赢", confidence=0.9)
    assert apply_fact_to_kb(cand, T.TestResult("verified", "一致")) == "inserted"
    assert (tmp_path / "facts.md").exists()


def test_apply_fact_duplicate_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(KB, "LEARNED_FACTS_FILE", str(tmp_path / "facts.md"))
    monkeypatch.setattr(KB, "CONTRADICTIONS_FILE", str(tmp_path / "cont.jsonl"))
    cand = _fact(concept="低PE", claim="PE<10 长期跑赢", confidence=0.9)
    apply_fact_to_kb(cand, T.TestResult("verified", "一致"))
    assert apply_fact_to_kb(cand, T.TestResult("verified", "一致")) == "skipped_dup"


def test_apply_fact_rejected_logs_contradiction(tmp_path, monkeypatch):
    monkeypatch.setattr(KB, "LEARNED_FACTS_FILE", str(tmp_path / "facts.md"))
    monkeypatch.setattr(KB, "CONTRADICTIONS_FILE", str(tmp_path / "cont.jsonl"))
    assert apply_fact_to_kb(_fact(confidence=0.9), T.TestResult("rejected", "冲突")) == "contradiction_logged"
    assert (tmp_path / "cont.jsonl").exists()


def test_apply_fact_not_covered_high_conf_pending_review(tmp_path, monkeypatch):
    # not_covered 高置信 → 待复核文件, 不入 learned_facts.md (E: 未覆盖事实复核通路)
    monkeypatch.setattr(KB, "LEARNED_FACTS_FILE", str(tmp_path / "facts.md"))
    monkeypatch.setattr(KB, "PENDING_REVIEW_FILE", str(tmp_path / "pending.jsonl"))
    cand = _fact(concept="T+1交易制度", claim="资金T+0可用", confidence=0.85)
    assert apply_fact_to_kb(cand, T.TestResult("inconclusive", "未覆盖")) == "pending_review"
    assert (tmp_path / "pending.jsonl").exists()
    assert not (tmp_path / "facts.md").exists()  # 不入库


def test_apply_fact_not_covered_low_conf_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(KB, "LEARNED_FACTS_FILE", str(tmp_path / "facts.md"))
    monkeypatch.setattr(KB, "PENDING_REVIEW_FILE", str(tmp_path / "pending.jsonl"))
    assert apply_fact_to_kb(_fact(confidence=0.5), T.TestResult("inconclusive", "未覆盖")) == "skipped"
    assert not (tmp_path / "pending.jsonl").exists()


# ── P4b: fact 一致性判据修正 (沉默≠冲突) ──

def test_fact_prompt_disambiguates_conflict_vs_uncovered():
    # 回归锁: 规则库"没写"必须判 not_covered, 不能误判 contradicts (见 T+1资金T0 误报)
    from agent.learning.tester import _FACT_PROMPT
    assert "沉默永远不构成冲突" in _FACT_PROMPT
    assert "以**未覆盖**为准" in _FACT_PROMPT
    assert "选 not_covered" in _FACT_PROMPT


# ── D: 查重语义化 (bge-m3 embedding API, 全程零模拟) ──

from knowledge.manager import _semantic_embed, _EMBED_DIM


class _FakeEmbedResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_semantic_embed_missing_key_raises(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "")
    with pytest.raises(RuntimeError):
        _semantic_embed("止损")


def test_semantic_embed_parses_bge_m3(monkeypatch):
    import requests
    vec = [0.0] * _EMBED_DIM
    vec[7] = 1.0
    monkeypatch.setenv("SILICONFLOW_API_KEY", "dummy")

    def fake_post(url, headers, json, timeout):
        return _FakeEmbedResp({"data": [{"embedding": vec, "index": 0}]})

    monkeypatch.setattr(requests, "post", fake_post)
    out = _semantic_embed("止损")
    assert len(out) == _EMBED_DIM
    assert out[7] == 1.0


def test_semantic_embed_dimension_mismatch_raises(monkeypatch):
    import requests
    monkeypatch.setenv("SILICONFLOW_API_KEY", "dummy")

    def fake_post(url, headers, json, timeout):
        return _FakeEmbedResp({"data": [{"embedding": [0.0] * 256, "index": 0}]})

    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(RuntimeError):
        _semantic_embed("止损")
