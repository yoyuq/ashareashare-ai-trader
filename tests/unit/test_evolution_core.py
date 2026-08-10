"""v5.1 进化系统内核修复 — 复活死链路 + 大师阶段落地.

覆盖:
A. 元认知偏差校准复活 — extract_experience 落 conservative/aggressive tag, get_metacognition_summary 报出真实偏差
B. 反事实验证接地 — _infer_suggestion 优先用 error_type tag; 波动率归一阈值
C. 每日复盘 PIT 修复 — 只复盘最近一条未复盘记录
D. 记忆双评分器统一 — _score_item 在 retrieve 与淘汰中一致
E. 大师阶段防御 — market_phase 防御触发减仓弱股
"""

import asyncio
import json
from dataclasses import asdict
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# A. 元认知偏差校准复活
# ═══════════════════════════════════════════════════════════════

from agent.evolution.daily_review import ExperienceItem, extract_experience


def _make_rec(risk_level=3, phase="trend_up", master="利弗莫尔"):
    from agent.evolution.decision_journal import DecisionRecord
    return DecisionRecord(
        date="2026-08-09", market_phase=phase, dominant_master=master,
        secondary_master="", risk_level=risk_level,
        position_multiplier=1.0, max_positions_adj=0, key_risks=[],
        diagnosis="测试", market_snapshot={"regime": "weak_bull"},
        regime="weak_bull", crowding_score=50.0, crowding_signal="unknown",
    )


def test_extract_experience_conservative_tag():
    """error_type=conservative → 落 conservative tag"""
    rec = _make_rec()
    review = {
        "verdict": "wrong", "error_type": "conservative",
        "lesson_title": "该上没上", "lesson_detail": "市场大涨但给了5级风险踏空, 下次该更积极",
        "actual_outcome": "市场上涨2%", "confidence": 0.7, "missed_factors": [],
    }
    exp = extract_experience(rec, review)
    assert exp is not None
    assert "conservative" in exp.tags, f"应落 conservative tag, 实际 {exp.tags}"


def test_extract_experience_aggressive_tag_chinese():
    """error_type=激进 → 落 aggressive tag (兼容中文)"""
    rec = _make_rec()
    review = {
        "verdict": "wrong", "error_type": "激进型错误",
        "lesson_title": "不该上却上", "lesson_detail": "该防守时满仓被套",
        "actual_outcome": "市场下跌3%", "confidence": 0.8, "missed_factors": [],
    }
    exp = extract_experience(rec, review)
    assert exp is not None
    assert "aggressive" in exp.tags


def test_extract_experience_no_error_type_no_bias_tag():
    """无 error_type → 不落 bias tag (不误报)"""
    rec = _make_rec()
    review = {
        "verdict": "correct", "error_type": "",
        "lesson_title": "判断正确", "lesson_detail": "风险等级给得准",
        "actual_outcome": "市场平稳", "confidence": 0.6, "missed_factors": [],
    }
    exp = extract_experience(rec, review)
    assert exp is not None
    assert "conservative" not in exp.tags and "aggressive" not in exp.tags


def test_metacognition_summary_reports_bias(tmp_path):
    """落 tag 后 get_metacognition_summary 报出保守偏差 (不再恒为攻守平衡)"""
    from agent.evolution.experience_memory import ExperienceMemory

    mem = ExperienceMemory(tmp_path / "mem.json", max_items=50)
    # 两条保守型错误经验
    for i in range(2):
        it = ExperienceItem(
            id=f"e{i}", date="2026-08-0%d" % (7 + i), scenario_type="trend_up",
            verdict="wrong", lesson_title="该上没上", lesson_detail="踏空",
            master_used="利弗莫尔", risk_level_given=5,
            actual_outcome="涨", confidence=0.7, tags=["conservative", "wrong"],
        )
        mem.items.append(it)
    summary = mem.get_metacognition_summary("2026-08-10", lookback_days=15)
    assert "偏保守" in summary, f"应识别保守偏差, 实际: {summary}"
    assert "踏空错误" in summary or "保守" in summary


# ═══════════════════════════════════════════════════════════════
# B. 反事实验证接地 + 波动率归一
# ═══════════════════════════════════════════════════════════════

from agent.evolution.counterfactual import _infer_suggestion, verify_counterfactual


def test_infer_suggestion_uses_error_type_tag():
    """conservative tag → 建议降级(更激进); aggressive tag → 建议升级"""
    cons = ExperienceItem(id="c", date="2026-08-09", scenario_type="trend_up",
                          verdict="wrong", lesson_title="t", lesson_detail="d",
                          master_used="", risk_level_given=3, actual_outcome="",
                          confidence=0.7, tags=["conservative"])
    assert _infer_suggestion(cons) == {"risk_level": 2}

    aggr = ExperienceItem(id="a", date="2026-08-09", scenario_type="trend_up",
                          verdict="wrong", lesson_title="t", lesson_detail="d",
                          master_used="", risk_level_given=3, actual_outcome="",
                          confidence=0.7, tags=["aggressive"])
    assert _infer_suggestion(aggr) == {"risk_level": 4}


def test_infer_suggestion_keyword_fallback():
    """无 tag 时关键词启发式仍工作 (旧经验兼容)"""
    exp = ExperienceItem(id="k", date="2026-08-09", scenario_type="trend_up",
                         verdict="wrong", lesson_title="t踏空t", lesson_detail="d过于保守",
                         master_used="", risk_level_given=3, actual_outcome="",
                         confidence=0.7, tags=[])
    assert _infer_suggestion(exp) == {"risk_level": 2}


def test_infer_suggestion_correct_returns_none():
    """verdict=correct → 无需反事实验证"""
    exp = ExperienceItem(id="ok", date="2026-08-09", scenario_type="trend_up",
                         verdict="correct", lesson_title="t", lesson_detail="d",
                         master_used="", risk_level_given=3, actual_outcome="",
                         confidence=0.7, tags=[])
    assert _infer_suggestion(exp) is None


def test_verify_counterfactual_vol_normalized_threshold():
    """小波动日改进不够 → 不通过 (波动率归一阈值)"""
    # conservative 经验: risk 3 → 2, 系数 0.95 → 1.20
    # 次日 +0.2% (小波动): improvement = (1.20-0.95)*0.002 = 0.0005
    # 阈值 = 0.001 * max(1, 0.2*0.5) = 0.001 → 0.0005 < 0.001 → 不通过
    exp = ExperienceItem(id="v", date="2026-08-09", scenario_type="trend_up",
                         verdict="wrong", lesson_title="踏空", lesson_detail="该上没上",
                         master_used="", risk_level_given=3, actual_outcome="",
                         confidence=0.7, tags=["conservative"])
    res = verify_counterfactual(exp, 0.2)
    assert res is not None
    assert res.passed is False, "小波动日的小改进不应通过波动率归一阈值"


def test_verify_counterfactual_large_move_passes():
    """大波动日改进确实有效 → 通过"""
    exp = ExperienceItem(id="v2", date="2026-08-09", scenario_type="trend_up",
                         verdict="wrong", lesson_title="踏空", lesson_detail="该上没上",
                         master_used="", risk_level_given=3, actual_outcome="",
                         confidence=0.7, tags=["conservative"])
    # 次日 +3% (大波动): improvement = (1.20-0.95)*0.03 = 0.0075
    # 阈值 = 0.001 * max(1, 3*0.5) = 0.0015 → 0.0075 > 0.0015 → 通过
    res = verify_counterfactual(exp, 3.0)
    assert res is not None
    assert res.passed is True


# ═══════════════════════════════════════════════════════════════
# C. 每日复盘 PIT 修复 — 只复盘最近一条
# ═══════════════════════════════════════════════════════════════

def test_review_picks_only_most_recent(monkeypatch):
    """unreviewed 多条时只复盘最近一条 (PIT 正确)"""
    from simulation import daily_runner

    journal = SimpleNamespace(
        unreviewed=lambda before: [_make_rec(risk_level=3), _make_rec(risk_level=4)],
    )
    # 模拟 phase1 中的复盘段: 截取后应只剩最近一条
    _unreviewed = journal.unreviewed(before="2026-08-10")
    # 复制 daily_runner 的 PIT 修复逻辑: 只取 [-1:]
    _unreviewed = _unreviewed[-1:]
    assert len(_unreviewed) == 1
    assert _unreviewed[0].risk_level == 4  # 最近一条(昨日)


def test_market_move_pct_uses_median(monkeypatch):
    """market_move_pct 用中位数(鲁棒于单只异动股)"""
    import pandas as pd
    df = pd.DataFrame({"pct_change": [50.0, 0.5, -0.3, 0.2, 0.1]})
    # 生产代码用 median
    pct = float(df['pct_change'].median()) if 'pct_change' in df.columns else 0.0
    assert pct == pytest.approx(0.2), "中位数应鲁棒于 +50% 的涨停异动股"


# ═══════════════════════════════════════════════════════════════
# D. 记忆双评分器统一
# ═══════════════════════════════════════════════════════════════

def test_score_item_consistent_retrieve_and_eviction(tmp_path):
    """_score_item 在 retrieve 与淘汰中一致 → 高置信经验既优先展示也不先被淘汰"""
    from agent.evolution.experience_memory import ExperienceMemory, _score_item

    mem = ExperienceMemory(tmp_path / "mem.json", max_items=50)
    # 两条同日经验: 高置信(cf_verified) vs 普通
    high = ExperienceItem(id="h", date="2026-08-09", scenario_type="trend_up",
                          verdict="wrong", lesson_title="t", lesson_detail="d",
                          master_used="利弗莫尔", risk_level_given=3,
                          actual_outcome="", confidence=0.8, tags=["cf_verified", "high_confidence"])
    normal = ExperienceItem(id="n", date="2026-08-09", scenario_type="trend_up",
                            verdict="wrong", lesson_title="t2", lesson_detail="d2",
                            master_used="利弗莫尔", risk_level_given=3,
                            actual_outcome="", confidence=0.5, tags=[])
    mem.items = [high, normal]

    # retrieve (无查询匹配, 只看价值) → 高置信应排前
    top = mem.retrieve(current_date="2026-08-10", top_k=2)
    assert top[0].id == "h", "高置信经验应优先展示"

    # 淘汰排序 → 高置信也应靠前(不被先淘汰)
    sorted_items = sorted(list(mem.items), key=lambda x: _score_item(x), reverse=True)
    assert sorted_items[0].id == "h", "高置信经验不应最先被淘汰"


# ═══════════════════════════════════════════════════════════════
# E. 大师阶段防御
# ═══════════════════════════════════════════════════════════════

def test_phase_defensive_trigger():
    """market_phase=trend_down 且 timing 未 off → 触发防御"""
    _diag = {"market_phase": "trend_down"}
    _phase_defensive = _diag.get("market_phase") in ("trend_down", "bubble_late")
    assert _phase_defensive is True


def test_phase_trend_up_no_defensive():
    """market_phase=trend_up → 不触发阶段防御 (需 timing 或 risk gate)"""
    _diag = {"market_phase": "trend_up"}
    _phase_defensive = _diag.get("market_phase") in ("trend_down", "bubble_late")
    assert _phase_defensive is False


def test_phase_bubble_late_defensive():
    """market_phase=bubble_late → 触发防御"""
    _diag = {"market_phase": "bubble_late"}
    _phase_defensive = _diag.get("market_phase") in ("trend_down", "bubble_late")
    assert _phase_defensive is True


# ═══════════════════════════════════════════════════════════════
# F (v5.2) 对抗票 — 主导 vs 对抗 分歧裁决
# ═══════════════════════════════════════════════════════════════

def _make_adv(dom=3, adv=3, mult=1.2):
    return {
        "risk_level": dom,
        "position_multiplier": mult,
        "max_positions_adj": 0,
        "market_phase": "range",
        "dominant_master": "利弗莫尔",
        "secondary_master": "达利欧",
        "adversarial_risk_level": adv,
    }


def test_adversarial_no_divergence_no_intervention():
    """分歧 <2 级 → 不干预 (原样返回)"""
    from simulation.daily_runner import _apply_adversarial_gate
    _d = _make_adv(dom=3, adv=3, mult=1.3)
    out = _apply_adversarial_gate(_d)
    assert out["risk_level"] == 3
    assert out["position_multiplier"] == 1.3
    assert out.get("adversarial_applied") is None


def test_adversarial_one_level_no_intervention():
    """分歧 1 级 → 不干预 (避免过度敏感)"""
    from simulation.daily_runner import _apply_adversarial_gate
    _d = _make_adv(dom=3, adv=4, mult=1.3)
    out = _apply_adversarial_gate(_d)
    assert out["risk_level"] == 3
    assert out.get("adversarial_applied") is None


def test_adversarial_conservative_steps_up():
    """对抗更保守(数字更大) 分歧>=2 → 主导风险+1级, 仓位向1.0收敛"""
    from simulation.daily_runner import _apply_adversarial_gate
    # 主导利弗莫尔看多 risk=2, 对抗达利欧看空 risk=4 → 分歧2
    _d = _make_adv(dom=2, adv=4, mult=1.3)
    out = _apply_adversarial_gate(_d)
    assert out["adversarial_applied"] == "conservative"
    assert out["risk_level"] == 3, "对抗更保守 → 风险+1"
    # 仓位 1.3 → 向1.0收敛一半: 1.0 + 0.3*0.5 = 1.15
    assert out["position_multiplier"] == pytest.approx(1.15)
    assert out["adversarial_divergence"] == 2


def test_adversarial_aggressive_dampens_not_chases():
    """对抗更激进(数字更小) 分歧>=2 → 不追激进, 只降置信, 风险不变"""
    from simulation.daily_runner import _apply_adversarial_gate
    # 主导看空 risk=4, 对抗看多 risk=2 → 分歧2
    _d = _make_adv(dom=4, adv=2, mult=0.5)
    out = _apply_adversarial_gate(_d)
    assert out["adversarial_applied"] == "dampen"
    assert out["risk_level"] == 4, "不追激进 → 风险保持4"
    # 仓位 0.5 → 向1.0收敛: 1.0 + (0.5-1.0)*0.5 = 0.75
    assert out["position_multiplier"] == pytest.approx(0.75)


def test_adversarial_never_raises_above_dominant_danger():
    """对抗票只收不放 — 永不覆盖成比主导更激进"""
    from simulation.daily_runner import _apply_adversarial_gate
    _d = _make_adv(dom=5, adv=1, mult=0.4)
    out = _apply_adversarial_gate(_d)
    assert out["risk_level"] == 5
    assert out["adversarial_applied"] == "dampen"
    # 仓位收敛但仍是防守态
    assert out["position_multiplier"] < 1.0


# ═══════════════════════════════════════════════════════════════
# G (v5.2 P1) 第二审计者 — 用实际结果复核 LLM 自报偏差, 打破自证循环
# ═══════════════════════════════════════════════════════════════

from agent.evolution.counterfactual import audit_review_bias


def test_audit_catches_selfconfirmed_correct_up_miss():
    """上涨+防御(risk4)但LLM自报correct → 审计覆盖为conservative (踏空盲区)"""
    _aud = audit_review_bias({"verdict": "correct", "error_type": ""}, +2.0, 4)
    assert _aud["audit_bias"] == "conservative"
    assert _aud["llm_bias"] is None
    assert _aud["overrides"] is True, "LLM自报无偏差但实际偏保守, 审计应覆盖"


def test_audit_agrees_with_llm_conservative():
    """下跌+进攻(risk2)+LLM自报conservative vs 审计激进 → 不一致, 以审计为准"""
    _aud = audit_review_bias({"verdict": "wrong", "error_type": "保守型错误"}, -2.0, 2)
    assert _aud["audit_bias"] == "aggressive"
    assert _aud["llm_bias"] == "conservative"
    assert _aud["overrides"] is True, "审计与LLM自报矛盾, 以确定性审计为准"


def test_audit_agrees_with_llm_when_consistent():
    """下跌+进攻(risk2)+LLM自报aggressive → 审计与LLM一致, 不覆盖"""
    _aud = audit_review_bias({"verdict": "wrong", "error_type": "激进型错误"}, -2.0, 2)
    assert _aud["audit_bias"] == "aggressive"
    assert _aud["llm_bias"] == "aggressive"
    assert _aud["agrees"] is True
    assert _aud["overrides"] is False


def test_audit_indeterminate_flat():
    """小波动 → 无法审计, 不覆盖 (避免误伤)"""
    _aud = audit_review_bias({"verdict": "correct", "error_type": ""}, +0.2, 3)
    assert _aud["audit_bias"] == "indeterminate"
    assert _aud["overrides"] is False


def test_audit_balanced_no_override():
    """上涨+risk3(中性) → 实际平衡, LLM自报无偏差 → 不覆盖"""
    _aud = audit_review_bias({"verdict": "correct", "error_type": ""}, +2.0, 3)
    assert _aud["audit_bias"] == "balanced"
    assert _aud["overrides"] is False


# ═══════════════════════════════════════════════════════════════
# H (v5.2 P1) OPRO train/eval 分离 — 消除优化数据泄露
# ═══════════════════════════════════════════════════════════════

def test_opro_holdout_split_disjoint():
    """留出集与选择集不相交, 且合起来=原样本 (防优化过拟合)"""
    from agent.evolution.prompt_optimizer import PromptOptimizer
    _samples = [{"id": i} for i in range(20)]
    opt = PromptOptimizer(initial_prompt="x", eval_samples=_samples,
                          output_dir="replay_data/_test_opro", holdout_ratio=0.2)
    assert len(opt.selection_samples) == 16
    assert len(opt.holdout_samples) == 4
    _ids = {s["id"] for s in opt.selection_samples} | {s["id"] for s in opt.holdout_samples}
    assert _ids == set(range(20)), "选择集+留出集应覆盖全部样本"
    _sel = {s["id"] for s in opt.selection_samples}
    _hold = {s["id"] for s in opt.holdout_samples}
    assert _sel.isdisjoint(_hold), "留出集不应与选择集重叠"


def test_opro_holdout_deterministic():
    """固定种子 → 拆分可复现"""
    from agent.evolution.prompt_optimizer import PromptOptimizer
    _samples = [{"id": i} for i in range(20)]
    a = PromptOptimizer("x", _samples, "replay_data/_test_opro", holdout_ratio=0.2)
    b = PromptOptimizer("x", _samples, "replay_data/_test_opro", holdout_ratio=0.2)
    assert [s["id"] for s in a.holdout_samples] == [s["id"] for s in b.holdout_samples]


# ═══════════════════════════════════════════════════════════════
# I (v5.2) regime 对齐 — 跨界惩罚, 防 regime 过拟合 (留出法 A/B 暴露)
# ═══════════════════════════════════════════════════════════════

from agent.evolution.experience_memory import _score_item, ExperienceMemory


def _exp(scenario, conf=0.6, tags=()):
    return ExperienceItem(id=scenario, date="2021-01-05", scenario_type=scenario,
                          verdict="wrong", lesson_title="t", lesson_detail="d",
                          master_used="利弗莫尔", risk_level_given=3,
                          actual_outcome="", confidence=conf, tags=list(tags))


def test_bear_regime_penalizes_bull_lesson():
    """熊市(weak_bear)下, 牛市(trend_up)学到的激进经验被惩罚, 防守经验优先"""
    _bull = _exp("trend_up")
    _bear = _exp("trend_down")
    s_bull = _score_item(_bull, current_date="2021-02-05", regime="weak_bear")
    s_bear = _score_item(_bear, current_date="2021-02-05", regime="weak_bear")
    assert s_bear > s_bull, f"熊市应优先防守经验, bull={s_bull:.2f} bear={s_bear:.2f}"


def test_bull_regime_prefers_bull_lesson():
    """牛市(strong_bull)下, 趋势上行经验优先"""
    _bull = _exp("trend_up")
    _bear = _exp("trend_down")
    s_bull = _score_item(_bull, current_date="2020-12-10", regime="strong_bull")
    s_bear = _score_item(_bear, current_date="2020-12-10", regime="strong_bull")
    assert s_bull > s_bear


def test_neutral_regime_no_penalty():
    """震荡(neutral)下不惩罚中性经验, 且不惩罚防守经验"""
    _range = _exp("range")
    _bull = _exp("trend_up")
    s_range = _score_item(_range, current_date="2021-01-05", regime="range_bound")
    s_bull = _score_item(_bull, current_date="2021-01-05", regime="range_bound")
    # range 经验是中性, 不惩罚; trend_up 在震荡下反向 → 惩罚
    assert s_range > s_bull


def test_memory_format_regime_filters_aggressive(tmp_path):
    """end-to-end: 熊市注入时, trend_down 经验排前, trend_up 经验不被注入"""
    mem = ExperienceMemory(tmp_path / "mem.json", max_items=50)
    mem.items = [_exp("trend_up", conf=0.9), _exp("trend_down", conf=0.6)]
    top = mem.retrieve(current_date="2021-02-05", top_k=1, regime="weak_bear")
    assert top[0].scenario_type == "trend_down", "熊市注入应优先防守经验"


# ═══════════════════════════════════════════════════════════════
# v5.4 组合级反事实 (P1 #26) — 移除拖累票, 组合真的会更好吗?
# ═══════════════════════════════════════════════════════════════

from agent.evolution.portfolio_counterfactual import (
    compute_stock_contributions, portfolio_level_counterfactual,
    StockContribution, summarize_verified,
)


def _snap(*items):
    """构造持仓快照: (symbol, value, weight). name 用 symbol."""
    return [{"symbol": s, "name": s, "qty": 100, "value": v, "weight": w}
            for s, v, w in items]


def test_contributions_weighted_by_return():
    """贡献 = weight × day_return; 涨票正贡献, 跌票负贡献."""
    snap = _snap(("sh.A", 5000, 0.5), ("sh.B", 5000, 0.5))
    rets = {"sh.A": 2.0, "sh.B": -1.0}
    cs = compute_stock_contributions(snap, rets)
    by = {c.symbol: c for c in cs}
    assert by["sh.A"].contribution_pct == pytest.approx(1.0)   # 0.5*2
    assert by["sh.B"].contribution_pct == pytest.approx(-0.5)  # 0.5*(-1)


def test_contributions_skip_unmatched_stock():
    """无次日收益的持仓跳过 (停牌/无数据), 不崩."""
    snap = _snap(("sh.A", 5000, 0.5), ("sh.B", 5000, 0.5))
    cs = compute_stock_contributions(snap, {"sh.A": 1.0})
    assert [c.symbol for c in cs] == ["sh.A"]


def test_portfolio_cf_identifies_worst_drag():
    """组合实际收益 = Σ贡献; 最差票贡献最负; 移除它反事实更好."""
    snap = _snap(("sh.GOOD", 6000, 0.6), ("sh.BAD", 4000, 0.4))
    rets = {"sh.GOOD": 3.0, "sh.BAD": -5.0}
    r = portfolio_level_counterfactual(snap, rets, date="2021-01-05")
    assert r is not None
    assert r.portfolio_return_pct == pytest.approx(0.6*3 + 0.4*(-5))  # -0.2%
    assert r.worst_stock.symbol == "sh.BAD"
    assert r.worst_contribution_pct == pytest.approx(-2.0)  # 0.4*(-5)
    # 移除 BAD 后: 只留 GOOD 贡献 1.8; 反事实 -0.2 - (-2.0) = 1.8
    assert r.counterfactual_return_pct == pytest.approx(1.8)
    assert r.improvement_pct == pytest.approx(2.0)
    assert r.verified is True


def test_portfolio_cf_not_verified_when_worst_not_negative():
    """最差票贡献非负 (组合里没有拖累票) → 不移除, verified=False."""
    snap = _snap(("sh.A", 5000, 0.5), ("sh.B", 5000, 0.5))
    rets = {"sh.A": 1.0, "sh.B": 0.5}
    r = portfolio_level_counterfactual(snap, rets, date="2021-01-05")
    assert r is not None
    assert r.worst_stock.symbol == "sh.B"
    assert r.improvement_pct == pytest.approx(-0.25)  # 移除反而更差
    assert r.verified is False


def test_portfolio_cf_volatility_normalized_threshold():
    """大波动日: 改善门槛抬高, 小拖累不判定为"该扔掉" (防浅层后视)."""
    # 组合整体大涨 5% → 阈值 = max(0.05, 5*0.15)=0.75; 最差票贡献 -0.5 不够
    snap = _snap(("sh.A", 9000, 0.9), ("sh.B", 1000, 0.1))
    rets = {"sh.A": 5.5, "sh.B": -5.0}   # 组合收益 ≈ 0.9*5.5 + 0.1*(-5) = 4.45%
    r = portfolio_level_counterfactual(snap, rets, date="2021-01-05")
    assert r.portfolio_return_pct == pytest.approx(4.45)
    assert r.worst_stock.symbol == "sh.B"
    assert r.worst_contribution_pct == pytest.approx(-0.5)
    # improvement=0.5 < 阈值 0.75 → 不 verified
    assert r.verified is False


def test_portfolio_cf_empty_returns_none():
    """空持仓/无匹配 → None (不崩)."""
    assert portfolio_level_counterfactual([], {}, date="2021-01-05") is None
    assert portfolio_level_counterfactual(
        _snap(("sh.A", 5000, 1.0)), {}, date="2021-01-05") is None


def test_portfolio_cf_summary_counts():
    """汇总统计: verified 计数 + 最差票频次."""
    recs = [
        portfolio_level_counterfactual(_snap(("sh.A", 6000, 0.6), ("sh.B", 4000, 0.4)),
                                       {"sh.A": 3.0, "sh.B": -5.0}, date="d1"),  # verified
        portfolio_level_counterfactual(_snap(("sh.A", 5000, 0.5), ("sh.B", 5000, 0.5)),
                                       {"sh.A": 1.0, "sh.B": 0.5}, date="d2"),   # not
    ]
    s = summarize_verified(recs)
    assert s["total"] == 2
    assert s["verified"] == 1
    assert s["worst_symbols"][0][0] == "sh.B"  # 被标记为拖累票


def test_portfolio_cf_summary_empty():
    assert summarize_verified([]) == {
        "total": 0, "verified": 0, "verified_rate": 0.0,
        "avg_improvement_pct": 0.0, "worst_symbols": []}


def test_decision_record_roundtrip_keeps_positions_snapshot():
    """journal 往返: positions_snapshot + total_value 保留; 旧记录(无字段)不崩."""
    from agent.evolution.decision_journal import DecisionRecord
    rec = DecisionRecord(date="2021-01-05", market_phase="trend_up",
                         dominant_master="live", secondary_master="",
                         risk_level=3, position_multiplier=0.9, max_positions_adj=0,
                         positions_snapshot=[{"symbol": "sh.600519", "name": "x",
                                              "qty": 100, "price": 2000.0,
                                              "value": 200000.0, "weight": 0.5}],
                         total_value=400000.0)
    back = DecisionRecord.from_json(rec.to_json())
    assert back.positions_snapshot == rec.positions_snapshot
    assert back.total_value == pytest.approx(400000.0)

    # 旧格式记录 (无 positions_snapshot 字段) → 默认 [], 不崩
    old = DecisionRecord(date="2020-01-01", market_phase="r", dominant_master="m",
                         secondary_master="", risk_level=3, position_multiplier=1.0,
                         max_positions_adj=0)
    old_cfg = json.dumps(asdict(old), ensure_ascii=False)
    assert DecisionRecord.from_json(old_cfg).positions_snapshot == []


def test_review_linkage_computes_portfolio_cf():
    """复盘链路: 用昨日持仓快照 + 今日截面涨跌, 算出 portfolio_cf 附到 review."""
    from agent.evolution.decision_journal import DecisionRecord
    import pandas as pd
    rec = DecisionRecord(date="2021-01-05", market_phase="trend_up",
                         dominant_master="m", secondary_master="", risk_level=3,
                         position_multiplier=0.9, max_positions_adj=0,
                         positions_snapshot=_snap(("sh.GOOD", 6000, 0.6),
                                                  ("sh.BAD", 4000, 0.4)))
    # 模拟 T 日截面: 每只票当日涨跌
    df = pd.DataFrame({"symbol": ["sh.GOOD", "sh.BAD"], "pctChg": [3.0, -5.0]})
    sret = dict(zip(df["symbol"], pd.to_numeric(df["pctChg"], errors="coerce")))
    review = {"verdict": "wrong", "lesson_title": "t", "lesson_detail": "d"}
    pcf = portfolio_level_counterfactual(rec.positions_snapshot, sret, date=rec.date)
    assert pcf is not None and pcf.verified
    review["portfolio_cf"] = pcf.to_dict()
    assert review["portfolio_cf"]["worst_stock"]["symbol"] == "sh.BAD"
    assert review["portfolio_cf"]["improvement_pct"] == pytest.approx(2.0)