"""v3.1-deerflow 多智能体协作测试

覆盖三项 DeerFlow 模式移植:
  1. EvaluatorAgent — 综合研判质量评估 (驱动反思回炉)
  2. DecisionValidator — 交易计划硬约束校验 + 拒绝日志落盘
  3. Brainstormer 多视角辩论 — 3视角×多空=6路并行 + 定向修订最弱视角
  4. 全局反思-修订循环 — evaluation 路由 + synthesis 回炉 + trade_params 写回
"""

import asyncio
import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# 通用 fakes
# ═══════════════════════════════════════════════════════════════

class _FakeRouteResult:
    def __init__(self, response):
        self.response = response
        self.tier = SimpleNamespace(value="flash")
        self.cost = 0.0
        self.model_name = "deepseek-v4-flash"


class _FakeRouter:
    """按序返回预设响应, 记录每次调用的 task_type + messages"""
    def __init__(self, responses=None):
        self._queue = list(responses or [])
        self.calls = []          # [(task_type, messages)]
        self.task_types = []

    async def route(self, messages=None, task_type=None, **kw):
        self.calls.append((task_type, messages))
        self.task_types.append(task_type)
        if self._queue:
            return self._queue.pop(0)
        return _FakeRouteResult("")

    def count(self, task_type):
        return sum(1 for t in self.task_types if t == task_type)


def _judge(action="BUY", conviction=0.6, improvement=False, crit_bull="",
           crit_bear="", weak_bull="trend", weak_bear="trend"):
    return _FakeRouteResult(json.dumps({
        "action": action, "conviction": conviction, "score": 7,
        "key_reasons": ["技术金叉"], "risks": ["回调风险"],
        "bull_quality": 4, "bear_quality": 4,
        "improvement_needed": improvement,
        "weakest_bull_lens": weak_bull, "weakest_bear_lens": weak_bear,
        "critique_for_bull": crit_bull, "critique_for_bear": crit_bear,
    }))


def _lens_response(label):
    return _FakeRouteResult(f"### {label} 看涨论据\n1. 论据 (数据: 收盘101.0)")


def _simple_df(close=101.0, prev=100.5):
    return pd.DataFrame({
        "open": [100.0, 100.5, close], "high": [101.0, 101.0, close + 1],
        "low": [99.0, 100.0, close - 1], "close": [100.0, prev, close],
        "volume": [1000, 2000, 3000],
    })


# ═══════════════════════════════════════════════════════════════
# 1. EvaluatorAgent
# ═══════════════════════════════════════════════════════════════

async def _eval_result(final_report, trading_params, recs, critic_results=None):
    from agent.sub_agents.evaluator import EvaluatorAgent
    ev = EvaluatorAgent()
    res = await ev.run(ev._start_context(
        task_id="t", final_report=final_report,
        trading_params=trading_params, stock_recommendations=recs,
        critic_results=critic_results or {}))
    return res.data


def test_evaluator_missing_params_fails():
    """BUY 推荐缺少交易参数 → 不通过 + missing_trade_params"""
    data = asyncio.run(_eval_result(
        '{"recommendations":[{"symbol":"sh.600519","action":"BUY"}]}',
        trading_params={},
        recs={"sh.600519": {"action": "BUY", "conviction": 0.5}},
    ))
    assert data["passed"] is False
    codes = [i["code"] for i in data["evaluation_result"]["issues"]]
    assert "missing_trade_params" in codes
    assert data["evaluation_result"]["score"] < 70


def test_evaluator_clean_report_passes():
    """完整可执行推荐 → 通过, 高分"""
    data = asyncio.run(_eval_result(
        final_report="ok",
        trading_params={"sh.600519": {"entry_price": 100, "stop_loss": 95,
                                      "take_profit": 110, "position_pct": 0.10}},
        recs={"sh.600519": {"action": "BUY", "conviction": 0.6,
                            "key_reasons": ["x"], "risks": ["y"]}},
        critic_results={"sh.600519": {"robustness_score": 90}},
    ))
    assert data["passed"] is True
    assert data["evaluation_result"]["score"] >= 90


def test_evaluator_conviction_contradiction():
    """BUY 却低确信 → 自相矛盾缺陷"""
    data = asyncio.run(_eval_result(
        final_report="ok",
        trading_params={"sh.600519": {"entry_price": 100, "stop_loss": 95,
                                      "take_profit": 110, "position_pct": 0.10}},
        recs={"sh.600519": {"action": "BUY", "conviction": 0.2,
                            "key_reasons": ["x"], "risks": ["y"]}},
    ))
    codes = [i["code"] for i in data["evaluation_result"]["issues"]]
    assert "conviction_contradiction" in codes
    assert data["passed"] is False


def test_evaluator_hallucinated_numbers():
    """报告 JSON 声称入场价 vs 代码计算值偏差>5% → 幻觉数字缺陷"""
    report = json.dumps({"recommendations": [
        {"symbol": "sh.600519", "action": "BUY",
         "entry_price": 120.0, "stop_loss": 95, "take_profit": 130}
    ]})
    data = asyncio.run(_eval_result(
        final_report=report,
        trading_params={"sh.600519": {"entry_price": 101.0, "stop_loss": 95,
                                      "take_profit": 110, "position_pct": 0.10}},
        recs={"sh.600519": {"action": "BUY", "conviction": 0.6,
                            "key_reasons": ["x"], "risks": ["y"]}},
    ))
    codes = [i["code"] for i in data["evaluation_result"]["issues"]]
    assert "hallucinated_numbers" in codes


def test_evaluator_low_robustness():
    """Critic 稳健性过低 → low_robustness 缺陷"""
    data = asyncio.run(_eval_result(
        final_report="ok",
        trading_params={"sh.600519": {"entry_price": 100, "stop_loss": 95,
                                      "take_profit": 110, "position_pct": 0.10}},
        recs={"sh.600519": {"action": "BUY", "conviction": 0.6,
                            "key_reasons": ["x"], "risks": ["y"]}},
        critic_results={"sh.600519": {"robustness_score": 30}},
    ))
    codes = [i["code"] for i in data["evaluation_result"]["issues"]]
    assert "low_robustness" in codes


# ═══════════════════════════════════════════════════════════════
# 2. DecisionValidator
# ═══════════════════════════════════════════════════════════════

_TMP_JOURNAL = None  # 延迟到首次使用时创建 (避免污染真实 simulation_data/)


def _tmp_journal():
    global _TMP_JOURNAL
    if _TMP_JOURNAL is None:
        import tempfile
        _TMP_JOURNAL = tempfile.mktemp(suffix=".jsonl")
    return _TMP_JOURNAL


async def _validate(trading_params, recs, market_data=None, portfolio=None):
    from agent.sub_agents.validator import DecisionValidator
    vd = DecisionValidator()
    vd.JOURNAL_PATH = _tmp_journal()  # 隔离到临时文件, 不污染 simulation_data/
    res = await vd.run(vd._start_context(
        task_id="t", trading_params=trading_params,
        stock_recommendations=recs, market_data=market_data or {},
        portfolio=portfolio))
    return res.data


def test_validator_rejects_bad_stop():
    """止损>=入场 → 拒绝 (critical bad_stop)"""
    data = asyncio.run(_validate(
        {"sh.600519": {"entry_price": 100, "stop_loss": 105, "take_profit": 110,
                       "position_pct": 0.10}},
        {"sh.600519": {"action": "BUY", "conviction": 0.6}},
    ))
    r = data["validation_results"]["sh.600519"]
    assert r["valid"] is False
    assert "bad_stop" in [v["code"] for v in r["violations"]]


def test_validator_rejects_bad_take():
    data = asyncio.run(_validate(
        {"sh.600519": {"entry_price": 100, "stop_loss": 95, "take_profit": 90,
                       "position_pct": 0.10}},
        {"sh.600519": {"action": "BUY", "conviction": 0.6}},
    ))
    r = data["validation_results"]["sh.600519"]
    assert r["valid"] is False
    assert "bad_take" in [v["code"] for v in r["violations"]]


def test_validator_position_out_of_range():
    """仓位超出 [3%,20%] → high 违规"""
    data = asyncio.run(_validate(
        {"sh.600519": {"entry_price": 100, "stop_loss": 95, "take_profit": 110,
                       "position_pct": 0.50}},
        {"sh.600519": {"action": "BUY", "conviction": 0.6}},
    ))
    r = data["validation_results"]["sh.600519"]
    assert "position_range" in [v["code"] for v in r["violations"]]


def test_validator_limit_unbuyable():
    """最新收盘已触及涨停 → warning 无法追高"""
    df = _simple_df(close=110.0, prev=100.0)  # +10% 主板涨停
    data = asyncio.run(_validate(
        {"sh.600519": {"entry_price": 110, "stop_loss": 104, "take_profit": 120,
                       "position_pct": 0.10}},
        {"sh.600519": {"action": "BUY", "conviction": 0.6}},
        market_data={"sh.600519": df},
    ))
    r = data["validation_results"]["sh.600519"]
    codes = [v["code"] for v in r["violations"]]
    assert "limit_unbuyable" in codes


def test_validator_lot_too_small():
    """资金不足以买满1手 → high 违规"""
    data = asyncio.run(_validate(
        {"sh.600519": {"entry_price": 100, "stop_loss": 95, "take_profit": 110,
                       "position_pct": 0.10}},
        {"sh.600519": {"action": "BUY", "conviction": 0.6}},
        portfolio={"capital": 500},  # 500×10% = 50元 < 100股×100元
    ))
    r = data["validation_results"]["sh.600519"]
    assert "lot_too_small" in [v["code"] for v in r["violations"]]


def test_validator_t1_pending():
    """当日买入不可卖出 → warning t1_pending"""
    data = asyncio.run(_validate(
        {"sh.600519": {"entry_price": 100, "stop_loss": 95, "take_profit": 110,
                       "position_pct": 0.10}},
        {"sh.600519": {"action": "SELL", "conviction": 0.6}},
        portfolio={"positions": {"sh.600519": {"quantity": 100,
                                               "buy_date": date.today().isoformat()}}},
    ))
    r = data["validation_results"]["sh.600519"]
    assert "t1_pending" in [v["code"] for v in r["violations"]]


def test_validator_clean_passes_and_journal(tmp_path):
    """完整可执行推荐 → 通过; 违规时 journal 落盘"""
    from agent.sub_agents.validator import DecisionValidator
    journal = tmp_path / "validation_journal.jsonl"

    async def _run(tp, rec):
        vd = DecisionValidator()
        vd.JOURNAL_PATH = journal
        res = await vd.run(vd._start_context(
            task_id="t", trading_params=tp, stock_recommendations=rec,
            market_data={}))
        return res.data

    # 干净推荐 → 通过, 不写 journal
    data = asyncio.run(_run(
        {"sh.600519": {"entry_price": 100, "stop_loss": 95, "take_profit": 110,
                       "position_pct": 0.10}},
        {"sh.600519": {"action": "BUY", "conviction": 0.6}},
    ))
    r = data["validation_results"]["sh.600519"]
    assert r["valid"] is True
    assert data["reject_count"] == 0
    assert not journal.exists()

    # 违规推荐 → 拒绝, journal 落盘可审计
    data2 = asyncio.run(_run(
        {"sh.600519": {"entry_price": 100, "stop_loss": 105, "take_profit": 110,
                       "position_pct": 0.10}},
        {"sh.600519": {"action": "BUY", "conviction": 0.6}},
    ))
    assert data2["reject_count"] == 1
    assert data2["violation_count"] == 1
    lines = journal.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["code"] == "bad_stop"
    assert record["symbol"] == "sh.600519"


def test_limit_pct_for_symbol():
    from agent.sub_agents.validator import limit_pct_for_symbol
    assert limit_pct_for_symbol("sh.600519") == 10.0
    assert limit_pct_for_symbol("sz.300750") == 20.0
    assert limit_pct_for_symbol("sh.688111") == 20.0
    assert limit_pct_for_symbol("bj.831010") == 30.0


# ═══════════════════════════════════════════════════════════════
# 3. Brainstormer 多视角辩论
# ═══════════════════════════════════════════════════════════════

def _make_workflow(router):
    from agent.orchestration.workflow import AnalysisWorkflow
    return AnalysisWorkflow(router=router, knowledge_manager=None)


def test_debate_six_lenses_parallel():
    """3视角×多空=6路并行论证 + Judge 整合 (无修订)"""
    responses = [_lens_response(l) for l in ("技术趋势", "估值与基本面", "风险与市场")] * 2
    responses.append(_judge(action="BUY", conviction=0.6, improvement=False))
    router = _FakeRouter(responses)
    wf = _make_workflow(router)
    df = _simple_df()

    rec = asyncio.run(wf._debate_single_stock(
        "sh.600519", "range_bound", {}, df, {}, [], []))

    # 6 路论证 + 1 次 Judge
    assert router.count("adversarial_debate") == 6
    assert router.count("judge_verdict") == 1
    assert rec["action"] == "BUY"
    assert rec["conviction"] == pytest.approx(0.6)
    assert rec["lenses_used"] == ["trend", "valuation", "risk"]
    # 摘要应标注视角来源
    assert "技术趋势" in rec["bull_summary"]
    assert "估值与基本面" in rec["bull_summary"]


def test_debate_revises_weakest_lens():
    """Judge 指出估值视角最弱 → 只修订该视角多头, 再评估"""
    responses = [_lens_response(l) for l in ("技术趋势", "估值与基本面", "风险与市场")] * 2
    responses.append(_judge(action="BUY", conviction=0.5, improvement=True,
                            weak_bull="valuation", crit_bull="论据不足,补充数据"))
    responses.append(_lens_response("估值与基本面修订"))  # 修订响应
    responses.append(_judge(action="BUY", conviction=0.7, improvement=False))
    router = _FakeRouter(responses)
    wf = _make_workflow(router)

    rec = asyncio.run(wf._debate_single_stock(
        "sh.600519", "range_bound", {}, _simple_df(), {}, [], []))

    # 6 首轮 + 1 修订 = 7 次论证; 2 次 Judge
    assert router.count("adversarial_debate") == 7
    assert router.count("judge_verdict") == 2
    assert rec["conviction"] == pytest.approx(0.7)

    # 修订调用应指向"估值与基本面"视角
    revise_msgs = router.calls[-3][1]  # 修订那次调用的 messages
    joined = json.dumps(revise_msgs, ensure_ascii=False)
    assert "估值与基本面" in joined


# ═══════════════════════════════════════════════════════════════
# 4. 全局反思-修订循环
# ═══════════════════════════════════════════════════════════════

class _FakeKnowledge:
    chroma_available = False

    def get_system_prompt(self, name):
        return "你是A股综合研判Agent。只输出JSON。"


def _synthesis_state():
    df = _simple_df(close=101.0, prev=100.5)
    return {
        "task_id": "t", "date": "2026-08-01", "symbols": ["sh.600519"],
        "stock_recommendations": {"sh.600519": {
            "action": "BUY", "conviction": 0.6,
            "key_reasons": ["x"], "risks": ["y"]}},
        "critic_results": {"sh.600519": {"robustness_score": 90, "flaw_count": 0}},
        "technical_indicators": {}, "market_data": {"sh.600519": df},
        "market_regime": "range_bound", "regime_confidence": 0.5,
        "scan_results": [], "analysis_reports": {}, "strategy_matches": {},
        "backtest_results": {}, "debate_result": {},
        "errors": [], "reflection_round": 0, "synthesis_revisions": [],
    }


def test_route_after_evaluation():
    from langgraph.graph import END
    from agent.orchestration.workflow import AnalysisWorkflow

    route = AnalysisWorkflow._route_after_evaluation
    # 通过 → END
    assert route({"evaluation_result": {"passed": True}}) == END
    # 不通过 + 未达上限 → 回炉
    assert route({"evaluation_result": {"passed": False}, "reflection_round": 0}) == "synthesis_revision"
    # 不通过 + 已达2轮 → END (防止无限回炉)
    assert route({"evaluation_result": {"passed": False}, "reflection_round": 2}) == END


def test_synthesis_writes_back_trade_params_and_validates(tmp_path, monkeypatch):
    """trade_params 写回 stock_recommendations + validation_results 生成"""
    from agent.sub_agents.validator import DecisionValidator
    DecisionValidator.JOURNAL_PATH = tmp_path / "validation_journal.jsonl"
    monkeypatch.setenv("VALIDATION_JOURNAL_PATH", str(tmp_path / "validation_journal.jsonl"))

    report = json.dumps({"recommendations": [
        {"symbol": "sh.600519", "action": "BUY", "conviction": 0.6}]})
    router = _FakeRouter([_FakeRouteResult(report)])
    wf = _make_workflow(router)
    wf.knowledge = _FakeKnowledge()

    state = _synthesis_state()
    out = asyncio.run(wf._synthesis_node(state))

    # 写回 bug 修复: 推荐携带代码计算的 trade_params
    rec = out["stock_recommendations"]["sh.600519"]
    tp = rec.get("trade_params", {})
    assert tp.get("entry_price", 0) > 0
    assert tp["stop_loss"] < tp["entry_price"] < tp["take_profit"]
    # 验证器已运行
    assert "sh.600519" in out["validation_results"]


def test_synthesis_revision_injects_critique():
    """回炉修订: 批评注入第二轮 synthesis, reflection_round 递增"""
    report1 = json.dumps({"recommendations": [
        {"symbol": "sh.600519", "action": "BUY", "conviction": 0.6}]})
    report2 = json.dumps({"recommendations": [
        {"symbol": "sh.600519", "action": "BUY", "conviction": 0.7,
         "entry_price": 101.0, "stop_loss": 95, "take_profit": 110,
         "position_pct": 0.14}]})
    router = _FakeRouter([_FakeRouteResult(report1), _FakeRouteResult(report2)])
    wf = _make_workflow(router)
    wf.knowledge = _FakeKnowledge()

    state = _synthesis_state()
    state["evaluation_result"] = {"critique": "报告缺少 trade_params, 请从代码计算的交易参数中填入"}
    state["reflection_round"] = 0

    out = asyncio.run(wf._synthesis_revision_node(state))

    assert out["reflection_round"] == 1
    assert len(out["synthesis_revisions"]) == 1
    # 批评已注入本轮 synthesis (回炉重跑) 的 user 消息
    assert router.count("daily_synthesis") == 1
    replayed_user = router.calls[0][1][-1]["content"]
    assert "trade_params" in replayed_user
    assert "报告缺少 trade_params" in replayed_user
    # 消费后清空, 避免下轮重复注入
    assert out["synthesis_critique"] == ""


def test_workflow_graph_has_reflection_nodes():
    """LangGraph 图应包含 evaluation + synthesis_revision 节点"""
    from agent.orchestration.workflow import AnalysisWorkflow
    wf = AnalysisWorkflow(router=None, knowledge_manager=None)
    nodes = list(wf.graph.nodes.keys())
    assert "evaluation" in nodes
    assert "synthesis_revision" in nodes
    assert "synthesis" in nodes
