"""v5.5 实盘可信度优化 — P0 数据脱节拦截 / 真实基准 / 北交所前缀统一.

覆盖:
P0-1 数据脱节拦截: 缓存滞后>3天 → 降级 dry-run (不真交易)
P0-2 benchmark 接上证指数: alpha 反映"是否跑赢大盘"而非"是否正收益"
P0-3 北交所前缀统一为 8/4: 腾讯行情取对板块
"""

from datetime import date

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# P0-1 数据脱节拦截
# ═══════════════════════════════════════════════════════════════

from simulation.daily_runner import _effective_dry_run_for_lag


def test_p01_lag_over_3_degrades_to_dry_run():
    """缓存滞后>3天且非显式dry-run → 强制只读, 不真交易."""
    analysis = {"data_lag_days": 7, "data_source": "cache"}
    assert _effective_dry_run_for_lag(analysis, base_dry_run=False, skip_analyze=False) is True


def test_p01_fresh_data_normal_trading():
    """数据新鲜(lag<=3) → 维持原值(正常交易)."""
    analysis = {"data_lag_days": 2, "data_source": "cache"}
    assert _effective_dry_run_for_lag(analysis, base_dry_run=False, skip_analyze=False) is False
    # live 数据 lag=0
    assert _effective_dry_run_for_lag({"data_lag_days": 0}, base_dry_run=False, skip_analyze=False) is False


def test_p01_explicit_dry_run_stays():
    """显式 --dry-run 恒为 True, 即使数据新鲜."""
    analysis = {"data_lag_days": 0}
    assert _effective_dry_run_for_lag(analysis, base_dry_run=True, skip_analyze=False) is True


def test_p01_skip_analyze_no_degrade():
    """skip_analyze(无analysis) → 不降级, 维持原值."""
    assert _effective_dry_run_for_lag(None, base_dry_run=False, skip_analyze=True) is False
    assert _effective_dry_run_for_lag(None, base_dry_run=False, skip_analyze=False) is False


def test_p01_boundary_lag_3_not_degrade():
    """边界: lag=3 不降级 (仅 >3 才降)."""
    analysis = {"data_lag_days": 3}
    assert _effective_dry_run_for_lag(analysis, base_dry_run=False, skip_analyze=False) is False


# ═══════════════════════════════════════════════════════════════
# P0-2 benchmark 接上证指数
# ═══════════════════════════════════════════════════════════════

from simulation.daily_runner import _index_benchmark_ret


def _make_index():
    # 上证收盘序列: 7/28=2950, 8/01=3000, 8/08=3100
    dates = pd.to_datetime(["2026-07-28", "2026-08-01", "2026-08-08"])
    return pd.Series([2950.0, 3000.0, 3100.0], index=dates).sort_index()


def test_p02_positive_benchmark():
    """决策日→回顾日指数上涨 → benchmark>0, alpha 反映跑赢大盘."""
    idx = _make_index()
    ret = _index_benchmark_ret(idx, "2026-07-28", "2026-08-08")
    assert ret == pytest.approx(3100 / 2950 - 1, abs=1e-6)
    assert ret > 0


def test_p02_takes_last_close_on_or_before_date():
    """取各日期最后一个 <= 该日收盘 (周末/节假日对齐)."""
    idx = _make_index()
    # 7/30 无数据 → 用 7/28 的 2950
    ret = _index_benchmark_ret(idx, "2026-07-30", "2026-08-08")
    assert ret == pytest.approx(3100 / 2950 - 1, abs=1e-6)


def test_p02_none_or_empty_returns_zero():
    assert _index_benchmark_ret(None, "2026-07-28", "2026-08-08") == 0.0
    assert _index_benchmark_ret(pd.Series(dtype=float), "2026-07-28", "2026-08-08") == 0.0


def test_p02_no_date_index_returns_zero():
    # 无 DatetimeIndex → 无法定位日期, 回退 0 (不抛错)
    s = pd.Series([2950.0, 3000.0, 3100.0])
    assert _index_benchmark_ret(s, "2026-07-28", "2026-08-08") == 0.0


def test_p02_bad_dates_returns_zero():
    idx = _make_index()
    assert _index_benchmark_ret(idx, "not-a-date", "2026-08-08") == 0.0


# ═══════════════════════════════════════════════════════════════
# P0-3 北交所前缀统一
# ═══════════════════════════════════════════════════════════════

from simulation.daily_runner import _tencent_prefix


def test_p03_sh_prefix():
    assert _tencent_prefix("600519") == "sh"


def test_p03_sz_prefix():
    assert _tencent_prefix("000858") == "sz"
    assert _tencent_prefix("300750") == "sz"   # 创业板
    assert _tencent_prefix("002594") == "sz"


def test_p03_bse_prefix_8_4():
    # 北交所正确判据 8/4 (原 startswith('9') 取错行情)
    assert _tencent_prefix("830799") == "bj"
    assert _tencent_prefix("430047") == "bj"


def test_p03_bse_new_920_prefix():
    # 北交所新代码 920 开头
    assert _tencent_prefix("920001") == "bj"


# ═══════════════════════════════════════════════════════════════
# P1-4 组合级反事实闭环
# ═══════════════════════════════════════════════════════════════

from agent.evolution.portfolio_counterfactual import (
    PortfolioCounterfactualResult, StockContribution,
    accumulate_drag_history, drag_experiences,
)


def _pcf_result(symbol, imp, name="测试股", verified=True):
    return PortfolioCounterfactualResult(
        date="2026-08-09", portfolio_return_pct=-1.0, counterfactual_return_pct=-1.0 + imp,
        improvement_pct=imp, worst_stock=StockContribution(symbol=symbol, name=name),
        worst_contribution_pct=-imp, n_positions=3, n_match=3, verified=verified,
    )


def test_p14_accumulate_history_counts_verified():
    h = {}
    h = accumulate_drag_history(h, [_pcf_result("sz.000001", 0.5, "平安")], "2026-08-09")
    h = accumulate_drag_history(h, [_pcf_result("sz.000001", 0.3, "平安")], "2026-08-10")
    assert h["sz.000001"]["count"] == 2
    assert h["sz.000001"]["last_name"] == "平安"
    # 平均提升 = (0.5+0.3)/2
    assert h["sz.000001"]["avg_improvement_pct"] == pytest.approx(0.4, abs=1e-6)


def test_p14_non_verified_not_counted():
    h = accumulate_drag_history({}, [_pcf_result("sz.000001", 0.5, verified=False)], "2026-08-09")
    assert h == {}


def test_p14_drag_experiences_threshold():
    """同一票 >=2 次才写经验; 只 1 次不写 (防单次偶然)."""
    h = {"sz.000001": {"count": 1, "avg_improvement_pct": 0.5, "last_name": "平安"}}
    assert drag_experiences(h, "2026-08-10") == []
    h["sz.000001"]["count"] = 2
    items = drag_experiences(h, "2026-08-10")
    assert len(items) == 1
    assert items[0].scenario_type == "range"
    assert "downweight" in items[0].tags
    assert "组合拖累票" in items[0].tags
    # 置信度随次数提升
    assert items[0].confidence == pytest.approx(0.7, abs=1e-6)


def test_p14_drag_experience_confidence_caps():
    h = {"sz.000001": {"count": 10, "avg_improvement_pct": 0.5, "last_name": "平安"}}
    items = drag_experiences(h, "2026-08-10")
    assert items[0].confidence <= 0.9


# ═══════════════════════════════════════════════════════════════
# P1-5 实盘诊断喂真实拥挤度
# ═══════════════════════════════════════════════════════════════

from analysis.crowding import market_crowding


def _mk_df(n=200, col="turn_pct_60d"):
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame({col: rng.random(n)})


def test_p15_turnover_fallback_shape():
    """实盘无 turn_pct_60d 但有用 turnover → 回退 cross-sectional 分位, 输出同形状."""
    df = _mk_df(col="turnover")
    cd = market_crowding(df)
    assert set(cd.keys()) == {"score", "signal", "hot_ratio"}
    assert 0.0 <= cd["score"] <= 100.0
    assert cd["signal"] in ("hot", "warm", "cool")
    assert 0.0 <= cd["hot_ratio"] <= 1.0


def test_p15_hot_turnover_detected():
    """large 高尾 turnover → 极端活跃广度 → hot."""
    import numpy as np
    turnover = [1.0] * 150 + [40.0] * 50  # 25% 极高换手
    df = pd.DataFrame({"turnover": turnover})
    cd = market_crowding(df)
    assert cd["signal"] == "hot"
    assert cd["score"] >= 60


def test_p15_cool_when_missing_data():
    """无 turn_pct_60d 也无 turnover → 中性 cool, 不崩."""
    df = pd.DataFrame({"price": [1, 2, 3]})
    cd = market_crowding(df)
    assert cd == {"score": 50.0, "signal": "cool", "hot_ratio": 0.0}


def test_p15_priority_uses_turn_pct_60d():
    """优先用 turn_pct_60d (即使有 turnover 也先用 60日分位)."""
    df = pd.DataFrame({
        "turn_pct_60d": [0.5] * 200,
        "turnover": [30.0] * 200,
    })
    cd = market_crowding(df)
    # turn_pct_60d 全 0.5 (不极端) → 不过热
    assert cd["signal"] == "cool"


# ═══════════════════════════════════════════════════════════════
# P1-6 消除共享 journal 污染
# ═══════════════════════════════════════════════════════════════

from simulation.daily_runner import _init_evolution_system


def test_p16_analysis_path_uses_separate_journal():
    """单股分析路径用独立 journal/memory/evolution, 不污染 diag_journal (市场级复盘)."""
    diag_j, diag_m, diag_e = _init_evolution_system(name="diag")
    ana_j, ana_m, ana_e = _init_evolution_system(name="analysis")
    assert diag_j.path.name == "diag_journal.jsonl"
    assert ana_j.path.name == "analysis_journal.jsonl"
    # 三个组件路径全部隔离, 分析路径不写市场级 diag 文件
    assert str(diag_j.path) != str(ana_j.path)
    assert str(diag_m.path) != str(ana_m.path)
    assert str(diag_e.path) != str(ana_e.path)


# ═══════════════════════════════════════════════════════════════
# P1-7 反事实置信度写入prompt
# ═══════════════════════════════════════════════════════════════

from agent.evolution.experience_memory import ExperienceMemory
from agent.evolution.daily_review import ExperienceItem


def _mk_item(tags, confidence=0.7, title="t", detail="d", verdict="wrong"):
    return ExperienceItem(
        id="x", date="2026-08-09", scenario_type="trend_up", verdict=verdict,
        lesson_title=title, lesson_detail=detail, master_used="利弗莫尔",
        risk_level_given=3, actual_outcome="", confidence=confidence, tags=list(tags),
    )


def test_p17_prompt_shows_confidence(tmp_path):
    mem = ExperienceMemory(tmp_path / "m.json")
    mem.items = [_mk_item(["high_confidence"], confidence=0.9)]
    txt = mem.format_for_prompt(current_date="2026-08-10")
    assert "置信度90%" in txt
    assert "★高置信" in txt


def test_p17_prompt_shows_cf_verified(tmp_path):
    mem = ExperienceMemory(tmp_path / "m.json")
    mem.items = [_mk_item(["cf_verified"], confidence=0.7)]
    txt = mem.format_for_prompt(current_date="2026-08-10")
    assert "反事实验证通过" in txt


def test_p17_prompt_shows_drag_warning(tmp_path):
    mem = ExperienceMemory(tmp_path / "m.json")
    mem.items = [_mk_item(["组合拖累票", "portfolio_cf"], confidence=0.8)]
    txt = mem.format_for_prompt(current_date="2026-08-10")
    assert "组合拖累警示" in txt


def test_p17_prompt_plain_confidence_only(tmp_path):
    mem = ExperienceMemory(tmp_path / "m.json")
    mem.items = [_mk_item([], confidence=0.5)]
    txt = mem.format_for_prompt(current_date="2026-08-10")
    assert "置信度50%" in txt
    assert "★" not in txt
    assert "验证" not in txt


# ═══════════════════════════════════════════════════════════════
# P2-8 回放与实盘逻辑分叉收拢
# ═══════════════════════════════════════════════════════════════

from simulation.daily_runner import _apply_adversarial_gate


def test_p28_replay_buy_sets_peak_for_trailing():
    """回放买入记录 peak=入场价, 供移动止损."""
    from scripts.historical_replay import ReplayPortfolio
    pf = ReplayPortfolio()
    pf.cash = 100000
    ok = pf.buy("sz.000001", "平安", 10.0, 100, "2026-08-09", stop=9.3, take=11.2)
    assert ok
    # peak = 入场价 (含默认滑点) → 10.0 * (1+10bp) = 10.01
    assert pf.positions["sz.000001"]["peak"] == pytest.approx(10.01, abs=1e-6)
    assert pf.positions["sz.000001"]["stop"] == 9.3


def test_p28_float32_price_does_not_pollute_cash():
    """端到端暴露: df float32 价格入账会污染 self.cash → weight float32 → JSON 序列化失败.
    buy/sell 强转 float, cash 始终 Python float, 持仓快照可 JSON 序列化."""
    import json
    import numpy as np
    from scripts.historical_replay import ReplayPortfolio
    pf = ReplayPortfolio()
    pf.cash = 100000.0
    # 用 float32 价格买入 (模拟 df 列)
    ok = pf.buy("sz.000001", "平安", np.float32(10.0), np.int32(100), "2026-08-09")
    assert ok
    assert type(pf.cash) is float, f"cash 被污染为 {type(pf.cash)}"
    # 持仓快照 weight 必须可 JSON 序列化 (float32 不能被 json 序列化)
    _snap = [{"symbol": "sz.000001", "qty": int(p["qty"]), "price": float(p["entry_price"]),
              "value": float(p["qty"] * p["entry_price"]), "weight": 0.0} for p in pf.positions.values()]
    json.dumps(_snap)  # 不应抛 TypeError
    # 卖出同样不污染
    pf.sell("sz.000001", np.float32(11.0), "2026-08-10")
    assert type(pf.cash) is float


def test_p28_adversarial_gate_reused_in_replay_shape():
    """复用 _apply_adversarial_gate: 分歧>=2 且对抗更保守 → 风险+1级, 仓位收敛."""
    diag = {"risk_level": 3, "position_multiplier": 1.2,
            "adversarial_risk_level": 5}
    out = _apply_adversarial_gate(dict(diag))
    assert out["adversarial_applied"] == "conservative"
    assert out["risk_level"] == 4          # 3→4
    assert out["position_multiplier"] < 1.2  # 向 1.0 收敛
    assert out["adversarial_divergence"] == 2


def test_p28_adversarial_gate_noop_when_agrees():
    """分歧<2 → 不干预 (与回放默认缺失时行为一致)."""
    diag = {"risk_level": 3, "position_multiplier": 1.2, "adversarial_risk_level": 3}
    out = _apply_adversarial_gate(dict(diag))
    assert out["risk_level"] == 3
    assert out["position_multiplier"] == 1.2


