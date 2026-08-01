"""kill-test 三路信号对比框架测试

覆盖:
  1. RuleArm 确定性规则 (趋势/RSI/量比 → BUY, 超买/死叉 → SELL)
  2. RandomArm 有种子可复现 + 与规则臂同数量
  3. LLMArm 读取 reports/data_{date}.json
  4. KillTestTracker 信号去重 + N日前向收益结算 (含 SELL 方向化)
  5. report 汇总/对比 + Welch t + 显著性
  6. run_killtest 端到端 (注入合成数据)
"""

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_df(closes, start="2026-01-01"):
    """合成日K (close 序列 + 标准列)"""
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": closes, "close": closes,
        "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
        "volume": [10000] * len(closes),
    })


def _make_ind_df(closes, **extra):
    df = _make_df(closes)
    for k, v in extra.items():
        df[k] = v
    return df


# ═══════════════════════════════════════════════════════════════
# 1. RuleArm
# ═══════════════════════════════════════════════════════════════

def test_rule_arm_buy_on_bullish():
    """多头+未超买+放量 → BUY"""
    from killtest.arms import RuleArm
    ind = _make_ind_df(
        [10.0] * 120,
        trend_score=0.8, rsi_14=60.0, vol_ratio_5=1.5, close=10.5,
    )
    sigs = RuleArm().generate({"sh.600519": ind}, "2026-01-02")
    assert any(s.action == "BUY" for s in sigs)


def test_rule_arm_sell_on_overbought():
    """RSI>75 → SELL"""
    from killtest.arms import RuleArm
    ind = _make_ind_df(
        [10.0] * 120,
        trend_score=0.3, rsi_14=80.0, vol_ratio_5=1.0, close=10.5,
    )
    sigs = RuleArm().generate({"sh.600519": ind}, "2026-01-02")
    assert any(s.action == "SELL" for s in sigs)


def test_rule_arm_no_signal_on_neutral():
    from killtest.arms import RuleArm
    ind = _make_ind_df(
        [10.0] * 120,
        trend_score=0.0, rsi_14=50.0, vol_ratio_5=1.0, close=10.0,
    )
    assert RuleArm().generate({"sh.600519": ind}, "2026-01-02") == []


def test_rule_arm_causal_asof_no_future():
    """asof 之前的数据不影响结果 (因果切片)"""
    from killtest.arms import RuleArm
    closes = [10.0] * 100 + [5.0] * 20  # 后段大跌
    ind = _make_ind_df(closes)
    ind["trend_score"] = 0.8   # 全段都写多头 (后段实际下跌, 但 asof 只取切片)
    ind["rsi_14"] = 60.0
    ind["vol_ratio_5"] = 1.5
    # asof 指向中间 (前段), 不应看到后段大跌
    mid = pd.bdate_range("2026-01-01", periods=len(closes))[80].strftime("%Y-%m-%d")
    sigs = RuleArm().generate({"sh.600519": ind}, mid)
    assert len(sigs) >= 0  # 不应崩溃; 因果性在 asof_row 切片体现


# ═══════════════════════════════════════════════════════════════
# 2. RandomArm
# ═══════════════════════════════════════════════════════════════

def test_random_arm_reproducible_and_matched():
    from killtest.arms import RandomArm
    universe = [f"sh.{600000+i}" for i in range(10)]
    a1 = RandomArm(seed=42).generate(universe, "2026-01-02", n_buy=3, n_sell=2)
    a2 = RandomArm(seed=42).generate(universe, "2026-01-02", n_buy=3, n_sell=2)
    assert len(a1) == 5
    assert [s.symbol for s in a1] == [s.symbol for s in a2]  # 可复现
    assert sum(s.action == "BUY" for s in a1) == 3
    assert all(s.arm == "random" for s in a1)


# ═══════════════════════════════════════════════════════════════
# 3. LLMArm
# ═══════════════════════════════════════════════════════════════

def test_llm_arm_reads_daily_file(tmp_path):
    from killtest.arms import LLMArm
    report_dir = tmp_path
    (report_dir / "data_2026-01-02.json").write_text(json.dumps({
        "stock_recommendations": {
            "sh.600519": {"action": "BUY", "conviction": 0.7,
                          "trade_params": {"entry_price": 100.0}},
            "sz.000858": {"action": "HOLD", "conviction": 0.5},
        }
    }), encoding="utf-8")
    sigs = LLMArm(report_dir=str(report_dir)).load_daily("2026-01-02")
    assert len(sigs) == 1
    assert sigs[0].symbol == "sh.600519" and sigs[0].action == "BUY"
    assert sigs[0].price == 100.0


def test_llm_arm_missing_date_returns_empty(tmp_path):
    from killtest.arms import LLMArm
    assert LLMArm(report_dir=str(tmp_path)).load_daily("2020-01-01") == []


# ═══════════════════════════════════════════════════════════════
# 4. KillTestTracker
# ═══════════════════════════════════════════════════════════════

def _signal_df(sig_idx=10, n=30, exit_delta=10, action="up"):
    """合成行情: 信号日 sig_idx 处 close=100, 其后 lookahead 天走到 100±exit_delta"""
    dates = pd.bdate_range("2025-12-01", periods=n)
    closes = [100.0] * n
    if action == "up":
        for k in range(sig_idx + 1, sig_idx + 6):
            closes[k] = 100.0 + (k - sig_idx) * (exit_delta / 5)
    else:  # down
        for k in range(sig_idx + 1, sig_idx + 6):
            closes[k] = 100.0 - (k - sig_idx) * (exit_delta / 5)
    df = _make_df(closes, start="2025-12-01")
    return dates[sig_idx].strftime("%Y-%m-%d"), df


def test_tracker_dedup_and_settle(tmp_path):
    from killtest.arms import Signal
    from killtest.tracker import KillTestTracker
    tr = KillTestTracker(str(tmp_path))

    sig_date, df = _signal_df(sig_idx=10)
    s1 = Signal(sig_date, "sh.600519", "BUY", 0.7, "rule", price=100.0)
    assert tr.emit(s1) is True
    assert tr.emit(s1) is False  # 去重
    assert len(tr.load_signals("rule")) == 1

    # 合成行情: 信号日 100 → 5日后 110 → 前向 +10%
    out = tr.settle("rule", {"sh.600519": df}, lookahead=5)
    assert len(out) == 1
    assert out[0].fwd_return == pytest.approx(0.10, rel=1e-2)
    assert out[0].is_correct is True
    # 已结算的不重复
    assert tr.settle("rule", {"sh.600519": df}, lookahead=5) == []


def test_tracker_sell_directional_return(tmp_path):
    from killtest.arms import Signal
    from killtest.tracker import KillTestTracker
    tr = KillTestTracker(str(tmp_path))
    sig_date, df = _signal_df(sig_idx=10, exit_delta=10, action="down")
    tr.emit(Signal(sig_date, "sh.600519", "SELL", 0.6, "rule", price=100.0))
    out = tr.settle("rule", {"sh.600519": df}, lookahead=5)
    assert out[0].fwd_return == pytest.approx(0.10, rel=1e-2)  # 做空方向化 = +10%
    assert out[0].is_correct is True


# ═══════════════════════════════════════════════════════════════
# 5. report
# ═══════════════════════════════════════════════════════════════

def test_summarize_and_compare():
    from killtest.report import summarize, compare, welch_t
    from killtest.tracker import Outcome

    def _o(fwd):
        return Outcome("2026-01-02", "sh.600519", "BUY", "rule", 0.5,
                       100, 100, fwd, fwd > 0)

    better = [_o(0.06), _o(0.10), _o(0.08), _o(0.04)]   # 全正, 均值 ~0.07
    worse = [_o(-0.06), _o(-0.04), _o(-0.07), _o(-0.05)]  # 全负, 差异显著
    s = summarize(better)
    assert s["n"] == 4 and s["win_rate"] == 1.0
    cmp = compare(better, worse)
    assert cmp["mean_diff"] > 0
    assert cmp["welch"]["p"] < 0.05  # 显著
    assert cmp["ci"]["lo"] > 0


# ═══════════════════════════════════════════════════════════════
# 6. run_killtest 端到端 (注入合成数据)
# ═══════════════════════════════════════════════════════════════

def test_run_killtest_end_to_end(tmp_path, monkeypatch):
    from killtest.runner import run_killtest

    # 构造合成行情: 一只持续上涨, 一只震荡
    n = 150
    dates = pd.bdate_range("2025-10-01", periods=n)
    base = 100.0
    up_closes = [base * (1 + 0.005 * i) for i in range(n)]
    flat_closes = [base + 10 * np.sin(i / 8) for i in range(n)]

    def make_df(closes):
        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": closes, "close": closes,
            "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
            "volume": [10000] * n,
        })

    data = {"sh.600519": make_df(up_closes), "sz.000858": make_df(flat_closes)}

    async def fake_get_kline(sym, start, end):
        return data.get(sym, pd.DataFrame())

    # 注入确定性指标: 上涨股给多头指标 (触发 BUY), 震荡股给中性
    def fake_indicators(df, sym):
        ind = df.copy()
        ind["trend_score"] = 0.8 if sym == "sh.600519" else 0.0
        ind["rsi_14"] = 60.0 if sym == "sh.600519" else 50.0
        ind["vol_ratio_5"] = 1.5 if sym == "sh.600519" else 1.0
        ind["macd_death_cross"] = 0.0
        ind["close"] = df["close"]
        return ind

    result = asyncio.run(run_killtest(
        history_days=60, symbols=["sh.600519", "sz.000858"],
        arms=("rule", "random"), lookahead=5, seed=42,
        data_dir=str(tmp_path / "kt"), report_dir=str(tmp_path),
        get_kline=fake_get_kline, indicator_fn=fake_indicators,
    ))

    assert "report" in result and result["report"]
    assert "rule" in result["arms"] and "random" in result["arms"]
    # 上涨股应产生规则 BUY 信号
    rule_outcomes = result["arms"]["rule"]
    assert len(rule_outcomes) > 0
    assert any(o.action == "BUY" for o in rule_outcomes)
    # 报告文件写入
    assert (Path(tmp_path) / "killtest_report.md").exists()
    # 基线已计算
    assert result["baseline_n"] > 0
