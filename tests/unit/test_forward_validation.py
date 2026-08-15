"""analysis.forward_validation — 前瞻纸面验证闭环单元测试 (阶段2 #108)。

验证: 等权篮子收益 (含停牌 ffill) / 交易日计数 / 预注册判据应用 (成功/失效/未到 horizon) /
扣费口径 / 注册表读写 (判据注册后不被 tracking 改动)。
"""

from datetime import date

import pytest

from analysis import forward_validation as fv


# --------------------------------------------------------------------------- #
# equalweight_return
# --------------------------------------------------------------------------- #

def test_equalweight_return_basic():
    entry = {"a": 100.0, "b": 50.0}
    cur = {"a": 110.0, "b": 55.0}  # 各 +10%
    assert fv.equalweight_return(entry, cur) == pytest.approx(0.10, abs=1e-9)


def test_equalweight_return_ffill_missing():
    """停牌/退市票 (current 缺失) → ffill 入场价, 贡献 0 收益。"""
    entry = {"a": 100.0, "b": 50.0}
    cur = {"a": 110.0}  # b 缺失 → 0 收益
    # a +10%, b 0% → 等权 +5%
    assert fv.equalweight_return(entry, cur) == pytest.approx(0.05, abs=1e-9)


def test_equalweight_return_empty_entry():
    assert fv.equalweight_return({}, {"a": 100.0}) == 0.0


def test_equalweight_return_skips_nonpositive():
    entry = {"a": 100.0, "b": 0.0}  # b 无有效入场价
    cur = {"a": 110.0}
    assert fv.equalweight_return(entry, cur) == pytest.approx(0.10, abs=1e-9)


# --------------------------------------------------------------------------- #
# trading_days_between
# --------------------------------------------------------------------------- #

def test_trading_days_skip_weekend():
    # 周五 08-14 → 周一 08-17 = 1 个交易日 (周末跳过)
    assert fv.trading_days_between(date(2026, 8, 14), date(2026, 8, 17)) == 1


def test_trading_days_same_day_zero():
    assert fv.trading_days_between(date(2026, 8, 14), date(2026, 8, 14)) == 0


def test_trading_days_full_week():
    # 周一 08-17 → 周五 08-21 = 4 个交易日
    assert fv.trading_days_between(date(2026, 8, 17), date(2026, 8, 21)) == 4


# --------------------------------------------------------------------------- #
# compute_status
# --------------------------------------------------------------------------- #

def _bet(entry_date="2026-08-14", horizon=60, cost_bps=0.0, fail_thr=-10.0):
    return {
        "edge_id": "test",
        "entry": {
            "entry_date": entry_date,
            "basket_prices": {"a": 100.0, "b": 50.0},
            "universe_prices": {"a": 100.0, "b": 50.0, "c": 100.0},
            "sh_index_close": 3000.0,
        },
        "construction": {"cost_bps": cost_bps},
        "criterion": {"horizon_trading_days": horizon, "failure_threshold_pp": fail_thr},
    }


def test_compute_status_success_at_horizon():
    """篮子 +10% vs universe +5% vs 指数 +5% → 主/副判据都成功。"""
    bet = _bet(horizon=0)  # horizon=0 → 任意 elapsed 即落定
    cur_b = {"a": 110.0, "b": 55.0}      # +10%
    cur_u = {"a": 105.0, "b": 52.5, "c": 105.0}  # +5%
    st = fv.compute_status(bet, cur_b, cur_u, 3150.0, "2026-08-14")
    assert st["basket_return_pct"] == pytest.approx(10.0, abs=1e-6)
    assert st["universe_return_pct"] == pytest.approx(5.0, abs=1e-6)
    assert st["sh_index_return_pct"] == pytest.approx(5.0, abs=1e-6)
    assert st["selection_alpha_pp"] == pytest.approx(5.0, abs=1e-6)
    assert st["vs_index_pp"] == pytest.approx(5.0, abs=1e-6)
    assert st["primary_success"] is True
    assert st["secondary_success"] is True
    assert st["edge_failing"] is False


def test_compute_status_edge_failing():
    """篮子 -5% vs universe +5% → 偏离 -10pp → 触发 edge_failing。"""
    bet = _bet(horizon=0)
    cur_b = {"a": 95.0, "b": 47.5}      # -5%
    cur_u = {"a": 105.0, "b": 52.5, "c": 105.0}  # +5%
    st = fv.compute_status(bet, cur_b, cur_u, 3000.0, "2026-08-14")
    assert st["selection_alpha_pp"] == pytest.approx(-10.0, abs=1e-6)
    assert st["edge_failing"] is True
    assert st["primary_success"] is False  # alpha 非负 → 负 → 失败


def test_compute_status_horizon_not_reached():
    """未到 horizon → 主/副判据为 None (尚无结论, 不是失败)。"""
    bet = _bet(horizon=60)
    cur_b = {"a": 95.0, "b": 47.5}
    cur_u = {"a": 105.0, "b": 52.5, "c": 105.0}
    st = fv.compute_status(bet, cur_b, cur_u, 3000.0, "2026-08-14")  # 同一天, 0 交易日
    assert st["horizon_reached"] is False
    assert st["primary_success"] is None
    assert st["secondary_success"] is None


def test_compute_status_cost_applied_to_basket_only():
    """31bp 一次性往返上界只扣篮子 (基准 universe 免费)。"""
    bet = _bet(horizon=0, cost_bps=31.0)
    cur_b = {"a": 110.0, "b": 55.0}      # 毛 +10%
    cur_u = {"a": 105.0, "b": 52.5, "c": 105.0}  # +5%
    st = fv.compute_status(bet, cur_b, cur_u, 3150.0, "2026-08-14")
    # 净 = 10% - 0.31% = 9.69%; alpha = 9.69% - 5% = 4.69pp
    assert st["basket_return_pct"] == pytest.approx(9.69, abs=1e-6)
    assert st["selection_alpha_pp"] == pytest.approx(4.69, abs=1e-6)


# --------------------------------------------------------------------------- #
# 注册表读写 (判据注册后 tracking 不覆盖判据/入场)
# --------------------------------------------------------------------------- #

def test_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(fv, "REGISTRY_PATH", tmp_path / "reg.json")
    bet = _bet()
    bet["edge_id"] = "test_edge"
    fv.register_bet(bet)

    reg = fv.load_registry()
    assert "test_edge" in reg["bets"]
    assert reg["bets"]["test_edge"]["criterion"]["horizon_trading_days"] == 60

    # 追加 tracking 不改判据/入场
    fv.append_tracking("test_edge", {"current_date": "2026-08-17", "elapsed_trading_days": 1})
    reg2 = fv.load_registry()
    b = reg2["bets"]["test_edge"]
    assert b["criterion"]["horizon_trading_days"] == 60
    assert b["entry"]["basket_prices"]["a"] == 100.0
    assert len(b["tracking"]) == 1
    assert b["tracking"][0]["current_date"] == "2026-08-17"


def test_registry_append_unknown_edge_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(fv, "REGISTRY_PATH", tmp_path / "reg.json")
    with pytest.raises(KeyError):
        fv.append_tracking("nope", {"current_date": "2026-08-17"})
