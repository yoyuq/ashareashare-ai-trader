"""analysis.attribution — alpha/beta 归因层单元测试 (阶段1 #106)。

用可手算的案例验证: 完美 beta / 纯 alpha / 日期对齐 / 数据不足 / 恒等式 / 净值曲线等价。
"""

import numpy as np
import pandas as pd
import pytest

from analysis.attribution import (
    alpha_beta_attribution,
    attribute_equity_curves,
    load_index_series,
    two_factor_attribution,
    two_factor_attribute_equity_curves,
)


def _series(values, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_perfect_beta_two_zero_alpha():
    """策略 = 2×基准 → beta=2, r²=1, 年化 alpha=0。"""
    b = _series([0.01, -0.02, 0.03, 0.01, 0.005])
    p = _series([0.02, -0.04, 0.06, 0.02, 0.01])
    r = alpha_beta_attribution(p, b)
    assert r["n_aligned"] == 5
    assert r["beta"] == pytest.approx(2.0, abs=1e-6)
    assert r["r_squared"] == pytest.approx(1.0, abs=1e-6)
    assert r["alpha_annualized_pct"] == pytest.approx(0.0, abs=1e-6)
    # beta 贡献 = β × 基准总收益 (3 位小数舍入 → 容差 2bp)
    assert r["beta_contribution_pct"] == pytest.approx(2.0 * r["benchmark_return_pct"], abs=0.002)
    # 总收益恒等式: strategy = beta_contribution + alpha_contribution (舍入容差)
    assert r["strategy_return_pct"] == pytest.approx(
        r["beta_contribution_pct"] + r["alpha_contribution_pct"], abs=0.002)


def test_constant_alpha_zero_beta():
    """策略 = 恒定日收益 (与基准无关) → beta=0, alpha 全部来自截距。"""
    b = _series([0.01, -0.02, 0.03, 0.01, -0.01])
    p = _series([0.001] * 5)
    r = alpha_beta_attribution(p, b)
    assert r["beta"] == pytest.approx(0.0, abs=1e-9)
    # alpha_contribution == 策略总收益 (beta=0 → 没有 beta 贡献; 舍入容差)
    assert r["alpha_contribution_pct"] == pytest.approx(r["strategy_return_pct"], abs=0.002)
    # 年化 alpha = (1+0.001)^252 - 1 (3 位小数舍入 → 容差 1bp)
    expected_annual = ((1.001) ** 252 - 1.0) * 100.0
    assert r["alpha_annualized_pct"] == pytest.approx(expected_annual, abs=0.001)


def test_alignment_on_date_intersection():
    """两序列日期不同 → 只按交集对齐, 不误配。"""
    p = _series([0.01, 0.02, 0.03, 0.04], start="2026-01-01")
    b = _series([0.02, 0.03, 0.04, 0.05], start="2026-01-02")  # 错开一天
    r = alpha_beta_attribution(p, b)
    # 交集 = 3 天 (01-02, 01-05, 01-06 之类, 取决于工作日)
    assert r["n_aligned"] == 3


def test_insufficient_data_returns_nan():
    """对齐后 < 3 点 → 返回 NaN, n_aligned=0, 不抛异常。"""
    p = _series([0.01, 0.02], start="2026-01-01")
    b = _series([0.01, 0.02], start="2026-01-01")
    r = alpha_beta_attribution(p, b)
    assert r["n_aligned"] == 0
    assert np.isnan(r["beta"])
    assert np.isnan(r["alpha_contribution_pct"])


def test_nan_handling():
    """含 NaN 的日收益被剔除, 不污染回归。"""
    b = _series([0.01, -0.02, 0.03, 0.01])
    p = _series([0.02, np.nan, 0.06, 0.02])
    r = alpha_beta_attribution(p, b)
    assert r["n_aligned"] == 3
    assert r["beta"] == pytest.approx(2.0, abs=1e-6)


def test_equity_curve_equivalent_to_returns():
    """净值曲线包装与直接日收益输入等价。"""
    b_ret = _series([0.01, -0.02, 0.03, 0.01, 0.005])
    p_ret = _series([0.02, -0.04, 0.06, 0.02, 0.01])
    b_eq = (1.0 + b_ret).cumprod()
    p_eq = (1.0 + p_ret).cumprod()
    r_ret = alpha_beta_attribution(p_ret, b_ret)
    r_eq = attribute_equity_curves(p_eq, b_eq)
    assert r_eq["beta"] == pytest.approx(r_ret["beta"], abs=1e-6)
    assert r_eq["alpha_annualized_pct"] == pytest.approx(r_ret["alpha_annualized_pct"], abs=1e-6)


def test_excess_return_decomposition():
    """excess = (β−1)×benchmark + alpha_contribution (跑赢量拆解)。"""
    b = _series([0.01, -0.02, 0.03, 0.01, 0.005, 0.01])
    p = _series([0.015, -0.03, 0.045, 0.015, 0.008, 0.015])
    r = alpha_beta_attribution(p, b)
    # excess_return_pct == strategy − benchmark (舍入容差)
    assert r["excess_return_pct"] == pytest.approx(
        r["strategy_return_pct"] - r["benchmark_return_pct"], abs=0.002)


def test_load_index_series_missing_file(tmp_path):
    """文件不存在 → 返回 None (不抛异常)。"""
    assert load_index_series(tmp_path, "2026-01-01", "2026-12-31") is None


def test_two_factor_recovers_market_and_size_beta():
    """策略 = 1.0×市场 + 0.5×SMB + 恒定 0.001 → 还原 β_mkt=1, β_smb=0.5, α=0.001/日。"""
    mkt = _series([0.01, -0.02, 0.03, 0.01, 0.005, -0.01, 0.02])
    smb_raw = _series([0.02, 0.01, -0.01, 0.02, 0.01, -0.005, 0.015])
    ew = mkt + smb_raw
    p = mkt + 0.5 * smb_raw + 0.001
    r = two_factor_attribution(p, mkt, ew)
    assert r["n_aligned"] == 7
    assert r["beta_mkt"] == pytest.approx(1.0, abs=1e-6)
    assert r["beta_smb"] == pytest.approx(0.5, abs=1e-6)
    expected_annual = ((1.001) ** 252 - 1.0) * 100.0
    assert r["alpha_annualized_pct"] == pytest.approx(expected_annual, abs=0.001)
    # 市场贡献 = β_mkt × 市场总收益; 小盘贡献 = β_smb × SMB 总收益
    assert r["market_contribution_pct"] == pytest.approx(1.0 * r["market_return_pct"], abs=0.002)
    assert r["size_contribution_pct"] == pytest.approx(0.5 * r["smb_return_pct"], abs=0.002)
    # 恒等式: strategy = market + size + alpha (舍入容差)
    assert r["strategy_return_pct"] == pytest.approx(
        r["market_contribution_pct"] + r["size_contribution_pct"] + r["alpha_contribution_pct"],
        abs=0.002)


def test_two_factor_equity_curve_wrapper_equivalent():
    """双因子净值曲线包装与直接日收益输入等价。"""
    mkt = _series([0.01, -0.02, 0.03, 0.01, 0.005, -0.01, 0.02])
    smb_raw = _series([0.02, 0.01, -0.01, 0.02, 0.01, -0.005, 0.015])
    ew = mkt + smb_raw
    p = mkt + 0.5 * smb_raw + 0.001
    r_ret = two_factor_attribution(p, mkt, ew)
    r_eq = two_factor_attribute_equity_curves(
        (1.0 + p).cumprod(), (1.0 + mkt).cumprod(), (1.0 + ew).cumprod())
    assert r_eq["beta_mkt"] == pytest.approx(r_ret["beta_mkt"], abs=1e-6)
    assert r_eq["beta_smb"] == pytest.approx(r_ret["beta_smb"], abs=1e-6)
    assert r_eq["alpha_annualized_pct"] == pytest.approx(r_ret["alpha_annualized_pct"], abs=1e-6)


def test_two_factor_insufficient_data_returns_nan():
    """对齐后 < 3 点 → 全 NaN, 不抛异常。"""
    mkt = _series([0.01, 0.02], start="2026-01-01")
    ew = _series([0.01, 0.02], start="2026-01-01")
    p = _series([0.01, 0.02], start="2026-01-01")
    r = two_factor_attribution(p, mkt, ew)
    assert r["n_aligned"] == 0
    assert np.isnan(r["beta_mkt"])
