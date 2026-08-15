"""v5.6 P1-11 过拟合检测真实传参/统一入口

覆盖:
  - Walk-Forward 未提供 oos_returns 时返回 NaN (不再恒 0)
  - 参数稳健性无参数扫描时返回 NaN (不再静默 0); 提供参数值列时算逐参数相关性
  - DSR 单一来源: 公开 deflated_sharpe_ratio 与 OverfittingGuard 内部口径一致
"""

import numpy as np
import pandas as pd
import pytest

from backtest.overfitting import (
    OverfittingGuard,
    deflated_sharpe_ratio,
)


def _returns(n=120, seed=5):
    return pd.Series(np.random.default_rng(seed).normal(0.001, 0.012, n))


# ═══════════════════════════════════════════════════════════════
# Walk-Forward 语义 (NaN 未定义, 非恒 0)
# ═══════════════════════════════════════════════════════════════

def test_walk_forward_nan_when_no_oos():
    """未提供 oos_returns → WF 应为 NaN 且标记未定义 (不伪装成 0)"""
    guard = OverfittingGuard()
    rep = guard.evaluate(returns=_returns())
    assert np.isnan(rep.walk_forward_sharpe)
    assert np.isnan(rep.walk_forward_stability)
    assert rep.details["walk_forward_defined"] is False


def test_walk_forward_defined_with_oos():
    """提供足够长 (≥63 且跨≥2季度) 的 oos_returns → WF 应被真实计算"""
    idx = pd.date_range("2021-01-01", periods=200, freq="B")
    oos = pd.Series(np.random.default_rng(7).normal(0.0005, 0.01, 200), index=idx)
    guard = OverfittingGuard()
    rep = guard.evaluate(returns=_returns(), oos_returns=oos)
    assert rep.details["walk_forward_defined"] is True
    assert np.isfinite(rep.walk_forward_sharpe)


def test_walk_forward_nan_when_too_short():
    """oos_returns 不足一个季度 → WF NaN (未定义)"""
    guard = OverfittingGuard()
    rep = guard.evaluate(returns=_returns(), oos_returns=pd.Series([0.01, -0.02, 0.01]))
    assert np.isnan(rep.walk_forward_sharpe)
    assert rep.details["walk_forward_defined"] is False


# ═══════════════════════════════════════════════════════════════
# 参数稳健性 (NaN 语义 + 真实参数列)
# ═══════════════════════════════════════════════════════════════

def test_parameter_sensitivity_nan_when_undefined():
    """无 param_results → 参数稳健性 NaN (未定义), 而非静默 0"""
    guard = OverfittingGuard()
    rep = guard.evaluate(returns=_returns())
    assert np.isnan(rep.parameter_sensitivity)
    assert rep.details["parameter_sensitivity_metric"] == "sharpe_cv"
    assert rep.details["parameter_sensitivity_corr"] is None


def test_parameter_sensitivity_nan_for_single_sharpe():
    """仅 1 个 sharpe 样本 → 无法估变异系数, 返回 NaN"""
    guard = OverfittingGuard()
    rep = guard.evaluate(
        returns=_returns(),
        param_results=pd.DataFrame({"variant": ["a"], "sharpe": [1.0]}),
    )
    assert np.isnan(rep.parameter_sensitivity)


def test_parameter_sensitivity_cv_with_sharpes():
    """跨变体 Sharpe 变异系数 (CV) 应与手算一致"""
    sharpes = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.15])
    expected_cv = float(np.std(sharpes) / abs(np.mean(sharpes)))
    param_results = pd.DataFrame({"variant": list(range(len(sharpes))), "sharpe": sharpes})

    guard = OverfittingGuard()
    rep = guard.evaluate(returns=_returns(), param_results=param_results)
    assert rep.parameter_sensitivity == pytest.approx(expected_cv, abs=1e-3)
    assert rep.details["parameter_sensitivity_metric"] == "sharpe_cv"
    assert rep.details["parameter_sensitivity_corr"] is None


def test_param_sensitivity_meta_detects_param_columns():
    """提供真实参数值列 → 逐参数相关性 (|Pearson r|) 被计算并暴露"""
    param_results = pd.DataFrame({
        "fast": [5, 10, 15, 20, 25, 30, 35, 40],
        "sharpe": [1.0, 1.1, 1.3, 1.2, 1.0, 0.8, 0.6, 0.5],
    })
    guard = OverfittingGuard()
    rep = guard.evaluate(returns=_returns(), param_results=param_results)
    assert rep.details["parameter_sensitivity_metric"] == "param_corr"
    corr = rep.details["parameter_sensitivity_corr"]
    assert corr is not None
    assert 0.0 <= corr <= 1.0


# ═══════════════════════════════════════════════════════════════
# DSR 单一来源
# ═══════════════════════════════════════════════════════════════

def test_deflated_sharpe_ratio_matches_guard():
    """公开 deflated_sharpe_ratio 与 OverfittingGuard 内部 DSR 口径一致"""
    returns = _returns(seed=11)
    sr = returns.mean() / max(returns.std(), 1e-10)
    skew = float(returns.skew())
    kurt = float(returns.kurt())

    deflated, pval = deflated_sharpe_ratio(
        sharpe=sr, n_trials=1, n_obs=len(returns),
        skew=skew, kurt_excess=kurt, annualize=True,
    )
    guard = OverfittingGuard()
    rep = guard.evaluate(returns=returns)

    assert deflated == pytest.approx(rep.deflated_sharpe, abs=1e-3)
    assert pval == pytest.approx(rep.deflated_sharpe_pvalue, abs=1e-4)


def test_deflated_sharpe_ratio_nonpositive_sharpe():
    """非正 Sharpe → DSR 退化 (0, 1.0)"""
    assert deflated_sharpe_ratio(sharpe=0.0) == (0.0, 1.0)
    assert deflated_sharpe_ratio(sharpe=-0.5, n_trials=5) == (0.0, 1.0)


def test_deflated_sharpe_ratio_k_penalty():
    """试验次数越多 (K↑), 多重检验基准越高 → 同一 Sharpe 的 p 值上升 (更不显著)"""
    sharpe, n_obs = 0.05, 252
    _, p1 = deflated_sharpe_ratio(sharpe=sharpe, n_trials=1, n_obs=n_obs)
    _, p8 = deflated_sharpe_ratio(sharpe=sharpe, n_trials=8, n_obs=n_obs)
    assert p8 >= p1 - 1e-9
