"""冷落溢价 (neglect/cold_tilt) 单元测试 — 验证 PreScreener 低换手选股接线。

实证背景 (run_improved_portfolio_ab.py / verify_neglect_tilt.py):
  低换手 bottom-100 tilt 5/5 窗口跑赢上证综指 return+Sharpe (小盘+冷落系统性 beta)。
本测试锁定接线正确性, 不重新估计 alpha。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.pre_screener import PreScreener, REGIME_WEIGHTS  # noqa: E402


def _sample_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "code": [f"sh.{600000 + i}" for i in range(n)],
        "name": ["测试"] * n,
        "price": rng.uniform(5, 60, n),
        "pct_change": rng.uniform(-5, 5, n),
        "volume": rng.uniform(1e6, 5e7, n),
        "amount": rng.uniform(1e8, 5e9, n),
        "turnover": rng.uniform(0.3, 20, n),
        "pe_ttm": rng.uniform(8, 80, n),
        "pb": rng.uniform(0.8, 8, n),
        "total_mv": rng.uniform(2e9, 5e10, n),
    })


def test_neglect_present_in_all_regimes():
    for regime, w in REGIME_WEIGHTS.items():
        assert "neglect" in w, f"{regime} 缺少 neglect 权重"
        assert w["neglect"] == 0.15
        nums = {k: v for k, v in w.items() if k != "rationale"}
        assert all(isinstance(v, (int, float)) for v in nums.values())


def test_neglect_factor_rewards_low_turnover():
    df = _sample_df()
    r = PreScreener().screen(df, regime="range_bound", top_n=50, neglect=True)
    assert "score_neglect" in r.df.columns
    # 换手率最低的 5 只冷落分应高于换手率最高的 5 只
    cold = r.df.nsmallest(5, "turnover")["score_neglect"].mean()
    hot = r.df.nlargest(5, "turnover")["score_neglect"].mean()
    assert cold > hot


def test_neglect_off_removes_score_column():
    df = _sample_df()
    r = PreScreener().screen(df, regime="range_bound", top_n=50, neglect=False)
    assert "score_neglect" not in r.df.columns


def test_cold_tilt_selects_lowest_turnover():
    df = _sample_df()
    top_n = 50
    r = PreScreener().screen(df, regime="range_bound", top_n=top_n, cold_tilt=True)
    assert len(r.df) <= top_n
    # 冷落模式选出的应是「通过硬过滤后的 universe」里换手率最低的 top_n
    assert r.df["turnover"].median() < df["turnover"].median()
    # 硬过滤 (仅 amount 分位 + price + turnover 下限, 本样本其余全过) 决定存活集
    survived = df[(df["amount"] >= df["amount"].quantile(0.15))
                  & (df["price"] >= 3.0) & (df["turnover"] >= 0.20)]
    expected = survived.nsmallest(len(r.df), "turnover")["turnover"].sort_values().tolist()
    got = r.df["turnover"].sort_values().tolist()
    assert np.allclose(expected, got)
