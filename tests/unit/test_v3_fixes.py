"""v3.0 新增修复的回归测试: CSCV-PBO 接线 / DSR 多重检验 / 哈希向量检索确定性"""

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# 1. 哈希向量检索确定性 (ChromaDB 真实余弦相似度的基础)
# ═══════════════════════════════════════════════════════════════

def test_hash_embed_deterministic_and_ordering():
    """同文本余弦=1.0, 不同文本余弦更低 (跨进程 md5 稳定)"""
    from knowledge.manager import _stable_hash_embed

    a = _stable_hash_embed("十字星 趋势反转")
    b = _stable_hash_embed("十字星 趋势反转")
    c = _stable_hash_embed("红三兵 看涨持续")

    sim_ab = sum(x * y for x, y in zip(a, b))
    sim_ac = sum(x * y for x, y in zip(a, c))
    assert abs(sim_ab - 1.0) < 1e-6, "同文本应完全一致"
    assert sim_ac < sim_ab, "不同文本相似度应更低"


# ═══════════════════════════════════════════════════════════════
# 2. CSCV-PBO 接线 + DSR 多重检验 K
# ═══════════════════════════════════════════════════════════════

def _variant_matrix(n_variants: int = 8, n_periods: int = 120, alpha: float = 0.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    cols = {}
    for i in range(n_variants):
        mean = alpha if i == 0 else 0.0
        cols[f"v{i}"] = pd.Series(rng.normal(mean, 0.012, n_periods))
    return pd.DataFrame(cols)


def test_pbo_defined_when_variants_supplied():
    """提供变体矩阵 → 真 CSCV-PBO 不再是 NaN, DSR 用 K=变体数"""
    from backtest.overfitting import OverfittingGuard

    matrix = _variant_matrix(n_variants=8, n_periods=120)
    param_results = pd.DataFrame({
        "variant": list(matrix.columns),
        "sharpe": [matrix[c].mean() / matrix[c].std() for c in matrix.columns],
    })
    guard = OverfittingGuard(n_simulations=50)
    rep = guard.evaluate(returns=matrix.iloc[:, 0], variant_returns=matrix, param_results=param_results)

    assert not np.isnan(rep.pbo), "提供变体矩阵后 PBO 应被真实计算"
    assert 0.0 <= rep.pbo <= 1.0
    assert "pbo_defined" in rep.details and rep.details["pbo_defined"] is True


def test_pbo_nan_without_variants():
    """单一收益序列无法做 CSCV → PBO=NaN 且不参与判定"""
    from backtest.overfitting import OverfittingGuard

    guard = OverfittingGuard()
    rep = guard.evaluate(returns=_variant_matrix(n_variants=1, n_periods=120).iloc[:, 0])
    assert np.isnan(rep.pbo)
    assert rep.details["pbo_defined"] is False


def test_dsr_k_correction_applies():
    """K>1 时多重检验校正应使 p 值上升 (单序列 K=1 时更低)"""
    from backtest.overfitting import OverfittingGuard

    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.001, 0.012, 120))
    guard = OverfittingGuard()

    # K=1
    rep_k1 = guard.evaluate(returns=returns)
    # K=8 (8 个变体)
    matrix = _variant_matrix(n_variants=8, n_periods=120, alpha=0.001, seed=3)
    param_results = pd.DataFrame({"variant": list(matrix.columns), "sharpe": 1.0})
    rep_k8 = guard.evaluate(returns=returns, param_results=param_results)

    assert rep_k8.deflated_sharpe_pvalue >= rep_k1.deflated_sharpe_pvalue - 1e-9, \
        "试验次数越多, 通过多重检验校正的 p 值应越高 (更不显著)"


# ═══════════════════════════════════════════════════════════════
# 3. ChromaDB 文本查询走真实向量 (kline_text)
# ═══════════════════════════════════════════════════════════════

def test_kline_text_retrieval_real_vector():
    """文本查询应返回真实相似度排名 (doji 对'十字星'最高)"""
    from knowledge.manager import KnowledgeManager

    km = KnowledgeManager()
    if not km.chroma_available:
        pytest.skip("ChromaDB 不可用")
    results = km.search_similar_klines("十字星", top_k=3)
    assert results, "应返回真实检索结果"
    assert results[0]["pattern"] == "doji", "十字星查询应命中 doji"
    # 相似度应为真实余弦值 (v3.1-deerflow: hnsw:space=cosine), 而非硬编码 0.5 / 完美 1.0
    sims = {r["similarity"] for r in results}
    assert 0.0 < max(sims) < 1.0
