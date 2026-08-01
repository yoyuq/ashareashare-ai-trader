"""
FactorEvaluator — IC/IR 因子评估器 (v3.1)

提供因子评估的标准方法论:
  - Spearman Rank IC (横截面排名相关性)
  - Information Ratio (IR = IC_mean / IC_std)
  - IC 衰减分析 (多周期前瞻收益)
  - 分层回测 (Layer Test / Quintile Analysis)

用法:
    evaluator = FactorEvaluator()
    result = evaluator.evaluate(factor_values, forward_returns)
    print(f"IC={result.ic_mean:.4f}, IR={result.ir:.3f}, Rank_IC={result.rank_ic:.4f}")
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FactorEvaluation:
    """单个因子的评估结果"""
    name: str = ""
    category: str = "custom"

    # IC 指标
    ic_mean: float = 0.0          # 平均 IC (Spearman Rank)
    ic_std: float = 0.0           # IC 标准差
    ir: float = 0.0               # Information Ratio = ic_mean / ic_std
    ic_positive_ratio: float = 0.0  # IC > 0 的周期占比

    # Rank IC
    rank_ic: float = 0.0          # Spearman Rank IC
    rank_ic_std: float = 0.0

    # 衰减
    ic_decay: Dict[int, float] = field(default_factory=dict)  # horizon -> IC

    # 分层
    layer_returns: List[float] = field(default_factory=list)  # 各分组的平均收益
    top_bottom_spread: float = 0.0  # 多空组合收益差

    # 元信息
    periods: int = 0
    grade: str = "C"

    def _assign_grade(self):
        """Based on IR: A(>1.0) B(>0.5) C(>0.3) D(>0.1) F(<=0.1)"""
        if self.ir >= 1.0:
            self.grade = "A"
        elif self.ir >= 0.5:
            self.grade = "B"
        elif self.ir >= 0.3:
            self.grade = "C"
        elif self.ir >= 0.1:
            self.grade = "D"
        else:
            self.grade = "F"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "ic_mean": round(self.ic_mean, 4),
            "ic_std": round(self.ic_std, 4),
            "ir": round(self.ir, 3),
            "ic_positive_ratio": round(self.ic_positive_ratio, 3),
            "rank_ic": round(self.rank_ic, 4),
            "ic_decay": {str(k): round(v, 4) for k, v in self.ic_decay.items()},
            "layer_returns": [round(x, 4) for x in self.layer_returns],
            "top_bottom_spread": round(self.top_bottom_spread, 4),
            "periods": self.periods,
            "grade": self.grade,
        }


class FactorEvaluator:
    """因子评估器 — 基于 IC/IR 的标准量化评估"""

    def evaluate(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        name: str = "unknown",
        category: str = "custom",
    ) -> FactorEvaluation:
        """
        评估单个因子

        Args:
            factor_values: 因子值 (index = stock_id)
            forward_returns: 前瞻收益 (index = stock_id, 同一时间截面)
            name: 因子名称
            category: 因子类别

        Returns:
            FactorEvaluation with IC/IR/RankIC results
        """
        result = FactorEvaluation(name=name, category=category)

        # 数据对齐
        common_idx = factor_values.index.intersection(forward_returns.index)
        if len(common_idx) < 10:
            logger.debug(f"Factor '{name}': insufficient data ({len(common_idx)} rows)")
            return result

        fv = factor_values.loc[common_idx].astype(float)
        fr = forward_returns.loc[common_idx].astype(float)

        # 去除 NaN/Inf
        valid = fv.notna() & fr.notna() & np.isfinite(fv) & np.isfinite(fr)
        fv = fv[valid]
        fr = fr[valid]

        if len(fv) < 10:
            return result

        result.periods = len(fv)

        # Rank IC (Spearman)
        result.rank_ic = self._spearman_ic(fv, fr)
        result.ic_mean = result.rank_ic  # alias

        # IR approximation (single cross-section: use rank dispersion)
        result.ir = self._estimate_ir_single(fv, fr)

        # IC positive ratio approximation
        result.ic_positive_ratio = 1.0 if result.rank_ic > 0 else 0.0

        # Layer test
        layers = self._layer_test(fv, fr, n_groups=5)
        result.layer_returns = layers
        result.top_bottom_spread = layers[-1] - layers[0] if len(layers) >= 2 else 0.0

        result._assign_grade()
        return result

    def evaluate_time_series(
        self,
        factor_panel: pd.DataFrame,       # Date x Stock
        returns_panel: pd.DataFrame,      # Date x Stock
        name: str = "unknown",
        category: str = "custom",
    ) -> FactorEvaluation:
        """
        时间序列评估 — 每个截面计算 IC, 汇总统计

        Args:
            factor_panel: 因子面板 (行=日期, 列=股票)
            returns_panel: 前瞻收益面板
        """
        result = FactorEvaluation(name=name, category=category)
        ics = []

        common_dates = factor_panel.index.intersection(returns_panel.index)
        if len(common_dates) < 5:
            return result

        for dt in common_dates:
            fv = factor_panel.loc[dt]
            fr = returns_panel.loc[dt]

            valid = fv.notna() & fr.notna()
            fv_valid = fv[valid]
            fr_valid = fr[valid]

            if len(fv_valid) < 10:
                continue

            ic = self._spearman_ic(fv_valid, fr_valid)
            if not np.isnan(ic):
                ics.append(ic)

        if not ics:
            return result

        ics = np.array(ics)
        result.ic_mean = float(np.mean(ics))
        result.ic_std = float(np.std(ics, ddof=1))
        result.ir = result.ic_mean / result.ic_std if result.ic_std > 0 else 0.0
        result.ic_positive_ratio = float(np.mean(ics > 0))
        result.rank_ic = result.ic_mean
        result.periods = len(ics)

        result._assign_grade()
        return result

    def compute_decay(
        self,
        factor_values: pd.Series,
        forward_returns_multi: Dict[int, pd.Series],
    ) -> Dict[int, float]:
        """
        Compute IC decay across multiple forward horizons.

        Args:
            factor_values: Single-period factor values
            forward_returns_multi: {horizon_days: forward_returns_series}

        Returns:
            {horizon: IC_value}
        """
        decay = {}
        for horizon, fwd_ret in forward_returns_multi.items():
            ic = self._spearman_ic(factor_values, fwd_ret)
            decay[horizon] = ic if not np.isnan(ic) else 0.0
        return decay

    @staticmethod
    def _spearman_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
        """Compute Spearman Rank IC (cross-sectional)"""
        try:
            rank_fv = factor_values.rank(pct=False)
            rank_fr = forward_returns.rank(pct=False)
            ic = rank_fv.corr(rank_fr, method="spearman")
            return float(ic) if not np.isnan(ic) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _estimate_ir_single(factor_values: pd.Series, forward_returns: pd.Series) -> float:
        """
        Estimate IR from a single cross-section using rank dispersion.

        IR = IC_mean / IC_std. For a single cross-section, we approximate
        using the dispersion of top vs bottom ranks.
        """
        n = min(len(factor_values), len(forward_returns))
        if n < 10:
            return 0.0

        try:
            # Split into 5 groups by factor rank
            rank_fv = factor_values.rank()
            top = forward_returns[rank_fv >= rank_fv.quantile(0.8)]
            bottom = forward_returns[rank_fv <= rank_fv.quantile(0.2)]

            if len(top) < 2 or len(bottom) < 2:
                return 0.0

            spread = top.mean() - bottom.mean()
            spread_std = max(abs(spread), 1e-8)

            # Rough IR approximation
            rank_ic = FactorEvaluator._spearman_ic(factor_values, forward_returns)
            return abs(rank_ic) / max(0.05, abs(spread_std - abs(rank_ic)))

        except Exception:
            return 0.0

    @staticmethod
    def _layer_test(
        factor_values: pd.Series,
        forward_returns: pd.Series,
        n_groups: int = 5,
    ) -> List[float]:
        """
        Layer test: divide stocks into N groups by factor value,
        compute mean forward return for each group.

        Expects monotonically increasing returns from bottom to top group.
        """
        try:
            groups = pd.qcut(factor_values.rank(method="first"), n_groups, labels=False, duplicates="drop")
            layer_returns = forward_returns.groupby(groups).mean().sort_index().tolist()
            return layer_returns
        except Exception:
            return [0.0] * n_groups
