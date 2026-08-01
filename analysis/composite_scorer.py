"""
Composite Scorer — 复合惩罚函数评分系统 (v3.0-competition, 2026 Best Practice)

基于2026年量化研究最佳实践:
  "Use a composite objective function including penalties for overfitting drift,
   effective parameter count, transaction-cost sensitivity, search-intensity,
   and fragility to regime changes."

替代单一指标(如夏普比率)的策略评估,使用复合惩罚函数:
  1. 基础收益 (Base Return): 年化收益率
  2. 风险惩罚 (Risk Penalty): 波动率 + 最大回撤
  3. 过拟合惩罚 (Overfitting Penalty): PBO + DSR + 参数数量
  4. 成本惩罚 (Cost Penalty): 交易成本 + 滑点 + 冲击成本
  5. 脆弱性惩罚 (Fragility Penalty): 体制变化敏感度

输出: 复合评分 (0-100, 越高越好)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CompositeScore:
    """复合评分结果"""
    total_score: float = 0.0             # 综合评分 0-100
    base_return: float = 0.0             # 基础收益分
    risk_penalty: float = 0.0            # 风险惩罚
    overfitting_penalty: float = 0.0     # 过拟合惩罚
    cost_penalty: float = 0.0            # 成本惩罚
    fragility_penalty: float = 0.0       # 脆弱性惩罚

    # 详细分解
    details: Dict[str, Any] = field(default_factory=dict)
    grade: str = "C"                     # A/B/C/D/F

    def to_dict(self) -> dict:
        return {
            "total_score": round(self.total_score, 1),
            "grade": self.grade,
            "breakdown": {
                "base_return": round(self.base_return, 1),
                "risk_penalty": round(self.risk_penalty, 1),
                "overfitting_penalty": round(self.overfitting_penalty, 1),
                "cost_penalty": round(self.cost_penalty, 1),
                "fragility_penalty": round(self.fragility_penalty, 1),
            },
            "details": self.details,
        }

    def _assign_grade(self):
        """根据总分分配等级"""
        if self.total_score >= 85:
            self.grade = "A"
        elif self.total_score >= 70:
            self.grade = "B"
        elif self.total_score >= 55:
            self.grade = "C"
        elif self.total_score >= 40:
            self.grade = "D"
        else:
            self.grade = "F"


class CompositeScorer:
    """
    复合惩罚函数评分器

    用法:
        scorer = CompositeScorer()
        score = scorer.evaluate(
            backtest_metrics={...},
            strategy_config={...},
            regime_history={...},
        )
        print(f"复合评分: {score.total_score:.1f}/100 ({score.grade}级)")
    """

    # 权重配置
    WEIGHTS = {
        "base_return": 1.0,       # 正贡献
        "risk": -0.5,             # 惩罚
        "overfitting": -0.4,      # 惩罚
        "cost": -0.3,             # 惩罚
        "fragility": -0.3,        # 惩罚
    }

    def evaluate(
        self,
        backtest_metrics: Dict[str, Any],
        strategy_config: Optional[Dict[str, Any]] = None,
        regime_history: Optional[Dict[str, Any]] = None,
    ) -> CompositeScore:
        """
        计算复合评分

        Args:
            backtest_metrics: 回测绩效指标
            strategy_config: 策略配置
            regime_history: 市场体制历史

        Returns:
            CompositeScore
        """
        score = CompositeScore()

        # 1. 基础收益 (0-40分)
        score.base_return = self._score_base_return(backtest_metrics)
        score.details["base_return"] = (
            f"年化收益={backtest_metrics.get('annual_return_pct', 0):.1f}%, "
            f"夏普={backtest_metrics.get('sharpe_ratio', 0):.2f}"
        )

        # 2. 风险惩罚 (0-30分扣减)
        score.risk_penalty = self._score_risk_penalty(backtest_metrics)
        score.details["risk_penalty"] = (
            f"最大回撤={backtest_metrics.get('max_drawdown_pct', 0):.1f}%, "
            f"波动率={backtest_metrics.get('volatility', 0):.1f}%"
        )

        # 3. 过拟合惩罚 (0-20分扣减)
        score.overfitting_penalty = self._score_overfitting_penalty(
            backtest_metrics, strategy_config or {}
        )
        pbo = backtest_metrics.get("PBO", backtest_metrics.get("pbo"))
        dsr = backtest_metrics.get("DSR", backtest_metrics.get("deflated_sharpe"))
        score.details["overfitting_penalty"] = (
            f"PBO={pbo or 'N/A'}, DSR={dsr or 'N/A'}, "
            f"参数数量={self._count_params(strategy_config or {})}"
        )

        # 4. 成本惩罚 (0-15分扣减)
        score.cost_penalty = self._score_cost_penalty(
            backtest_metrics, strategy_config or {}
        )
        score.details["cost_penalty"] = (
            f"交易成本={strategy_config.get('transaction_cost_pct', 0.0031) if strategy_config else 0.0031:.2%}, "
            f"换手率={backtest_metrics.get('turnover', 0):.1f}x"
        )

        # 5. 脆弱性惩罚 (0-15分扣减)
        score.fragility_penalty = self._score_fragility_penalty(
            backtest_metrics, regime_history or {}
        )
        score.details["fragility_penalty"] = (
            f"体制覆盖率={regime_history.get('regimes_covered', 0) if regime_history else 0}/6"
        )

        # 计算总分
        score.total_score = max(0.0, min(100.0,
            score.base_return * self.WEIGHTS["base_return"] +
            score.risk_penalty * self.WEIGHTS["risk"] +
            score.overfitting_penalty * self.WEIGHTS["overfitting"] +
            score.cost_penalty * self.WEIGHTS["cost"] +
            score.fragility_penalty * self.WEIGHTS["fragility"]
        ))

        score._assign_grade()
        return score

    # ═══════════════════════════════════════════════════════════════
    # 子评分方法
    # ═══════════════════════════════════════════════════════════════

    def _score_base_return(self, m: dict) -> float:
        """基础收益评分 (0-40)"""
        score = 20.0  # 中性起点
        ann_ret = m.get("annual_return_pct", m.get("total_return_pct", 0))
        sharpe = m.get("sharpe_ratio", 0)

        # 收益贡献
        if ann_ret > 20: score += 15.0
        elif ann_ret > 10: score += 10.0
        elif ann_ret > 0: score += 5.0
        elif ann_ret < -20: score -= 10.0

        # 风险调整
        if sharpe > 2.0: score += 5.0
        elif sharpe > 1.0: score += 2.0
        elif sharpe < 0: score -= 5.0

        return max(5.0, min(40.0, score))

    def _score_risk_penalty(self, m: dict) -> float:
        """风险惩罚 (越小越好,返回扣分值 0-30)"""
        penalty = 5.0  # 基础风险

        max_dd = abs(m.get("max_drawdown_pct", m.get("max_drawdown", 0)))
        if max_dd > 30: penalty += 15.0
        elif max_dd > 20: penalty += 10.0
        elif max_dd > 10: penalty += 5.0

        vol = m.get("volatility", m.get("annual_volatility", 20))
        if vol > 40: penalty += 10.0
        elif vol > 25: penalty += 5.0

        return min(30.0, penalty)

    def _score_overfitting_penalty(self, m: dict, config: dict) -> float:
        """过拟合惩罚 (0-20)"""
        penalty = 2.0

        # PBO (Probability of Backtest Overfitting)
        pbo = m.get("PBO", m.get("pbo", m.get("overfitting_checks", {}).get("PBO")))
        if pbo is not None:
            if pbo > 0.7: penalty += 10.0
            elif pbo > 0.5: penalty += 6.0
            elif pbo > 0.3: penalty += 3.0

        # DSR (Deflated Sharpe Ratio)
        dsr = m.get("DSR", m.get("deflated_sharpe", m.get("overfitting_checks", {}).get("DSR")))
        if dsr is not None:
            if dsr < 0.3: penalty += 8.0
            elif dsr < 0.7: penalty += 4.0

        # 参数数量惩罚 (log penalty on search intensity)
        param_count = self._count_params(config)
        if param_count > 20: penalty += 5.0
        elif param_count > 10: penalty += 2.0

        return min(20.0, penalty)

    def _score_cost_penalty(self, m: dict, config: dict) -> float:
        """成本惩罚 (0-15)"""
        penalty = 3.0

        cost_pct = config.get("transaction_cost_pct", 0.0031)
        if cost_pct < 0.001: penalty += 8.0  # 严重低估
        elif cost_pct < 0.002: penalty += 4.0  # 轻度低估

        # 换手率惩罚
        turnover = m.get("turnover", m.get("annual_turnover", 5))
        if turnover > 20: penalty += 4.0
        elif turnover > 10: penalty += 2.0

        return min(15.0, penalty)

    def _score_fragility_penalty(self, m: dict, regime_history: dict) -> float:
        """脆弱性惩罚 (0-15)"""
        penalty = 3.0

        # 体制覆盖率
        regimes_covered = regime_history.get("regimes_covered", 0)
        if regimes_covered < 3: penalty += 8.0
        elif regimes_covered < 5: penalty += 4.0

        # 压力测试
        if not m.get("stress_tested"):
            penalty += 4.0

        return min(15.0, penalty)

    def _count_params(self, config: dict) -> int:
        """递归统计参数数量"""
        if not config:
            return 0
        count = 0
        for k, v in config.items():
            if k in ("name", "description", "category", "version", "note", "id"):
                continue
            if isinstance(v, dict):
                count += self._count_params(v)
            elif isinstance(v, (int, float)):
                count += 1
            elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                count += len(v)
        return count


# ── 便捷函数 ──

def composite_score(backtest_metrics: dict, strategy_config: dict = None,
                    regime_history: dict = None) -> CompositeScore:
    """快速计算复合评分"""
    scorer = CompositeScorer()
    return scorer.evaluate(backtest_metrics, strategy_config, regime_history)
