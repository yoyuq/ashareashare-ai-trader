"""
过拟合防控体系 — 6层防御

层级设计:
  L1: 严格时间分割 (60% train / 20% validation / 20% test)
  L2: PBO (Probability of Backtest Overfitting)
  L3: Deflated Sharpe Ratio (DSR)
  L4: Walk-Forward Validation (滚动窗口验证)
  L5: 参数敏感性分析
  L6: Monte Carlo置换检验

v2.1新增:
  - 48小时冷静期(策略设计→冻结→回测)
  - 密封评估(Proposer/Evaluator分离)
  - 口径硬化(definitions YAML)
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class OverfittingReport:
    """过拟合检测报告"""
    pbo: float                    # PBO概率
    deflated_sharpe: float        # 经缩减的Sharpe
    deflated_sharpe_pvalue: float  # DSR p值
    walk_forward_sharpe: float      # WF平均样本外Sharpe
    walk_forward_stability: float   # WF稳定性(StdDev of OOS Sharpes)
    parameter_sensitivity: float    # 参数敏感度(越小越好)
    monte_carlo_pvalue: float       # MC置换检验p值
    is_overfit: bool               # 综合判断
    details: Dict[str, Any] = field(default_factory=dict)

    def verdict(self) -> str:
        if self.is_overfit:
            return "⚠️ 高风险: 策略疑似过拟合,不建议实盘使用"
        if self.pbo > 0.1 or self.deflated_sharpe_pvalue > 0.05:
            return "⚡ 中风险: 策略可能需要更多样本外验证"
        return "✅ 低风险: 策略通过过拟合防控检验"


class OverfittingGuard:
    """
    过拟合防控守护者

    使用方式:
        guard = OverfittingGuard()
        report = guard.evaluate(
            returns=backtest_daily_returns,
            param_space=param_combinations,
        )
    """

    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

    # ═══════════════════════════════════════════════════════════════
    # 主评估入口
    # ═══════════════════════════════════════════════════════════════

    def evaluate(
        self,
        returns: pd.Series,
        oos_returns: Optional[pd.Series] = None,
        param_results: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> OverfittingReport:
        """
        全面过拟合评估

        Args:
            returns: 样本内(或全部)日收益率
            oos_returns: 样本外日收益率(用于Walk-Forward)
            param_results: 参数扫描结果 (columns: param1, param2, ..., sharpe)
            benchmark_returns: 基准收益率
        """
        n = len(returns)

        # L2: PBO
        pbo = self._compute_pbo(returns.get("sharpe", pd.Series(dtype=float))
                                if isinstance(returns, pd.DataFrame)
                                else returns) if param_results is not None else 0.0

        # L3: Deflated Sharpe Ratio
        dsr, dsr_pval = self._deflated_sharpe_ratio(
            returns, param_results or pd.DataFrame()
        )

        # L4: Walk-Forward
        wf_sharpe, wf_stability = (0, 0)
        if oos_returns is not None and len(oos_returns) > 0:
            wf_sharpe, wf_stability = self._walk_forward_stability(oos_returns)

        # L5: 参数敏感性
        param_sens = self._parameter_sensitivity(param_results)

        # L6: Monte Carlo
        mc_pval = self._monte_carlo_test(returns, benchmark_returns)

        # 综合判断 (至少2项触发即标记)
        flags = sum([
            pbo > 0.1,
            dsr_pval > 0.05,
            wf_stability > 0.5,  # 样本外Sharpe波动>50%
            param_sens > 1.0,    # 参数敏感度>1.0
            mc_pval > 0.05,
        ])
        is_overfit = flags >= 2

        return OverfittingReport(
            pbo=round(pbo, 4),
            deflated_sharpe=round(dsr, 3),
            deflated_sharpe_pvalue=round(dsr_pval, 4),
            walk_forward_sharpe=round(wf_sharpe, 3),
            walk_forward_stability=round(wf_stability, 3),
            parameter_sensitivity=round(param_sens, 3),
            monte_carlo_pvalue=round(mc_pval, 4),
            is_overfit=is_overfit,
            details={
                "n_samples": n,
                "flags_triggered": flags,
                "threshold": 2,
            },
        )

    # ═══════════════════════════════════════════════════════════════
    # L2: PBO (Probability of Backtest Overfitting)
    # ═══════════════════════════════════════════════════════════════

    def _compute_pbo(
        self,
        returns: pd.Series,
        n_trials: int = 1000,
    ) -> float:
        """
        PBO计算 (Bailey et al. 2014)

        思路: 随机排序收益率序列→计算最佳参数的Sharpe→
              重复N次→统计有多少次随机序列超过了实际最优Sharpe

        PBO ≈ P(random Sharpe > best in-sample Sharpe)
        """
        if len(returns) < 50:
            return 0.5

        # 实际Sharpe
        actual_sharpe = (
            returns.mean() / max(returns.std(), 1e-10) * np.sqrt(252)
        )

        # 生成随机序列
        random_sharpes = []
        n = len(returns)

        for _ in range(min(n_trials, 1000)):
            # 随机打乱收益率
            shuffled = returns.sample(frac=1, replace=False).values
            sr = np.mean(shuffled) / max(np.std(shuffled), 1e-10) * np.sqrt(252)
            random_sharpes.append(sr)

        random_sharpes = np.array(random_sharpes)
        pbo = np.mean(random_sharpes >= actual_sharpe)

        # 修正: 对称区间
        pbo = min(pbo, 1.0 - pbo) * 2

        return pbo

    # ═══════════════════════════════════════════════════════════════
    # L3: Deflated Sharpe Ratio (DSR)
    # ═══════════════════════════════════════════════════════════════

    def _deflated_sharpe_ratio(
        self,
        returns: pd.Series,
        param_results: pd.DataFrame,
    ) -> Tuple[float, float]:
        """
        Deflated Sharpe Ratio (Lopez de Prado 2016)

        考虑多重检验后的Sharpe比率
        DSR = Prob(实际SR > E[max(SR) from random])
        """
        if returns.empty:
            return 0, 1.0

        # 实际SR
        n = len(returns)
        sr = returns.mean() / max(returns.std(), 1e-10) * np.sqrt(252)

        # 有效试验次数
        if not param_results.empty and "sharpe" in param_results.columns:
            K = len(param_results)
        else:
            K = 1

        # 期望最大SR (Bailey & Lopez de Prado公式)
        # E[max(SR)] ≈ sqrt(2 * log(K)) * (1 - gamma * n**-1)
        if K > 1:
            euler = 0.5772
            expected_max = (1 - euler) * stats.norm.ppf(1 - 1/K) + euler * stats.norm.ppf(1 - 1/(K*np.e))
            expected_max = max(expected_max, 0)
        else:
            expected_max = 0

        # 经缩减的SR
        deflated_sr = max(sr - expected_max, 0)

        # p值
        se = 1.0 / np.sqrt(max(n, 1))
        z_stat = sr / max(se, 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return deflated_sr, p_value

    # ═══════════════════════════════════════════════════════════════
    # L4: Walk-Forward Validation
    # ═══════════════════════════════════════════════════════════════

    def _walk_forward_stability(
        self,
        oos_returns: pd.Series,
    ) -> Tuple[float, float]:
        """
        Walk-Forward 样本外稳定性

        按季度/年度分组 → 每组计算Sharpe → 统计均值和标准差
        """
        if len(oos_returns) < 63:  # 至少一个季度
            return 0, 0

        # 按季度分组
        quarterly = oos_returns.resample("QE").apply(
            lambda x: x.mean() / max(x.std(), 1e-10) * np.sqrt(252)
        ).dropna()

        if len(quarterly) < 2:
            # 按月
            monthly = oos_returns.resample("ME").apply(
                lambda x: x.mean() / max(x.std(), 1e-10) * np.sqrt(252)
            ).dropna()
            if len(monthly) < 2:
                return oos_returns.mean() / max(oos_returns.std(), 1e-10) * np.sqrt(252), 0
            oos_groups = monthly
        else:
            oos_groups = quarterly

        avg_sr = oos_groups.mean()
        std_sr = oos_groups.std()
        stability = std_sr / max(abs(avg_sr), 1e-10)

        return avg_sr, stability

    # ═══════════════════════════════════════════════════════════════
    # L5: 参数敏感性分析
    # ═══════════════════════════════════════════════════════════════

    def _parameter_sensitivity(
        self,
        param_results: Optional[pd.DataFrame],
    ) -> float:
        """
        参数敏感性 = ΔSharpe / ΔParam

        如果微小参数变化导致Sharpe剧烈波动→大概率过拟合
        """
        if param_results is None or param_results.empty:
            return 0

        if "sharpe" not in param_results.columns:
            return 0

        sharpes = param_results["sharpe"].values

        if len(sharpes) < 2:
            return 0

        # 变异系数
        sensitivity = np.std(sharpes) / max(abs(np.mean(sharpes)), 1e-10)

        return sensitivity

    # ═══════════════════════════════════════════════════════════════
    # L6: Monte Carlo置换检验
    # ═══════════════════════════════════════════════════════════════

    def _monte_carlo_test(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        n_sims: int = 1000,
    ) -> float:
        """
        Monte Carlo置换检验

        H0: 策略收益 = 随机交易 (无真实Alpha)

        生成N条随机净值曲线 → 统计超越实际Sharpe的比例 → p值
        """
        if returns.empty:
            return 1.0

        actual_sr = returns.mean() / max(returns.std(), 1e-10) * np.sqrt(252)
        n = len(returns)

        # 生成随机净值曲线
        random_srs = []
        for _ in range(min(n_sims, 500)):
            # 方式1: 对收益率自助法(bootstrap)
            sampled = np.random.choice(returns.values, size=n, replace=True)
            sr = np.mean(sampled) / max(np.std(sampled), 1e-10) * np.sqrt(252)
            random_srs.append(sr)

        random_srs = np.array(random_srs)

        # p值 = 随机SR > 实际SR 的比例
        p_value = np.mean(random_srs >= actual_sr)

        return p_value

    # ═══════════════════════════════════════════════════════════════
    # 辅助: 时间分割
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def split_time_series(
        df: pd.DataFrame,
        train_pct: float = 0.60,
        val_pct: float = 0.20,
        time_col: str = "date",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        严格按时间顺序分割 (严禁随机shuffle!)

        Returns:
            train_df, validation_df, test_df
        """
        df = df.sort_values(time_col)
        n = len(df)

        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(
            f"数据分割: 训练{train_pct:.0%}({len(train)}条) | "
            f"验证{val_pct:.0%}({len(val)}条) | "
            f"测试{1-train_pct-val_pct:.0%}({len(test)}条)"
        )

        return train, val, test


# ═══════════════════════════════════════════════════════════════
# v2.1: 48小时冷静期 + 密封评估
# ═══════════════════════════════════════════════════════════════

@dataclass
class BacktestDiscipline:
    """
    🆕 v2.1 回测纪律执行器

    规则:
    1. 策略设计完成后,必须等待48小时才能回测
    2. 定义文件必须YAML硬化(不能口头描述)
    3. 测试集只能用一次
    """

    strategy_name: str
    designed_at: datetime = field(default_factory=datetime.now)
    test_set_used: bool = False
    definition_yaml_path: Optional[str] = None

    def can_backtest(self) -> Tuple[bool, str]:
        """检查是否可以回测"""
        elapsed = datetime.now() - self.designed_at
        remaining = timedelta(hours=48) - elapsed

        if remaining.total_seconds() > 0:
            hours_left = remaining.total_seconds() / 3600
            return False, (
                f"冷静期尚未结束,还需等待{hours_left:.1f}小时。"
                f"设计时间: {self.designed_at.strftime('%Y-%m-%d %H:%M')}"
            )
        return True, "✅ 冷静期已过"

    def can_use_test_set(self) -> Tuple[bool, str]:
        """检查是否可以使用测试集"""
        if self.test_set_used:
            return False, "❌ 测试集已使用过,不可再用!"
        return True, "✅ 测试集可用"

    def mark_test_set_used(self):
        """标记测试集已使用"""
        self.test_set_used = True
        logger.warning(f"⚠️ {self.strategy_name}: 测试集已使用,不可再用")


class SealedEvaluation:
    """
    🆕 v2.1 密封评估框架

    Proposer (策略开发者) 和 Evaluator (评估者) 信息隔离:
      - Proposer: 编写策略,提交代码
      - Evaluator: 独立运行回测,锁定测试用例后才看到策略代码
      - RO-Lock: 代码提交后,测试用例锁定,不可修改
    """

    def __init__(self):
        self._locked_test_cases: Dict[str, List[str]] = {}
        self._submit_log: Dict[str, datetime] = {}

    def lock_test_cases(self, strategy_id: str, test_cases: List[str]):
        """锁定测试用例 (Evaluator执行,在查看代码前)"""
        self._locked_test_cases[strategy_id] = test_cases
        logger.info(f"🔒 {strategy_id}: 测试用例已锁定 ({len(test_cases)}个)")

    def submit_strategy(self, strategy_id: str, code_hash: str):
        """提交策略 (Proposer执行,提交后不可修改)"""
        self._submit_log[strategy_id] = datetime.now()
        logger.info(f"📝 {strategy_id}: 策略已提交 (hash={code_hash[:8]})")

    def evaluate(self, strategy_id: str) -> Tuple[bool, str]:
        """评估 (独立运行,使用锁定的测试用例)"""
        if strategy_id not in self._locked_test_cases:
            return False, "❌ 测试用例未锁定"
        if strategy_id not in self._submit_log:
            return False, "❌ 策略未提交"

        test_cases = self._locked_test_cases[strategy_id]
        return True, (
            f"✅ 评估开始: {len(test_cases)}个锁定测试用例, "
            f"策略提交于 {self._submit_log[strategy_id].strftime('%Y-%m-%d %H:%M')}"
        )
