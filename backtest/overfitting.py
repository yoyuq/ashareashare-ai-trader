"""
过拟合防控体系 — 6层防御

层级设计:
  L1: 严格时间分割 (60% train / 20% validation / 20% test)
  L2: PBO (Probability of Backtest Overfitting, CSCV法 — 需多变体绩效矩阵)
  L3: Deflated Sharpe Ratio (DSR, 多重检验校正 + Lo 2002 标准误)
  L4: Walk-Forward Validation (滚动窗口验证)
  L5: 参数敏感性分析
  L6: Monte Carlo 随机化检验 (符号翻转零假设)

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
        variant_returns: Optional[pd.DataFrame] = None,
    ) -> OverfittingReport:
        """
        全面过拟合评估

        Args:
            returns: 样本内(或全部)日收益率
            oos_returns: 样本外日收益率(用于Walk-Forward)
            param_results: 参数扫描结果 (columns: param1, param2, ..., sharpe)
            benchmark_returns: 基准收益率
            variant_returns: 多策略变体逐期收益矩阵 (index=时间顺序, columns=各参数/策略变体),
                用于真正的 CSCV-PBO。不提供时 PBO 为 NaN (单一序列不存在"多重选择"问题, 不参与判定)。
        """
        n = len(returns)

        # L2: PBO — 真正的 CSCV 需要多变体绩效矩阵; 单一收益序列无法计算 (返回 NaN)
        pbo = self._compute_pbo(variant_returns) if variant_returns is not None else float("nan")

        # L3: Deflated Sharpe Ratio
        # v3.0: 修复 param_results 为 DataFrame 时 `or` 触发歧义真值 ValueError
        dsr, dsr_pval = self._deflated_sharpe_ratio(
            returns, param_results if param_results is not None else pd.DataFrame()
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
                "pbo_defined": variant_returns is not None,  # False: 未提供多变体矩阵, PBO=NaN 不参与判定
            },
        )

    # ═══════════════════════════════════════════════════════════════
    # L2: PBO (Probability of Backtest Overfitting)
    # ═══════════════════════════════════════════════════════════════

    def _compute_pbo(
        self,
        variant_returns: Optional[pd.DataFrame],
        max_partitions: int = 100,
    ) -> float:
        """
        PBO计算 — 真正的 CSCV 法 (Bailey, Borwein, López de Prado, Zhu 2014)

        PBO = P(样本内最优变体 在样本外排名低于中位数), 通过组合对称交叉验证估计:
          1. 将时间轴切成 S 个等长块 (S 为偶数)
          2. 对每个组合 C(S, S/2): 选中的 S/2 块作样本内(IS), 其余作样本外(OOS)
          3. 找出 IS 最优变体 n*, 统计其 OOS 绩效是否低于 OOS 中位数
          4. PBO = 低于中位数的划分数 / 有效划分数

        注意: "打乱收益序列再比Sharpe"不是PBO (打乱不改变均值/标准差, 原实现恒返回0)。
        PBO 度量的是"从多个候选中挑选"带来的选择过拟合, 因此必须有多变体矩阵。

        Args:
            variant_returns: 多变体逐期收益矩阵 (index=时间顺序, columns=变体)

        Returns:
            PBO ∈ [0,1]; 变体或样本不足时返回 NaN (未定义)
        """
        if variant_returns is None:
            return float("nan")

        T, N = variant_returns.shape
        if N < 2 or T < 16:
            return float("nan")

        # 1. 切成 S 块 (每块≥8期, S 为偶数, 至少2块)
        S = min(10, (T // 8) * 2)
        S = max(S, 2)
        block_size = T // S
        if block_size < 4:
            return float("nan")
        blocks = [variant_returns.iloc[i * block_size:(i + 1) * block_size] for i in range(S)]

        # 2. 枚举 C(S, S/2) 划分 (过多时固定种子采样, 保证可复现)
        from itertools import combinations
        half = S // 2
        combs = list(combinations(range(S), half))
        if len(combs) > max_partitions:
            rng = np.random.default_rng(42)
            pick = rng.choice(len(combs), max_partitions, replace=False)
            combs = [combs[int(i)] for i in pick]

        def _sharpe_by_variant(df: pd.DataFrame) -> pd.Series:
            sd = df.std()
            out = df.mean() / sd.replace(0, np.nan)
            return out.fillna(-np.inf)  # 排名用, 无需年化 (单调变换不影响名次)

        # 3. 逐划分: IS最优变体在OOS是否低于中位数
        below_median, valid = 0, 0
        for is_idx in combs:
            is_set = set(is_idx)
            oos_idx = [j for j in range(S) if j not in is_set]
            is_perf = _sharpe_by_variant(pd.concat([blocks[j] for j in is_idx]))
            oos_perf = _sharpe_by_variant(pd.concat([blocks[j] for j in oos_idx]))
            if not np.isfinite(is_perf).any() or not np.isfinite(oos_perf).any():
                continue
            n_star = is_perf.idxmax()
            if oos_perf[n_star] < oos_perf.median():
                below_median += 1
            valid += 1

        if valid == 0:
            return float("nan")
        return below_median / valid

    # ═══════════════════════════════════════════════════════════════
    # L3: Deflated Sharpe Ratio (DSR)
    # ═══════════════════════════════════════════════════════════════

    def _deflated_sharpe_ratio(
        self,
        returns: pd.Series,
        param_results: pd.DataFrame,
    ) -> Tuple[float, float]:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado 2014)

        DSR 检验 "观测SR 是否在扣除多重检验基准 SR₀=E[max SR|K次试验] 后仍显著":
            DSR = Φ( (SR̂ - SR₀) / √Var[SR̂] )
        其中 Var[SR̂] = (1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²) / (T-1)  (Lo 2002, 含偏度γ₃/峰度γ₄修正)

        修复: 旧实现返回的 p 值基于"未缩减的原始SR", 多重检验校正从未进入 p 值。

        Returns:
            (deflated_sharpe_年化, p_value) — p 小表示通过多重检验校正
        """
        if returns.empty:
            return 0, 1.0

        n = len(returns)
        # 每期 SR (不年化 — DSR 公式要求与 T=期数 同单位)
        sr = returns.mean() / max(returns.std(), 1e-10)
        skew = float(returns.skew()) if n > 2 else 0.0
        kurt_excess = float(returns.kurt()) if n > 3 else 0.0  # pandas 返回超额峰度 (γ₄-3)

        # 有效试验次数 K (参数扫描的组数)
        if not param_results.empty and "sharpe" in param_results.columns:
            K = max(len(param_results), 1)
        else:
            K = 1

        # 多重检验基准 SR₀ = E[max SR] (K 个独立试验, Bailey & López de Prado)
        if K > 1:
            euler = 0.5772156649
            sr0 = (1 - euler) * stats.norm.ppf(1 - 1.0 / K) \
                + euler * stats.norm.ppf(1 - 1.0 / (K * np.e))
            sr0 = max(sr0, 0.0)
        else:
            sr0 = 0.0

        # SR 估计量方差 (Lo 2002; (γ₄-1)/4 = (超额峰度+2)/4)
        var_sr = (1.0 - skew * sr + (kurt_excess + 2) / 4.0 * sr ** 2) / max(n - 1, 1)
        se = float(np.sqrt(max(var_sr, 1e-12)))

        dsr_stat = (sr - sr0) / se
        p_value = float(1.0 - stats.norm.cdf(dsr_stat))  # 单侧: SR 显著高于多重检验基准

        deflated_sr_annualized = (sr - sr0) * np.sqrt(252)
        return deflated_sr_annualized, p_value

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
        Monte Carlo 随机化检验

        H0: 策略无方向选择能力 (收益与信号无关)。
        零分布经"随机符号翻转"构造: r* = s⊙r, s∈{±1} iid ——
        保留收益的幅度结构 (波动率), 同时令期望收益为0。

        修复: 旧实现对收益本身做 bootstrap 重采样, SR 天然围绕样本SR波动 →
        p 值恒≈0.5, 任何策略 (含纯噪声) 都不报警。符号翻转零分布以0为中心,
        有效策略 p→0, 噪声策略 p≈0.5 → 触发过拟合标记。

        Returns:
            p值 = P(SR* ≥ SR_actual | H0)
        """
        if returns.empty:
            return 1.0

        r = returns.to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 10:
            return 1.0

        actual_sr = np.mean(r) / max(np.std(r, ddof=1), 1e-10) * np.sqrt(252)
        rng = np.random.default_rng(1234)  # 固定种子, 结果可复现
        sims = min(n_sims, 1000)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(sims, len(r)))
        sim_returns = r[None, :] * signs
        sim_std = np.maximum(sim_returns.std(axis=1, ddof=1), 1e-10)
        sim_srs = sim_returns.mean(axis=1) / sim_std * np.sqrt(252)

        p_value = float(np.mean(sim_srs >= actual_sr))
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
