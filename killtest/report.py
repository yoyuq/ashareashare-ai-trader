"""
kill-test 对比报告 — 三路信号质量 + 统计显著性

指标 (方向化收益 fwd_return, BUY=做多, SELL=做空, 正=方向正确):
  - n / mean_return / win_rate / expected_value / profit_factor
  - vs 基线 (随机臂 / 全市场均值): 均值差 + Welch t 统计量 + 自助法置信区间
"""

import math
from typing import Dict, List, Optional

import numpy as np

from killtest.tracker import Outcome


def summarize(outcomes: List[Outcome]) -> Dict:
    """单臂汇总指标"""
    if not outcomes:
        return {"n": 0, "mean_return": 0.0, "win_rate": 0.0, "expected_value": 0.0,
                "profit_factor": 0.0, "median": 0.0}
    rets = np.array([o.fwd_return for o in outcomes], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    if not np.isfinite(pf):
        pf = 99.0 if len(wins) else 0.0
    return {
        "n": len(outcomes),
        "mean_return": round(float(rets.mean()), 4),
        "median": round(float(np.median(rets)), 4),
        "win_rate": round(float((rets > 0).mean()), 4),
        "expected_value": round(float(rets.mean()), 4),
        "profit_factor": round(float(pf), 3),
    }


def welch_t(a: List[float], b: List[float]) -> Dict:
    """Welch t 检验 (两独立样本, 方差不相等假设), 返回 t 值/自由度/p 值"""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"t": 0.0, "p": 1.0, "df": 0.0}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return {"t": 0.0, "p": 1.0, "df": 0.0}
    t = (a.mean() - b.mean()) / se
    # Welch-Satterthwaite 自由度
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    )
    p = _t_tail(abs(t), df)
    return {"t": round(float(t), 3), "p": round(float(p), 4), "df": round(float(df), 1)}


def _t_tail(t: float, df: float) -> float:
    """t 分布双侧尾概率 (无 scipy, 用近似正态 + 修正)"""
    # Student t CDF 近似 (Abramowitz-Stegun 7.1.26 误差函数)
    if df > 200:
        return 2 * (1 - _normal_cdf(t))
    return 2 * (1 - _t_cdf_approx(t, df))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_cdf_approx(t: float, df: float) -> float:
    """t CDF 近似: 大样本接近正态; 小样本用 beta 展开近似"""
    x = df / (df + t * t)
    # 用不完全 beta 函数近似的简化版: 对大 df 直接用正态
    if df >= 30:
        return _normal_cdf(t)
    # 小样本近似 (精度一般, kill-test 只用方向性判断)
    return _normal_cdf(t * math.sqrt((df - 1.5) / df))


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 42) -> Dict:
    """自助法 95% 置信区间"""
    arr = np.array(values, dtype=float)
    if len(arr) < 3:
        return {"lo": float(arr.mean()) if len(arr) else 0.0,
                "hi": float(arr.mean()) if len(arr) else 0.0,
                "n_boot": 0}
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"lo": round(float(lo), 4), "hi": round(float(hi), 4), "n_boot": n_boot}


def compare(arm_outcomes: List[Outcome], baseline_outcomes: List[Outcome]) -> Dict:
    """臂 vs 基线的对比: 均值差 + t 检验 + 自助 CI"""
    a = [o.fwd_return for o in arm_outcomes]
    b = [o.fwd_return for o in baseline_outcomes]
    return {
        "mean_diff": round((np.mean(a) - np.mean(b)) if a and b else 0.0, 4),
        "welch": welch_t(a, b),
        "ci": bootstrap_ci(a),
        "baseline_n": len(b),
    }


def render(arms: Dict[str, List[Outcome]], baseline: Dict, title: str = "kill-test") -> str:
    """渲染 markdown 对比报告"""
    lines = [f"# {title}\n"]
    lines.append("| 臂 | n | 平均收益% | 胜率 | 期望值% | 盈亏比 | vs随机均值差% | t | p |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")

    arm_names = list(arms.keys())
    random_outcomes = arms.get("random", [])
    random_means = [o.fwd_return for o in random_outcomes]
    baseline_means = [o.fwd_return for o in baseline.get("outcomes", [])]

    for name, outcomes in arms.items():
        s = summarize(outcomes)
        # 对比基准: 优先随机臂, 其次全市场基线
        base = random_means if (random_means and name != "random") else baseline_means
        cmp = compare(outcomes, random_outcomes) if (name != "random" and random_outcomes) else (
            compare(outcomes, baseline.get("outcomes", [])))
        t = cmp["welch"]["t"]
        p = cmp["welch"]["p"]
        sig = "✅" if p < 0.05 else ("🔸" if p < 0.1 else "—")
        lines.append(
            f"| {name} | {s['n']} | {s['mean_return']*100:.2f}% | {s['win_rate']*100:.1f}% "
            f"| {s['expected_value']*100:.2f}% | {s['profit_factor']:.2f} | "
            f"{cmp['mean_diff']*100:+.2f}% | {t:.2f} | {p:.3f} {sig} |"
        )

    lines.append(f"\n> 基线: 全市场 {len(baseline_means)} 条前向收益, 均值 "
                 f"{np.mean(baseline_means)*100:.2f}% (n={len(baseline_means)})"
                 if baseline_means else "\n> 基线: 无")
    lines.append("\n> 口径: 方向化前向收益 (BUY=做多, SELL=做空, 正=方向正确)。"
                 "p<0.05 表示显著优于随机/基线。")
    return "\n".join(lines)
