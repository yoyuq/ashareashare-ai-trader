"""
Alpha/beta 因子归因层 (阶段1 #106)

把策略收益分解为「多少是 beta (市场敞口)、多少是 alpha (选股/择时技能)」。
项目反复把 beta 当 alpha 发现 (冷落 beta 跑赢上证 = 小盘 tilt 的 beta 不是 alpha),
因为没有自动归因这一步。本模块补上它, 让每个 A/B 都能自动报 beta/alpha。

方法 (单因子市场模型, Jensen 分解, 全程零模拟 — 只用真实日收益回归):
    r_p,t = α + β·r_b,t + ε_t          (OLS 日收益回归)
      β   = Cov(r_p, r_b) / Var(r_b)   市场 beta
      α   = mean(r_p) − β·mean(r_b)    日度 alpha (截距)

总收益分解 (同一窗口, 简单收益率口径的恒等式):
    strategy_return ≈ beta_contribution + alpha_contribution
      beta_contribution  = β × benchmark_return   (赚市场 beta 的钱)
      alpha_contribution = strategy_return − beta_contribution  (超额技能)
      excess_return      = strategy_return − benchmark_return    (「跑赢」量)

诚实判读:
    alpha_contribution 接近 0 → 跑赢几乎全是 beta (小盘/高波动 tilt), 不是选股 alpha。
    alpha_contribution 显著为正 → 有真实 alpha (在扣掉 beta 之后仍跑赢)。

用法:
    from analysis.attribution import attribute_equity_curves
    attr = attribute_equity_curves(strategy_equity, benchmark_equity)
    print(attr["beta"], attr["alpha_contribution_pct"])
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# 年化交易日数 (A 股一年约 244 交易日; 用 252 与国际口径一致, 相对比较不受影响)
PERIODS_PER_YEAR = 252
_MIN_POINTS = 3  # 少于 3 个对齐日收益不做回归 (拟合无意义)


def alpha_beta_attribution(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """
    对日收益序列做 alpha/beta 分解。

    Args:
        strategy_returns: 策略日收益 (简单收益率, e.g. equity.pct_change())。索引为日期。
        benchmark_returns: 基准日收益 (同上)。索引为日期。
        rf_annual: 年化无风险利率 (默认 0; 相对分解中 rf 几乎抵消, 保持 0 更诚实)。
        periods_per_year: 年化周期数 (默认 252)。

    Returns:
        dict: beta / alpha_annualized_pct / beta_contribution_pct /
              alpha_contribution_pct / excess_return_pct / r_squared /
              tracking_error_annualized_pct / information_ratio / n_aligned。
        数据不足时返回全 NaN (n_aligned=0), 不抛异常 (软增强, 不阻塞 A/B)。
    """
    out: Dict[str, float] = {
        "beta": float("nan"),
        "alpha_annualized_pct": float("nan"),
        "strategy_return_pct": float("nan"),
        "benchmark_return_pct": float("nan"),
        "excess_return_pct": float("nan"),
        "beta_contribution_pct": float("nan"),
        "alpha_contribution_pct": float("nan"),
        "r_squared": float("nan"),
        "tracking_error_annualized_pct": float("nan"),
        "information_ratio": float("nan"),
        "n_aligned": 0,
    }

    rp = strategy_returns.dropna()
    rb = benchmark_returns.dropna()
    if rp.empty or rb.empty:
        return out

    # 按日期内连接对齐 (只保留两序列都有的交易日)
    df = pd.concat([rp.rename("p"), rb.rename("b")], axis=1, join="inner").dropna()
    n = len(df)
    if n < _MIN_POINTS:
        return out

    p = df["p"].astype(float).to_numpy()
    b = df["b"].astype(float).to_numpy()

    # OLS: p = α + β·b (闭式解, 无需 statsmodels)
    var_b = float(np.var(b, ddof=1))
    if var_b <= 0 or not np.isfinite(var_b):
        return out
    beta = float(np.cov(p, b, ddof=1)[0, 1] / var_b)
    alpha_daily = float(np.mean(p) - beta * np.mean(b))

    # 年化 Jensen's alpha (几何口径, 更贴合复利语义)
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    alpha_net_daily = alpha_daily - rf_daily
    alpha_annualized = (1.0 + alpha_net_daily) ** periods_per_year - 1.0

    # 拟合优度 R² = 1 − SS_res / SS_tot
    resid = p - (alpha_daily + beta * b)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((p - np.mean(p)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 总收益分解 (用原序列的完整收益, 对齐后仍用对齐序列的复利口径)
    def _total(series: np.ndarray) -> float:
        # 简单收益的复利累计: prod(1+r) − 1
        return float(np.prod(1.0 + series) - 1.0)

    total_p = _total(p)
    total_b = _total(b)
    beta_contribution = beta * total_b
    alpha_contribution = total_p - beta_contribution
    excess = total_p - total_b

    # 主动收益 (r_p − r_b) 的跟踪误差与信息比率 (年化)
    active = p - b
    te_daily = float(np.std(active, ddof=1))
    te_annualized = te_daily * np.sqrt(periods_per_year)
    ir = (float(np.mean(active)) / te_daily) * np.sqrt(periods_per_year) if te_daily > 0 else float("nan")

    out.update({
        "beta": round(beta, 4),
        "alpha_annualized_pct": round(alpha_annualized * 100.0, 3),
        "strategy_return_pct": round(total_p * 100.0, 3),
        "benchmark_return_pct": round(total_b * 100.0, 3),
        "excess_return_pct": round(excess * 100.0, 3),
        "beta_contribution_pct": round(beta_contribution * 100.0, 3),
        "alpha_contribution_pct": round(alpha_contribution * 100.0, 3),
        "r_squared": round(r_squared, 4),
        "tracking_error_annualized_pct": round(te_annualized * 100.0, 3),
        "information_ratio": round(ir, 4) if np.isfinite(ir) else float("nan"),
        "n_aligned": n,
    })
    return out


def attribute_equity_curves(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """
    从净值曲线 (价格/权益序列) 计算 alpha/beta 分解。

    Args:
        strategy_equity: 策略净值曲线 (索引为日期, 值>=0)。
        benchmark_equity: 基准净值曲线 (同上)。

    Returns:
        同 alpha_beta_attribution。内部先算日收益再分解。
    """
    rp = strategy_equity.pct_change()
    rb = benchmark_equity.pct_change()
    return alpha_beta_attribution(rp, rb, rf_annual=rf_annual, periods_per_year=periods_per_year)


def two_factor_attribute_equity_curves(
    strategy_equity: pd.Series,
    market_equity: pd.Series,
    equalweight_equity: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """
    从三条净值曲线算双因子 (市场 + 小盘/等权) 分解。strategy/market/equalweight 均先算日收益。

    用于回答「冷落/低换手 tilt 跑赢上证 = 小盘 beta 还是选股 alpha」: 传入冷落组合净值、
    上证指数净值、以及它所属 universe 的等权净值; 若 alpha_contribution 接近 0 而 size_contribution
    显著为正, 则「跑赢」几乎全是小盘 beta, 不是选股技能。
    """
    rp = strategy_equity.pct_change()
    rm = market_equity.pct_change()
    re_ = equalweight_equity.pct_change()
    return two_factor_attribution(rp, rm, re_, periods_per_year=periods_per_year)


def two_factor_attribution(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    equalweight_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """
    双因子 (市场 + 小盘/等权) 归因: r_p = α + β_mkt·r_mkt + β_smb·(r_ew − r_mkt) + ε。

    关键用途: 上证 (r_mkt) 是大盘市值加权, 小盘/冷落溢价不会体现在 β_mkt 里, 会漏进单因子
    模型的「alpha」残差 → 项目反复把「小盘 beta」当「选股 alpha」。本函数引入 SMB
    (等权 − 市值加权) 因子, 把小盘溢价单独拆出来, 剩下的 α 才是「扣掉市场+小盘 beta 后」的
    真实选股 alpha。

    Args:
        strategy_returns: 策略日收益。
        market_returns: 市值加权市场日收益 (上证 sh.000001)。
        equalweight_returns: 等权 universe 日收益 (小盘 tilt 的「自然 beta」)。
        periods_per_year: 年化周期数。

    Returns:
        dict: beta_mkt / beta_smb / alpha_annualized_pct / market_contribution_pct /
              size_contribution_pct / alpha_contribution_pct / r_squared / n_aligned。
    """
    out: Dict[str, float] = {
        "beta_mkt": float("nan"),
        "beta_smb": float("nan"),
        "alpha_annualized_pct": float("nan"),
        "strategy_return_pct": float("nan"),
        "market_return_pct": float("nan"),
        "equalweight_return_pct": float("nan"),
        "smb_return_pct": float("nan"),
        "market_contribution_pct": float("nan"),
        "size_contribution_pct": float("nan"),
        "alpha_contribution_pct": float("nan"),
        "r_squared": float("nan"),
        "n_aligned": 0,
    }

    rp = strategy_returns.dropna()
    rm = market_returns.dropna()
    re_ = equalweight_returns.dropna()
    if rp.empty or rm.empty or re_.empty:
        return out

    df = pd.concat([rp.rename("p"), rm.rename("mkt"), re_.rename("ew")],
                   axis=1, join="inner").dropna()
    n = len(df)
    if n < _MIN_POINTS:
        return out

    p = df["p"].astype(float).to_numpy()
    mkt = df["mkt"].astype(float).to_numpy()
    ew = df["ew"].astype(float).to_numpy()
    smb = ew - mkt  # 小盘溢价因子 (等权 − 市值加权)

    # 多元 OLS 含截距: p = α + β_mkt·mkt + β_smb·smb
    X = np.column_stack([np.ones(n), mkt, smb])
    coef = _ols_beta(p, X)
    if coef is None:
        return out
    alpha_daily = float(coef[0])
    beta_mkt = float(coef[1])
    beta_smb = float(coef[2])
    alpha_annualized = (1.0 + alpha_daily) ** periods_per_year - 1.0

    resid = p - (alpha_daily + beta_mkt * mkt + beta_smb * smb)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((p - np.mean(p)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def _total(x: np.ndarray) -> float:
        return float(np.prod(1.0 + x) - 1.0)

    total_p = _total(p)
    total_mkt = _total(mkt)
    total_ew = _total(ew)
    total_smb = total_ew - total_mkt  # 窗口内小盘溢价 (等权跑赢市值加权的量)
    market_contribution = beta_mkt * total_mkt
    size_contribution = beta_smb * total_smb
    alpha_contribution = total_p - market_contribution - size_contribution

    out.update({
        "beta_mkt": round(beta_mkt, 4),
        "beta_smb": round(beta_smb, 4),
        "alpha_annualized_pct": round(alpha_annualized * 100.0, 3),
        "strategy_return_pct": round(total_p * 100.0, 3),
        "market_return_pct": round(total_mkt * 100.0, 3),
        "equalweight_return_pct": round(total_ew * 100.0, 3),
        "smb_return_pct": round(total_smb * 100.0, 3),
        "market_contribution_pct": round(market_contribution * 100.0, 3),
        "size_contribution_pct": round(size_contribution * 100.0, 3),
        "alpha_contribution_pct": round(alpha_contribution * 100.0, 3),
        "r_squared": round(r_squared, 4),
        "n_aligned": n,
    })
    return out


def _ols_beta(y: np.ndarray, X: np.ndarray):
    """多元 OLS 闭式解: β = (XᵀX)⁻¹Xᵀy。X 为设计矩阵 (含截距列时截距作为首列系数返回)。
    奇异矩阵退化为 lstsq。"""
    XtX = X.T @ X
    try:
        return np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X, y, rcond=None)[0]


def load_index_series(
    root,
    start,
    end,
    symbol: str = "sh.000001",
    path: str = "replay_data/index_series.parquet",
) -> Optional[pd.Series]:
    """
    从 index_series.parquet 切出指定窗口的指数收盘价序列 (日期索引)。

    消除 A/B 脚本里重复的 `idx[(idx.date>=T)&(idx.date<=T_end)].set_index("date")["close"]`。

    Args:
        root: 项目根路径 (Path)。
        start/end: 窗口起止 (pd.Timestamp / date / str)。
        symbol: 指数符号 (sh.000001 上证 / sh.000300 沪深300)。
        path: 相对 root 的 parquet 路径。

    Returns:
        指数收盘价 Series (索引为 datetime), 缺失或文件不存在返回 None。
    """
    fp = root / path
    if not fp.exists():
        return None
    idx = pd.read_parquet(fp)
    if "date" not in idx.columns or "symbol" not in idx.columns:
        return None
    idx["date"] = pd.to_datetime(idx["date"])
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    win = idx[(idx["date"] >= start) & (idx["date"] <= end)]
    sub = win[win["symbol"] == symbol]
    if sub.empty:
        return None
    return sub.set_index("date")["close"].sort_index()
