"""
FactorFactory — 因子自动发现引擎 (v3.1)

自动从基础算子生成候选因子,评估 IC/IR, 排序输出因子榜单。

Pipeline:
  1. 定义基础算子 (returns/volume/volatility/rsi/bb/etc.)
  2. 参数化组合: 算子 x 窗口 x 变换
  3. 计算每个候选因子的截面 IC/IR
  4. 去相关: 相关因子(>0.7)保留最高 IR
  5. 排序输出因子榜单 → reports/factor_leaderboard.json

用法:
    factory = FactorFactory()
    candidates = factory.generate(ohlcv_data, symbols)
    ranked = factory.screen(candidates, forward_returns, min_ic=0.02, min_ir=0.3)
    factory.save_leaderboard(ranked)
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from analysis.factor_evaluator import FactorEvaluation, FactorEvaluator


@dataclass
class FactorCandidate:
    """候选因子"""
    name: str
    category: str                    # momentum | value | volatility | volume | quality | custom
    formula: str                     # Human-readable description
    params: Dict[str, Any] = field(default_factory=dict)
    evaluation: Optional[FactorEvaluation] = None

    def to_dict(self) -> dict:
        base = {
            "name": self.name,
            "category": self.category,
            "formula": self.formula,
            "params": self.params,
        }
        if self.evaluation:
            base.update(self.evaluation.to_dict())
        return base


class FactorFactory:
    """
    因子自动发现引擎

    生成候选因子 → IC/IR 评估 → 去相关 → 排行榜
    """

    def __init__(self, report_dir: str = "reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = FactorEvaluator()

    # ── 基础算子库 ──
    BASE_OPERATORS: Dict[str, Callable] = {
        "returns":          lambda c, n: c.pct_change(n),
        "log_returns":      lambda c, n: np.log(c / c.shift(n)),
        "volume_ratio":     lambda v, n: v / v.rolling(n).mean(),
        "volume_std":       lambda v, n: v.rolling(n).std() / v.rolling(n).mean(),
        "volatility":       lambda c, n: c.pct_change().rolling(n).std(),
        "rsi":              lambda c, n: _compute_rsi(c, n),
        "bias_ma":          lambda c, n: c / c.rolling(n).mean() - 1,
        "ma_cross":         lambda c, n: (c.rolling(n).mean() - c.rolling(2 * n).mean()) / c.rolling(2 * n).mean(),
        "bb_position":      lambda c, n: _compute_bb_position(c, n),
        "drawdown":         lambda c, n: c / c.rolling(n).max() - 1,
        "high_low_ratio":   lambda h, l, c, n: (h - l).rolling(n).mean() / c.rolling(n).mean(),
        "price_volume_corr": lambda c, v, n: c.pct_change().rolling(n).corr(v.pct_change()),
    }

    # ── 参数窗口 ──
    DEFAULT_WINDOWS = [5, 10, 20, 30, 60, 120]

    def generate(
        self,
        data: Dict[str, pd.DataFrame],
        symbols: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
    ) -> List[FactorCandidate]:
        """
        Generate candidate factors from base operators.

        Args:
            data: {symbol: DataFrame} with columns [close, volume, high, low]
            symbols: symbols to include
            windows: rolling windows to use

        Returns:
            List of FactorCandidate objects
        """
        if windows is None:
            windows = self.DEFAULT_WINDOWS
        if symbols is None:
            symbols = list(data.keys())

        candidates = []

        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty or "close" not in df.columns:
                continue

            close = df["close"]
            volume = df.get("volume", pd.Series(index=close.index))
            high = df.get("high", close)
            low = df.get("low", close)

            for w in windows:
                if len(close) < w + 5:
                    continue

                # Momentum factors
                candidates.append(FactorCandidate(
                    name=f"returns_{w}d", category="momentum",
                    formula=f"Price return over {w} days",
                    params={"window": w},
                ))
                candidates.append(FactorCandidate(
                    name=f"log_returns_{w}d", category="momentum",
                    formula=f"Log price return over {w} days",
                    params={"window": w},
                ))

                # Mean-reversion factors
                candidates.append(FactorCandidate(
                    name=f"bias_ma_{w}d", category="mean_reversion",
                    formula=f"Price deviation from {w}-day MA",
                    params={"window": w},
                ))
                if w >= 10:
                    candidates.append(FactorCandidate(
                        name=f"ma_cross_{w}d", category="trend",
                        formula=f"{w}/{2*w} day MA crossover signal",
                        params={"window": w},
                    ))

                # Volatility factors
                candidates.append(FactorCandidate(
                    name=f"volatility_{w}d", category="volatility",
                    formula=f"{w}-day return volatility",
                    params={"window": w},
                ))

                # Volume factors
                if not volume.empty:
                    candidates.append(FactorCandidate(
                        name=f"vol_ratio_{w}d", category="volume",
                        formula=f"Volume / {w}-day average volume",
                        params={"window": w},
                    ))
                    candidates.append(FactorCandidate(
                        name=f"vol_std_{w}d", category="volume",
                        formula=f"{w}-day volume coefficient of variation",
                        params={"window": w},
                    ))

                # Technical indicators
                candidates.append(FactorCandidate(
                    name=f"rsi_{w}d", category="momentum",
                    formula=f"RSI with {w}-day window",
                    params={"window": w},
                ))
                candidates.append(FactorCandidate(
                    name=f"bb_position_{w}d", category="volatility",
                    formula=f"Bollinger Band position ({w}-day, 2 std)",
                    params={"window": w},
                ))

                # Drawdown
                candidates.append(FactorCandidate(
                    name=f"drawdown_{w}d", category="risk",
                    formula=f"Drawdown from {w}-day high",
                    params={"window": w},
                ))

                # Cross-sectional
                if not volume.empty:
                    candidates.append(FactorCandidate(
                        name=f"price_vol_corr_{w}d", category="volume",
                        formula=f"Price-volume correlation over {w} days",
                        params={"window": w},
                    ))

        # Deduplicate by name
        seen = set()
        unique = []
        for c in candidates:
            if c.name not in seen:
                seen.add(c.name)
                unique.append(c)

        logger.info(f"[FactorFactory] Generated {len(unique)} candidate factors "
                    f"({len(symbols)} stocks x {len(windows)} windows)")
        return unique

    def screen(
        self,
        candidates: List[FactorCandidate],
        data: Dict[str, pd.DataFrame],
        forward_period: int = 5,
        min_ic: float = 0.02,
        min_ir: float = 0.3,
        correlation_threshold: float = 0.7,
    ) -> List[FactorCandidate]:
        """
        Screen and evaluate candidate factors.

        1. Compute factor values and forward returns for each stock
        2. Evaluate cross-sectional IC/IR
        3. Remove factors below min_ic/min_ir
        4. De-duplicate correlated factors (keep highest IR)
        5. Rank by IR descending

        Returns:
            Ranked list of FactorCandidate with evaluation results
        """
        symbols = list(data.keys())
        windows = list(self.DEFAULT_WINDOWS)
        evaluated = []

        for cand in candidates:
            w = cand.params.get("window", 20)
            op_name = cand.name.split(f"_{w}d")[0] if f"_{w}d" in cand.name else ""
            op_name = op_name or cand.category or "unknown"

            # Compute factor values for all stocks
            factor_vals = {}
            fwd_returns = {}

            for sym in symbols:
                df = data.get(sym)
                if df is None or df.empty or "close" not in df.columns:
                    continue

                close = df["close"]
                volume = df.get("volume", pd.Series(dtype=float))
                high = df.get("high", close)
                low = df.get("low", close)

                if len(close) < w + forward_period + 5:
                    continue

                # 计算因子值 — 因子时点必须与前瞻收益对齐:
                # 用"截至 forward_period 根之前"的数据计算因子值, 使其时点等于
                # 未来收益的起点, 避免目标泄漏 (此前因子取最后一根bar → IC 恒等式虚高)
                try:
                    if forward_period > 0:
                        ev_close = close.iloc[:-forward_period]
                        ev_volume = volume.iloc[:-forward_period]
                        ev_high = high.iloc[:-forward_period]
                        ev_low = low.iloc[:-forward_period]
                    else:
                        ev_close, ev_volume, ev_high, ev_low = close, volume, high, low

                    fv = self._compute_factor_value(
                        cand.name, ev_close, ev_volume, ev_high, ev_low, w
                    )
                    if fv is not None and not np.isnan(fv):
                        factor_vals[sym] = fv

                        # Forward return: 因子时点起未来 N 日的已实现收益
                        fwd_ret = close.iloc[-1] / close.iloc[-(1 + forward_period)] - 1
                        fwd_returns[sym] = fwd_ret
                except Exception:
                    continue

            if len(factor_vals) < 10:
                continue

            # Evaluate
            fv_series = pd.Series(factor_vals)
            fr_series = pd.Series(fwd_returns)

            eval_result = self.evaluator.evaluate(
                fv_series, fr_series,
                name=cand.name,
                category=cand.category,
            )

            if abs(eval_result.rank_ic) >= min_ic and eval_result.ir >= min_ir:
                cand.evaluation = eval_result
                evaluated.append(cand)

        # De-correlate: group by correlation, keep highest IR
        screened = self._decorrelate(evaluated, correlation_threshold)

        # Rank by IR
        screened.sort(key=lambda c: c.evaluation.ir if c.evaluation else 0, reverse=True)

        logger.info(f"[FactorFactory] Screened: {len(screened)}/{len(candidates)} factors "
                    f"(IC>={min_ic}, IR>={min_ir}, corr<{correlation_threshold})")
        return screened

    def rank_factors(
        self,
        data: Dict[str, pd.DataFrame],
        symbols: Optional[List[str]] = None,
        top_n: int = 30,
    ) -> List[FactorCandidate]:
        """
        Full pipeline: generate → screen → rank

        Returns top-N factors by IR.
        """
        candidates = self.generate(data, symbols)
        ranked = self.screen(candidates, data)
        return ranked[:top_n]

    def save_leaderboard(self, factors: List[FactorCandidate], path: str = None):
        """Save factor leaderboard to JSON"""
        if path is None:
            path = self.report_dir / f"factor_leaderboard_{date.today().isoformat()}.json"
        else:
            path = Path(path)

        data = {
            "generated_at": date.today().isoformat(),
            "total_factors": len(factors),
            "factors": [
                {
                    "rank": i + 1,
                    **f.to_dict(),
                }
                for i, f in enumerate(factors)
            ],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[FactorFactory] Leaderboard saved: {path}")
        return str(path)

    def get_leaderboard(self, path: str = None) -> List[Dict]:
        """Read the latest factor leaderboard"""
        if path is None:
            # Find latest
            files = sorted(self.report_dir.glob("factor_leaderboard_*.json"))
            if not files:
                return []
            path = files[-1]

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("factors", [])

    # ── Internal helpers ──

    def _compute_factor_value(
        self,
        name: str,
        close: pd.Series,
        volume: pd.Series,
        high: pd.Series,
        low: pd.Series,
        window: int,
    ) -> Optional[float]:
        """Compute the latest factor value for a single stock"""
        try:
            parts = name.split("_")
            base = parts[0].replace("ratio", "ratio").replace("position", "position")

            if name.startswith("returns_"):
                return float(close.pct_change(window).iloc[-1])
            elif name.startswith("log_returns_"):
                return float(np.log(close.iloc[-1] / close.iloc[-window-1]) if len(close) > window else 0)
            elif name.startswith("bias_ma_"):
                return float(close.iloc[-1] / close.rolling(window).mean().iloc[-1] - 1)
            elif name.startswith("ma_cross_"):
                ma_s = close.rolling(window).mean()
                ma_l = close.rolling(2 * window).mean()
                return float((ma_s.iloc[-1] - ma_l.iloc[-1]) / ma_l.iloc[-1] if ma_l.iloc[-1] != 0 else 0)
            elif name.startswith("volatility_"):
                return float(close.pct_change().rolling(window).std().iloc[-1])
            elif name.startswith("vol_ratio_") and not volume.empty:
                return float((volume.iloc[-1] / volume.rolling(window).mean().iloc[-1]) if volume.rolling(window).mean().iloc[-1] > 0 else 1)
            elif name.startswith("vol_std_") and not volume.empty:
                return float(volume.rolling(window).std().iloc[-1] / max(volume.rolling(window).mean().iloc[-1], 1))
            elif name.startswith("rsi_"):
                return _compute_rsi(close, window).iloc[-1] if len(close) > window else 50
            elif name.startswith("bb_position_"):
                return _compute_bb_position(close, window)
            elif name.startswith("drawdown_"):
                return float(close.iloc[-1] / close.rolling(window).max().iloc[-1] - 1)
            elif name.startswith("price_vol_corr_") and not volume.empty:
                return float(close.pct_change().rolling(window).corr(volume.pct_change()).iloc[-1])
            elif name.startswith("high_low_ratio_"):
                return float(((high - low).rolling(window).mean().iloc[-1] / close.iloc[-1]) if close.iloc[-1] > 0 else 0)
            else:
                return 0.0
        except Exception:
            return None

    def _decorrelate(
        self,
        candidates: List[FactorCandidate],
        threshold: float = 0.7,
    ) -> List[FactorCandidate]:
        """Remove correlated factors, keeping the one with highest IR"""
        if len(candidates) <= 1:
            return candidates

        # Collect factor values matrix
        # Since we have candidates, we need to recompute values... expensive.
        # For now: group by category, keep top-2 per category
        by_category: Dict[str, List[FactorCandidate]] = {}
        for c in candidates:
            cat = c.category or "other"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(c)

        result = []
        for cat, cats in by_category.items():
            cats.sort(key=lambda x: x.evaluation.ir if x.evaluation else 0, reverse=True)
            result.extend(cats[:2])  # Keep top 2 per category

        return result


# ── Free functions for indicator computation ──

def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-8)
    return 100 - (100 / (1 + rs))


def _compute_bb_position(close: pd.Series, window: int = 20, n_std: float = 2.0) -> float:
    """Compute Bollinger Band position (0=lower, 1=upper)"""
    try:
        ma = close.rolling(window).mean().iloc[-1]
        std = close.rolling(window).std().iloc[-1]
        price = close.iloc[-1]
        lower = ma - n_std * std
        upper = ma + n_std * std
        if upper - lower < 1e-8:
            return 0.5
        return float((price - lower) / (upper - lower))
    except Exception:
        return 0.5
