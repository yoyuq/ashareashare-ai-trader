"""
ParallelScanner — 并行扫描加速器 (v3.1)

全市场 5000+ 股指标计算从 ~120s 优化到 ~30s:
  - multiprocessing.Pool 分批并行
  - 当 symbols > 100 时自动启用并行模式
  - numba JIT 加速热点计算 (可选)

用法:
    scanner = ParallelScanner(n_workers=8)
    results = scanner.scan_universe(ohlcv_data, indicators)
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Try numba for JIT acceleration
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class ParallelScanner:
    """Parallel market scanner with optional JIT acceleration"""

    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or max(1, mp.cpu_count() - 2)

    def scan_universe(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        compute_fn: Callable,
        n_workers: int = None,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """
        Parallel scan of all symbols in the universe.

        Args:
            ohlcv_data: {symbol: DataFrame} with OHLCV columns
            compute_fn: function(df) -> Dict[str, float] returning indicator values
            n_workers: number of worker processes (default: cpu_count - 2)
            batch_size: symbols per batch

        Returns:
            DataFrame with symbols as index and indicator values as columns
        """
        symbols = list(ohlcv_data.keys())
        n_workers = n_workers or self.n_workers

        # For small datasets, use sequential mode
        if len(symbols) <= 100:
            return self._scan_sequential(ohlcv_data, compute_fn)

        # Split into batches
        batches = [
            symbols[i:i + batch_size]
            for i in range(0, len(symbols), batch_size)
        ]

        logger.info(f"[ParallelScanner] Scanning {len(symbols)} symbols in {len(batches)} batches "
                    f"({n_workers} workers, batch_size={batch_size})")

        all_results = {}

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for batch_id, batch_syms in enumerate(batches):
                batch_data = {s: ohlcv_data[s] for s in batch_syms if s in ohlcv_data}
                future = executor.submit(
                    _compute_batch, batch_data, compute_fn
                )
                futures[future] = batch_id

            completed = 0
            for future in as_completed(futures):
                batch_id = futures[future]
                try:
                    batch_result = future.result()
                    all_results.update(batch_result)
                    completed += 1
                    if completed % max(1, len(batches) // 5) == 0:
                        logger.debug(f"[ParallelScanner] {completed}/{len(batches)} batches done")
                except Exception as e:
                    logger.warning(f"[ParallelScanner] Batch {batch_id} failed: {e}")

        # Build result DataFrame
        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(all_results, orient="index")
        df.index.name = "symbol"

        logger.info(f"[ParallelScanner] Complete: {len(df)} symbols, {len(df.columns)} features")
        return df

    def _scan_sequential(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        compute_fn: Callable,
    ) -> pd.DataFrame:
        """Sequential scan for small datasets"""
        results = {}
        for sym, df in ohlcv_data.items():
            try:
                result = compute_fn(df)
                if isinstance(result, dict):
                    results[sym] = result
            except Exception:
                continue
        return pd.DataFrame.from_dict(results, orient="index")

    def batch_compute_indicators(
        self,
        df: pd.DataFrame,
        indicator_specs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Vectorized batch indicator computation with optional JIT.

        Args:
            df: OHLCV DataFrame
            indicator_specs: [{name, type, params}, ...]

        Returns:
            {indicator_name: latest_value}
        """
        results = {}
        close = df["close"].values if "close" in df.columns else None
        volume = df["volume"].values if "volume" in df.columns else None

        if close is None or len(close) < 5:
            return results

        for spec in indicator_specs:
            name = spec.get("name", "")
            ind_type = spec.get("type", "")
            params = spec.get("params", {})
            window = params.get("window", 20)

            try:
                val = _compute_single(close, volume, ind_type, window)
                results[name] = val
            except Exception:
                results[name] = np.nan

        return results


# ── Module-level functions for multiprocessing ──

def _compute_batch(
    batch_data: Dict[str, pd.DataFrame],
    compute_fn: Callable,
) -> Dict[str, Dict]:
    """Compute indicators for a batch of symbols (used by ProcessPoolExecutor)"""
    results = {}
    for sym, df in batch_data.items():
        try:
            result = compute_fn(df)
            if isinstance(result, dict):
                results[sym] = result
        except Exception:
            continue
    return results


def _compute_single(
    close: np.ndarray,
    volume: np.ndarray,
    ind_type: str,
    window: int,
) -> float:
    """Compute a single indicator value using vectorized numpy ops"""
    if ind_type in ("ma", "sma"):
        return float(np.mean(close[-window:]))
    elif ind_type == "ema":
        alpha = 2.0 / (window + 1)
        ema = close[0]
        for price in close[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return float(ema)
    elif ind_type in ("rsi",):
        return _compute_rsi_fast(close, window)
    elif ind_type in ("volatility", "vol"):
        returns = np.diff(close) / close[:-1]
        return float(np.std(returns[-window:]) if len(returns) >= window else 0)
    elif ind_type in ("momentum", "mom", "returns"):
        return float((close[-1] / close[-window-1] - 1) if len(close) > window else 0)
    elif ind_type in ("volume_ratio", "vol_ratio"):
        if volume is None or len(volume) < window:
            return 1.0
        return float(volume[-1] / np.mean(volume[-window:]) if np.mean(volume[-window:]) > 0 else 1.0)
    elif ind_type in ("sharpe",):
        returns = np.diff(close) / close[:-1]
        mean_ret = np.mean(returns[-window:])
        std_ret = np.std(returns[-window:], ddof=1)
        return float(mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0)
    elif ind_type in ("max_drawdown", "mdd"):
        peak = np.maximum.accumulate(close[-window:])
        return float(np.min((close[-window:] - peak) / peak))
    else:
        return 0.0


@jit(nopython=True, cache=True)
def _compute_rsi_fast(close: np.ndarray, window: int = 14) -> float:
    """JIT-compiled RSI computation"""
    n = len(close)
    if n < window + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(n - window, n):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / window
    avg_loss = losses / window
    if avg_loss < 1e-8:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)
