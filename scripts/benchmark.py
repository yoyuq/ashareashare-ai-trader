#!/usr/bin/env python3
"""
性能基准测试 — 测量核心操作延迟和吞吐量

使用:
    python scripts/benchmark.py                # 全部基准
    python scripts/benchmark.py --quick         # 快速模式 (减少样本)
    python scripts/benchmark.py --module indicators  # 仅测指标
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def _make_ohlcv(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 50 + np.cumsum(rng.normal(0.02, 1.5, n))
    close = np.maximum(close, 2)
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="B"),
        "open": close - rng.uniform(-0.3, 0.3, n),
        "high": close + rng.uniform(0.2, 2.0, n),
        "low": close - rng.uniform(0.2, 2.0, n),
        "close": close,
        "volume": rng.integers(100000, 5000000, n).astype(float),
    })


def bench_indicators(df: pd.DataFrame, n_runs: int = 5) -> dict:
    """130+ 技术指标计算性能"""
    from analysis.indicators import TechnicalAnalyzer
    analyzer = TechnicalAnalyzer()
    # 预热
    analyzer.compute_all(df.head(100))
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = analyzer.compute_all(df)
        times.append(time.perf_counter() - t0)
    return {
        "operation": "130+ indicators (500 bars)",
        "avg_ms": round(np.mean(times) * 1000, 1),
        "min_ms": round(np.min(times) * 1000, 1),
        "max_ms": round(np.max(times) * 1000, 1),
        "bars_per_sec": round(len(df) / np.mean(times), 0),
    }


def bench_strategies(df: pd.DataFrame, n_runs: int = 10) -> list:
    """9 个策略回测性能"""
    from analysis.recommender import STRATEGY_BACKTESTERS
    results = []
    for sid, bt_func in STRATEGY_BACKTESTERS.items():
        bt_func(df)  # 预热
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            bt_func(df)
            times.append(time.perf_counter() - t0)
        results.append({
            "strategy": sid,
            "avg_ms": round(np.mean(times) * 1000, 2),
            "total_trades": bt_func(df).get("signals", 0),
        })
    results.sort(key=lambda r: r["avg_ms"])
    return results


def bench_regime(df: pd.DataFrame, n_runs: int = 10) -> dict:
    """市场状态识别性能"""
    from analysis.regime import MarketRegimeDetector
    detector = MarketRegimeDetector()
    detector.detect(df.head(100))  # 预热
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = detector.detect(df)
        times.append(time.perf_counter() - t0)
    return {
        "operation": "Market regime detection (252 bars)",
        "avg_ms": round(np.mean(times) * 1000, 1),
        "regime": result.regime.value,
        "confidence": round(result.confidence, 2),
    }


def bench_risk_controls(n_runs: int = 20) -> list:
    """风控层性能"""
    from analysis.risk_controls import SignalGrader, PortfolioRiskManager
    results = []

    # 信号分级
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        SignalGrader.grade(72, 0.58, "weak_bull", "neutral", 3)
        times.append(time.perf_counter() - t0)
    results.append({
        "operation": "SignalGrader.grade()",
        "avg_us": round(np.mean(times) * 1_000_000, 1),
    })

    # 风控更新
    rm = PortfolioRiskManager(100000)
    rets = pd.Series(np.random.normal(0.001, 0.012, 60))
    rm.update(105000, rets)  # 预热
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        rm.update(105000 + np.random.uniform(-1000, 1000), rets)
        times.append(time.perf_counter() - t0)
    results.append({
        "operation": "PortfolioRiskManager.update()",
        "avg_us": round(np.mean(times) * 1_000_000, 1),
    })

    # 防踩踏
    positions = {f"S{i}": None for i in range(5)}
    changes = {f"S{i}": -0.04 + i * 0.01 for i in range(5)}
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        rm.check_stampede_risk(positions, changes)
        times.append(time.perf_counter() - t0)
    results.append({
        "operation": "check_stampede_risk (5 pos)",
        "avg_us": round(np.mean(times) * 1_000_000, 1),
    })

    return results


def bench_backtest(df: pd.DataFrame, n_runs: int = 3) -> dict:
    """事件驱动回测引擎性能"""
    from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
    from datetime import date

    config = BacktestConfig(
        initial_capital=100000.0,
        start_date=date(2022, 1, 1),
        end_date=date(2024, 12, 31),
    )

    def simple_strat(today, bars, broker):
        sym = list(bars.keys())[0] if bars else None
        if sym is None:
            return
        if sym not in broker.account.positions:
            broker.buy(sym, 100)
        elif broker.account.positions[sym].quantity >= 300:
            broker.sell(sym, 300)

    # Ensure ticker symbol for broker
    df_indexed = df.copy()
    df_indexed["date"] = pd.to_datetime(df_indexed["date"])

    # Warm-up
    engine = EventDrivenBacktestEngine(config)
    engine.load_data("test", df_indexed)
    engine.run(simple_strat, progress_bar=False)

    times = []
    for _ in range(n_runs):
        engine = EventDrivenBacktestEngine(config)
        engine.load_data("test", df_indexed)
        t0 = time.perf_counter()
        result = engine.run(simple_strat, progress_bar=False)
        times.append(time.perf_counter() - t0)

    return {
        "operation": f"Event-driven backtest ({len(df)} bars)",
        "avg_ms": round(np.mean(times) * 1000, 1),
        "trades": result.total_trades,
        "sharpe": round(result.sharpe_ratio, 2),
    }


def bench_overfitting(df: pd.DataFrame, n_runs: int = 3) -> dict:
    """过拟合检测性能"""
    from backtest.overfitting import OverfittingGuard

    rets = df["close"].pct_change().dropna()
    guard = OverfittingGuard(n_simulations=100)
    guard.evaluate(rets)  # 预热

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        report = guard.evaluate(rets)
        times.append(time.perf_counter() - t0)

    return {
        "operation": "OverfittingGuard.evaluate (PBO+DSR+MC, 100 sims)",
        "avg_ms": round(np.mean(times) * 1000, 1),
        "pbo": round(report.pbo, 3),
        "is_overfit": report.is_overfit,
    }


def bench_paper_trading(n_runs: int = 20) -> dict:
    """模拟交易性能"""
    from simulation.portfolio import PortfolioManager
    from simulation.paper_trader import PaperTradingEngine

    manager = PortfolioManager()
    manager.reset()
    engine = PaperTradingEngine(manager)

    # 买入
    times = []
    for _ in range(n_runs):
        manager.reset()
        t0 = time.perf_counter()
        engine.execute_buy("sh.600519", "茅台", 100.0,
                           recommendation={"strategy_id": "test", "win_rate": 0.55})
        times.append(time.perf_counter() - t0)

    buy_result = {
        "operation": "PaperTradingEngine.execute_buy()",
        "avg_us": round(np.mean(times) * 1_000_000, 1),
    }

    # MTM
    manager.reset()
    engine.execute_buy("sh.600519", "茅台", 100.0)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        engine.mark_to_market({"sh.600519": 105.0})
        times.append(time.perf_counter() - t0)

    mtm_result = {
        "operation": "PaperTradingEngine.mark_to_market()",
        "avg_us": round(np.mean(times) * 1_000_000, 1),
    }

    return [buy_result, mtm_result]


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="A股智能分析Agent — 性能基准")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    parser.add_argument("--module", type=str, help="仅测试指定模块")
    args = parser.parse_args()

    n_runs_map = {"indicators": 5, "strategies": 10, "regime": 10,
                  "risk": 20, "backtest": 3, "overfitting": 3, "paper": 20}
    if args.quick:
        n_runs_map = {k: max(1, v // 3) for k, v in n_runs_map.items()}

    print("=" * 65)
    print("  A股智能分析Agent v2.14 — 性能基准测试")
    print("=" * 65)

    df = _make_ohlcv(500)

    benchmarks = []

    def run(label, fn):
        if args.module and args.module != label:
            return
        print(f"\n--- {label} ---")
        try:
            result = fn(df if label not in ("risk", "paper") else None)
            if isinstance(result, list):
                for r in result:
                    _print_result(r)
                    benchmarks.append(r)
            else:
                _print_result(result)
                benchmarks.append(result)
        except Exception as e:
            print(f"  FAIL: {e}")

    def _print_result(r):
        if "avg_ms" in r:
            label = r.get("operation", r.get("strategy", "?"))
            print(f"  {label}: {r['avg_ms']}ms avg")
            extra = {k: v for k, v in r.items()
                    if k not in ("operation", "strategy", "avg_ms", "min_ms", "max_ms")}
            if extra:
                print(f"    {' | '.join(f'{k}={v}' for k, v in extra.items())}")
        elif "avg_us" in r:
            label = r.get("operation", "?")
            print(f"  {label}: {r['avg_us']}us avg")

    def run_list(label, fn):
        if args.module and args.module != label:
            return
        print(f"\n--- {label} ---")
        try:
            results = fn(df if label not in ("risk", "paper") else None)
            for r in results:
                _print_result(r)
                benchmarks.append(r)
        except Exception as e:
            print(f"  FAIL: {e}")

    run("indicators", lambda df: [bench_indicators(df, n_runs_map["indicators"])])
    run_list("strategies", lambda df: bench_strategies(df, n_runs_map["strategies"]))
    run("regime", lambda df: [bench_regime(df.head(252), n_runs_map["regime"])])
    run_list("risk", lambda _: bench_risk_controls(n_runs_map["risk"]))
    run("backtest", lambda df: [bench_backtest(df, n_runs_map["backtest"])])
    run("overfitting", lambda df: [bench_overfitting(df, n_runs_map["overfitting"])])
    run_list("paper", lambda _: bench_paper_trading(n_runs_map["paper"]))

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    for b in benchmarks:
        label = b.get("operation", b.get("strategy", "?"))
        if "avg_ms" in b:
            perf = "[FAST]" if b.get("avg_ms", 999) < 50 else ("[OK]" if b.get("avg_ms", 999) < 500 else "[SLOW]")
            print(f"  {perf} {label}: {b['avg_ms']}ms")
        elif "avg_us" in b:
            print(f"  [FAST] {label}: {b['avg_us']}us")
    print()


if __name__ == "__main__":
    main()
