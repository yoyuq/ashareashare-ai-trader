#!/usr/bin/env python3
"""
全量历史回测验证 — 9策略 × 60+标的 × 多市场状态

运行所有策略在所有精选池标的上回测, 按市场状态/行业分组统计,
输出综合性能报告。

使用:
    python scripts/historical_validation.py                    # 全量 (较慢, ~10分钟)
    python scripts/historical_validation.py --quick             # 快速 (前10只, 2年)
    python scripts/historical_validation.py --pool tech         # 仅科技池
    python scripts/historical_validation.py --output report.json
"""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def fetch_data(symbols: List[str], years: int = 3) -> Dict[str, pd.DataFrame]:
    """从 Baostock 拉取历史日K线"""
    import baostock as bs

    bs.login()
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    data = {}
    for i, sym in enumerate(symbols):
        try:
            rs = bs.query_history_k_data_plus(
                sym, "date,open,high,low,close,volume,preclose",
                start_date=start, end_date=end,
                frequency="d", adjustflag="2",
            )
            if rs.error_code == "0":
                rows = [rs.get_row_data() for _ in range(10000) if rs.next()]
                if rows:
                    df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume","preclose"])
                    for col in ["open","high","low","close","volume","preclose"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df.dropna()
                    if len(df) > 100:
                        data[sym] = df
        except Exception as e:
            print(f"  {sym}: {e}")
    bs.logout()
    return data


def detect_regime(df: pd.DataFrame) -> str:
    """快速市场状态检测 (基于这段数据的MA排列)"""
    if len(df) < 60:
        return "unknown"
    close = df["close"].values
    ma20 = pd.Series(close).rolling(20).mean().values[-1]
    ma60 = pd.Series(close).rolling(60).mean().values[-1]
    ret_60 = (close[-1] / close[-60] - 1) * 100 if len(close) >= 60 else 0

    if close[-1] > ma20 > ma60 and ret_60 > 10:
        return "strong_bull"
    elif close[-1] > ma20:
        return "weak_bull"
    elif close[-1] < ma20 < ma60 and ret_60 < -10:
        return "strong_bear"
    elif close[-1] < ma20:
        return "weak_bear"
    else:
        return "range_bound"


def run_validation(symbols: List[str], years: int = 3, max_stocks: int = 0) -> Dict:
    """全量验证主函数"""
    from analysis.recommender import STRATEGY_BACKTESTERS

    # 获取行业映射
    from scripts.shared import NAME_MAP, load_watchlist
    import yaml

    # 行业分类
    industry_map = {}
    cfg_path = Path(__file__).parent.parent / "config" / "symbols.yaml"
    try:
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        for pool_name, pool_data in cfg.get("watchlist", {}).items():
            if isinstance(pool_data, dict) and "symbols" in pool_data:
                for sym in pool_data["symbols"]:
                    if pool_name not in ("default",):
                        industry_map[sym] = pool_name
    except Exception:
        pass

    target = symbols[:max_stocks] if max_stocks > 0 else symbols
    print(f"拉取 {len(target)} 只标的历史数据 ({years}年)...")
    data = fetch_data(target, years)
    print(f"成功获取 {len(data)} 只")

    if not data:
        return {"error": "no data"}

    # 结果容器
    strategy_results: Dict[str, List[Dict]] = {sid: [] for sid in STRATEGY_BACKTESTERS}
    regime_results: Dict[str, List[Dict]] = {
        "strong_bull": [], "weak_bull": [], "range_bound": [],
        "weak_bear": [], "strong_bear": [],
    }
    stock_scores: List[Dict] = []

    total = len(data)
    print(f"\n运行 {len(STRATEGY_BACKTESTERS)} 策略 × {total} 只标的 = {len(STRATEGY_BACKTESTERS) * total} 次回测...")

    for i, (sym, df) in enumerate(data.items()):
        regime = detect_regime(df)
        name = NAME_MAP.get(sym, sym)
        industry = industry_map.get(sym, "其他")

        best_strategy = None
        best_sharpe = -999

        for sid, bt_func in STRATEGY_BACKTESTERS.items():
            try:
                result = bt_func(df)
                signals = result.get("signals", 0)
                if signals < 5:
                    continue  # 信号太少, 不可靠

                entry = {
                    "symbol": sym, "name": name, "strategy": sid,
                    "regime": regime, "industry": industry,
                    "signals": signals,
                    "win_rate": round(result["win_rate"], 3),
                    "profit_factor": round(result["profit_factor"], 2),
                    "expected_value": round(result["expected_value"], 2),
                    "sharpe": round(result["sharpe"], 3),
                    "max_dd": round(result["max_dd"], 1),
                    "avg_win": round(result.get("avg_win", 0), 2),
                    "avg_loss": round(result.get("avg_loss", 0), 2),
                }
                strategy_results[sid].append(entry)
                if regime in regime_results:
                    regime_results[regime].append(entry)

                if result["sharpe"] > best_sharpe:
                    best_sharpe = result["sharpe"]
                    best_strategy = sid

            except Exception as e:
                print(f"  {sym} {sid}: ERROR {e}")

        # 标的最佳策略
        if best_strategy:
            stock_scores.append({
                "symbol": sym, "name": name, "best_strategy": best_strategy,
                "best_sharpe": round(best_sharpe, 3), "regime": regime,
                "industry": industry,
                "close": float(df["close"].iloc[-1]),
            })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{total} 完成...")

    # 汇总统计
    summary = _build_summary(strategy_results, regime_results, stock_scores)
    return {
        "meta": {
            "date": date.today().isoformat(),
            "symbols": len(data),
            "years": years,
            "total_backtests": sum(len(v) for v in strategy_results.values()),
        },
        "per_strategy": _summarize_strategy(strategy_results),
        "per_regime": _summarize_regime(regime_results),
        "top_stocks": sorted(stock_scores, key=lambda s: s["best_sharpe"], reverse=True)[:20],
        "summary": summary,
    }


def _summarize_strategy(results: Dict[str, List[Dict]]) -> Dict:
    """按策略汇总"""
    summary = {}
    for sid, entries in results.items():
        if not entries:
            summary[sid] = {"total": 0}
            continue
        wr = np.mean([e["win_rate"] for e in entries])
        sr = np.mean([e["sharpe"] for e in entries])
        pf = np.mean([e["profit_factor"] for e in entries])
        dd = np.mean([e["max_dd"] for e in entries])
        summary[sid] = {
            "total": len(entries),
            "avg_win_rate": round(wr, 3),
            "avg_sharpe": round(sr, 3),
            "avg_profit_factor": round(pf, 2),
            "avg_max_dd": round(dd, 1),
            "rank": 0,  # filled later
        }
    # 按夏普排名
    ranked = sorted(summary.items(), key=lambda x: x[1].get("avg_sharpe", -999), reverse=True)
    for rank, (sid, _) in enumerate(ranked, 1):
        summary[sid]["rank"] = rank
    return summary


def _summarize_regime(results: Dict[str, List[Dict]]) -> Dict:
    """按市场状态汇总"""
    summary = {}
    for regime, entries in results.items():
        if not entries:
            summary[regime] = {"total": 0}
            continue
        # 每种策略在该市场状态下的表现
        by_strategy = {}
        for e in entries:
            sid = e["strategy"]
            if sid not in by_strategy:
                by_strategy[sid] = []
            by_strategy[sid].append(e)

        strategy_stats = {}
        for sid, s_entries in by_strategy.items():
            strategy_stats[sid] = {
                "count": len(s_entries),
                "avg_win_rate": round(np.mean([e["win_rate"] for e in s_entries]), 3),
                "avg_sharpe": round(np.mean([e["sharpe"] for e in s_entries]), 3),
                "avg_profit_factor": round(np.mean([e["profit_factor"] for e in s_entries]), 2),
            }

        summary[regime] = {
            "total": len(entries),
            "avg_win_rate": round(np.mean([e["win_rate"] for e in entries]), 3),
            "avg_sharpe": round(np.mean([e["sharpe"] for e in entries]), 3),
            "best_strategies": sorted(strategy_stats.items(),
                                      key=lambda x: x[1]["avg_sharpe"], reverse=True)[:3],
        }
    return summary


def _build_summary(strategy, regime, stocks) -> Dict:
    """构建摘要"""
    all_entries = [e for v in strategy.values() for e in v]

    if not all_entries:
        return {"total": 0}

    overall_wr = np.mean([e["win_rate"] for e in all_entries])
    overall_sr = np.mean([e["sharpe"] for e in all_entries])

    # 最佳/最差策略
    per_s = _summarize_strategy(strategy)
    best_s = max(per_s.items(), key=lambda x: x[1].get("avg_sharpe", -999))
    worst_s = min(per_s.items(), key=lambda x: x[1].get("avg_sharpe", 999) if x[1].get("total", 0) > 0 else 999)

    # 最佳/最差市场状态
    per_r = _summarize_regime(regime)
    best_r = max(per_r.items(), key=lambda x: x[1].get("avg_sharpe", -999) if x[1].get("total", 0) > 0 else -999)
    worst_r = min(per_r.items(), key=lambda x: x[1].get("avg_sharpe", 999) if x[1].get("total", 0) > 0 else 999)

    return {
        "total_backtests": len(all_entries),
        "overall_win_rate": round(overall_wr, 3),
        "overall_sharpe": round(overall_sr, 3),
        "best_strategy": {"id": best_s[0], "avg_sharpe": best_s[1].get("avg_sharpe", 0)},
        "worst_strategy": {"id": worst_s[0], "avg_sharpe": worst_s[1].get("avg_sharpe", 0)},
        "best_regime": {"regime": best_r[0], "avg_sharpe": best_r[1].get("avg_sharpe", 0)},
        "worst_regime": {"regime": worst_r[0], "avg_sharpe": worst_r[1].get("avg_sharpe", 0)},
    }


def print_report(results: Dict):
    """打印可读报告"""
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    meta = results["meta"]
    summary = results["summary"]
    per_s = results["per_strategy"]
    per_r = results["per_regime"]

    print("\n" + "=" * 70)
    print(f"  全量历史回测验证报告")
    print(f"  {meta['date']} | {meta['symbols']}只标的 × {len(per_s)}策略 = {meta['total_backtests']}次回测")
    print("=" * 70)

    print(f"\n   总览: 综合胜率 {summary['overall_win_rate']:.1%} | 综合夏普 {summary['overall_sharpe']:.3f}")
    print(f"   最佳策略: {summary['best_strategy']['id']} (夏普 {summary['best_strategy']['avg_sharpe']:.3f})")
    print(f"    最差策略: {summary['worst_strategy']['id']} (夏普 {summary['worst_strategy']['avg_sharpe']:.3f})")
    print(f"   最佳市态: {summary['best_regime']['regime']} (夏普 {summary['best_regime']['avg_sharpe']:.3f})")
    print(f"   最差市态: {summary['worst_regime']['regime']} (夏普 {summary['worst_regime']['avg_sharpe']:.3f})")

    print(f"\n  -- 策略排名 --")
    ranked = sorted(per_s.items(), key=lambda x: x[1].get("avg_sharpe", -999), reverse=True)
    print(f"  {'排名':<4} {'策略':<25} {'样本':<6} {'胜率':<8} {'夏普':<8} {'盈亏比':<8} {'回撤':<8}")
    print(f"  {'-'*65}")
    for sid, stats in ranked:
        if stats["total"] > 0:
            print(f"  {stats['rank']:<4} {sid:<25} {stats['total']:<6} "
                  f"{stats['avg_win_rate']:<8.1%} {stats['avg_sharpe']:<8.3f} "
                  f"{stats['avg_profit_factor']:<8.2f} {stats['avg_max_dd']:<8.1f}%")

    print(f"\n  -- 按市场状态 --")
    for regime in ["strong_bull", "weak_bull", "range_bound", "weak_bear", "strong_bear"]:
        info = per_r.get(regime, {})
        if info.get("total", 0) > 0:
            best = info.get("best_strategies", [])
            best_str = best[0][0] if best else "N/A"
            print(f"  {regime:<15} {info['total']:>5}样本  "
                  f"胜率{info['avg_win_rate']:.1%}  夏普{info['avg_sharpe']:.3f}  "
                  f"最佳: {best_str}")

    print(f"\n  -- Top 10 标的 (按最佳策略夏普) --")
    for i, s in enumerate(results["top_stocks"][:10]):
        print(f"  {i+1:>2}. {s['name']:<8} ({s['symbol']}) "
              f"最佳={s['best_strategy']:<22} 夏普={s['best_sharpe']:.3f} "
              f"市态={s['regime']}")


def main():
    parser = argparse.ArgumentParser(description="全量历史回测验证")
    parser.add_argument("--quick", action="store_true", help="快速模式 (10只, 2年)")
    parser.add_argument("--pool", type=str, default="all_industries", help="股票池")
    parser.add_argument("--years", type=int, default=3, help="回测年数")
    parser.add_argument("--max", type=int, default=0, help="最多N只")
    parser.add_argument("--output", type=str, default="", help="输出JSON文件")
    args = parser.parse_args()

    if args.quick:
        args.max = min(args.max or 10, 10)
        args.years = min(args.years, 2)

    from scripts.shared import load_watchlist
    symbols = load_watchlist(args.pool)
    print(f"股票池: {args.pool} ({len(symbols)}只)")

    results = run_validation(symbols, args.years, args.max)
    print_report(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str),
                              encoding="utf-8")
        print(f"\nJSON报告已保存: {args.output}")


if __name__ == "__main__":
    main()
