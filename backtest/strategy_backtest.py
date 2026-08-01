"""
策略级别回测 — 每策略在全市场精选股票池上回测，汇总策略健康度

股票池优先级:
  1. reports/deep_analysis_top100.json (全市场AI筛选Top结果)
  2. config/symbols.yaml (全行业60+代表股票)
  3. 默认蓝筹15只 (兜底)

输出: reports/strategy_backtest.json
用法: python -m backtest.strategy_backtest [--days 1000]
"""

import argparse, asyncio, json, random, sys, time, os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()
from analysis.recommender import STRATEGY_BACKTESTERS
from analysis.optimized_strategies import OPTIMIZED_BACKTESTERS, OPTIMIZED_NAMES
from analysis.strategies_v3 import V3_BACKTESTERS, V3_NAMES

# 合并: 保留v1全量 + 动量突破v2(唯一通过验证) + v3替换过拟合v2
OPTIMIZED_BACKTESTERS = {
    "momentum_breakout_v2": OPTIMIZED_BACKTESTERS["momentum_breakout_v2"],
}
OPTIMIZED_BACKTESTERS.update(V3_BACKTESTERS)

OPTIMIZED_NAMES = {
    "momentum_breakout_v2": "动量突破v2",
}
OPTIMIZED_NAMES.update(V3_NAMES)

# ── 策略名映射 ──
STRATEGY_NAMES = {
    "dual_ma_trend": "双均线趋势",
    "macd_trend": "MACD趋势",
    "bollinger_reversal": "布林带回归",
    "rsi_mean_reversion": "RSI均值回归",
    "momentum_breakout": "动量突破",
    "limit_up_chase": "涨停追击",
    "multi_factor": "多因子选股",
    "low_volatility": "低波动异象",
    "northbound_follow": "北向跟随",
    "turtle_trend": "海龟趋势",
}
STRATEGY_NAMES.update(OPTIMIZED_NAMES)

# 合并所有策略: v1 + v2
ALL_BACKTESTERS = {}
ALL_BACKTESTERS.update(STRATEGY_BACKTESTERS)
ALL_BACKTESTERS.update(OPTIMIZED_BACKTESTERS)

# ── 兜底股票池 ──
FALLBACK_STOCKS = [
    "sh.600519", "sz.000858", "sh.601318", "sh.600036", "sz.300750",
    "sz.002594", "sh.600900", "sh.600276", "sh.600031", "sz.002415",
    "sh.601899", "sz.000333", "sh.601088", "sh.600009", "sz.002714",
]


def _load_stock_universe(max_stocks: int = 0) -> List[str]:
    """加载股票池: 优先全市场分析结果 → symbols.yaml → 兜底蓝筹"""
    root = Path(__file__).parent.parent

    # ── Priority 1: 全市场AI分析Top结果 (5884→100) ──
    analysis_path = root / "reports" / "deep_analysis_top100.json"
    if analysis_path.exists():
        try:
            with open(analysis_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("results", [])
            stocks = []
            seen: Set[str] = set()
            for r in results:
                code = str(r.get("code", ""))
                if not code or code in seen:
                    continue
                seen.add(code)
                if code.startswith("6"):
                    stocks.append(f"sh.{code}")
                elif code.startswith(("0", "3")):
                    stocks.append(f"sz.{code}")
                elif code.startswith(("4", "8", "9")):
                    stocks.append(f"bj.{code}")
            if len(stocks) >= 10:
                logger.info(f"从全市场分析结果加载 {len(stocks)} 只股票 (deep_analysis_top100.json)")
                return stocks[:max_stocks] if max_stocks else stocks
        except Exception as e:
            logger.warning(f"加载分析结果失败: {e}")

    # ── Priority 2: symbols.yaml 全行业代表池 ──
    symbols_path = root / "config" / "symbols.yaml"
    if symbols_path.exists():
        try:
            with open(symbols_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            stocks = []
            seen = set()
            watchlist = cfg.get("watchlist", {})
            for pool_name, pool_data in watchlist.items():
                symbols = pool_data if isinstance(pool_data, list) else pool_data.get("symbols", [])
                for sym in symbols:
                    if sym not in seen:
                        seen.add(sym)
                        stocks.append(sym)
            if len(stocks) >= 10:
                logger.info(f"从 symbols.yaml 加载 {len(stocks)} 只股票 ({len(watchlist)} 个池)")
                return stocks[:max_stocks] if max_stocks else stocks
        except Exception as e:
            logger.warning(f"加载 symbols.yaml 失败: {e}")

    # ── Priority 3: 兜底 ──
    logger.info(f"使用兜底股票池: {len(FALLBACK_STOCKS)} 只")
    return FALLBACK_STOCKS[:max_stocks] if max_stocks else FALLBACK_STOCKS


async def run_strategy_backtests(
    stocks: List[str] = None,
    lookback_days: int = 1000,
) -> Dict[str, Any]:
    """对每个策略在每个股票上回测, 汇总结果"""
    from data.router import get_data_router
    from data.providers.base import DataFrequency, DataRequest

    if stocks is None:
        stocks = _load_stock_universe()

    router = get_data_router()
    today = date.today()
    total_stocks = len(stocks)
    strategy_results: Dict[str, Dict] = {}

    for si, (sid, bt_func) in enumerate(ALL_BACKTESTERS.items()):
        sname = STRATEGY_NAMES.get(sid, sid)
        all_trades = []
        stock_details = []
        total_wins = 0
        total_trades_count = 0
        success_count = 0

        logger.info(f"[{si+1}/{len(ALL_BACKTESTERS)}] {sname}: 回测 {total_stocks} 只...")

        for sym in stocks:
            try:
                req = DataRequest(sym, today - timedelta(days=lookback_days), today, DataFrequency.DAILY)
                result = await router.get_daily_kline(req)
                df = result.data
                if df.empty or len(df) < 60:
                    continue

                bt = bt_func(df)
                n_trades = bt.get("signals", 0)
                if n_trades < 3:
                    continue

                success_count += 1
                wr = bt.get("win_rate", 0)
                ev = bt.get("expected_value", 0)
                sr = bt.get("sharpe", 0)
                dd = bt.get("max_dd", 0)
                pf = bt.get("profit_factor", 0)
                avg_w = bt.get("avg_win", 0)
                avg_l = bt.get("avg_loss", 0)

                n_wins = int(n_trades * wr)
                n_losses = n_trades - n_wins
                all_trades.extend([avg_w] * n_wins + [-avg_l] * n_losses)
                total_wins += n_wins
                total_trades_count += n_trades

                stock_details.append({
                    "symbol": sym, "trades": n_trades,
                    "win_rate": round(wr, 3), "expected_value": round(ev, 2),
                    "sharpe": round(sr, 2), "max_dd": round(dd, 2),
                    "profit_factor": round(pf, 2),
                })

            except Exception as e:
                logger.debug(f"  {sym}: {e}")
                continue

        # 汇总
        n = total_trades_count
        if n > 0:
            arr = np.array(all_trades)
            avg_return = float(np.mean(arr))
            win_rate_agg = total_wins / n
            total_return = float(np.sum(arr))
            sharpe = float(np.mean(arr) / (np.std(arr) + 1e-9) * np.sqrt(n))
            cumulative = np.cumsum(arr)
            peak = np.maximum.accumulate(cumulative)
            max_dd = float(np.min(cumulative - peak))
            profit_factor = abs(float(np.sum(arr[arr > 0])) / (np.sum(arr[arr < 0]) + 1e-9))
        else:
            avg_return = win_rate_agg = total_return = sharpe = max_dd = profit_factor = 0

        strategy_results[sid] = {
            "name": sname,
            "stocks_tested": success_count,
            "total_trades": n,
            "avg_return_per_trade": round(avg_return, 2),
            "win_rate": round(win_rate_agg, 3),
            "total_return_pct": round(total_return, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "profit_factor": round(profit_factor, 2),
            "stock_details": stock_details,
            "grade": _grade(win_rate_agg, sharpe, max_dd, profit_factor),
        }

        logger.info(f"  => {sname}: {success_count}/{total_stocks}只 "
                    f"| {n}笔 | 胜率{win_rate_agg:.0%} | 夏普{sharpe:.1f} | {strategy_results[sid]['grade']}")

    return {
        "generated_at": datetime.now().isoformat(),
        "lookback_days": lookback_days,
        "stocks_total": total_stocks,
        "stocks_source": "deep_analysis" if (Path(__file__).parent.parent / "reports" / "deep_analysis_top100.json").exists() else "symbols_yaml",
        "results": strategy_results,
    }


async def _overfitting_check(
    stocks: List[str],
    backtesters: Dict,
    names: Dict,
    lookback_days: int,
) -> Dict:
    """股票池交叉验证: 60/20/20 分组, 测策略泛化能力。

    核心逻辑: 策略在A组股票上的表现应能泛化到B组。
    如果训练组夏普远高于验证组 → 策略只在见过的股票上有效 → 过拟合。
    """
    import random
    from data.router import get_data_router
    from data.providers.base import DataFrequency, DataRequest

    router = get_data_router()
    today = date.today()
    n_strats = len(backtesters)

    # 固定种子保证可复现
    rng = random.Random(42)
    shuffled = list(stocks)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_stocks = shuffled[:train_end]
    val_stocks = shuffled[train_end:val_end]
    test_stocks = shuffled[val_end:]

    logger.info(f"  交叉验证分组: train={len(train_stocks)} val={len(val_stocks)} test={len(test_stocks)}")

    results = {}

    for sid, bt_func in backtesters.items():
        sname = names.get(sid, sid)

        async def _run_on_group(group_stocks):
            """在股票组上运行策略, 聚合所有交易"""
            all_trades = []
            tested = 0
            for sym in group_stocks:
                try:
                    req = DataRequest(sym, today - timedelta(days=lookback_days), today, DataFrequency.DAILY)
                    r = await router.get_daily_kline(req)
                    df = r.data
                    if df.empty or len(df) < 60:
                        continue
                    bt = bt_func(df)
                    n = bt.get("signals", 0)
                    if n >= 2:
                        tested += 1
                        # 用 avg_win/avg_loss 重建交易序列
                        wr = bt["win_rate"]
                        aw = bt["avg_win"]
                        al = bt["avg_loss"]
                        n_wins = int(n * wr)
                        all_trades.extend([aw] * n_wins + [-al] * (n - n_wins))
                except Exception:
                    continue
            return all_trades, tested

        train_trades, train_n = await _run_on_group(train_stocks)
        val_trades, val_n = await _run_on_group(val_stocks)
        test_trades, test_n = await _run_on_group(test_stocks)

        if len(train_trades) < 10 or len(val_trades) < 5:
            results[sid] = {"overfit_score": 0, "train_sr": 0, "val_sr": 0,
                           "sr_decay": 0, "deflated_sr": 0, "risk": "unknown",
                           "n_train": train_n, "n_val": val_n, "n_test": test_n}
            continue

        # 计算各组的年化夏普
        def _group_sharpe(trades):
            arr = np.array(trades)
            if len(arr) < 3:
                return 0.0
            daily = arr / 100  # % → 小数
            return float(np.mean(daily) / (np.std(daily) + 1e-10) * np.sqrt(252))

        train_sr = _group_sharpe(train_trades)
        val_sr = _group_sharpe(val_trades)
        test_sr = _group_sharpe(test_trades)

        # SR 衰减: (验证 - 训练) / |训练|
        sr_decay = (val_sr - train_sr) / (abs(train_sr) + 1e-10)

        # Deflated Sharpe (Harvey & Liu 2015)
        total_val_trades = len(val_trades)
        if val_sr > 0:
            expected_max = np.sqrt(2 * np.log(max(n_strats, 2)))
            se_sr = np.sqrt((1 + 0.5 * val_sr**2) / max(total_val_trades, 1))
            dsr = (val_sr - expected_max) / (se_sr + 1e-10)
        else:
            dsr = 0.0

        # 过拟合判定: valSR为正=可用, valSR>1且衰减<30%=稳健
        of_score = abs(sr_decay) + (0.5 if dsr < 0 else 0)
        if val_sr > 1.0 and sr_decay > -0.3:
            risk = "low"
        elif val_sr > 0:
            risk = "medium"
        else:
            risk = "high"

        results[sid] = {
            "overfit_score": round(of_score, 3),
            "train_sr": round(train_sr, 2),
            "val_sr": round(val_sr, 2),
            "test_sr": round(test_sr, 2),
            "sr_decay": round(float(sr_decay), 2),
            "deflated_sr": round(float(dsr), 2),
            "risk": risk,
            "n_train": train_n,
            "n_val": val_n,
            "n_test": test_n,
        }

        logger.info(f"  OF {sname}: trainSR={train_sr:+.1f} valSR={val_sr:+.1f} "
                    f"decay={sr_decay:+.1%} dsr={dsr:+.1f} stocks={train_n}/{val_n}/{test_n} risk={risk}")

    return results


def _grade(win_rate: float, sharpe: float, max_dd: float, pf: float) -> str:
    score = 0
    if win_rate > 0.55: score += 1
    if win_rate > 0.60: score += 1
    if sharpe > 0.5: score += 1
    if sharpe > 1.0: score += 1
    if max_dd > -10: score += 1
    if max_dd > -5: score += 1
    if pf > 1.5: score += 1
    if pf > 2.0: score += 1

    if score >= 7: return "exc"
    if score >= 5: return "good"
    if score >= 3: return "ok"
    return "weak"


async def main():
    parser = argparse.ArgumentParser(description="策略级别回测 — 全市场股票池")
    parser.add_argument("--days", type=int, default=1000, help="回看天数 (默认1000)")
    parser.add_argument("--max-stocks", type=int, default=0, help="限制股票数 (0=全部)")
    args = parser.parse_args()

    stocks = _load_stock_universe(max_stocks=args.max_stocks)

    logger.info(f"策略回测: {len(stocks)}只股票 x {len(ALL_BACKTESTERS)}个策略 (v1+v2) | 回看{args.days}天")
    t0 = time.time()

    result = await run_strategy_backtests(stocks=stocks, lookback_days=args.days)

    # ═══════════════════════════════════════════════════════════════
    # 过拟合验证: 60/20/20 时间分割 + Deflated Sharpe
    # ═══════════════════════════════════════════════════════════════
    logger.info("运行过拟合验证 (60/20/20 时间分割)...")
    of_result = await _overfitting_check(
        stocks=stocks,  # 全部股票池交叉验证
        backtesters=ALL_BACKTESTERS,
        names=STRATEGY_NAMES,
        lookback_days=args.days,
    )
    # 将过拟合分数合并到每个策略结果
    for sid, of_info in of_result.items():
        if sid in result["results"]:
            result["results"][sid]["overfit_score"] = of_info.get("overfit_score", 0)
            result["results"][sid]["train_sr"] = of_info.get("train_sr", 0)
            result["results"][sid]["val_sr"] = of_info.get("val_sr", 0)
            result["results"][sid]["sr_decay"] = of_info.get("sr_decay", 0)
            result["results"][sid]["deflated_sr"] = of_info.get("deflated_sr", 0)
            result["results"][sid]["overfit_risk"] = of_info.get("risk", "unknown")
            # 过拟合严重的降级
            old_grade = result["results"][sid]["grade"]
            if of_info.get("risk") == "high" and old_grade in ("ok", "good", "exc"):
                result["results"][sid]["grade"] = "weak"
    result["overfitting"] = {
        "method": "cross-sectional CV (60/20/20 stock split) + Deflated Sharpe",
        "stocks_used": len(stocks[:min(99, len(stocks))]),
        "results": of_result,
    }

    # 保存
    outdir = Path(__file__).parent.parent / "reports"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "strategy_backtest.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {outpath}")

    # 排名
    ranked = sorted(result["results"].items(), key=lambda x: x[1]["sharpe"], reverse=True)
    grade_map = {"exc": "A", "good": "B", "ok": "C", "weak": "D"}
    print(f"\n{'='*65}")
    print(f"策略回测排名 ({len(stocks)}只股票, {args.days}天)")
    print(f"{'='*65}")
    print(f"{'排名':<4} {'策略':<12} {'胜率':<6} {'夏普':<7} {'回撤':<7} {'PF':<6} {'评级':<4}")
    print(f"{'-'*65}")
    for i, (sid, r) in enumerate(ranked, 1):
        print(f"{i:<4} {r['name']:<12} {r['win_rate']:.0%}   "
              f"{r['sharpe']:+.1f}   {r['max_drawdown']:+.1f}% "
              f"{r['profit_factor']:.1f}  {grade_map.get(r['grade'], '?')}")
    print(f"{'='*65}")

    elapsed = time.time() - t0
    logger.info(f"完成, 耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
