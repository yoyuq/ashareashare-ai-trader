#!/usr/bin/env python3
"""
晚间总结脚本 — A股模拟交易

收盘后执行两项任务:
  1. --analyze: 运行分析流水线, 生成次日交易推荐
  2. --summarize: 持仓Mark-to-Market, 止盈止损检查, 记录每日快照

使用:
    # 仅运行分析 (生成推荐)
    python scripts/evening_summary.py --analyze

    # 仅运行总结 (评估持仓)
    python scripts/evening_summary.py --summarize

    # 完整流程 (分析 + 总结)
    python scripts/evening_summary.py --analyze --summarize

    # 无LLM模式 (快速规则引擎)
    python scripts/evening_summary.py --analyze --summarize --no-llm
"""

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# 项目内部导入
from simulation.portfolio import PortfolioManager
from simulation.paper_trader import PaperTradingEngine

REPORT_DIR = Path(__file__).parent.parent / "reports"
SIMULATION_DIR = Path(__file__).parent.parent / "simulation_data"


from scripts.shared import load_watchlist, NAME_MAP, resolve_name, get_fallback_price


async def run_analysis(pool: str = "default", no_llm: bool = False,
                       run_date: str = "") -> Dict[str, Any]:
    """
    运行盘后分析 — 复用现有 DailyPipeline
    增强: 将收盘价和策略信息保存到 pending_recommendations
    """
    today_str = run_date or date.today().isoformat()
    logger.info(f"═══ 盘后分析 (pool={pool}, date={today_str}, LLM={'禁用' if no_llm else '启用'}) ═══")

    from scripts.run_daily_analysis import DailyPipeline

    symbols = load_watchlist(pool)
    logger.info(f"标的数: {len(symbols)}")

    pipeline = DailyPipeline(
        symbols=symbols,
        enable_notify=False,
        force_no_llm=no_llm,
    )
    result = await pipeline.run()

    # 提取每只股票的分析数据 (含收盘价、RSI、趋势分等)
    analysis_data = result.get("analysis", {})

    # 构建增强的买入推荐 (含价格和策略信息)
    recs = result.get("stock_recommendations", {})
    buy_recs = []
    for sym, rec in recs.items():
        if isinstance(rec, dict) and rec.get("action") == "BUY":
            if rec.get("conviction", 0) >= 0.45:  # 阈值0.45 (低于规则引擎上限0.55以增加信号)
                # 从分析数据中提取价格和策略
                info = analysis_data.get(sym, {})
                indicators = info.get("indicators", {}) if isinstance(info, dict) else {}
                strategies = info.get("strategies", []) if isinstance(info, dict) else []

                close_price = indicators.get("close", 0)
                composite_score = indicators.get("composite_score", rec.get("score", 5) * 10)

                # 提取最佳策略
                best_strategy = {}
                if strategies:
                    best = max(strategies, key=lambda s: s.get("win_rate", 0))
                    best_strategy = {
                        "strategy_id": best.get("id", ""),
                        "strategy_name": best.get("name", ""),
                        "win_rate": best.get("win_rate", 0),
                        "sharpe": best.get("sharpe", 0),
                    }

                buy_recs.append({
                    "symbol": sym,
                    "action": "BUY",
                    "conviction": rec.get("conviction", 0),
                    "score": rec.get("score", 0),
                    "composite_score": composite_score,
                    "close_price": close_price,
                    "rsi": indicators.get("rsi_14", 50),
                    "trend_score": indicators.get("trend_score", 0),
                    "key_reasons": rec.get("key_reasons", []),
                    "risks": rec.get("risks", []),
                    "verdict_summary": rec.get("verdict_summary", ""),
                    **best_strategy,
                })

    # 按确信度×评分排序
    buy_recs.sort(key=lambda r: r["conviction"] * r["score"], reverse=True)

    # 保存到 portfolio
    manager = PortfolioManager()
    manager.state.pending_recommendations = buy_recs
    manager.state.last_analysis_date = today_str
    manager.save()

    logger.info(f"分析完成: {len(buy_recs)}条买入推荐 (含收盘价和策略信息)")
    for i, r in enumerate(buy_recs[:8]):
        logger.info(f"  [{i+1}] {r['symbol']}: score={r['score']:.1f} conf={r['conviction']:.0%} "
                   f"price={r['close_price']:.2f} {r.get('strategy_name', '')}")

    # 🆕 v2.14: 信号追踪 — 记录今日信号 + 回顾N日前信号
    _track_signals(buy_recs, result, today_str)

    return result


def _track_signals(today_recs: list, analysis_result: dict, today_str: str = ""):
    """N日信号回顾 + 今日信号记录"""
    signal_file = SIMULATION_DIR / "signal_tracker.json"
    today_str = today_str or date.today().isoformat()

    # 加载历史
    tracker = {"signals": {}, "reviews": []}
    if signal_file.exists():
        try:
            tracker = json.loads(signal_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # 回顾5天前的信号
    review_date = (date.fromisoformat(today_str) - timedelta(days=5)).isoformat()
    pending = tracker.get("signals", {}).get(review_date, [])
    if pending:
        analysis_data = analysis_result.get("analysis", {})
        prices_data = analysis_result.get("prices", {})
        reviewed = 0; correct = 0
        for sig in pending:
            sym = sig["symbol"]
            close_price = prices_data.get(sym, {}).get("close", 0) if isinstance(
                prices_data.get(sym, {}), dict) else 0
            if close_price > 0:
                actual_return = (close_price / sig["entry_price"] - 1) * 100
                is_correct = (sig["direction"] == "long" and actual_return > 0) or \
                            (sig["direction"] == "short" and actual_return < 0)
                reviewed += 1
                if is_correct: correct += 1
        if reviewed > 0:
            accuracy = correct / reviewed * 100
            logger.info(f"[信号回顾] {review_date}: {correct}/{reviewed} 正确 ({accuracy:.0f}%)")
            tracker.setdefault("reviews", []).append({
                "date": review_date, "signals": len(pending),
                "reviewed": reviewed, "correct": correct, "accuracy_pct": round(accuracy, 1),
            })

    # 记录今日信号
    today_signals = []
    for r in today_recs:
        today_signals.append({
            "symbol": r["symbol"],
            "direction": "long",
            "entry_price": r.get("close_price", 0),
            "conviction": r.get("conviction", 0),
            "score": r.get("score", 0),
            "strategy": r.get("strategy_name", ""),
        })
    if today_signals:
        tracker.setdefault("signals", {})[today_str] = today_signals

    # 保存
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    signal_file.write_text(json.dumps(tracker, ensure_ascii=False, indent=2), encoding="utf-8")


async def get_closing_prices(symbols: List[str]) -> Dict[str, float]:
    """获取收盘价"""
    prices = {}
    try:
        from data.router import get_data_router
        from data.providers.base import DataFrequency, DataRequest

        router = get_data_router()
        today = date.today()

        for sym in symbols:
            try:
                req = DataRequest(sym, today - timedelta(days=5),
                                today, DataFrequency.DAILY)
                result = await router.get_daily_kline(req)
                df = result.data
                if not df.empty:
                    prices[sym] = float(df["close"].iloc[-1])
            except Exception as e:
                logger.debug(f"获取{sym}收盘价失败: {e}")
    except Exception as e:
        logger.warning(f"数据获取失败: {e}")

    for sym in symbols:
        if sym not in prices or prices[sym] <= 0:
            fb_price, warning = get_fallback_price(sym)
            if fb_price is not None:
                prices[sym] = fb_price
            elif warning:
                logger.warning(warning)

    return prices


def load_sell_signals(today_str: str) -> Dict[str, Dict]:
    """从今日分析中加载卖出信号"""
    datapath = REPORT_DIR / f"data_{today_str}.json"
    if not datapath.exists():
        return {}

    try:
        with open(datapath, "r", encoding="utf-8") as f:
            data = json.load(f)
        recs = data.get("stock_recommendations", {})
        return {
            sym: rec for sym, rec in recs.items()
            if isinstance(rec, dict) and rec.get("action") in ("SELL",)
        }
    except Exception:
        return {}


async def run_summary(run_date: str = "") -> Dict[str, Any]:
    """
    运行收盘总结: Mark-to-Market + 止盈止损 + 快照
    """
    today_str = run_date or date.today().isoformat()
    logger.info(f"═══ 收盘总结 (date={today_str}) ═══")

    manager = PortfolioManager()
    engine = PaperTradingEngine(manager)
    state = engine.state

    positions = list(state.positions.keys())
    logger.info(f"当前持仓: {len(positions)}只 | 现金: ¥{state.cash:,.2f}")

    if not positions:
        # 无持仓, 仅记录快照
        logger.info("无持仓, 记录空快照")
        engine.record_daily_snapshot()
        return {
            "status": "ok", "positions": 0,
            "total_value": state.total_value,
            "daily_pnl": 0, "exits": [],
        }

    # 获取收盘价
    prices = await get_closing_prices(positions)
    logger.info(f"获取到 {len(prices)} 只股票收盘价")

    # Mark-to-Market
    mtm = engine.mark_to_market(prices)
    for sym, info in mtm.items():
        logger.info(f"  {info['name']}({sym}): {info['quantity']}股 "
                   f"@¥{info['current_price']:.2f} "
                   f"浮动盈亏¥{info['unrealized_pnl']:+,.2f} ({info['unrealized_pnl_pct']:+.1f}%)")

    # 检查止盈止损
    sell_signals = load_sell_signals(today_str)
    triggers = engine.check_exit_conditions(prices, sell_signals)

    exits = []
    for trigger in triggers:
        if trigger["should_sell"]:
            logger.info(f"触发退出: {trigger['name']}({trigger['symbol']}) "
                       f"— {trigger['reason']}: {trigger['detail']}")

            trade = engine.execute_sell(
                symbol=trigger["symbol"],
                price=trigger.get("price"),
                exit_reason=trigger["reason"],
            )
            if trade:
                exits.append({
                    "symbol": trigger["symbol"],
                    "name": trigger["name"],
                    "reason": trigger["reason"],
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                })

    # 记录快照
    snapshot = engine.record_daily_snapshot()

    # 🆕 v2.14: 交易绩效分析 (WinRateAnalyzer)
    trade_stats = {}
    if len(state.trade_history) >= 5:
        try:
            from analysis.winrate import WinRateAnalyzer
            import pandas as pd
            # 将交易历史转为DataFrame
            trades_df = pd.DataFrame([
                {
                    "date": t.date, "symbol": t.symbol, "side": t.side,
                    "price": t.price, "pnl": t.pnl or 0,
                    "pnl_pct": t.pnl_pct or 0,
                    "holding_days": 1,
                    "exit_reason": t.exit_reason or "",
                }
                for t in state.trade_history
                if t.side == "sell" and t.pnl is not None
            ])
            if not trades_df.empty:
                analyzer = WinRateAnalyzer()
                wr_report = analyzer.analyze(trades_df, pd.DataFrame())
                trade_stats = {
                    "total_trades": wr_report.total_signals,
                    "win_rate": round(wr_report.overall_win_rate * 100, 1),
                    "profit_factor": wr_report.profit_factor,
                    "expected_value": wr_report.expected_value,
                    "max_consecutive_loss": wr_report.max_consecutive_loss,
                    "scenario_breakdown": {
                        k: {
                            "win_rate": round(v["win_rate"] * 100, 1),
                            "trades": v["total"],
                        }
                        for k, v in wr_report.scenario_breakdown.items()
                        if v["total"] > 0
                    },
                }
                logger.info(f"交易绩效: 胜率{trade_stats['win_rate']:.0f}% "
                           f"盈亏比{trade_stats['profit_factor']:.1f} "
                           f"最大连亏{trade_stats['max_consecutive_loss']}次")
        except Exception as e:
            logger.debug(f"WinRateAnalyzer 跳过: {e}")

    # 保存交易日志
    trade_log_path = SIMULATION_DIR / f"paper_trades_{today_str}.json"
    engine.manager.save()

    # 汇总输出
    state = engine.state  # 刷新
    print("\n" + "=" * 60)
    print(f"Evening Summary -- {today_str}")
    print("-" * 60)
    print(f"Total: {state.total_value:,.2f} (Initial {state.initial_capital:,.0f})")
    print(f"Return: {state.total_return:+,.2f} ({state.total_return_pct:+.2f}%)")
    print(f"Cash: {state.cash:,.2f}")
    print(f"Positions: {state.position_value:,.2f} ({len(state.positions)} stocks)")
    print(f"Realized PnL: {state.total_realized_pnl:+,.2f}")
    print(f"Fees: {state.total_commission + state.total_stamp_duty:,.2f}")
    print(f"Today: {snapshot.daily_pnl:+,.2f} ({snapshot.daily_return_pct:+.2f}%)")
    print(f"WinRate: {state.win_count}W/{state.loss_count}L "
          f"({state.win_rate*100:.0f}%)")
    if trade_stats:
        print(f"Trade Stats: 盈亏比{trade_stats['profit_factor']:.1f}x "
              f"期望值{trade_stats['expected_value']:+.1f}% "
              f"最大连亏{trade_stats['max_consecutive_loss']}次")
        if trade_stats.get("scenario_breakdown"):
            for scenario, info in trade_stats["scenario_breakdown"].items():
                print(f"  {scenario}: 胜率{info['win_rate']:.0f}% ({info['trades']}笔)")

    if exits:
        print(f"\nExits triggered ({len(exits)}):")
        for e in exits:
            emoji = "[WIN]" if (e.get("pnl") or 0) > 0 else "[LOSS]"
            print(f"  {emoji} {e['name']}({e['symbol']}): {e['reason']} "
                  f"PnL={e.get('pnl', 0):+,.2f}")

    if state.positions:
        print(f"\nCurrent Positions:")
        for sym, pos in state.positions.items():
            pnl_sign = "+" if pos.unrealized_pnl > 0 else ""
            print(f"  {pos.name}({sym}): {pos.quantity}shares "
                  f"@{pos.current_price:.2f} "
                  f"PnL={pnl_sign}{pos.unrealized_pnl:,.2f} ({pos.unrealized_pnl_pct:+.1f}%) "
                  f"| SL={pos.stop_loss:.2f} TP={pos.take_profit:.2f}")

    print(f"\nData: {manager.filepath}")
    print("=" * 60)

    return {
        "status": "ok",
        "date": today_str,
        "total_value": round(state.total_value, 2),
        "cash": round(state.cash, 2),
        "position_value": round(state.position_value, 2),
        "daily_pnl": snapshot.daily_pnl,
        "daily_return_pct": snapshot.daily_return_pct,
        "cumulative_return_pct": snapshot.cumulative_return_pct,
        "positions": len(state.positions),
        "exits": len(exits),
        "exit_details": exits,
        "trade_stats": trade_stats,  # 🆕 v2.14: 交易绩效统计
    }


async def main():
    parser = argparse.ArgumentParser(description="A股模拟交易 — 晚间总结")
    parser.add_argument("--analyze", action="store_true",
                       help="运行盘后分析流水线")
    parser.add_argument("--summarize", action="store_true",
                       help="运行收盘总结(MTM+止盈止损+快照)")
    parser.add_argument("--pool", type=str, default="all_industries",
                       help="股票池: all_industries(全行业60只) / tech / default / broad")
    parser.add_argument("--no-llm", action="store_true",
                       help="禁用LLM,使用规则引擎")
    parser.add_argument("--date", type=str, default="",
                       help="指定日期 (默认今天)")
    args = parser.parse_args()

    if not args.analyze and not args.summarize:
        # 默认: 全跑
        args.analyze = True
        args.summarize = True

    if args.analyze:
        logger.info("Phase 1/2: 盘后分析...")
        await run_analysis(pool=args.pool, no_llm=args.no_llm, run_date=args.date)

    if args.summarize:
        logger.info("Phase 2/2: 收盘总结...")
        await run_summary(run_date=args.date)

    logger.info("✅ 晚间流程完成")


if __name__ == "__main__":
    asyncio.run(main())
