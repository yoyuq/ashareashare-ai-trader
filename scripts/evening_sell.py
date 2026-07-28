#!/usr/bin/env python3
"""
盘后自动卖出检查脚本 — A股模拟交易 v2.12

检查维度:
  1. 止损/止盈触发 (基于当日收盘价)
  2. 策略失效 (胜率<25% 或 今日推荐SELL)
  3. 浮亏超限 (市场状态自适应: 熊市-5%, 强熊-3%, 震荡-8%)
  4. 🆕 实时价格 fallback (分析数据缺失时自动用腾讯行情)

使用:
    python scripts/evening_sell.py                  # 正常执行
    python scripts/evening_sell.py --dry-run        # 试运行
"""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from simulation.portfolio import PortfolioManager
from simulation.paper_trader import PaperTradingEngine

REPORT_DIR = Path(__file__).parent.parent / "reports"
from scripts.shared import (NAME_MAP, resolve_name, get_best_price,
    refresh_fallback_prices_from_tencent, calc_trailing_stop)

# 🆕 v2.12: 市场状态 → 浮亏硬止损阈值
REGIME_MAX_LOSS = {
    "strong_bull":  -0.10,   # 牛市容忍更大回撤
    "weak_bull":    -0.08,
    "range_bound":  -0.08,
    "weak_bear":    -0.05,   # 熊市收紧止损
    "strong_bear":  -0.03,   # 强熊快跑
    "crisis":       0.00,    # 危机全清 (正数=触发所有)
}


def check_position(sym: str, pos, analysis_prices: dict,
                   recommendations: dict, regime: str = "range_bound",
                   rt_quotes: dict = None) -> List[Dict[str, Any]]:
    """
    检查单只持仓是否需要卖出。pos 是 PaperPosition dataclass。

    v2.12: 价格fallback + 市场状态自适应止损
    """
    name = resolve_name(sym)
    signals = []

    # 🆕 统一价格获取: 分析价 > 腾讯实时 > 兜底价
    close = get_best_price(sym, analysis_prices, rt_quotes)
    if close <= 0:
        close = getattr(pos, 'current_price', 0)
    if close <= 0:
        close = pos.avg_cost  # 最后兜底: 用成本价

    rec = recommendations.get(sym, {})

    entry = pos.avg_cost
    sl = getattr(pos, 'stop_loss', 0)
    tp = getattr(pos, 'take_profit', 0)

    pnl_pct = (close / entry - 1) * 100 if entry > 0 else 0

    # ── 维度1: 止损止盈触发 ──
    if sl > 0 and close <= sl:
        signals.append({
            "reason": "stop_loss",
            "detail": f"收盘{close:.2f}触发止损{sl:.2f}",
            "urgency": "high",
        })
    if tp > 0 and close >= tp:
        signals.append({
            "reason": "take_profit",
            "detail": f"收盘{close:.2f}触发止盈{tp:.2f}",
            "urgency": "high",
        })

    # ── 维度2: 策略失效 ──
    ap = analysis_prices.get(sym, {})
    if not isinstance(ap, dict):
        ap = {}
    best_wr = ap.get("best_strategy_win_rate", 0)
    best_name = ap.get("best_strategy_name", "")
    if best_wr < 0.25 and best_wr > 0:
        signals.append({
            "reason": "strategy_failure",
            "detail": f"{best_name}胜率仅{best_wr:.0%},策略不可靠",
            "urgency": "medium",
        })

    # 今日明确推荐卖出
    if rec.get("action") == "SELL" and rec.get("conviction", 0) >= 0.4:
        signals.append({
            "reason": "recommend_sell",
            "detail": f"今日建议SELL(确信度{rec['conviction']:.0%})",
            "urgency": "high",
        })

    # ── 维度3: 移动止盈 (🆕 v2.13: ATR trailing stop) ──
    ap_data = analysis_prices.get(sym, {})
    atr = ap_data.get("atr_14", 0) if isinstance(ap_data, dict) else 0
    if atr <= 0:
        atr = close * 0.03  # 默认3%波动率
    # 从持仓获取entry价格
    highest = getattr(pos, 'current_price', close)  # 近似: 用现价作为高点
    # 实际场景中highest_since_entry应存在PaperPosition中
    # 暂时用current_price和close的最大值
    peak = max(close, getattr(pos, 'current_price', close))

    should_trail, trail_level, trail_detail = calc_trailing_stop(
        pos.avg_cost, close, peak, atr, regime)
    if should_trail:
        signals.append({
            "reason": "trailing_stop",
            "detail": trail_detail,
            "urgency": "high",
        })

    # ── 维度4: 浮亏超限 (🆕 v2.12: 市场状态自适应) ──
    max_loss_pct = REGIME_MAX_LOSS.get(regime, -0.08)
    if max_loss_pct == 0.0:
        # crisis: 全仓卖出
        signals.append({
            "reason": "crisis_exit",
            "detail": f"市场危机状态, 清仓避险 (浮亏{pnl_pct:.1f}%)",
            "urgency": "critical",
        })
    elif pnl_pct < max_loss_pct * 100:
        signals.append({
            "reason": "max_loss",
            "detail": f"浮亏{pnl_pct:.1f}%超过{max_loss_pct*100:.0f}%硬止损线(市场:{regime})",
            "urgency": "critical",
        })

    return signals


async def run_evening_sell(dry_run: bool = False) -> Dict[str, Any]:
    """执行盘后卖出检查 (v2.12: Tencent fallback + 市场自适应)"""
    logger.info("=== 盘后卖出检查 v2.12 ===")
    logger.info(f"模式: {'试运行' if dry_run else '执行'}")

    # 1. 加载持仓
    manager = PortfolioManager()
    engine = PaperTradingEngine(manager)
    state = engine.state

    if not state.positions:
        logger.info("无持仓,无需检查")
        return {"status": "no_positions", "sells": 0}

    logger.info(f"当前持仓: {len(state.positions)}只 | 现金: {state.cash:,.2f}")

    # 2. 加载最新分析数据 + 市场状态
    today = date.today().isoformat()
    datapath = REPORT_DIR / f"data_{today}.json"
    analysis_prices = {}
    recommendations = {}
    regime = "range_bound"

    if datapath.exists():
        with open(datapath, "r", encoding="utf-8") as f:
            data = json.load(f)
        analysis_prices = data.get("analysis_prices", {})
        recommendations = data.get("stock_recommendations", {})
        regime_info = data.get("market_regime", {})
        regime = regime_info.get("regime", "range_bound")
        logger.info(f"加载分析数据: {len(analysis_prices)}只 | 市场: {regime}")
    else:
        logger.warning(f"无今日分析数据 ({datapath})")

    # 🆕 v2.12: 刷新兜底价格 + 获取腾讯实时行情 (价格fallback)
    await refresh_fallback_prices_from_tencent()
    rt_quotes = {}
    try:
        from data.providers.tencent_provider import TencentFinanceProvider
        tp = TencentFinanceProvider()
        position_symbols = list(state.positions.keys())
        rt_quotes = await tp.get_realtime_quotes(position_symbols)
        logger.info(f"腾讯实时行情: {len(rt_quotes)}只")
    except Exception:
        pass

    max_loss_pct = REGIME_MAX_LOSS.get(regime, -0.08)
    logger.info(f"市场状态: {regime} | 浮亏硬止损: {max_loss_pct*100:.0f}%")

    # 3. 逐只检查
    all_signals = {}
    for sym, pos in list(state.positions.items()):
        signals = check_position(sym, pos, analysis_prices, recommendations,
                                 regime=regime, rt_quotes=rt_quotes)
        if signals:
            all_signals[sym] = signals

    if not all_signals:
        logger.info("所有持仓正常,无需卖出")
        return {"status": "ok", "sells": 0, "checked": len(state.positions)}

    # 4. 执行卖出
    executed = []
    skipped = []

    for sym, signals in all_signals.items():
        pos = state.positions[sym]
        name = resolve_name(sym)
        qty = pos.quantity
        close = get_best_price(sym, analysis_prices, rt_quotes)
        if close <= 0:
            close = pos.current_price

        # 汇总信号
        reasons_text = "; ".join(s["detail"] for s in signals)
        is_critical = any(s["urgency"] == "critical" for s in signals)
        is_high = any(s["urgency"] == "high" for s in signals)

        # 确定卖出数量
        if is_critical:
            sell_qty = qty  # 全部清仓
        elif is_high:
            sell_qty = qty  # 全部卖出
        else:
            sell_qty = max(qty // 2, 100)  # 减半仓(至少100股)
            sell_qty = min(sell_qty, qty)

        if dry_run:
            logger.info(
                f"[DRY-RUN] {name}({sym}): 卖{sell_qty}股 @{close:.2f} "
                f"理由: {reasons_text}"
            )
            executed.append({
                "symbol": sym, "name": name, "quantity": sell_qty,
                "price": close, "reasons": reasons_text, "dry_run": True,
            })
            continue

        trade = engine.execute_sell(
            symbol=sym,
            price=close,
            quantity=sell_qty,
            reason=reasons_text,
        )

        if trade:
            pnl = (close - pos.avg_cost) * sell_qty
            executed.append({
                "symbol": sym, "name": name,
                "quantity": sell_qty, "price": close,
                "amount": close * sell_qty,
                "pnl": round(pnl, 2),
                "reasons": reasons_text,
            })
        else:
            skipped.append({"symbol": sym, "name": name, "reason": "卖出执行失败"})

    # 5. 汇总
    state = engine.state
    print("\n" + "=" * 60)
    print(f"Evening Sell Summary | {today}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'} | "
          f"Checked: {len(state.positions)} | Sold: {len(executed)}")
    print("-" * 60)

    if executed:
        for e in executed:
            if e.get("dry_run"):
                print(f"  [DRY-RUN] {e['name']}({e['symbol']}) "
                      f"{e['quantity']}sh @{e['price']:.2f}")
                print(f"           理由: {e['reasons']}")
            else:
                print(f"  [SOLD] {e['name']}({e['symbol']}) "
                      f"{e['quantity']}sh @{e['price']:.2f} "
                      f"金额={e['amount']:,.0f} 盈亏={e['pnl']:+,.0f}")
                print(f"         理由: {e['reasons']}")

    if skipped:
        for s in skipped:
            print(f"  [SKIP] {s['name']}({s['symbol']}): {s['reason']}")

    print(f"\nCash: {state.cash:,.2f} | Positions: {len(state.positions)} | "
          f"Total: {state.total_value:,.2f}")
    print("=" * 60)

    return {
        "status": "ok",
        "date": today,
        "dry_run": dry_run,
        "checked": len(state.positions),
        "signaled": len(all_signals),
        "executed": len(executed),
        "skipped": len(skipped),
    }


async def main():
    parser = argparse.ArgumentParser(description="A股模拟交易 -- 盘后卖出检查 v2.10")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式")
    args = parser.parse_args()
    await run_evening_sell(dry_run=args.dry_run)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
