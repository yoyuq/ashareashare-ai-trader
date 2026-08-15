"""落点 B 端到端验证 — 真实 ChromaDB + 真实 Baostock + 真实 execute_sell。

验证 `simulation.daily_runner._apply_learned_sell` (低估值规则高估卖出) 在真实数据下是否真能触发:
1. `recall_verified_rules(1)` 从真实 ChromaDB 取回 verified 规则 (确认 low_pe_value 在库、applicable)。
2. 对每只真实持仓, 用真实 `data_router.get_daily_kline` 取 pe (确认列名 peTTM/pe_ttm + 取值)。
3. 用 `sell_signals_for_positions` (与 `_apply_learned_sell` 同款谓词) 算信号, 列出会卖/不卖。
4. `--execute`: 调 `_apply_learned_sell(engine, router, mtm_pct, gate=1)` 真实执行卖出 (改 portfolio.json)。

默认 report-only (只读, 不动持仓)。`--execute` 才真卖 (改真实 paper portfolio)。

用法:
    python scripts/verify_learned_sell_live.py            # 只读报告
    python scripts/verify_learned_sell_live.py --execute  # 真实卖出
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from timeutil import today_cn


async def _fetch_pe(data_router, sym):
    """取单票最新 pe_ttm。返回 (pe: float|None, 备注)。真实取数, 失败不伪造。"""
    from data.providers.base import DataFrequency, DataRequest

    _today = today_cn()
    try:
        req = DataRequest(sym, _today - timedelta(days=200), _today, DataFrequency.DAILY)
        r = await data_router.get_daily_kline(req)
        df = r.data
    except Exception as e:
        return None, f"取数失败: {e}"
    if df is None or df.empty:
        return None, "无数据"
    pe_col = "pe_ttm" if "pe_ttm" in df.columns else ("peTTM" if "peTTM" in df.columns else None)
    if pe_col is None:
        return None, f"无pe列 (cols={list(df.columns)[:8]})"
    pe = pd.to_numeric(df[pe_col], errors="coerce").iloc[-1]
    if not pd.notna(pe):
        return None, "pe=NaN"
    return float(pe), pe_col


async def report_only(engine, data_router) -> int:
    from agent.learning.knowledge_apply import recall_verified_rules, sell_signals_for_positions

    rules = recall_verified_rules(1)
    if not rules:
        logger.error("recall_verified_rules(1) 返回空 → 真实 ChromaDB 无 verified 规则, 落点 B 会 no-op")
        return 1

    applicable = [r for r in rules if r.get("applicable")]
    logger.info(f"取回 {len(rules)} 条 verified 规则 (applicable={len(applicable)}):")
    for r in rules:
        logger.info(f"  - {r['concept']}: template={r['template']} applicable={r['applicable']} "
                    f"params={r['params']}")
    if not applicable:
        logger.error("无 applicable 规则 (全时序型) → 落点 B no-op")
        return 1

    max_pe = float(applicable[0]["params"].get("max_pe", 15.0))
    thr = max_pe * 1.5
    logger.info(f"\n卖出阈值 pe > {thr:.1f} (max_pe={max_pe})")

    would_sell = []
    logger.info(f"\n{'持仓':<24} {'pe_ttm':>8} {'触发':>4}  备注/来源列")
    for sym, pos in engine.state.positions.items():
        pe, note = await _fetch_pe(data_router, sym)
        name = f"{pos.name}({sym})"
        if pe is None:
            logger.info(f"{name:<24} {'?':>8} {'-':>4}  {note}")
            continue
        _one = pd.DataFrame([{"code": sym.split(".")[-1], "pe_ttm": pe}])
        trigger = bool(sell_signals_for_positions(_one, rules).any())
        mark = "卖" if trigger else "-"
        if trigger:
            would_sell.append((sym, pos.name, pe))
        logger.info(f"{name:<24} {pe:>8.1f} {mark:>4}  ({note})")

    logger.info(f"\n会触发高估卖出: {len(would_sell)} 只")
    for sym, name, pe in would_sell:
        logger.info(f"  → {name}({sym}) pe={pe:.1f} > {thr:.1f}")
    return 0


async def run_execute(engine, data_router, mtm_pct) -> int:
    from simulation.daily_runner import _apply_learned_sell

    sold = await _apply_learned_sell(engine, data_router, mtm_pct, gate=1)
    logger.info(f"真实卖出 {len(sold)} 只: {[s['symbol'] for s in sold]}")
    for s in sold:
        logger.info(f"  → {s['name']}({s['symbol']}) @ {s['price']}")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="真实执行卖出 (改 portfolio.json)")
    args = ap.parse_args()

    from simulation.portfolio import PortfolioManager
    from simulation.paper_trader import PaperTradingEngine
    from data.router import get_data_router

    manager = PortfolioManager()
    engine = PaperTradingEngine(manager)
    data_router = get_data_router()

    logger.info(f"当前真实持仓 {len(engine.state.positions)} 只: "
                f"{[f'{p.name}({s})' for s, p in engine.state.positions.items()]}")

    if not args.execute:
        return await report_only(engine, data_router)

    from simulation.daily_runner import _tencent_quote

    mtm_pct = {}
    for sym in engine.state.positions:
        _px, _pct = _tencent_quote(sym)
        if _pct is not None:
            mtm_pct[sym] = _pct
    logger.info(f"mtm_pct (今日涨跌幅%): {mtm_pct}")

    rc = await run_execute(engine, data_router, mtm_pct)
    manager.save()
    logger.info("已保存 portfolio.json")
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
