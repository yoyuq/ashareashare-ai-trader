#!/usr/bin/env python
"""今日仓位变动执行脚本 — 2026-07-31"""
import sys, json, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from simulation.portfolio import PortfolioManager
from simulation.paper_trader import PaperTradingEngine

manager = PortfolioManager()
engine = PaperTradingEngine(manager)
state = engine.state

print("=== Pre-Execution ===")
print(f"Positions: {len(state.positions)} | Cash: {state.cash:,.2f} | Total: {state.total_value:,.2f}")
print()

# ── Trade Plan ──
trades = [
    {
        "sym": "sz.001318", "name": "阳光乳业", "qty": None,
        "reason": "强熊止损:中小创杀跌-4%,距止损线(10.54)仅1%,换手率5.78%异常放量"
    },
    {
        "sym": "sz.300492", "name": "华图山鼎", "qty": 50,
        "reason": "强熊减仓:创业板小票,锁定+4%利润,降低中小创风险敞口"
    },
    {
        "sym": "sz.300138", "name": "晨光生物", "qty": 100,
        "reason": "强熊减仓:创业板小票,锁定+4.5%利润,降低中小创风险敞口"
    },
]

# Get all live prices first
live_prices = {}
for trade in trades:
    sym = trade["sym"]
    code = sym.replace("sh.", "").replace("sz.", "").replace("bj.", "")
    prefix = "sh" if code.startswith("6") else "sz"
    tc = f"{prefix}{code}"
    try:
        resp = requests.get(
            f"https://qt.gtimg.cn/q={tc}", timeout=5,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"},
        )
        resp.encoding = "gbk"
        for line in resp.text.split("\n"):
            if "=" in line and "~" in line:
                fields = line.split("=", 1)[1].strip('"').split("~")
                if len(fields) > 3 and fields[2] == code:
                    live_prices[sym] = float(fields[3]) if fields[3] else 0
                    break
    except Exception as e:
        print(f"  Failed to get live price for {trade['name']}: {e}")

# Execute
for trade in trades:
    sym = trade["sym"]
    name = trade["name"]
    sell_qty = trade["qty"]
    reason = trade["reason"]

    pos = state.positions.get(sym)
    if not pos or pos.quantity <= 0:
        print(f"  SKIP {name}: no position")
        continue

    if sell_qty is None:
        sell_qty = pos.quantity

    price = live_prices.get(sym, pos.current_price)
    if price <= 0:
        price = pos.current_price

    result = engine.execute_sell(
        symbol=sym,
        quantity=sell_qty,
        price=price,
        exit_reason=reason,
    )

    if result:
        pnl = (price - pos.avg_cost) * sell_qty
        pnl_pct = (price / pos.avg_cost - 1) * 100
        fees = result.commission + result.stamp_duty
        print(f"[OK] SELL {name}({sym}): {sell_qty} shares @{price:.2f}")
        print(f"     PnL: {pnl:+,.2f} ({pnl_pct:+.2f}%) | Fees: {fees:.2f}")
        print(f"     Reason: {reason}")
        remaining = pos.quantity - sell_qty if hasattr(pos, 'quantity') else 0
        if remaining > 0:
            print(f"     Remaining: {remaining} shares")
        else:
            print(f"     ALL SOLD")
    else:
        print(f"[FAIL] {name}({sym}) - sell failed")
    print()

# Save and print final state
state = engine.state
manager.save()

print("=== Post-Execution ===")
active_positions = {k: v for k, v in state.positions.items() if v.quantity > 0}
print(f"Positions: {len(active_positions)} | Cash: {state.cash:,.2f} | Total: {state.total_value:,.2f}")
print()
print("Current Holdings:")
for sym, pos in sorted(active_positions.items()):
    pnl_pct = (pos.current_price / pos.avg_cost - 1) * 100 if pos.avg_cost > 0 else 0
    print(f"  {pos.name:8s} ({sym}) {pos.quantity:>5d} shares @{pos.avg_cost:.2f} -> {pos.current_price:.2f} ({pnl_pct:+.2f}%)")

# Also update current prices for remaining positions
print()
print("Updating MTM prices for remaining positions...")
for sym in active_positions:
    code = sym.replace("sh.", "").replace("sz.", "").replace("bj.", "")
    prefix = "sh" if code.startswith("6") else "sz"
    tc = f"{prefix}{code}"
    try:
        resp = requests.get(
            f"https://qt.gtimg.cn/q={tc}", timeout=5,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"},
        )
        resp.encoding = "gbk"
        for line in resp.text.split("\n"):
            if "=" in line and "~" in line:
                fields = line.split("=", 1)[1].strip('"').split("~")
                if len(fields) > 3 and fields[2] == code:
                    price = float(fields[3]) if fields[3] else 0
                    if price > 0:
                        active_positions[sym].current_price = price
                    break
    except Exception:
        pass

manager.save()
print("MTM prices updated and saved.")
print()
print("=== DONE ===")
