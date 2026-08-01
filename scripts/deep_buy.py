#!/usr/bin/env python3
"""使用全市场DeepSeek深度分析结果执行模拟买入"""
import sys, json, os
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv; load_dotenv()

from simulation.portfolio import PortfolioManager
from simulation.paper_trader import PaperTradingEngine

# 市场状态 → 仓位限制 (与 morning_buy.py 一致)
REGIME_POSITION_LIMITS = {
    "strong_bull":  {"max_positions": 10, "single_pct": 0.20, "label": "强牛-满仓进攻"},
    "weak_bull":    {"max_positions": 8,  "single_pct": 0.15, "label": "弱牛-积极"},
    "range_bound":  {"max_positions": 6,  "single_pct": 0.12, "label": "震荡-中性"},
    "weak_bear":    {"max_positions": 3,  "single_pct": 0.10, "label": "弱熊-防守"},
    "strong_bear":  {"max_positions": 1,  "single_pct": 0.05, "label": "强熊-空仓"},
    "crisis":       {"max_positions": 1,  "single_pct": 0.05, "label": "危机-最低参与"},
}

# 加载市场状态
today_data = Path(__file__).parent.parent / "reports" / f"data_{date.today().isoformat()}.json"
regime = "range_bound"
if today_data.exists():
    with open(today_data, "r", encoding="utf-8") as f:
        ad = json.load(f)
    regime = ad.get("market_regime", {}).get("regime", "range_bound")

limits = REGIME_POSITION_LIMITS.get(regime, REGIME_POSITION_LIMITS["range_bound"])
max_positions = limits["max_positions"]
single_pct = limits["single_pct"]
print(f"市场状态: {regime} → {limits['label']} | 最大{max_positions}只 | 单只{single_pct:.0%}")

# 加载深度分析结果
deep_file = Path(__file__).parent.parent / "reports" / "deep_analysis_top100.json"
with open(deep_file, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
buys = [r for r in results if r.get("action") == "BUY"]
print(f"深度分析: {len(results)}只 → BUY {len(buys)}只")

# 初始化
manager = PortfolioManager()
engine = PaperTradingEngine(manager)
state = engine.state

print(f"当前持仓: {len(state.positions)}只 | 现金: RMB{state.cash:,.2f} | 总资产: RMB{state.total_value:,.2f}")

# 卖出旧持仓 (先回笼资金)
if state.positions:
    old_positions = list(state.positions.items())
    for sym, pos in old_positions:
        # 用当前市价卖出
        engine.execute_sell(symbol=sym, quantity=None, price=pos.current_price, exit_reason="换仓:全市场AI精选")
    print(f"已卖出 {len(old_positions)} 只旧持仓 | 现金: RMB{state.cash:,.2f}")

# 执行买入 (严格按市场状态限制仓位)
print(f"最大持仓: {max_positions}只 | 单只上限: {single_pct:.0%}")
executed = []

for rec in buys:
    if len(state.positions) >= max_positions:
        break

    code = rec.get("code", "")
    name = rec.get("name", "")
    score = rec.get("final_score", rec.get("score", 0))
    conv = rec.get("conviction", 0.5)
    technical = rec.get("technical", "")
    fundamental = rec.get("fundamental", "")
    risk = rec.get("risk", "")

    if not code:
        print(f"  SKIP: no code for {name}")
        continue

    # 获取实时价格 — 转换代码格式
    price = 0
    try:
        import requests
        # 纯数字 → 带前缀的腾讯代码
        if code.startswith("6"): tc_code = "sh" + code
        elif code.startswith(("0","3")): tc_code = "sz" + code
        elif code.startswith("9"): tc_code = "bj" + code
        else: tc_code = code.replace("sh.","sh").replace("sz.","sz")
        resp = requests.get(f"https://qt.gtimg.cn/q={tc_code}", timeout=5,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
        resp.encoding = "gbk"
        for line in resp.text.split("\n"):
            if "=" in line and "~" in line:
                fields = line.split("=", 1)[1].strip('"').split("~")
                if len(fields) > 3 and fields[2] == code:
                    price = float(fields[3]) if fields[3] else 0
                    break
    except Exception:
        pass

    if price <= 0:
        print(f"  SKIP {name}({code}): no price")
        continue

    # 计算止损止盈
    stop_loss = round(price * 0.93, 2)
    take_profit = round(price * 1.12, 2)

    # 构建推荐
    enhanced = {
        "conviction": conv,
        "score": score / 10,
        "composite_score": score,
        "key_reasons": [technical, fundamental],
        "risks": [risk],
        "verdict_summary": f"DeepSeek: BUY({conv:.0%}) score={score}",
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }

    if code.startswith("6"): symbol = f"sh.{code}"
    elif code.startswith(("0","3")): symbol = f"sz.{code}"
    elif code.startswith("9"): symbol = f"bj.{code}"
    else: symbol = code
    trade = engine.execute_buy(
        symbol=symbol,
        name=name,
        price=price,
        recommendation=enhanced,
        max_position_pct=single_pct,
        max_positions=max_positions,
    )

    if trade:
        executed.append({"name": name, "code": code, "price": price, "qty": trade.quantity, "score": score})
        print(f"  [OK] {name}({code}) {trade.quantity}股 @RMB{price:.2f} score={score}")
    else:
        print(f"  [FAIL] {name}({code}) 买入失败(资金不足)")

# 保存
manager.save()
state = engine.state

print(f"\n{'='*50}")
print(f"执行完成: {len(executed)} 只新持仓")
print(f"现金: RMB{state.cash:,.2f} | 持仓: {len(state.positions)}只 | 总资产: RMB{state.total_value:,.2f}")
for e in executed:
    print(f"  {e['name']}({e['code']}) {e['qty']}股 @RMB{e['price']:.2f} score={e['score']}")
