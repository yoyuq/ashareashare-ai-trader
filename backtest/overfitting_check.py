"""
策略过拟合验证 CLI — 60/20/20 时间分割交叉验证

口径统一 (v5.6 P1-11): 过拟合统计一律委托 `backtest.overfitting` (单源), 本脚本不再
自带 DSR / 成交价 / 成本实现。策略函数复用 `analysis` 注册表 (含价量张力反转 v2/v3),
涨跌停板块感知与费率由策略函数内部 (broker 单源) 处理。

流程: 对每只股票 60/20/20 按时间切分 → 训练/验证/测试三组分别回测 → 聚合 SR 衰减、
胜率衰减、Deflated Sharpe (Bailey & López de Prado) → 输出过拟合风险。
"""

import asyncio, json, sys, time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

from backtest.overfitting import deflated_sharpe_ratio


# ═══════════════════════════════════════════════════════════════
# 过拟合验证: 时间分割 + Deflated Sharpe
# ═══════════════════════════════════════════════════════════════

def time_split_backtest(strategy_func, df: pd.DataFrame, symbol: str = "") -> Dict:
    """60/20/20 时间分割回测 (T日信号/T+1开盘成交, 涨跌停板块感知由策略函数处理)"""
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_bt = strategy_func(df.iloc[:train_end], symbol=symbol)
    val_bt = strategy_func(df.iloc[train_end:val_end], symbol=symbol)
    test_bt = strategy_func(df.iloc[val_end:], symbol=symbol)

    # 过拟合指标: 训练vs验证夏普比衰减 (策略函数早期返回 {"signals":0} 无 sharpe, 用 .get 兜底)
    train_sr = train_bt.get("sharpe", 0.0)
    val_sr = val_bt.get("sharpe", 0.0)
    sr_decay = (val_sr - train_sr) / (abs(train_sr) + 1e-10)
    wr_decay = val_bt.get("win_rate", 0.0) - train_bt.get("win_rate", 0.0)

    return {
        "train": train_bt, "val": val_bt, "test": test_bt,
        "sr_decay": sr_decay, "wr_decay": wr_decay,
        "overfit_score": abs(sr_decay) + abs(wr_decay),
    }


async def main():
    from data.router import get_data_router
    from data.providers.base import DataFrequency, DataRequest
    from analysis.optimized_strategies import OPTIMIZED_BACKTESTERS, OPTIMIZED_NAMES
    from analysis.recommender import STRATEGY_BACKTESTERS

    # 加载股票池
    root = Path(__file__).parent.parent
    ap = root / "reports" / "deep_analysis_top100.json"
    stocks = []
    if ap.exists():
        with open(ap, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            code = str(r.get("code", ""))
            if code.startswith("6"): stocks.append(f"sh.{code}")
            elif code.startswith(("0","3")): stocks.append(f"sz.{code}")
    stocks = stocks[:30]  # 取30只代表性股票

    router = get_data_router()
    today = date.today()

    # 合并所有策略 (v1 + v2/v3, 含价量张力反转 pv_tension)
    all_strats: Dict = {}
    all_strats.update(STRATEGY_BACKTESTERS)
    all_strats.update(OPTIMIZED_BACKTESTERS)

    names = {**{k: k for k in STRATEGY_BACKTESTERS}, **OPTIMIZED_NAMES}

    logger.info(f"过拟合验证: {len(stocks)}只 x {len(all_strats)}策略")

    results = {}
    for sid, func in all_strats.items():
        sname = names.get(sid, sid)
        splits = []
        overfit_scores = []
        total_val_trades = 0

        for sym in stocks:
            try:
                req = DataRequest(sym, today - timedelta(days=1000), today, DataFrequency.DAILY)
                r = await router.get_daily_kline(req)
                df = r.data
                if df.empty or len(df) < 100:
                    continue

                split = time_split_backtest(func, df, symbol=sym)
                if split["train"].get("signals", 0) >= 3 and split["val"].get("signals", 0) >= 1:
                    splits.append(split)
                    overfit_scores.append(split["overfit_score"])
                    total_val_trades += split["val"].get("signals", 0)

            except Exception as e:
                continue

        if not splits:
            results[sid] = {"name": sname, "n_stocks": 0, "overfit_risk": "N/A", "note": "no valid splits"}
            continue

        # 聚合
        avg_train_sr = np.mean([s["train"].get("sharpe", 0.0) for s in splits])
        avg_val_sr = np.mean([s["val"].get("sharpe", 0.0) for s in splits])
        avg_test_sr = np.mean([s["test"].get("sharpe", 0.0) for s in splits])
        avg_of_score = np.mean(overfit_scores)
        avg_wr_decay = np.mean([s["wr_decay"] for s in splits])

        # Deflated Sharpe (Bailey & López de Prado, 单源) — 策略返回 sharpe 为年化, /√252 换回每期
        dsr, dsr_pval = deflated_sharpe_ratio(
            sharpe=avg_val_sr / np.sqrt(252),
            n_trials=len(all_strats),
            n_obs=total_val_trades,
            annualize=True,
        )

        # 风险判定
        if avg_of_score > 0.5 or avg_wr_decay < -0.15:
            risk = "high"
        elif avg_of_score > 0.3 or avg_wr_decay < -0.08:
            risk = "medium"
        else:
            risk = "low"

        results[sid] = {
            "name": sname,
            "n_stocks": len(splits),
            "train_sr": round(avg_train_sr, 2),
            "val_sr": round(avg_val_sr, 2),
            "test_sr": round(avg_test_sr, 2),
            "sr_decay": round((avg_val_sr - avg_train_sr) / (abs(avg_train_sr) + 1e-10), 2),
            "wr_decay": round(avg_wr_decay, 3),
            "deflated_sr": round(dsr, 2),
            "dsr_pval": round(dsr_pval, 3),
            "overfit_score": round(avg_of_score, 3),
            "risk": risk,
        }

    # 保存
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stocks": len(stocks),
        "split": "60/20/20",
        "results": results,
    }
    outpath = root / "reports" / "overfitting_report.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 打印
    risk_icon = {"low": "✅", "medium": "⚡", "high": "⚠️"}
    print(f"\n{'='*80}")
    print(f"过拟合验证报告 (30只股票, 60/20/20时间分割)")
    print(f"{'='*80}")
    print(f"{'策略':<16} {'训练SR':<8} {'验证SR':<8} {'测试SR':<8} {'SR衰减':<8} {'DSR':<7} {'风险':<6}")
    print(f"{'-'*80}")
    for sid, r in sorted(results.items(), key=lambda x: x[1].get("val_sr", -999), reverse=True):
        if r.get("n_stocks", 0) == 0:
            continue
        print(f"{r['name']:<16} {r['train_sr']:+.2f}    {r['val_sr']:+.2f}    {r['test_sr']:+.2f}    "
              f"{r['sr_decay']:+.2f}     {r['deflated_sr']:+.2f}   {risk_icon.get(r['risk'], '?')} {r['risk']}")
    print(f"{'='*80}")
    print(f"\n风险: ✅低 ⚡中 ⚠️高")
    print(f"DSR = Deflated Sharpe Ratio (Bailey & López de Prado 2014), 越高越可靠")
    print(f"SR衰减 = (验证SR - 训练SR) / |训练SR|, 负值越大=过拟合越严重")

    logger.info(f"报告已保存: {outpath}")


if __name__ == "__main__":
    asyncio.run(main())
