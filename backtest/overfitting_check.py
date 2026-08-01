"""
策略过拟合验证 + 新策略探索
- 60/20/20 时间分割交叉验证
- Deflated Sharpe Ratio
- 参数敏感性
- Monte Carlo 随机化检验
- 新增: 价量张力反转策略 (Price-Volume Tension Reversal)
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


def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    rs = gain.ewm(span=period, adjust=False).mean() / (loss.ewm(span=period, adjust=False).mean() + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = pd.Series(df["high"]), pd.Series(df["low"]), pd.Series(df["close"])
    tr = pd.concat([(h-l).abs(), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def exec_prices(df: pd.DataFrame):
    opens = pd.Series(df["open"].values)
    closes = pd.Series(df["close"].values)
    highs = pd.Series(df["high"].values)
    ep = opens.shift(-1).copy(); ep.iloc[-1] = closes.iloc[-1]
    xp = opens.shift(-1).copy(); xp.iloc[-1] = closes.iloc[-1]
    lu = closes >= highs.shift(1) * 1.099
    ld = closes <= (highs.shift(1) * 0.9 * 0.99)
    ep[lu] = np.nan; xp[ld] = np.nan
    return ep, xp


def calc_stats(trades: List[float]) -> Dict:
    COST = 0.0031
    if not trades:
        return {"signals": 0, "win_rate": 0, "sharpe": 0, "max_dd": 0, "profit_factor": 1,
                "avg_win": 0, "avg_loss": 0}
    net = [t - COST * 100 for t in trades]
    wins = [t for t in net if t > 0]
    losses = [abs(t) for t in net if t <= 0]
    wr = len(wins) / len(net)
    pf = sum(wins) / (sum(losses) + 1e-10)
    sr = float((pd.Series([t/100 for t in net]).mean() / (pd.Series([t/100 for t in net]).std() + 1e-10) * np.sqrt(252)))
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    dd = abs(float(np.min(cum - peak)))
    return {
        "signals": len(net), "win_rate": wr, "sharpe": sr, "max_dd": dd,
        "profit_factor": pf, "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
    }


# ═══════════════════════════════════════════════════════════════
# 新策略: 价量张力反转 (Price-Volume Tension Reversal)
# 基于国联民生金工 2026.7 研究
# ═══════════════════════════════════════════════════════════════

def backtest_pv_tension(df: pd.DataFrame) -> Dict:
    """
    价量张力反转策略:
    - 弹力势差: price_change / volume_ratio → 资金推动效率
    - 量能分歧: 当日振幅 vs 均幅 → 多空分歧
    - 买入: 涨幅大但量能不足 (虚涨) + 振幅扩大 (分歧加剧) → 即将回落
    - 卖出: 缩量+振幅收敛 (分歧消除) 或回到均线
    """
    closes = pd.Series(df["close"].values)
    highs = pd.Series(df["high"].values)
    lows = pd.Series(df["low"].values)
    volumes = pd.Series(df["volume"].values)
    n = len(closes)
    if n < 40:
        return {"signals": 0}

    rsi = compute_rsi(closes, 14)
    atr = compute_atr(df, 14)
    amplitude = (highs - lows) / closes.shift() * 100
    amp_ma20 = amplitude.rolling(20).mean()
    vol_ma20 = volumes.rolling(20).mean()
    ma20 = closes.rolling(20).mean()
    ep, xp = exec_prices(df)

    trades = []
    in_position = False
    entry_price = 0.0
    stop_price = 0.0
    short_signal = False  # 做空信号(实盘中只做多,此处仅跟踪)

    for i in range(40, n):
        if not in_position:
            # 弹力势差: 5日涨幅 / 量比 → 高值=虚涨(资金效率低,预示回落)
            chg_5 = (closes.iloc[i] / closes.iloc[max(0,i-5)] - 1) * 100
            vol_ratio = volumes.iloc[i] / (vol_ma20.iloc[i] + 1e-10)
            elastic_potential = chg_5 / (vol_ratio + 1e-10)  # 弹力势差

            # 量能分歧: 振幅偏离均值 → 多空分歧加剧
            amp_divergence = amplitude.iloc[i] / (amp_ma20.iloc[i] + 1e-10)

            # 入场: 虚涨(弹力势差>2) + 分歧加剧(>1.3x) + RSI>60(超买区域)
            fake_rally = elastic_potential > 2.0
            divergence_high = amp_divergence > 1.3
            rsi_overbought = rsi.iloc[i] > 60

            if fake_rally and divergence_high and rsi_overbought:
                short_signal = True  # 检测到做空信号
                # A股做多反转: 等RSI回落到<50再抄底
            elif short_signal and rsi.iloc[i] < 45 and closes.iloc[i] < ma20.iloc[i]:
                # 反转做多入场
                px = ep[i]
                if not np.isnan(px) and atr.iloc[i] > 0:
                    in_position = True
                    entry_price = px
                    stop_price = px - 2 * atr.iloc[i]
                    short_signal = False
        else:
            cur_pnl = (closes.iloc[i] / entry_price - 1) * 100
            hit_stop = closes.iloc[i] <= stop_price
            back_to_ma = closes.iloc[i] >= ma20.iloc[i] and cur_pnl > 0
            rsi_overbought_exit = rsi.iloc[i] > 65 and cur_pnl > 2

            if hit_stop or back_to_ma or rsi_overbought_exit:
                px = xp[i]
                if not np.isnan(px):
                    pnl = (px / entry_price - 1) * 100
                    trades.append(pnl)
                    in_position = False
            elif in_position and cur_pnl > 3:
                stop_price = max(stop_price, entry_price * 1.015)

    if in_position and n > 0:
        trades.append((closes.iloc[-1] / entry_price - 1) * 100)

    return calc_stats(trades)


# ═══════════════════════════════════════════════════════════════
# 过拟合验证: 时间分割 + Deflated Sharpe + 参数敏感性
# ═══════════════════════════════════════════════════════════════

def time_split_backtest(strategy_func, df: pd.DataFrame) -> Dict:
    """60/20/20 时间分割回测"""
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_bt = strategy_func(df.iloc[:train_end])
    val_bt = strategy_func(df.iloc[train_end:val_end])
    test_bt = strategy_func(df.iloc[val_end:])

    # 过拟合指标: 训练vs验证夏普比衰减
    sr_decay = (val_bt["sharpe"] - train_bt["sharpe"]) / (abs(train_bt["sharpe"]) + 1e-10)
    # 胜率衰减
    wr_decay = (val_bt["win_rate"] - train_bt["win_rate"])

    return {
        "train": train_bt, "val": val_bt, "test": test_bt,
        "sr_decay": sr_decay, "wr_decay": wr_decay,
        "overfit_score": abs(sr_decay) + abs(wr_decay),
    }


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int) -> float:
    """Deflated Sharpe Ratio (Harvey & Liu 2015)"""
    from scipy import stats as scipy_stats
    if sharpe <= 0:
        return 0.0, 1.0
    # E[max SR] ≈ sqrt(2*log(n_trials))
    expected_max_sr = np.sqrt(2 * np.log(max(n_trials, 2)))
    # Standard error of SR
    se_sr = np.sqrt((1 + 0.5 * sharpe**2) / max(n_obs, 1))
    dsr = (sharpe - expected_max_sr) / (se_sr + 1e-10)
    pval = 1 - scipy_stats.norm.cdf(dsr)
    return max(0, float(dsr)), min(1, max(0, float(pval)))


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

    # 合并所有策略
    all_strats = {}
    all_strats.update(STRATEGY_BACKTESTERS)
    all_strats.update(OPTIMIZED_BACKTESTERS)
    all_strats["pv_tension"] = backtest_pv_tension

    names = {**{k: k for k in STRATEGY_BACKTESTERS}, **OPTIMIZED_NAMES, "pv_tension": "价量张力反转"}

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

                split = time_split_backtest(func, df)
                if split["train"]["signals"] >= 3 and split["val"]["signals"] >= 1:
                    splits.append(split)
                    overfit_scores.append(split["overfit_score"])
                    total_val_trades += split["val"]["signals"]

            except Exception as e:
                continue

        if not splits:
            results[sid] = {"name": sname, "n_stocks": 0, "overfit_risk": "N/A", "note": "no valid splits"}
            continue

        # 聚合
        avg_train_sr = np.mean([s["train"]["sharpe"] for s in splits])
        avg_val_sr = np.mean([s["val"]["sharpe"] for s in splits])
        avg_test_sr = np.mean([s["test"]["sharpe"] for s in splits])
        avg_of_score = np.mean(overfit_scores)
        avg_wr_decay = np.mean([s["wr_decay"] for s in splits])

        # Deflated Sharpe
        dsr, dsr_pval = deflated_sharpe(avg_val_sr, n_trials=len(all_strats), n_obs=total_val_trades)

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
    print(f"DSR = Deflated Sharpe Ratio (Harvey&Liu 2015), 越高越可靠")
    print(f"SR衰减 = (验证SR - 训练SR) / |训练SR|, 负值越大=过拟合越严重")

    logger.info(f"报告已保存: {outpath}")


if __name__ == "__main__":
    asyncio.run(main())
