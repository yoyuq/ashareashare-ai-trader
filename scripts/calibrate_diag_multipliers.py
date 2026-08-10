"""
诊断官仓位系数校准研究 (优化版: 选股预计算, 网格搜索只调系数)

目标: 找出"风险等级 → 仓位系数"的最优规则映射,
验证规则化的诊断官是否能超过 LLM 诊断官.

方法:
1. 预计算每天的 Top30 选股 (PreScreener 规则排序)
2. 预计算每天的风险等级 (规则判断, 1-5级)
3. 网格搜索最优系数表: 每个等级对应不同的仓位乘数
4. 和 LLM 诊断官 (diag2_top30) 对比
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd

from analysis.pre_screener import PreScreener

REPLAY_DIR = ROOT / "replay_data"


def compute_daily_features(df_day: pd.DataFrame) -> dict:
    """单日截面市场特征."""
    n = len(df_day)
    if n == 0:
        return {}
    pct = df_day["pctChg"].astype(float)
    pe = pd.to_numeric(df_day["peTTM"], errors="coerce").dropna()
    pb = pd.to_numeric(df_day["pbMRQ"], errors="coerce").dropna()
    amt = pd.to_numeric(df_day["amount"], errors="coerce").dropna()
    return {
        "n": n,
        "up_ratio": float((pct > 0).mean()),
        "limit_up_ratio": float((pct >= 9.5).mean()),
        "limit_down_ratio": float((pct <= -9.5).mean()),
        "med_pe": float(pe.median()) if len(pe) else 0.0,
        "med_pb": float(pb.median()) if len(pb) else 0.0,
        "total_amt": float(amt.sum()) / 1e8,
    }


def rule_risk_level(feat: dict, amt_ma5: float, pe_pct: float, p60_pos: float) -> int:
    """规则化风险等级 (1-5). 完全可解释, 无 LLM."""
    score = 3.0

    # 广度
    up_r = feat.get("up_ratio", 0.5)
    if up_r >= 0.7: score -= 0.8
    elif up_r >= 0.55: score -= 0.3
    elif up_r <= 0.3: score += 0.8
    elif up_r <= 0.4: score += 0.3

    # 涨跌停情绪
    lur = feat.get("limit_up_ratio", 0)
    ldr = feat.get("limit_down_ratio", 0)
    if ldr > 0.015: score += 1.0
    elif ldr > 0.005: score += 0.4
    if lur > 0.02: score += 0.5
    elif lur > 0.01: score += 0.2

    # 估值分位
    if pe_pct > 0.85: score += 0.6
    elif pe_pct > 0.7: score += 0.3
    elif pe_pct < 0.2: score -= 0.5
    elif pe_pct < 0.35: score -= 0.2

    # 60日强度 (用上涨占比近似)
    if p60_pos > 0.7: score += 0.4
    elif p60_pos > 0.55: score += 0.1
    elif p60_pos < 0.3: score -= 0.3
    elif p60_pos < 0.4: score -= 0.1

    # 成交额相对水平
    if amt_ma5 > 0:
        ratio = feat.get("total_amt", 0) / amt_ma5
        if ratio > 1.5: score += 0.3
        elif ratio < 0.6: score += 0.2

    return int(round(np.clip(score, 1, 5)))


def precompute(df: pd.DataFrame, window: list, top_n: int = 30) -> list:
    """预计算每天的选股和特征. 返回每天的 dict 列表."""
    print("预计算每日选股和市场特征...")
    screener = PreScreener()
    daily = []

    for i, d in enumerate(window):
        df_day = df[df["date"] == d].copy()
        feat = compute_daily_features(df_day)

        # PreScreener 选股
        try:
            screened = screener.screen(df_day, regime="range_bound", top_n=top_n * 3).df
            if "composite_score" in screened.columns:
                picks = screened.sort_values("composite_score", ascending=False).head(top_n)
            else:
                picks = screened.head(top_n)
            codes = picks["code"].astype(str).tolist()
            scores = picks["composite_score"].astype(float).tolist() if "composite_score" in picks.columns else [50.0]*len(picks)
        except Exception:
            codes = df_day.head(top_n)["code"].astype(str).tolist()
            scores = [50.0] * len(codes)

        daily.append({
            "date": d, "feat": feat,
            "picks": list(zip(codes, scores)),  # [(code, score), ...]
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(window)} 天完成")

    # 计算衍生指标 (PE分位, 成交额MA, 60日强度近似)
    pe_list = [d["feat"].get("med_pe", 0) for d in daily]
    amt_list = [d["feat"].get("total_amt", 0) for d in daily]

    for i in range(len(daily)):
        # PE 60日分位
        pe_win = pe_list[max(0, i-59):i+1]
        if len(pe_win) >= 10 and pe_win[-1] > 0:
            pe_pct = sum(1 for p in pe_win if p <= pe_win[-1]) / len(pe_win)
        else:
            pe_pct = 0.5
        daily[i]["pe_pct"] = pe_pct

        # 成交额 5日MA
        amt_win = amt_list[max(0, i-4):i+1]
        daily[i]["amt_ma5"] = np.mean(amt_win) if amt_win else 0

        # 60日强度近似: 用 20日前的截面 (如果没有 pct_60d 列)
        # 简化: 用最近20日上涨占比的均值
        up_list = [daily[j]["feat"].get("up_ratio", 0.5) for j in range(max(0, i-19), i+1)]
        daily[i]["p60_pos_est"] = np.mean(up_list) if up_list else 0.5

        # 风险等级
        daily[i]["risk_level"] = rule_risk_level(
            daily[i]["feat"], daily[i]["amt_ma5"],
            daily[i]["pe_pct"], daily[i]["p60_pos_est"]
        )

    return daily


def fast_simulate(daily: list, mult_table: dict, max_pos: int = 15,
                  stop_pct: float = 0.07, take_pct: float = 0.12,
                  df_full: pd.DataFrame = None) -> dict:
    """快速模拟: 用预计算的选股, 只调仓位系数.

    因为 PreScreener 输出的是每日排序后的 TopN, 不同系数只影响买入仓位大小,
    不影响选哪些股 (只要 max_pos >= top_n 就全买). 所以模拟非常快.
    """
    capital = 100000.0
    cash = capital
    positions = {}  # {code: {qty, stop, take}}
    equity = []

    for i, day in enumerate(daily):
        if i == 0:
            equity.append({"date": str(day["date"])[:10], "total": capital,
                           "positions": 0, "risk": day["risk_level"]})
            continue

        prev = daily[i - 1]
        d = day["date"]
        risk = day["risk_level"]
        mult = mult_table.get(risk, 1.0)

        # 获取当日行情 (用于开盘买入和收盘估值)
        df_d = df_full[df_full["date"] == d] if df_full is not None else None
        if df_d is None or len(df_d) == 0:
            continue

        # 先处理卖出 (用当日开盘/收盘价判断止损止盈)
        for code in list(positions.keys()):
            row = df_d[df_d["code"].astype(str) == code]
            if len(row) == 0:
                continue
            open_px = float(row["open"].iloc[0])
            close_px = float(row["close"].iloc[0])
            pos = positions[code]
            if open_px <= pos["stop"]:
                # 开盘跳空跌破止损, 用开盘价成交
                cash += pos["qty"] * open_px
                del positions[code]
            elif close_px >= pos["take"]:
                cash += pos["qty"] * close_px
                del positions[code]
            elif close_px <= pos["stop"]:
                cash += pos["qty"] * close_px
                del positions[code]

        # 再处理买入 (前一日的 picks, 今日开盘买)
        pct_per_stock_base = 1.0 / max_pos
        pct_per_stock = min(0.15, max(0.02, pct_per_stock_base * mult))

        for code, score in prev["picks"]:
            if len(positions) >= max_pos:
                break
            if code in positions:
                continue
            row = df_d[df_d["code"].astype(str) == code]
            if len(row) == 0:
                continue
            px = float(row["open"].iloc[0])
            if px <= 0:
                continue
            qty = int(cash * pct_per_stock / px / 100) * 100
            if qty < 100:
                continue
            cost = qty * px
            if cost > cash:
                continue
            cash -= cost
            positions[code] = {
                "qty": qty,
                "stop": px * (1 - stop_pct),
                "take": px * (1 + take_pct),
            }

        # 日终市值
        total = cash
        for code, pos in positions.items():
            row = df_d[df_d["code"].astype(str) == code]
            if len(row) > 0:
                total += pos["qty"] * float(row["close"].iloc[0])

        equity.append({"date": str(d)[:10], "total": round(total, 2),
                       "positions": len(positions), "risk": risk})

    # 指标
    vals = [e["total"] for e in equity]
    if len(vals) < 2:
        return {"return": 0, "max_dd": 0, "sharpe": 0, "win_rate": 0, "equity": equity}

    rets = [(vals[i] / vals[i-1] - 1) for i in range(1, len(vals))]
    total_ret = (vals[-1] / vals[0] - 1) * 100
    peak = vals[0]
    max_dd = 0
    for v in vals:
        if v > peak: peak = v
        dd = (v / peak - 1) * 100
        if dd < max_dd: max_dd = dd
    avg_r = np.mean(rets)
    std_r = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9
    sharpe = (avg_r / std_r * np.sqrt(252)) if std_r > 0 else 0
    win_rate = sum(1 for r in rets if r > 0) / len(rets) * 100

    # 风险等级分布
    rl_counts = {}
    for e in equity:
        rl = e.get("risk", 3)
        rl_counts[rl] = rl_counts.get(rl, 0) + 1

    return {
        "return": total_ret, "max_dd": max_dd, "sharpe": sharpe,
        "win_rate": win_rate, "risk_dist": rl_counts,
        "final_positions": len(positions), "equity": equity,
    }


def main():
    data_file = REPLAY_DIR / "daily_2020-06-01_2021-02-28_idx.parquet"
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        return

    print("加载数据...")
    df = pd.read_parquet(data_file)
    # 去掉指数 (只留个股)
    df = df[~df["code"].astype(str).str.match(r"^(sh\.000|sz\.399)")].copy()
    df["code"] = df["code"].astype(str).str.split(".").str[-1]

    all_dates = sorted(df["date"].unique())
    window = all_dates[-57:]  # 最后57个交易日, 同 A/B 窗口
    print(f"窗口: {str(window[0])[:10]} ~ {str(window[-1])[:10]} ({len(window)}天)")
    print()

    # 预计算 (只做一次)
    daily = precompute(df, window, top_n=30)

    # 风险等级分布
    rl_dist = {}
    for d in daily:
        rl = d["risk_level"]
        rl_dist[rl] = rl_dist.get(rl, 0) + 1
    print(f"\n规则风险等级分布: {dict(sorted(rl_dist.items()))}")
    print()

    # 基准: 全 ×1.0
    base = fast_simulate(daily, {1:1,2:1,3:1,4:1,5:1}, max_pos=15, df_full=df)
    print(f"基准 (全×1.0, 等权15只): 收益{base['return']:+.2f}%  回撤{base['max_dd']:+.2f}%  "
          f"夏普{base['sharpe']:.2f}  胜率{base['win_rate']:.1f}%")
    print(f"  (注: 基准无诊断, 等权买入 Top30 中的前 15 只, 仓位 = 1/15)")
    print()

    # 网格搜索
    print("=== 网格搜索最优系数表 (Top30 选股, 15只上限) ===")
    r1_grid = [1.0, 1.2, 1.4, 1.6, 1.8]
    r2_grid = [0.9, 1.1, 1.3, 1.5]
    r3_grid = [0.7, 0.9, 1.1, 1.3]
    r4_grid = [0.4, 0.6, 0.8, 1.0]
    r5_grid = [0.2, 0.4, 0.6]

    results = []
    for r1 in r1_grid:
        for r2 in r2_grid:
            if r2 > r1: continue
            for r3 in r3_grid:
                if r3 > r2: continue
                for r4 in r4_grid:
                    if r4 > r3: continue
                    for r5 in r5_grid:
                        if r5 > r4: continue
                        mult = {1:r1, 2:r2, 3:r3, 4:r4, 5:r5}
                        res = fast_simulate(daily, mult, max_pos=15, df_full=df)
                        res["mult"] = mult
                        results.append(res)

    print(f"有效组合: {len(results)} 个")
    print()

    # Top 10 by 夏普
    results.sort(key=lambda r: r["sharpe"], reverse=True)
    print(f"{'排名':>4s} {'R1':>5s} {'R2':>5s} {'R3':>5s} {'R4':>5s} {'R5':>5s}  "
          f"{'收益':>7s} {'回撤':>7s} {'夏普':>6s} {'胜率':>6s}")
    print("-" * 75)
    for i, r in enumerate(results[:10]):
        m = r["mult"]
        print(f"{i+1:4d} {m[1]:5.2f} {m[2]:5.2f} {m[3]:5.2f} {m[4]:5.2f} {m[5]:5.2f}  "
              f"{r['return']:+7.2f}% {r['max_dd']:+7.2f}% {r['sharpe']:6.2f} {r['win_rate']:5.1f}%")

    # Top 5 by 收益
    print()
    print("=== Top 5 收益 ===")
    by_ret = sorted(results, key=lambda r: r["return"], reverse=True)
    for i, r in enumerate(by_ret[:5]):
        m = r["mult"]
        print(f"  {i+1}. R1={m[1]:.2f} R2={m[2]:.2f} R3={m[3]:.2f} R4={m[4]:.2f} R5={m[5]:.2f}  "
              f"收益{r['return']:+.2f}%  回撤{r['max_dd']:+.2f}%  夏普{r['sharpe']:.2f}")

    # 和 LLM 诊断官对比
    print()
    print("=== 对比参考 ===")
    print(f"  LLM 诊断官 (diag2_top30): 收益 +2.36%  回撤 -3.35%  夏普 1.08  胜率 50.9%")
    print(f"  base_v34 (LLM全流程):   收益 +2.43%  回撤 -2.50%  夏普 1.74  胜率 49.1%")

    # 最优夏普 vs LLM
    best_sharpe = results[0]
    print()
    print(f"最优夏普组合: R1={best_sharpe['mult'][1]} R2={best_sharpe['mult'][2]} "
          f"R3={best_sharpe['mult'][3]} R4={best_sharpe['mult'][4]} R5={best_sharpe['mult'][5]}")
    print(f"  收益 {best_sharpe['return']:+.2f}%  回撤 {best_sharpe['max_dd']:+.2f}%  "
          f"夏普 {best_sharpe['sharpe']:.2f}  胜率 {best_sharpe['win_rate']:.1f}%")
    print(f"  vs LLM诊断官: 夏普 {'胜出' if best_sharpe['sharpe'] > 1.08 else '输给 LLM'} "
          f"({best_sharpe['sharpe']:.2f} vs 1.08)")

    # 保存
    out = {
        "window": {"start": str(window[0])[:10], "end": str(window[-1])[:10], "days": len(window)},
        "baseline": {k: v for k, v in base.items() if k != "equity"},
        "best_sharpe": {
            "multipliers": best_sharpe["mult"],
            "metrics": {k: v for k, v in best_sharpe.items() if k not in ("mult", "equity")},
        },
        "best_return": {
            "multipliers": by_ret[0]["mult"],
            "metrics": {k: v for k, v in by_ret[0].items() if k not in ("mult", "equity")},
        },
        "top10_by_sharpe": [
            {"multipliers": r["mult"],
             "metrics": {k: v for k, v in r.items() if k not in ("mult", "equity")}}
            for r in results[:10]
        ],
        "llm_diag_reference": {
            "return": 2.36, "max_dd": -3.35, "sharpe": 1.08, "win_rate": 50.9,
            "note": "diag2_top30 实盘回放结果, 含 LLM 诊断官动态调节",
        },
    }
    out_path = REPLAY_DIR / "calibration_diag_multipliers.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
