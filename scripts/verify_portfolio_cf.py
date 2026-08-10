"""v5.4 组合级反事实 — 真实数据聚焦验证 (复用可靠缓存, 绕过 baostock 全市场重拉).

验证两点:
1. journal 持仓快照构建 (symbol 匹配 / weight 计算) 在真实数据结构下正确.
2. portfolio_level_counterfactual 在真实次日涨跌下产出合理 portfolio_cf.
"""
import pandas as pd
from pathlib import Path

from agent.evolution.portfolio_counterfactual import portfolio_level_counterfactual

cache = Path("replay_data/daily_2020-06-01_2021-02-28.parquet")
print(f"加载缓存: {cache}")
big = pd.read_parquet(cache)
big["date"] = pd.to_datetime(big["date"])
print(f"缓存: {len(big)} 行, {big['symbol'].nunique()} 只, "
      f"{big['date'].min().date()} ~ {big['date'].max().date()}")

# 用一个涨跌分化窗口: 2021-01 抱团牛 (leader 涨, 平均跌)
for tgt_s, tgt_n in [("2021-01-05", "2021-01-06"), ("2021-02-24", "2021-02-25")]:
    tgt = pd.Timestamp(tgt_s)
    nxt = pd.Timestamp(tgt_n)
    day = big[big["date"] == tgt]
    nd = big[big["date"] == nxt]
    if day.empty or nd.empty:
        print(f"[{tgt_s}] 无数据, 跳过")
        continue
    # 次日截面涨跌
    nret = dict(zip(nd["symbol"], pd.to_numeric(nd["pctChg"], errors="coerce")))
    # 构造持仓: 取当日收盘价最高的 5 只 (模拟已持有)
    top = day.sort_values("close", ascending=False).head(5)
    cash = 100000.0
    snap, pos_val = [], 0.0
    for _, row in top.iterrows():
        px = float(row["close"])
        val = 1000 * px
        pos_val += val
        snap.append({"symbol": row["symbol"], "name": row["symbol"],
                     "qty": 1000, "price": round(px, 3),
                     "value": round(val, 2), "weight": 0.0})
    total = cash + pos_val
    for x in snap:
        x["weight"] = round(x["value"] / total, 4)
    r = portfolio_level_counterfactual(snap, nret, date=tgt_s)
    print(f"\n[{tgt_s}] 持仓{len(snap)}只, 总资产¥{total:,.0f}")
    if r is None:
        print("  → 无匹配, None")
        continue
    print(f"  组合次日收益: {r.portfolio_return_pct:+.3f}%")
    print(f"  最差票: {r.worst_stock.symbol} 贡献{r.worst_contribution_pct:+.3f}% "
          f"(权重{r.worst_stock.weight:.3f}, 次日{r.worst_stock.day_return_pct:+.2f}%)")
    print(f"  移除后反事实: {r.counterfactual_return_pct:+.3f}%  改善{r.improvement_pct:+.3f}pp  "
          f"verified={r.verified}")
print("\nOK")