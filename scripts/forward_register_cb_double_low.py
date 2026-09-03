"""④ 前瞻纸面验证 — 注册「低溢价主导 top10」(A 臂原公式) 的前瞻 bet (复用 #108 harness).

命名澄清 (2026-08-30 核查): 面板 `转股溢价率` 为**百分数** (113050: close 144.967 / 转股价值
143.391 → 1.099%, 面板原值 1.098725), 故 A 臂公式 score=−(close+prem×100) 中溢价权重是价格的
~100 倍 — 实际选的是**最低溢价 (近似 0/负溢价, 股性债)**, 是「低溢价主导周轮动」而非经典双低
(价格+溢价率, 诊断回测 5/8 FAIL 未过线)。本 bet 忠实注册**实际过线的公式**, 与回测 A 臂完全同口径。

做什么: 冻结 top10 (score=−(close+prem×100), 取 nlargest 即最小 close+prem×100) 的入场名单,
连同预注册成功/失效判据写进前瞻注册表, 之后由 `scripts/forward_track_cb.py` 跟踪。

忠实度:
  - 入场快照取自 replay_data 面板最后交易日 (真实 sina 收盘 + 东财 point-in-time 溢价率), 零模拟。
  - 双低公式与回测 A 臂完全同口径: score = −(close + prem×100), nlargest (方向与回测一致)。
  - 基准 = 入场日在籍转债全体等权 (成员冻结, 退市 ffill — harness 口径)。
  - 形态为「持有」, 非周轮动 (如实标注; 周轮动前瞻待 cb_forward 快照攒积后重放)。

用法:
  python scripts/forward_register_cb_double_low.py

输出: simulation_data/forward_validation/registry.json (追加 edge_id=cb_double_low_top10_hold)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.forward_validation import register_bet  # noqa: E402

K = 10
COST_BPS = 20.0
EDGE_ID = "cb_lowprem_top10_hold"

CRITERION = {
    "horizon_trading_days": 60,
    "primary_benchmark": "cb_equal_weight_frozen",
    "primary_success": "basket_net_return >= cb_equal_weight_return (低溢价因子 OOS alpha 非负)",
    "secondary_success": "basket_net_return >= sh_index_return (跑赢上证综指)",
    "failure_threshold_pp": -10.0,
}


def fetch_index(symbol: str = "sh000001") -> float:
    r = requests.get(f"https://qt.gtimg.cn/q={symbol}", timeout=8,
                     headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
    r.encoding = "gbk"
    for line in r.text.strip().split("\n"):
        if "=" not in line or "~" not in line:
            continue
        f = line.split("=", 1)[1].strip('"').split("~")
        if len(f) > 3 and f[3]:
            return float(f[3])
    raise RuntimeError(f"腾讯未返回 {symbol} 指数价")


def load_entry_snapshot() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """面板最后交易日的 (date, 全体在籍快照, 双低 top10)。

    在籍 = 当日同时有收盘价与转股溢价率 (sina 日线 + 东财 point-in-time)。
    """
    daily = pd.read_parquet(ROOT / "replay_data" / "cb_daily.parquet")
    value = pd.read_parquet(ROOT / "replay_data" / "cb_value.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    value["date"] = pd.to_datetime(value["日期"])
    value["symbol"] = value["symbol"].astype(str)

    last_date = daily["date"].max()
    d = daily[daily["date"] == last_date][["symbol", "close"]].copy()
    v = value[value["date"] == last_date][["symbol", "转股溢价率"]].rename(
        columns={"转股溢价率": "prem"})
    snap = d.merge(v, on="symbol", how="inner").dropna(subset=["close", "prem"])
    snap = snap[snap["close"] > 0]
    if snap.empty:
        raise RuntimeError(f"面板 {last_date.date()} 无在籍快照 (price+prem 均缺失), 中止注册")
    snap["score"] = -(snap["close"] + snap["prem"] * 100.0)
    top = snap.nlargest(K, "score")  # 与回测 select_top 同方向: 最小 close+prem×100 = 最低溢价
    if len(top) < K:
        raise RuntimeError(f"双低精选不足 {K} 只 (实际 {len(top)}), 中止注册 (不伪造)")
    return str(last_date.date()), snap, top


def main() -> int:
    entry_date, snap, top = load_entry_snapshot()
    sh_close = fetch_index("sh000001")

    bet = {
        "edge_id": EDGE_ID,
        "edge_name": "转债低溢价主导 top10 (score=−(close+prem×100), A臂原公式, 冻结入场名单持有)",
        "entry_date": entry_date,
        "source": "replay_data/cb_daily.parquet + cb_value.parquet (sina收盘+东财pt溢价) + qt.gtimg.cn 上证",
        "total_return": False,
        "construction": {
            "universe_filter": "入场日同时有收盘价与转股溢价率的全部上市转债",
            "universe": "全体在籍转债等权 (成员冻结)",
            "selection": "top-10 lowest score=-(close+prem*100)",
            "weighting": "equalweight",
            "rebalance": "hold (冻结入场名单, 不换仓; 非周轮动, 如实标注)",
            "cost_bps": COST_BPS,
        },
        "entry": {
            "entry_date": entry_date,
            "basket_symbols": [str(s) for s in top["symbol"].tolist()],
            "basket_prices": {str(r["symbol"]): float(r["close"]) for _, r in top.iterrows()},
            "universe_symbols": [str(s) for s in snap["symbol"].tolist()],
            "universe_prices": {str(r["symbol"]): float(r["close"]) for _, r in snap.iterrows()},
            "sh_index_close": sh_close,
        },
        "criterion": CRITERION,
    }

    register_bet(bet)
    print(f"已注册前瞻 bet: {EDGE_ID}")
    print(f"  入场日: {entry_date} | 低溢价 top10 | 在籍 universe {len(snap)} 只 | 上证 {sh_close:.2f}")
    for _, r in top.iterrows():
        print(f"    {r['symbol']}  close={r['close']:.3f}  prem={r['prem']:.2f}%  score={r['score']:.2f}")
    print(f"  预注册判据: {json.dumps(CRITERION, ensure_ascii=False)}")
    print("  之后用 scripts/forward_track_cb.py 跟踪; 判据注册后只读 (禁止事后调参追赢)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
