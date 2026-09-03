"""③ 前瞻纸面验证 — 注册「冷落低波精选 top5」的前瞻 bet (复用 #108 harness, 见 preregistration §八).

做什么: 冻结「冷落低波精选」(score=−低换手rank−低波rank−小市值rank, 主板 top5) 的入场名单,
连同预注册成功/失效判据写进前瞻注册表, 之后由 `scripts/forward_track.py` 在真实未来行情上纸面跟踪。

忠实度:
  - 低换手 = full_market_cache 当日换手率 (与回测 `df["turn"]` 同口径).
  - 低波 std20 = replay_data 20日收盘滚动std, 取最新 (2026-07-31, 数据所及; 相对 08-14 快照约滞后~10交易日, 披露).
  - 小市值 = 流通市值对数 log1p(amount*100/turnover/1e8), 与回测 `log_mkt` 同公式.
  - 主板限制: code 前缀 60 (沪) / 00 (深), 与回测 MAIN_BOARD=sh.60+sz.00 一致.

用法:
  python scripts/forward_register_cold_lowvol.py   # 依赖 full_market_cache (已存在 2026-08-14 快照)

输出: simulation_data/forward_validation/registry.json (追加 edge_id=cold_lowvol_top5_hold)

诚实边界: 价格收益口径 (无分红); 持有口径 (冻结不换仓); 因子 std20 滞后 ~10 交易日; 集中 5 只方差大。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.forward_validation import register_bet  # noqa: E402
from data.full_market_cache import read_full_market_cache  # noqa: E402

K = 5
UNIVERSE_N = 800
COST_BPS = 31.0
EDGE_ID = "cold_lowvol_top5_hold"

CRITERION = {
    "horizon_trading_days": 60,
    "primary_benchmark": "matched_universe",
    "primary_success": "basket_net_return >= universe_return (选股 alpha 非负)",
    "secondary_success": "basket_net_return >= sh_index_return (跑赢上证价格指数)",
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


def _std20_map() -> dict[str, float]:
    """replay_data 最新窗口算 20日收盘滚动std, 取每只最新值 → {6位code: std20}."""
    fp = ROOT / "replay_data" / "daily_2026-02-21_2026-07-31.parquet"
    d = pd.read_parquet(fp, columns=["date", "symbol", "close"])
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["symbol", "date"])
    d["std20"] = d.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    last = d.dropna(subset=["std20"]).groupby("symbol").tail(1)
    out: dict[str, float] = {}
    for _, r in last.iterrows():
        code6 = str(r["symbol"]).split(".")[-1]
        out[code6] = float(r["std20"])
    return out


def _filter_liquid(df: pd.DataFrame) -> pd.DataFrame:
    """非ST / 主板(60/00) / 可交易(turnover>0) / amount>0 / pe>0 / pb>0 (与回测 load_base_window 同口径)."""
    df = df.copy()
    df["name"] = df["name"].astype(str)
    df["code"] = df["code"].astype(str)
    df = df[~df["name"].str.contains("ST", na=False)]
    df = df[df["code"].str.startswith(("60", "00"))]  # 主板: 沪60 / 深00
    df = df[df["turnover"] > 0]
    df = df[df["amount"] > 0]
    df = df[(df["pe_ttm"] > 0) & (df["pb"] > 0)]
    return df


def _price_map(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        px = r.get("price")
        if px is None or float(px) <= 0:
            continue
        out[str(r["code"])] = float(px)
    return out


def main() -> int:
    df, date = read_full_market_cache()
    if df is None:
        print("无缓存, 先跑 scripts/refresh_market_cache.py")
        return 1

    df = _filter_liquid(df)
    std20 = _std20_map()
    df["std20"] = df["code"].map(std20).astype(float)

    # 三因子 (与回测 _cold_lowvol 同口径)
    df["turn"] = df["turnover"].astype(float)
    df["log_mkt"] = np.log1p(df["amount"].astype(float) * 100.0 / df["turnover"].astype(float) / 1e8)
    df = df[df["std20"].notna()].copy()
    r_turn = df["turn"].rank(pct=True)
    r_std = df["std20"].rank(pct=True)
    r_mkt = df["log_mkt"].rank(pct=True)
    df["score"] = -(r_turn + r_std + r_mkt)

    top5 = df.nlargest(K, "score")
    universe = df.nlargest(UNIVERSE_N, "amount")

    if len(top5) < K:
        print(f"精选不足 {K} 只 (实际 {len(top5)}), 中止注册 (不伪造)。")
        return 1

    sh_close = fetch_index("sh000001")

    bet = {
        "edge_id": EDGE_ID,
        "edge_name": "冷落低波精选 top5 (低换手+低波+小市值, 主板, 持有)",
        "entry_date": str(date),
        "source": "full_market_cache (tencent_realtime) + replay_data std20(至07-31) + qt.gtimg.cn 上证",
        "total_return": False,
        "construction": {
            "universe_filter": "非ST / 主板(60/00) / turnover>0 / amount>0 / pe>0 / pb>0",
            "universe": "top-800 by amount (主板)",
            "selection": "top-5 by score=-(turn_rank+std20_rank+log_mkt_rank)",
            "weighting": "equalweight",
            "rebalance": "hold (冻结入场名单, 不换仓)",
            "cost_bps": COST_BPS,
        },
        "entry": {
            "entry_date": str(date),
            "basket_symbols": [str(c) for c in top5["code"].tolist()],
            "basket_prices": _price_map(top5),
            "universe_symbols": [str(c) for c in universe["code"].tolist()],
            "universe_prices": _price_map(universe),
            "sh_index_close": sh_close,
        },
        "criterion": CRITERION,
    }

    register_bet(bet)
    print(f"已注册前瞻 bet: {EDGE_ID}")
    print(f"  入场日: {date} | 精选 {len(top5)} 只 | universe {len(universe)} 只 | 上证 {sh_close:.2f}")
    for _, r in top5.iterrows():
        print(f"    {r['code']} {r['name']}  换手{r['turn']:.2f}%  std20={r['std20']:.3f}  "
              f"流通市值≈{np.expm1(r['log_mkt']):.0f}亿  score={r['score']:+.3f}")
    print(f"  预注册判据: {json.dumps(CRITERION, ensure_ascii=False)}")
    print("  之后用 scripts/forward_track.py 跟踪; 判据注册后只读 (禁止事后调参追赢)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
