"""前瞻纸面验证 (#108) — 注册第一个试点「冷落 beta」的前瞻 bet。

做什么: 冻结今天 (最近交易日) 的 bottom-100 低换手等权篮子 + 匹配 universe (top-800 流动性) +
上证综指收盘, 连同**预先注册的成功/失效判据**一起写进前瞻注册表, 之后由 `scripts/forward_track.py`
在真实未来行情上纸面跟踪。判据一经注册只读, 满足「禁止事后调参追赢」。

用法 (先刷新快照, 再注册):
  python scripts/refresh_market_cache.py     # 腾讯免代理, 日期=最近交易日
  python scripts/forward_register_cold_tilt.py

输出: simulation_data/forward_validation/registry.json (追加/覆盖 edge_id=cold_tilt_bottom100_hold)

诚实边界:
  - 前瞻跟踪用价格收益 (full_market_cache 无分红); #107 已证分红在篮子 vs universe 间大致抵消, 公平近似。
  - 「持有」口径 = 冻结入场名单不换仓, 扣 31bp 一次性全额往返上界 (与 run_cold_tilt_rebalance.py「持有」一致)。
  - 这是 beta 不是 alpha, 且回测已知牛市跑输自家池子 (2019/2021/2025-26), 见 reports/benchmark_discipline.md。
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
from data.full_market_cache import read_full_market_cache  # noqa: E402

K = 100
UNIVERSE_N = 800
COST_BPS = 31.0  # 一次性全额往返上界 (与「持有」口径一致)

EDGE_ID = "cold_tilt_bottom100_hold"

# 预注册判据 (注册后只读; 见 reports/forward_validation_harness.md)
CRITERION = {
    "horizon_trading_days": 60,          # 主判据在入场后 60 交易日落定 (~3 个月)
    "primary_benchmark": "matched_universe",  # 诚实基准 = 自家可投池子等权 (不是错配的上证)
    "primary_success": "basket_net_return >= universe_return (选股 alpha 非负)",
    "secondary_success": "basket_net_return >= sh_index_return (跑赢上证价格指数)",
    "failure_threshold_pp": -10.0,       # 滚动偏离: 篮子累计落后 universe ≥10pp → edge_failing
}


def fetch_index(symbol: str = "sh000001") -> float:
    """腾讯实时取上证综指收盘 (免代理)。失败抛错, 不兜底不模拟。"""
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


def _filter_liquid(df: pd.DataFrame) -> pd.DataFrame:
    """非ST / 可交易(换手>0) / 成交额>0 / pe>0 / pb>0 (与 cold_tilt_live.py 同口径)。"""
    df = df.copy()
    df["name"] = df["name"].astype(str)
    df = df[~df["name"].str.contains("ST", na=False)]
    df = df[df["turnover"] > 0]
    df = df[df["amount"] > 0]
    df = df[(df["pe_ttm"] > 0) & (df["pb"] > 0)]
    return df


def _price_map(df: pd.DataFrame) -> dict[str, float]:
    """code → price (入场价快照); code 缺失/价<=0 剔除。"""
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        code = str(r["code"])
        px = r.get("price")
        if px is None or float(px) <= 0:
            continue
        out[code] = float(px)
    return out


def main() -> int:
    df, date = read_full_market_cache()
    if df is None:
        print("无缓存, 先跑 scripts/refresh_market_cache.py")
        return 1

    df = _filter_liquid(df)
    top800 = df.nlargest(UNIVERSE_N, "amount")
    bottom100 = top800.nsmallest(K, "turnover")

    if len(bottom100) < K:
        print(f"篮子不足 {K} 只 (实际 {len(bottom100)}), 中止注册 (不伪造)。")
        return 1

    sh_close = fetch_index("sh000001")

    bet = {
        "edge_id": EDGE_ID,
        "edge_name": "冷落 beta (bottom-100 低换手等权, 持有)",
        "entry_date": str(date),
        "source": "full_market_cache (tencent_realtime) + qt.gtimg.cn 上证综指",
        "total_return": False,  # 前瞻用价格收益 (见 docstring 诚实边界)
        "construction": {
            "universe_filter": "非ST / turnover>0 / amount>0 / pe_ttm>0 / pb>0",
            "universe": "top-800 by amount",
            "selection": "bottom-100 by turnover",
            "weighting": "equalweight",
            "rebalance": "hold (冻结入场名单, 不换仓)",
            "cost_bps": COST_BPS,
        },
        "entry": {
            "entry_date": str(date),
            "basket_symbols": [str(c) for c in bottom100["code"].tolist()],
            "basket_prices": _price_map(bottom100),
            "universe_symbols": [str(c) for c in top800["code"].tolist()],
            "universe_prices": _price_map(top800),
            "sh_index_close": sh_close,
        },
        "criterion": CRITERION,
    }

    register_bet(bet)
    print(f"已注册前瞻 bet: {EDGE_ID}")
    print(f"  入场日: {date} | 篮子 {len(bet['entry']['basket_symbols'])} 只 | "
          f"universe {len(bet['entry']['universe_symbols'])} 只 | 上证 {sh_close:.2f}")
    print(f"  篮子换手中位数 {bottom100['turnover'].median():.2f}% | "
          f"换手范围 {bottom100['turnover'].min():.2f}%~{bottom100['turnover'].max():.2f}%")
    print(f"  预注册判据: {json.dumps(CRITERION, ensure_ascii=False)}")
    print("  之后用 scripts/forward_track.py 跟踪; 判据注册后只读 (禁止事后调参追赢)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
