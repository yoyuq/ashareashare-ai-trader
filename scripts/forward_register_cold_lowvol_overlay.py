"""前瞻注册 — 冷落低波精选 top5 + 解禁规避叠层 (iter16 PASS 产物, 队列#11 前瞻验证).

与 forward_register_cold_lowvol.py 唯一差异: 精选时剔除「未来30交易日解禁≥5%解禁前流通市值」的票
(与回测 run_restricted_overlay.py 同规则)。universe 基准不变 (top-800 by amount, 不剔除 — 与回测口径一致)。
判据预注册: 60交易日, vs matched_universe 主判据 (alpha 非负), vs 上证副判据, 失效阈值 -10pp。

输出: simulation_data/forward_validation/registry.json (edge_id=cold_lowvol_top5_unlock_screen)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.forward_validation import register_bet  # noqa: E402
from data.full_market_cache import read_full_market_cache  # noqa: E402
from forward_register_cold_lowvol import (  # noqa: E402
    CRITERION, K, UNIVERSE_N, COST_BPS, _filter_liquid, _price_map, _std20_map, fetch_index,
)
from run_restricted_screen import load_events  # noqa: E402

EDGE_ID = "cold_lowvol_top5_unlock_screen"
AHEAD_TD = 30
RATIO_MIN = 0.05


def blocked_symbols(cache_date: str, codes: pd.Series) -> set:
    """缓存日 T 起 30 交易日 (日历近似: 60 自然日) 内解禁≥5% 的票。

    披露: 回测用交易日口径; 注册时点用「解禁时间 ∈ [T, T+60自然日]」日历近似
    (60 自然日 ≈ 40 交易日, 比回测 30 交易日宽 — 更宽的窗只多剔不漏剔, 方向保守)。
    """
    ev = load_events()
    big = ev[ev["ratio"] >= RATIO_MIN]
    t0 = pd.Timestamp(cache_date)
    t1 = t0 + pd.Timedelta(days=60)
    ev_in = big[(big["free_date"] >= t0) & (big["free_date"] <= t1)]
    syms = set(ev_in["sym"])
    return {c for c in codes if (("sh." if str(c).startswith("6") else "sz.") + str(c)) in syms}


def main() -> int:
    df, date = read_full_market_cache()
    if df is None:
        print("无缓存, 先跑 scripts/refresh_market_cache.py")
        return 1

    df = _filter_liquid(df)
    std20 = _std20_map()
    df["std20"] = df["code"].map(std20).astype(float)

    df["turn"] = df["turnover"].astype(float)
    df["log_mkt"] = np.log1p(df["amount"].astype(float) * 100.0 / df["turnover"].astype(float) / 1e8)
    df = df[df["std20"].notna()].copy()
    r_turn = df["turn"].rank(pct=True)
    r_std = df["std20"].rank(pct=True)
    r_mkt = df["log_mkt"].rank(pct=True)
    df["score"] = -(r_turn + r_std + r_mkt)

    blk = blocked_symbols(date, df["code"])
    n_before = len(df)
    df_sel = df[~df["code"].isin(blk)].copy()
    print(f"解禁规避叠层: 剔除 {len(blk)} 只 (未来60自然日解禁≥5%), 精选池 {n_before} → {len(df_sel)}")

    top5 = df_sel.nlargest(K, "score")
    universe = df.nlargest(UNIVERSE_N, "amount")  # 基准不剔除 (与回测 B vs A 同口径)

    if len(top5) < K:
        print(f"精选不足 {K} 只 (实际 {len(top5)}), 中止注册 (不伪造)。")
        return 1

    sh_close = fetch_index("sh000001")

    bet = {
        "edge_id": EDGE_ID,
        "edge_name": "冷落低波精选 top5 + 解禁规避叠层 (iter16 PASS, 前瞻验证)",
        "entry_date": str(date),
        "source": "full_market_cache (tencent_realtime) + replay_data std20(至07-31) "
                  "+ restricted_release 未来时间表 + qt.gtimg.cn 上证",
        "total_return": False,
        "construction": {
            "universe_filter": "非ST / 主板(60/00) / turnover>0 / amount>0 / pe>0 / pb>0",
            "universe": "top-800 by amount (主板, 不剔除解禁票)",
            "selection": "top-5 by score=-(turn_rank+std20_rank+log_mkt_rank), "
                         f"剔除未来{AHEAD_TD}交易日(注册口径60自然日)解禁≥{RATIO_MIN:.0%}的票",
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
    print(f"  入场日: {date} | 精选 {len(top5)} 只 (剔 {len(blk)}) | universe {len(universe)} 只 | 上证 {sh_close:.2f}")
    for _, r in top5.iterrows():
        print(f"    {r['code']} {r['name']}  换手{r['turn']:.2f}%  std20={r['std20']:.3f}  "
              f"流通市值≈{np.expm1(r['log_mkt']):.0f}亿  score={r['score']:+.3f}")
    print("  之后用 scripts/forward_track.py 跟踪; 判据注册后只读 (禁止事后调参追赢)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
