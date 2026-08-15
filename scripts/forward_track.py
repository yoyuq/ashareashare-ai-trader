"""前瞻纸面验证 (#108) — 在真实未来行情上跟踪已注册的 edge, 判其是否失效。

对注册表里每个 bet (当前仅冷落 beta 试点), 用当前全市场快照算篮子 vs 匹配 universe vs 上证综指的前瞻
收益, 应用预注册判据, 追加一次 tracking 快照, 打印滚动偏离 + edge_failing 状态。

用法 (先刷新快照, 再跟踪):
  python scripts/refresh_market_cache.py     # 更新全市场快照到最近交易日
  python scripts/forward_track.py

输出: simulation_data/forward_validation/registry.json (各 bet.tracking 追加) + stdout 报告。

诚实边界:
  - 快照日期与入场日相同 (无新交易日) 时跳过, 不重复追加 day-0 空快照。
  - 上证综指腾讯实时获取失败即报错 (零模拟: 缺数据报错不兜底)。
  - 停牌/退市票 ffill (equalweight_return 内部), 与回测口径一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.forward_validation import (  # noqa: E402
    append_tracking,
    compute_status,
    load_registry,
)
from data.full_market_cache import read_full_market_cache  # noqa: E402


def fetch_index(symbol: str = "sh000001") -> float:
    """腾讯实时取上证综指收盘 (免代理)。失败抛错。"""
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


def _current_prices(df: pd.DataFrame, codes: list[str]) -> dict[str, float]:
    """当前快照里 code → price (只回传篮子在籍票; 缺失交给 equalweight_return 的 ffill)。"""
    sub = df[df["code"].astype(str).isin(codes)]
    return {str(r["code"]): float(r["price"]) for _, r in sub.iterrows()
            if r.get("price") is not None and float(r["price"]) > 0}


def main() -> int:
    df, date = read_full_market_cache()
    if df is None:
        print("无缓存, 先跑 scripts/refresh_market_cache.py")
        return 1

    reg = load_registry()
    bets = reg.get("bets", {})
    if not bets:
        print("注册表为空, 先跑 scripts/forward_register_cold_tilt.py")
        return 1

    sh_close = fetch_index("sh000001")

    for edge_id, bet in bets.items():
        entry = bet["entry"]
        tracking = bet.setdefault("tracking", [])

        # 无前瞻新数据 (快照日 <= 入场日) → 跳过, 不追加 day-0 空快照
        from datetime import date as _date_cls
        cur_date = _date_cls.fromisoformat(str(date))
        entry_date = _date_cls.fromisoformat(str(bet["entry"]["entry_date"]))
        if cur_date <= entry_date:
            print(f"[{edge_id}] 无前瞻新数据 (快照日 {date} <= 入场日 {entry_date}), 跳过。")
            continue
        if tracking and tracking[-1].get("current_date") == str(date):
            print(f"[{edge_id}] 快照日期 {date} 已跟踪过, 跳过 (无新交易日)。")
            continue

        basket_px = _current_prices(df, entry["basket_symbols"])
        universe_px = _current_prices(df, entry["universe_symbols"])
        status = compute_status(bet, basket_px, universe_px, sh_close, str(date))
        append_tracking(edge_id, status)

        print("\n" + "=" * 84)
        print(f"前瞻跟踪: {edge_id}  ({bet['entry']['entry_date']} → {date})")
        print("=" * 84)
        print(f"  已过交易日 {status['elapsed_trading_days']} | "
              f"horizon {bet['criterion']['horizon_trading_days']} 交易日")
        print(f"  篮子净收益    {status['basket_return_pct']:>+7.2f}%  (毛 {status['basket_gross_return_pct']:+.2f}%)")
        print(f"  匹配universe  {status['universe_return_pct']:>+7.2f}%")
        print(f"  上证综指      {status['sh_index_return_pct']:>+7.2f}%")
        print(f"  选股 alpha    {status['selection_alpha_pp']:>+7.2f}pp  "
              f"(失败阈值 {bet['criterion']['failure_threshold_pp']}pp)")
        print(f"  跑赢上证      {status['vs_index_pp']:>+7.2f}pp")
        flag = "⚠️ EDGE 失效" if status["edge_failing"] else "ok"
        print(f"  edge_failing: {flag} | horizon_reached: {status['horizon_reached']} | "
              f"primary_success: {status['primary_success']} | secondary_success: {status['secondary_success']}")

    print(f"\n跟踪完成, 已写 {ROOT / 'simulation_data' / 'forward_validation' / 'registry.json'}")

    # 可 grep 的失效标记 (供日志监控/告警): 任一 edge 失效则显式打印
    reg = load_registry()
    for edge_id, bet in reg.get("bets", {}).items():
        tr = bet.get("tracking", [])
        if tr and tr[-1].get("edge_failing"):
            print(f"[EDGE_FAILING] {edge_id} selection_alpha={tr[-1]['selection_alpha_pp']}pp "
                  f"@ {tr[-1]['current_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
