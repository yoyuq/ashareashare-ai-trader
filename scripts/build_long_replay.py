"""重建长窗口全市场日K (Baostock, 免代理) — 供多 regime 分段回放使用

v3.3 健壮版: 节流(0.08s/查询) + 每200只重连 + 中途保存 + 断点续跑。
上次 830 只快速拉取触发了 Baostock 黑名单 (10001011), 本版避免重蹈。

用法:
  python scripts/build_long_replay.py --start 2025-10-08 --end 2026-07-31
输出:
  replay_data/daily_{start}_{end}.parquet  (全市场, symbol=category, 数值=float32)
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.historical_replay import _bs_query, _optimize_dtypes, load_snapshot_basic  # noqa: E402


def robust_fetch(universe, start: date, end: date, out_path: Path,
                 throttle: float = 0.08, reconnect_every: int = 200):
    """节流 + 重连 + 中途保存 + 断点续跑 的全市场日K拉取。"""
    import baostock as bs

    done = set()
    all_dfs = []
    if out_path.exists():
        try:
            prev = pd.read_parquet(out_path)
            done = set(prev["symbol"].unique())
            all_dfs.append(prev)
            print(f"续跑: 已有 {len(done)} 只, 继续拉取剩余")
        except Exception as e:
            print(f"续跑加载失败, 从零开始: {e}")

    pending = [s for s in universe if s not in done]
    print(f"待拉取: {len(pending)} 只 | 窗口 {start}~{end} | 节流 {throttle}s/查询")

    def _save():
        if all_dfs:
            try:
                pd.concat(all_dfs, ignore_index=True).to_parquet(out_path, index=False)
            except Exception as e:
                print(f"中途保存失败: {e}")

    bs.login()
    t0 = time.time()
    ok = 0
    for i, sym in enumerate(pending):
        try:
            df = _bs_query(sym, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            if not df.empty:
                df["symbol"] = sym
                all_dfs.append(df)
                done.add(sym)
                ok += 1
        except Exception:
            pass
        if (i + 1) % reconnect_every == 0:
            try:
                bs.logout()
            except Exception:
                pass
            bs.login()
        time.sleep(throttle)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(pending)} 处理中 | 成功 {ok} | 耗时 {time.time()-t0:.0f}s")
            _save()
    try:
        bs.logout()
    except Exception:
        pass

    if not all_dfs:
        print("未拉到任何数据!")
        return
    big = pd.concat(all_dfs, ignore_index=True)
    big = _optimize_dtypes(big)
    big.to_parquet(out_path, index=False)
    print(f"完成: {big['symbol'].nunique()} 只 | {len(big)} 行 → {out_path} | 耗时 {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description="重建长窗口全市场日K (节流+重连+续跑)")
    ap.add_argument("--start", default="2025-10-08", help="开始日期 YYYY-MM-DD")
    ap.add_argument("--end", default="2026-07-31", help="结束日期 YYYY-MM-DD")
    ap.add_argument("--throttle", type=float, default=0.08, help="每次查询间 sleep 秒")
    args = ap.parse_args()

    basic, universe = load_snapshot_basic()
    if not universe:
        print("缺少 full_market_cache.json 快照, 无法确定股票池")
        return 1
    y0, m0, d0 = map(int, args.start.split("-"))
    y1, m1, d1 = map(int, args.end.split("-"))
    start, end = date(y0, m0, d0), date(y1, m1, d1)
    out = Path(f"replay_data/daily_{start.isoformat()}_{end.isoformat()}.parquet")
    robust_fetch(universe, start, end, out, throttle=args.throttle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
