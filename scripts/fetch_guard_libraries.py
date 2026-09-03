"""北向资金 + SSE两融 历史库构建 (预注册 §十五, 外生守卫信号源).

零模拟纪律: 真实拉取, 空/失败即报错非0退出, 不补假数据.
产出:
  replay_data/hsgt.parquet       北向资金 日成交净买额 (2014-11→今)
  replay_data/margin_sse.parquet SSE 融资融券余额 (2015→今)

用法: python scripts/fetch_guard_libraries.py
"""
from __future__ import annotations

import datetime as dt
import os
import socket
import sys
import time
from pathlib import Path

socket.setdefaulttimeout(15)
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)

import akshare as ak  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT_HSGT = ROOT / "replay_data" / "hsgt.parquet"
OUT_MARGIN = ROOT / "replay_data" / "margin_sse.parquet"


def force_utf8():
    for n in ("stdout", "stderr"):
        s = getattr(sys, n, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def save(df, path, cols):
    if df is None or not len(df):
        raise RuntimeError(f"空数据, 不落地: {path}")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=cols)
    df.to_parquet(path)
    print(f"  -> {path}: {len(df)} 行, {sorted(df.columns)}")
    return df


def main() -> int:
    force_utf8()
    print("=== 北向资金 (hsgt) ===")
    hs = ak.stock_hsgt_hist_em(symbol="北向资金")
    hs = save(hs, OUT_HSGT, {"日期": "date", "当日成交净买额": "net_buy"})
    print(f"    期间 {hs['date'].min()} ~ {hs['date'].max()}")

    print("=== SSE 两融 (margin) ===")
    # stock_margin_sse(start,end) 区间一拉全史; 分成若干区间防一次过大
    start = dt.date(2015, 1, 1)
    end = dt.date.today()
    frames = []
    cur = start
    while cur <= end:
        seg_end = min(cur + dt.timedelta(days=365 * 2) - dt.timedelta(days=1), end)
        df = ak.stock_margin_sse(start_date=cur.strftime("%Y%m%d"), end_date=seg_end.strftime("%Y%m%d"))
        if df is not None and len(df):
            frames.append(df)
        print(f"    区间 {cur}~{seg_end}: +{len(df) if df is not None else 0} 行")
        cur = seg_end + dt.timedelta(days=1)
        time.sleep(0.4)
    mg = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["信用交易日期"])
    mg = save(mg, OUT_MARGIN, {"信用交易日期": "date", "融资融券余额": "margin"})
    print(f"    期间 {mg['date'].min()} ~ {mg['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())