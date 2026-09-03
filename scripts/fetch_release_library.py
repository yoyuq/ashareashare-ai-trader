"""限售解禁历史库 (预注册 §十六, 个股风险因子).

零模拟纪律: 真实拉取, 空/失败重试仍败 → 报错非0退出, 不补假数据.
产出: replay_data/release.parquet (2016→今, 股票代码/解禁时间/实际解禁市值/占解禁前流通市值比例).

用法: python scripts/fetch_release_library.py [START] [END]   (默认 2016-01-01 ~ 今天, 按自然月)
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
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
OUT = ROOT / "replay_data" / "release.parquet"
META = ROOT / "replay_data" / "release_meta.json"
RETRIES = 3
SLEEP = 0.3


def force_utf8():
    for n in ("stdout", "stderr"):
        s = getattr(sys, n, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def months_between(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        last = calendar.monthrange(y, m)[1]
        yield dt.date(y, m, 1), dt.date(y, m, last)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch_month(s, e):
    for i in range(RETRIES):
        try:
            df = ak.stock_restricted_release_detail_em(
                start_date=s.strftime("%Y%m%d"), end_date=e.strftime("%Y%m%d"))
            return df if df is not None else pd.DataFrame()
        except Exception as ex:
            if i == RETRIES - 1:
                raise RuntimeError(f"{s}~{e}: {type(ex).__name__} {str(ex)[:110]}")
            time.sleep(1.0 * (i + 1))
    raise AssertionError


def save(frames, done, failed):
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    # 归一化
    ck = next((c for c in ("股票代码", "股票简称") if c in df.columns), None)
    df["_code"] = df["股票代码"].astype(str).str.zfill(6)
    df["_date"] = pd.to_datetime(df["解禁时间"]).dt.date
    df.to_parquet(OUT)
    META.write_text(json.dumps({"done_months": sorted(done), "failed": failed,
                                "built": dt.datetime.now().isoformat(), "rows": int(len(df))},
                               ensure_ascii=False), encoding="utf-8")
    print(f"  已经存 {OUT} ({len(df)} 行)")
    return df


def main() -> int:
    force_utf8()
    start = dt.date(2016, 1, 1); end = dt.date.today()
    if len(sys.argv) >= 3:
        start = dt.date.fromisoformat(sys.argv[1]); end = dt.date.fromisoformat(sys.argv[2])
    frames = []
    done = set()
    if OUT.exists():
        frames.append(pd.read_parquet(OUT))
    if META.exists():
        done = set(json.loads(META.read_text(encoding="utf-8")).get("done_months", []))
    failed = []
    months = list(months_between(start, end))
    print(f"限售解禁历史库: {start}~{end}, {len(months)} 个月 | 已缓存 {len(frames) and len(frames[0])} 行")
    for s, e in months:
        key = s.strftime("%Y-%m")
        if key in done:
            continue
        try:
            df = fetch_month(s, e)
        except Exception as ex:
            failed.append((key, str(ex))); print(f"  [FAIL] {key}: {str(ex)[:60]}"); continue
        frames.append(df); done.add(key)
        print(f"  [OK] {key}: +{len(df)} 行 | 累计 {sum(len(f) for f in frames)}")
        time.sleep(SLEEP)
        if len(done) % 8 == 0:
            save(frames, done, failed)
    if failed:
        print(f"\n失败 {len(failed)} 个: {[k for k,_ in failed]}; 不补假, 退出码非0. 重跑断点续传.")
        save(frames, done, failed); return 1
    final = save(frames, done, failed)
    print(f"\n解禁库建成: {OUT} {len(final)} 行, {len(done)} 个月, {start}~{end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())