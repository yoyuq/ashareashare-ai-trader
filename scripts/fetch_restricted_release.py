"""限售解禁明细历史库 (agent loop 队列#4) — 按月分片.

零模拟: ak.stock_restricted_release_detail_em(start,end) 按月拉 (该接口半月以上区间会挂, 见
memory signal-sources-availability), 失败重试仍败 → 0字节哨兵, 不造数。
产出: replay_data/restricted_release/{YYYYMM}.parquet (断点续传, 一文件=一月)。

用法: NEW_LIMIT=6 python scripts/fetch_restricted_release.py
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

socket.setdefaulttimeout(20)

import akshare as ak  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "replay_data" / "restricted_release"
RETRIES = 3
SLEEP = 2.0
START = (2016, 1)   # 数据 2016 起 (memory 已证)
END = (2026, 8)


def force_utf8():
    for n in ("stdout", "stderr"):
        s = getattr(sys, n, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def months() -> list[str]:
    out = []
    y, m = START
    while (y, m) <= END:
        out.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def fetch_month(ym: str) -> pd.DataFrame | None:
    # akshare 接口要求 YYYYMMDD (横杠格式会拼出垃圾 filter → result=None)
    start = ym + "01"
    y, m = int(ym[:4]), int(ym[4:6])
    e_y, e_m = (y, m + 1) if m < 12 else (y + 1, 1)
    end = f"{e_y}{e_m:02d}01"
    for i in range(RETRIES):
        try:
            d = ak.stock_restricted_release_detail_em(start_date=start, end_date=end)
            break
        except Exception:
            time.sleep(3 + i * 3)
    else:
        raise TimeoutError(f"{ym}: 重试 {RETRIES} 次仍失败")
    if d is None or d.empty:
        return None
    return d


def main() -> int:
    force_utf8()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ms = months()
    print(f"months: {len(ms)} ({START}->{END})", flush=True)
    new_limit = int(os.environ.get("NEW_LIMIT", "0") or 0)
    done = new_done = 0
    for ym in ms:
        fp = OUT_DIR / f"{ym}.parquet"
        if fp.exists():
            done += 1
            continue
        if new_limit and new_done >= new_limit:
            print(f"[batch] 达到本次上限 {new_limit}, 已完成 {done}/{len(ms)}", flush=True)
            return 0
        try:
            d = fetch_month(ym)
        except TimeoutError as e:
            print(f"[FAIL] {e}", flush=True)
            fp.write_bytes(b"")
            done += 1
            new_done += 1
            continue
        if d is None:
            fp.write_bytes(b"")
        else:
            d.to_parquet(fp)
        time.sleep(SLEEP)
        done += 1
        new_done += 1
        if done % 12 == 0:
            print(f"  {done}/{len(ms)}", flush=True)
    print(f"DONE fetched/cached {done}/{len(ms)} -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
