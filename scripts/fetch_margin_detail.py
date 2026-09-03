"""两融个股明细历史库 (agent loop 队列#9) — 按日截面分片.

零模拟: 每交易日拉 sse+szse 个股两融明细, 只留主板 60/00, 失败重试仍败 → 0字节哨兵, 不造数。
产出: replay_data/margin_detail/{YYYYMMDD}.parquet (断点续传, 每文件=一交易日, 列: code, rz_balance)。

用法: NEW_LIMIT=250 python scripts/fetch_margin_detail.py
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
OUT_DIR = ROOT / "replay_data" / "margin_detail"
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


def trading_dates() -> list[str]:
    """交易日历: 2015-01 → 2026-08 各年度日线并集。"""
    dates = set()
    for y in range(2015, 2027):
        fp = ROOT / "replay_data" / f"daily_{y}-01-01_{y}-12-31.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp, columns=["date"])
        dates |= set(pd.to_datetime(d["date"]).dt.strftime("%Y%m%d"))
    return sorted(dates)


def fetch_date(ds: str) -> pd.DataFrame | None:
    """单日: sse+szse 并集, 主板 60/00, 归一化为 code+rz_balance。"""
    frames = []
    for fn, code_col in ((ak.stock_margin_detail_sse, "标的证券代码"),
                         (ak.stock_margin_detail_szse, "证券代码")):
        for i in range(RETRIES):
            try:
                d = fn(date=ds)
                break
            except Exception:
                time.sleep(2 + i * 2)
        else:
            raise TimeoutError(f"{ds}: 重试 {RETRIES} 次仍失败")
        if d is None or d.empty:
            continue
        d = d[[code_col, "融资余额"]].rename(columns={code_col: "code", "融资余额": "rz_balance"})
        d["code"] = d["code"].astype(str).str.zfill(6)
        d = d[d["code"].str.startswith(("60", "00"))]
        frames.append(d)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"], keep="last")
    return out


def main() -> int:
    force_utf8()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = trading_dates()
    print(f"trading dates: {len(dates)} (2015→2026)", flush=True)
    new_limit = int(os.environ.get("NEW_LIMIT", "0") or 0)
    done = new_done = 0
    for ds in dates:
        fp = OUT_DIR / f"{ds}.parquet"
        if fp.exists():
            done += 1
            continue
        if new_limit and new_done >= new_limit:
            print(f"[batch] 达到本次上限 {new_limit}, 已完成 {done}/{len(dates)}", flush=True)
            return 0
        try:
            d = fetch_date(ds)
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
        if done % 200 == 0:
            print(f"  {done}/{len(dates)}", flush=True)
    print(f"DONE fetched/cached {done}/{len(dates)} -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
