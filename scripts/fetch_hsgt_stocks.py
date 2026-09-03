"""北向个股持仓历史库 (agent loop 队列#5).

零模拟: 真实拉取 ak.stock_hsgt_individual_em (需代理), 空/失败重试仍败 → 写 0 字节哨兵并继续
(哨兵=确认无数据), 不造数。产出: replay_data/hsgt_stock/{code}.parquet 分片缓存 (断点续传)。

用法: python scripts/fetch_hsgt_stocks.py
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

socket.setdefaulttimeout(15)
os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["http_proxy"] = os.environ["https_proxy"] = "http://127.0.0.1:7897"

import akshare as ak  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "replay_data" / "hsgt_stock"
RETRIES = 4
SLEEP = 0.25


def force_utf8():
    for n in ("stdout", "stderr"):
        s = getattr(sys, n, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def universe() -> list[str]:
    """主板 60/00 全集 (2017-2024 各年度日线并集, 含已退市在册票)。"""
    codes = set()
    for y in range(2017, 2025):
        fp = ROOT / "replay_data" / f"daily_{y}-01-01_{y}-12-31.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp, columns=["symbol"])
        codes |= {s[3:] for s in d["symbol"] if str(s).startswith(("sh.60", "sz.00"))}
    return sorted(codes)


def fetch_one(code: str) -> pd.DataFrame | None:
    for i in range(RETRIES):
        try:
            df = ak.stock_hsgt_individual_em(symbol=code)
            if df is None or df.empty:
                return None
            return df
        except Exception:
            time.sleep(2 + i * 2)
    raise TimeoutError(f"{code}: 重试 {RETRIES} 次仍失败")


def main() -> int:
    force_utf8()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = universe()
    print(f"universe: {len(codes)} 只主板票", flush=True)
    new_limit = int(os.environ.get("NEW_LIMIT", "0") or 0)  # >0: 本次最多抓 N 只新票 (前台分批用)
    done = 0
    new_done = 0
    for i, code in enumerate(codes):
        fp = OUT_DIR / f"{code}.parquet"
        if fp.exists():  # 分片缓存 (0字节=确认无数据)
            done += 1
            continue
        if new_limit and new_done >= new_limit:
            print(f"[batch] 达到本次上限 {new_limit}, 已完成 {done}/{len(codes)}", flush=True)
            return 0
        try:
            df = fetch_one(code)
        except TimeoutError as e:
            print(f"[FAIL] {e}", flush=True)
            fp.write_bytes(b"")  # 哨兵: 确认无数据, 不再重试
            done += 1
            new_done += 1
            continue
        if df is None:
            fp.write_bytes(b"")
        else:
            df.to_parquet(fp)
        time.sleep(SLEEP)
        done += 1
        new_done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(codes)}", flush=True)
    print(f"DONE fetched/cached {done}/{len(codes)} -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
