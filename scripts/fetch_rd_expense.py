"""研发费用历史库 (agent loop 队列#8 数据准备).

零模拟: 同花顺利润表 stock_financial_benefit_ths 含 研发费用/营业总收入 标准科目
(2005→2026, 值为"25.35亿"字符串). 失败重试仍败 → 0字节哨兵 (确认无数据), 不造数。
产出: replay_data/rd_expense/{code}.parquet (断点续传)。

用法: NEW_LIMIT=300 python scripts/fetch_rd_expense.py
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
OUT_DIR = ROOT / "replay_data" / "rd_expense"
RETRIES = 4
SLEEP = 0.3
KEEP_COLS = ["报告期", "研发费用", "一、营业总收入"]  # 源表营收列带"一、"前缀 (v2 修正)


def force_utf8():
    for n in ("stdout", "stderr"):
        s = getattr(sys, n, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def universe() -> list[str]:
    """主板 60/00 全集 (与日线面板同口径)。"""
    codes = set()
    for y in range(2017, 2027):
        fp = ROOT / "replay_data" / f"daily_{y}-01-01_{y}-12-31.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp, columns=["symbol"])
        codes |= {s[3:] for s in d["symbol"] if str(s).startswith(("sh.60", "sz.00"))}
    return sorted(codes)


def parse_val(v):
    """'25.35亿' → 2.535e7*1e10? 不: 亿=1e8 → 2.535e9. 非数值(bool/None/'--') → NaN。"""
    if isinstance(v, str):
        s = v.strip()
        if s.endswith("亿"):
            try:
                return float(s[:-1]) * 1e8
            except ValueError:
                return float("nan")
        if s.endswith("万"):
            try:
                return float(s[:-1]) * 1e4
            except ValueError:
                return float("nan")
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    return float("nan")


def fetch_one(code: str) -> pd.DataFrame | None:
    for i in range(RETRIES):
        try:
            df = ak.stock_financial_benefit_ths(symbol=code, indicator="按报告期")
            if df is None or df.empty:
                return None
            cols = [c for c in KEEP_COLS if c in df.columns]
            if "研发费用" not in cols:
                return None
            df = df[cols].copy()
            df = df.rename(columns={"一、营业总收入": "营业总收入"})
            for c in ("研发费用", "营业总收入"):
                if c in df.columns:
                    df[c] = df[c].map(parse_val).astype("float64")
            return df
        except Exception:
            time.sleep(2 + i * 2)
    raise TimeoutError(f"{code}: 重试 {RETRIES} 次仍失败")


def main() -> int:
    force_utf8()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = universe()
    print(f"universe: {len(codes)} 只主板票", flush=True)
    new_limit = int(os.environ.get("NEW_LIMIT", "0") or 0)  # >0: 本次最多抓 N 只新票 (分批用)
    done = new_done = 0
    for code in codes:
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
            fp.write_bytes(b"")  # 哨兵
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
