"""探测龙头战法真身所需信号源的真实可用性 (全程零模拟纪律).

测三类源的历史深度 + 字段:
  1. 涨停板池/封板强度  ak.stock_zt_pool_em(date)  — 封资金额/首末封板时间/封板次数/炸板次数/连板数
  2. 龙虎榜/游资席位     akshare 龙虎榜接口 (试点几个), point-in-time 按日归档
  3. (若 1/2 有历史) 评估能否回接到龙头背测.

纪律: 真实拉取, HTTP 失败/空表即报错, 绝不模拟兜底. 探清深度后决定接入方案.
"""
from __future__ import annotations

import os
import sys
import time

# 清代理直连东财 (中国IP不需要代理; 若有代理残留会被东财拒)
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)

PROBE_DATES = ["2026-07-15", "2025-12-15", "2024-11-15", "2023-10-16",
               "2022-09-15", "2021-08-16", "2020-07-15", "2019-06-14",
               "2018-05-15", "2016-04-15", "2015-03-16"]  # 跨11窗各探一交易日


def force_utf8():
    for n in ("stdout", "stderr"):
        s = getattr(sys, n, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def probe_zt_pool():
    import akshare as ak
    print("\n" + "=" * 100)
    print("涨停板池 stock_zt_pool_em — 历史覆盖探测 (封板强度信号源)")
    print("=" * 100)
    for d in PROBE_DATES:
        try:
            df = ak.stock_zt_pool_em(date=d)
            print(f"  {d}: rows={len(df)}  cols={list(df.columns)}")
            if len(df):
                print(f"        sample row: {df.iloc[0].to_dict()}")
        except Exception as e:
            print(f"  {d}: ERROR({type(e).__name__}) {str(e)[:100]}")
        time.sleep(0.3)


def probe_lhb():
    import akshare as ak
    print("\n" + "=" * 100)
    print("龙虎榜 — 历史覆盖探测 (游资席位信号源)")
    print("=" * 100)
    # 候选接口, 逐个探
    cands = [
        ("stock_lhb_detail_daily_sina", "2024-11-15"),
        ("stock_lhb_detail_em", "2024-11-15"),
        ("stock_lhb_ggtj_sina", "2024-11-15"),
        ("stock_lhb_detail_daily_sina", "2020-07-15"),
    ]
    for fn, d in cands:
        if not hasattr(ak, fn):
            print(f"  akshare 无 {fn} — 跳过 (接口版本可能不含)")
            continue
        try:
            df = getattr(ak, fn)(date=d)
            print(f"  {fn}({d}): rows={len(df)}  cols={list(df.columns)}")
            if len(df):
                print(f"        sample: {df.iloc[0].to_dict()}")
        except TypeError as e:
            try:  # 某些接口签名不同
                df = getattr(ak, fn)(start_date=d, end_date=d)
                print(f"  {fn}(start=end={d}): rows={len(df)}  cols={list(df.columns)}")
            except Exception as e2:
                print(f"  {fn}({d}): both-sig ERROR({type(e2).__name__}) {str(e2)[:90]}")
        except Exception as e:
            print(f"  {fn}({d}): ERROR({type(e).__name__}) {str(e)[:90]}")
        time.sleep(0.3)


def main() -> int:
    force_utf8()
    print(f"akshare {__import__('akshare').__version__} | 探测文档: 龙头战法真身信号源")
    probe_zt_pool()
    probe_lhb()
    print("\n" + "=" * 100)
    print("探测完成 — 根据上述真实返回决定接入方案 (拉不到即报错, 零模拟)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())