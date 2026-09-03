"""龙虎榜历史库 — 从东财 `stock_lhb_detail_em` 按月区间拉全窗口真历史 (预注册 §十四).

零模拟纪律: 单月失败重试3次仍失败 → 记入 failed_months 并以非0退出, 绝不补假数据.
point-in-time 语义: 每条记录含 上榜日, 背测只用 上榜日<=买入日 的信息; 东财返的"上榜后N日"
是它自带统计, 本库/**背测均不引用**, 只用 上榜日/净买额/买入额/上榜原因/解读 作信号.

用法: python scripts/fetch_lhb_history.py [START] [END]   (默认 2015-01-01 ~ 今天, 按自然月)
产出: replay_data/lhb_history.parquet + replay_data/lhb_history_meta.json
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)  # 清代理直连东财

import akshare as ak  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "replay_data" / "lhb_history.parquet"
META = ROOT / "replay_data" / "lhb_history_meta.json"
RETRIES = 3
SLEEP = 0.4


def force_utf8():
    for n in ("stdout", "stderr"):
        s = getattr(sys, n, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def months_between(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        last = calendar.monthrange(y, m)[1]
        yield dt.date(y, m, 1), dt.date(y, m, last)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch_month(s: dt.date, e: dt.date):
    for i in range(RETRIES):
        try:
            df = ak.stock_lhb_detail_em(start_date=s.strftime("%Y%m%d"), end_date=e.strftime("%Y%m%d"))
            if df is None:
                df = pd.DataFrame()
            return df
        except Exception as ex:
            if i == RETRIES - 1:
                raise RuntimeError(f"{s}~{e}: {type(ex).__name__} {str(ex)[:120]}")
            time.sleep(1.0 * (i + 1))
    raise AssertionError  # unreachable


def _norm_code(c: str) -> str:
    c = str(c).strip().zfill(6)
    return c


def main() -> int:
    force_utf8()
    start = dt.date(2015, 1, 1)
    end = dt.date.today()
    if len(sys.argv) >= 3:
        start = dt.date.fromisoformat(sys.argv[1]); end = dt.date.fromisoformat(sys.argv[2])

    # 断点续传: 已缓存数据 + 已完成的月份区间
    frames = []
    done_months = set()
    if OUT.exists():
        frames.append(pd.read_parquet(OUT))
    if META.exists():
        done_months = set(json.loads(META.read_text(encoding="utf-8")).get("done_months", []))

    failed = []
    all_months = list(months_between(start, end))
    print(f"龙虎榜历史库: {start} ~ {end}, 共 {len(all_months)} 个自然月 | 已缓存 {len(frames) and len(frames[0])} 行")

    for s, e in all_months:
        key = s.strftime("%Y-%m")
        if key in done_months:
            continue
        try:
            df = fetch_month(s, e)
        except Exception as ex:
            failed.append((key, str(ex)))
            print(f"  [FAIL] {key}: {str(ex)[:70]}")
            continue
        if len(df):
            # 归一化: 代码转6位, 上榜日转 date
            code_col = next((c for c in ("代码", "股票代码", "symbol") if c in df.columns), None)
            if code_col is None:
                failed.append((key, "no code col"))
                print(f"  [FAIL] {key}: 缺代码列 {list(df.columns)}")
                continue
            df = df.copy()
            df["_code"] = df[code_col].map(_norm_code)
            df["_date"] = pd.to_datetime(df["上榜日"]).dt.date if "上榜日" in df.columns else None
            if df["_date"].isna().all():
                failed.append((key, "no 上榜日"))
                print(f"  [FAIL] {key}: 缺上榜日")
                continue
            drop = [c for c in ("上榜后1日", "上榜后2日", "上榜后5日", "上榜后10日") if c in df.columns]
            frames.append(df.drop(columns=drop))
        done_months.add(key)
        print(f"  [OK] {key}: +{len(df)} 行 | 累计 {sum(len(f) for f in frames)}")
        time.sleep(SLEEP)
        # 每 5 月写一次盘中缓存, 防中断丢进度
        if len(done_months) % 5 == 0:
            _save(frames, done_months, failed)

    if failed:
        print(f"\n失败月份 {len(failed)} 个: {[k for k, _ in failed]}")
        print("零模拟纪律: 不补假数据, 退出码非0. 可重跑以断点续传.")
        _save(frames, done_months, failed)
        return 1
    final = _save(frames, done_months, failed)
    print(f"\n龙虎榜历史库建成: {OUT} 共 {len(final)} 条, 覆盖 {len(done_months)} 个月, {start}~{end}")
    return 0


def _save(frames, done_months, failed):
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["_code", "_date"]).reset_index(drop=True)
    df.to_parquet(OUT)
    META.write_text(json.dumps({
        "done_months": sorted(done_months), "failed": failed,
        "built": dt.datetime.now().isoformat(), "rows": int(len(df)),
    }, ensure_ascii=False), encoding="utf-8")
    print(f"  已经存 {OUT} ({len(df)} 行)")
    return df


if __name__ == "__main__":
    raise SystemExit(main())